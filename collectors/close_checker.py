"""Background worker that refreshes market status for stale rows.

Each cycle:
  1. Picks the N oldest-checked Kalshi markets whose status is not yet
     'closed' or 'settled', calls the Kalshi /markets/{ticker} endpoint,
     and updates `status`, `result`, `settlement_value`, `close_time`,
     `expiration_time`, `settle_time`, plus `fetched_at = now`.
  2. Picks the N oldest-checked Polymarket markets whose `closed` is false,
     calls the CLOB /markets/{condition_id} endpoint, and updates `closed`,
     `active`, `accepting_orders` plus `fetched_at = now`.

Markets that 404 on the platform side have their `fetched_at` bumped only,
so they fall to the back of the queue without blocking the next cycle.

This is the only writer that updates `status`/`closed` for already-stored
markets; the snapshot/history collectors only upsert from bulk listings,
which silently miss settled markets the API stops returning.
"""
from __future__ import annotations

import argparse
import logging
import threading
import time
from datetime import datetime, timezone
from typing import Optional

import requests
from sqlalchemy import text
from sqlalchemy.orm import Session

import config
from clients.kalshi import KalshiClient
from clients.polymarket import PolymarketCLOBClient
from db.models import init_db

log = logging.getLogger("collectors.close_checker")

# Hard wall-clock cap for a single market HTTP call. requests' own timeout is
# read-only and does NOT bound a hung DNS getaddrinfo or a trickling keepalive
# socket — which is exactly what froze close_checker for 15h (Kalshi) and 4h+
# (Polymarket). This wrapper runs the call on a daemon thread and abandons it
# if it blows the deadline, so one bad socket can never stall the whole loop.
HTTP_DEADLINE_S = 45


class _DeadlineExceeded(Exception):
    pass


def _with_deadline(fn, *args, deadline: float = HTTP_DEADLINE_S):
    box: dict = {}

    def _run() -> None:
        try:
            box["v"] = fn(*args)
        except Exception as exc:  # noqa: BLE001 — re-raised on the caller thread
            box["e"] = exc

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(deadline)
    if t.is_alive():
        # Thread is wedged in a syscall with no timeout; leak it (daemon) and
        # move on. At a few stalls/day this never accumulates meaningfully.
        raise _DeadlineExceeded(f"hard {deadline:.0f}s deadline exceeded")
    if "e" in box:
        raise box["e"]
    return box.get("v")


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _oldest_kalshi(session: Session, batch: int) -> list[str]:
    rows = session.execute(
        text("""
            SELECT ticker FROM kalshi_markets
            WHERE status IS NULL OR status NOT IN ('closed', 'settled')
            ORDER BY fetched_at ASC NULLS FIRST
            LIMIT :batch
        """),
        {"batch": batch},
    ).fetchall()
    return [r[0] for r in rows]


def _oldest_polymarket(session: Session, batch: int) -> list[str]:
    rows = session.execute(
        text("""
            SELECT condition_id FROM polymarket_markets
            WHERE closed IS NULL OR closed = 0
            ORDER BY fetched_at ASC NULLS FIRST
            LIMIT :batch
        """),
        {"batch": batch},
    ).fetchall()
    return [r[0] for r in rows]


def _touch_kalshi(session: Session, ticker: str) -> None:
    """Bump fetched_at without changing other fields — for markets that 404'd."""
    session.execute(
        text("UPDATE kalshi_markets SET fetched_at = :now WHERE ticker = :t"),
        {"now": _now(), "t": ticker},
    )


def _touch_polymarket(session: Session, condition_id: str) -> None:
    session.execute(
        text("UPDATE polymarket_markets SET fetched_at = :now WHERE condition_id = :c"),
        {"now": _now(), "c": condition_id},
    )


def _is_http_404(exc: Exception) -> bool:
    if isinstance(exc, requests.HTTPError):
        resp = exc.response
        return resp is not None and resp.status_code == 404
    return False


def _refresh_kalshi(
    session: Session, client: KalshiClient, ticker: str
) -> tuple[str, Optional[str]]:
    """Returns (outcome, new_status). outcome is one of:
       'updated', 'unchanged', '404', 'error'."""
    try:
        market = _with_deadline(client.get_market, ticker)
    except _DeadlineExceeded as exc:
        log.warning("kalshi get_market(%s) %s — skipping", ticker, exc)
        _touch_kalshi(session, ticker)
        return ("error", None)
    except Exception as exc:
        if _is_http_404(exc):
            _touch_kalshi(session, ticker)
            return ("404", None)
        log.warning("kalshi get_market(%s) failed: %s", ticker, exc)
        return ("error", None)

    new_status = market.get("status")
    new_result = market.get("result")
    if not new_status:
        _touch_kalshi(session, ticker)
        return ("unchanged", None)

    session.execute(
        text("""
            UPDATE kalshi_markets
               SET status = :status,
                   result = :result,
                   settlement_value = :sv,
                   close_time = :ct,
                   expiration_time = :et,
                   settle_time = :st,
                   fetched_at = :now
             WHERE ticker = :t
        """),
        {
            "status": new_status,
            "result": new_result,
            "sv": market.get("settlement_value"),
            "ct": market.get("close_time"),
            "et": market.get("expiration_time"),
            "st": market.get("settle_time"),
            "now": _now(),
            "t": ticker,
        },
    )
    return ("updated", new_status)


def _refresh_polymarket(
    session: Session, clob: PolymarketCLOBClient, condition_id: str
) -> tuple[str, Optional[bool]]:
    try:
        market = _with_deadline(clob.get_market, condition_id)
    except _DeadlineExceeded as exc:
        log.warning("polymarket get_market(%s) %s — skipping", condition_id, exc)
        _touch_polymarket(session, condition_id)
        return ("error", None)
    except Exception as exc:
        if _is_http_404(exc):
            _touch_polymarket(session, condition_id)
            return ("404", None)
        log.warning("polymarket get_market(%s) failed: %s", condition_id, exc)
        return ("error", None)

    if not market:
        _touch_polymarket(session, condition_id)
        return ("unchanged", None)

    closed = market.get("closed")
    active = market.get("active")
    accepting = market.get("accepting_orders")

    session.execute(
        text("""
            UPDATE polymarket_markets
               SET closed = :closed,
                   active = :active,
                   accepting_orders = :accepting,
                   fetched_at = :now
             WHERE condition_id = :c
        """),
        {
            "closed": bool(closed) if closed is not None else None,
            "active": bool(active) if active is not None else None,
            "accepting": bool(accepting) if accepting is not None else None,
            "now": _now(),
            "c": condition_id,
        },
    )
    return ("updated", bool(closed) if closed is not None else None)


def _kalshi_cycle(sf, client: KalshiClient, batch: int) -> None:
    with sf() as session:
        tickers = _oldest_kalshi(session, batch)

    if not tickers:
        log.info("kalshi: no candidates")
        return

    counts = {"updated": 0, "unchanged": 0, "404": 0, "error": 0}
    closed_now = 0
    t0 = time.time()
    for ticker in tickers:
        with sf() as session:
            outcome, new_status = _refresh_kalshi(session, client, ticker)
            session.commit()
        counts[outcome] += 1
        if outcome == "updated" and new_status in ("closed", "settled"):
            closed_now += 1
    log.info(
        "kalshi cycle  n=%d  updated=%d  closed_or_settled=%d  unchanged=%d  "
        "404=%d  error=%d  elapsed=%.1fs",
        len(tickers), counts["updated"], closed_now, counts["unchanged"],
        counts["404"], counts["error"], time.time() - t0,
    )


def _polymarket_cycle(sf, clob: PolymarketCLOBClient, batch: int) -> None:
    with sf() as session:
        cids = _oldest_polymarket(session, batch)

    if not cids:
        log.info("polymarket: no candidates")
        return

    counts = {"updated": 0, "unchanged": 0, "404": 0, "error": 0}
    closed_now = 0
    t0 = time.time()
    for cid in cids:
        with sf() as session:
            outcome, is_closed = _refresh_polymarket(session, clob, cid)
            session.commit()
        counts[outcome] += 1
        if outcome == "updated" and is_closed:
            closed_now += 1
    log.info(
        "polymarket cycle  n=%d  updated=%d  closed=%d  unchanged=%d  "
        "404=%d  error=%d  elapsed=%.1fs",
        len(cids), counts["updated"], closed_now, counts["unchanged"],
        counts["404"], counts["error"], time.time() - t0,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Refresh status for the oldest-checked markets on Kalshi and Polymarket."
    )
    parser.add_argument("--db", default="./data/markets.db")
    parser.add_argument("--batch", type=int, default=1000,
                        help="Markets per platform per cycle (default 1000).")
    parser.add_argument("--interval", type=int, default=600,
                        help="Seconds between cycles (default 600 = 10 min).")
    parser.add_argument("--no-kalshi", action="store_true")
    parser.add_argument("--no-polymarket", action="store_true")
    parser.add_argument("--once", action="store_true",
                        help="Run a single cycle and exit.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    sf = init_db(args.db)

    # Indexes for the oldest-first scans. Without these, a single cycle
    # scans 4M+ rows. CREATE INDEX IF NOT EXISTS is idempotent and cheap.
    with sf() as session:
        session.execute(text(
            "CREATE INDEX IF NOT EXISTS ix_kalshi_close_check "
            "ON kalshi_markets (fetched_at) WHERE status IS NULL OR status NOT IN ('closed','settled')"
        ))
        session.execute(text(
            "CREATE INDEX IF NOT EXISTS ix_polymarket_close_check "
            "ON polymarket_markets (fetched_at) WHERE closed IS NULL OR closed = 0"
        ))
        session.commit()

    kalshi_client = None
    if not args.no_kalshi:
        kalshi_client = KalshiClient(
            api_key_id=config.KALSHI_API_KEY_ID,
            private_key_pem=config.KALSHI_PRIVATE_KEY,
            rate_limit=config.KALSHI_RATE_LIMIT,
        )

    poly_clob = None
    if not args.no_polymarket:
        poly_clob = PolymarketCLOBClient(rate_limit=config.POLYMARKET_RATE_LIMIT)

    log.info(
        "close_checker started — batch=%d  interval=%ds  kalshi=%s  polymarket=%s",
        args.batch, args.interval, not args.no_kalshi, not args.no_polymarket,
    )

    while True:
        if kalshi_client is not None:
            try:
                _kalshi_cycle(sf, kalshi_client, args.batch)
            except Exception as exc:
                log.warning("kalshi cycle aborted: %s", exc)
        if poly_clob is not None:
            try:
                _polymarket_cycle(sf, poly_clob, args.batch)
            except Exception as exc:
                log.warning("polymarket cycle aborted: %s", exc)

        if args.once:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
