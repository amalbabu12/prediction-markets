"""
Stream new markets from Kalshi and Polymarket, with optional DB persistence.

Neither platform pushes "market created" events over WebSocket — their WS APIs
are for price/orderbook updates on already-known markets. Instead, this script
polls both REST APIs on an interval and prints any markets it hasn't seen before.

First pass: silently seeds the seen-set with all currently open markets.
Subsequent passes: prints only markets that appear after startup.

When --db is passed, every market seen is upserted into the local SQLite DB
on every poll (so prices stay current). New markets are still printed.
The poller runs until --duration hours have elapsed, then exits cleanly.

Usage:
    python stream_markets.py [--interval SECONDS] [--db PATH] [--duration HOURS]

Examples:
    python stream_markets.py                                    # print only, forever
    python stream_markets.py --db ./data/markets.db \\
        --interval 900 --duration 10                           # 15-min polls for 10h
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import threading
import time
from datetime import datetime, timezone
from typing import Optional

import config
from clients.kalshi import KalshiClient
from clients.polymarket import PolymarketGammaClient

logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s %(name)s: %(message)s",
)


# ── Formatting ────────────────────────────────────────────────────────────────

def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%H:%M:%S")


def print_kalshi(market: dict) -> None:
    ticker = market.get("ticker", "?")
    title = market.get("title", "")
    close = market.get("close_time", "")[:10]
    yes_ask = market.get("yes_ask", "?")
    print(f"[{_ts()}] KALSHI  {ticker:<40s}  yes_ask={yes_ask:>3}¢  closes={close}  {title[:60]}")


def print_polymarket(market: dict) -> None:
    cid = market.get("conditionId", "?")
    question = market.get("question", "")
    end = (market.get("endDate") or "")[:10]
    volume = market.get("volume", 0)
    try:
        vol_str = f"${float(volume):,.0f}"
    except (TypeError, ValueError):
        vol_str = str(volume)
    print(f"[{_ts()}] POLY    {cid[:12]}...  vol={vol_str:<10s}  ends={end}  {question[:60]}")


# ── DB persistence ────────────────────────────────────────────────────────────

def _upsert_kalshi(SessionFactory, market: dict) -> None:
    """Upsert a Kalshi market dict into kalshi_markets (merge on ticker PK)."""
    from db.models import KalshiMarket
    ticker = market.get("ticker", "")
    if not ticker:
        return
    row = KalshiMarket(
        ticker=ticker,
        event_ticker=market.get("event_ticker"),
        series_ticker=market.get("series_ticker"),
        title=market.get("title"),
        subtitle=market.get("subtitle"),
        status=market.get("status"),
        market_type=market.get("market_type"),
        yes_bid=market.get("yes_bid"),
        yes_ask=market.get("yes_ask"),
        no_bid=market.get("no_bid"),
        no_ask=market.get("no_ask"),
        last_price=market.get("last_price"),
        previous_yes_bid=market.get("previous_yes_bid"),
        previous_yes_ask=market.get("previous_yes_ask"),
        previous_price=market.get("previous_price"),
        volume=market.get("volume"),
        volume_24h=market.get("volume_24h"),
        open_interest=market.get("open_interest"),
        liquidity=market.get("liquidity"),
        result=market.get("result"),
        settlement_value=market.get("settlement_value"),
        open_time=market.get("open_time"),
        close_time=market.get("close_time"),
        expected_expiration_time=market.get("expected_expiration_time"),
        expiration_time=market.get("expiration_time"),
        settle_time=market.get("settle_time"),
        raw_json=json.dumps(market, default=str),
    )
    with SessionFactory() as session:
        session.merge(row)
        session.commit()


def _upsert_polymarket(SessionFactory, market: dict) -> None:
    """Upsert a Polymarket Gamma market dict into polymarket_markets (merge on condition_id PK)."""
    from db.models import PolymarketMarket
    cid = market.get("conditionId", "")
    if not cid:
        return

    clob_tokens = market.get("clobTokenIds") or []
    token_yes = clob_tokens[0] if len(clob_tokens) > 0 else None
    token_no  = clob_tokens[1] if len(clob_tokens) > 1 else None

    outcome_prices = market.get("outcomePrices") or []
    try:
        price_yes = float(outcome_prices[0]) if outcome_prices else None
        price_no  = float(outcome_prices[1]) if len(outcome_prices) > 1 else None
    except (ValueError, TypeError):
        price_yes = price_no = None

    row = PolymarketMarket(
        condition_id=cid,
        question_id=market.get("questionId"),
        event_id=str(market.get("eventId", "") or ""),
        question=market.get("question"),
        description=market.get("description"),
        market_slug=market.get("slug"),
        active=market.get("active"),
        closed=market.get("closed"),
        archived=market.get("archived"),
        accepting_orders=market.get("acceptingOrders"),
        token_id_yes=token_yes,
        token_id_no=token_no,
        outcomes=json.dumps(market.get("outcomes") or []),
        outcome_prices=json.dumps(outcome_prices),
        price_yes=price_yes,
        price_no=price_no,
        volume=market.get("volume"),
        volume_24h=market.get("volume24hr"),
        liquidity=market.get("liquidity"),
        end_date=market.get("endDate"),
        game_start_time=market.get("gameStartTime"),
        neg_risk=market.get("negRisk"),
        fee_rate_bps=market.get("feeRateBps"),
        minimum_order_size=market.get("minimumOrderSize"),
        minimum_tick_size=market.get("minimumTickSize"),
        resolution_source=market.get("resolutionSource"),
        raw_json=json.dumps(market, default=str),
    )
    with SessionFactory() as session:
        session.merge(row)
        session.commit()


# ── Pollers ───────────────────────────────────────────────────────────────────

def poll_kalshi(
    client: KalshiClient,
    interval: int,
    deadline: float,
    SessionFactory=None,
    lookback: int = 300,
    on_new_market=None,
    seed_first_pass: bool = False,
) -> None:
    """
    Poll Kalshi for open markets created in the last `lookback` seconds.

    Uses min_created_ts on every poll so only recently-created markets
    are fetched. The seen-set deduplicates markets that span poll boundaries.

    If SessionFactory is provided, every market found is upserted.

    If seed_first_pass=True, the first poll silently fills the seen-set and
    upserts to the DB without firing on_new_market. This prevents a burst of
    LLM calls on startup for markets that pre-date the process launch.
    """
    seen: set[str] = set()
    is_seed = seed_first_pass

    while time.monotonic() < deadline:
        try:
            min_ts = int(time.time()) - lookback
            batch_count = 0

            for market in client.iter_markets(status="open", min_created_ts=min_ts):
                ticker = market.get("ticker", "")
                if not ticker:
                    continue

                if SessionFactory:
                    _upsert_kalshi(SessionFactory, market)
                    batch_count += 1

                if ticker not in seen:
                    seen.add(ticker)
                    if not is_seed:
                        print_kalshi(market)
                        if on_new_market is not None:
                            on_new_market(market)

            if SessionFactory:
                label = "seed" if is_seed else "poll"
                print(f"[{_ts()}] KALSHI  {label} upserted {batch_count} markets", flush=True)

        except Exception as exc:
            print(f"[{_ts()}] KALSHI  poll error: {exc}", file=sys.stderr)

        is_seed = False  # only the first iteration is a seed pass
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        time.sleep(min(interval, remaining))

    print(f"[{_ts()}] KALSHI  done — {len(seen)} total new markets seen")


def poll_polymarket(
    client: PolymarketGammaClient,
    interval: int,
    deadline: float,
    SessionFactory=None,
    lookback: int = 300,
    on_new_market=None,
    seed_first_pass: bool = False,
) -> None:
    """
    Poll Polymarket for active markets created in the last `lookback` seconds.

    Uses start_date_min on every poll so only recently-created markets
    are fetched. The seen-set deduplicates markets that span poll boundaries.

    If SessionFactory is provided, every market found is upserted.

    If seed_first_pass=True, the first poll silently fills the seen-set and
    upserts to the DB without firing on_new_market. This prevents a burst of
    LLM calls on startup for markets that pre-date the process launch.
    """
    seen: set[str] = set()
    is_seed = seed_first_pass

    while time.monotonic() < deadline:
        try:
            cutoff = datetime.fromtimestamp(time.time() - lookback, tz=timezone.utc).isoformat()
            new_markets: list[dict] = []
            batch_count = 0

            for market in client.iter_markets(
                active=True, closed=False,
                order="volume", ascending=False,
                start_date_min=cutoff,
            ):
                cid = market.get("conditionId", "")
                if not cid:
                    continue

                if SessionFactory:
                    _upsert_polymarket(SessionFactory, market)
                    batch_count += 1

                if cid not in seen:
                    seen.add(cid)
                    if not is_seed:
                        new_markets.append(market)

            for market in reversed(new_markets):
                print_polymarket(market)
                if on_new_market is not None:
                    on_new_market(market)

            if SessionFactory:
                label = "seed" if is_seed else "poll"
                print(f"[{_ts()}] POLY    {label} upserted {batch_count} markets", flush=True)

        except Exception as exc:
            print(f"[{_ts()}] POLY    poll error: {exc}", file=sys.stderr)

        is_seed = False  # only the first iteration is a seed pass
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        time.sleep(min(interval, remaining))

    print(f"[{_ts()}] POLY    done — {len(seen)} total markets seen")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Stream new Kalshi + Polymarket markets")
    parser.add_argument(
        "--interval", type=int, default=30,
        help="Poll interval in seconds (default: 30)",
    )
    parser.add_argument(
        "--duration", type=float, default=0,
        help="Stop after this many hours (default: 0 = run forever)",
    )
    parser.add_argument(
        "--db", type=str, default=None,
        help="SQLite DB path for persistence, e.g. ./data/markets.db",
    )
    parser.add_argument("--no-kalshi", action="store_true", help="Skip Kalshi")
    parser.add_argument("--no-polymarket", action="store_true", help="Skip Polymarket")
    args = parser.parse_args()

    deadline = (
        time.monotonic() + args.duration * 3600
        if args.duration > 0
        else float("inf")
    )
    duration_str = f"{args.duration}h" if args.duration > 0 else "∞"
    db_str = args.db or "none"
    print(
        f"Streaming markets  |  poll every {args.interval}s"
        f"  |  duration={duration_str}  |  db={db_str}  |  Ctrl+C to stop\n"
    )

    SessionFactory = None
    if args.db:
        from db.models import init_db
        SessionFactory = init_db(args.db)

    kalshi_client = KalshiClient(
        api_key_id=config.KALSHI_API_KEY_ID,
        private_key_pem=config.KALSHI_PRIVATE_KEY,
        rate_limit=config.KALSHI_RATE_LIMIT,
    )
    gamma_client = PolymarketGammaClient(rate_limit=config.POLYMARKET_RATE_LIMIT)

    threads: list[threading.Thread] = []

    if not args.no_kalshi:
        threads.append(threading.Thread(
            target=poll_kalshi,
            args=(kalshi_client, args.interval, deadline, SessionFactory),
            daemon=True,
            name="kalshi-poller",
        ))

    if not args.no_polymarket:
        threads.append(threading.Thread(
            target=poll_polymarket,
            args=(gamma_client, args.interval, deadline, SessionFactory),
            daemon=True,
            name="polymarket-poller",
        ))

    for t in threads:
        t.start()

    try:
        for t in threads:
            t.join()
        print("\nAll pollers finished.")
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
