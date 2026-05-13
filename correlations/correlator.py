"""Compute historical correlation for candidate_pairs older than the age cutoff.

Polls candidate_pairs WHERE correlator_processed_at IS NULL AND
created_at < now - CORRELATOR_AGE_DAYS, then for each pair:

  1. JIT-fetch hourly price history (last CORRELATOR_WINDOW_DAYS days) for both
     markets via Kalshi candlesticks / Polymarket CLOB prices-history.
  2. Time-align onto a shared hour grid.
  3. If we have ≥ CORRELATOR_MIN_SAMPLES joint observations, compute Pearson r
     and the mean / std of the spread (price_b − price_a).
  4. If |r| ≥ CORRELATOR_PEARSON_MIN, write to correlated_pairs (the
     divergence_watcher's input).
  5. Always stamp correlator_processed_at so the candidate isn't reprocessed.

No price-collection background worker — history is fetched on demand from the
platform APIs at the moment we need it.
"""
from __future__ import annotations

import argparse
import logging
import time
from typing import Optional

import numpy as np
from sqlalchemy import text
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session

import config
from clients.kalshi import KalshiClient
from clients.polymarket import PolymarketCLOBClient
from db.models import init_db
from correlations import models  # noqa: F401  — register tables with Base
from correlations.config import (
    CORRELATOR_AGE_DAYS,
    CORRELATOR_MIN_SAMPLES,
    CORRELATOR_PEARSON_MIN,
    CORRELATOR_POLL_INTERVAL_SEC,
    CORRELATOR_WINDOW_DAYS,
)
from correlations.models import CorrelatedPair

log = logging.getLogger("correlations.correlator")

HOUR = 3600


def _due_candidates(session: Session, limit: int) -> list[dict]:
    sql = """
    SELECT id, id_a, platform_a, id_b, platform_b, jaccard
    FROM candidate_pairs
    WHERE correlator_processed_at IS NULL
      AND created_at < datetime('now', :cutoff)
    ORDER BY created_at ASC
    LIMIT :limit
    """
    rows = session.execute(text(sql), {
        "cutoff": f"-{CORRELATOR_AGE_DAYS} days",
        "limit": limit,
    }).fetchall()
    return [
        {"id": r[0], "id_a": r[1], "platform_a": r[2],
         "id_b": r[3], "platform_b": r[4], "jaccard": r[5]}
        for r in rows
    ]


def _kalshi_lookup(session: Session, ticker: str) -> Optional[str]:
    row = session.execute(
        text("SELECT series_ticker FROM kalshi_markets WHERE ticker = :t"),
        {"t": ticker},
    ).fetchone()
    return row[0] if row and row[0] else None


def _polymarket_token_yes(session: Session, condition_id: str) -> Optional[str]:
    row = session.execute(
        text("SELECT token_id_yes FROM polymarket_markets WHERE condition_id = :c"),
        {"c": condition_id},
    ).fetchone()
    return row[0] if row and row[0] else None


def _fetch_history(
    session: Session,
    kalshi_client: KalshiClient,
    poly_clob: PolymarketCLOBClient,
    market_id: str,
    platform: str,
    start_ts: int,
    end_ts: int,
) -> list[tuple[int, float]]:
    """Return [(unix_ts, price_in_[0,1]), ...] hourly. Empty list on failure."""
    if platform == "kalshi":
        series = _kalshi_lookup(session, market_id)
        if not series:
            return []
        try:
            candles = kalshi_client.get_candlesticks(
                series_ticker=series, ticker=market_id,
                start_ts=start_ts, end_ts=end_ts, period_interval=60,
            )
        except Exception as exc:
            log.warning("kalshi history failed for %s: %s", market_id, exc)
            return []
        out: list[tuple[int, float]] = []
        for c in candles:
            ts = c.get("end_period_ts")
            close_cents = (c.get("price") or {}).get("close")
            if ts is None or close_cents is None:
                continue
            out.append((int(ts), float(close_cents) / 100.0))
        return out

    if platform == "polymarket":
        token = _polymarket_token_yes(session, market_id)
        if not token:
            return []
        try:
            points = poly_clob.get_price_history(
                token_id=token, start_ts=start_ts, end_ts=end_ts, fidelity=60,
            )
        except Exception as exc:
            log.warning("polymarket history failed for %s: %s", market_id, exc)
            return []
        return [(int(p["t"]), float(p["p"])) for p in points
                if p.get("t") is not None and p.get("p") is not None]

    return []


def _align_hourly(a: list[tuple[int, float]], b: list[tuple[int, float]]) -> tuple[np.ndarray, np.ndarray]:
    """Bucket each series into hour-aligned timestamps (last value wins per hour)
    and return arrays for the intersecting hours."""
    def bucketize(pts: list[tuple[int, float]]) -> dict[int, float]:
        out: dict[int, float] = {}
        for ts, p in pts:
            hour = (ts // HOUR) * HOUR
            out[hour] = p
        return out

    bucket_a = bucketize(a)
    bucket_b = bucketize(b)
    common = sorted(set(bucket_a) & set(bucket_b))
    if not common:
        return np.array([]), np.array([])
    return (
        np.array([bucket_a[t] for t in common]),
        np.array([bucket_b[t] for t in common]),
    )


def _process_candidate(
    session: Session,
    kalshi_client: KalshiClient,
    poly_clob: PolymarketCLOBClient,
    cand: dict,
    now_ts: int,
) -> tuple[bool, Optional[float], int]:
    """Returns (high_corr, pearson_r_or_None, n_samples)."""
    start_ts = now_ts - CORRELATOR_WINDOW_DAYS * 86400

    a_pts = _fetch_history(session, kalshi_client, poly_clob,
                           cand["id_a"], cand["platform_a"], start_ts, now_ts)
    b_pts = _fetch_history(session, kalshi_client, poly_clob,
                           cand["id_b"], cand["platform_b"], start_ts, now_ts)

    arr_a, arr_b = _align_hourly(a_pts, b_pts)
    n = len(arr_a)

    if n < CORRELATOR_MIN_SAMPLES:
        return (False, None, n)

    # Constant series → pearson is undefined; treat as no correlation.
    if arr_a.std() == 0 or arr_b.std() == 0:
        return (False, None, n)

    r = float(np.corrcoef(arr_a, arr_b)[0, 1])
    if not np.isfinite(r):
        return (False, None, n)

    if abs(r) < CORRELATOR_PEARSON_MIN:
        return (False, r, n)

    spread = arr_b - arr_a
    mean_spread = float(spread.mean())
    std_spread = float(spread.std())

    stmt = sqlite_insert(CorrelatedPair).values(
        id_a=cand["id_a"], platform_a=cand["platform_a"],
        id_b=cand["id_b"], platform_b=cand["platform_b"],
        pearson_r=r,
        n_samples=n,
        mean_spread=mean_spread,
        std_spread=std_spread,
    )
    stmt = stmt.on_conflict_do_nothing(index_elements=["id_a", "id_b"])
    session.execute(stmt)
    return (True, r, n)


def _mark_processed(session: Session, candidate_id: int) -> None:
    session.execute(
        text("UPDATE candidate_pairs SET correlator_processed_at = datetime('now') WHERE id = :id"),
        {"id": candidate_id},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute Pearson correlation for aged candidate pairs.")
    parser.add_argument("--db", default="./data/markets.db")
    parser.add_argument("--batch", type=int, default=20,
                        help="Max candidates processed per polling cycle (default: 20).")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    sf = init_db(args.db)

    kalshi_client = KalshiClient(
        api_key_id=config.KALSHI_API_KEY_ID,
        private_key_pem=config.KALSHI_PRIVATE_KEY,
        rate_limit=config.KALSHI_RATE_LIMIT,
    )
    poly_clob = PolymarketCLOBClient(rate_limit=config.POLYMARKET_RATE_LIMIT)

    log.info("correlator started — poll=%ds  age>=%dd  window=%dd  pearson_min=%.2f  min_samples=%d",
             CORRELATOR_POLL_INTERVAL_SEC, CORRELATOR_AGE_DAYS, CORRELATOR_WINDOW_DAYS,
             CORRELATOR_PEARSON_MIN, CORRELATOR_MIN_SAMPLES)

    while True:
        with sf() as session:
            candidates = _due_candidates(session, args.batch)

        if not candidates:
            time.sleep(CORRELATOR_POLL_INTERVAL_SEC)
            continue

        promoted = 0
        skipped_low_samples = 0
        skipped_low_corr = 0
        now_ts = int(time.time())

        for cand in candidates:
            with sf() as session:
                high_corr, r, n = _process_candidate(
                    session, kalshi_client, poly_clob, cand, now_ts,
                )
                _mark_processed(session, cand["id"])
                session.commit()

            if high_corr:
                promoted += 1
                log.info("promoted pair %s/%s ↔ %s/%s  r=%.3f  n=%d",
                         cand["platform_a"], cand["id_a"][:18],
                         cand["platform_b"], cand["id_b"][:18], r or 0, n)
            elif n < CORRELATOR_MIN_SAMPLES:
                skipped_low_samples += 1
            else:
                skipped_low_corr += 1

        log.info("cycle done — processed=%d  promoted=%d  low_samples=%d  low_corr=%d",
                 len(candidates), promoted, skipped_low_samples, skipped_low_corr)

        time.sleep(CORRELATOR_POLL_INTERVAL_SEC)


if __name__ == "__main__":
    main()
