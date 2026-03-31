"""
Bulk-collect resolved Polymarket markets to a parquet file.

Fetches all closed markets from Polymarket's Gamma API and writes them to a
parquet file for use in cross-platform evaluation.

Usage:
    python collect_resolved_polymarket.py [--output PATH] [--limit N]

After running, use eval_relationships.py with --platform kalshi,polymarket
and --poly-resolved PATH to evaluate cross-platform pair accuracy.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
from datetime import datetime, timezone

import pandas as pd

import config
from clients.polymarket import PolymarketGammaClient

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%H:%M:%S")


def _parse_prices(market: dict) -> list[float]:
    """Parse outcomePrices from either a list or JSON string."""
    raw = market.get("outcomePrices") or []
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return []
    try:
        return [float(p) for p in raw]
    except (ValueError, TypeError):
        return []


def _is_resolved(market: dict) -> bool:
    """True if the market has a clear YES or NO resolution price."""
    prices = _parse_prices(market)
    if not prices:
        return False
    try:
        p = prices[0]
        if math.isnan(p):
            return False
        # Gamma API returns 1.0/0.0 for fully resolved, or 0.99/0.01 for near-resolved
        return p >= 0.95 or p <= 0.05
    except (ValueError, TypeError):
        return False


def market_to_row(market: dict) -> dict:
    prices = _parse_prices(market)
    price_yes = prices[0] if prices else None
    return {
        "id": market.get("conditionId", ""),
        "platform": "polymarket",
        "question": (market.get("question") or "").strip(),
        "outcome": "YES" if (price_yes is not None and price_yes >= 0.95) else "NO",
        "price_yes": price_yes,
        "end_date": market.get("endDate"),
        "volume": market.get("volume"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect resolved Polymarket markets")
    parser.add_argument("--output", default="./data/polymarket_resolved.csv")
    parser.add_argument("--limit", type=int, default=0,
                        help="Stop after N markets (0 = fetch all)")
    args = parser.parse_args()

    client = PolymarketGammaClient(rate_limit=config.POLYMARKET_RATE_LIMIT)

    rows = []
    total = 0
    batch = 0

    print(f"[{_ts()}] Fetching resolved Polymarket markets → {args.output}")

    try:
        for market in client.iter_markets(closed=True, active=False, order="volume", ascending=False):
            cid = market.get("conditionId", "")
            if not cid or not market.get("question"):
                continue

            total += 1
            batch += 1

            if _is_resolved(market):
                rows.append(market_to_row(market))

            if batch >= 500:
                print(f"[{_ts()}]  {total:,} fetched  {len(rows):,} resolved", flush=True)
                batch = 0

            if args.limit and total >= args.limit:
                break

    except KeyboardInterrupt:
        print("\nInterrupted.")

    if rows:
        df = pd.DataFrame(rows).drop_duplicates("id")
        df.to_csv(args.output, index=False)
        print(f"\n[{_ts()}] Done — {total:,} fetched, {len(df):,} resolved saved to {args.output}")
    else:
        print(f"\n[{_ts()}] Done — {total:,} fetched, 0 resolved found")


if __name__ == "__main__":
    main()
