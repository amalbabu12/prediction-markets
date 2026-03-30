"""
Bulk-collect resolved Polymarket markets into the DB.

Fetches all closed markets from Polymarket's Gamma API and upserts them into
the polymarket_markets table. Run this once to build a resolved dataset for
cross-platform evaluation.

Usage:
    python collect_resolved_polymarket.py [--db PATH] [--limit N]

After running, use eval_relationships.py with --platform kalshi,polymarket
to evaluate cross-platform pair accuracy.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from datetime import datetime, timezone

import config
from clients.polymarket import PolymarketGammaClient
from db.models import init_db, PolymarketMarket

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%H:%M:%S")


def _is_resolved(market: dict) -> bool:
    """True if the market has a clear YES or NO resolution price."""
    prices = market.get("outcomePrices") or []
    try:
        p = float(prices[0]) if prices else None
        if p is None or math.isnan(p):
            return False
        return p >= 0.99 or p <= 0.01
    except (ValueError, TypeError):
        return False


def upsert(sf, market: dict) -> None:
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
        condition_id=market.get("conditionId", ""),
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
    with sf() as session:
        session.merge(row)
        session.commit()


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect resolved Polymarket markets")
    parser.add_argument("--db", default="./data/markets.db")
    parser.add_argument("--limit", type=int, default=0,
                        help="Stop after N markets (0 = fetch all)")
    args = parser.parse_args()

    sf = init_db(args.db)
    client = PolymarketGammaClient(rate_limit=config.POLYMARKET_RATE_LIMIT)

    total = 0
    resolved = 0
    batch = 0

    print(f"[{_ts()}] Fetching resolved Polymarket markets → {args.db}")

    try:
        for market in client.iter_markets(closed=True, active=False, order="volume", ascending=False):
            cid = market.get("conditionId", "")
            if not cid or not market.get("question"):
                continue

            total += 1
            batch += 1

            if _is_resolved(market):
                upsert(sf, market)
                resolved += 1

            if batch >= 500:
                print(f"[{_ts()}]  {total:,} fetched  {resolved:,} resolved", flush=True)
                batch = 0

            if args.limit and total >= args.limit:
                break

    except KeyboardInterrupt:
        print("\nInterrupted.")

    print(f"\n[{_ts()}] Done — {total:,} markets fetched, {resolved:,} resolved stored in DB")


if __name__ == "__main__":
    main()
