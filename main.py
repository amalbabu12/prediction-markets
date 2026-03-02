"""
Prediction Markets Data Collector
==================================
Fetches data from Kalshi and Polymarket and stores it in a local SQLite DB.

Usage:
    python main.py --help
    python main.py snapshot              # Quick snapshot: markets + orderbooks
    python main.py history               # Full historical data (slow)
    python main.py continuous --interval 300  # Repeat snapshot every 5 minutes
    python main.py kalshi-only snapshot
    python main.py polymarket-only snapshot
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime, timezone

import config
from clients.kalshi import KalshiClient
from clients.polymarket import (
    PolymarketCLOBClient,
    PolymarketDataClient,
    PolymarketGammaClient,
)
from collectors.kalshi_collector import KalshiCollector
from collectors.polymarket_collector import PolymarketCollector
from db.models import init_db

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("main")


# ── Default time range for historical data ────────────────────────────────────
# Go back 2 years by default
_TWO_YEARS_AGO = int(datetime(2023, 1, 1, tzinfo=timezone.utc).timestamp())
_NOW = int(datetime.now(timezone.utc).timestamp())


def build_clients() -> tuple[KalshiClient, PolymarketGammaClient, PolymarketCLOBClient, PolymarketDataClient]:
    kalshi = KalshiClient(
        api_key_id=config.KALSHI_API_KEY_ID,
        private_key_pem=config.KALSHI_PRIVATE_KEY,
        rate_limit=config.KALSHI_RATE_LIMIT,
    )
    gamma = PolymarketGammaClient(rate_limit=config.POLYMARKET_RATE_LIMIT)
    clob = PolymarketCLOBClient(rate_limit=80.0)
    data = PolymarketDataClient(rate_limit=15.0)
    return kalshi, gamma, clob, data


def run_outcomes(
    kalshi_collector: KalshiCollector,
    pm_collector: PolymarketCollector,
    skip_kalshi: bool = False,
    skip_polymarket: bool = False,
) -> None:
    """
    Collect only market questions and their resolved outcomes. Fast — no orderbooks,
    no price history, no trades. Best mode for building a dataset of questions and
    binary outcomes across platforms.

    Kalshi:      series + markets (all statuses, including settled with result=yes/no)
    Polymarket:  markets only (question text + price_yes/price_no on closed markets)
    """
    if not skip_kalshi:
        logger.info("=== Kalshi outcomes ===")
        kalshi_collector.collect_series()
        kalshi_collector.collect_markets()

    if not skip_polymarket:
        logger.info("=== Polymarket outcomes ===")
        pm_collector.collect_markets()

    logger.info("Outcomes collection complete.")


def run_snapshot(
    kalshi_collector: KalshiCollector,
    pm_collector: PolymarketCollector,
    skip_kalshi: bool = False,
    skip_polymarket: bool = False,
) -> None:
    """
    Fast snapshot: refresh market metadata and take orderbook snapshots.
    Doesn't collect full trade or candlestick history.
    """
    if not skip_kalshi:
        logger.info("=== Kalshi snapshot ===")
        kalshi_collector.collect_series()
        kalshi_collector.collect_events()
        kalshi_collector.collect_markets()
        kalshi_collector.snapshot_orderbooks()

    if not skip_polymarket:
        logger.info("=== Polymarket snapshot ===")
        pm_collector.collect_events()
        pm_collector.collect_markets()
        pm_collector.snapshot_orderbooks(only_active=True)

    logger.info("Snapshot complete.")


def run_history(
    kalshi_collector: KalshiCollector,
    pm_collector: PolymarketCollector,
    skip_kalshi: bool = False,
    skip_polymarket: bool = False,
    start_ts: int = _TWO_YEARS_AGO,
    end_ts: int = _NOW,
    candle_intervals: tuple[int, ...] = (60, 1440),
    price_fidelity: int = 60,
) -> None:
    """
    Full historical data collection. This is slow and should be run once
    after the initial snapshot to backfill historical OHLCV and trade data.
    """
    if not skip_kalshi:
        logger.info("=== Kalshi history ===")
        # Markets must be in DB first
        kalshi_collector.collect_series()
        kalshi_collector.collect_events()
        kalshi_collector.collect_markets()
        logger.info("Collecting Kalshi global trades...")
        kalshi_collector.collect_trades_global(min_ts=start_ts, max_ts=end_ts)
        logger.info("Collecting Kalshi candlesticks (%s)...", candle_intervals)
        kalshi_collector.collect_candlesticks(
            start_ts=start_ts,
            end_ts=end_ts,
            intervals=candle_intervals,
        )

    if not skip_polymarket:
        logger.info("=== Polymarket history ===")
        pm_collector.collect_events()
        pm_collector.collect_markets()
        logger.info("Collecting Polymarket price history (fidelity=%dm)...", price_fidelity)
        pm_collector.collect_price_history(fidelity=price_fidelity)
        logger.info("Collecting Polymarket trades...")
        pm_collector.collect_trades()

    logger.info("History collection complete.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prediction markets data collector for Kalshi and Polymarket."
    )
    parser.add_argument(
        "mode",
        choices=["outcomes", "snapshot", "history", "continuous"],
        help=(
            "outcomes:   questions + resolved outcomes only — no orderbooks (fastest).\n"
            "snapshot:   refresh market metadata + orderbooks (fast).\n"
            "history:    full historical candles + trades (slow, run once).\n"
            "continuous: repeat snapshot on an interval."
        ),
    )
    parser.add_argument("--db", default=config.DB_PATH, help="SQLite database path.")
    parser.add_argument("--no-kalshi", action="store_true", help="Skip Kalshi collection.")
    parser.add_argument("--no-polymarket", action="store_true", help="Skip Polymarket collection.")
    parser.add_argument(
        "--interval",
        type=int,
        default=300,
        help="Seconds between snapshots in continuous mode (default: 300).",
    )
    parser.add_argument(
        "--start-ts",
        type=int,
        default=_TWO_YEARS_AGO,
        help="History start as Unix timestamp (default: 2023-01-01).",
    )
    parser.add_argument(
        "--end-ts",
        type=int,
        default=_NOW,
        help="History end as Unix timestamp (default: now).",
    )
    parser.add_argument(
        "--price-fidelity",
        type=int,
        default=60,
        choices=[1, 5, 10, 30, 60, 1440],
        help="Polymarket price history bar size in minutes (default: 60).",
    )
    parser.add_argument(
        "--candle-intervals",
        nargs="+",
        type=int,
        default=[60, 1440],
        choices=[1, 60, 1440],
        help="Kalshi candle resolutions in minutes (default: 60 1440).",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging.")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info("Initializing database at %s", args.db)
    Session = init_db(args.db)

    kalshi_client, gamma, clob, data_client = build_clients()
    kalshi_collector = KalshiCollector(kalshi_client, Session)
    pm_collector = PolymarketCollector(gamma, clob, data_client, Session)

    if args.mode == "outcomes":
        run_outcomes(
            kalshi_collector,
            pm_collector,
            skip_kalshi=args.no_kalshi,
            skip_polymarket=args.no_polymarket,
        )

    elif args.mode == "snapshot":
        run_snapshot(
            kalshi_collector,
            pm_collector,
            skip_kalshi=args.no_kalshi,
            skip_polymarket=args.no_polymarket,
        )

    elif args.mode == "history":
        run_history(
            kalshi_collector,
            pm_collector,
            skip_kalshi=args.no_kalshi,
            skip_polymarket=args.no_polymarket,
            start_ts=args.start_ts,
            end_ts=args.end_ts,
            candle_intervals=tuple(args.candle_intervals),
            price_fidelity=args.price_fidelity,
        )

    elif args.mode == "continuous":
        logger.info("Continuous mode: snapshot every %ds. Ctrl-C to stop.", args.interval)
        try:
            while True:
                t0 = time.monotonic()
                run_snapshot(
                    kalshi_collector,
                    pm_collector,
                    skip_kalshi=args.no_kalshi,
                    skip_polymarket=args.no_polymarket,
                )
                elapsed = time.monotonic() - t0
                sleep_for = max(0.0, args.interval - elapsed)
                logger.info("Next snapshot in %.0fs.", sleep_for)
                time.sleep(sleep_for)
        except KeyboardInterrupt:
            logger.info("Stopped.")
            sys.exit(0)


if __name__ == "__main__":
    main()
