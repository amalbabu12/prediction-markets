"""
Kalshi data collector.

Collection order:
  1. Series (fast — O(100) records)
  2. Events — all statuses, paginated
  3. Markets — all statuses, paginated
  4. Trades — per market or global, paginated
  5. Candlesticks — 1m / 60m / 1440m per market (heavy — throttled)
  6. Orderbook snapshots — open markets only (requires auth or public fallback)
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Optional

from tqdm import tqdm

from clients.kalshi import KalshiClient
from db.models import (
    KalshiCandlestick,
    KalshiEvent,
    KalshiMarket,
    KalshiOrderbookSnap,
    KalshiSeries,
    KalshiTrade,
    to_json,
)
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _ohlc(d: dict | None, field: str, key: str) -> Optional[int]:
    if not d:
        return None
    sub = d.get(field)
    if not sub:
        return None
    return sub.get(key)


class KalshiCollector:
    """
    Orchestrates fetching and persisting Kalshi data.

    Usage:
        from clients.kalshi import KalshiClient
        from db.models import init_db
        from collectors.kalshi_collector import KalshiCollector

        Session = init_db("./data/markets.db")
        client = KalshiClient()          # no auth needed for most data
        collector = KalshiCollector(client, Session)

        collector.collect_series()
        collector.collect_events()
        collector.collect_markets()
        collector.collect_trades()       # global trade stream
        collector.collect_candlesticks() # heavy — open/closed markets
    """

    def __init__(self, client: KalshiClient, session_factory) -> None:
        self.client = client
        self.Session = session_factory

    # ── Series ────────────────────────────────────────────────────────────────

    def collect_series(self) -> int:
        """Upsert all series. Returns number of series stored."""
        series_list = self.client.get_series_list(include_volume=True)
        logger.info("Fetched %d Kalshi series.", len(series_list))

        with self.Session() as session:
            for s in series_list:
                obj = session.get(KalshiSeries, s.get("ticker")) or KalshiSeries(
                    ticker=s.get("ticker")
                )
                obj.title = s.get("title")
                obj.category = s.get("category")
                obj.frequency = s.get("frequency")
                obj.tags = to_json(s.get("tags", []))
                obj.raw_json = to_json(s)
                obj.fetched_at = _now()
                session.merge(obj)
            session.commit()

        logger.info("Upserted %d series.", len(series_list))
        return len(series_list)

    # ── Events ────────────────────────────────────────────────────────────────

    def collect_events(
        self,
        statuses: tuple[str, ...] = ("open", "closed", "settled"),
        with_nested_markets: bool = False,
    ) -> int:
        """Upsert all events. Returns total count."""
        total = 0
        for status in statuses:
            count = 0
            with self.Session() as session:
                for e in self.client.iter_events(
                    status=status, with_nested_markets=with_nested_markets
                ):
                    obj = session.get(KalshiEvent, e.get("event_ticker")) or KalshiEvent(
                        event_ticker=e.get("event_ticker")
                    )
                    obj.series_ticker = e.get("series_ticker")
                    obj.title = e.get("title")
                    obj.sub_title = e.get("sub_title")
                    obj.category = e.get("category")
                    obj.mutually_exclusive = e.get("mutually_exclusive")
                    obj.status = e.get("status")
                    obj.close_time = e.get("close_time")
                    obj.settle_time = e.get("settle_time")
                    obj.expected_expiration_time = e.get("expected_expiration_time")
                    obj.raw_json = to_json(e)
                    obj.fetched_at = _now()
                    session.merge(obj)
                    count += 1

                    if count % 200 == 0:
                        session.commit()
                        logger.debug("Events (%s): %d committed.", status, count)

                session.commit()
            logger.info("Events status=%s: %d upserted.", status, count)
            total += count

        return total

    # ── Markets ───────────────────────────────────────────────────────────────

    def collect_markets(
        self,
        statuses: tuple[str, ...] = ("unopened", "open", "paused", "closed", "settled"),
    ) -> int:
        """Upsert all markets. Returns total count."""
        total = 0
        for status in statuses:
            count = 0
            with self.Session() as session:
                for m in self.client.iter_markets(status=status):
                    obj = session.get(KalshiMarket, m.get("ticker")) or KalshiMarket(
                        ticker=m.get("ticker")
                    )
                    obj.event_ticker = m.get("event_ticker")
                    obj.series_ticker = m.get("series_ticker")
                    obj.title = m.get("title")
                    obj.subtitle = m.get("subtitle")
                    obj.status = m.get("status")
                    obj.market_type = m.get("market_type")
                    obj.yes_bid = m.get("yes_bid")
                    obj.yes_ask = m.get("yes_ask")
                    obj.no_bid = m.get("no_bid")
                    obj.no_ask = m.get("no_ask")
                    obj.last_price = m.get("last_price")
                    obj.previous_yes_bid = m.get("previous_yes_bid")
                    obj.previous_yes_ask = m.get("previous_yes_ask")
                    obj.previous_price = m.get("previous_price")
                    obj.volume = m.get("volume")
                    obj.volume_24h = m.get("volume_24h")
                    obj.open_interest = m.get("open_interest")
                    obj.liquidity = m.get("liquidity")
                    obj.result = m.get("result")
                    obj.settlement_value = m.get("settlement_value")
                    obj.open_time = m.get("open_time")
                    obj.close_time = m.get("close_time")
                    obj.expected_expiration_time = m.get("expected_expiration_time")
                    obj.expiration_time = m.get("expiration_time")
                    obj.settle_time = m.get("settle_time")
                    obj.raw_json = to_json(m)
                    obj.fetched_at = _now()
                    session.merge(obj)
                    count += 1

                    if count % 500 == 0:
                        session.commit()
                        logger.debug("Markets (%s): %d committed.", status, count)

                session.commit()
            logger.info("Markets status=%s: %d upserted.", status, count)
            total += count

        return total

    # ── Trades ────────────────────────────────────────────────────────────────

    def collect_trades_global(
        self,
        min_ts: Optional[int] = None,
        max_ts: Optional[int] = None,
    ) -> int:
        """
        Collect the global trade stream (all markets).
        This is the most efficient way to ingest historical trades.
        """
        count = 0
        skipped = 0
        with self.Session() as session:
            for t in self.client.iter_trades(min_ts=min_ts, max_ts=max_ts):
                trade_id = t.get("trade_id")
                if not trade_id:
                    continue
                # Skip if already stored
                if session.get(KalshiTrade, trade_id):
                    skipped += 1
                    continue
                obj = KalshiTrade(
                    trade_id=trade_id,
                    ticker=t.get("ticker"),
                    yes_price=t.get("yes_price"),
                    no_price=t.get("no_price"),
                    count=t.get("count"),
                    taker_side=t.get("taker_side"),
                    created_time=t.get("created_time"),
                    raw_json=to_json(t),
                )
                session.add(obj)
                count += 1

                if count % 1000 == 0:
                    session.commit()
                    logger.debug("Trades: %d inserted (%d skipped).", count, skipped)

            session.commit()

        logger.info("Trades: %d inserted, %d already existed.", count, skipped)
        return count

    def collect_trades_for_market(self, ticker: str) -> int:
        """Collect all trades for a specific market."""
        count = 0
        with self.Session() as session:
            for t in self.client.iter_trades(ticker=ticker):
                trade_id = t.get("trade_id")
                if not trade_id or session.get(KalshiTrade, trade_id):
                    continue
                obj = KalshiTrade(
                    trade_id=trade_id,
                    ticker=ticker,
                    yes_price=t.get("yes_price"),
                    no_price=t.get("no_price"),
                    count=t.get("count"),
                    taker_side=t.get("taker_side"),
                    created_time=t.get("created_time"),
                    raw_json=to_json(t),
                )
                session.add(obj)
                count += 1
            session.commit()
        return count

    # ── Candlesticks ──────────────────────────────────────────────────────────

    def collect_candlesticks(
        self,
        start_ts: int,
        end_ts: int,
        statuses: tuple[str, ...] = ("open", "closed", "settled"),
        intervals: tuple[int, ...] = (1, 60, 1440),
        max_markets: Optional[int] = None,
    ) -> int:
        """
        Fetch OHLCV candlesticks for all markets with a known series_ticker.

        Args:
            start_ts: History start (Unix seconds)
            end_ts: History end (Unix seconds)
            statuses: Which market statuses to include
            intervals: Candle resolutions in minutes (1, 60, and/or 1440)
            max_markets: Stop after this many markets (useful for testing)

        Returns:
            Total number of candle rows inserted.
        """
        # Load markets from DB that have a series_ticker
        with self.Session() as session:
            markets_q = (
                session.query(KalshiMarket)
                .filter(KalshiMarket.status.in_(statuses))
                .filter(KalshiMarket.series_ticker.isnot(None))
                .all()
            )
        markets = markets_q[:max_markets] if max_markets else markets_q
        logger.info("Collecting candlesticks for %d markets.", len(markets))

        total_inserted = 0
        with self.Session() as session:
            for m in tqdm(markets, desc="Kalshi candlesticks", unit="market"):
                for interval in intervals:
                    try:
                        candles = self.client.get_candlesticks(
                            m.series_ticker, m.ticker, start_ts, end_ts, interval
                        )
                    except Exception as exc:
                        logger.warning(
                            "Candle error %s %dm: %s", m.ticker, interval, exc
                        )
                        continue

                    for c in candles:
                        ts = c.get("end_period_ts")
                        # Check for existing row
                        existing = (
                            session.query(KalshiCandlestick)
                            .filter_by(ticker=m.ticker, period_interval=interval, end_period_ts=ts)
                            .first()
                        )
                        if existing:
                            continue

                        obj = KalshiCandlestick(
                            ticker=m.ticker,
                            series_ticker=m.series_ticker,
                            period_interval=interval,
                            end_period_ts=ts,
                            yes_bid_open=_ohlc(c, "yes_bid", "open"),
                            yes_bid_high=_ohlc(c, "yes_bid", "high"),
                            yes_bid_low=_ohlc(c, "yes_bid", "low"),
                            yes_bid_close=_ohlc(c, "yes_bid", "close"),
                            yes_ask_open=_ohlc(c, "yes_ask", "open"),
                            yes_ask_high=_ohlc(c, "yes_ask", "high"),
                            yes_ask_low=_ohlc(c, "yes_ask", "low"),
                            yes_ask_close=_ohlc(c, "yes_ask", "close"),
                            price_open=_ohlc(c, "price", "open"),
                            price_high=_ohlc(c, "price", "high"),
                            price_low=_ohlc(c, "price", "low"),
                            price_close=_ohlc(c, "price", "close"),
                            volume=c.get("volume"),
                            open_interest=c.get("open_interest"),
                            raw_json=to_json(c),
                        )
                        session.add(obj)
                        total_inserted += 1

                    if total_inserted % 5000 == 0 and total_inserted:
                        session.commit()

            session.commit()

        logger.info("Candlesticks: %d rows inserted.", total_inserted)
        return total_inserted

    # ── Orderbook Snapshots ───────────────────────────────────────────────────

    def snapshot_orderbooks(
        self,
        tickers: Optional[list[str]] = None,
        depth: int = 0,
    ) -> int:
        """
        Take a single orderbook snapshot for each specified ticker (or all open markets).

        Args:
            tickers: List of market tickers. If None, fetches open markets from DB.
            depth: 0 = all levels; 1-100 = top N levels per side.

        Returns:
            Number of snapshots stored.
        """
        if tickers is None:
            with self.Session() as session:
                open_markets = (
                    session.query(KalshiMarket.ticker)
                    .filter(KalshiMarket.status == "open")
                    .all()
                )
            tickers = [r[0] for r in open_markets]

        logger.info("Snapshotting orderbooks for %d markets.", len(tickers))
        count = 0
        snap_time = _now()

        with self.Session() as session:
            for ticker in tqdm(tickers, desc="Kalshi orderbooks", unit="market"):
                try:
                    book = self.client.get_orderbook(ticker, depth=depth)
                except Exception as exc:
                    logger.warning("Orderbook error %s: %s", ticker, exc)
                    continue

                if not book:
                    continue

                obj = KalshiOrderbookSnap(
                    ticker=ticker,
                    snapshot_time=snap_time,
                    yes_bids=to_json(book.get("yes", [])),
                    no_bids=to_json(book.get("no", [])),
                    raw_json=to_json(book),
                )
                session.add(obj)
                count += 1

                if count % 100 == 0:
                    session.commit()

            session.commit()

        logger.info("Orderbook snapshots stored: %d.", count)
        return count
