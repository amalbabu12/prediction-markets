"""
Polymarket data collector.

Collection order:
  1. Tags (Gamma — fast)
  2. Events (Gamma — paginated)
  3. Markets (Gamma + CLOB merge — paginated)
  4. Price history for each token (CLOB /prices-history — heavy)
  5. Trades per market (Data API)
  6. Orderbook snapshots (CLOB — periodic)
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from clients.polymarket import (
    PolymarketCLOBClient,
    PolymarketDataClient,
    PolymarketGammaClient,
)
from db.models import (
    PolymarketEvent,
    PolymarketMarket,
    PolymarketOBSnap,
    PolymarketPriceHistory,
    PolymarketTrade,
    to_json,
)

logger = logging.getLogger(__name__)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _extract_token_ids(market: dict) -> tuple[Optional[str], Optional[str]]:
    """
    Extract Yes and No token IDs from a Gamma or CLOB market dict.
    Gamma: market["clobTokenIds"] = ["YES_TOKEN_ID", "NO_TOKEN_ID"]
    CLOB:  market["tokens"] = [{"token_id": "...", "outcome": "Yes"}, ...]
    """
    # CLOB format
    tokens = market.get("tokens")
    if tokens and isinstance(tokens, list):
        yes_id = next((t["token_id"] for t in tokens if t.get("outcome", "").lower() == "yes"), None)
        no_id  = next((t["token_id"] for t in tokens if t.get("outcome", "").lower() == "no"), None)
        return yes_id, no_id

    # Gamma format (clobTokenIds may be a JSON-encoded string or a list)
    clob_ids = market.get("clobTokenIds")
    if clob_ids and isinstance(clob_ids, str):
        try:
            clob_ids = json.loads(clob_ids)
        except (json.JSONDecodeError, ValueError):
            clob_ids = []
    if isinstance(clob_ids, list) and len(clob_ids) >= 2:
        return clob_ids[0], clob_ids[1]

    return None, None


def _parse_price(val) -> Optional[float]:
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


class PolymarketCollector:
    """
    Orchestrates fetching and persisting Polymarket data from all three APIs.

    Usage:
        from clients.polymarket import PolymarketGammaClient, PolymarketCLOBClient, PolymarketDataClient
        from db.models import init_db
        from collectors.polymarket_collector import PolymarketCollector

        Session = init_db("./data/markets.db")
        collector = PolymarketCollector(
            gamma=PolymarketGammaClient(),
            clob=PolymarketCLOBClient(),
            data=PolymarketDataClient(),
            session_factory=Session,
        )

        collector.collect_events()
        collector.collect_markets()
        collector.collect_price_history(fidelity=60)  # hourly bars, full history
        collector.collect_trades()
    """

    def __init__(
        self,
        gamma: PolymarketGammaClient,
        clob: PolymarketCLOBClient,
        data: PolymarketDataClient,
        session_factory,
    ) -> None:
        self.gamma = gamma
        self.clob = clob
        self.data = data
        self.Session = session_factory

    # ── Events ────────────────────────────────────────────────────────────────

    def collect_events(
        self,
        active: Optional[bool] = None,
        closed: Optional[bool] = None,
    ) -> int:
        """
        Upsert all events from the Gamma API.

        Passing active=None, closed=None fetches everything (active + closed).
        Returns total count.
        """
        # Fetch active and closed events separately for full coverage
        all_events: dict[str, dict] = {}
        for kwargs in [
            {"active": True, "closed": False, "archived": False},
            {"active": False, "closed": True},
            {"archived": True},
        ]:
            for e in self.gamma.iter_events(**kwargs):
                eid = e.get("id")
                if eid:
                    all_events[eid] = e

        logger.info("Fetched %d unique Polymarket events.", len(all_events))

        with self.Session() as session:
            count = 0
            for eid, e in all_events.items():
                obj = session.get(PolymarketEvent, eid) or PolymarketEvent(id=eid)
                obj.slug = e.get("slug")
                obj.title = e.get("title")
                obj.description = e.get("description")
                obj.active = e.get("active")
                obj.closed = e.get("closed")
                obj.archived = e.get("archived")
                obj.volume = _parse_price(e.get("volume"))
                obj.volume_24h = _parse_price(e.get("volume24hr"))
                obj.liquidity = _parse_price(e.get("liquidity"))
                obj.start_date = e.get("startDate") or e.get("start_date")
                obj.end_date = e.get("endDate") or e.get("end_date")
                obj.tags = to_json(e.get("tags", []))
                obj.raw_json = to_json(e)
                obj.fetched_at = _now()
                session.merge(obj)
                count += 1

                if count % 500 == 0:
                    session.commit()

            session.commit()

        logger.info("Polymarket events upserted: %d.", count)
        return count

    # ── Markets ───────────────────────────────────────────────────────────────

    def collect_markets(
        self,
        include_closed: bool = True,
        include_archived: bool = True,
        page_size: int = 500,
    ) -> int:
        """
        Collect markets from the Gamma API, writing each page to DB immediately.

        Uses bulk INSERT OR REPLACE so there is no per-row SELECT, and memory
        usage stays flat regardless of total market count.

        Returns: total markets upserted.
        """
        query_sets = [{"active": True, "closed": False, "archived": False}]
        if include_closed:
            query_sets.append({"active": False, "closed": True})
        if include_archived:
            query_sets.append({"archived": True})

        total = 0
        fetched_at = _now()

        for kwargs in query_sets:
            label = "active" if kwargs.get("active") else ("closed" if kwargs.get("closed") else "archived")
            batch: list[dict] = []
            page = 0

            for m in self.gamma.iter_markets(limit=page_size, **kwargs):
                cid = m.get("conditionId") or m.get("condition_id")
                if not cid:
                    continue

                yes_id, no_id = _extract_token_ids(m)

                outcome_prices = m.get("outcomePrices", [])
                if isinstance(outcome_prices, str):
                    try:
                        outcome_prices = json.loads(outcome_prices)
                    except (json.JSONDecodeError, ValueError):
                        outcome_prices = []
                outcomes = m.get("outcomes", [])
                if isinstance(outcomes, str):
                    try:
                        outcomes = json.loads(outcomes)
                    except (json.JSONDecodeError, ValueError):
                        outcomes = []

                batch.append({
                    "condition_id": cid,
                    "question_id": m.get("questionID") or m.get("question_id"),
                    "event_id": str(m.get("eventId") or m.get("event_id") or ""),
                    "question": m.get("question"),
                    "description": m.get("description"),
                    "market_slug": m.get("slug") or m.get("market_slug"),
                    "active": m.get("active"),
                    "closed": m.get("closed"),
                    "archived": m.get("archived"),
                    "accepting_orders": m.get("acceptingOrders") or m.get("accepting_orders"),
                    "token_id_yes": yes_id,
                    "token_id_no": no_id,
                    "outcomes": to_json(outcomes),
                    "outcome_prices": to_json(outcome_prices),
                    "price_yes": _parse_price(outcome_prices[0]) if len(outcome_prices) > 0 else None,
                    "price_no": _parse_price(outcome_prices[1]) if len(outcome_prices) > 1 else None,
                    "volume": _parse_price(m.get("volume")),
                    "volume_24h": _parse_price(m.get("volume24hr")),
                    "liquidity": _parse_price(m.get("liquidity")),
                    "end_date": m.get("endDate") or m.get("end_date"),
                    "game_start_time": m.get("gameStartTime") or m.get("game_start_time"),
                    "seconds_delay": m.get("secondsDelay") or m.get("seconds_delay"),
                    "neg_risk": m.get("negRisk") or m.get("neg_risk"),
                    "fee_rate_bps": m.get("feeRateBps") or m.get("fee_rate_bps"),
                    "minimum_order_size": _parse_price(
                        m.get("minimumOrderSize") or m.get("minimum_order_size")
                    ),
                    "minimum_tick_size": _parse_price(
                        m.get("minimumTickSize") or m.get("minimum_tick_size")
                    ),
                    "resolution_source": m.get("resolutionSource") or m.get("resolution_source"),
                    "raw_json": to_json(m),
                    "fetched_at": fetched_at,
                })

                if len(batch) >= page_size:
                    self._bulk_upsert_markets(batch)
                    total += len(batch)
                    page += 1
                    logger.info("Polymarket %s markets: %d upserted (page %d).", label, total, page)
                    batch = []

            if batch:
                self._bulk_upsert_markets(batch)
                total += len(batch)
                page += 1
                logger.info("Polymarket %s markets: %d upserted (page %d).", label, total, page)

        logger.info("Polymarket markets total upserted: %d.", total)
        return total

    def _bulk_upsert_markets(self, rows: list[dict]) -> None:
        """Insert or replace a batch of market rows."""
        stmt = sqlite_insert(PolymarketMarket).values(rows)
        stmt = stmt.on_conflict_do_update(
            index_elements=["condition_id"],
            set_={c: stmt.excluded[c] for c in rows[0] if c != "condition_id"},
        )
        with self.Session() as session:
            session.execute(stmt)
            session.commit()

    # ── Price History ─────────────────────────────────────────────────────────

    def collect_price_history(
        self,
        fidelity: int = 60,
        max_markets: Optional[int] = None,
        only_active: bool = False,
    ) -> int:
        """
        Fetch full price history for all markets' Yes and No tokens.

        Args:
            fidelity: Bar resolution in minutes.
                      1 = 1-minute bars (large), 60 = hourly (recommended), 1440 = daily
            max_markets: Stop after N markets (useful for testing)
            only_active: If True, only fetch history for currently active markets

        Returns: total price point rows inserted.
        """
        with self.Session() as session:
            q = session.query(PolymarketMarket)
            if only_active:
                q = q.filter(PolymarketMarket.active == True)
            markets = q.all()

        if max_markets:
            markets = markets[:max_markets]

        logger.info(
            "Collecting price history for %d markets (fidelity=%dm).", len(markets), fidelity
        )

        total_inserted = 0
        with self.Session() as session:
            for mkt in tqdm(markets, desc="PM price history", unit="market"):
                tokens = [
                    (mkt.token_id_yes, "YES"),
                    (mkt.token_id_no, "NO"),
                ]
                for token_id, outcome in tokens:
                    if not token_id:
                        continue
                    try:
                        history = self.clob.get_full_price_history(token_id, fidelity=fidelity)
                    except Exception as exc:
                        logger.warning(
                            "Price history error %s (%s): %s", mkt.condition_id, outcome, exc
                        )
                        continue

                    for point in history:
                        ts = point.get("t")
                        price = point.get("p")
                        if ts is None or price is None:
                            continue

                        existing = (
                            session.query(PolymarketPriceHistory)
                            .filter_by(token_id=token_id, fidelity=fidelity, ts=ts)
                            .first()
                        )
                        if existing:
                            continue

                        obj = PolymarketPriceHistory(
                            token_id=token_id,
                            condition_id=mkt.condition_id,
                            outcome=outcome,
                            fidelity=fidelity,
                            ts=ts,
                            price=float(price),
                        )
                        session.add(obj)
                        total_inserted += 1

                    if total_inserted % 10_000 == 0 and total_inserted:
                        session.commit()
                        logger.debug("Price history: %d rows committed.", total_inserted)

            session.commit()

        logger.info("Price history rows inserted: %d.", total_inserted)
        return total_inserted

    # ── Trades ────────────────────────────────────────────────────────────────

    def collect_trades(
        self,
        max_markets: Optional[int] = None,
        only_active: bool = False,
    ) -> int:
        """
        Collect trades for all markets from the Data API.
        Note: Data API returns up to 500 trades per market (most recent).
        For deeper history use CLOB /trades.

        Returns: total trade rows inserted.
        """
        with self.Session() as session:
            q = session.query(PolymarketMarket)
            if only_active:
                q = q.filter(PolymarketMarket.active == True)
            markets = q.all()

        if max_markets:
            markets = markets[:max_markets]

        logger.info("Collecting trades for %d markets.", len(markets))
        total_inserted = 0

        with self.Session() as session:
            for mkt in tqdm(markets, desc="PM trades", unit="market"):
                # Try Data API first (richer fields)
                try:
                    trades = self.data.get_trades(market=mkt.condition_id, limit=500)
                except Exception:
                    trades = []

                # Fall back to CLOB /trades if Data API returned nothing
                if not trades:
                    try:
                        trades = self.clob.get_trades(mkt.condition_id, limit=500)
                    except Exception as exc:
                        logger.warning("Trades error %s: %s", mkt.condition_id, exc)
                        continue

                for t in trades:
                    # Build a stable dedup key from available fields
                    tx = t.get("transactionHash") or t.get("transaction_hash") or ""
                    ts = t.get("timestamp") or t.get("created_at") or ""
                    trade_id_key = f"{mkt.condition_id}:{tx}:{ts}"

                    existing = (
                        session.query(PolymarketTrade)
                        .filter_by(trade_id=trade_id_key)
                        .first()
                    )
                    if existing:
                        continue

                    asset_id = t.get("asset_id") or t.get("assetId")
                    outcome = "YES" if asset_id == mkt.token_id_yes else (
                        "NO" if asset_id == mkt.token_id_no else None
                    )

                    obj = PolymarketTrade(
                        trade_id=trade_id_key,
                        condition_id=mkt.condition_id,
                        token_id=asset_id,
                        outcome=outcome,
                        side=t.get("side"),
                        size=_parse_price(t.get("size")),
                        price=_parse_price(t.get("price")),
                        fee_rate_bps=t.get("fee_rate_bps") or t.get("feeRateBps"),
                        tx_hash=tx,
                        trade_time=ts,
                        raw_json=to_json(t),
                    )
                    session.add(obj)
                    total_inserted += 1

                if total_inserted % 5000 == 0 and total_inserted:
                    session.commit()

            session.commit()

        logger.info("Polymarket trades inserted: %d.", total_inserted)
        return total_inserted

    # ── Orderbook Snapshots ───────────────────────────────────────────────────

    def snapshot_orderbooks(
        self,
        max_markets: Optional[int] = None,
        only_active: bool = True,
        batch_size: int = 20,
    ) -> int:
        """
        Snapshot orderbooks for all active markets (both Yes and No tokens).

        Uses batch POST /books endpoint for efficiency (20 tokens per call).

        Returns: number of snapshots stored.
        """
        with self.Session() as session:
            q = session.query(PolymarketMarket)
            if only_active:
                q = q.filter(PolymarketMarket.active == True)
            markets = q.all()

        if max_markets:
            markets = markets[:max_markets]

        # Build flat list of (token_id, condition_id, outcome) tuples
        tokens: list[tuple[str, str, str]] = []
        for m in markets:
            if m.token_id_yes:
                tokens.append((m.token_id_yes, m.condition_id, "YES"))
            if m.token_id_no:
                tokens.append((m.token_id_no, m.condition_id, "NO"))

        logger.info("Snapshotting orderbooks for %d tokens (%d markets).", len(tokens), len(markets))
        snap_time = _now()
        count = 0

        with self.Session() as session:
            for i in range(0, len(tokens), batch_size):
                batch = tokens[i : i + batch_size]
                token_ids = [t[0] for t in batch]
                try:
                    books = self.clob.get_orderbooks_batch(token_ids)
                except Exception as exc:
                    logger.warning("Batch orderbook error (batch %d): %s", i // batch_size, exc)
                    continue

                # books may be a list or dict keyed by token_id
                if isinstance(books, list):
                    book_map = {b.get("asset_id"): b for b in books if b}
                elif isinstance(books, dict):
                    book_map = books
                else:
                    continue

                for token_id, condition_id, outcome in batch:
                    book = book_map.get(token_id, {})
                    obj = PolymarketOBSnap(
                        token_id=token_id,
                        condition_id=condition_id,
                        outcome=outcome,
                        snapshot_time=snap_time,
                        bids=to_json(book.get("bids", [])),
                        asks=to_json(book.get("asks", [])),
                        raw_json=to_json(book),
                    )
                    session.add(obj)
                    count += 1

                if count % 200 == 0:
                    session.commit()

            session.commit()

        logger.info("Polymarket orderbook snapshots stored: %d.", count)
        return count
