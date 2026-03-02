"""
SQLAlchemy models for the local backtesting database (SQLite).

Schema overview:
  Kalshi:
    kalshi_series           — market categories/series
    kalshi_events           — parent events grouping markets
    kalshi_markets          — individual binary/scalar markets
    kalshi_orderbook_snaps  — periodic order book snapshots
    kalshi_trades           — executed trades (public)
    kalshi_candlesticks     — OHLCV bars (1m / 60m / 1440m)

  Polymarket:
    polymarket_events       — events from Gamma API
    polymarket_markets      — markets (Gamma + CLOB merged)
    polymarket_ob_snaps     — order book snapshots (per outcome token)
    polymarket_trades       — trades from CLOB / Data API
    polymarket_price_hist   — price time series from CLOB /prices-history

Raw JSON is stored in every table so no information is discarded.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    event,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


# ══════════════════════════════════════════════════════════════════════════════
# Kalshi tables
# ══════════════════════════════════════════════════════════════════════════════

class KalshiSeries(Base):
    __tablename__ = "kalshi_series"

    ticker = Column(String, primary_key=True)
    title = Column(String)
    category = Column(String)
    frequency = Column(String)
    tags = Column(Text)          # JSON list
    raw_json = Column(Text)
    fetched_at = Column(DateTime, default=_now, onupdate=_now)


class KalshiEvent(Base):
    __tablename__ = "kalshi_events"

    event_ticker = Column(String, primary_key=True)
    series_ticker = Column(String, index=True)
    title = Column(String)
    sub_title = Column(String)
    category = Column(String)
    mutually_exclusive = Column(Boolean)
    status = Column(String, index=True)
    # Timestamps stored as UTC ISO strings (API returns ISO 8601)
    close_time = Column(String)
    settle_time = Column(String)
    expected_expiration_time = Column(String)
    raw_json = Column(Text)
    fetched_at = Column(DateTime, default=_now, onupdate=_now)


class KalshiMarket(Base):
    __tablename__ = "kalshi_markets"

    ticker = Column(String, primary_key=True)
    event_ticker = Column(String, index=True)
    series_ticker = Column(String, index=True)
    title = Column(String)
    subtitle = Column(String)
    status = Column(String, index=True)
    market_type = Column(String)       # "binary" | "scalar"
    # Prices in cents (1–99)
    yes_bid = Column(Integer)
    yes_ask = Column(Integer)
    no_bid = Column(Integer)
    no_ask = Column(Integer)
    last_price = Column(Integer)
    previous_yes_bid = Column(Integer)
    previous_yes_ask = Column(Integer)
    previous_price = Column(Integer)
    # Volume & OI
    volume = Column(Integer)
    volume_24h = Column(Integer)
    open_interest = Column(Integer)
    liquidity = Column(Integer)
    # Settlement
    result = Column(String)
    settlement_value = Column(Integer)
    # Timestamps
    open_time = Column(String)
    close_time = Column(String)
    expected_expiration_time = Column(String)
    expiration_time = Column(String)
    settle_time = Column(String)
    raw_json = Column(Text)
    fetched_at = Column(DateTime, default=_now, onupdate=_now)


class KalshiOrderbookSnap(Base):
    __tablename__ = "kalshi_orderbook_snaps"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String, index=True)
    snapshot_time = Column(DateTime, default=_now, index=True)
    # JSON: [[price_cents, quantity], ...]
    yes_bids = Column(Text)
    no_bids = Column(Text)
    raw_json = Column(Text)

    __table_args__ = (Index("ix_kalshi_ob_ticker_time", "ticker", "snapshot_time"),)


class KalshiTrade(Base):
    __tablename__ = "kalshi_trades"

    trade_id = Column(String, primary_key=True)
    ticker = Column(String, index=True)
    yes_price = Column(Integer)     # cents
    no_price = Column(Integer)      # cents
    count = Column(Integer)         # contracts
    taker_side = Column(String)     # "yes" | "no"
    created_time = Column(String, index=True)   # ISO 8601
    raw_json = Column(Text)


class KalshiCandlestick(Base):
    __tablename__ = "kalshi_candlesticks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String, index=True)
    series_ticker = Column(String)
    period_interval = Column(Integer)     # 1 | 60 | 1440
    end_period_ts = Column(Integer, index=True)  # Unix seconds
    # yes_bid OHLC (cents)
    yes_bid_open = Column(Integer)
    yes_bid_high = Column(Integer)
    yes_bid_low = Column(Integer)
    yes_bid_close = Column(Integer)
    # yes_ask OHLC (cents)
    yes_ask_open = Column(Integer)
    yes_ask_high = Column(Integer)
    yes_ask_low = Column(Integer)
    yes_ask_close = Column(Integer)
    # midpoint price OHLC (cents)
    price_open = Column(Integer)
    price_high = Column(Integer)
    price_low = Column(Integer)
    price_close = Column(Integer)
    volume = Column(Integer)
    open_interest = Column(Integer)
    raw_json = Column(Text)

    __table_args__ = (
        UniqueConstraint("ticker", "period_interval", "end_period_ts", name="uq_kalshi_candle"),
        Index("ix_kalshi_candle_ticker_interval", "ticker", "period_interval"),
    )


# ══════════════════════════════════════════════════════════════════════════════
# Polymarket tables
# ══════════════════════════════════════════════════════════════════════════════

class PolymarketEvent(Base):
    __tablename__ = "polymarket_events"

    id = Column(String, primary_key=True)
    slug = Column(String, unique=True, index=True)
    title = Column(String)
    description = Column(Text)
    active = Column(Boolean)
    closed = Column(Boolean)
    archived = Column(Boolean)
    volume = Column(Float)
    volume_24h = Column(Float)
    liquidity = Column(Float)
    start_date = Column(String)
    end_date = Column(String, index=True)
    tags = Column(Text)       # JSON list of tag objects
    raw_json = Column(Text)
    fetched_at = Column(DateTime, default=_now, onupdate=_now)


class PolymarketMarket(Base):
    """
    Merged view of Gamma + CLOB market data.

    condition_id  — Gamma / CLOB primary key (one per market)
    token_id_yes  — CLOB ERC1155 token ID for YES outcome
    token_id_no   — CLOB ERC1155 token ID for NO outcome
    Prices in [0.0, 1.0] (decimal probability).
    """
    __tablename__ = "polymarket_markets"

    condition_id = Column(String, primary_key=True)
    question_id = Column(String)
    event_id = Column(String, index=True)     # FK → polymarket_events.id
    question = Column(Text)
    description = Column(Text)
    market_slug = Column(String, index=True)
    active = Column(Boolean, index=True)
    closed = Column(Boolean)
    archived = Column(Boolean)
    accepting_orders = Column(Boolean)
    # Outcome tokens
    token_id_yes = Column(String, index=True)
    token_id_no = Column(String, index=True)
    outcomes = Column(Text)       # JSON ["Yes", "No"] or multi-outcome
    outcome_prices = Column(Text) # JSON current prices per outcome
    # Prices
    price_yes = Column(Float)
    price_no = Column(Float)
    # Volume & liquidity
    volume = Column(Float)
    volume_24h = Column(Float)
    liquidity = Column(Float)
    # Metadata
    end_date = Column(String, index=True)
    game_start_time = Column(String)
    seconds_delay = Column(Integer)
    neg_risk = Column(Boolean)
    fee_rate_bps = Column(Integer)
    minimum_order_size = Column(Float)
    minimum_tick_size = Column(Float)
    resolution_source = Column(String)
    raw_json = Column(Text)
    fetched_at = Column(DateTime, default=_now, onupdate=_now)


class PolymarketOBSnap(Base):
    __tablename__ = "polymarket_ob_snaps"

    id = Column(Integer, primary_key=True, autoincrement=True)
    token_id = Column(String, index=True)
    condition_id = Column(String, index=True)
    outcome = Column(String)         # "YES" | "NO"
    snapshot_time = Column(DateTime, default=_now, index=True)
    # JSON: [{"price": "0.65", "size": "100"}, ...]
    bids = Column(Text)
    asks = Column(Text)
    raw_json = Column(Text)

    __table_args__ = (Index("ix_pm_ob_token_time", "token_id", "snapshot_time"),)


class PolymarketTrade(Base):
    __tablename__ = "polymarket_trades"

    id = Column(Integer, primary_key=True, autoincrement=True)
    # Use a unique constraint on external IDs to avoid duplicates
    trade_id = Column(String, unique=True, index=True, nullable=True)
    condition_id = Column(String, index=True)
    token_id = Column(String, index=True)
    outcome = Column(String)       # "YES" | "NO"
    side = Column(String)          # "BUY" | "SELL"
    size = Column(Float)
    price = Column(Float)
    fee_rate_bps = Column(Integer)
    tx_hash = Column(String)
    trade_time = Column(String, index=True)   # ISO 8601
    raw_json = Column(Text)


class PolymarketPriceHistory(Base):
    __tablename__ = "polymarket_price_hist"

    id = Column(Integer, primary_key=True, autoincrement=True)
    token_id = Column(String, index=True)
    condition_id = Column(String, index=True)
    outcome = Column(String)      # "YES" | "NO"
    fidelity = Column(Integer)    # resolution in minutes
    ts = Column(Integer, index=True)      # Unix timestamp
    price = Column(Float)

    __table_args__ = (
        UniqueConstraint("token_id", "fidelity", "ts", name="uq_pm_price_hist"),
        Index("ix_pm_price_token_fidelity", "token_id", "fidelity"),
    )


# ══════════════════════════════════════════════════════════════════════════════
# Engine / session factory
# ══════════════════════════════════════════════════════════════════════════════

def init_db(db_path: str) -> sessionmaker:
    """
    Create the SQLite database, enable WAL mode for better concurrency,
    create all tables, and return a sessionmaker.

    Args:
        db_path: Path to the SQLite file (e.g., "./data/markets.db").
                 Parent directory is created automatically.

    Returns:
        A bound sessionmaker — call it to get a Session.

    Example:
        Session = init_db("./data/markets.db")
        with Session() as session:
            session.add(...)
            session.commit()
    """
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(f"sqlite:///{db_path}", echo=False)

    # Enable WAL for better write concurrency and crash safety
    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_conn, _record):
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    Base.metadata.create_all(engine)
    return sessionmaker(engine, expire_on_commit=False)


# ── Helpers ───────────────────────────────────────────────────────────────────

def to_json(obj: Any) -> str:  # type: ignore[name-defined]
    return json.dumps(obj, default=str)

def from_json(s: str | None) -> Any:  # type: ignore[name-defined]
    return json.loads(s) if s else None
