"""
Load resolved market questions and outcomes from the database.

Produces a unified DataFrame with columns:
  id          - unique market identifier (ticker or condition_id)
  platform    - "kalshi" or "polymarket"
  question    - the full question text
  outcome     - "YES", "NO", or None (unresolved)
  resolved_at - ISO datetime string of resolution, or None
"""
from __future__ import annotations

import logging
from typing import Optional

import pandas as pd
from sqlalchemy.orm import sessionmaker

from db.models import KalshiMarket, PolymarketMarket

logger = logging.getLogger(__name__)


def load_markets(
    session_factory: sessionmaker,
    platforms: tuple[str, ...] = ("kalshi", "polymarket"),
    resolved_only: bool = False,
    max_markets: Optional[int] = None,
    pm_session_factory: Optional[sessionmaker] = None,
) -> pd.DataFrame:
    """
    Load all markets (or only resolved ones) into a unified DataFrame.

    Args:
        session_factory: SQLAlchemy sessionmaker — used for Kalshi (and Polymarket
                         if pm_session_factory is not provided)
        platforms: Which platforms to include
        resolved_only: If True, only return markets with a confirmed outcome
        max_markets: Cap total rows returned (useful for quick tests)
        pm_session_factory: Optional separate sessionmaker for Polymarket DB.
                            If omitted, Polymarket is read from session_factory.

    Returns:
        DataFrame with columns: id, platform, question, outcome, resolved_at
    """
    rows = []
    pm_sf = pm_session_factory or session_factory

    with session_factory() as session:
        if "kalshi" in platforms:
            markets = session.query(KalshiMarket).all()
            for m in markets:
                question = m.title or ""
                if m.subtitle:
                    question = f"{question} — {m.subtitle}"
                question = question.strip()
                if not question:
                    continue

                outcome: Optional[str] = None
                if m.result == "yes":
                    outcome = "YES"
                elif m.result == "no":
                    outcome = "NO"

                if resolved_only and outcome is None:
                    continue

                rows.append({
                    "id": m.ticker,
                    "platform": "kalshi",
                    "question": question,
                    "outcome": outcome,
                    "resolved_at": m.settle_time,
                })

    with pm_sf() as session:
        if "polymarket" in platforms:
            markets = session.query(PolymarketMarket).all()
            for m in markets:
                question = (m.question or "").strip()
                if not question:
                    continue

                outcome = None
                if m.closed and m.price_yes is not None:
                    if m.price_yes >= 0.99:
                        outcome = "YES"
                    elif m.price_yes <= 0.01:
                        outcome = "NO"

                if resolved_only and outcome is None:
                    continue

                rows.append({
                    "id": m.condition_id,
                    "platform": "polymarket",
                    "question": question,
                    "outcome": outcome,
                    "resolved_at": m.end_date,
                })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    if max_markets and len(df) > max_markets:
        df = df.sample(n=max_markets, random_state=42).reset_index(drop=True)

    logger.info(
        "Loaded %d markets (%d resolved) from %s",
        len(df),
        int(df["outcome"].notna().sum()),
        ", ".join(platforms),
    )
    return df
