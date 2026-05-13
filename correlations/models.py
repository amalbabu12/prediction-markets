"""SQLAlchemy models for the correlation pipeline.

Imports the existing Base from db.models so all tables live in one SQLite file.
Importing this module before init_db() is called is enough to register the
tables; create_all() picks them up.
"""
from __future__ import annotations

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)

from db.models import Base, _now


class TagDictionary(Base):
    __tablename__ = "tag_dictionary"

    tag = Column(String, primary_key=True)
    description = Column(Text)
    market_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=_now)
    updated_at = Column(DateTime, default=_now, onupdate=_now)


class MarketTag(Base):
    __tablename__ = "market_tags"

    market_id = Column(String, nullable=False)
    platform = Column(String, nullable=False)
    tag = Column(String, ForeignKey("tag_dictionary.tag"), nullable=False)
    confidence = Column(Float)
    tagged_at = Column(DateTime, default=_now)

    __table_args__ = (
        UniqueConstraint("market_id", "platform", "tag", name="uq_market_tag"),
        Index("ix_market_tags_lookup", "market_id", "platform"),
        Index("ix_market_tags_tag", "tag"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)


class CandidatePair(Base):
    """Tag-overlap pair, awaiting historical Pearson correlation."""

    __tablename__ = "candidate_pairs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    id_a = Column(String, nullable=False)
    platform_a = Column(String, nullable=False)
    id_b = Column(String, nullable=False)
    platform_b = Column(String, nullable=False)
    jaccard = Column(Float)
    shared_tags = Column(Text)             # JSON list
    created_at = Column(DateTime, default=_now, index=True)
    correlator_processed_at = Column(DateTime, index=True)  # NULL until correlator runs

    __table_args__ = (
        UniqueConstraint("id_a", "id_b", name="uq_candidate_pair"),
    )


class CorrelatedPair(Base):
    """A candidate pair whose historical Pearson r exceeded the threshold.
    Captures the spread baseline used for live divergence z-scores.
    """

    __tablename__ = "correlated_pairs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    id_a = Column(String, nullable=False)
    platform_a = Column(String, nullable=False)
    id_b = Column(String, nullable=False)
    platform_b = Column(String, nullable=False)
    pearson_r = Column(Float)
    n_samples = Column(Integer)
    mean_spread = Column(Float)            # μ over correlation window
    std_spread = Column(Float)             # σ over correlation window
    started_watching_at = Column(DateTime, default=_now, index=True)

    __table_args__ = (
        UniqueConstraint("id_a", "id_b", name="uq_correlated_pair"),
    )


class PriceDivergence(Base):
    __tablename__ = "price_divergences"

    id = Column(Integer, primary_key=True, autoincrement=True)
    pair_id = Column(Integer, ForeignKey("correlated_pairs.id"), index=True)
    price_a = Column(Float)
    price_b = Column(Float)
    spread = Column(Float)
    z_score = Column(Float)
    detected_at = Column(DateTime, default=_now, index=True)
