"""
Cross-encoder based pair scoring — local alternative to the LLM relationship pipeline.

Uses `cross-encoder/quora-distilroberta-base` (or any CrossEncoder model) to score
whether two questions are asking the same thing. Trained on Quora duplicate question
pairs, so it distinguishes true duplicates from merely topically-similar questions.

Advantages over LLM:
  - No API key, no cost, runs locally
  - Batched inference — much faster on large pair sets
  - Deterministic (no temperature)
  - No JSON parsing issues

Trade-off:
  - Can't produce rationales
  - Weaker on subtle market-specific nuance vs a 70B reasoning model
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from forecasting.relationships import DiscoveredPair

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "cross-encoder/quora-distilroberta-base"
_DEFAULT_THRESHOLD = 0.5


class CrossEncoderScorer:
    """
    Wraps a sentence-transformers CrossEncoder for pairwise question scoring.

    Args:
        model_name: Any sentence-transformers cross-encoder model ID.
        threshold: Score >= threshold → is_same_outcome=True.
        batch_size: Inference batch size (tune for your hardware).
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        threshold: float = _DEFAULT_THRESHOLD,
        batch_size: int = 64,
    ) -> None:
        from sentence_transformers import CrossEncoder

        self.model_name = model_name
        self.threshold = threshold
        self.batch_size = batch_size
        self._model = CrossEncoder(model_name)
        logger.info(
            "CrossEncoderScorer loaded: %s  threshold=%.2f  batch_size=%d",
            model_name, threshold, batch_size,
        )

    def score(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        """
        Score a list of (question_a, question_b) pairs.

        Returns:
            np.ndarray of float scores in [0, 1], one per pair.
            Higher = more likely same question / same outcome.
        """
        scores = self._model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=len(pairs) > 200,
        )
        return np.array(scores, dtype=np.float32)


def discover_with_crossencoder(
    df: pd.DataFrame,
    neighbor_pairs: pd.DataFrame,
    scorer: CrossEncoderScorer,
    min_confidence: float = 0.0,
) -> list[DiscoveredPair]:
    """
    Score all neighbor pairs directly with the cross-encoder.

    Much simpler than the LLM pipeline: no grouping, no async, no JSON parsing.
    Each row in neighbor_pairs becomes exactly one DiscoveredPair.

    Args:
        df: Market DataFrame with columns: id, platform, question, outcome, resolved_at.
        neighbor_pairs: Output of find_neighbors().
        scorer: Initialised CrossEncoderScorer.
        min_confidence: Drop pairs whose score is below this threshold.

    Returns:
        Deduplicated list of DiscoveredPair, sorted by confidence descending.
    """
    if neighbor_pairs.empty:
        logger.warning("discover_with_crossencoder: neighbor_pairs is empty")
        return []

    id_to_row: dict[str, dict] = {row["id"]: row.to_dict() for _, row in df.iterrows()}

    # Deduplicate pairs before scoring (KNN produces A→B and B→A)
    seen_keys: set[tuple[str, str]] = set()
    deduped_rows = []
    for _, row in neighbor_pairs.iterrows():
        key = tuple(sorted([row["id_a"], row["id_b"]]))
        if key not in seen_keys:
            seen_keys.add(key)
            deduped_rows.append(row)

    logger.info(
        "Scoring %d unique pairs with %s ...", len(deduped_rows), scorer.model_name
    )

    pairs_input = [(row["question_a"], row["question_b"]) for row in deduped_rows]
    scores = scorer.score(pairs_input)

    result: list[DiscoveredPair] = []
    for row, score in zip(deduped_rows, scores):
        score_f = float(score)
        if score_f < min_confidence:
            continue

        row_a = id_to_row.get(row["id_a"])
        row_b = id_to_row.get(row["id_b"])
        if row_a is None or row_b is None:
            continue

        result.append(DiscoveredPair(
            anchor_id=row["id_a"],
            group_category="other",       # cross-encoder doesn't classify topics
            question_a=row["question_a"],
            question_b=row["question_b"],
            id_a=row["id_a"],
            id_b=row["id_b"],
            platform_a=row["platform_a"],
            platform_b=row["platform_b"],
            is_same_outcome=score_f >= scorer.threshold,
            confidence_score=score_f,
            rationale=f"cross-encoder score {score_f:.3f}",
            outcome_a=row_a.get("outcome"),
            outcome_b=row_b.get("outcome"),
            resolved_at_a=row_a.get("resolved_at"),
            resolved_at_b=row_b.get("resolved_at"),
        ))

    result.sort(key=lambda p: p.confidence_score, reverse=True)
    logger.info(
        "Cross-encoder: %d pairs above min_confidence=%.2f  "
        "(same_outcome=%d  diff_outcome=%d)",
        len(result),
        min_confidence,
        sum(1 for p in result if p.is_same_outcome),
        sum(1 for p in result if not p.is_same_outcome),
    )
    return result
