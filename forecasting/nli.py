"""
NLI-based same-outcome classifier for prediction market pairs.

Wraps cross-encoder/nli-deberta-v3-large to classify whether two prediction
market questions resolve the same way (ENTAILMENT) or oppositely (CONTRADICTION).
"""
from __future__ import annotations

import asyncio
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class NLIClassifier:
    """
    Classify whether two prediction market questions resolve the same way.

    Uses cross-encoder/nli-deberta-v3-large to compute entailment/contradiction
    probability for the pair.

    Returns (is_same_outcome, confidence):
      True  + score  if P(ENTAILMENT)    >= entailment_threshold
      False + score  if P(CONTRADICTION) >= contradiction_threshold
      None  + score  if neither threshold met (ambiguous)
    """

    MODEL_ID = "cross-encoder/nli-deberta-v3-large"

    def __init__(
        self,
        entailment_threshold: float = 0.25,
        contradiction_threshold: float = 0.25,
        device: str = "cpu",
    ) -> None:
        self._ent_thresh = entailment_threshold
        self._con_thresh = contradiction_threshold
        self._device = device
        self._pipeline = None  # lazy-loaded on first call

    def _load(self) -> None:
        if self._pipeline is not None:
            return
        import warnings
        warnings.filterwarnings("ignore", category=UserWarning)
        from transformers import pipeline as hf_pipeline
        logger.info("Loading %s on %s ...", self.MODEL_ID, self._device)
        self._pipeline = hf_pipeline(
            "text-classification",
            model=self.MODEL_ID,
            device=self._device,
            top_k=None,
        )
        logger.info("NLI model ready.")

    def classify(
        self,
        question_a: str,
        question_b: str,
    ) -> tuple[Optional[bool], float]:
        """
        Classify a pair of market questions.

        Args:
            question_a: Question from market A
            question_b: Question from market B

        Returns:
            (is_same_outcome, confidence)
            is_same_outcome: True (same), False (opposite), None (ambiguous)
            confidence: score of the winning label
        """
        self._load()

        premise    = f'If the answer to "{question_a}" is YES, then'
        hypothesis = f'the answer to "{question_b}" is also YES'

        results = self._pipeline({"text": premise, "text_pair": hypothesis})
        scores = {r["label"].upper(): r["score"] for r in results}

        ent = scores.get("ENTAILMENT", 0.0)
        con = scores.get("CONTRADICTION", 0.0)

        if ent >= self._ent_thresh:
            return True, float(ent)
        if con >= self._con_thresh:
            return False, float(con)
        return None, max(ent, con)

    async def classify_async(
        self,
        question_a: str,
        question_b: str,
    ) -> tuple[Optional[bool], float]:
        """Async wrapper — runs classify() in a thread pool."""
        return await asyncio.to_thread(self.classify, question_a, question_b)
