"""
NLI-based same-outcome classifier for prediction market pairs.

Wraps cross-encoder/nli-deberta-v3-large to classify whether two prediction
market questions resolve the same way (ENTAILMENT) or oppositely (CONTRADICTION).

Provides two interfaces:
  NLIClassifier         — simple classify(q_a, q_b) → (bool|None, float)
  HuggingFaceNLIBackend — full LLMBackend implementation; handles the
                          _PAIRS_USER prompt format used by relationships.py
                          so it can be dropped in as a local, API-free backend.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Optional

from forecasting.llm import LLMBackend

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


class HuggingFaceNLIBackend(LLMBackend):
    """
    Full LLMBackend implementation backed by a local NLI model.

    No API key required — model is downloaded from HuggingFace Hub on first use
    and cached locally (~900 MB for cross-encoder/nli-deberta-v3-large).

    Implements the same generate() interface as OpenAICompatibleBackend so it
    can be used anywhere a LLMBackend is expected, including relationships.py's
    _discover_pairs_in_group which uses the _PAIRS_USER prompt format.

    When the prompt contains '"pairs"' (the _PAIRS_USER format), it:
      - Extracts all numbered questions from the prompt
      - Runs NLI on every (i, j) pair
      - Returns JSON matching the expected {category, pairs} schema

    Category labelling is keyword-based (no model call needed).
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
        self._pipeline = None  # lazy-loaded

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

    def _keyword_category(self, text: str) -> str:
        p = text.lower()
        if any(w in p for w in ("btc", "bitcoin", "crypto", "eth")):
            return "crypto"
        if any(w in p for w in ("election", "president", "vote", "senate")):
            return "elections"
        if any(w in p for w in ("fed", "rate", "inflation", "gdp", "economy",
                                 "unemployment", "recession", "cpi")):
            return "economy"
        return "other"

    def _discover_pairs(self, prompt: str) -> str:
        self._load()
        category = self._keyword_category(prompt)
        questions = re.findall(r"^\d+\.\s+(.+)$", prompt, re.MULTILINE)
        if len(questions) < 2:
            return json.dumps({"category": category, "pairs": []})

        pairs: list[dict] = []
        for i in range(len(questions)):
            for j in range(i + 1, len(questions)):
                q_a, q_b = questions[i], questions[j]
                premise    = f'If the answer to "{q_a}" is YES, then'
                hypothesis = f'the answer to "{q_b}" is also YES'
                results = self._pipeline({"text": premise, "text_pair": hypothesis})
                scores  = {r["label"].upper(): r["score"] for r in results}
                ent = scores.get("ENTAILMENT", 0.0)
                con = scores.get("CONTRADICTION", 0.0)
                if ent >= self._ent_thresh:
                    pairs.append({
                        "question_a": q_a, "question_b": q_b,
                        "is_same_outcome": True,
                        "confidence_score": round(float(ent), 4),
                        "rationale": f"NLI entailment={ent:.2f}",
                    })
                elif con >= self._con_thresh:
                    pairs.append({
                        "question_a": q_a, "question_b": q_b,
                        "is_same_outcome": False,
                        "confidence_score": round(float(con), 4),
                        "rationale": f"NLI contradiction={con:.2f}",
                    })

        return json.dumps({"category": category, "pairs": pairs})

    async def generate(
        self,
        user_prompt: str,
        system_prompt: str = "",
        max_new_tokens: int = 2048,
    ) -> str:
        if '"pairs"' in user_prompt:
            return await asyncio.to_thread(self._discover_pairs, user_prompt)
        return json.dumps({"category": self._keyword_category(user_prompt)})
