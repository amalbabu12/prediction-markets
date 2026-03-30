"""
LLM-based relationship discovery using KNN neighbor groups.

For each market A and its K nearest neighbors, the LLM:
  1. Labels the group with a topic category
  2. Finds pairs whose outcomes are likely related and predicts whether
     they resolve the same way (is_same_outcome=True) or oppositely (False)

Each market anchors one LLM call. Discovered pairs are deduplicated across
groups since the same pair may surface from both sides (A→B and B→A).

Uses prompt-based JSON output — works with any LLMBackend implementation.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from forecasting.llm import LLMBackend, extract_json

logger = logging.getLogger(__name__)

_CATEGORIES = [
    "politics", "geopolitics", "elections", "economy",
    "finance", "earnings", "crypto", "tech", "sports", "culture", "other",
]

# ── Prompts ───────────────────────────────────────────────────────────────────

_LABEL_SYSTEM = (
    "You are an expert at categorising prediction markets. "
    "Respond with valid JSON only — no explanation, no markdown."
)

_LABEL_USER = """\
Assign the following group of prediction market questions to exactly one category.

Valid categories: {categories}

Questions:
{numbered}

Respond with exactly this JSON:
{{"category": "<one of the valid categories>"}}"""


_PAIRS_SYSTEM = (
    "You are an expert at analysing prediction markets. "
    "Respond with valid JSON only — no explanation, no markdown, no code fences."
)

_PAIRS_USER = """\
Below are prediction market questions from different platforms. First assign the \
group a category, then identify pairs that are ARBITRAGE-WORTHY — i.e. where \
the two questions are so closely matched that buying YES on one and YES on the \
other is a near-certain combined win (or loss).

SAME OUTCOME (is_same_outcome=true): The two markets resolve on the EXACT SAME \
underlying event with the SAME threshold, the SAME time window, and the SAME \
direction. Both will resolve YES together or NO together with very high certainty. \
DO NOT use this just because markets are on the same topic. \
BAD examples (do NOT flag): "BTC above $80k by March" vs "BTC above $85k by April" \
(different threshold AND date), "Team A wins game 1" vs "Team A wins game 2" \
(different events), "BTC market cap #1" vs "ETH market cap #2" (different assets). \
GOOD examples: identical question text on two platforms, or questions that differ \
only in platform-specific wording but resolve on identical criteria.

DIFFERENT OUTCOME (is_same_outcome=false): Direct logical inverses where one \
resolving YES forces the other to resolve NO. Only genuine inverses — "price \
above X" vs "price below X" for the same asset, date, and threshold.

OMIT a pair entirely if you are not highly confident (>= 0.8) in the relationship. \
It is better to miss a pair than to flag a false arbitrage. \
When in doubt, leave it out.

Copy question text exactly as it appears below.

Valid categories: {categories}

Questions:
{numbered}

Respond with exactly this JSON format:
{{
  "category": "<one of the valid categories>",
  "pairs": [
    {{
      "question_a": "<exact text>",
      "question_b": "<exact text>",
      "is_same_outcome": true,
      "confidence_score": 0.85,
      "rationale": "one sentence explanation"
    }}
  ]
}}

Return {{"category": "other", "pairs": []}} if no high-confidence arbitrage pairs exist."""


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class DiscoveredPair:
    anchor_id: str              # the market that anchored this neighbor group
    group_category: str
    question_a: str
    question_b: str
    id_a: str
    id_b: str
    platform_a: str
    platform_b: str
    is_same_outcome: bool
    confidence_score: float
    rationale: str
    outcome_a: Optional[str]    # "YES", "NO", or None if unresolved
    outcome_b: Optional[str]
    resolved_at_a: Optional[str]
    resolved_at_b: Optional[str]


# ── Per-group helpers ─────────────────────────────────────────────────────────

async def _label_group(
    backend: LLMBackend,
    questions: list[str],
) -> str:
    numbered = "\n".join(f"{i + 1}. {q}" for i, q in enumerate(questions))
    prompt = _LABEL_USER.format(
        categories=", ".join(_CATEGORIES),
        numbered=numbered,
    )
    try:
        raw = await backend.generate(prompt, system_prompt=_LABEL_SYSTEM, max_new_tokens=32)
        data = extract_json(raw)
        if isinstance(data, dict):
            cat = data.get("category", "other")
            return cat if cat in _CATEGORIES else "other"
    except Exception as exc:
        logger.warning("Group labeling failed: %s", exc)
    return "other"


async def _discover_pairs_in_group(
    backend: LLMBackend,
    group_df: pd.DataFrame,
) -> tuple[str, list[dict]]:
    """Returns (category, pairs_list) in a single LLM call."""
    questions = group_df["question"].tolist()
    if len(questions) < 2:
        return "other", []

    numbered = "\n".join(f"{i + 1}. {q}" for i, q in enumerate(questions))
    prompt = _PAIRS_USER.format(
        categories=", ".join(_CATEGORIES),
        numbered=numbered,
    )
    try:
        raw = await backend.generate(prompt, system_prompt=_PAIRS_SYSTEM, max_new_tokens=2048)
        data = extract_json(raw)
        if isinstance(data, dict):
            category = data.get("category", "other")
            category = category if category in _CATEGORIES else "other"
            return category, data.get("pairs", [])
    except Exception as exc:
        logger.warning("Pair discovery failed: %s", exc)
    return "other", []


# ── Main entry point ──────────────────────────────────────────────────────────

async def discover_relationships(
    df: pd.DataFrame,
    neighbor_pairs: pd.DataFrame,
    backend: LLMBackend,
    min_confidence: float = 0.5,
    concurrency: int = 8,
) -> list[DiscoveredPair]:
    """
    For each market and its KNN neighbors, discover related pairs via LLM.

    Args:
        df: Full market DataFrame with columns: id, platform, question,
            outcome, resolved_at. Row order must match the embeddings used
            to compute neighbor_pairs.
        neighbor_pairs: Output of find_neighbors() — long-form pair DataFrame
            with columns: idx_a, idx_b, id_a, id_b, question_a, question_b, ...
        backend: Any LLMBackend implementation
        min_confidence: Pairs below this threshold are discarded
        concurrency: Max simultaneous LLM calls

    Returns:
        Deduplicated list of DiscoveredPair dataclasses
    """
    semaphore = asyncio.Semaphore(concurrency)
    id_to_row = {row["id"]: row for _, row in df.iterrows()}

    # Reconstruct per-anchor groups: anchor_id → [anchor_row, neighbor_row, ...]
    # neighbor_pairs has idx_a as the anchor; each row is one (anchor, neighbor) pair
    groups: dict[str, list[dict]] = {}
    for _, row in neighbor_pairs.iterrows():
        anchor_id = row["id_a"]
        if anchor_id not in groups:
            anchor_row = id_to_row.get(anchor_id)
            if anchor_row is None:
                continue
            groups[anchor_id] = [dict(anchor_row)]
        neighbor_row = id_to_row.get(row["id_b"])
        if neighbor_row is not None:
            groups[anchor_id].append(dict(neighbor_row))

    logger.info(
        "Discovering relationships across %d neighbor groups (concurrency=%d) ...",
        len(groups), concurrency,
    )

    async def process_group(anchor_id: str, members: list[dict]) -> list[DiscoveredPair]:
        async with semaphore:
            group_df = pd.DataFrame(members)
            category, raw_pairs = await _discover_pairs_in_group(backend, group_df)

            q_to_row = {r["question"]: r for r in members}

            pairs: list[DiscoveredPair] = []
            for p in raw_pairs:
                conf = p.get("confidence_score", 0)
                try:
                    conf = float(conf)
                except (TypeError, ValueError):
                    conf = 0.0
                if conf < min_confidence:
                    continue

                q_a = p.get("question_a", "")
                q_b = p.get("question_b", "")
                row_a = q_to_row.get(q_a)
                row_b = q_to_row.get(q_b)

                if row_a is None or row_b is None:
                    logger.debug(
                        "Anchor %s: skipping pair — LLM may have paraphrased. "
                        "A=%r", anchor_id, q_a[:60]
                    )
                    continue

                pairs.append(DiscoveredPair(
                    anchor_id=anchor_id,
                    group_category=category,
                    question_a=q_a,
                    question_b=q_b,
                    id_a=row_a["id"],
                    id_b=row_b["id"],
                    platform_a=row_a["platform"],
                    platform_b=row_b["platform"],
                    is_same_outcome=bool(p.get("is_same_outcome", False)),
                    confidence_score=conf,
                    rationale=p.get("rationale", ""),
                    outcome_a=row_a.get("outcome"),
                    outcome_b=row_b.get("outcome"),
                    resolved_at_a=row_a.get("resolved_at"),
                    resolved_at_b=row_b.get("resolved_at"),
                ))
            return pairs

    results = await asyncio.gather(*[
        process_group(aid, members) for aid, members in groups.items()
    ])
    all_pairs = [p for batch in results for p in batch]

    # Deduplicate — same pair may have been discovered from both A's and B's group
    seen: set[tuple[str, str]] = set()
    deduped: list[DiscoveredPair] = []
    for p in all_pairs:
        key = tuple(sorted([p.id_a, p.id_b]))
        if key not in seen:
            seen.add(key)
            deduped.append(p)

    logger.info(
        "Discovered %d unique pairs (from %d total, min_confidence=%.2f)",
        len(deduped), len(all_pairs), min_confidence,
    )
    return deduped
