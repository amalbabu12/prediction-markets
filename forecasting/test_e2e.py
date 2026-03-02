"""
End-to-end pipeline test using mock data.

No database, no real LLM, no HuggingFace credentials needed.

Run:
    python -m forecasting.test_e2e
"""
from __future__ import annotations

import asyncio
import json
import re
import sys
from dataclasses import asdict

import numpy as np
import pandas as pd

from forecasting.llm import LLMBackend, extract_json
from forecasting.neighbors import find_neighbors
from forecasting.relationships import discover_relationships
from forecasting.evaluator import evaluate


# ── Mock LLM backend ──────────────────────────────────────────────────────────

class MockLLMBackend(LLMBackend):
    """
    Returns deterministic JSON responses without calling any model.

    Label calls  → assigns category based on keywords in the questions.
    Pair calls   → pairs question 1 with question 2 in each group,
                   predicting is_same_outcome=True with confidence 0.85.
    """

    async def generate(
        self,
        user_prompt: str,
        system_prompt: str = "",
        max_new_tokens: int = 2048,
    ) -> str:
        # Detect call type from prompt content
        if '"category"' in user_prompt:
            return self._label(user_prompt)
        return self._pairs(user_prompt)

    def _label(self, prompt: str) -> str:
        p = prompt.lower()
        if any(w in p for w in ("btc", "bitcoin", "crypto", "eth")):
            return '{"category": "crypto"}'
        if any(w in p for w in ("election", "president", "vote", "senate")):
            return '{"category": "elections"}'
        if any(w in p for w in ("fed", "rate", "inflation", "gdp", "economy")):
            return '{"category": "economy"}'
        if any(w in p for w in ("war", "sanction", "nato", "military")):
            return '{"category": "geopolitics"}'
        return '{"category": "other"}'

    def _pairs(self, prompt: str) -> str:
        # Parse numbered questions from the prompt
        questions = re.findall(r"^\d+\.\s+(.+)$", prompt, re.MULTILINE)
        if len(questions) < 2:
            return '{"pairs": []}'
        # Pair consecutive questions
        pairs = []
        for i in range(0, len(questions) - 1, 2):
            pairs.append({
                "question_a": questions[i],
                "question_b": questions[i + 1],
                "is_same_outcome": True,
                "confidence_score": 0.85,
                "rationale": "mock: consecutive questions likely related",
            })
        return json.dumps({"pairs": pairs})


# ── Mock market data ──────────────────────────────────────────────────────────

def make_mock_markets() -> pd.DataFrame:
    """
    30 synthetic markets across 3 topics (crypto, elections, economy).
    Each group of 10 has known outcomes so evaluation is meaningful.
    """
    markets = [
        # Crypto — 10 markets
        ("k1",  "kalshi",     "Will BTC close above $50,000 by end of March?",        "YES"),
        ("k2",  "kalshi",     "Will BTC close above $55,000 by end of March?",        "NO"),
        ("p1",  "polymarket", "Bitcoin above $50k on March 31?",                      "YES"),
        ("p2",  "polymarket", "Bitcoin above $60k on March 31?",                      "NO"),
        ("k3",  "kalshi",     "Will ETH close above $3,000 by end of March?",         "YES"),
        ("p3",  "polymarket", "Ethereum price above $3000 on March 31?",              "YES"),
        ("k4",  "kalshi",     "Will crypto market cap exceed $2T in Q1?",             "YES"),
        ("p4",  "polymarket", "Total crypto market cap over $2 trillion in Q1?",      "YES"),
        ("k5",  "kalshi",     "Will BTC ETF inflows exceed $1B in March?",            "NO"),
        ("p5",  "polymarket", "Bitcoin ETF sees over $1B net inflows in March?",      "NO"),

        # Elections — 10 markets
        ("k6",  "kalshi",     "Will Democrats win the 2026 Senate majority?",         "NO"),
        ("p6",  "polymarket", "Democratic Party controls Senate after 2026 midterms?","NO"),
        ("k7",  "kalshi",     "Will voter turnout exceed 50% in 2026 midterms?",      "YES"),
        ("p7",  "polymarket", "2026 midterm turnout above 50%?",                      "YES"),
        ("k8",  "kalshi",     "Will Republicans gain seats in the House in 2026?",    "YES"),
        ("p8",  "polymarket", "GOP gains House seats in 2026 midterms?",              "YES"),
        ("k9",  "kalshi",     "Will there be a third-party candidate in 2028?",       "YES"),
        ("p9",  "polymarket", "Independent candidate runs for president in 2028?",    "YES"),
        ("k10", "kalshi",     "Will the 2026 midterms see record spending?",          "NO"),
        ("p10", "polymarket", "Record campaign spending in 2026 midterms?",           "NO"),

        # Economy — 10 markets
        ("k11", "kalshi",     "Will the Fed cut rates in Q1 2026?",                   "NO"),
        ("p11", "polymarket", "Federal Reserve cuts interest rates in Q1 2026?",      "NO"),
        ("k12", "kalshi",     "Will US CPI exceed 3% in February 2026?",              "YES"),
        ("p12", "polymarket", "US inflation above 3% in February 2026?",              "YES"),
        ("k13", "kalshi",     "Will US GDP growth exceed 2% in 2026?",                "YES"),
        ("p13", "polymarket", "US GDP grows more than 2% in 2026?",                   "YES"),
        ("k14", "kalshi",     "Will the unemployment rate fall below 4% in 2026?",    "NO"),
        ("p14", "polymarket", "US unemployment rate drops below 4% in 2026?",         "NO"),
        ("k15", "kalshi",     "Will the US enter recession in 2026?",                 "NO"),
        ("p15", "polymarket", "United States recession in 2026?",                     "NO"),
    ]

    df = pd.DataFrame(markets, columns=["id", "platform", "question", "outcome"])
    df["resolved_at"] = "2026-03-31T00:00:00Z"
    return df


def make_mock_embeddings(df: pd.DataFrame) -> np.ndarray:
    """
    Synthetic embeddings where markets in the same topic cluster together.
    Each topic group shares a base vector; individual markets add small noise.
    """
    rng = np.random.default_rng(42)
    d = 64  # embedding dimension

    # One base vector per topic group (every 10 rows)
    n = len(df)
    embeddings = np.zeros((n, d), dtype=np.float32)
    for i in range(n):
        group = i // 10       # 0=crypto, 1=elections, 2=economy
        base = rng.standard_normal(d) * 0.01  # shared group direction
        base[group * (d // 3): (group + 1) * (d // 3)] += 1.0  # group signal
        noise = rng.standard_normal(d) * 0.1
        embeddings[i] = base + noise

    # L2-normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / norms


# ── Test runner ───────────────────────────────────────────────────────────────

async def run_test() -> None:
    print("=" * 60)
    print("E2E Pipeline Test — Mock Data")
    print("=" * 60)

    # 1. Mock markets
    df = make_mock_markets()
    print(f"\n[1] Markets: {len(df)} total, {df['outcome'].notna().sum()} resolved")
    print(df.groupby(["platform", "outcome"]).size().to_string())

    # 2. Mock embeddings
    embeddings = make_mock_embeddings(df)
    print(f"\n[2] Embeddings: shape={embeddings.shape}")

    # 3. KNN neighbor search
    neighbor_pairs = find_neighbors(df, embeddings, k=4)
    print(f"\n[3] Neighbor pairs: {len(neighbor_pairs)} unique pairs")
    print(f"    Sample:")
    for _, row in neighbor_pairs.head(3).iterrows():
        print(f"      {row['id_a']} ↔ {row['id_b']}  rank={row['similarity_rank']}")
        print(f"        A: {row['question_a'][:60]}")
        print(f"        B: {row['question_b'][:60]}")

    # 4. Relationship discovery with mock LLM
    backend = MockLLMBackend()
    pairs = await discover_relationships(
        df,
        neighbor_pairs=neighbor_pairs,
        backend=backend,
        min_confidence=0.5,
        concurrency=4,
    )
    print(f"\n[4] Discovered pairs: {len(pairs)}")
    print(f"    Sample:")
    for p in pairs[:3]:
        print(f"      {p.id_a} ({p.platform_a}) ↔ {p.id_b} ({p.platform_b})")
        print(f"      is_same_outcome={p.is_same_outcome}  conf={p.confidence_score}")
        print(f"      actual: outcome_a={p.outcome_a}  outcome_b={p.outcome_b}")
        print(f"      category={p.group_category}")

    # 5. Evaluate
    report = evaluate(pairs)
    print(f"\n[5] Accuracy Report:")
    print(f"    Total pairs:      {report.total_pairs}")
    print(f"    Evaluable pairs:  {report.evaluable_pairs}")
    print(f"    Overall accuracy: {report.overall_accuracy:.1%}")
    print(f"    Anchor accuracy:  {report.cluster_accuracy:.1%}")
    print(f"    Confusion: {report.confusion}")
    if report.by_category:
        print(f"    By category:")
        for cat, acc in sorted(report.by_category.items(), key=lambda x: -x[1]):
            print(f"      {cat:<15} {acc:.1%}")
    if report.by_platform_pair:
        print(f"    By platform pair:")
        for pair, acc in sorted(report.by_platform_pair.items(), key=lambda x: -x[1]):
            print(f"      {pair:<30} {acc:.1%}")

    print("\n[OK] End-to-end test complete.")


if __name__ == "__main__":
    asyncio.run(run_test())
