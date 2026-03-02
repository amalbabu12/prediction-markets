"""
Run the arb-detection pipeline on the live Polymarket DB.

Uses MockLLMBackend so no HuggingFace credentials are needed.
Opens the DB read-only (SQLite WAL) so ongoing writes are not disturbed.

Run:
    python -m forecasting.test_polymarket [--max-markets N] [--k K]
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from db.models import init_db
from forecasting.loader import load_markets
from forecasting.embedder import embed_questions
from forecasting.neighbors import find_neighbors
from forecasting.relationships import discover_relationships
from forecasting.cross_encoder import CrossEncoderScorer, discover_with_crossencoder
from forecasting.evaluator import evaluate
from forecasting.test_e2e import MockLLMBackend

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

PM_DB = "data/polymarket.db"


async def run(
    max_markets: int, k: int, min_confidence: float, concurrency: int,
    use_cross_encoder: bool, model: str | None, api_key: str | None, base_url: str | None,
    csv_path: str | None = None, rpm: int | None = None,
) -> None:
    if use_cross_encoder:
        mode = "Cross-Encoder"
    elif model:
        mode = f"LLM ({model})"
    else:
        mode = "Mock LLM"
    print("=" * 60)
    print(f"Polymarket Pipeline Test — Real Data, {mode}")
    print("=" * 60)

    # 1. Load markets — from CSV snapshot if provided, else live DB
    if csv_path:
        import pandas as _pd
        df = _pd.read_csv(csv_path).dropna(subset=["question","outcome"])
        if max_markets and len(df) > max_markets:
            df = df.sample(n=max_markets, random_state=42).reset_index(drop=True)
        logger.info("Loaded %d markets from %s", len(df), csv_path)
    else:
        pm_sf = init_db(PM_DB)
        df = load_markets(
            pm_sf,
            platforms=("polymarket",),
            resolved_only=True,
            max_markets=max_markets,
        )
    if df.empty:
        print("No resolved markets found. Is collection still running?")
        return

    print(f"\n[1] Loaded {len(df)} resolved Polymarket markets")
    print(f"    YES: {(df['outcome']=='YES').sum()}  NO: {(df['outcome']=='NO').sum()}")
    print(f"    Sample questions:")
    for _, row in df.head(3).iterrows():
        print(f"      [{row['outcome']}] {row['question'][:80]}")

    # 2. Embed
    print(f"\n[2] Embedding {len(df)} questions...")
    embeddings = embed_questions(df)
    print(f"    Shape: {embeddings.shape}")

    # 3. KNN
    print(f"\n[3] Finding {k} nearest neighbors...")
    neighbor_pairs = find_neighbors(df, embeddings, k=k)
    print(f"    Found {len(neighbor_pairs)} candidate pairs")
    print(f"    Sample:")
    for _, row in neighbor_pairs.head(3).iterrows():
        print(f"      sim_rank={row['similarity_rank']}")
        print(f"        A: {row['question_a'][:70]}")
        print(f"        B: {row['question_b'][:70]}")

    # 4. Relationship discovery
    if use_cross_encoder:
        print(f"\n[4] Scoring pairs with cross-encoder (local, no API)...")
        scorer = CrossEncoderScorer(threshold=min_confidence)
        pairs = discover_with_crossencoder(df, neighbor_pairs, scorer, min_confidence=0.0)
        print(f"    Scored {len(pairs)} pairs  (threshold={min_confidence})")
    else:
        if model:
            from forecasting.llm import HuggingFaceBackend, OpenAICompatibleBackend
            # Auto-detect sensible RPM defaults; override with --rpm
            if rpm is None:
                if base_url and "groq" in base_url:
                    rpm = 20
                elif base_url and "googleapis" in base_url:
                    rpm = 12  # Gemini free: 15 RPM, stay conservative
                else:
                    rpm = 0
            # Use OpenAICompatibleBackend for any explicit base_url
            # (Gemini, Groq, Together, etc.) — correct URL construction + 429 handling
            BackendClass = OpenAICompatibleBackend if base_url else HuggingFaceBackend
            backend = BackendClass(
                model=model,
                api_key=api_key,
                base_url=base_url,
                temperature=0.0,
                rpm_limit=rpm,
            )
        else:
            backend = MockLLMBackend()
        print(f"\n[4] Discovering relationships ({mode})...")
        pairs = await discover_relationships(
            df,
            neighbor_pairs=neighbor_pairs,
            backend=backend,
            min_confidence=min_confidence,
            concurrency=concurrency,
        )
        print(f"    Discovered {len(pairs)} pairs above confidence {min_confidence}")
    if pairs:
        print(f"    Sample:")
        for p in pairs[:3]:
            print(f"      [{p.outcome_a}] {p.question_a[:60]}")
            print(f"      [{p.outcome_b}] {p.question_b[:60]}")
            print(f"      is_same_outcome={p.is_same_outcome}  conf={p.confidence_score:.2f}  cat={p.group_category}")

    # 5. Evaluate
    report = evaluate(pairs)
    print(f"\n[5] Accuracy Report:")
    print(f"    Total pairs:      {report.total_pairs}")
    print(f"    Evaluable pairs:  {report.evaluable_pairs}")
    if report.evaluable_pairs > 0:
        print(f"    Overall accuracy: {report.overall_accuracy:.1%}")
        print(f"    Anchor accuracy:  {report.cluster_accuracy:.1%}")
        print(f"    Confusion:        TP={report.confusion['TP']}  FP={report.confusion['FP']}  TN={report.confusion['TN']}  FN={report.confusion['FN']}")
        if report.by_category:
            print(f"    By category:")
            for cat, acc in sorted(report.by_category.items(), key=lambda x: -x[1]):
                print(f"      {cat:<15} {acc:.1%}")

    print("\n[OK] Done.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-markets", type=int, default=2000,
                        help="Max resolved markets to load (default: 2000)")
    parser.add_argument("--k", type=int, default=10,
                        help="KNN neighbors per market (default: 10)")
    parser.add_argument("--min-confidence", type=float, default=0.5)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--csv", type=str, default=None,
                        help="Load markets from a CSV snapshot instead of the live DB")
    parser.add_argument("--cross-encoder", action="store_true",
                        help="Use cross-encoder scorer instead of LLM")
    parser.add_argument("--model", type=str, default=None,
                        help="HF/Groq model ID")
    parser.add_argument("--api-key", type=str, default=None,
                        help="API key")
    parser.add_argument("--base-url", type=str, default=None,
                        help="OpenAI-compatible base URL (e.g. https://api.groq.com/openai/v1)")
    parser.add_argument("--rpm", type=int, default=None,
                        help="Override requests-per-minute rate limit (0 = unlimited)")
    args = parser.parse_args()

    asyncio.run(run(
        args.max_markets, args.k, args.min_confidence, args.concurrency,
        args.cross_encoder, args.model, args.api_key, args.base_url,
        csv_path=args.csv, rpm=args.rpm,
    ))


if __name__ == "__main__":
    main()
