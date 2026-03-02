"""
Semantic market relationship pipeline.

Implements the four-stage workflow from:
  "Semantic Trading: Agentic AI for Clustering and Relationship Discovery
   in Prediction Markets" (Capponi, Gliozzo, Zhu — arXiv:2512.02436)

Stages:
  1. Load    — pull market questions + outcomes from the DB
  2. Embed   — sentence-transformer embeddings (cached to disk)
  3. Cluster — K-means, K ≈ N/10
  4. Discover — LLM labels clusters + finds related pairs
  5. Evaluate — accuracy vs ground-truth resolved outcomes

Outputs (saved to data/forecasting/):
  cluster_manifest.csv    — every market with its cluster_id
  discovered_pairs.csv    — all LLM-predicted pairs with metadata
  accuracy_report.json    — accuracy metrics

Usage:
    python -m forecasting.pipeline \\
        --model "mistralai/Mistral-7B-Instruct-v0.3" \\
        --hf-api-key "hf_..." \\
        --resolved-only

    # Against a local TGI / vLLM server:
    python -m forecasting.pipeline \\
        --model "meta-llama/Meta-Llama-3-8B-Instruct" \\
        --base-url "http://localhost:8080" \\
        --resolved-only --max-markets 5000
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
from dataclasses import asdict
from pathlib import Path

import pandas as pd

import config
from db.models import init_db
from forecasting.embedder import embed_questions
from forecasting.evaluator import evaluate
from forecasting.llm import HuggingFaceBackend, LLMBackend
from forecasting.loader import load_markets
from forecasting.neighbors import find_neighbors
from forecasting.relationships import discover_relationships

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("forecasting.pipeline")

OUT_DIR = Path("data/forecasting")


async def run(
    backend: LLMBackend,
    args,
) -> None:
    db_path = args.db
    platforms = tuple(args.platforms)
    resolved_only = args.resolved_only
    max_markets = args.max_markets
    min_confidence = args.min_confidence
    concurrency = args.concurrency
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    Session = init_db(db_path)

    # ── 1. Load ───────────────────────────────────────────────────────────────
    df = load_markets(Session, platforms=platforms, resolved_only=resolved_only)
    if df.empty:
        logger.error(
            "No markets found. Run 'python main.py outcomes' first."
        )
        return

    if max_markets and len(df) > max_markets:
        # Prefer resolved markets when subsampling so evaluation has more signal
        resolved = df[df["outcome"].notna()]
        unresolved = df[df["outcome"].isna()]
        n_res = min(len(resolved), max_markets)
        n_unres = min(len(unresolved), max(0, max_markets - n_res))
        parts = [resolved.sample(n=n_res, random_state=42)]
        if n_unres > 0:
            parts.append(unresolved.sample(n=n_unres, random_state=42))
        df = pd.concat(parts).reset_index(drop=True)
        logger.info(
            "Subsampled to %d markets (%d resolved).",
            len(df), int(df["outcome"].notna().sum()),
        )

    # ── 2. Embed ──────────────────────────────────────────────────────────────
    embeddings = embed_questions(df)

    # ── 3. KNN neighbor search ────────────────────────────────────────────────
    neighbor_pairs = find_neighbors(
        df, embeddings,
        k=args.k,
        use_faiss=args.faiss,
    )

    neighbor_pairs.to_csv(OUT_DIR / "neighbor_pairs.csv", index=False)
    logger.info("Neighbor pairs → %s  (%d rows)", OUT_DIR / "neighbor_pairs.csv", len(neighbor_pairs))

    # ── 4. Discover relationships ─────────────────────────────────────────────
    pairs = await discover_relationships(
        df,
        neighbor_pairs=neighbor_pairs,
        backend=backend,
        min_confidence=min_confidence,
        concurrency=concurrency,
    )

    if not pairs:
        logger.warning("No pairs discovered.")
        return

    pairs_df = pd.DataFrame([asdict(p) for p in pairs])
    pairs_path = OUT_DIR / "discovered_pairs.csv"
    pairs_df.to_csv(pairs_path, index=False)
    logger.info("Discovered pairs → %s  (%d rows)", pairs_path, len(pairs_df))

    # ── 5. Evaluate ───────────────────────────────────────────────────────────
    report = evaluate(pairs)

    report_path = OUT_DIR / "accuracy_report.json"
    with open(report_path, "w") as f:
        json.dump(asdict(report), f, indent=2)
    logger.info("Accuracy report → %s", report_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Semantic market relationship pipeline (arXiv:2512.02436).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Model config ──────────────────────────────────────────────────────────
    model_group = parser.add_argument_group("Model")
    model_group.add_argument(
        "--model", required=True,
        help="HuggingFace model ID (e.g. 'mistralai/Mistral-7B-Instruct-v0.3') "
             "or any model name recognised by your --base-url server.",
    )
    model_group.add_argument(
        "--hf-api-key", default=None,
        help="HuggingFace API token (hf_...). Not needed for local servers.",
    )
    model_group.add_argument(
        "--base-url", default=None,
        help="Override endpoint URL (e.g. 'http://localhost:8080' for a local "
             "TGI or vLLM server). Leave unset to use the HF Inference API.",
    )
    model_group.add_argument(
        "--temperature", type=float, default=0.0,
        help="Sampling temperature passed to the model.",
    )

    # ── Data config ───────────────────────────────────────────────────────────
    data_group = parser.add_argument_group("Data")
    data_group.add_argument("--db", default=config.DB_PATH)
    data_group.add_argument(
        "--platforms", nargs="+", default=["kalshi", "polymarket"],
        choices=["kalshi", "polymarket"],
    )
    data_group.add_argument(
        "--resolved-only", action="store_true",
        help="Only include markets with a confirmed outcome.",
    )
    data_group.add_argument(
        "--max-markets", type=int, default=None,
        help="Cap total markets (useful for a quick test run).",
    )

    # ── Pipeline config ───────────────────────────────────────────────────────
    pipe_group = parser.add_argument_group("Pipeline")
    pipe_group.add_argument(
        "--k", type=int, default=10,
        help="Nearest neighbors per market.",
    )
    pipe_group.add_argument(
        "--faiss", action="store_true",
        help="Use persisted faiss index instead of batched numpy search.",
    )
    pipe_group.add_argument(
        "--min-confidence", type=float, default=0.5,
        help="Minimum LLM confidence score to keep a pair.",
    )
    pipe_group.add_argument(
        "--concurrency", type=int, default=8,
        help="Max parallel LLM calls.",
    )

    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    backend = HuggingFaceBackend(
        model=args.model,
        api_key=args.hf_api_key,
        base_url=args.base_url,
        temperature=args.temperature,
    )

    asyncio.run(run(backend=backend, args=args))


if __name__ == "__main__":
    main()
