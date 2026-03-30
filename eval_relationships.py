"""
Evaluate relationship-discovery accuracy against resolved markets.

Mirrors the methodology from "Semantic Trading" (arXiv:2512.02436):
  1. Load N resolved binary markets from the DB
  2. Embed questions with all-MiniLM-L6-v2
  3. K-means cluster into groups of ~10 (K = N // 10)
  4. Run LLM relationship discovery on each cluster
  5. Compare predicted is_same_outcome against ground truth
     (ground truth = both markets resolved identically)
  6. Report accuracy at multiple confidence thresholds

Usage:
    python eval_relationships.py [--db PATH] [--n N] [--platform PLATFORM]
                                  [--output PATH]

Results are written to a JSON file and printed as a summary table.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

import config
from forecasting.embedder import embed_questions
from forecasting.llm import OpenAICompatibleBackend
from forecasting.relationships import _discover_pairs_in_group

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

CONFIDENCE_THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9]


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%H:%M:%S")


def kmeans_cluster(embeddings: np.ndarray, k: int, seed: int = 42) -> np.ndarray:
    """Assign each embedding to one of k clusters. Returns integer label array."""
    import faiss
    d = embeddings.shape[1]
    kmeans = faiss.Kmeans(d, k, niter=20, seed=seed, gpu=False)
    kmeans.train(embeddings.astype(np.float32))
    _, labels = kmeans.index.search(embeddings.astype(np.float32), 1)
    return labels.flatten()


def evaluate_pairs(
    pairs: list[dict],
    min_confidence: float,
) -> dict:
    """Compute accuracy metrics for pairs above the confidence threshold."""
    filtered = [p for p in pairs if p["confidence_score"] >= min_confidence and p["evaluable"]]
    if not filtered:
        return {"n": 0, "correct": 0, "accuracy": None}
    correct = sum(1 for p in filtered if p["predicted_correct"])
    return {
        "n": len(filtered),
        "correct": correct,
        "accuracy": correct / len(filtered),
    }


async def run_cluster(
    backend: OpenAICompatibleBackend,
    cluster_id: int,
    members: list[dict],
    verbose: bool,
) -> list[dict]:
    """Run LLM on one cluster, return raw pair dicts augmented with ground-truth."""
    if len(members) < 2:
        return []

    group_df = pd.DataFrame(members)
    t0 = time.time()
    try:
        category, raw_pairs = await _discover_pairs_in_group(backend, group_df)
    except Exception as exc:
        logger.warning("Cluster %d LLM failed: %s", cluster_id, exc)
        return []

    elapsed = time.time() - t0
    if verbose:
        print(f"  cluster {cluster_id:3d}  size={len(members):3d}  "
              f"pairs={len(raw_pairs):2d}  cat={category}  {elapsed:.1f}s")

    q_to_row = {r["question"]: r for r in members}
    results = []
    for p in raw_pairs:
        conf = float(p.get("confidence_score", 0))
        q_a = p.get("question_a", "")
        q_b = p.get("question_b", "")
        row_a = q_to_row.get(q_a)
        row_b = q_to_row.get(q_b)
        if row_a is None or row_b is None:
            continue

        outcome_a = row_a.get("outcome")
        outcome_b = row_b.get("outcome")
        evaluable = outcome_a is not None and outcome_b is not None

        actual_same = (outcome_a == outcome_b) if evaluable else None
        predicted_same = bool(p.get("is_same_outcome", False))
        predicted_correct = (predicted_same == actual_same) if evaluable else None

        results.append({
            "cluster_id": cluster_id,
            "category": category,
            "question_a": q_a,
            "question_b": q_b,
            "id_a": row_a["id"],
            "id_b": row_b["id"],
            "platform_a": row_a["platform"],
            "platform_b": row_b["platform"],
            "outcome_a": outcome_a,
            "outcome_b": outcome_b,
            "is_same_outcome_predicted": predicted_same,
            "is_same_outcome_actual": actual_same,
            "confidence_score": conf,
            "rationale": p.get("rationale", ""),
            "evaluable": evaluable,
            "predicted_correct": predicted_correct,
        })
    return results


def _load_resolved_markets(db_path: str, platforms: tuple[str, ...]) -> pd.DataFrame:
    """Load resolved markets directly via sqlite3 (bypasses SQLAlchemy WAL conflict)."""
    import sqlite3
    rows = []
    conn = sqlite3.connect(f"file:{db_path}?immutable=1", uri=True)
    try:
        cur = conn.cursor()
        if "kalshi" in platforms:
            # Rowid-based random sampling: O(1) per row vs full table scan.
            cur.execute("SELECT MAX(rowid) FROM kalshi_markets")
            max_rowid = cur.fetchone()[0] or 1
            rng = np.random.default_rng(42)
            # Oversample generously since most rows won't match our filter.
            # ~30% of rows are resolved (2.8M / 3.9M), ~0% are KXMV among resolved.
            sample_size = min(max_rowid, 30_000)
            sample_rowids = rng.choice(max_rowid, size=sample_size, replace=False) + 1
            placeholders = ",".join("?" * len(sample_rowids))
            cur.execute(f"""
                SELECT ticker, title, subtitle, result, yes_ask
                FROM kalshi_markets
                WHERE rowid IN ({placeholders})
                  AND result IN ('yes', 'no')
                  AND ticker NOT LIKE 'KXMV%'
                  AND title IS NOT NULL AND title != ''
            """, sample_rowids.tolist())
            for ticker, title, subtitle, result, yes_ask in cur.fetchall():
                question = title.strip()
                if subtitle:
                    question = f"{question} — {subtitle.strip()}"
                outcome = "YES" if result == "yes" else "NO"
                price = (yes_ask / 100.0) if yes_ask is not None else None
                rows.append({"id": ticker, "platform": "kalshi",
                             "question": question, "outcome": outcome,
                             "price_yes": price})
        if "polymarket" in platforms:
            cur.execute("""
                SELECT condition_id, question, price_yes
                FROM polymarket_markets
                WHERE closed = 1
                  AND price_yes IS NOT NULL
                  AND (price_yes >= 0.99 OR price_yes <= 0.01)
                  AND question IS NOT NULL AND question != ''
            """)
            for cid, question, price_yes in cur.fetchall():
                outcome = "YES" if price_yes >= 0.99 else "NO"
                rows.append({"id": cid, "platform": "polymarket",
                             "question": question.strip(), "outcome": outcome,
                             "price_yes": price_yes})
    finally:
        conn.close()
    return pd.DataFrame(rows)


async def main_async(args: argparse.Namespace) -> None:
    # ── Load resolved markets ──────────────────────────────────────────────────
    platforms = tuple(args.platform.split(","))

    print(f"[{_ts()}] Loading resolved markets (platforms={platforms}, n={args.n}) ...")
    df = _load_resolved_markets(args.db, platforms)

    if df.empty:
        print("ERROR: No resolved markets found in the DB.")
        return

    # Balance YES/NO outcomes for a fair evaluation
    yes_df = df[df["outcome"] == "YES"]
    no_df  = df[df["outcome"] == "NO"]
    n_each = min(args.n // 2, len(yes_df), len(no_df))
    df = pd.concat([
        yes_df.sample(n=n_each, random_state=42),
        no_df.sample(n=n_each, random_state=42),
    ]).reset_index(drop=True)

    print(f"[{_ts()}] Sampled {len(df)} markets  "
          f"(YES={int((df['outcome']=='YES').sum())}, NO={int((df['outcome']=='NO').sum())})")

    # ── Embed ──────────────────────────────────────────────────────────────────
    from sentence_transformers import SentenceTransformer
    device = os.getenv("EMBEDDER_DEVICE", "cpu")
    print(f"[{_ts()}] Embedding with all-MiniLM-L6-v2 (device={device}) ...")
    model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    embeddings = embed_questions(df, model=model)

    # ── Cluster ────────────────────────────────────────────────────────────────
    k = max(2, len(df) // 10)
    print(f"[{_ts()}] K-means clustering into {k} clusters ...")
    labels = kmeans_cluster(embeddings, k)
    df["cluster"] = labels

    cluster_sizes = pd.Series(labels).value_counts()
    print(f"[{_ts()}] Cluster size — mean={cluster_sizes.mean():.1f}  "
          f"median={cluster_sizes.median():.0f}  max={cluster_sizes.max()}")

    # ── LLM discovery ──────────────────────────────────────────────────────────
    # Use a smaller/faster model with higher TPM limits for bulk eval.
    # llama-3.1-8b-instant: 30k TPM vs llama-3.3-70b-versatile: 6k TPM on Groq free tier.
    eval_model = os.getenv("EVAL_LLM_MODEL", "llama-3.1-8b-instant")
    backend = OpenAICompatibleBackend(
        model=eval_model,
        api_key=config.GROQ_API_KEY,
        base_url="https://api.groq.com/openai/v1",
        rpm_limit=25,
    )
    print(f"[{_ts()}] LLM model: {eval_model}")

    records = df.to_dict("records")
    clusters: dict[int, list[dict]] = defaultdict(list)
    for r in records:
        clusters[int(r["cluster"])].append(r)

    print(f"[{_ts()}] Running LLM on {k} clusters (rpm_limit=5) ...")
    all_pairs: list[dict] = []
    for cid in sorted(clusters.keys()):
        members = clusters[cid]
        pairs = await run_cluster(backend, cid, members, verbose=args.verbose)
        all_pairs.extend(pairs)

    # ── Evaluate ───────────────────────────────────────────────────────────────
    evaluable = [p for p in all_pairs if p["evaluable"]]
    print(f"\n[{_ts()}] Total pairs discovered: {len(all_pairs)}  evaluable: {len(evaluable)}")

    if not evaluable:
        print("No evaluable pairs (need both markets resolved). "
              "Check that resolved_only markets are present.")
        return

    # Per-confidence-threshold accuracy
    print("\nAccuracy by confidence threshold:")
    print(f"  {'threshold':>10}  {'n_pairs':>8}  {'correct':>8}  {'accuracy':>9}")
    print("  " + "-" * 40)
    threshold_results = {}
    for thresh in CONFIDENCE_THRESHOLDS:
        result = evaluate_pairs(all_pairs, thresh)
        acc_str = f"{result['accuracy']:.1%}" if result["accuracy"] is not None else "  n/a"
        print(f"  {thresh:>10.1f}  {result['n']:>8}  {result['correct']:>8}  {acc_str:>9}")
        threshold_results[str(thresh)] = result

    # Per-category accuracy (at conf >= 0.5)
    category_groups: dict[str, list[dict]] = defaultdict(list)
    for p in evaluable:
        category_groups[p["category"]].append(p)

    print("\nAccuracy by category (conf >= 0.5):")
    print(f"  {'category':>15}  {'n_pairs':>8}  {'accuracy':>9}")
    print("  " + "-" * 37)
    for cat, pairs in sorted(category_groups.items(), key=lambda x: -len(x[1])):
        r = evaluate_pairs(pairs, 0.5)
        acc_str = f"{r['accuracy']:.1%}" if r["accuracy"] is not None else "  n/a"
        print(f"  {cat:>15}  {r['n']:>8}  {acc_str:>9}")

    # Cross-platform breakdown
    cross = [p for p in evaluable if p["platform_a"] != p["platform_b"]]
    same_plat = [p for p in evaluable if p["platform_a"] == p["platform_b"]]
    print(f"\nCross-platform pairs (conf>=0.5): {len([p for p in cross if p['confidence_score']>=0.5])}")
    print(f"Same-platform pairs (conf>=0.5):  {len([p for p in same_plat if p['confidence_score']>=0.5])}")

    # ── Save results ───────────────────────────────────────────────────────────
    output = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "args": vars(args),
        "n_markets": len(df),
        "n_clusters": k,
        "n_pairs_total": len(all_pairs),
        "n_pairs_evaluable": len(evaluable),
        "threshold_results": threshold_results,
        "pairs": all_pairs,
    }
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n[{_ts()}] Results saved to {args.output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate relationship-discovery accuracy")
    parser.add_argument("--db", default="./data/markets.db")
    parser.add_argument("--n", type=int, default=500,
                        help="Number of resolved markets to evaluate on (balanced YES/NO)")
    parser.add_argument("--platform", default="kalshi",
                        help="Comma-separated platforms: kalshi, polymarket, or kalshi,polymarket")
    parser.add_argument("--output", default="./output/eval_results.json")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-cluster LLM call details")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
