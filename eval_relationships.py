"""
Evaluate relationship-discovery accuracy against resolved markets.

Two evaluation modes:

  --mode kmeans  (default, paper methodology — arXiv:2512.02436)
    1. Load N resolved markets
    2. Embed + K-means cluster into groups of ~10
    3. Run LLM on each cluster
    4. Evaluate all discovered pairs against ground truth

  --mode knn  (detector methodology — mirrors the streaming detector)
    1. Load N resolved markets from each platform
    2. Embed all, build a FAISS index per platform
    3. For each market on platform A, find K nearest neighbors on platform B
       with cosine similarity >= MIN_CROSS_PLATFORM_SIM
    4. Form anchor+neighbors group, run LLM
    5. Evaluate only the cross-platform pairs

Usage:
    python eval_relationships.py [--db PATH] [--n N] [--mode MODE]
                                  [--poly-csv PATH] [--output PATH]
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


def _load_resolved_markets(db_path: str, platforms: tuple[str, ...],
                           poly_csv: str | None = None) -> pd.DataFrame:
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
        if "polymarket" in platforms and poly_csv:
            poly_df = pd.read_csv(poly_csv)
            for _, r in poly_df.iterrows():
                if r.get("question") and r.get("id"):
                    rows.append({
                        "id": r["id"], "platform": "polymarket",
                        "question": str(r["question"]).strip(),
                        "outcome": r.get("outcome", "YES"),
                        "price_yes": r.get("price_yes"),
                    })
    finally:
        conn.close()
    return pd.DataFrame(rows)


async def main_async(args: argparse.Namespace) -> None:
    # ── Load resolved markets ──────────────────────────────────────────────────
    platforms = tuple(args.platform.split(","))

    print(f"[{_ts()}] Loading resolved markets (platforms={platforms}, n={args.n}) ...")
    poly_csv = getattr(args, "poly_csv", None)
    df = _load_resolved_markets(args.db, platforms, poly_csv=poly_csv)

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

    # ── LLM backend ────────────────────────────────────────────────────────────
    eval_model = os.getenv("EVAL_LLM_MODEL", "llama-3.1-8b-instant")
    backend = OpenAICompatibleBackend(
        model=eval_model,
        api_key=config.GROQ_API_KEY,
        base_url="https://api.groq.com/openai/v1",
        rpm_limit=25,
    )
    print(f"[{_ts()}] LLM model: {eval_model}  mode: {args.mode}")

    all_pairs: list[dict] = []

    if args.mode == "kmeans":
        # ── K-means clustering (paper methodology) ────────────────────────────
        k = max(2, len(df) // 10)
        print(f"[{_ts()}] K-means clustering into {k} clusters ...")
        labels = kmeans_cluster(embeddings, k)
        df["cluster"] = labels
        cluster_sizes = pd.Series(labels).value_counts()
        print(f"[{_ts()}] Cluster size — mean={cluster_sizes.mean():.1f}  "
              f"median={cluster_sizes.median():.0f}  max={cluster_sizes.max()}")

        records = df.to_dict("records")
        clusters: dict[int, list[dict]] = defaultdict(list)
        for r in records:
            clusters[int(r["cluster"])].append(r)

        print(f"[{_ts()}] Running LLM on {k} clusters ...")
        for cid in sorted(clusters.keys()):
            pairs = await run_cluster(backend, cid, clusters[cid], verbose=args.verbose)
            all_pairs.extend(pairs)

    else:
        # ── KNN cross-platform (detector methodology) ─────────────────────────
        import faiss
        from detect_arbitrage import MIN_CROSS_PLATFORM_SIM

        records = df.to_dict("records")
        platforms_present = df["platform"].unique().tolist()
        if len(platforms_present) < 2:
            print("ERROR: KNN mode needs at least 2 platforms. Use --platform kalshi,polymarket")
            return

        # Build per-platform FAISS indexes
        plat_indexes: dict[str, tuple] = {}  # platform -> (index, [record])
        for plat in platforms_present:
            mask = df["platform"] == plat
            plat_embs = embeddings[mask.values].astype(np.float32)
            plat_recs = [r for r in records if r["platform"] == plat]
            idx = faiss.IndexFlatIP(plat_embs.shape[1])
            idx.add(plat_embs)
            plat_indexes[plat] = (idx, plat_recs)

        print(f"[{_ts()}] Built {len(plat_indexes)} platform indexes: "
              + ", ".join(f"{p}={len(v[1])}" for p, v in plat_indexes.items()))

        # For each market on platform A, find K nearest on platform B
        k_neighbors = args.k
        seen_groups: set[frozenset] = set()
        groups: list[list[dict]] = []

        for i, rec in enumerate(records):
            anchor_platform = rec["platform"]
            anchor_emb = embeddings[i:i+1].astype(np.float32)

            for other_platform, (other_idx, other_recs) in plat_indexes.items():
                if other_platform == anchor_platform:
                    continue
                k_actual = min(k_neighbors, other_idx.ntotal)
                scores, indices = other_idx.search(anchor_emb, k_actual)
                neighbors = [
                    other_recs[idx]
                    for score, idx in zip(scores[0], indices[0])
                    if 0 <= idx < len(other_recs) and score >= MIN_CROSS_PLATFORM_SIM
                ]
                if not neighbors:
                    continue
                group = [rec] + neighbors
                group_key = frozenset(r["id"] for r in group)
                if group_key in seen_groups:
                    continue
                seen_groups.add(group_key)
                groups.append(group)

        print(f"[{_ts()}] Found {len(groups)} cross-platform KNN groups ...")
        for gid, group in enumerate(groups):
            pairs = await run_cluster(backend, gid, group, verbose=args.verbose)
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
        "n_clusters": k if args.mode == "kmeans" else len(groups) if args.mode == "knn" else 0,
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
    parser.add_argument("--mode", default="kmeans", choices=["kmeans", "knn"],
                        help="kmeans: paper methodology; knn: mirrors the streaming detector")
    parser.add_argument("--k", type=int, default=10,
                        help="Nearest neighbors per market in knn mode (default: 10)")
    parser.add_argument("--poly-csv", default=None,
                        help="CSV from collect_resolved_polymarket.py for Polymarket resolved data")
    parser.add_argument("--output", default="./output/eval_results.json")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-cluster LLM call details")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
