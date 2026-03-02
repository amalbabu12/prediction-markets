"""
K-nearest neighbor search for market relationship discovery.

Replaces K-means clustering with per-market neighbor lookup so every market
finds its K most semantically similar markets regardless of cluster boundaries.

Two backends:
  - Batched numpy (default) — exact cosine similarity, memory-bounded
  - Faiss (optional)        — approximate search, fast at any scale,
                              index persisted to disk for incremental updates

For batch backtesting, batched numpy is fine up to ~100k markets.
For streaming (new markets arriving continuously), load the persisted faiss
index and call add() + search() incrementally.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_FAISS_INDEX_PATH = Path("data/faiss_index.bin")
_FAISS_IDS_PATH = Path("data/faiss_ids.npy")


# ── Batched exact search ──────────────────────────────────────────────────────

def _batched_knn(
    embeddings: np.ndarray,
    k: int,
    batch_size: int = 512,
) -> np.ndarray:
    """
    Exact cosine KNN via batched dot products on L2-normalized vectors.

    Processes `batch_size` query rows at a time so the full N×N matrix is
    never materialized in memory.

    Args:
        embeddings: L2-normalized array of shape (N, D)
        k: Number of neighbors per market (excluding self)
        batch_size: Rows to process per batch

    Returns:
        Integer array of shape (N, k) — neighbor indices into embeddings
    """
    n = len(embeddings)
    k = min(k, n - 1)
    neighbor_indices = np.empty((n, k), dtype=np.int32)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        # (batch, D) @ (D, N) → (batch, N) similarity scores
        sims = embeddings[start:end] @ embeddings.T
        # Zero out self-similarity
        for i in range(end - start):
            sims[i, start + i] = -1.0
        # Top-k indices (unsorted within top-k is fine — we sort below)
        top_k = np.argpartition(sims, -k, axis=1)[:, -k:]
        # Sort each row by descending similarity
        for i in range(end - start):
            row = top_k[i]
            order = np.argsort(sims[i, row])[::-1]
            neighbor_indices[start + i] = row[order]

        if (start // batch_size) % 10 == 0:
            logger.debug("KNN batch %d/%d", start // batch_size, n // batch_size)

    return neighbor_indices


# ── Faiss search ──────────────────────────────────────────────────────────────

def _faiss_knn(
    embeddings: np.ndarray,
    k: int,
    index_path: Path = _FAISS_INDEX_PATH,
    ids_path: Path = _FAISS_IDS_PATH,
    force_rebuild: bool = False,
) -> np.ndarray:
    """
    Approximate KNN using a persisted faiss IndexFlatIP (exact inner product).

    The index is built once and saved to disk. On subsequent runs it is loaded
    directly, making this suitable for incremental / streaming use.

    Args:
        embeddings: L2-normalized array of shape (N, D)
        k: Neighbors per market (excluding self)
        index_path: Where to persist the faiss index
        ids_path: Parallel array mapping faiss positions → original row indices
        force_rebuild: Ignore cached index and rebuild from scratch

    Returns:
        Integer array of shape (N, k) — neighbor indices into embeddings
    """
    import faiss  # optional dependency

    n, d = embeddings.shape
    k = min(k, n - 1)
    vecs = embeddings.astype(np.float32)

    if not force_rebuild and index_path.exists() and ids_path.exists():
        logger.info("Loading faiss index from %s", index_path)
        index = faiss.read_index(str(index_path))
    else:
        logger.info("Building faiss IndexFlatIP for %d vectors (d=%d) ...", n, d)
        index = faiss.IndexFlatIP(d)
        index.add(vecs)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(index_path))
        np.save(ids_path, np.arange(n, dtype=np.int64))
        logger.info("Faiss index saved to %s", index_path)

    # k+1 because the top result for each vector is itself
    _, indices = index.search(vecs, k + 1)

    # Remove self from results (always rank 0 for exact inner product)
    neighbor_indices = np.empty((n, k), dtype=np.int32)
    for i in range(n):
        row = indices[i]
        row = row[row != i][:k]
        neighbor_indices[i] = row

    return neighbor_indices


# ── Public API ────────────────────────────────────────────────────────────────

def find_neighbors(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    k: int = 10,
    use_faiss: bool = False,
    batch_size: int = 512,
    faiss_index_path: Path = _FAISS_INDEX_PATH,
    force_rebuild: bool = False,
) -> pd.DataFrame:
    """
    For each market, find its K nearest semantic neighbors.

    Returns a long-form DataFrame of (market, neighbor) pairs — deduplicated
    so each pair appears only once regardless of which side initiated the match.

    Args:
        df: Market DataFrame with at least 'id', 'platform', 'question',
            'outcome', 'resolved_at' columns. Row order must match embeddings.
        embeddings: L2-normalized embeddings, shape (N, D)
        k: Neighbors per market
        use_faiss: Use persisted faiss index instead of batched numpy
        batch_size: Batch size for numpy backend (ignored if use_faiss=True)
        faiss_index_path: Path for the faiss index file
        force_rebuild: Force faiss index rebuild even if cached

    Returns:
        DataFrame with columns: id_a, id_b, platform_a, platform_b,
        question_a, question_b, outcome_a, outcome_b, resolved_at_a,
        resolved_at_b, similarity_rank (1 = closest neighbor of market_a)
    """
    n = len(df)
    logger.info(
        "Finding %d nearest neighbors for %d markets (backend=%s) ...",
        k, n, "faiss" if use_faiss else "numpy",
    )

    if use_faiss:
        neighbor_indices = _faiss_knn(
            embeddings, k,
            index_path=faiss_index_path,
            force_rebuild=force_rebuild,
        )
    else:
        neighbor_indices = _batched_knn(embeddings, k, batch_size=batch_size)

    # Build long-form pair DataFrame
    rows = []
    df_records = df.to_dict("records")

    for i, neighbors in enumerate(neighbor_indices):
        a = df_records[i]
        for rank, j in enumerate(neighbors, start=1):
            b = df_records[j]
            # Canonical ordering: smaller index first to deduplicate (i,j) vs (j,i)
            if i < j:
                rows.append({
                    "idx_a": i,
                    "idx_b": int(j),
                    "id_a": a["id"],
                    "id_b": b["id"],
                    "platform_a": a["platform"],
                    "platform_b": b["platform"],
                    "question_a": a["question"],
                    "question_b": b["question"],
                    "outcome_a": a.get("outcome"),
                    "outcome_b": b.get("outcome"),
                    "resolved_at_a": a.get("resolved_at"),
                    "resolved_at_b": b.get("resolved_at"),
                    "similarity_rank": rank,
                })

    pairs_df = pd.DataFrame(rows)

    # Drop duplicate pairs that appeared from both sides
    pairs_df = (
        pairs_df
        .sort_values("similarity_rank")
        .drop_duplicates(subset=["id_a", "id_b"])
        .reset_index(drop=True)
    )

    logger.info(
        "KNN complete — %d unique pairs from %d markets (k=%d)",
        len(pairs_df), n, k,
    )
    return pairs_df


def add_to_index(
    new_embeddings: np.ndarray,
    index_path: Path = _FAISS_INDEX_PATH,
    ids_path: Path = _FAISS_IDS_PATH,
) -> None:
    """
    Incrementally add new embeddings to a persisted faiss index.

    Used in streaming mode when new markets arrive and need to be indexed
    without rebuilding from scratch.

    Args:
        new_embeddings: L2-normalized array of shape (M, D)
        index_path: Path to an existing faiss index file
        ids_path: Path to the parallel ID array
    """
    import faiss

    if not index_path.exists():
        raise FileNotFoundError(
            f"No index found at {index_path}. "
            "Run find_neighbors() first to build the initial index."
        )

    index = faiss.read_index(str(index_path))
    existing_ids = np.load(ids_path)
    next_id = int(existing_ids.max()) + 1 if len(existing_ids) else 0

    index.add(new_embeddings.astype(np.float32))
    new_ids = np.arange(next_id, next_id + len(new_embeddings), dtype=np.int64)

    faiss.write_index(index, str(index_path))
    np.save(ids_path, np.concatenate([existing_ids, new_ids]))
    logger.info("Added %d vectors to faiss index (%s)", len(new_embeddings), index_path)
