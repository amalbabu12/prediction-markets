"""
Embed market questions using sentence-transformers.

Embeddings are cached to disk to avoid recomputation on repeat runs.
Cache is keyed on the ordered list of questions + model name, so any
change to the question set triggers a fresh computation.
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_CACHE_DIR = Path("data/embeddings")
_MODEL_NAME = "all-MiniLM-L6-v2"


def embed_questions(
    df: pd.DataFrame,
    cache_dir: Path = _CACHE_DIR,
    model_name: str = _MODEL_NAME,
) -> np.ndarray:
    """
    Compute (or load cached) L2-normalized sentence embeddings for each row in df.

    Args:
        df: DataFrame with a 'question' column; order must be stable
        cache_dir: Directory to cache .npy embedding arrays
        model_name: sentence-transformers model name

    Returns:
        np.ndarray of shape (N, D) — one row per market, L2-normalized
    """
    from sentence_transformers import SentenceTransformer

    cache_dir.mkdir(parents=True, exist_ok=True)

    # Cache key depends on model and the ordered question list (order-sensitive)
    key = hashlib.md5(
        (model_name + "||".join(df["question"].tolist())).encode()
    ).hexdigest()[:16]
    cache_file = cache_dir / f"embeddings_{key}.npy"

    if cache_file.exists():
        logger.info("Loading cached embeddings from %s", cache_file)
        return np.load(cache_file)

    logger.info(
        "Computing embeddings for %d questions with %s ...", len(df), model_name
    )
    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        df["question"].tolist(),
        batch_size=256,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    np.save(cache_file, embeddings)
    logger.info("Embeddings cached to %s", cache_file)
    return embeddings
