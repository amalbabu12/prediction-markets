"""
K-means clustering of market embeddings.

Cluster count K ≈ floor(N / 10) per the paper, targeting ~10 markets per
cluster on average — small enough for per-cluster LLM pair analysis while
still capturing paraphrases and topically related propositions.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def cluster_markets(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    k: int | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Assign each market to a cluster via K-means.

    Args:
        df: Market DataFrame with at least 'id' and 'question' columns
        embeddings: L2-normalized embeddings, shape (N, D)
        k: Number of clusters. Defaults to floor(N / 10).
        random_state: Seed for reproducibility

    Returns:
        df copy with an added 'cluster_id' integer column
    """
    from sklearn.cluster import KMeans, MiniBatchKMeans

    n = len(df)
    if k is None:
        k = max(1, n // 10)

    logger.info("Clustering %d markets into K=%d clusters ...", n, k)

    # MiniBatchKMeans is substantially faster for large datasets
    if n > 10_000:
        kmeans = MiniBatchKMeans(n_clusters=k, random_state=random_state, n_init=3)
    else:
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)

    labels = kmeans.fit_predict(embeddings)

    df = df.copy()
    df["cluster_id"] = labels

    sizes = pd.Series(labels).value_counts()
    logger.info(
        "Clustering done. Cluster sizes — min=%d  median=%.0f  max=%d",
        sizes.min(), sizes.median(), sizes.max(),
    )
    return df
