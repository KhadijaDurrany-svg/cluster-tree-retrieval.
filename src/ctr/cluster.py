"""Clustering for one tree layer.

k-means with k chosen by the silhouette score over a small range. Deterministic
given the seed. Always returns at least two clusters when there is something to
split, so the tree keeps shrinking layer by layer.
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def choose_and_cluster(x: np.ndarray, max_k: int = 8, seed: int = 0) -> tuple[np.ndarray, int]:
    n = x.shape[0]
    if n <= 2:
        return np.zeros(n, dtype=int), 1
    best_k, best_score, best_labels = 0, -1.0, None
    for k in range(2, min(max_k, n - 1) + 1):
        labels = KMeans(n_clusters=k, n_init=5, random_state=seed).fit_predict(x)
        if len(set(labels)) < 2:
            continue
        score = silhouette_score(x, labels)
        if score > best_score:
            best_k, best_score, best_labels = k, score, labels
    if best_labels is None:
        best_labels = KMeans(n_clusters=2, n_init=5, random_state=seed).fit_predict(x)
        best_k = 2
    return best_labels, best_k
