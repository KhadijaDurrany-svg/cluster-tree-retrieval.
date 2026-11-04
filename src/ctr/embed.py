"""Offline text embedder: TF-IDF over the corpus, projected by truncated SVD.

Deterministic and dependency-light (no model download), so the tree and the
retrieval are fully reproducible. Vectors are L2-normalized, so cosine
similarity is a dot product.
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from .text import tokenize


def _l2(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return mat / norms


class Embedder:
    def __init__(self, dim: int = 64, seed: int = 13) -> None:
        self.dim = dim
        self.seed = seed
        self._vec = TfidfVectorizer(analyzer=tokenize, sublinear_tf=True, norm="l2")
        self._svd: TruncatedSVD | None = None
        self._fitted = False

    def fit(self, texts: list[str]) -> Embedder:
        if not texts:
            raise ValueError("cannot fit on empty corpus")
        tfidf = self._vec.fit_transform(texts)
        n, f = tfidf.shape
        dim = max(1, min(self.dim, n - 1, f - 1))
        self._svd = TruncatedSVD(n_components=dim, random_state=self.seed)
        self._svd.fit(tfidf)
        self.dim = dim
        self._fitted = True
        return self

    def encode(self, texts: list[str]) -> np.ndarray:
        if not self._fitted or self._svd is None:
            raise RuntimeError("embedder must be fit() before encode()")
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return _l2(self._svd.transform(self._vec.transform(texts)).astype(np.float32))
