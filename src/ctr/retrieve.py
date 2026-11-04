"""Retrieval methods over the tree and flat baselines, under a token budget.

Every method returns the node ids it packed and the set of leaf chunks those
nodes cover (a summary node covers every leaf beneath it). Packing is greedy and
skips a node that does not fit rather than stopping, so the budget is filled.
"""

from __future__ import annotations

import math

from .embed import Embedder
from .schemas import Chunk
from .text import tokenize
from .tree import Tree


class _BM25:
    def __init__(self, ids: list[str], texts: list[str], k1: float = 1.2, b: float = 0.75) -> None:
        self.ids, self.k1, self.b = ids, k1, b
        self._tf = [dict() for _ in texts]
        self._len = []
        df: dict[str, int] = {}
        for i, t in enumerate(texts):
            toks = tokenize(t)
            self._len.append(len(toks))
            for w in toks:
                self._tf[i][w] = self._tf[i].get(w, 0) + 1
            for w in set(toks):
                df[w] = df.get(w, 0) + 1
        n = len(texts)
        self._avgdl = sum(self._len) / n if n else 0.0
        self._idf = {w: math.log(1 + (n - d + 0.5) / (d + 0.5)) for w, d in df.items()}

    def scores(self, query: str) -> dict[str, float]:
        q = [w for w in tokenize(query) if w in self._idf]
        out: dict[str, float] = {}
        for i, tf in enumerate(self._tf):
            s = 0.0
            for w in q:
                f = tf.get(w, 0)
                if f:
                    denom = f + self.k1 * (1 - self.b + self.b * self._len[i] / self._avgdl)
                    s += self._idf[w] * f * (self.k1 + 1) / denom
            if s > 0:
                out[self.ids[i]] = s
        return out


class Retriever:
    def __init__(self, tree: Tree, embedder: Embedder, chunks: list[Chunk]) -> None:
        self.tree = tree
        self.embedder = embedder
        self.chunks = chunks
        self.leaf_ids = [c.id for c in chunks]
        self._bm25 = _BM25(self.leaf_ids, [c.text for c in chunks])

    # ------------------------------------------------------------- helpers
    def _tokens(self, nid: str) -> int:
        return self.tree.nodes[nid].token_len()

    def _pack(self, ranked: list[str], budget: int) -> list[str]:
        packed, total = [], 0
        for nid in ranked:
            t = self._tokens(nid)
            if total + t <= budget:
                packed.append(nid)
                total += t
        return packed

    def _covered(self, packed: list[str]) -> set[str]:
        covered: set[str] = set()
        for nid in packed:
            covered.update(self.tree.nodes[nid].leaf_ids)
        return covered

    def _dense_scores(self, query: str, ids: list[str]) -> dict[str, float]:
        q = self.embedder.encode([query])[0]
        mat = self.tree.matrix(ids)
        sims = mat @ q
        return {ids[i]: float(sims[i]) for i in range(len(ids))}

    # ------------------------------------------------------------- methods
    def truncation(self, budget: int) -> tuple[list[str], set[str]]:
        packed = self._pack(self.leaf_ids, budget)
        return packed, self._covered(packed)

    def bm25(self, query: str, budget: int) -> tuple[list[str], set[str]]:
        scores = self._bm25.scores(query)
        ranked = sorted(self.leaf_ids, key=lambda i: -scores.get(i, 0.0))
        packed = self._pack(ranked, budget)
        return packed, self._covered(packed)

    def dense(self, query: str, budget: int) -> tuple[list[str], set[str]]:
        scores = self._dense_scores(query, self.leaf_ids)
        ranked = sorted(self.leaf_ids, key=lambda i: -scores[i])
        packed = self._pack(ranked, budget)
        return packed, self._covered(packed)

    def collapsed_tree(self, query: str, budget: int) -> tuple[list[str], set[str]]:
        ids = self.tree.all_ids()
        scores = self._dense_scores(query, ids)
        ranked = sorted(ids, key=lambda i: -scores[i])
        packed = self._pack(ranked, budget)
        return packed, self._covered(packed)

    def traversal(self, query: str, budget: int, beam: int = 3) -> tuple[list[str], set[str]]:
        q = self.embedder.encode([query])[0]

        def score(nid: str) -> float:
            return float(self.tree.embeddings[nid] @ q)

        frontier = list(self.tree.layers[-1])
        while frontier and self.tree.nodes[frontier[0]].layer > 0:
            frontier.sort(key=lambda i: -score(i))
            kept = frontier[:beam]
            children = [c for nid in kept for c in self.tree.nodes[nid].children]
            frontier = children or kept  # stop if no children
            if all(self.tree.nodes[i].layer == 0 for i in frontier):
                break
        leaves = [i for i in frontier if self.tree.nodes[i].is_leaf] or \
                 [lid for nid in frontier for lid in self.tree.nodes[nid].leaf_ids]
        ranked = sorted(set(leaves), key=lambda i: -score(i))
        packed = self._pack(ranked, budget)
        return packed, self._covered(packed)
