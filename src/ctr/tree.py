"""Recursive tree construction: cluster a layer, summarize each cluster, repeat.

Leaves are the document chunks. Each higher layer clusters the layer below and
replaces each cluster with one extractive-summary node that remembers every leaf
beneath it (provenance). Building stops when a layer is small enough or stops
shrinking.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .cluster import choose_and_cluster
from .embed import Embedder
from .schemas import Chunk, Node
from .summarize import extractive_summary
from .text import sentences


@dataclass
class Tree:
    nodes: dict[str, Node]
    embeddings: dict[str, np.ndarray]
    layers: list[list[str]]

    def leaf_ids(self) -> list[str]:
        return list(self.layers[0])

    def all_ids(self) -> list[str]:
        return [nid for layer in self.layers for nid in layer]

    def matrix(self, ids: list[str]) -> np.ndarray:
        return np.stack([self.embeddings[i] for i in ids]) if ids else np.zeros((0, 1))

    def depth(self) -> int:
        return len(self.layers)


def build_tree(chunks: list[Chunk], embedder: Embedder, max_summary_tokens: int = 50,
               min_top: int = 4, max_k: int = 8, seed: int = 0) -> Tree:
    if not chunks:
        raise ValueError("cannot build a tree from no chunks")
    nodes: dict[str, Node] = {}
    embeddings: dict[str, np.ndarray] = {}

    leaf_vecs = embedder.encode([c.text for c in chunks])
    for i, c in enumerate(chunks):
        nodes[c.id] = Node(id=c.id, layer=0, text=c.text, leaf_ids=(c.id,), is_leaf=True)
        embeddings[c.id] = leaf_vecs[i]
    layers: list[list[str]] = [[c.id for c in chunks]]

    current = list(layers[0])
    layer_idx = 0
    counter = 0
    while len(current) > min_top:
        layer_idx += 1
        x = np.stack([embeddings[i] for i in current])
        labels, _ = choose_and_cluster(x, max_k=max_k, seed=seed)
        new_ids: list[str] = []
        for cl in sorted(set(labels)):
            members = [current[i] for i in range(len(current)) if labels[i] == cl]
            leaf_ids = tuple(sorted({lid for m in members for lid in nodes[m].leaf_ids}))
            sents = sentences(" ".join(nodes[m].text for m in members))
            if sents:
                s_emb = embedder.encode(sents)
                centroid = np.mean([embeddings[m] for m in members], axis=0)
                centroid = centroid / (np.linalg.norm(centroid) or 1.0)
                summary = extractive_summary(sents, s_emb, centroid, max_summary_tokens)
            else:
                summary = " ".join(nodes[m].text for m in members)[:200]
            nid = f"L{layer_idx}n{counter}"
            counter += 1
            nodes[nid] = Node(id=nid, layer=layer_idx, text=summary, leaf_ids=leaf_ids,
                              children=tuple(members))
            embeddings[nid] = embedder.encode([summary])[0]
            new_ids.append(nid)
        layers.append(new_ids)
        if len(new_ids) >= len(current):
            break  # stopped shrinking
        current = new_ids
    return Tree(nodes=nodes, embeddings=embeddings, layers=layers)
