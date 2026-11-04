"""Extractive summarization by Maximal Marginal Relevance (MMR).

Picks sentences close to the cluster centroid while penalizing redundancy, up to
a token budget. Summaries are verbatim sentences (no generation), so they stay
reproducible and faithful - at the cost, noted in the README, of repeating text.
"""

from __future__ import annotations

import numpy as np

from .text import token_len


def extractive_summary(sents: list[str], sent_emb: np.ndarray, centroid: np.ndarray,
                       max_tokens: int = 50, lambda_mmr: float = 0.6) -> str:
    if not sents:
        return ""
    selected: list[int] = []
    total = 0
    candidates = list(range(len(sents)))
    while candidates and total < max_tokens:
        best, best_val = candidates[0], -1e9
        for i in candidates:
            rel = float(sent_emb[i] @ centroid)
            red = max((float(sent_emb[i] @ sent_emb[j]) for j in selected), default=0.0)
            val = lambda_mmr * rel - (1.0 - lambda_mmr) * red
            if val > best_val:
                best, best_val = i, val
        tk = token_len(sents[best])
        if total + tk > max_tokens and selected:
            break
        selected.append(best)
        total += tk
        candidates.remove(best)
    selected.sort()  # restore document order
    return " ".join(sents[i] for i in selected)
