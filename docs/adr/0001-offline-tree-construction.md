# ADR 0001: Offline tree construction

## Status
Accepted.

## Context
RAPTOR builds a tree by recursively clustering chunk embeddings and summarizing
each cluster. The canonical implementation uses neural embeddings, UMAP
dimensionality reduction, Gaussian-mixture clustering, and an LLM summarizer —
all of which need downloads or an API and make results hard to reproduce.

## Decision
Build the same shape of tree with offline, deterministic parts:

- **Embeddings:** TF-IDF over the corpus projected by truncated SVD (LSA),
  L2-normalized.
- **Clustering:** k-means per layer, with k chosen by the silhouette score over a
  small range; always at least two clusters while a layer is large, so the tree
  keeps shrinking.
- **Summaries:** extractive MMR — sentences near the cluster centroid, penalized
  for redundancy, capped at a token budget. Verbatim, so they are faithful and
  reproducible.

Each internal node records every leaf beneath it (provenance), so retrieval can
map a summary back to the exact chunks it represents.

## Consequences
- The whole pipeline runs with no network and no model files; the benchmark is
  reproducible anywhere.
- Extractive summaries repeat text rather than abstracting it — a real limitation
  the README states. A neural embedder or an abstractive summarizer drops in
  behind the same interfaces.
- Silhouette-selected k-means is simpler than UMAP + GMM and good enough to show
  the budget/compression effect the project is about.
