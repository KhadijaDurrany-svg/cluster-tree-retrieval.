# cluster-tree-retrieval

A RAPTOR-style hierarchical retriever, built and measured **fully offline**. It
splits a document into leaf chunks, recursively clusters and extractively
summarizes them into a tree, then retrieves either from the **collapsed tree**
or by **traversing** it — against flat baselines (truncation, BM25, dense). The
question it answers: under a fixed token budget, when does a summary tree beat
flat retrieval?

**No LLM, no model downloads.** Embeddings are TF-IDF + SVD, summaries are
extractive, clustering is k-means. Every number regenerates with one command.

```
document ─▶ chunk ─▶ [cluster layer ─▶ extractive-summarize each cluster] ⟲ ─▶ tree
                                      retrieve: collapsed tree | traversal | flat
```

## The idea, and the honest result

A summary node is small but points to many leaves. Under a tight token budget
that cannot fit many raw chunks, one summary node can "cover" a whole region
cheaply. So the tree should help **global** questions (the answer is spread
across many chunks) and add nothing to **local** questions (the answer is in one
chunk, which flat retrieval already finds). The benchmark is built to test
exactly that split — and it bears out.

## What is in the box

- **`ctr.text`** — sentence segmentation and budget-aware chunking.
- **`ctr.embed`** — offline TF-IDF + truncated-SVD embedder.
- **`ctr.cluster`** — k-means with k chosen by silhouette, per layer.
- **`ctr.summarize`** — MMR extractive summaries, capped to a token budget.
- **`ctr.tree`** — recursive tree construction with full leaf provenance.
- **`ctr.retrieve`** — collapsed-tree, traversal, and truncation/BM25/dense
  baselines, with budget-aware packing that skips oversized nodes.
- **`ctr.metrics` / `ctr.harness`** — supporting-chunk recall and all-support,
  bootstrap CIs, a paired permutation test.
- **`ctr.service`** — a FastAPI app (`/index`, `/retrieve`).

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Python 3.10+.

## Quickstart

```bash
ctr gen-data --data-dir data     # deterministic synthetic documents
ctr tree --data-dir data --index 0   # build a tree, show its layers
ctr bench --data-dir data        # reproduce the table below
uvicorn ctr.service:app --reload  # POST /index a document, /retrieve a context
```

## Benchmark

32 synthetic documents (≈40 facts each), with local-detail and global-synthesis
questions. Retrieval runs at a **120-token budget**. A supporting chunk counts as
covered if it is packed directly or sits under a packed summary node.
`all-support` is 1 only if the whole gold set is covered; `recall` is the
fraction covered. CIs are a 2,000-sample bootstrap; significance is a
10,000-sample paired permutation test.

Reproduce with `ctr bench --data-dir data`. Source of truth:
[`bench/results.json`](bench/results.json).

| Method | Local all-support | Global all-support (95% CI) | Global recall |
|---|---|---|---|
| truncation (first tokens) | 0.231 | 0.000 [0.00, 0.00] | 0.269 |
| BM25 | 1.000 | 0.071 [0.03, 0.11] | 0.587 |
| dense | 1.000 | 0.071 [0.03, 0.11] | 0.587 |
| **collapsed tree** | 1.000 | **0.137 [0.09, 0.19]** | **0.663** |
| traversal | 0.988 | 0.071 [0.03, 0.11] | 0.585 |

Collapsed tree vs best flat (BM25) on global all-support: **+0.066, p = 0.013**.

### How to read this honestly

- **The tree helps global questions, significantly.** On questions whose answer
  is spread across many chunks, the collapsed tree nearly doubles all-support
  (0.137 vs 0.071) and lifts recall (0.663 vs 0.587), p = 0.013. That is the
  compression mechanism: within a 120-token budget, a summary node covers a whole
  region that flat retrieval cannot fit as raw chunks.
- **The tree adds nothing to local questions.** When the answer is in one chunk,
  BM25, dense and the tree all score 1.000 — flat retrieval already finds it, so
  the hierarchy is wasted effort there.
- **Absolute global all-support is low for everyone** (0.14 even for the tree),
  because the gold sets are large and the budget is deliberately tight. Recall is
  the gentler read of the same effect. Truncation is near useless on global
  questions, as expected.
- **Traversal does not beat the collapsed tree** here (it ties the flat
  baselines), matching the original RAPTOR finding that collapsed-tree retrieval
  is the stronger of the two modes.

## Known gaps and caveats

- Synthetic documents and extractive summaries isolate the budget/compression
  mechanism; they are not QuALITY or a real long-document corpus. The published
  RAPTOR reproduction on QuALITY finds the tree ties BM25/dense — because there
  the budget is generous enough that flat retrieval already fits what it needs.
  The effect here appears precisely because the budget is tight relative to the
  global gold sets.
- Embeddings are TF-IDF + SVD, not a neural model; clustering is flat k-means,
  not UMAP + GMM. Both are deliberate, for a zero-download build; swapping in a
  neural embedder is a local change.
- No reader/LLM is run. The benchmark measures retrieval coverage, not end-task
  QA accuracy.

## Design notes

Decision records are in [`docs/adr/`](docs/adr/): the tree construction choices
and the budget-aware retrieval and metric.

## Testing

```bash
pytest -q       # 20 tests
ruff check .
```

Tests cover tree invariants (every leaf reachable from the top, provenance,
extractive-only summaries), budget compliance for every retrieval method, and
the metrics.

## License

MIT — see [LICENSE](LICENSE).
