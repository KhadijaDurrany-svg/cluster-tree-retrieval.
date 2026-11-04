# Benchmark methodology

## Documents
Synthetic and deterministic, from `ctr.dataset` (seed 29). 32 documents, each
describing ~40 people with one fact-sentence apiece (place, object, job), packed
into ~36-token chunks.

## Questions
- **local**: "Where did PersonNN live?" — answer in one chunk; gold supporting is
  that single chunk.
- **global**: "Which people owned a <object>?" — answer spread across every chunk
  mentioning an owner of that object; gold supporting is that set (≥3 chunks).

Global gold sets are deliberately larger than the token budget can hold as raw
chunks, so covering them all requires either luck or a summary node.

## Retrieval and budget
Fixed budget of 120 tokens. Methods: `truncation` (first tokens), `bm25`,
`dense`, `collapsed_tree`, `traversal`. Each ranks candidates and packs them into
the budget, skipping oversized nodes.

## Metrics
- **all-support** — 1 only if every gold supporting chunk is covered (packed
  directly or under a packed summary node).
- **recall** — fraction of gold supporting chunks covered.
- **bootstrap CI** — 2,000-sample percentile bootstrap.
- **paired permutation test** — 10,000-sample, collapsed tree vs best flat on
  global all-support.

## Reproducing
```bash
ctr gen-data --data-dir data   # deterministic
ctr bench --data-dir data      # prints the table, writes bench/
```

## Reading the result
The tree's advantage is concentrated on global questions and is a tested effect
(p ≈ 0.01). It depends on the budget being tight relative to the global gold
sets; with a generous budget, flat retrieval fits what it needs and the tree
ties — which is what the RAPTOR reproduction on QuALITY reports.
