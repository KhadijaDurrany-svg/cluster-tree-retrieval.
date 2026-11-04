# ADR 0002: Budget-aware retrieval and the coverage metric

## Status
Accepted.

## Context
The interesting question is not raw ranking quality but what fits in a fixed
context budget. A summary node is cheap in tokens but stands in for many leaves,
so the budget is where a tree can win or lose.

## Decision
**Packing.** Every method ranks candidate nodes, then greedily packs them into
the token budget, skipping a node that does not fit rather than stopping. The
collapsed tree ranks all nodes (leaves and summaries) together; traversal beam-
searches down the layers and then packs the leaves under the final beam; the
baselines pack leaves only.

**Metric.** A gold supporting chunk counts as covered if it is packed directly or
sits under a packed summary node (provenance). `all-support` requires the whole
gold set to be covered; `recall` is the fraction. Reporting both, split by
local vs global questions, separates "found the one chunk" from "assembled the
whole set".

## Consequences
- Crediting a summary node with its leaves is what lets the tree show its
  advantage on global questions — and the split by question kind keeps that
  honest, because the same crediting gives the tree no edge on local questions,
  which it does not get.
- The comparison is apples-to-apples: all methods pack to the same budget with
  the same packer.
- The headline is a paired permutation test of the collapsed tree against the
  best flat baseline on global all-support, so the claim is a tested one, not an
  eyeballed gap.
