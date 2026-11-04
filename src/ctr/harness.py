"""Benchmark harness: tree vs flat retrieval under a fixed token budget.

For every document it builds one tree, then runs each method on every question
at a fixed budget, scoring supporting-chunk recall and all-support (did the
retrieved context cover the whole gold set). Results are reported overall and
split by question kind (local vs global), with bootstrap CIs and a paired
permutation test of the collapsed tree against the best flat baseline.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from statistics import mean

from .dataset import load_dataset
from .embed import Embedder
from .metrics import all_support, bootstrap_ci, paired_permutation_test, support_recall
from .retrieve import Retriever
from .tree import build_tree

BUDGET = 120
METHODS = ["truncation", "bm25", "dense", "collapsed_tree", "traversal"]


def _run_method(retr: Retriever, method: str, query: str) -> set[str]:
    if method == "truncation":
        return retr.truncation(BUDGET)[1]
    if method == "bm25":
        return retr.bm25(query, BUDGET)[1]
    if method == "dense":
        return retr.dense(query, BUDGET)[1]
    if method == "collapsed_tree":
        return retr.collapsed_tree(query, BUDGET)[1]
    return retr.traversal(query, BUDGET)[1]


def run_benchmark(data_dir: str | Path) -> dict:
    docs = load_dataset(Path(data_dir) / "eval.jsonl")
    # Per (method, kind) accumulators of all-support and recall.
    allsup: dict[str, dict[str, list[float]]] = {m: {"local": [], "global": []} for m in METHODS}
    recall: dict[str, dict[str, list[float]]] = {m: {"local": [], "global": []} for m in METHODS}
    # Paired vectors on global all-support: collapsed_tree vs best flat (bm25/dense).
    paired: dict[str, list[float]] = {m: [] for m in METHODS}

    for doc in docs:
        emb = Embedder().fit([c.text for c in doc.chunks])
        tree = build_tree(doc.chunks, emb)
        retr = Retriever(tree, emb, doc.chunks)
        for q in doc.questions:
            for m in METHODS:
                covered = _run_method(retr, m, q.text)
                allsup[m][q.kind].append(all_support(covered, q.supporting))
                recall[m][q.kind].append(support_recall(covered, q.supporting))
                if q.kind == "global":
                    paired[m].append(all_support(covered, q.supporting))

    def agg(d: dict[str, list[float]]) -> dict:
        out = {}
        for kind, vals in d.items():
            m, lo, hi = bootstrap_ci(vals)
            out[kind] = {"mean": round(m, 4), "ci_lo": round(lo, 4), "ci_hi": round(hi, 4), "n": len(vals)}
        return out

    results = {m: {"all_support": agg(allsup[m]), "recall": agg(recall[m])} for m in METHODS}

    best_flat = max(("bm25", "dense"), key=lambda m: mean(paired[m]) if paired[m] else 0.0)
    tree_vec = paired["collapsed_tree"]
    delta = round(mean(tree_vec) - mean(paired[best_flat]), 4) if tree_vec else 0.0
    sig = {
        "metric": "global all_support",
        "collapsed_tree_vs_best_flat": {
            "best_flat": best_flat,
            "delta": delta,
            "p_value": round(paired_permutation_test(tree_vec, paired[best_flat]), 4),
        },
    }
    return {"budget_tokens": BUDGET, "n_docs": len(docs), "configs": results, "significance": sig}


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def write_results(data_dir: str | Path, out_dir: str | Path) -> dict:
    data_dir, out_dir = Path(data_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = run_benchmark(data_dir)
    manifest = {"git_sha": _git_sha(),
                "eval_sha256": hashlib.sha256((data_dir / "eval.jsonl").read_bytes()).hexdigest()}
    (out_dir / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return result


def format_table(result: dict) -> str:
    lines = [f"budget: {result['budget_tokens']} tokens, {result['n_docs']} documents"]
    header = f"{'method':<16} {'local all-sup':>14} {'global all-sup':>16} {'global recall':>14}"
    lines.append(header)
    lines.append("-" * len(header))
    for m, agg in result["configs"].items():
        la = agg["all_support"]["local"]["mean"]
        ga = agg["all_support"]["global"]
        gr = agg["recall"]["global"]["mean"]
        lines.append(f"{m:<16} {la:>14.3f} {ga['mean']:>9.3f} [{ga['ci_lo']:.2f},{ga['ci_hi']:.2f}] {gr:>14.3f}")
    s = result["significance"]["collapsed_tree_vs_best_flat"]
    lines.append("")
    lines.append(f"global all-support, collapsed_tree vs {s['best_flat']}: "
                 f"delta={s['delta']:+.3f}, p={s['p_value']:.4f}")
    return "\n".join(lines)
