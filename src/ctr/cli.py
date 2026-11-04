"""CLI: generate data, inspect a document's tree, run the benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .dataset import load_dataset, write_dataset
from .embed import Embedder
from .harness import format_table, run_benchmark, write_results
from .retrieve import Retriever
from .tree import build_tree


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ctr", description="cluster-tree (RAPTOR-style) retrieval")
    sub = parser.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("gen-data", help="write the synthetic dataset")
    g.add_argument("--data-dir", default="data")
    g.add_argument("--seed", type=int, default=29)

    t = sub.add_parser("tree", help="build and describe the tree for one document")
    t.add_argument("--data-dir", default="data")
    t.add_argument("--index", type=int, default=0)

    b = sub.add_parser("bench", help="run the benchmark")
    b.add_argument("--data-dir", default="data")
    b.add_argument("--out", default="bench")
    b.add_argument("--no-write", action="store_true")

    args = parser.parse_args(argv)

    if args.cmd == "gen-data":
        write_dataset(Path(args.data_dir) / "eval.jsonl", seed=args.seed)
        print(f"wrote dataset to {args.data_dir}/eval.jsonl")
        return 0

    if args.cmd == "tree":
        doc = load_dataset(Path(args.data_dir) / "eval.jsonl")[args.index]
        emb = Embedder().fit([c.text for c in doc.chunks])
        tree = build_tree(doc.chunks, emb)
        retr = Retriever(tree, emb, doc.chunks)
        q = next((x for x in doc.questions if x.kind == "global"), doc.questions[0])
        covered = retr.collapsed_tree(q.text, 120)[1]
        print(json.dumps({
            "doc": doc.id, "n_chunks": len(doc.chunks),
            "layers": [len(layer) for layer in tree.layers], "depth": tree.depth(),
            "example_question": q.text, "kind": q.kind,
            "gold_supporting": list(q.supporting),
            "collapsed_tree_covered_supporting": sorted(covered & set(q.supporting)),
        }, indent=2))
        return 0

    if args.cmd == "bench":
        if args.no_write:
            print(format_table(run_benchmark(args.data_dir)))
        else:
            print(format_table(write_results(args.data_dir, args.out)))
            print(f"\nwrote {args.out}/results.json and {args.out}/manifest.json")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
