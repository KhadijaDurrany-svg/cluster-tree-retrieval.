"""cluster-tree-retrieval (ctr).

A RAPTOR-style hierarchical retriever, built and measured fully offline. It
splits a document into leaf chunks, recursively clusters and extractively
summarizes them into a tree, and retrieves either from the collapsed tree or by
traversing it, against flat baselines (truncation, BM25, dense). A synthetic
benchmark with both local-detail and global-synthesis questions shows where the
summary tree helps and where flat retrieval is already enough. No LLM, no model
downloads.
"""

from .schemas import Chunk, Document, Node, Question
from .tree import Tree, build_tree

__all__ = ["Tree", "build_tree", "Chunk", "Node", "Document", "Question"]
__version__ = "0.1.0"
