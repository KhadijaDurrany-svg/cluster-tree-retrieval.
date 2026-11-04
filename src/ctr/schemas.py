"""Typed data models for chunks, tree nodes, documents and questions."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Chunk:
    """A leaf chunk of the source document."""

    id: str
    text: str
    order: int  # position in the document, for truncation baselines


@dataclass
class Node:
    """A tree node. Leaves carry a chunk id; internal nodes carry a summary and
    the leaf ids reachable beneath them (provenance)."""

    id: str
    layer: int
    text: str
    leaf_ids: tuple[str, ...]
    children: tuple[str, ...] = ()
    is_leaf: bool = False

    def token_len(self) -> int:
        return len(self.text.split())


@dataclass
class Question:
    """A QA item. `kind` is 'local' (answer in one chunk) or 'global' (needs
    several chunks). `supporting` are the gold leaf chunk ids."""

    id: str
    text: str
    answer: str
    supporting: tuple[str, ...]
    kind: str = "local"


@dataclass
class Document:
    id: str
    chunks: list[Chunk] = field(default_factory=list)
    questions: list[Question] = field(default_factory=list)
