"""Sentence segmentation, chunking and tokenization (approximate, offline).

Token counts are whitespace words - a deterministic proxy for a real tokenizer,
which keeps the whole pipeline reproducible without a model dependency.
"""

from __future__ import annotations

import re

from .schemas import Chunk

_SENT_RE = re.compile(r"[^.!?]+[.!?]?")
_WORD_RE = re.compile(r"[a-z0-9]+")

STOPWORDS = frozenset(
    "a an and are as at be by for from in is it of on or that the to was were with".split()
)


def sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_RE.findall(text) if s.strip()]


def tokenize(text: str) -> list[str]:
    return [w for w in _WORD_RE.findall(text.lower()) if w not in STOPWORDS and len(w) > 1]


def token_len(text: str) -> int:
    return len(text.split())


def chunk_document(doc_id: str, text: str, max_tokens: int = 40) -> list[Chunk]:
    """Pack whole sentences into leaf chunks of at most `max_tokens` words."""
    chunks: list[Chunk] = []
    buf: list[str] = []
    buf_len = 0
    order = 0

    def flush() -> None:
        nonlocal buf, buf_len, order
        if buf:
            chunks.append(Chunk(id=f"{doc_id}-c{order}", text=" ".join(buf), order=order))
            order += 1
            buf, buf_len = [], 0

    for sent in sentences(text):
        n = token_len(sent)
        if buf_len + n > max_tokens and buf:
            flush()
        buf.append(sent)
        buf_len += n
    flush()
    return chunks
