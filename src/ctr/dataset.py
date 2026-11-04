"""Synthetic documents with local-detail and global-synthesis questions.

Each document describes many people, one fact-sentence each, packed into chunks.
A *local* question asks about one person (answer lives in one chunk). A *global*
question asks which people share an attribute (answer is spread across several
chunks). Global questions need more supporting chunks than the token budget fits
as raw leaves, which is exactly where a summary node - one small node that covers
many leaves - can help.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from .schemas import Chunk, Document, Question
from .text import chunk_document

_PLACES = ["Bremen", "Calden", "Dorne", "Ellis", "Faro", "Gelt", "Harwick", "Iver"]
_OBJECTS = ["boat", "lantern", "ledger", "compass", "loom", "kiln"]
_JOBS = ["baker", "sailor", "scribe", "smith", "weaver", "miller"]


def _person(i: int) -> str:
    return f"Person{i:02d}"


def generate(n_docs: int = 32, people_per_doc: int = 40, seed: int = 29) -> list[Document]:
    rng = random.Random(seed)
    docs: list[Document] = []
    for d in range(n_docs):
        facts = []
        obj_of: dict[str, str] = {}
        place_of: dict[str, str] = {}
        for i in range(people_per_doc):
            p = _person(i)
            place = rng.choice(_PLACES)
            obj = rng.choice(_OBJECTS)
            job = rng.choice(_JOBS)
            place_of[p], obj_of[p] = place, obj
            facts.append(f"{p} lived in {place}, owned a {obj}, and worked as a {job}.")
        rng.shuffle(facts)
        text = " ".join(facts)
        chunks = chunk_document(f"d{d}", text, max_tokens=36)

        # Map each person to the chunk that mentions them.
        chunk_of: dict[str, str] = {}
        for c in chunks:
            for i in range(people_per_doc):
                if _person(i) in c.text:
                    chunk_of[_person(i)] = c.id

        questions: list[Question] = []
        # Local questions: a few random people.
        for i in rng.sample(range(people_per_doc), k=5):
            p = _person(i)
            questions.append(Question(
                id=f"d{d}-local{i}", text=f"Where did {p} live?", answer=place_of[p],
                supporting=(chunk_of[p],), kind="local"))
        # Global questions: which people owned a given object (spread out).
        for obj in _OBJECTS:
            owners = [_person(i) for i in range(people_per_doc) if obj_of[_person(i)] == obj]
            support = tuple(sorted({chunk_of[p] for p in owners}))
            if len(support) >= 3:
                questions.append(Question(
                    id=f"d{d}-global-{obj}", text=f"Which people owned a {obj}?",
                    answer=",".join(owners), supporting=support, kind="global"))

        docs.append(Document(id=f"d{d}", chunks=chunks, questions=questions))
    return docs


# ------------------------------------------------------------------- IO
def write_dataset(path: str | Path, **kwargs) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for doc in generate(**kwargs):
            fh.write(json.dumps({
                "id": doc.id,
                "chunks": [{"id": c.id, "text": c.text, "order": c.order} for c in doc.chunks],
                "questions": [{"id": q.id, "text": q.text, "answer": q.answer,
                               "supporting": list(q.supporting), "kind": q.kind} for q in doc.questions],
            }) + "\n")


def load_dataset(path: str | Path) -> list[Document]:
    out: list[Document] = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            chunks = [Chunk(id=c["id"], text=c["text"], order=c["order"]) for c in r["chunks"]]
            questions = [Question(id=q["id"], text=q["text"], answer=q["answer"],
                                  supporting=tuple(q["supporting"]), kind=q["kind"]) for q in r["questions"]]
            out.append(Document(id=r["id"], chunks=chunks, questions=questions))
    return out
