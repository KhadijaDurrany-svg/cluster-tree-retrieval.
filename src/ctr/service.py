"""FastAPI service: index a document, retrieve a context for a question.

  GET  /healthz
  POST /index    {doc_id, text}           -> builds the tree
  POST /retrieve {question, method, budget} -> packed nodes + covered chunks
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .embed import Embedder
from .retrieve import Retriever
from .text import chunk_document
from .tree import build_tree


class IndexRequest(BaseModel):
    doc_id: str = "doc"
    text: str = Field(..., min_length=1)


class RetrieveRequest(BaseModel):
    question: str = Field(..., min_length=1)
    method: str = Field("collapsed_tree", pattern="^(truncation|bm25|dense|collapsed_tree|traversal)$")
    budget: int = Field(120, ge=16, le=4000)


def create_app() -> FastAPI:
    state: dict = {"retr": None}

    app = FastAPI(title="cluster-tree-retrieval", version="0.1.0")

    @app.get("/healthz")
    def healthz() -> dict:
        return {"status": "ok", "indexed": state["retr"] is not None}

    @app.post("/index")
    def index(req: IndexRequest) -> dict:
        chunks = chunk_document(req.doc_id, req.text)
        if not chunks:
            raise HTTPException(status_code=400, detail="no chunks produced from text")
        emb = Embedder().fit([c.text for c in chunks])
        state["retr"] = Retriever(build_tree(chunks, emb), emb, chunks)
        state["tree"] = state["retr"].tree
        return {"status": "indexed", "n_chunks": len(chunks), "layers": [len(x) for x in state["tree"].layers]}

    @app.post("/retrieve")
    def retrieve(req: RetrieveRequest) -> dict:
        retr: Retriever | None = state["retr"]
        if retr is None:
            raise HTTPException(status_code=409, detail="no index; POST /index first")
        method = getattr(retr, req.method)
        packed, covered = (method(req.budget) if req.method == "truncation"
                           else method(req.question, req.budget))
        return {"packed_nodes": packed, "covered_chunks": sorted(covered),
                "context": " ".join(retr.tree.nodes[n].text for n in packed)}

    return app


app = create_app()
