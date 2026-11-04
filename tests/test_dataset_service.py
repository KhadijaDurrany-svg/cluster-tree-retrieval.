from fastapi.testclient import TestClient

from ctr.dataset import generate
from ctr.service import create_app


def test_generation_deterministic_and_typed():
    a = generate(n_docs=3, people_per_doc=20, seed=7)
    b = generate(n_docs=3, people_per_doc=20, seed=7)
    assert [d.id for d in a] == [d.id for d in b]
    kinds = {q.kind for doc in a for q in doc.questions}
    assert kinds == {"local", "global"}


def test_supporting_chunks_exist():
    for doc in generate(n_docs=2, people_per_doc=20, seed=1):
        ids = {c.id for c in doc.chunks}
        for q in doc.questions:
            assert set(q.supporting) <= ids
            if q.kind == "global":
                assert len(q.supporting) >= 3


def _client():
    return TestClient(create_app())


def test_service_index_then_retrieve():
    with _client() as c:
        text = (". ".join(f"Record {i} concerns boats and sailing and the sea" for i in range(12)) + ".")
        r = c.post("/index", json={"doc_id": "d", "text": text})
        assert r.status_code == 200
        assert r.json()["n_chunks"] >= 1
        r = c.post("/retrieve", json={"question": "boats sailing", "method": "collapsed_tree", "budget": 60})
        assert r.status_code == 200
        assert "covered_chunks" in r.json()


def test_service_retrieve_before_index_409():
    with _client() as c:
        assert c.post("/retrieve", json={"question": "x"}).status_code == 409
