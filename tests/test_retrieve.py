from ctr.embed import Embedder
from ctr.retrieve import Retriever
from ctr.schemas import Chunk
from ctr.text import token_len
from ctr.tree import build_tree


def _retriever(n=18):
    topics = ["boats sailing harbor sea voyage", "bread baking flour oven dough", "ledger numbers audit tax coin"]
    chunks = [Chunk(id=f"c{i}", text=f"Record {i} concerns {topics[i % 3]} in detail.", order=i)
              for i in range(n)]
    emb = Embedder().fit([c.text for c in chunks])
    return Retriever(build_tree(chunks, emb), emb, chunks), chunks


def test_budget_is_respected_for_every_method():
    retr, _ = _retriever()
    budget = 40
    for packed, _covered in (
        retr.truncation(budget),
        retr.bm25("boats sailing", budget),
        retr.dense("boats sailing", budget),
        retr.collapsed_tree("boats sailing", budget),
        retr.traversal("boats sailing", budget),
    ):
        total = sum(token_len(retr.tree.nodes[n].text) for n in packed)
        assert total <= budget


def test_covered_leaves_are_real_chunks():
    retr, chunks = _retriever()
    ids = {c.id for c in chunks}
    _packed, covered = retr.collapsed_tree("bread baking", 60)
    assert covered <= ids


def test_bm25_prefers_on_topic_chunks():
    retr, _ = _retriever()
    _packed, covered = retr.bm25("ledger audit tax", 40)
    # At least one retrieved chunk should be a ledger chunk (ids c2, c5, c8, ...).
    assert any(c.endswith(("2", "5", "8")) for c in covered)
