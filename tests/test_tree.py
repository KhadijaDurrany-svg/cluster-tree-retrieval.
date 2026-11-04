from ctr.embed import Embedder
from ctr.schemas import Chunk
from ctr.tree import build_tree


def _chunks(n=16):
    topics = ["boats sailing harbor sea", "bread baking flour oven", "ledgers numbers accounts audit"]
    out = []
    for i in range(n):
        t = topics[i % len(topics)]
        out.append(Chunk(id=f"c{i}", text=f"Entry {i} about {t} and more {t} details here.", order=i))
    return out


def _tree():
    chunks = _chunks()
    emb = Embedder().fit([c.text for c in chunks])
    return build_tree(chunks, emb), chunks


def test_tree_shrinks_and_has_depth():
    tree, chunks = _tree()
    assert tree.depth() >= 2
    assert len(tree.layers[0]) == len(chunks)
    # Each higher layer is smaller than the one below it.
    for below, above in zip(tree.layers, tree.layers[1:], strict=False):
        assert len(above) < len(below)


def test_every_leaf_is_reachable_from_the_top():
    tree, chunks = _tree()
    top_leaves = set()
    for nid in tree.layers[-1]:
        top_leaves.update(tree.nodes[nid].leaf_ids)
    assert top_leaves == {c.id for c in chunks}


def test_summaries_use_only_corpus_words():
    # Extractive summaries copy sentences, so every summary word comes from the
    # source text (nothing is generated).
    tree, chunks = _tree()
    corpus_words = set(" ".join(c.text for c in chunks).split())
    for nid in tree.all_ids():
        node = tree.nodes[nid]
        if not node.is_leaf:
            assert set(node.text.split()) <= corpus_words


def test_provenance_leaf_ids_are_real():
    tree, chunks = _tree()
    ids = {c.id for c in chunks}
    for nid in tree.all_ids():
        assert set(tree.nodes[nid].leaf_ids) <= ids
