from ctr.text import chunk_document, sentences, token_len, tokenize


def test_sentences_split():
    assert sentences("A cat sat. A dog ran! Ok?") == ["A cat sat.", "A dog ran!", "Ok?"]


def test_tokenize_drops_stopwords():
    assert "the" not in tokenize("the quick brown fox")


def test_chunks_respect_budget():
    text = " ".join(f"Sentence number {i} has several words here." for i in range(20))
    chunks = chunk_document("d", text, max_tokens=20)
    assert len(chunks) > 1
    # No chunk wildly exceeds the budget (a single long sentence may).
    assert all(token_len(c.text) <= 40 for c in chunks)


def test_chunks_cover_all_sentences_in_order():
    text = "One two three. Four five six. Seven eight nine."
    chunks = chunk_document("d", text, max_tokens=4)
    joined = " ".join(c.text for c in chunks)
    for s in ("One two three.", "Four five six.", "Seven eight nine."):
        assert s in joined
    assert [c.order for c in chunks] == list(range(len(chunks)))
