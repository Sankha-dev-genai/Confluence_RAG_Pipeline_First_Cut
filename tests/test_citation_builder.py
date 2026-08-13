from app.retrieval.citation_builder import CitationBuilder


def test_build_basic():
    chunks = [
        {"chunk_id": "a1", "page_id": "1", "title": "Getting Started",
         "section": "Overview", "text": "word " * 100,
         "score": 0.9, "embedding_score": 0.7, "source_url": "http://x"},
        {"chunk_id": "b1", "page_id": "2", "title": "API",
         "section": "Auth", "text": "short text", "score": 0.5,
         "embedding_score": 0.5, "source_url": ""},
    ]
    cites = CitationBuilder.build(chunks)
    assert [c["index"] for c in cites] == [1, 2]
    assert cites[0]["similarity"] == 0.7      # uses embedding_score
    assert cites[0]["rerank_score"] == 0.9
    assert cites[0]["snippet"].endswith("...")  # long text trimmed
    assert cites[1]["snippet"] == "short text"


def test_max_citations():
    chunks = [{"chunk_id": str(i), "page_id": str(i), "title": "t",
               "text": "x", "score": 0.1, "embedding_score": 0.1} for i in range(5)]
    assert len(CitationBuilder.build(chunks, max_citations=3)) == 3
