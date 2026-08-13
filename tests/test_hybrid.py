from app.retrieval.hybrid import HybridSearcher, _minmax


class FakeStore:
    def __init__(self, rows): self.rows = rows
    def search(self, query, top_k): return self.rows


class FakeBM25:
    def __init__(self, rows): self.rows = rows
    def search(self, query, top_k): return self.rows


def test_minmax_basic():
    out = _minmax({"a": 0.0, "b": 10.0, "c": 5.0})
    assert out["a"] == 0.0 and out["b"] == 1.0 and 0.49 < out["c"] < 0.51


def test_minmax_all_equal():
    out = _minmax({"a": 3.0, "b": 3.0})
    assert out["a"] == 1.0 and out["b"] == 1.0


def test_fusion_prefers_docs_in_both():
    dense = [{"chunk_id": "A", "score": 0.9}, {"chunk_id": "B", "score": 0.8}]
    sparse = [{"chunk_id": "A", "bm25_score": 5.0}, {"chunk_id": "C", "bm25_score": 4.0}]
    h = HybridSearcher(FakeStore(dense), FakeBM25(sparse), alpha=0.5)
    fused = h.search("q", top_k=10)
    top = fused[0]
    assert top["chunk_id"] == "A"                 # appears in both -> ranks first
    assert top["cosine_score"] == 0.9 and top["bm25_score"] == 5.0
    assert "hybrid_score" in top and top["score"] == top["hybrid_score"]
    ids = {r["chunk_id"] for r in fused}
    assert ids == {"A", "B", "C"}                 # union of both lists


def test_alpha_shifts_weight():
    dense = [{"chunk_id": "D", "score": 1.0}]         # only dense
    sparse = [{"chunk_id": "S", "bm25_score": 1.0}]   # only sparse
    dense_heavy = HybridSearcher(FakeStore(dense), FakeBM25(sparse), alpha=1.0).search("q", 10)
    top = max(dense_heavy, key=lambda r: r["score"])
    assert top["chunk_id"] == "D"                 # alpha=1 -> pure dense wins
