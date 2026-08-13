from app.evaluation.comparison import run_comparison


class FakeRetriever:
    """Returns a fixed ranking for every query."""
    def __init__(self, page_id):
        self.page_id = page_id
    def retrieve(self, query):
        return [{"chunk_id": f"{self.page_id}_1", "page_id": self.page_id,
                 "text": "the api rate limit is 100 per minute widget celery kubectl",
                 "score": 0.9, "embedding_score": 0.9}]


def test_comparison_structure_and_delta():
    res = run_comparison(
        dense_retriever=FakeRetriever("000"),      # never correct
        hybrid_retriever=FakeRetriever("131133"),  # correct for some questions
        save=False,
    )
    for key in ("dense", "hybrid", "delta", "per_query", "num_questions"):
        assert key in res
    # delta is arithmetic difference of the aggregates
    for m in res["delta"]:
        assert abs(res["delta"][m]
                   - round(res["hybrid"].get(m, 0.0) - res["dense"].get(m, 0.0), 4)) < 1e-9
    # hybrid (sometimes correct) should not be worse than dense (never correct) on hit@1
    assert res["hybrid"].get("hit@1", 0) >= res["dense"].get("hit@1", 0)
    assert len(res["per_query"]) == res["num_questions"]
