from app.evaluation.metrics import (
    precision_at_k, recall_at_k, hit_at_k, reciprocal_rank,
    average_precision, evaluate_single,
)

R = ["A", "B", "C", "D", "E"]


def test_precision():
    assert precision_at_k(R, {"A"}, 1) == 1.0
    assert abs(precision_at_k(R, {"A"}, 3) - 1 / 3) < 1e-9
    assert precision_at_k(R, {"B", "D"}, 4) == 0.5


def test_recall_and_hit():
    assert recall_at_k(R, {"A"}, 3) == 1.0
    assert hit_at_k(R, {"C"}, 2) == 0.0
    assert hit_at_k(R, {"C"}, 3) == 1.0


def test_rank_metrics():
    assert reciprocal_rank(R, {"A"}) == 1.0
    assert abs(reciprocal_rank(R, {"C"}) - 1 / 3) < 1e-9
    assert reciprocal_rank(R, {"Z"}) == 0.0
    assert abs(average_precision(["A", "X", "C"], {"A", "C"}) - (1 + 2 / 3) / 2) < 1e-9


def test_evaluate_single_keys():
    out = evaluate_single(R, {"A"}, (1, 3, 5))
    for key in ("precision@3", "precision@5", "recall@3", "hit@1", "mrr", "ap"):
        assert key in out


def test_context_sufficiency():
    from app.evaluation.metrics import context_sufficiency
    texts = ["The API rate limit is 100 requests per minute."]
    assert context_sufficiency(texts, ["100", "minute"]) == 1.0
    assert context_sufficiency(texts, ["100", "nonsense"]) == 0.5
    assert context_sufficiency(texts, []) == 0.0
