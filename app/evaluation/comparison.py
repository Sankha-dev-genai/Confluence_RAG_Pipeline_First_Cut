"""
Dense-only vs Hybrid retrieval comparison over the golden set.

Runs the evaluation twice (same questions, same reranker/metrics) with hybrid
OFF then ON, and reports aggregate deltas + per-question movement. This is the
"why hybrid" evidence for a demo.
"""
from __future__ import annotations

from typing import Any

from app.evaluation.evaluator import RetrievalEvaluator

HEADLINE = [
    "precision@3", "precision@5", "recall@5",
    "hit@1", "hit@3", "mrr", "ap", "context_sufficiency@5",
]


def run_comparison(
    dense_retriever: Any | None = None,
    hybrid_retriever: Any | None = None,
    k_values=(1, 3, 5),
    top_k: int = 20,
    min_score: float = 0.0,
    hybrid_alpha: float = 0.5,
    golden_path: str | None = None,
    save: bool = True,
) -> dict[str, Any]:
    if dense_retriever is None or hybrid_retriever is None:
        from app.retrieval.retriever import Retriever
        dense_retriever = dense_retriever or Retriever(
            top_k=top_k, min_score=min_score, use_hybrid=False)
        hybrid_retriever = hybrid_retriever or Retriever(
            top_k=top_k, min_score=min_score, use_hybrid=True, hybrid_alpha=hybrid_alpha)

    kw = {"k_values": k_values}
    if golden_path:
        kw["golden_path"] = golden_path
    rep_dense = RetrievalEvaluator(retriever=dense_retriever, **kw).run(save=False)
    rep_hybrid = RetrievalEvaluator(retriever=hybrid_retriever, **kw).run(save=False)

    agg_d, agg_h = rep_dense["aggregate"], rep_hybrid["aggregate"]
    metrics = sorted(set(agg_d) | set(agg_h))
    delta = {m: round(agg_h.get(m, 0.0) - agg_d.get(m, 0.0), 4) for m in metrics}

    dq = {q["id"]: q for q in rep_dense["per_query"]}
    per_query = []
    for q in rep_hybrid["per_query"]:
        dm = dq.get(q["id"], {}).get("metrics", {})
        hm = q["metrics"]
        per_query.append({
            "id": q["id"],
            "question": q["question"],
            "dense_mrr": round(dm.get("mrr", 0.0), 3),
            "hybrid_mrr": round(hm.get("mrr", 0.0), 3),
            "dense_p@3": round(dm.get("precision@3", 0.0), 3),
            "hybrid_p@3": round(hm.get("precision@3", 0.0), 3),
            "dense_ctx@5": round(dm.get("context_sufficiency@5", 0.0), 3),
            "hybrid_ctx@5": round(hm.get("context_sufficiency@5", 0.0), 3),
        })

    result = {
        "num_questions": rep_hybrid["num_questions"],
        "hybrid_alpha": hybrid_alpha,
        "dense": agg_d,
        "hybrid": agg_h,
        "delta": delta,
        "per_query": per_query,
    }

    if save:
        from app.core.config import EVAL_DIR
        import json
        (EVAL_DIR / "comparison.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    return result


def print_comparison(result: dict[str, Any]) -> None:
    d, h, delta = result["dense"], result["hybrid"], result["delta"]
    print("=" * 64)
    print(f"DENSE vs HYBRID  (alpha={result['hybrid_alpha']}, "
          f"{result['num_questions']} questions)")
    print("=" * 64)
    print(f"{'metric':22s} {'dense':>8s} {'hybrid':>8s} {'delta':>8s}")
    print("-" * 64)
    for m in HEADLINE:
        if m in d or m in h:
            dv, hv, dl = d.get(m, 0.0), h.get(m, 0.0), delta.get(m, 0.0)
            arrow = "  ▲" if dl > 1e-6 else ("  ▼" if dl < -1e-6 else "   ")
            print(f"{m:22s} {dv:8.3f} {hv:8.3f} {dl:+8.3f}{arrow}")
    print("=" * 64)
