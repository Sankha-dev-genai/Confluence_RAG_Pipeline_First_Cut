from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Sequence

from app.evaluation.metrics import evaluate_single, _dedup_keep_order, context_sufficiency

GOLDEN_PATH = Path("data/evaluation/golden_qa.json")
RESULTS_PATH = Path("data/evaluation/results.json")


class RetrievalEvaluator:
    """
    Runs the golden question set through the retriever and computes
    ranked-retrieval metrics (Precision@k, Recall@k, Hit@k, MRR, MAP)
    at the PAGE level (the unit a citation points a user to).

    No LLM is required — this measures the retrieval foundation, which
    is what Precision@k is actually about.
    """

    def __init__(
        self,
        retriever: Any | None = None,
        golden_path: str | Path = GOLDEN_PATH,
        k_values: Sequence[int] = (1, 3, 5),
        top_k: int = 20,
        min_score: float = 0.0,
    ) -> None:
        self.golden_path = Path(golden_path)
        self.k_values = tuple(k_values)

        if retriever is None:
            # Lazy import so metrics can be used without torch/faiss installed.
            from app.retrieval.retriever import Retriever
            retriever = Retriever(top_k=top_k, min_score=min_score)
        self.retriever = retriever

    def load_golden(self) -> list[dict[str, Any]]:
        data = json.loads(self.golden_path.read_text(encoding="utf-8"))
        return data["questions"]

    def ranked_page_ids(self, question: str) -> list[str]:
        """Retrieve, then collapse chunks to a ranked list of unique pages."""
        results = self.retriever.retrieve(question)
        return _dedup_keep_order([str(r.get("page_id", "")) for r in results])

    def ranked_pages_and_texts(self, question: str, k: int = 5):
        """Return (ranked unique page_ids, texts of the top-k retrieved chunks)."""
        results = self.retriever.retrieve(question)
        pages = _dedup_keep_order([str(r.get("page_id", "")) for r in results])
        texts = [str(r.get("text", "")) for r in results[:k]]
        return pages, texts

    def run(self, save: bool = True) -> dict[str, Any]:
        golden = self.load_golden()
        per_query: list[dict[str, Any]] = []

        t0 = time.perf_counter()
        for item in golden:
            relevant = {str(p) for p in item["relevant_page_ids"]}
            retrieved, texts = self.ranked_pages_and_texts(item["question"], k=5)
            metrics = evaluate_single(retrieved, relevant, self.k_values)
            metrics["context_sufficiency@5"] = context_sufficiency(
                texts, item.get("answer_keywords", [])
            )
            per_query.append(
                {
                    "id": item.get("id"),
                    "question": item["question"],
                    "relevant_page_ids": sorted(relevant),
                    "retrieved_top5": retrieved[:5],
                    "metrics": metrics,
                }
            )
        elapsed = time.perf_counter() - t0

        # Aggregate (mean over queries)
        metric_names = list(per_query[0]["metrics"].keys()) if per_query else []
        aggregate = {
            name: round(
                sum(q["metrics"][name] for q in per_query) / len(per_query), 4
            )
            for name in metric_names
        }

        report = {
            "num_questions": len(per_query),
            "k_values": list(self.k_values),
            "aggregate": aggregate,
            "per_query": per_query,
            "eval_time_s": round(elapsed, 3),
        }

        if save:
            RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
            RESULTS_PATH.write_text(
                json.dumps(report, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

        return report

    @staticmethod
    def print_report(report: dict[str, Any]) -> None:
        agg = report["aggregate"]
        print("=" * 60)
        print("RETRIEVAL EVALUATION")
        print("=" * 60)
        print(f"Questions evaluated : {report['num_questions']}")
        print(f"Eval time           : {report['eval_time_s']}s")
        print("-" * 60)
        # headline metrics first
        order = [
            "precision@3", "precision@5",
            "recall@3", "recall@5",
            "hit@1", "hit@3", "hit@5",
            "mrr", "ap", "context_sufficiency@5",
        ]
        for name in order:
            if name in agg:
                label = {"ap": "MAP", "context_sufficiency@5": "CTX_SUFF@5"}.get(name, name.upper())
                print(f"{label:14s}: {agg[name]:.4f}")
        print("=" * 60)
