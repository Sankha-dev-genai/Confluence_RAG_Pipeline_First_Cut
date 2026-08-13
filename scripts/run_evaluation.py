"""
Run the retrieval evaluation over the golden question set.

    python scripts/run_evaluation.py

Writes data/evaluation/results.json and prints a summary table.
Requires the vector store to exist (run the ingestion/embedding pipeline first).
No OpenAI key needed — this measures retrieval quality.
"""
from app.evaluation.evaluator import RetrievalEvaluator


def main() -> None:
    evaluator = RetrievalEvaluator(top_k=20, min_score=0.0)
    report = evaluator.run(save=True)
    RetrievalEvaluator.print_report(report)
    print("\nSaved: data/evaluation/results.json")


if __name__ == "__main__":
    main()
