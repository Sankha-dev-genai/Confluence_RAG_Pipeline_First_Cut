"""Dense vs Hybrid comparison on the synthetic benchmark corpus.

    python scripts/run_benchmark.py [--alpha 0.5]
"""
import argparse

from app.evaluation.comparison import run_comparison, print_comparison

STORE = "data/benchmark/vectorstore"
GOLDEN = "data/evaluation/golden_benchmark.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=0.5,
                    help="dense weight; (1-alpha) is BM25 weight")
    args = ap.parse_args()

    from app.retrieval.retriever import Retriever
    dense = Retriever(top_k=20, min_score=0.0, use_hybrid=False, vectorstore_dir=STORE)
    hybrid = Retriever(top_k=20, min_score=0.0, use_hybrid=True,
                       hybrid_alpha=args.alpha, vectorstore_dir=STORE)

    result = run_comparison(dense_retriever=dense, hybrid_retriever=hybrid,
                            golden_path=GOLDEN, hybrid_alpha=args.alpha, save=True)
    print_comparison(result)
    print("\nSaved: data/evaluation/comparison.json")


if __name__ == "__main__":
    main()
