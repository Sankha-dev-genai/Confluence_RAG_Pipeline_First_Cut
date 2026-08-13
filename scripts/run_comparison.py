"""Compare dense-only vs hybrid retrieval on the golden set.

    python scripts/run_comparison.py
"""
from app.evaluation.comparison import run_comparison, print_comparison


def main() -> None:
    result = run_comparison(save=True)
    print_comparison(result)
    print("\nSaved: data/evaluation/comparison.json")


if __name__ == "__main__":
    main()
