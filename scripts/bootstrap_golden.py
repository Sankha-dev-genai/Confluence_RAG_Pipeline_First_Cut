"""
Bootstrap a golden evaluation set for a collection.

    # deterministic candidates from titles/sections (no LLM):
    python scripts/bootstrap_golden.py --collection default --method auto

    # LLM-generated candidates (needs OPENAI_API_KEY):
    python scripts/bootstrap_golden.py --collection default --method llm

By default candidates are written for review; pass --approve to merge them
straight into the collection's golden_qa.json.
"""
import argparse

from app.evaluation.golden_bootstrap import (
    auto_bootstrap, llm_bootstrap, save_candidates, approve_into_golden,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection", default="default")
    ap.add_argument("--method", choices=["auto", "llm"], default="auto")
    ap.add_argument("--per-page", type=int, default=2)
    ap.add_argument("--approve", action="store_true",
                    help="merge directly into golden_qa.json instead of candidates")
    args = ap.parse_args()

    gen = auto_bootstrap if args.method == "auto" else llm_bootstrap
    candidates = gen(args.collection, per_page=args.per_page)
    print(f"Generated {len(candidates)} candidate questions "
          f"({args.method}) for collection '{args.collection}'.")

    if args.approve:
        n = approve_into_golden(args.collection, candidates)
        print(f"Merged {n} into golden_qa.json.")
    else:
        save_candidates(args.collection, candidates)
        print("Saved to golden_candidates.json for review "
              "(approve in the app's Evaluation tab, or re-run with --approve).")


if __name__ == "__main__":
    main()
