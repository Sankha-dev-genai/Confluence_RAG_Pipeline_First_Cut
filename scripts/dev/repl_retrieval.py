import statistics
import time

from app.retrieval.retriever import Retriever


def print_separator(title: str = ""):
    print("\n" + "=" * 90)

    if title:
        print(title)
        print("=" * 90)


def print_chunk(rank: int, chunk: dict):

    print_separator(f"RANK {rank}")

    print(f"Similarity Score : {chunk['score']:.4f}")
    print(f"Final Score      : {chunk['score']:.4f}")
    print(
    f"Embedding Score  : "
    f"{chunk.get('embedding_score',0):.4f}"
    )
    print(f"Chunk ID         : {chunk.get('chunk_id')}")

    print(f"Parent Page      : {chunk.get('parent_page', 'N/A')}")
    print(f"Child Page       : {chunk.get('child_page', 'N/A')}")
    print(f"Breadcrumb       : {chunk.get('breadcrumb', 'N/A')}")

    print(f"Title            : {chunk.get('title', '')}")
    print(f"Section          : {chunk.get('section', '')}")

    print(
        f"Heading Path     : "
        f"{' > '.join(chunk.get('heading_path', []))}"
    )

    print(f"Owner            : {chunk.get('owner', '')}")
    print(f"Status           : {chunk.get('status', '')}")
    print(f"Audience         : {chunk.get('audience', '')}")
    print(f"Last Reviewed    : {chunk.get('last_reviewed', '')}")

    print(f"Word Count       : {chunk.get('word_count', 0)}")
    print(f"Characters       : {chunk.get('character_count', 0)}")

    print(f"Source URL       : {chunk.get('source_url', '')}")

    print()

    print("Score Breakdown")

    breakdown = chunk.get(
        "score_breakdown",
        {},
    )

    for key, value in breakdown.items():

        print(
            f"  {key:12s}: {value:.4f}"
        )
    print("\nPreview\n")

    preview = chunk.get("text", "")[:500]

    print(preview)

    print()


def main():

    print_separator("LOADING RETRIEVER")

    retriever = Retriever(
        top_k=10,
        min_score=0.05,
    )

    while True:

        print()

        question = input("Question (exit to quit): ")

        if question.lower() == "exit":
            break

        print_separator("QUERY")

        print(question)

        start = time.perf_counter()

        results = retriever.retrieve(question)

        retrieval_time = (
            time.perf_counter()
            - start
        )

        if not results:

            print_separator("NO RESULTS")

            print("No relevant chunks found.")

            continue

        scores = []

        parent_pages = set()

        child_pages = set()

        for rank, chunk in enumerate(
            results,
            start=1,
        ):

            scores.append(chunk["score"])

            if chunk.get("parent_page"):
                parent_pages.add(chunk["parent_page"])

            if chunk.get("child_page"):
                child_pages.add(chunk["child_page"])

            print_chunk(
                rank,
                chunk,
            )

        print_separator("SUMMARY")

        print(
            f"Highest Score       : {max(scores):.4f}"
        )

        print(
            f"Lowest Score        : {min(scores):.4f}"
        )

        print(
            f"Average Score       : {statistics.mean(scores):.4f}"
        )

        print(
            f"Median Score        : {statistics.median(scores):.4f}"
        )

        if len(scores) > 1:

            print(
                f"Std Deviation       : {statistics.stdev(scores):.4f}"
            )

        else:

            print(
                "Std Deviation       : N/A"
            )

        print()

        print(
            f"Unique Parent Pages : {len(parent_pages)}"
        )

        print(
            f"Unique Child Pages  : {len(child_pages)}"
        )

        print(
            f"Chunks Retrieved    : {len(results)}"
        )

        print(
            f"Retrieval Time      : {retrieval_time * 1000:.2f} ms"
        )


if __name__ == "__main__":
    main()