from app.retrieval.retriever import Retriever


def main():

    retriever = Retriever(
        top_k=5,
        min_score=0.25,
    )

    while True:

        print()

        query = input("Question : ")

        if query.lower() == "exit":
            break

        results = retriever.retrieve(query)

        print()

        print("=" * 80)

        print(
            f"Retrieved {len(results)} chunks"
        )

        print("=" * 80)

        for index, result in enumerate(
            results,
            start=1,
        ):

            print()

            print(f"Rank : {index}")

            print(
                f"Score : {result['score']:.4f}"
            )

            print(
                f"Chunk : {result['chunk_id']}"
            )

            print(
                f"Title : {result['title']}"
            )

            print(
                f"Section : {result['section']}"
            )

            print()

            print(result["text"][:350])

            print()

            print("-" * 80)


if __name__ == "__main__":
    main()