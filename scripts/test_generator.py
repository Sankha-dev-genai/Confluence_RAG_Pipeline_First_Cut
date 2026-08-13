from app.llm.generator import LLMGenerator
from app.retrieval.retriever import Retriever


def main():

    retriever = Retriever(
        top_k=5,
        min_score=0.25,
    )

    generator = LLMGenerator()

    while True:

        question = input("\nQuestion: ")

        if question.lower() == "exit":
            break

        chunks = retriever.retrieve(question)

        if not chunks:
            print("\nNo relevant documentation found.")
            continue

        result = generator.generate(
        question,
        chunks,
        )

        print()

        print("=" * 80)

        if result["success"]:

            print(result["answer"])

        else:

            print(result["error"])

        print("=" * 80)


if __name__ == "__main__":
    main()