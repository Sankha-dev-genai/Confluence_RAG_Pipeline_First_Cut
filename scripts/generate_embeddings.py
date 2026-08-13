from app.embeddings.embedding_generator import (
    EmbeddingGenerator,
)


def main() -> None:

    generator = EmbeddingGenerator(
        batch_size=32,
    )

    result = generator.process_all()

    print()
    print("=" * 60)
    print("EMBEDDING GENERATION COMPLETE")
    print("=" * 60)
    print(
        f"Chunks              : {result['chunks']}"
    )
    print(
        "Embedding Dimension : "
        f"{result['embedding_dimension']}"
    )
    print(
        f"Embeddings File     : "
        f"{result['embeddings_path']}"
    )
    print(
        f"Metadata File       : "
        f"{result['metadata_path']}"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()