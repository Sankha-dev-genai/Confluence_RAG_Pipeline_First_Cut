from app.embeddings.embedding_model import (
    EmbeddingModel,
)


def main():

    model = EmbeddingModel.get_model()

    vector = model.encode(
        "Hello World",
        normalize_embeddings=True,
    )

    print()

    print("Embedding Size :", len(vector))

    print("Embedding Type :", type(vector))

    print()

    print(vector[:10])


if __name__ == "__main__":

    main()