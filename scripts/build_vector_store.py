from app.embeddings.vector_store import VectorStore


def main():

    store = VectorStore()

    store.build_index()


if __name__ == "__main__":

    main()