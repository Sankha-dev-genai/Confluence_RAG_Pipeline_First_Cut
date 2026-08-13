from sentence_transformers import SentenceTransformer


class EmbeddingModel:

    _model = None

    @classmethod
    def get_model(cls):

        if cls._model is None:

            print(
                "Loading embedding model..."
            )

            cls._model = SentenceTransformer(
                "all-MiniLM-L6-v2"
            )

            print(
                "Embedding model loaded."
            )

        return cls._model