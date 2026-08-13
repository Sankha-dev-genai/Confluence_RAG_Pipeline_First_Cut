from pathlib import Path
import json

import faiss
import numpy as np

from app.embeddings.embedding_model import EmbeddingModel


class VectorStore:

    def __init__(
        self,
        vectorstore_dir: str = "data/vectorstore",
    ):

        self.vectorstore_dir = Path(vectorstore_dir)

        self.embeddings_path = (
            self.vectorstore_dir / "embeddings.npy"
        )

        self.metadata_path = (
            self.vectorstore_dir / "chunks_metadata.json"
        )

        self.index_path = (
            self.vectorstore_dir / "faiss.index"
        )

        self.index = None
        self.metadata = None

    def build_index(self):

        embeddings = np.load(
            self.embeddings_path
        ).astype(np.float32)

        dimension = embeddings.shape[1]

        index = faiss.IndexFlatIP(
            dimension
        )

        index.add(
            embeddings
        )

        self.index = index

        faiss.write_index(
            index,
            str(self.index_path),
        )

        with open(
            self.metadata_path,
            "r",
            encoding="utf-8",
        ) as f:

            self.metadata = json.load(f)

        print()

        print(
            f"Vectors Added : {index.ntotal}"
        )

        print(
            f"Dimension     : {dimension}"
        )

        print(
            f"Index Saved   : {self.index_path}"
        )

    def load_index(self):

        self.index = faiss.read_index(
            str(self.index_path)
        )

        with open(
            self.metadata_path,
            "r",
            encoding="utf-8",
        ) as f:

            self.metadata = json.load(f)

    def search(
        self,
        query: str,
        top_k: int = 5,
    ):

        model = EmbeddingModel.get_model()

        query_embedding = model.encode(
            query,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)

        scores, indices = self.index.search(
            query_embedding.reshape(
                1,
                -1,
            ),
            top_k,
        )

        results = []

        for score, idx in zip(
            scores[0],
            indices[0],
        ):

            if idx == -1:
                continue

            record = self.metadata[idx].copy()

            record["score"] = float(score)

            results.append(record)

        return results