import json
from pathlib import Path
from typing import Any

import numpy as np

from app.embeddings.embedding_model import EmbeddingModel


class EmbeddingGenerator:
    """
    Load chunk files, generate normalized embeddings,
    and save vectors with their chunk metadata.
    """

    def __init__(
        self,
        chunk_dir: str | Path = "data/chunks",
        output_dir: str | Path = "data/vectorstore",
        batch_size: int = 32,
    ) -> None:
        self.chunk_dir = Path(chunk_dir)
        self.output_dir = Path(output_dir)
        self.batch_size = batch_size

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.embeddings_path = (
            self.output_dir / "embeddings.npy"
        )

        self.metadata_path = (
            self.output_dir / "chunks_metadata.json"
        )

    def load_chunks(self) -> list[dict[str, Any]]:
        """
        Load and flatten all chunk JSON files.
        """

        if not self.chunk_dir.exists():
            raise FileNotFoundError(
                f"Chunk directory not found: {self.chunk_dir}"
            )

        chunk_files = sorted(
            self.chunk_dir.glob("*.json")
        )

        if not chunk_files:
            raise FileNotFoundError(
                f"No chunk JSON files found in {self.chunk_dir}"
            )

        all_chunks: list[dict[str, Any]] = []

        for file_path in chunk_files:

            try:
                with file_path.open(
                    "r",
                    encoding="utf-8",
                ) as file:
                    file_data = json.load(file)

            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON file: {file_path}"
                ) from exc

            if isinstance(file_data, list):
                chunks = file_data

            elif (
                isinstance(file_data, dict)
                and isinstance(
                    file_data.get("chunks"),
                    list,
                )
            ):
                chunks = file_data["chunks"]

            else:
                raise ValueError(
                    "Expected a JSON list of chunks or "
                    f"a dictionary containing 'chunks': {file_path}"
                )

            for chunk in chunks:

                if not isinstance(chunk, dict):
                    raise ValueError(
                        f"Invalid chunk in file: {file_path}"
                    )

                chunk_id = chunk.get("chunk_id")
                text = chunk.get("text", "")

                if not chunk_id:
                    raise ValueError(
                        f"Missing chunk_id in file: {file_path}"
                    )

                if not isinstance(text, str) or not text.strip():
                    raise ValueError(
                        f"Empty text for chunk: {chunk_id}"
                    )

                all_chunks.append(chunk)

        if not all_chunks:
            raise ValueError(
                "No valid chunks were loaded."
            )

        return all_chunks

    def generate_embeddings(
        self,
        chunks: list[dict[str, Any]],
    ) -> np.ndarray:
        """
        Generate one normalized embedding per chunk.
        """

        texts = []

        for chunk in chunks:

            embedding_text = f"""
        Parent Page:
        {chunk.get("parent_page", "")}

        Child Page:
        {chunk.get("child_page", "")}

        Title:
        {chunk.get("title", "")}

        Breadcrumb:
        {chunk.get("breadcrumb", "")}

        Section:
        {chunk.get("section", "")}

        Owner:
        {chunk.get("owner", "")}

        Status:
        {chunk.get("status", "")}

        Audience:
        {chunk.get("audience", "")}

        Content:
        {chunk.get("text", "").strip()}
        """.strip()

            texts.append(embedding_text)

        model = EmbeddingModel.get_model()

        embeddings = model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        embeddings = np.asarray(
            embeddings,
            dtype=np.float32,
        )

        if embeddings.ndim != 2:
            raise ValueError(
                "Embeddings must be a two-dimensional array."
            )

        if embeddings.shape[0] != len(chunks):
            raise ValueError(
                "Embedding count does not match chunk count."
            )

        return embeddings

    def save_outputs(
        self,
        chunks: list[dict[str, Any]],
        embeddings: np.ndarray,
    ) -> None:
        """
        Save the embedding matrix and chunk metadata.
        """

        np.save(
            self.embeddings_path,
            embeddings,
        )

        metadata_records = []

        for index, chunk in enumerate(chunks):
            record = chunk.copy()
            record["vector_id"] = index
            metadata_records.append(record)


            # metadata_records.append(
            #     {
            #         "vector_id": index,
            #         "chunk_id": chunk.get("chunk_id"),
            #         "page_id": chunk.get("page_id"),
            #         "title": chunk.get("title", ""),
            #         "section": chunk.get("section", ""),
            #         "heading_path": chunk.get(
            #             "heading_path",
            #             [],
            #         ),
            #         "source_url": chunk.get(
            #             "source_url",
            #             "",
            #         ),
            #         "owner": chunk.get("owner"),
            #         "status": chunk.get("status"),
            #         "audience": chunk.get("audience"),
            #         "last_reviewed": chunk.get(
            #             "last_reviewed"
            #         ),
            #         "text": chunk.get("text", ""),
            #         "word_count": chunk.get(
            #             "word_count",
            #             0,
            #         ),
            #         "character_count": chunk.get(
            #             "character_count",
            #             0,
            #         ),
            #         "chunk_number": chunk.get(
            #             "chunk_number"
            #         ),
            #         "total_chunks": chunk.get(
            #             "total_chunks"
                    # ),
                # }
            # )

        with self.metadata_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                metadata_records,
                file,
                ensure_ascii=False,
                indent=2,
            )

    def process_all(self) -> dict[str, Any]:
        """
        Run the complete embedding-generation process.
        """

        chunks = self.load_chunks()

        print(
            f"Chunks loaded       : {len(chunks)}"
        )

        embeddings = self.generate_embeddings(
            chunks
        )

        print(
            f"Embeddings generated: {embeddings.shape[0]}"
        )
        print(
            f"Embedding dimension : {embeddings.shape[1]}"
        )

        self.save_outputs(
            chunks=chunks,
            embeddings=embeddings,
        )

        return {
            "chunks": len(chunks),
            "embedding_dimension": int(
                embeddings.shape[1]
            ),
            "embeddings_path": str(
                self.embeddings_path
            ),
            "metadata_path": str(
                self.metadata_path
            ),
        }