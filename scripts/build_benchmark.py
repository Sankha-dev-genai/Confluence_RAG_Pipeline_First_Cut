"""Embed + index the synthetic benchmark corpus into its own vector store.

    python scripts/build_benchmark.py

Keeps the benchmark index separate from the real Confluence index. Needs the
embedding model (downloads once).
"""
from app.embeddings.embedding_generator import EmbeddingGenerator
from app.embeddings.vector_store import VectorStore

CHUNKS = "data/benchmark/chunks"
STORE = "data/benchmark/vectorstore"


def main() -> None:
    res = EmbeddingGenerator(chunk_dir=CHUNKS, output_dir=STORE).process_all()
    print(f"Embedded {res['chunks']} chunks (dim={res['embedding_dimension']}).")
    VectorStore(STORE).build_index()
    print(f"Benchmark index built at {STORE}")


if __name__ == "__main__":
    main()
