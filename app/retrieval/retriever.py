from __future__ import annotations

from typing import Any

from app.core.config import settings
from app.core.logger import get_logger
from app.embeddings.vector_store import VectorStore
from app.retrieval.reranker import RetrievalReranker

log = get_logger("retriever")


class Retriever:
    """
    Retrieval pipeline

        Query
          -> Candidate generation
               dense (FAISS cosine)  [+ BM25 sparse, fused]   <- hybrid
          -> Metadata re-ranking      (existing RetrievalReranker)
          -> Score filtering
          -> Duplicate removal
          -> Final ranking

    Hybrid adds BM25 candidates fused with dense ones; it does NOT replace the
    metadata reranker — the fused relevance becomes the base the reranker boosts.
    True cosine similarity is preserved on each result for citations/confidence.
    """

    def __init__(
        self,
        top_k: int = 20,
        min_score: float = 0.20,
        use_hybrid: bool | None = None,
        hybrid_alpha: float | None = None,
        vectorstore_dir: str | None = None,
    ) -> None:
        self.top_k = top_k
        self.min_score = min_score
        self.use_hybrid = settings.use_hybrid if use_hybrid is None else use_hybrid
        self.hybrid_alpha = settings.hybrid_alpha if hybrid_alpha is None else hybrid_alpha

        self.store = VectorStore(vectorstore_dir) if vectorstore_dir else VectorStore()
        self.store.load_index()
        self._vectorstore_dir = vectorstore_dir
        self.reranker = RetrievalReranker()

        self._hybrid = None
        if self.use_hybrid:
            try:
                from pathlib import Path
                from app.retrieval.bm25_index import BM25Index
                from app.retrieval.hybrid import HybridSearcher
                meta = (Path(vectorstore_dir) / "chunks_metadata.json"
                        if vectorstore_dir else None)
                self._hybrid = HybridSearcher(
                    self.store, BM25Index(meta), alpha=self.hybrid_alpha
                )
                log.info(f"Hybrid retrieval enabled (alpha={self.hybrid_alpha}).")
            except Exception as exc:  # noqa: BLE001 - fall back to dense-only
                log.info(f"Hybrid unavailable ({exc}); using dense-only.")
                self.use_hybrid = False

    def _candidates(self, query: str) -> list[dict[str, Any]]:
        if self.use_hybrid and self._hybrid is not None:
            return self._hybrid.search(query=query, top_k=self.top_k)
        return self.store.search(query=query, top_k=self.top_k)

    def retrieve(self, query: str) -> list[dict[str, Any]]:
        raw_results = self._candidates(query)
        reranked = self.reranker.rerank(query=query, chunks=raw_results)

        filtered: list[dict[str, Any]] = []
        seen: set[str] = set()

        for result in reranked:
            chunk_id = result.get("chunk_id")
            if not chunk_id or chunk_id in seen:
                continue
            seen.add(chunk_id)
            if result["score"] < self.min_score:
                continue

            # Preserve TRUE cosine similarity for citations/confidence,
            # even though the reranker's base score may be the fused value.
            if "cosine_score" in result:
                result["embedding_score"] = result["cosine_score"]
                if isinstance(result.get("score_breakdown"), dict):
                    result["score_breakdown"]["embedding"] = result["cosine_score"]

            filtered.append(result)

        filtered.sort(key=lambda x: x["score"], reverse=True)
        return filtered

    def retrieve_one(self, query: str) -> dict[str, Any] | None:
        results = self.retrieve(query)
        return results[0] if results else None
