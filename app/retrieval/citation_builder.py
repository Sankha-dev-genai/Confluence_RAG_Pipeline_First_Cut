from __future__ import annotations

import re
from typing import Any


class CitationBuilder:
    """
    Turn retrieved chunks into clean, presentable citations.

    A citation is the exact evidence a chunk provides for an answer:
    its page, section, breadcrumb, similarity score, a short snippet,
    and a link back to the source Confluence page.
    """

    SNIPPET_CHARS = 240

    @staticmethod
    def _snippet(text: str, limit: int = SNIPPET_CHARS) -> str:
        """Collapse whitespace and trim to a short preview."""
        if not text:
            return ""
        clean = re.sub(r"\s+", " ", text).strip()
        if len(clean) <= limit:
            return clean
        return clean[:limit].rsplit(" ", 1)[0] + " ..."

    @classmethod
    def build(
        cls,
        chunks: list[dict[str, Any]],
        max_citations: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Build an ordered list of citation records from retrieved chunks.

        Each citation exposes BOTH scores so nothing is hidden:
        - similarity   : raw cosine similarity from FAISS (0..1)
        - rerank_score : final score after metadata boosting
        """
        if max_citations is not None:
            chunks = chunks[:max_citations]

        citations: list[dict[str, Any]] = []

        for i, chunk in enumerate(chunks, start=1):
            similarity = float(
                chunk.get("embedding_score", chunk.get("score", 0.0))
            )
            rerank_score = float(chunk.get("score", similarity))

            citations.append(
                {
                    "index": i,
                    "chunk_id": chunk.get("chunk_id", ""),
                    "page_id": chunk.get("page_id", ""),
                    "title": chunk.get("title", ""),
                    "child_page": chunk.get("child_page", chunk.get("title", "")),
                    "parent_page": chunk.get("parent_page", ""),
                    "section": chunk.get("section", ""),
                    "breadcrumb": chunk.get("breadcrumb", ""),
                    "heading_path": chunk.get("heading_path", []),
                    "owner": chunk.get("owner", ""),
                    "status": chunk.get("status", ""),
                    "last_reviewed": chunk.get("last_reviewed", ""),
                    "similarity": round(similarity, 4),
                    "rerank_score": round(rerank_score, 4),
                    "bm25_score": round(float(chunk.get("bm25_score", 0.0)), 4),
                    "hybrid_score": round(float(chunk.get("hybrid_score", 0.0)), 4),
                    "score_breakdown": chunk.get("score_breakdown", {}),
                    "source_url": chunk.get("source_url", ""),
                    "snippet": cls._snippet(chunk.get("text", "")),
                }
            )

        return citations

    @staticmethod
    def format_inline(citations: list[dict[str, Any]]) -> str:
        """A compact one-line reference list, e.g. for logs or CLI."""
        parts = [
            f"[{c['index']}] {c['title']} § {c.get('section') or '—'} "
            f"(sim={c['similarity']:.2f})"
            for c in citations
        ]
        return "\n".join(parts)
