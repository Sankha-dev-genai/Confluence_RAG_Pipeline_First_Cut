from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import re
from app.core.constants import STOP_WORDS


@dataclass(slots=True)
class ScoreBreakdown:
    """
    Explains how the final retrieval score was calculated.
    """

    embedding: float = 0.0
    title: float = 0.0
    section: float = 0.0
    heading: float = 0.0
    breadcrumb: float = 0.0
    keyword: float = 0.0
    exact_phrase: float = 0.0
    child_page: float = 0.0
    owner: float = 0.0

    @property
    def total(self) -> float:
        return (
            self.embedding
            + self.title
            + self.section
            + self.heading
            + self.breadcrumb
            + self.keyword
            + self.exact_phrase
            + self.child_page
            + self.owner
        )


class RetrievalReranker:
    """
    Metadata-aware reranker.

    Uses document metadata to improve FAISS ranking.

    This module NEVER replaces embedding similarity.
    It only boosts already relevant chunks.
    """

    TITLE_BOOST = 0.15
    SECTION_BOOST = 0.20
    HEADING_BOOST = 0.25
    BREADCRUMB_BOOST = 0.05
    KEYWORD_BOOST = 0.20
    EXACT_PHRASE_BOOST = 0.35
    CHILD_PAGE_BOOST = 0.20
    OWNER_BOOST = 0.15

    def exact_phrase_score(
        self,
        query: str,
        text: str,
    ) -> float:

        if not query or not text:
            return 0.0

        query = query.lower().strip()
        text = text.lower()

        if query in text:
            return 1.0

        return 0.0


    def child_page_score(
        self,
        query: str,
        child_page: str,
    ) -> float:

        return self.exact_phrase_score(
            query,
            child_page,
        )


    def owner_score(
        self,
        query_tokens: set[str],
        owner: str,
    ) -> float:

        return self.match_score(
            query_tokens,
            owner,
        )

    def rerank(
        self,
        query: str,
        chunks: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:

        query_tokens = self._tokenize(query)

        reranked = []

        for chunk in chunks:

            breakdown = self.score_chunk(
                query,
                query_tokens,
                chunk,
            )

            record = chunk.copy()

            record["embedding_score"] = chunk.get(
                "score",
                0.0,
            )

            record["score_breakdown"] = {
                "embedding": breakdown.embedding,
                "title": breakdown.title,
                "section": breakdown.section,
                "heading": breakdown.heading,
                "breadcrumb": breakdown.breadcrumb,
                "keyword": breakdown.keyword,
                "exact_phrase": breakdown.exact_phrase,
                "child_page": breakdown.child_page,
                "owner": breakdown.owner,
            }

            record["score"] = breakdown.total

            reranked.append(record)

        reranked.sort(
            key=lambda x: x["score"],
            reverse=True,
        )

        return reranked

    def score_chunk(
        self,
        query: str,
        query_tokens: set[str],
        chunk: dict[str, Any],
    ) -> ScoreBreakdown:

        breakdown = ScoreBreakdown()

        breakdown.embedding = float(
            chunk.get("score", 0.0)
        )

        title = chunk.get(
            "title",
            "",
        )

        section = chunk.get(
            "section",
            "",
        )

        breadcrumb = chunk.get(
            "breadcrumb",
            "",
        )

        heading = " ".join(
            chunk.get(
                "heading_path",
                [],
            )
        )

        breakdown.title = (
            self.match_score(
                query_tokens,
                title,
            )
            * self.TITLE_BOOST
        )

        breakdown.section = (
            self.match_score(
                query_tokens,
                section,
            )
            * self.SECTION_BOOST
        )

        breakdown.heading = (
            self.match_score(
                query_tokens,
                heading,
            )
            * self.HEADING_BOOST
        )

        breakdown.breadcrumb = (
            self.match_score(
                query_tokens,
                breadcrumb,
            )
            * self.BREADCRUMB_BOOST
        )

        breakdown.keyword = (
            self.keyword_overlap(
                query_tokens,
                chunk.get(
                    "text",
                    "",
                ),
            )
            * self.KEYWORD_BOOST
        )

        breakdown.exact_phrase = (
            self.exact_phrase_score(
                query,
                chunk.get("text", "")
            )
            * self.EXACT_PHRASE_BOOST
        )

        breakdown.child_page = (
            self.child_page_score(
                query,
                chunk.get("child_page", "")
            )
            * self.CHILD_PAGE_BOOST
        )

        breakdown.owner = (
            self.owner_score(
                query_tokens,
                chunk.get("owner", ""),
            )
            * self.OWNER_BOOST
        )

        return breakdown

    @staticmethod
    def _tokenize(
        text: str,
    ) -> set[str]:
        """
        Normalize text for retrieval.

        - lowercase
        - remove punctuation
        - remove stop words
        """

        tokens = {
            token
            for token in re.findall(
                r"[a-zA-Z0-9]+",
                text.lower(),
            )
        }

        return {
            token
            for token in tokens
            if token not in STOP_WORDS
        }

    def match_score(
        self,
        query_tokens: set[str],
        text: str,
    ) -> float:

        if not text:
            return 0.0

        text_tokens = self._tokenize(text)

        if not text_tokens:
            return 0.0

        overlap = len(
            query_tokens & text_tokens
        )

        return overlap / len(query_tokens)

    def keyword_overlap(
        self,
        query_tokens: set[str],
        text: str,
    ) -> float:

        if not text:
            return 0.0

        text_lower = text.lower()

        matches = sum(
            token in text_lower
            for token in query_tokens
        )

        return matches / max(
            len(query_tokens),
            1,
        )