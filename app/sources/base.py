"""
Source-agnostic contracts. Everything downstream (clean -> chunk -> embed ->
retrieve -> answer) operates on the RAW HTML + a normalized metadata sidecar,
so adding a new source only means implementing KnowledgeSource.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from app.ingestion.models import ConfluencePage  # reused as the generic tree node


@dataclass
class RawDocument:
    page_id: str
    title: str
    html: str
    url: str
    source_type: str
    breadcrumb: str | None = None
    ancestor_titles: list[str] = field(default_factory=list)
    ancestor_ids: list[str] = field(default_factory=list)
    parent_page_id: str | None = None
    parent_page: str | None = None
    space_key: str | None = None
    space_name: str | None = None
    version: Any = None
    updated_at: str | None = None
    created_at: str | None = None
    created_by: str | None = None
    extra: dict = field(default_factory=dict)

    def sidecar(self) -> dict:
        """Normalized metadata written next to raw HTML for the processing stage."""
        return {
            "title": self.title,
            "url": self.url,
            "source_type": self.source_type,
            "breadcrumb": self.breadcrumb,
            "ancestor_titles": self.ancestor_titles,
            "ancestor_ids": self.ancestor_ids,
            "parent_page_id": self.parent_page_id,
            "parent_page": self.parent_page,
            "space_key": self.space_key,
            "space_name": self.space_name,
            "version": self.version,
            "updated_at": self.updated_at,
            "created_at": self.created_at,
            "created_by": self.created_by,
        }


class KnowledgeSource(ABC):
    """Interface every knowledge source implements."""

    name: str = "source"

    @abstractmethod
    def test_connection(self) -> dict:
        """Return a small dict describing a successful connection."""

    @abstractmethod
    def build_tree(self) -> ConfluencePage:
        """Return the root node of the page hierarchy (children populated)."""

    @abstractmethod
    def download_all(self, root: ConfluencePage) -> None:
        """Write raw HTML (and a .meta.json sidecar) for every page under root."""

    @abstractmethod
    def metadata_extractor(self):
        """Return the metadata extractor instance appropriate for this source."""
