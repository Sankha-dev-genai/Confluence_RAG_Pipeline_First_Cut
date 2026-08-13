"""
Generic metadata extractor for non-Confluence sources.

Subclasses MetadataExtractor to reuse its markdown parsers (Page-Properties
table, headings) but sources page-level metadata from the .meta.json sidecar
written by the KnowledgeSource — so it never calls the Confluence API.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.core.config import METADATA_DIR, RAW_DIR
from app.processing.metadata_extractor import MetadataExtractor


class WikiMetadataExtractor(MetadataExtractor):
    def __init__(self) -> None:
        # Deliberately do NOT call super().__init__ (it creates a Confluence fetcher).
        self.fetcher = None
        METADATA_DIR.mkdir(parents=True, exist_ok=True)

    def _load_sidecar(self, page_id: str) -> dict:
        path = RAW_DIR / f"{page_id}.meta.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        return {}

    def extract_file(self, markdown_file: Path) -> dict[str, Any]:
        page_id = markdown_file.stem
        markdown = markdown_file.read_text(encoding="utf-8", errors="ignore")

        side = self._load_sidecar(page_id)
        document_fields = self._extract_field_value_table(markdown)  # reused
        headings = self._extract_headings(markdown)                  # reused

        title = side.get("title", page_id)
        ancestor_titles = side.get("ancestor_titles", [])
        breadcrumb = side.get("breadcrumb") or " > ".join(ancestor_titles + [title])

        return {
            "page_id": page_id,
            "title": title,
            "parent_page": side.get("parent_page")
            or (ancestor_titles[-1] if ancestor_titles else None),
            "child_page": title,
            "breadcrumb": breadcrumb,
            "source_type": side.get("source_type", "wiki"),
            "source_url": side.get("url", ""),
            "space_key": side.get("space_key"),
            "space_name": side.get("space_name"),
            "version": side.get("version"),
            "created_at": side.get("created_at"),
            "updated_at": side.get("updated_at"),
            "created_by": side.get("created_by"),
            "parent_page_id": side.get("parent_page_id"),
            "ancestor_ids": side.get("ancestor_ids", []),
            "ancestor_titles": ancestor_titles,
            "headings": headings,
            "owner": document_fields.get("owner"),
            "status": document_fields.get("status"),
            "last_reviewed": document_fields.get("last_reviewed"),
            "audience": document_fields.get("audience"),
            "custom_fields": document_fields.get("custom_fields", {}),
            "markdown_file": markdown_file.name,
            "character_count": len(markdown),
            "word_count": len(markdown.split()),
            "extracted_at": datetime.now(timezone.utc).isoformat(),
        }
