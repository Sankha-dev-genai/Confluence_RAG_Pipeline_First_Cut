from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any


@dataclass
class Chunk:
    """
    Represents one semantic chunk of a document.
    """

    # ---------- REQUIRED ----------
    chunk_id: str

    page_id: str

    title: str

    section: str

    heading_path: list[str]

    source_url: str

    owner: str | None

    status: str | None

    audience: str | None

    last_reviewed: str | None

    text: str

    word_count: int

    character_count: int

    chunk_number: int

    # ---------- OPTIONAL ----------
    total_chunks: int = 0

    parent_page: str | None = None

    child_page: str | None = None

    breadcrumb: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)