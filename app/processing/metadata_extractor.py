from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.core.config import (
    CLEANED_DIR,
    CONFLUENCE_BASE_URL,
    METADATA_DIR,
)
from app.ingestion.page_fetcher import PageFetcher


class MetadataExtractor:
    """
    Extracts structured metadata for cleaned Confluence pages.

    Metadata sources:
    1. Confluence REST API
    2. The first Field/Value Markdown table
    3. Cleaned Markdown headings
    """

    STANDARD_FIELDS = {
        "owner": "owner",
        "status": "status",
        "last reviewed": "last_reviewed",
        "last_reviewed": "last_reviewed",
        "audience": "audience",
    }

    def __init__(self) -> None:
        self.fetcher = PageFetcher()
        METADATA_DIR.mkdir(parents=True, exist_ok=True)

    def extract_file(self, markdown_file: Path) -> dict[str, Any]:
        page_id = markdown_file.stem

        markdown = markdown_file.read_text(
            encoding="utf-8",
            errors="ignore",
        )

        page = self.fetcher.get_page(page_id)

        document_fields = self._extract_field_value_table(markdown)
        headings = self._extract_headings(markdown)
        api_metadata = self._extract_api_metadata(page)

        metadata: dict[str, Any] = {
            "page_id": page_id,
            "title": api_metadata["title"],
            "parent_page": (
                api_metadata["ancestor_titles"][-1]
                if api_metadata["ancestor_titles"]
                else None
            ),
            "child_page": api_metadata["title"],
            "breadcrumb": " > ".join(
            api_metadata["ancestor_titles"] + [api_metadata["title"]]
            ),
            "source_type": "confluence",
            "source_url": self._build_source_url(page_id),
            "space_key": api_metadata["space_key"],
            "space_name": api_metadata["space_name"],
            "version": api_metadata["version"],
            "created_at": api_metadata["created_at"],
            "updated_at": api_metadata["updated_at"],
            "created_by": api_metadata["created_by"],
            "parent_page_id": api_metadata["parent_page_id"],
            "ancestor_ids": api_metadata["ancestor_ids"],
            "ancestor_titles": api_metadata["ancestor_titles"],
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

        return metadata

    def extract_all(self) -> list[dict[str, Any]]:
        markdown_files = sorted(CLEANED_DIR.glob("*.md"))
        results: list[dict[str, Any]] = []

        for markdown_file in markdown_files:
            try:
                metadata = self.extract_file(markdown_file)
                self._save_metadata(metadata)
                results.append(metadata)

                print(
                    f"Metadata extracted: "
                    f"{metadata['title']} ({metadata['page_id']})"
                )

            except Exception as exc:
                print(
                    f"Metadata extraction failed for "
                    f"{markdown_file.name}: {exc}"
                )

        print(
            f"\nMetadata extraction complete. "
            f"{len(results)} of {len(markdown_files)} files processed."
        )

        return results

    def _extract_field_value_table(
        self,
        markdown: str,
    ) -> dict[str, Any]:
        """
        Extracts values from a Markdown table such as:

        | Field | Value |
        | --- | --- |
        | Owner | Docs Team |
        | Status | COMPLETE |
        """

        result: dict[str, Any] = {
            "custom_fields": {},
        }

        tables = self._find_markdown_tables(markdown)

        for table in tables:
            if len(table) < 2:
                continue

            header = [self._normalize_cell(cell) for cell in table[0]]

            if len(header) < 2:
                continue

            if header[0].lower() != "field":
                continue

            if header[1].lower() != "value":
                continue

            for row in table[1:]:
                if len(row) < 2:
                    continue

                field_name = self._normalize_cell(row[0])
                field_value = self._normalize_cell(row[1])

                if not field_name:
                    continue

                normalized_name = self._normalize_field_name(field_name)

                standard_key = self.STANDARD_FIELDS.get(normalized_name)

                if standard_key:
                    result[standard_key] = field_value or None
                else:
                    result["custom_fields"][normalized_name] = (
                        field_value or None
                    )

            break

        return result

    def _find_markdown_tables(
        self,
        markdown: str,
    ) -> list[list[list[str]]]:
        lines = markdown.splitlines()
        tables: list[list[list[str]]] = []

        index = 0

        while index < len(lines) - 1:
            current_line = lines[index].strip()
            next_line = lines[index + 1].strip()

            if (
                self._is_table_row(current_line)
                and self._is_separator_row(next_line)
            ):
                table_rows: list[list[str]] = [
                    self._split_table_row(current_line)
                ]

                index += 2

                while index < len(lines):
                    line = lines[index].strip()

                    if not self._is_table_row(line):
                        break

                    table_rows.append(
                        self._split_table_row(line)
                    )
                    index += 1

                tables.append(table_rows)
                continue

            index += 1

        return tables

    @staticmethod
    def _is_table_row(line: str) -> bool:
        return line.startswith("|") and line.endswith("|")

    @staticmethod
    def _is_separator_row(line: str) -> bool:
        if not line.startswith("|") or not line.endswith("|"):
            return False

        cells = [
            cell.strip()
            for cell in line.strip("|").split("|")
        ]

        if not cells:
            return False

        return all(
            bool(re.fullmatch(r":?-{3,}:?", cell))
            for cell in cells
        )

    @staticmethod
    def _split_table_row(line: str) -> list[str]:
        """
        Splits a Markdown table row while retaining escaped pipe characters.
        """

        content = line.strip().strip("|")
        cells = re.split(r"(?<!\\)\|", content)

        return [
            cell.replace(r"\|", "|").strip()
            for cell in cells
        ]

    def _extract_headings(self, markdown: str) -> list[dict[str, Any]]:
        headings: list[dict[str, Any]] = []

        inside_code_block = False

        for line in markdown.splitlines():
            stripped = line.strip()

            if stripped.startswith("```"):
                inside_code_block = not inside_code_block
                continue

            if inside_code_block:
                continue

            match = re.match(r"^(#{1,6})\s+(.+?)\s*$", stripped)

            if not match:
                continue

            headings.append(
                {
                    "level": len(match.group(1)),
                    "text": self._strip_markdown(match.group(2)),
                }
            )

        return headings

    def _extract_api_metadata(
        self,
        page: dict[str, Any],
    ) -> dict[str, Any]:
        space = page.get("space") or {}
        version = page.get("version") or {}
        history = page.get("history") or {}
        created_by = history.get("createdBy") or {}
        ancestors = page.get("ancestors") or []

        parent_page_id = None

        if ancestors:
            parent_page_id = str(
                ancestors[-1].get("id", "")
            ) or None

        return {
            "title": page.get("title") or "Untitled",
            "space_key": space.get("key"),
            "space_name": space.get("name"),
            "version": version.get("number"),
            "created_at": history.get("createdDate"),
            "updated_at": version.get("when"),
            "created_by": (
                created_by.get("displayName")
                or created_by.get("publicName")
                or created_by.get("accountId")
            ),
            "parent_page_id": parent_page_id,
            "ancestor_ids": [
                str(ancestor.get("id"))
                for ancestor in ancestors
                if ancestor.get("id")
            ],
            "ancestor_titles": [
                ancestor.get("title")
                for ancestor in ancestors
                if ancestor.get("title")
            ],
        }

    def _save_metadata(
        self,
        metadata: dict[str, Any],
    ) -> None:
        output_file = (
            METADATA_DIR / f"{metadata['page_id']}.json"
        )

        output_file.write_text(
            json.dumps(
                metadata,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    @staticmethod
    def _build_source_url(page_id: str) -> str:
        base_url = CONFLUENCE_BASE_URL.rstrip("/")

        return (
            f"{base_url}/wiki/pages/"
            f"viewpage.action?pageId={page_id}"
        )

    @staticmethod
    def _normalize_field_name(value: str) -> str:
        value = MetadataExtractor._strip_markdown(value)
        value = value.strip().lower()
        value = re.sub(r"[\s\-]+", "_", value)
        value = re.sub(r"[^a-z0-9_]", "", value)

        return value.strip("_")

    @staticmethod
    def _normalize_cell(value: str) -> str:
        value = value.replace("<br>", " ")
        value = value.replace("<br/>", " ")
        value = value.replace("<br />", " ")
        value = MetadataExtractor._strip_markdown(value)
        value = re.sub(r"\s+", " ", value)

        return value.strip()

    @staticmethod
    def _strip_markdown(value: str) -> str:
        value = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"\1", value)
        value = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", value)
        value = re.sub(r"(\*\*|__)(.*?)\1", r"\2", value)
        value = re.sub(r"(\*|_)(.*?)\1", r"\2", value)
        value = re.sub(r"`([^`]+)`", r"\1", value)
        value = value.replace(r"\|", "|")

        return value.strip()