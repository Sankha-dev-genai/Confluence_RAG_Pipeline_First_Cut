from __future__ import annotations

import json
import re
from pathlib import Path

from app.core.config import (
    CLEANED_DIR,
    CHUNKS_DIR,
    METADATA_DIR,
)

from app.processing.chunk_models import Chunk


class MarkdownChunker:
    """
    Intelligent Markdown chunker.

    Features
    --------
    ✓ Section-aware
    ✓ Table-aware
    ✓ Code-block-aware
    ✓ List-aware
    ✓ Heading hierarchy
    ✓ Metadata aware
    """

    TARGET_WORDS = 180

    MIN_WORDS = 30

    MAX_WORDS = 250

    OVERLAP_WORDS = 30

    def __init__(self):

        CHUNKS_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

    def load_document(
        self,
        page_id: str,
    ):

        markdown_path = (
            CLEANED_DIR /
            f"{page_id}.md"
        )

        metadata_path = (
            METADATA_DIR /
            f"{page_id}.json"
        )

        markdown = markdown_path.read_text(
            encoding="utf-8"
        )

        metadata = json.loads(
            metadata_path.read_text(
                encoding="utf-8"
            )
        )

        return markdown, metadata

    def find_headings(
        self,
        markdown: str,
    ):

        headings = []

        lines = markdown.splitlines()

        for line_number, line in enumerate(lines):

            match = re.match(
                r"^(#{1,6})\s+(.*)",
                line,
            )

            if not match:
                continue

            headings.append(
                {
                    "line": line_number,
                    "level": len(match.group(1)),
                    "title": match.group(2).strip(),
                }
            )

        return headings

    def detect_protected_blocks(
        self,
        markdown: str,
    ):

        blocks = []

        lines = markdown.splitlines()

        inside_code = False

        start = None

        for index, line in enumerate(lines):

            if line.startswith("```"):

                if not inside_code:

                    start = index
                    inside_code = True

                else:

                    blocks.append(
                        {
                            "type": "code",
                            "start": start,
                            "end": index,
                        }
                    )

                    inside_code = False

        return blocks

    def split_into_sections(
        self,
        markdown: str,
    ):
        """
        Split the markdown into logical sections based on headings.
        Each section retains its heading hierarchy.
        """

        lines = markdown.splitlines()

        headings = self.find_headings(markdown)

        if not headings:
            return [
                {
                    "heading_path": [],
                    "section": "Document",
                    "content": markdown,
                }
            ]

        sections = []

        heading_stack = []

        for index, heading in enumerate(headings):

            level = heading["level"]

            while len(heading_stack) >= level:
                heading_stack.pop()

            heading_stack.append(heading["title"])

            start = heading["line"]

            if index == len(headings) - 1:
                end = len(lines)
            else:
                end = headings[index + 1]["line"]

            content = "\n".join(lines[start:end]).strip()

            sections.append(
                {
                    "section": heading["title"],
                    "heading_path": heading_stack.copy(),
                    "content": content,
                }
            )

        return sections

    def is_table_line(
        self,
        line: str,
    ) -> bool:

        line = line.strip()

        return (
            line.startswith("|")
            and line.endswith("|")
        )

    def detect_tables(
        self,
        markdown: str,
    ):

        lines = markdown.splitlines()

        tables = []

        inside = False

        start = 0

        for i, line in enumerate(lines):

            if self.is_table_line(line):

                if not inside:

                    inside = True
                    start = i

            else:

                if inside:

                    tables.append(
                        {
                            "start": start,
                            "end": i - 1,
                        }
                    )

                    inside = False

        if inside:

            tables.append(
                {
                    "start": start,
                    "end": len(lines) - 1,
                }
            )

        return tables

    def detect_lists(
        self,
        markdown: str,
    ):

        lines = markdown.splitlines()

        lists = []

        inside = False

        start = 0

        pattern = re.compile(
            r"^\s*([-*+]|\d+\.)\s+"
        )

        for i, line in enumerate(lines):

            if pattern.match(line):

                if not inside:

                    inside = True
                    start = i

            else:

                if inside:

                    lists.append(
                        {
                            "start": start,
                            "end": i - 1,
                        }
                    )

                    inside = False

        if inside:

            lists.append(
                {
                    "start": start,
                    "end": len(lines) - 1,
                }
            )

        return lists

    @staticmethod
    def word_count(
        text: str,
    ) -> int:

        return len(
            re.findall(
                r"\b\w+\b",
                text,
            )
        )


    def parse_blocks(
        self,
        markdown: str,
    ) -> list[dict]:
        """
        Parse markdown into semantic blocks.

        Blocks can be:
            - heading
            - paragraph
            - table
            - code
            - list

        Each block keeps its heading hierarchy.
        """

        lines = markdown.splitlines()

        blocks = []

        heading_path = []

        i = 0

        while i < len(lines):

            line = lines[i]

            stripped = line.strip()

            # -----------------------------------
            # Heading
            # -----------------------------------

            heading = re.match(
                r"^(#{1,6})\s+(.*)",
                stripped,
            )

            if heading:

                level = len(heading.group(1))

                title = heading.group(2).strip()

                while len(heading_path) >= level:

                    heading_path.pop()

                heading_path.append(title)

                blocks.append(
                    {
                        "type": "heading",
                        "text": line,
                        "heading_path": heading_path.copy(),
                        "section": title,
                    }
                )

                i += 1
                continue

            # -----------------------------------
            # Code Block
            # -----------------------------------

            if stripped.startswith("```"):

                code = [line]

                i += 1

                while i < len(lines):

                    code.append(lines[i])

                    if lines[i].strip().startswith("```"):

                        i += 1
                        break

                    i += 1

                blocks.append(
                    {
                        "type": "code",
                        "text": "\n".join(code),
                        "heading_path": heading_path.copy(),
                        "section": heading_path[-1]
                        if heading_path else "",
                    }
                )

                continue

            # -----------------------------------
            # Table
            # -----------------------------------

            if self.is_table_line(line):

                table = []

                while (
                    i < len(lines)
                    and self.is_table_line(lines[i])
                ):

                    table.append(lines[i])

                    i += 1

                blocks.append(
                    {
                        "type": "table",
                        "text": "\n".join(table),
                        "heading_path": heading_path.copy(),
                        "section": heading_path[-1]
                        if heading_path else "",
                    }
                )

                continue

            # -----------------------------------
            # List
            # -----------------------------------

            if re.match(
                r"^\s*([-*+]|\d+\.)\s+",
                line,
            ):

                items = []

                while (
                    i < len(lines)
                    and re.match(
                        r"^\s*([-*+]|\d+\.)\s+",
                        lines[i],
                    )
                ):

                    items.append(lines[i])

                    i += 1

                blocks.append(
                    {
                        "type": "list",
                        "text": "\n".join(items),
                        "heading_path": heading_path.copy(),
                        "section": heading_path[-1]
                        if heading_path else "",
                    }
                )

                continue

            # -----------------------------------
            # Paragraph
            # -----------------------------------

            if stripped:

                paragraph = [line]

                i += 1

                while (
                    i < len(lines)
                    and lines[i].strip()
                    and not re.match(
                        r"^(#{1,6})\s+",
                        lines[i],
                    )
                    and not lines[i].startswith("```")
                    and not self.is_table_line(lines[i])
                    and not re.match(
                        r"^\s*([-*+]|\d+\.)\s+",
                        lines[i],
                    )
                ):

                    paragraph.append(lines[i])

                    i += 1

                blocks.append(
                    {
                        "type": "paragraph",
                        "text": "\n".join(paragraph),
                        "heading_path": heading_path.copy(),
                        "section": heading_path[-1]
                        if heading_path else "",
                    }
                )

                continue

            i += 1

        return blocks
    

    @staticmethod
    def _is_protected_block(block: dict) -> bool:
        """
        Protected blocks should not be mixed or broken casually.

        Tables, code blocks, and lists retain their Markdown structure.
        """
        return block.get("type") in {
            "table",
            "code",
            "list",
        }

    @staticmethod
    def _section_key(block: dict) -> tuple[str, ...]:
        """
        Return a stable semantic-section identifier.
        """
        heading_path = block.get("heading_path") or []

        return tuple(
            str(item).strip()
            for item in heading_path
            if str(item).strip()
        )

    def _split_paragraph_block(
        self,
        block: dict,
    ) -> list[dict]:
        """
        Split an oversized paragraph at sentence boundaries.
        """

        text = block.get("text", "").strip()

        if not text:
            return []

        if self.word_count(text) <= self.MAX_WORDS:
            return [block.copy()]

        sentences = re.split(
            r"(?<=[.!?])\s+",
            text,
        )

        pieces: list[dict] = []
        current_sentences: list[str] = []
        current_words = 0

        for sentence in sentences:

            sentence = sentence.strip()

            if not sentence:
                continue

            sentence_words = self.word_count(sentence)

            if (
                current_sentences
                and current_words + sentence_words
                > self.MAX_WORDS
            ):
                new_block = block.copy()
                new_block["text"] = " ".join(
                    current_sentences
                ).strip()

                pieces.append(new_block)

                current_sentences = []
                current_words = 0

            # A single unusually large sentence
            if sentence_words > self.MAX_WORDS:

                words = sentence.split()

                for start in range(
                    0,
                    len(words),
                    self.MAX_WORDS,
                ):
                    part = " ".join(
                        words[
                            start:
                            start + self.MAX_WORDS
                        ]
                    ).strip()

                    if part:
                        new_block = block.copy()
                        new_block["text"] = part
                        pieces.append(new_block)

                continue

            current_sentences.append(sentence)
            current_words += sentence_words

        if current_sentences:

            new_block = block.copy()
            new_block["text"] = " ".join(
                current_sentences
            ).strip()

            pieces.append(new_block)

        return pieces

    def _split_list_block(
        self,
        block: dict,
    ) -> list[dict]:
        """
        Split an oversized list between list items.
        """

        text = block.get("text", "").strip()

        if not text:
            return []

        if self.word_count(text) <= self.MAX_WORDS:
            return [block.copy()]

        lines = text.splitlines()

        groups: list[list[str]] = []
        current_lines: list[str] = []
        current_words = 0

        for line in lines:

            line_words = self.word_count(line)

            if (
                current_lines
                and current_words + line_words
                > self.MAX_WORDS
            ):
                groups.append(current_lines)

                current_lines = []
                current_words = 0

            current_lines.append(line)
            current_words += line_words

        if current_lines:
            groups.append(current_lines)

        output: list[dict] = []

        for group in groups:

            new_block = block.copy()
            new_block["text"] = "\n".join(
                group
            ).strip()

            if new_block["text"]:
                output.append(new_block)

        return output

    def _split_table_block(
        self,
        block: dict,
    ) -> list[dict]:
        """
        Split an oversized Markdown table while repeating
        the header and separator in every resulting block.
        """

        text = block.get("text", "").strip()

        if not text:
            return []

        if self.word_count(text) <= self.MAX_WORDS:
            return [block.copy()]

        lines = [
            line
            for line in text.splitlines()
            if line.strip()
        ]

        if len(lines) <= 2:
            return [block.copy()]

        header = lines[0]
        separator = lines[1]
        rows = lines[2:]

        output: list[dict] = []
        current_rows: list[str] = []

        base_words = (
            self.word_count(header)
            + self.word_count(separator)
        )

        current_words = base_words

        for row in rows:

            row_words = self.word_count(row)

            if (
                current_rows
                and current_words + row_words
                > self.MAX_WORDS
            ):
                new_block = block.copy()

                new_block["text"] = "\n".join(
                    [
                        header,
                        separator,
                        *current_rows,
                    ]
                ).strip()

                output.append(new_block)

                current_rows = []
                current_words = base_words

            current_rows.append(row)
            current_words += row_words

        if current_rows:

            new_block = block.copy()

            new_block["text"] = "\n".join(
                [
                    header,
                    separator,
                    *current_rows,
                ]
            ).strip()

            output.append(new_block)

        return output

    def _split_code_block(
        self,
        block: dict,
    ) -> list[dict]:
        """
        Keep ordinary code blocks intact.

        Only split exceptionally large code blocks, while
        preserving opening and closing Markdown fences.
        """

        text = block.get("text", "").strip()

        if not text:
            return []

        if self.word_count(text) <= self.MAX_WORDS:
            return [block.copy()]

        lines = text.splitlines()

        if len(lines) <= 2:
            return [block.copy()]

        opening_fence = lines[0]
        closing_fence = (
            lines[-1]
            if lines[-1].strip().startswith("```")
            else "```"
        )

        code_lines = lines[1:-1]

        output: list[dict] = []
        current_lines: list[str] = []
        current_words = 0

        for line in code_lines:

            line_words = max(
                self.word_count(line),
                1,
            )

            if (
                current_lines
                and current_words + line_words
                > self.MAX_WORDS
            ):
                new_block = block.copy()

                new_block["text"] = "\n".join(
                    [
                        opening_fence,
                        *current_lines,
                        closing_fence,
                    ]
                ).strip()

                output.append(new_block)

                current_lines = []
                current_words = 0

            current_lines.append(line)
            current_words += line_words

        if current_lines:

            new_block = block.copy()

            new_block["text"] = "\n".join(
                [
                    opening_fence,
                    *current_lines,
                    closing_fence,
                ]
            ).strip()

            output.append(new_block)

        return output

    def _split_oversized_block(
        self,
        block: dict,
    ) -> list[dict]:
        """
        Select the appropriate structure-aware splitter.
        """

        if (
            self.word_count(block.get("text", ""))
            <= self.MAX_WORDS
        ):
            return [block.copy()]

        block_type = block.get("type")

        if block_type == "table":
            return self._split_table_block(block)

        if block_type == "list":
            return self._split_list_block(block)

        if block_type == "code":
            return self._split_code_block(block)

        if block_type == "paragraph":
            return self._split_paragraph_block(block)

        return [block.copy()]

    def _expand_large_blocks(
        self,
        blocks: list[dict],
    ) -> list[dict]:
        """
        Ensure that unusually large semantic blocks are
        safely divided before final chunk assembly.
        """

        expanded: list[dict] = []

        for block in blocks:
            expanded.extend(
                self._split_oversized_block(block)
            )

        return expanded

    def _create_section_overlap(
        self,
        previous_blocks: list[dict],
        next_section_key: tuple[str, ...],
    ) -> list[dict]:
        """
        Carry overlap only within the same semantic section.

        This prevents content from Authentication being copied
        into Endpoints, Rate Limit, or another unrelated section.
        """

        overlap: list[dict] = []
        overlap_words = 0

        for block in reversed(previous_blocks):

            if block.get("type") == "heading":
                continue

            if (
                self._section_key(block)
                != next_section_key
            ):
                continue

            overlap.insert(
                0,
                block.copy(),
            )

            overlap_words += self.word_count(
                block.get("text", "")
            )

            if overlap_words >= self.OVERLAP_WORDS:
                break

        return overlap

    def build_document_chunks(
        self,
        markdown: str,
    ) -> list[dict]:
        """
        Build structure-aware semantic chunks.

        Rules
        -----
        1. A new heading starts a new semantic section.
        2. Different heading paths are not merged.
        3. Tables, lists, and code blocks remain structurally intact.
        4. Oversized blocks are split using type-specific logic.
        5. Overlap is added only inside the same section.
        6. Word-count limits remain a safety boundary, not the
           primary semantic boundary.
        """

        parsed_blocks = self.parse_blocks(
            markdown
        )

        blocks = self._expand_large_blocks(
            parsed_blocks
        )

        print("=" * 80)
        print("TOTAL BLOCKS:", len(blocks))

        for index, block in enumerate(
            blocks[:10]
        ):
            print(
                index,
                block.get("type"),
                self.word_count(
                    block.get("text", "")
                ),
            )

        chunks: list[dict] = []

        current_blocks: list[dict] = []
        current_words = 0
        current_section_key: tuple[str, ...] = ()

        for block in blocks:

            block_text = block.get(
                "text",
                "",
            ).strip()

            if not block_text:
                continue

            block_type = block.get(
                "type",
                "",
            )

            block_words = self.word_count(
                block_text
            )

            block_section_key = (
                self._section_key(block)
            )

            # -------------------------------------------------
            # Rule 1:
            # Every heading begins a new semantic section.
            # -------------------------------------------------
            if block_type == "heading":

                if current_blocks:

                    finalized = self._finalize_chunk(
                        current_blocks
                    )

                    if finalized.get("text", "").strip():
                        chunks.append(finalized)

                current_blocks = [
                    block.copy()
                ]

                current_words = block_words
                current_section_key = (
                    block_section_key
                )

                continue

            # -------------------------------------------------
            # Rule 2:
            # Do not combine content from different headings.
            # -------------------------------------------------
            section_changed = (
                bool(current_blocks)
                and bool(block_section_key)
                and bool(current_section_key)
                and block_section_key
                != current_section_key
            )

            if section_changed:

                finalized = self._finalize_chunk(
                    current_blocks
                )

                if finalized.get("text", "").strip():
                    chunks.append(finalized)

                current_blocks = []
                current_words = 0
                current_section_key = (
                    block_section_key
                )

            if (
                not current_section_key
                and block_section_key
            ):
                current_section_key = (
                    block_section_key
                )

            # -------------------------------------------------
            # Rule 3:
            # Protected structures begin a clean chunk when
            # they would otherwise mix unrelated content.
            # -------------------------------------------------
            protected_boundary = (
                self._is_protected_block(block)
                and current_blocks
                and current_words
                >= self.TARGET_WORDS
            )

            if protected_boundary:

                finalized = self._finalize_chunk(
                    current_blocks
                )

                if finalized.get("text", "").strip():
                    chunks.append(finalized)

                current_blocks = []
                current_words = 0

            # -------------------------------------------------
            # Rule 4:
            # Apply size limits inside the same semantic section.
            # -------------------------------------------------
            exceeds_limit = (
                current_blocks
                and current_words + block_words
                > self.MAX_WORDS
            )

            if exceeds_limit:

                previous_blocks = [
                    item.copy()
                    for item in current_blocks
                ]

                finalized = self._finalize_chunk(
                    current_blocks
                )

                if finalized.get("text", "").strip():
                    chunks.append(finalized)

                overlap = (
                    self._create_section_overlap(
                        previous_blocks,
                        block_section_key,
                    )
                )

                current_blocks = overlap

                current_words = sum(
                    self.word_count(
                        item.get("text", "")
                    )
                    for item in current_blocks
                )

            current_blocks.append(
                block.copy()
            )

            current_words += block_words

            if block_section_key:
                current_section_key = (
                    block_section_key
                )

        if current_blocks:

            finalized = self._finalize_chunk(
                current_blocks
            )

            if finalized.get("text", "").strip():
                chunks.append(finalized)

        # Remove accidental empty chunks.
        chunks = [
            chunk
            for chunk in chunks
            if chunk.get("text", "").strip()
        ]

        print("=" * 80)
        print("TOTAL CHUNKS:", len(chunks))

        for chunk in chunks:
            print(
                chunk.get(
                    "section",
                    "Document",
                ),
                "->",
                chunk.get(
                    "word_count",
                    0,
                ),
                "words",
            )

        return chunks

    def _finalize_chunk(
        self,
        blocks: list[dict],
    ) -> dict:
        """
        Convert collected semantic blocks into one chunk.
        """

        clean_blocks = [
            block
            for block in blocks
            if block.get("text", "").strip()
        ]

        text = "\n\n".join(
            block["text"].strip()
            for block in clean_blocks
        ).strip()

        heading_path: list[str] = []
        section = "Document"

        # Prefer the first heading block.
        for block in clean_blocks:

            if (
                block.get("type") == "heading"
                and block.get("heading_path")
            ):
                heading_path = list(
                    block["heading_path"]
                )

                section = (
                    block.get("section")
                    or heading_path[-1]
                )

                break

        # For document content before the first heading.
        if not heading_path:

            for block in clean_blocks:

                if block.get("heading_path"):

                    heading_path = list(
                        block["heading_path"]
                    )

                    section = (
                        block.get("section")
                        or heading_path[-1]
                    )

                    break

        return {
            "text": text,
            "heading_path": heading_path,
            "section": section,
            "word_count": self.word_count(text),
            "character_count": len(text),
            "block_types": [
                block.get(
                    "type",
                    "unknown",
                )
                for block in clean_blocks
            ],
        }

    def _create_overlap(
        self,
        previous_blocks: list[dict],
    ) -> list[dict]:
        """
        Create overlap using complete semantic blocks.
        """

        overlap = []

        words = 0

        for block in reversed(previous_blocks):

            overlap.insert(0, block)

            words += self.word_count(
                block["text"]
            )

            if words >= self.OVERLAP_WORDS:

                break

        return overlap
    
    def create_chunk_objects(
        self,
        page_id: str,
        metadata: dict,
        markdown: str,
    ) -> list[Chunk]:
        """
        Create Chunk objects from the entire document.
        """

        document_chunks = self.build_document_chunks(markdown)
        print("=" * 80)
        print("TOTAL CHUNKS:", len(document_chunks))

        for c in document_chunks:
            print(c["word_count"])

        chunk_objects = []

        total_chunks = len(document_chunks)

        for index, chunk_data in enumerate(document_chunks, start=1):

            chunk = Chunk(
                chunk_id=f"{page_id}_{index:03d}",
                page_id=page_id,
                title=metadata.get("title", ""),

                # NEW FIELDS
                parent_page=metadata.get("parent_page"),
                child_page=metadata.get("child_page"),
                breadcrumb=metadata.get("breadcrumb"),

                section=chunk_data["section"],
                heading_path=chunk_data["heading_path"],

                source_url=metadata.get("source_url", ""),

                owner=metadata.get("owner"),
                status=metadata.get("status"),
                audience=metadata.get("audience"),
                last_reviewed=metadata.get("last_reviewed"),

                text=chunk_data["text"],

                word_count=chunk_data["word_count"],
                character_count=chunk_data["character_count"],

                chunk_number=index,
                total_chunks=total_chunks,
            )

            chunk_objects.append(chunk)

        return chunk_objects

    def save_chunks(
        self,
        page_id: str,
        chunks: list[Chunk],
    ) -> None:
        """
        Save all chunks for one page.
        """

        output_file = (
            CHUNKS_DIR /
            f"{page_id}.json"
        )

        data = [
            chunk.to_dict()
            for chunk in chunks
        ]

        with open(
            output_file,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                data,
                f,
                indent=2,
                ensure_ascii=False,
            )

    def process_page(
        self,
        page_id: str,
    ) -> list[Chunk]:
        """
        Chunk one markdown page.
        """

        markdown, metadata = self.load_document(
            page_id
        )

        chunks = self.create_chunk_objects(
            page_id=page_id,
            metadata=metadata,
            markdown=markdown,
        )

        self.save_chunks(
            page_id,
            chunks,
        )

        print(
            f"{page_id}: "
            f"{len(chunks)} chunks created."
        )

        return chunks

    def process_all(
        self,
    ) -> None:
        """
        Process every markdown file.
        """

        markdown_files = sorted(
            CLEANED_DIR.glob("*.md")
        )

        total_pages = 0

        total_chunks = 0

        for markdown_file in markdown_files:

            page_id = markdown_file.stem

            try:

                chunks = self.process_page(
                    page_id
                )

                total_pages += 1

                total_chunks += len(chunks)

            except Exception as exc:

                print(
                    f"Failed: {page_id}"
                )

                print(exc)

        print()

        print("=" * 60)

        print(
            f"Pages Processed : {total_pages}"
        )

        print(
            f"Chunks Created  : {total_chunks}"
        )

        print("=" * 60)

    def merge_small_sections(
        self,
        sections: list,
    ) -> list:
        """
        Merge adjacent small sections to create better-sized chunks.
        """

        if not sections:
            return sections

        merged = []

        current = sections[0].copy()

        for section in sections[1:]:

            current_words = self.word_count(
                current["content"]
            )

            section_words = self.word_count(
                section["content"]
            )

            if (
                current_words < self.MIN_WORDS
                and current_words + section_words
                <= self.MAX_WORDS
            ):

                current["content"] += (
                    "\n\n"
                    + section["content"]
                )

                current["heading_path"] = (
                    section["heading_path"]
                )

                current["section"] = (
                    section["section"]
                )

            else:

                merged.append(current)

                current = section.copy()

        merged.append(current)

        return merged
    