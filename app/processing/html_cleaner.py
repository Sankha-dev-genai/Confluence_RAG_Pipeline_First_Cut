from __future__ import annotations

import re
from pathlib import Path

from bs4 import BeautifulSoup
from markdownify import MarkdownConverter

from app.core.config import RAW_DIR, CLEANED_DIR


class ConfluenceMarkdownConverter(MarkdownConverter):
    """
    Custom markdown converter for Confluence export_view HTML.
    """

    def convert_img(self, el, text, parent_tags):
        alt = el.get("alt", "") or "Image"
        src = el.get("src", "")

        if src:
            return f"![{alt}]({src})"

        return f"![{alt}]"

    def convert_table(self, el, text, parent_tags):
        return "\n" + super().convert_table(el, text, parent_tags) + "\n"

    def convert_pre(self, el, text, parent_tags):
        code = el.get_text()
        return f"\n```\n{code.rstrip()}\n```\n"


class HTMLCleaner:

    def __init__(self):

        self.converter = ConfluenceMarkdownConverter(
            heading_style="ATX",
            bullets="-",
            escape_asterisks=False,
            escape_underscores=False,
        )

    def clean_html(self, html: str) -> str:

        soup = BeautifulSoup(html, "lxml")

        self._cleanup_html(soup)

        markdown = self.converter.convert(str(soup))

        markdown = self._post_process(markdown)

        return markdown

    def _cleanup_html(self, soup: BeautifulSoup):

        for tag in soup(
            [
                "script",
                "style",
                "meta",
                "noscript",
            ]
        ):
            tag.decompose()

    def _post_process(self, markdown: str):

        markdown = markdown.replace("\r", "")

        markdown = re.sub(
            r"\n{3,}",
            "\n\n",
            markdown,
        )

        markdown = re.sub(
            r"[ \t]+\n",
            "\n",
            markdown,
        )

        return markdown.strip()

    def clean_file(
        self,
        input_file: Path,
        output_file: Path,
    ):

        html = input_file.read_text(
            encoding="utf-8",
            errors="ignore",
        )

        markdown = self.clean_html(html)

        output_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_file.write_text(
            markdown,
            encoding="utf-8",
        )

    def clean_all(self):

        CLEANED_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        html_files = sorted(RAW_DIR.glob("*.html"))

        for html_file in html_files:

            output = CLEANED_DIR / f"{html_file.stem}.md"

            self.clean_file(
                html_file,
                output,
            )

        print(f"Cleaned {len(html_files)} files.")