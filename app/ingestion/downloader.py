from pathlib import Path

from app.core.config import RAW_DIR
from app.ingestion.models import ConfluencePage
from app.ingestion.page_fetcher import PageFetcher


class PageDownloader:

    def __init__(self):
        self.fetcher = PageFetcher()

    def download_page(self, node: ConfluencePage):

        # Fetch the full page JSON
        page = self.fetcher.get_page(node.page_id)

        # Prefer export_view over storage
        body = page.get("body", {})

        export_html = (
            body.get("export_view", {})
                .get("value")
        )

        storage_html = (
            body.get("storage", {})
                .get("value")
        )

        html = export_html or storage_html or ""

        output_file = RAW_DIR / f"{node.page_id}.html"

        output_file.write_text(
            html,
            encoding="utf-8"
        )

        print(f"Downloaded {node.title}")

    def download_tree(self, node: ConfluencePage):

        self.download_page(node)

        for child in node.children:
            self.download_tree(child)