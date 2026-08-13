from __future__ import annotations

from app.core.config import settings
from app.core.logger import get_logger
from app.ingestion.client import ConfluenceClient
from app.ingestion.hierarchy import HierarchyBuilder
from app.ingestion.downloader import PageDownloader
from app.ingestion.models import ConfluencePage
from app.processing.metadata_extractor import MetadataExtractor
from app.sources.base import KnowledgeSource

log = get_logger("source.confluence")


class ConfluenceSource(KnowledgeSource):
    name = "confluence"

    def test_connection(self) -> dict:
        settings.require_confluence()
        data = ConfluenceClient().get("/wiki/rest/api/space")
        return {"spaces_visible": len(data.get("results", []))}

    def build_tree(self) -> ConfluencePage:
        settings.require_confluence()
        return HierarchyBuilder().build_tree(settings.confluence_parent_page_id)

    def download_all(self, root: ConfluencePage) -> None:
        PageDownloader().download_tree(root)

    def metadata_extractor(self):
        # The existing extractor already handles Confluence perfectly.
        return MetadataExtractor()
