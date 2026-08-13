"""
MediaWiki adapter — proves the platform is source-agnostic.

Uses the standard MediaWiki Action API (api.php). Works against any wiki
(Wikipedia, Wikimedia, a company MediaWiki, Fandom, etc.). Pages are flat, so
the "tree" is a single root with each page as a child.

Required env:  WIKI_API_URL   (e.g. https://en.wikipedia.org/w/api.php)
Optional:      WIKI_BASE_URL, WIKI_NAMESPACE, WIKI_CATEGORY, WIKI_PAGE_LIMIT
"""
from __future__ import annotations

import json

from app.core.config import RAW_DIR, settings
from app.core.http import HttpClient
from app.core.logger import get_logger
from app.ingestion.models import ConfluencePage
from app.processing.wiki_metadata import WikiMetadataExtractor
from app.sources.base import KnowledgeSource, RawDocument

log = get_logger("source.mediawiki")


class MediaWikiSource(KnowledgeSource):
    name = "mediawiki"

    def __init__(self) -> None:
        self.api = settings.wiki_api_url
        self.base = (settings.wiki_base_url or "").rstrip("/")
        self.http = HttpClient(
            retries=settings.http_retries,
            timeout=settings.request_timeout,
            headers={"Accept": "application/json",
                     "User-Agent": "confluence-rag/1.0 (source=mediawiki)"},
        )

    # ---- helpers ----
    def _query(self, params: dict) -> dict:
        params = {"format": "json", "formatversion": "2", **params}
        return self.http.get(self.api, params=params)

    def _page_url(self, title: str, pageid: str) -> str:
        if self.base:
            return f"{self.base}/{title.replace(' ', '_')}"
        return f"{self.api}?curid={pageid}"

    # ---- interface ----
    def test_connection(self) -> dict:
        settings.require_wiki()
        data = self._query({"action": "query", "meta": "siteinfo",
                            "siprop": "general"})
        general = data.get("query", {}).get("general", {})
        return {"sitename": general.get("sitename", "unknown"),
                "generator": general.get("generator", "")}

    def _list_pages(self) -> list[dict]:
        settings.require_wiki()
        pages: list[dict] = []
        if settings.wiki_category:
            params = {"action": "query", "list": "categorymembers",
                      "cmtitle": settings.wiki_category,
                      "cmlimit": min(settings.wiki_page_limit, 500),
                      "cmtype": "page"}
            data = self._query(params)
            for m in data.get("query", {}).get("categorymembers", []):
                pages.append({"pageid": str(m["pageid"]), "title": m["title"]})
        else:
            cont: dict = {}
            while len(pages) < settings.wiki_page_limit:
                params = {"action": "query", "list": "allpages",
                          "apnamespace": settings.wiki_namespace,
                          "aplimit": min(settings.wiki_page_limit - len(pages), 500)}
                params.update(cont)
                data = self._query(params)
                for m in data.get("query", {}).get("allpages", []):
                    pages.append({"pageid": str(m["pageid"]), "title": m["title"]})
                cont = data.get("continue", {})
                if not cont:
                    break
        return pages[: settings.wiki_page_limit]

    def build_tree(self) -> ConfluencePage:
        pages = self._list_pages()
        root_title = settings.wiki_category or "Wiki"
        root = ConfluencePage(page_id="__wiki_root__", title=root_title, level=0)
        for p in pages:
            root.children.append(
                ConfluencePage(page_id=p["pageid"], title=p["title"],
                               parent_id=root.page_id, level=1)
            )
        log.info(f"Discovered {len(root.children)} wiki pages.")
        return root

    def _fetch_html(self, pageid: str) -> str:
        data = self._query({"action": "parse", "pageid": pageid,
                            "prop": "text", "disableeditsection": True})
        return data.get("parse", {}).get("text", "") or ""

    def download_all(self, root: ConfluencePage) -> None:
        root_title = root.title
        for node in root.children:
            html = self._fetch_html(node.page_id)
            doc = RawDocument(
                page_id=node.page_id,
                title=node.title,
                html=html,
                url=self._page_url(node.title, node.page_id),
                source_type="mediawiki",
                breadcrumb=f"{root_title} > {node.title}",
                ancestor_titles=[root_title],
                parent_page=root_title,
                space_name=root_title,
            )
            (RAW_DIR / f"{node.page_id}.html").write_text(doc.html, encoding="utf-8")
            (RAW_DIR / f"{node.page_id}.meta.json").write_text(
                json.dumps(doc.sidecar(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            log.info(f"Downloaded wiki page: {node.title}")

    def metadata_extractor(self):
        return WikiMetadataExtractor()
