"""
Central configuration.

Exposes a validated `settings` object (production style) AND keeps the
original module-level constants so every existing import keeps working.
Nothing raises at import time — validation happens at point-of-use via the
`require_*` helpers, so the app still runs for retrieval/eval without keys.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel

from app.core.exceptions import ConfigError

load_dotenv()

# ---------------------------------------------------------------- paths
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
COLLECTIONS_DIR = DATA_DIR / "collections"


class CollectionPaths:
    """All directories for one named knowledge base (collection).

    The special name "default" maps to the original data/ layout, so existing
    data and behaviour are unchanged. Any other name lives under
    data/collections/<name>/ and is fully isolated.
    """

    def __init__(self, name: str = "default") -> None:
        self.name = name or "default"
        base = DATA_DIR if self.name == "default" else (COLLECTIONS_DIR / self.name)
        self.base = base
        self.raw = base / "raw"
        self.cleaned = base / "cleaned"
        self.chunks = base / "chunks"
        self.metadata = base / "metadata"
        self.vectorstore = base / "vectorstore"
        self.evaluation = base / "evaluation"
        self.golden = base / "evaluation" / "golden_qa.json"

    def mkdirs(self) -> "CollectionPaths":
        for d in (self.raw, self.cleaned, self.chunks, self.metadata,
                  self.vectorstore, self.evaluation):
            d.mkdir(parents=True, exist_ok=True)
        return self


def collection_paths(name: str = "default") -> "CollectionPaths":
    return CollectionPaths(name)


# The active collection for THIS process is chosen by the COLLECTION env var,
# read before any module binds the path constants below. Ingestion runs one
# collection per process (python main.py --collection <name>); the app queries
# many collections via explicit directory arguments.
ACTIVE_COLLECTION = os.getenv("COLLECTION", "default")
_paths = collection_paths(ACTIVE_COLLECTION).mkdirs()

RAW_DIR = _paths.raw
CLEANED_DIR = _paths.cleaned
CHUNKS_DIR = _paths.chunks
METADATA_DIR = _paths.metadata
VECTORSTORE_DIR = _paths.vectorstore
EVAL_DIR = _paths.evaluation


# ---------------------------------------------------------------- settings
class Settings(BaseModel):
    # which knowledge source to ingest from
    source: str = "confluence"          # "confluence" | "mediawiki"

    # --- Confluence ---
    confluence_base_url: str | None = None
    confluence_email: str | None = None
    confluence_api_token: str | None = None
    confluence_parent_page_id: str | None = None

    # --- MediaWiki / generic wiki ---
    wiki_api_url: str | None = None      # e.g. https://en.wikipedia.org/w/api.php
    wiki_base_url: str | None = None     # e.g. https://en.wikipedia.org/wiki
    wiki_namespace: int = 0
    wiki_category: str | None = None     # optional: only pages in this category
    wiki_page_limit: int = 200

    # --- LLM ---
    llm_provider: str = "openai"
    openai_api_key: str | None = None
    openai_model: str = "gpt-4.1-mini"

    # --- embeddings / retrieval ---
    embedding_model: str = "all-MiniLM-L6-v2"
    top_k: int = 20
    min_score: float = 0.20
    context_k: int = 5

    # --- hybrid retrieval ---
    use_hybrid: bool = True
    hybrid_alpha: float = 0.5   # dense weight; (1-alpha) is BM25 weight

    # --- http ---
    request_timeout: int = 30
    http_retries: int = 3

    @classmethod
    def from_env(cls) -> "Settings":
        g = os.getenv
        def _int(name, default): 
            try: return int(g(name, default))
            except (TypeError, ValueError): return default
        def _float(name, default):
            try: return float(g(name, default))
            except (TypeError, ValueError): return default
        return cls(
            source=g("SOURCE", "confluence").lower(),
            confluence_base_url=g("CONFLUENCE_BASE_URL"),
            confluence_email=g("CONFLUENCE_EMAIL"),
            confluence_api_token=g("CONFLUENCE_API_TOKEN"),
            confluence_parent_page_id=g("CONFLUENCE_PARENT_PAGE_ID"),
            wiki_api_url=g("WIKI_API_URL"),
            wiki_base_url=g("WIKI_BASE_URL"),
            wiki_namespace=_int("WIKI_NAMESPACE", "0"),
            wiki_category=g("WIKI_CATEGORY"),
            wiki_page_limit=_int("WIKI_PAGE_LIMIT", "200"),
            llm_provider=g("LLM_PROVIDER", "openai"),
            openai_api_key=g("OPENAI_API_KEY"),
            openai_model=g("OPENAI_MODEL", "gpt-4.1-mini"),
            embedding_model=g("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
            top_k=_int("TOP_K", "20"),
            min_score=_float("MIN_SCORE", "0.20"),
            context_k=_int("CONTEXT_K", "5"),
            use_hybrid=g("HYBRID", "true").lower() not in ("0","false","no"),
            hybrid_alpha=_float("HYBRID_ALPHA", "0.5"),
            request_timeout=_int("REQUEST_TIMEOUT", "30"),
            http_retries=_int("HTTP_RETRIES", "3"),
        )

    # ---- point-of-use validation (raise only when actually needed) ----
    def require_confluence(self) -> None:
        missing = [k for k, v in {
            "CONFLUENCE_BASE_URL": self.confluence_base_url,
            "CONFLUENCE_EMAIL": self.confluence_email,
            "CONFLUENCE_API_TOKEN": self.confluence_api_token,
            "CONFLUENCE_PARENT_PAGE_ID": self.confluence_parent_page_id,
        }.items() if not v]
        if missing:
            raise ConfigError("Missing Confluence config: " + ", ".join(missing))

    def require_wiki(self) -> None:
        if not self.wiki_api_url:
            raise ConfigError("Missing WIKI_API_URL for the mediawiki source.")

    def require_source(self) -> None:
        if self.source == "confluence":
            self.require_confluence()
        elif self.source == "mediawiki":
            self.require_wiki()
        else:
            raise ConfigError(f"Unknown SOURCE: {self.source!r}")

    def require_openai(self) -> None:
        if not self.openai_api_key:
            raise ConfigError("OPENAI_API_KEY is not set.")


settings = Settings.from_env()

# ------------------------------------------- backward-compatible constants
CONFLUENCE_BASE_URL = settings.confluence_base_url
CONFLUENCE_EMAIL = settings.confluence_email
CONFLUENCE_API_TOKEN = settings.confluence_api_token
CONFLUENCE_PARENT_PAGE_ID = settings.confluence_parent_page_id
OPENAI_API_KEY = settings.openai_api_key
OPENAI_MODEL = settings.openai_model
