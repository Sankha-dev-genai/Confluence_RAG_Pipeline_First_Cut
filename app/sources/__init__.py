"""Source factory: picks the adapter from settings.source."""
from __future__ import annotations

from app.core.config import settings
from app.core.exceptions import ConfigError
from app.sources.base import KnowledgeSource


def get_source(name: str | None = None) -> KnowledgeSource:
    name = (name or settings.source or "confluence").lower()
    if name == "confluence":
        from app.sources.confluence import ConfluenceSource
        return ConfluenceSource()
    if name in ("mediawiki", "wiki"):
        from app.sources.mediawiki import MediaWikiSource
        return MediaWikiSource()
    raise ConfigError(f"Unknown source: {name!r}. Use 'confluence' or 'mediawiki'.")
