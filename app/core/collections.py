"""
Collections registry: manage multiple, isolated knowledge bases.

Each collection is a named knowledge base with its own folder, index and golden
set (see config.CollectionPaths). The registry stores each collection's SOURCE
configuration so it can be re-ingested reproducibly. Secrets are NEVER stored
here \u2014 tokens are referenced by environment-variable name.

Registry file: data/collections/registry.json
"""
from __future__ import annotations

import json
import os
from typing import Any

from app.core.config import COLLECTIONS_DIR, DATA_DIR, collection_paths
from app.core.exceptions import ConfigError

REGISTRY_PATH = COLLECTIONS_DIR / "registry.json"

DEFAULT_ENTRY = {
    "name": "default",
    "title": "Default (current Confluence)",
    "source": "confluence",
    "description": "The originally ingested Confluence space.",
}


def _load_raw() -> dict[str, Any]:
    if REGISTRY_PATH.exists():
        try:
            return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"collections": []}


def _save_raw(data: dict[str, Any]) -> None:
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False),
                             encoding="utf-8")


def list_collections() -> list[dict[str, Any]]:
    """All registered collections. 'default' is always present first."""
    data = _load_raw()
    cols = data.get("collections", [])
    names = {c.get("name") for c in cols}
    if "default" not in names:
        cols = [DEFAULT_ENTRY, *cols]
    # only surface collections whose vectorstore actually exists, plus default
    out = []
    for c in cols:
        p = collection_paths(c["name"])
        c = dict(c)
        c["has_index"] = (p.vectorstore / "chunks_metadata.json").exists()
        c["has_golden"] = p.golden.exists()
        out.append(c)
    return out


def get_collection(name: str) -> dict[str, Any]:
    for c in list_collections():
        if c["name"] == name:
            return c
    raise ConfigError(f"Unknown collection: {name!r}. "
                      f"Registered: {[c['name'] for c in list_collections()]}")


def register_collection(name: str, source: str, title: str | None = None,
                        description: str = "", **source_cfg: Any) -> dict[str, Any]:
    """
    Add or update a collection. `source_cfg` holds source-specific settings, e.g.
    Confluence: confluence_base_url, confluence_parent_page_id, token_env
    MediaWiki:  wiki_api_url, wiki_base_url, wiki_category, wiki_page_limit
    Never pass raw secrets \u2014 use token_env='ENV_VAR_NAME'.
    """
    if name == "default":
        raise ConfigError("'default' is reserved for the original data/ layout.")
    if source not in ("confluence", "mediawiki"):
        raise ConfigError("source must be 'confluence' or 'mediawiki'.")
    data = _load_raw()
    cols = [c for c in data.get("collections", []) if c.get("name") != name]
    entry = {"name": name, "title": title or name, "source": source,
             "description": description, **source_cfg}
    cols.append(entry)
    data["collections"] = cols
    _save_raw(data)
    collection_paths(name).mkdirs()
    return entry


def remove_collection(name: str) -> None:
    if name == "default":
        raise ConfigError("Cannot remove the 'default' collection.")
    data = _load_raw()
    data["collections"] = [c for c in data.get("collections", []) if c.get("name") != name]
    _save_raw(data)


def apply_source_config(entry: dict[str, Any], settings) -> None:
    """
    Push a collection's registered source settings onto the live `settings`
    object before ingestion, so the pipeline ingests THIS collection's source.
    Tokens are read from the environment variable named by `token_env`.
    """
    src = entry.get("source", "confluence")
    settings.source = src
    if src == "confluence":
        if entry.get("confluence_base_url"):
            settings.confluence_base_url = entry["confluence_base_url"]
        if entry.get("confluence_parent_page_id"):
            settings.confluence_parent_page_id = str(entry["confluence_parent_page_id"])
        if entry.get("confluence_email"):
            settings.confluence_email = entry["confluence_email"]
        tok = os.getenv(entry.get("token_env", "CONFLUENCE_API_TOKEN"))
        if tok:
            settings.confluence_api_token = tok
    elif src == "mediawiki":
        for k in ("wiki_api_url", "wiki_base_url", "wiki_category"):
            if entry.get(k):
                setattr(settings, k, entry[k])
        if entry.get("wiki_page_limit"):
            settings.wiki_page_limit = int(entry["wiki_page_limit"])
