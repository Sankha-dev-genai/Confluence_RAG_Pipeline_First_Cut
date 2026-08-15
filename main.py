"""
End-to-end ingestion & indexing pipeline (the project's front door).

Multi-collection & source-agnostic. Each collection is an isolated knowledge
base (own folder, index and golden set). The special collection 'default' maps
to the original data/ layout.

Stages:  connect -> tree -> download -> clean -> metadata -> chunk -> embed -> index

Usage:
    python main.py                                   # ingest the 'default' collection
    python main.py --collection eng_wiki             # ingest a named collection
    python main.py --collection eng_wiki --steps chunk embed index
    python main.py --list-collections
    python main.py --add-collection eng_wiki --source mediawiki \
                   --wiki-api-url https://en.wikipedia.org/w/api.php \
                   --wiki-category Category:Machine_learning
    python main.py --add-collection space2 --source confluence \
                   --confluence-base-url https://site.atlassian.net \
                   --parent-page-id 12345 --token-env CONFLUENCE_API_TOKEN
    python main.py --list-steps
"""
from __future__ import annotations

import argparse
import os
import sys

# --- choose the active collection BEFORE importing config (paths depend on it) ---
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--collection", default="default")
_known, _ = _pre.parse_known_args()
os.environ["COLLECTION"] = _known.collection

from app.core.config import settings                       # noqa: E402
from app.core.exceptions import IngestionError             # noqa: E402
from app.core.logger import get_logger                     # noqa: E402
from app.core import collections as col                    # noqa: E402

log = get_logger("pipeline")


def stage_connect(state):
    from app.sources import get_source
    src = get_source(); info = src.test_connection()
    log.info(f"Connected to '{src.name}': {info}")


def stage_tree(state):
    from app.sources import get_source
    from app.ingestion.tree_utils import print_tree
    root = get_source().build_tree(); state["tree"] = root
    log.info("Page hierarchy:"); print_tree(root)


def stage_download(state):
    from app.sources import get_source
    src = get_source(); root = state.get("tree") or src.build_tree()
    src.download_all(root); log.info("Download complete.")


def stage_clean(state):
    from app.processing.html_cleaner import HTMLCleaner
    HTMLCleaner().clean_all(); log.info("HTML cleaned -> markdown.")


def stage_metadata(state):
    from app.sources import get_source
    get_source().metadata_extractor().extract_all(); log.info("Metadata extracted.")


def stage_chunk(state):
    from app.processing.chunker import MarkdownChunker
    MarkdownChunker().process_all(); log.info("Chunks created.")


def stage_embed(state):
    from app.embeddings.embedding_generator import EmbeddingGenerator
    EmbeddingGenerator().process_all(); log.info("Embeddings generated.")


def stage_index(state):
    from app.embeddings.vector_store import VectorStore
    VectorStore().build_index(); log.info("FAISS index built.")


STAGES = {"connect": stage_connect, "tree": stage_tree, "download": stage_download,
          "clean": stage_clean, "metadata": stage_metadata, "chunk": stage_chunk,
          "embed": stage_embed, "index": stage_index}
DEFAULT_ORDER = ["connect", "tree", "download", "clean", "metadata", "chunk", "embed", "index"]


def run(steps, collection):
    state = {}
    log.info("=" * 56)
    log.info(f"INGESTION  collection='{collection}'  source={settings.source}")
    log.info("=" * 56)
    for name in steps:
        log.info(f"[stage] {name}")
        try:
            STAGES[name](state)
        except Exception as exc:  # noqa: BLE001
            raise IngestionError(f"stage '{name}' failed: {exc}") from exc
    log.info(f"Pipeline finished. Collection '{collection}' is ready.")


def main():
    p = argparse.ArgumentParser(description="Knowledge ingestion pipeline (multi-collection)")
    p.add_argument("--collection", default="default", help="which knowledge base to build")
    p.add_argument("--source", choices=["confluence", "mediawiki"], help="override source")
    p.add_argument("--steps", nargs="+", choices=list(STAGES), help="subset of stages")
    p.add_argument("--list-steps", action="store_true")
    p.add_argument("--list-collections", action="store_true")
    # collection registration
    p.add_argument("--add-collection", metavar="NAME", help="register a new collection")
    p.add_argument("--title"); p.add_argument("--description", default="")
    p.add_argument("--confluence-base-url"); p.add_argument("--confluence-email")
    p.add_argument("--parent-page-id"); p.add_argument("--token-env", default="CONFLUENCE_API_TOKEN")
    p.add_argument("--wiki-api-url"); p.add_argument("--wiki-base-url")
    p.add_argument("--wiki-category"); p.add_argument("--wiki-page-limit", type=int)
    args = p.parse_args()

    if args.list_collections:
        print("Registered collections:")
        for c in col.list_collections():
            flags = ("index" if c["has_index"] else "no-index") + \
                    (", golden" if c["has_golden"] else "")
            print(f"  \u2022 {c['name']:16s} [{c['source']}]  ({flags})  {c.get('title','')}")
        return

    if args.add_collection:
        src = args.source or ("mediawiki" if args.wiki_api_url else "confluence")
        cfg = {}
        if src == "confluence":
            cfg = {"confluence_base_url": args.confluence_base_url,
                   "confluence_email": args.confluence_email,
                   "confluence_parent_page_id": args.parent_page_id,
                   "token_env": args.token_env}
        else:
            cfg = {"wiki_api_url": args.wiki_api_url, "wiki_base_url": args.wiki_base_url,
                   "wiki_category": args.wiki_category, "wiki_page_limit": args.wiki_page_limit}
        cfg = {k: v for k, v in cfg.items() if v is not None}
        entry = col.register_collection(args.add_collection, src, title=args.title,
                                        description=args.description, **cfg)
        print(f"Registered collection '{entry['name']}' [{src}].")
        print(f"Now ingest it:  python main.py --collection {entry['name']}")
        return

    if args.list_steps:
        print(f"Collection: {args.collection}  |  Stages:", " -> ".join(DEFAULT_ORDER))
        return

    # apply this collection's registered source config (unless it's default)
    if args.collection != "default":
        entry = col.get_collection(args.collection)
        col.apply_source_config(entry, settings)
    if args.source:
        settings.source = args.source

    steps = [s for s in DEFAULT_ORDER if s in (args.steps or DEFAULT_ORDER)]
    run(steps, args.collection)


if __name__ == "__main__":
    main()
