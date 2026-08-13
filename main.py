"""
End-to-end ingestion & indexing pipeline (the project's front door).

Source-agnostic: set SOURCE=confluence or SOURCE=mediawiki in .env.

Stages, in order:
    connect -> tree -> download -> clean -> metadata -> chunk -> embed -> index

Usage:
    python main.py                              # whole pipeline for the active source
    python main.py --source mediawiki           # override source for this run
    python main.py --steps tree                 # just print the page hierarchy
    python main.py --steps chunk embed index    # rebuild the index from cleaned data
    python main.py --list-steps
"""
from __future__ import annotations

import argparse

from app.core.config import settings
from app.core.exceptions import IngestionError
from app.core.logger import get_logger

log = get_logger("pipeline")


def stage_connect(state: dict) -> None:
    from app.sources import get_source
    src = get_source()
    info = src.test_connection()
    log.info(f"Connected to '{src.name}': {info}")


def stage_tree(state: dict) -> None:
    from app.sources import get_source
    from app.ingestion.tree_utils import print_tree
    root = get_source().build_tree()
    state["tree"] = root
    log.info("Page hierarchy:")
    print_tree(root)


def stage_download(state: dict) -> None:
    from app.sources import get_source
    src = get_source()
    root = state.get("tree") or src.build_tree()
    src.download_all(root)
    log.info("Download complete.")


def stage_clean(state: dict) -> None:
    from app.processing.html_cleaner import HTMLCleaner
    HTMLCleaner().clean_all()
    log.info("HTML cleaned -> markdown.")


def stage_metadata(state: dict) -> None:
    from app.sources import get_source
    get_source().metadata_extractor().extract_all()
    log.info("Metadata extracted.")


def stage_chunk(state: dict) -> None:
    from app.processing.chunker import MarkdownChunker
    MarkdownChunker().process_all()
    log.info("Chunks created.")


def stage_embed(state: dict) -> None:
    from app.embeddings.embedding_generator import EmbeddingGenerator
    EmbeddingGenerator().process_all()
    log.info("Embeddings generated.")


def stage_index(state: dict) -> None:
    from app.embeddings.vector_store import VectorStore
    VectorStore().build_index()
    log.info("FAISS index built.")


STAGES = {
    "connect": stage_connect, "tree": stage_tree, "download": stage_download,
    "clean": stage_clean, "metadata": stage_metadata, "chunk": stage_chunk,
    "embed": stage_embed, "index": stage_index,
}
DEFAULT_ORDER = ["connect", "tree", "download", "clean",
                 "metadata", "chunk", "embed", "index"]


def run(steps: list[str]) -> None:
    state: dict = {}
    log.info("=" * 52)
    log.info(f"INGESTION PIPELINE  (source={settings.source})")
    log.info("=" * 52)
    for name in steps:
        log.info(f"[stage] {name}")
        try:
            STAGES[name](state)
        except Exception as exc:  # noqa: BLE001
            raise IngestionError(f"stage '{name}' failed: {exc}") from exc
    log.info("Pipeline finished successfully.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Knowledge ingestion pipeline")
    parser.add_argument("--source", choices=["confluence", "mediawiki"],
                        help="override SOURCE for this run")
    parser.add_argument("--steps", nargs="+", choices=list(STAGES),
                        help="subset of stages (default: all, in order)")
    parser.add_argument("--list-steps", action="store_true")
    args = parser.parse_args()

    if args.source:
        settings.source = args.source

    if args.list_steps:
        print(f"Source: {settings.source}")
        print("Stages:", " -> ".join(DEFAULT_ORDER))
        return

    steps = [s for s in DEFAULT_ORDER if s in (args.steps or DEFAULT_ORDER)]
    run(steps)


if __name__ == "__main__":
    main()
