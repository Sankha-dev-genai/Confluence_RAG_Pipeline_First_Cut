"""
BM25 sparse (keyword) index over the same chunk corpus as the dense index.

Complements dense retrieval: BM25 excels at exact terms (error codes, API names,
IDs) that embeddings sometimes smooth over. Built in-memory from
chunks_metadata.json — trivial for this corpus size.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

from app.core.config import VECTORSTORE_DIR
from app.core.constants import STOP_WORDS

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall((text or "").lower()) if t not in STOP_WORDS]


class BM25Index:
    def __init__(self, metadata_path: str | Path | None = None) -> None:
        path = Path(metadata_path or (VECTORSTORE_DIR / "chunks_metadata.json"))
        self.metadata: list[dict[str, Any]] = json.loads(path.read_text(encoding="utf-8"))
        self.corpus_tokens = [tokenize(self._doc_text(m)) for m in self.metadata]
        self.bm25 = BM25Okapi(self.corpus_tokens)

    @staticmethod
    def _doc_text(m: dict) -> str:
        parts = [
            m.get("title", ""),
            m.get("section", ""),
            " ".join(m.get("heading_path", []) or []),
            m.get("breadcrumb", ""),
            m.get("text", ""),
        ]
        return " ".join(p for p in parts if p)

    def search(self, query: str, top_k: int = 30) -> list[dict[str, Any]]:
        scores = self.bm25.get_scores(tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        results: list[dict[str, Any]] = []
        for i in order[:top_k]:
            if scores[i] <= 0:
                continue
            rec = dict(self.metadata[i])
            rec["bm25_score"] = float(scores[i])
            results.append(rec)
        return results
