"""
Bootstrap a golden evaluation set for a NEW collection.

A brand-new knowledge base needs its own question -> correct-page labels before
it can be evaluated. Three ways to create them (all human-verified):

  1. Manual        -> the "Mark correct" button in the app (already available).
  2. auto_bootstrap -> draft candidate questions from page titles/sections
                       (no LLM, deterministic). Human approves/edits.
  3. llm_bootstrap  -> an LLM reads each page and writes Q&A pairs grounded in it
                       (needs an OpenAI key). Human approves.

Candidates are written to <collection>/evaluation/golden_candidates.json for
review; approving merges them into <collection>/evaluation/golden_qa.json.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from typing import Any

from app.core.config import collection_paths
from app.core.constants import STOP_WORDS

_TOKEN = re.compile(r"[A-Za-z0-9]+")


def _keywords(text: str, k: int = 4) -> list[str]:
    seen, out = set(), []
    for t in _TOKEN.findall(text or ""):
        tl = t.lower()
        if tl in STOP_WORDS or len(tl) < 3 or tl in seen:
            continue
        seen.add(tl); out.append(t)
        if len(out) >= k:
            break
    return out


def load_pages(collection: str) -> dict[str, list[dict]]:
    """Group a collection's chunks by page_id."""
    meta_path = collection_paths(collection).vectorstore / "chunks_metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"No index for collection '{collection}'. Ingest it first.")
    chunks = json.loads(meta_path.read_text(encoding="utf-8"))
    pages: dict[str, list[dict]] = defaultdict(list)
    for c in chunks:
        pages[str(c.get("page_id"))].append(c)
    return pages


def auto_bootstrap(collection: str, per_page: int = 2) -> list[dict[str, Any]]:
    """Deterministic candidate questions from titles and section headings."""
    pages = load_pages(collection)
    out: list[dict[str, Any]] = []
    i = 0
    for pid, chunks in pages.items():
        title = chunks[0].get("title") or pid
        # candidate 1: about the page
        i += 1
        out.append({
            "id": f"auto{i:03d}",
            "question": f"What information does the '{title}' page provide?",
            "relevant_page_ids": [pid],
            "answer_keywords": _keywords(title),
            "tag": "auto",
        })
        # further candidates: one per distinct section (up to per_page-1 more)
        sections = []
        for c in chunks:
            sec = (c.get("section") or "").strip()
            if sec and sec.lower() not in ("document", "overview") and sec not in sections:
                sections.append(sec)
        for sec in sections[: max(0, per_page - 1)]:
            i += 1
            out.append({
                "id": f"auto{i:03d}",
                "question": f"What does the '{sec}' section of '{title}' cover?",
                "relevant_page_ids": [pid],
                "answer_keywords": _keywords(f"{sec} {title}"),
                "tag": "auto",
            })
    return out


def llm_bootstrap(collection: str, model: str | None = None,
                  per_page: int = 2) -> list[dict[str, Any]]:
    """LLM-generated Q&A grounded in each page. Requires an OpenAI key."""
    from app.core.config import settings
    settings.require_openai()
    from openai import OpenAI  # local import; optional dependency path
    client = OpenAI(api_key=settings.openai_api_key)
    model = model or settings.openai_model

    pages = load_pages(collection)
    out: list[dict[str, Any]] = []
    i = 0
    for pid, chunks in pages.items():
        title = chunks[0].get("title") or pid
        text = "\n".join(c.get("text", "") for c in chunks)[:4000]
        prompt = (
            f"From the documentation page titled '{title}', write {per_page} "
            "natural questions a user might ask, each answerable ONLY from this "
            "page. Return STRICT JSON: a list of objects with keys "
            '"question" and "answer_keywords" (2-4 short keywords). '
            f"Page content:\n{text}"
        )
        try:
            resp = client.chat.completions.create(
                model=model, temperature=0.3, max_tokens=400,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = resp.choices[0].message.content.strip()
            raw = re.sub(r"^```json|^```|```$", "", raw, flags=re.MULTILINE).strip()
            items = json.loads(raw)
        except Exception:
            items = []
        for it in items[:per_page]:
            i += 1
            out.append({
                "id": f"llm{i:03d}",
                "question": it.get("question", "").strip(),
                "relevant_page_ids": [pid],
                "answer_keywords": it.get("answer_keywords", []),
                "tag": "llm",
            })
    return [q for q in out if q["question"]]


# ---------------- candidate storage & approval ----------------
def candidates_path(collection: str):
    return collection_paths(collection).evaluation / "golden_candidates.json"


def save_candidates(collection: str, questions: list[dict]) -> None:
    p = candidates_path(collection)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"questions": questions}, indent=2, ensure_ascii=False),
                 encoding="utf-8")


def load_candidates(collection: str) -> list[dict]:
    p = candidates_path(collection)
    if not p.exists():
        return []
    return json.loads(p.read_text(encoding="utf-8")).get("questions", [])


def approve_into_golden(collection: str, questions: list[dict]) -> int:
    """Merge approved candidates into the collection's golden set (dedup)."""
    golden = collection_paths(collection).golden
    golden.parent.mkdir(parents=True, exist_ok=True)
    if golden.exists():
        data = json.loads(golden.read_text(encoding="utf-8"))
    else:
        data = {"description": f"Golden set for collection '{collection}'.", "questions": []}
    existing = data.get("questions", [])
    seen = {(q["question"].strip().lower(), tuple(sorted(q.get("relevant_page_ids", []))))
            for q in existing}
    added = 0
    for q in questions:
        key = (q["question"].strip().lower(), tuple(sorted(q.get("relevant_page_ids", []))))
        if key in seen or not q["question"].strip():
            continue
        existing.append(q); seen.add(key); added += 1
    data["questions"] = existing
    golden.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return added
