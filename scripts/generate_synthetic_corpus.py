"""
Generate a synthetic benchmark corpus for dense-vs-hybrid retrieval evaluation.

Creates a fictional "Nimbus" platform of many services that are structurally and
lexically similar (near-duplicate prose -> hard for pure dense retrieval) but each
carries UNIQUE identifiers -- error codes, config keys, API endpoints, CLI commands
(-> where BM25 shines). This produces a corpus where hybrid retrieval shows a
measurable aggregate gain over dense-only.

Outputs (no embedding model needed):
    data/benchmark/chunks/<page_id>.json        # chunk records (pipeline schema)
    data/benchmark/cleaned/<page_id>.md         # readable page (optional/browsing)
    data/evaluation/golden_benchmark.json       # golden Q&A for this corpus

Then, locally:
    python scripts/build_benchmark.py           # embed + index (needs the model)
    python scripts/run_benchmark.py             # dense vs hybrid comparison

Usage:
    python scripts/generate_synthetic_corpus.py [--services 40] [--clean]
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

BASE = Path("data")
BENCH = BASE / "benchmark"
CHUNK_DIR = BENCH / "chunks"
CLEAN_DIR = BENCH / "cleaned"
GOLDEN = BASE / "evaluation" / "golden_benchmark.json"

# 48 distinct service names (deterministic corpus)
NAMES = [
    "Aurora", "Basalt", "Cobalt", "Dune", "Ember", "Flint", "Granite", "Halite",
    "Indigo", "Jasper", "Kepler", "Lumen", "Marble", "Nimbus", "Onyx", "Pyrite",
    "Quartz", "Rime", "Slate", "Talc", "Umber", "Verdant", "Willow", "Xenon",
    "Yarrow", "Zephyr", "Amber", "Beryl", "Cinder", "Delta", "Echo", "Fjord",
    "Glacier", "Harbor", "Ionic", "Jade", "Kraken", "Lyric", "Mica", "Nova",
    "Opal", "Prism", "Quill", "Raven", "Sable", "Topaz", "Ultra", "Vesper",
]

OWNERS = ["Platform Team", "SRE", "API Team", "Data Team", "Infra Guild"]
STATUSES = ["STABLE", "BETA", "DEPRECATED", "GA"]


def _chunk(page_id, title, section, number, total, text, source_url, breadcrumb, owner, status):
    return {
        "chunk_id": f"{page_id}_{number:03d}",
        "page_id": page_id,
        "title": title,
        "section": section,
        "heading_path": [section],
        "source_url": source_url,
        "owner": owner,
        "status": status,
        "audience": "Internal",
        "last_reviewed": "2026-08-01",
        "text": text.strip(),
        "word_count": len(text.split()),
        "character_count": len(text),
        "chunk_number": number,
        "total_chunks": total,
        "parent_page": "Nimbus Platform (Synthetic Benchmark)",
        "child_page": title,
        "breadcrumb": breadcrumb,
    }


def build_service(i: int, name: str):
    page_id = str(900001 + i)
    low = name.lower()
    up = name.upper()
    title = f"{name} Service"
    owner = OWNERS[i % len(OWNERS)]
    status = STATUSES[i % len(STATUSES)]
    url = f"https://nimbus.internal/docs/{low}"
    crumb = f"Nimbus Platform > Services > {title}"
    ver = f"{1 + i % 3}.{i % 10}.0"

    err_base = 1000 + i * 10
    codes = {
        err_base + 1: "connection pool exhausted",
        err_base + 2: "upstream timeout exceeded",
        err_base + 3: "invalid authentication token",
        err_base + 4: "payload schema validation failed",
    }
    cfg = {
        f"NIMBUS_{up}_TIMEOUT_MS": 3000 + i * 25,
        f"NIMBUS_{up}_MAX_RETRIES": 3 + (i % 4),
        f"NIMBUS_{up}_POOL_SIZE": 16 + (i % 8) * 4,
    }
    endpoints = {
        "restart": f"/api/v1/{low}/restart",
        "status": f"/api/v1/{low}/status",
        "flush": f"/api/v1/{low}/cache/flush",
    }
    cli = f"nimbusctl {low} restart"

    # 1) Overview  -- deliberately SHARED phrasing across services (dense-confusable)
    overview = (
        f"The {name} service is a component of the Nimbus platform responsible for "
        f"handling requests reliably and at high throughput. It provides horizontal "
        f"scalability, health checks, and graceful degradation under load. Like other "
        f"Nimbus services, {name} exposes metrics, supports rolling restarts, and "
        f"integrates with the shared observability stack. Current version: {name} v{ver}. "
        f"Owner: {owner}. This document describes configuration, endpoints, error codes, "
        f"and troubleshooting for {name}."
    )
    # 2) Configuration -- UNIQUE config keys (BM25-friendly exact tokens)
    conf = "Configuration keys and defaults for the service:\n" + "\n".join(
        f"- {k} (default {v})" for k, v in cfg.items()
    ) + (
        f"\nEach key can be overridden via environment variable or the nimbus.yaml file. "
        f"Changing {list(cfg)[0]} affects request timeout behavior for {name}."
    )
    # 3) API endpoints -- UNIQUE paths
    api = "HTTP endpoints exposed by the service:\n" + "\n".join(
        f"- POST {p}" if k != "status" else f"- GET {p}" for k, p in endpoints.items()
    ) + (
        f"\nUse POST {endpoints['restart']} to restart the {name} service, "
        f"GET {endpoints['status']} to check health, and "
        f"POST {endpoints['flush']} to clear its cache."
    )
    # 4) Error codes -- UNIQUE ERR-#### tokens
    errs = f"Error codes emitted by the {name} service:\n" + "\n".join(
        f"- ERR-{code}: {msg}" for code, msg in codes.items()
    ) + (
        f"\nWhen you see ERR-{err_base + 1}, the connection pool is exhausted; "
        f"increase {list(cfg)[-1]}. ERR-{err_base + 2} indicates an upstream timeout."
    )
    # 5) Troubleshooting -- SHARED phrasing + refs (mixed signal)
    trouble = (
        f"Troubleshooting the {name} service. If you observe connection drops or high "
        f"latency, first check GET {endpoints['status']}, then review recent error codes. "
        f"Restart with `{cli}` or POST {endpoints['restart']}. Persistent timeouts usually "
        f"mean {list(cfg)[0]} is set too low. As with all Nimbus services, verify upstream "
        f"health and inspect the shared dashboards before escalating to {owner}."
    )

    sections = [
        ("Overview", overview),
        ("Configuration", conf),
        ("API Endpoints", api),
        ("Error Codes", errs),
        ("Troubleshooting", trouble),
    ]
    total = len(sections)
    chunks = [
        _chunk(page_id, title, sec, n + 1, total, txt, url, crumb, owner, status)
        for n, (sec, txt) in enumerate(sections)
    ]

    md = f"# {title}\n\n" + "\n\n".join(f"## {sec}\n\n{txt}" for sec, txt in sections)

    # Golden questions.
    #  - EXACT-MATCH are NAME-FREE, keyed only by an opaque error code. Embeddings
    #    treat "ERR-1042" as near-meaningless and every service's error section reads
    #    almost identically, so pure dense retrieval struggles to pick the right one;
    #    BM25 matches the exact token. This is where hybrid wins.
    #  - SEMANTIC name the service (a strong signal), so dense does well -> both fine.
    qid = f"b{i:03d}"
    e1, e2 = err_base + 1, err_base + 2
    q_exact_1 = {"id": f"{qid}a", "question": f"What does error code ERR-{e1} indicate?",
                 "relevant_page_ids": [page_id], "answer_keywords": [f"ERR-{e1}", "pool"],
                 "tag": "exact_match"}
    q_semantic = {"id": f"{qid}b", "question": f"How do I fix connection drops in the {name} service?",
                  "relevant_page_ids": [page_id], "answer_keywords": ["restart", "status"],
                  "tag": "semantic"}
    q_exact_2 = {"id": f"{qid}c", "question": f"What is the meaning of error ERR-{e2}?",
                 "relevant_page_ids": [page_id], "answer_keywords": [f"ERR-{e2}", "timeout"],
                 "tag": "exact_match"}
    q_named = {"id": f"{qid}d", "question": f"How do I restart the {name} service?",
               "relevant_page_ids": [page_id], "answer_keywords": [endpoints["restart"], "nimbusctl"],
               "tag": "semantic"}
    # interleave: qps=2 -> 1 name-free exact + 1 named semantic (balanced)
    q = [q_exact_1, q_semantic, q_exact_2, q_named]
    return page_id, chunks, md, q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--services", type=int, default=40,
                    help="how many services to generate (max 48)")
    ap.add_argument("--questions-per-service", type=int, default=2,
                    help="golden questions kept per service (1-4)")
    ap.add_argument("--clean", action="store_true", help="remove the benchmark corpus")
    args = ap.parse_args()

    if args.clean:
        shutil.rmtree(BENCH, ignore_errors=True)
        GOLDEN.unlink(missing_ok=True)
        print("Removed benchmark corpus and golden set.")
        return

    n = max(1, min(args.services, len(NAMES)))
    qps = max(1, min(args.questions_per_service, 4))
    CHUNK_DIR.mkdir(parents=True, exist_ok=True)
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    GOLDEN.parent.mkdir(parents=True, exist_ok=True)

    total_chunks = 0
    golden = []
    for i in range(n):
        page_id, chunks, md, q = build_service(i, NAMES[i])
        (CHUNK_DIR / f"{page_id}.json").write_text(
            json.dumps(chunks, indent=2, ensure_ascii=False), encoding="utf-8")
        (CLEAN_DIR / f"{page_id}.md").write_text(md, encoding="utf-8")
        total_chunks += len(chunks)
        # keep a mix: always the exact-match ones first, then semantic
        golden.extend(q[:qps])

    GOLDEN.write_text(json.dumps(
        {"description": "Synthetic Nimbus benchmark: exact-match + semantic questions.",
         "questions": golden}, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Generated {n} services, {total_chunks} chunks -> {CHUNK_DIR}")
    print(f"Golden questions: {len(golden)} -> {GOLDEN}")
    print("\nNext:")
    print("  python scripts/build_benchmark.py     # embed + index (needs the model)")
    print("  python scripts/run_benchmark.py       # dense vs hybrid comparison")


if __name__ == "__main__":
    main()
