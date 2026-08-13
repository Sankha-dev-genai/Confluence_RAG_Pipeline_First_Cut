# Knowledge RAG — Grounded Q&A over Confluence *or* MediaWiki

A source-agnostic Retrieval-Augmented Generation system: ingest a wiki, ask
questions in natural language, and get **grounded answers with exact citations**,
backed by a **retrieval evaluation harness** (Precision@k, Recall@k, Hit@k, MRR,
MAP, context-sufficiency) and an **interactive Streamlit app**.

Switch the entire knowledge source with one line in `.env`:
`SOURCE=confluence` or `SOURCE=mediawiki`.

## Architecture

    SOURCE (Confluence | MediaWiki)
        -> ingest (tree + raw HTML + metadata sidecar)
        -> clean (HTML -> Markdown)
        -> metadata (source-aware provider)
        -> chunk (section/table/code-aware)
        -> embed (all-MiniLM-L6-v2, 384-d, normalized)
        -> FAISS (IndexFlatIP = cosine)

    question -> retrieve -> metadata rerank -> citations
                                    |               |
                                context -> LLM -> answer (+ citations + metrics)

    golden set -> evaluate -> Precision@k / Recall@k / Hit@k / MRR / MAP / context-sufficiency

The **source-adapter layer** (`app/sources/`) is the only source-specific code.
Everything downstream operates on normalized raw HTML + a metadata sidecar, so a
new source (SharePoint, Notion, ...) is just one class implementing
`KnowledgeSource`.

## Project layout

    app/
      core/        config (validated Settings), logger, exceptions, http (retries)
      sources/     KnowledgeSource ABC + ConfluenceSource + MediaWikiSource + factory
      ingestion/   Confluence client, page fetch, hierarchy, downloader
      processing/  HTML cleaning, chunking, metadata (Confluence + generic wiki)
      embeddings/  embedding model, generator, FAISS vector store
      retrieval/   retriever, metadata reranker, citation builder
      llm/         OpenAI generator, prompt builder, response parser
      pipeline/    RAGPipeline (query orchestrator)
      evaluation/  metrics + evaluator (+ context-sufficiency)
    data/          raw/ cleaned/ chunks/ metadata/ vectorstore/ evaluation/
    scripts/       pipeline utilities  (+ dev/ REPL debug tools)
    tests/         pytest suite
    main.py        ingestion CLI (source-agnostic, staged)
    streamlit_app.py
    Dockerfile · Makefile · pyproject.toml

## Setup

    python -m venv .venv
    # Windows: .venv\Scripts\activate  |  macOS/Linux: source .venv/bin/activate
    pip install -r requirements.txt
    cp .env.example .env        # then fill in your values

The vector store ships in `data/vectorstore/`, so the app and evaluation run
immediately without re-ingesting.

## Ingest (build the index)

    python main.py                           # full pipeline for the active SOURCE
    python main.py --list-steps              # show stages
    python main.py --steps tree              # just print the hierarchy
    python main.py --steps chunk embed index # rebuild index from cleaned data
    python main.py --source mediawiki        # override source for one run

Stages: `connect -> tree -> download -> clean -> metadata -> chunk -> embed -> index`.

### Using a MediaWiki source

Set in `.env`:

    SOURCE=mediawiki
    WIKI_API_URL=https://en.wikipedia.org/w/api.php
    WIKI_BASE_URL=https://en.wikipedia.org/wiki
    WIKI_CATEGORY=Category:Machine_learning     # optional; else lists all pages
    WIKI_PAGE_LIMIT=50

Then `python main.py`. The clean/chunk/embed/retrieve/answer stack is unchanged.

## Run the app

    streamlit run streamlit_app.py

Tabs: **Ask** (answer + confidence + metrics + citation cards), **Evaluation**
(Precision@3/@5, Recall, Hit@k, MRR, MAP, context-sufficiency + chart + table),
**Corpus**, **Architecture**. Runs with no OpenAI key (retrieval + citations only).

## Evaluate

    python scripts/run_evaluation.py     # writes data/evaluation/results.json
    python scripts/run_comparison.py     # dense vs hybrid, writes comparison.json

Metrics: **Precision@k / Recall@k / Hit@k / MRR / MAP** (page-level ranking) plus
**context-sufficiency@5** — the fraction of expected answer facts present in the
retrieved context (a groundedness proxy; high value means retrieval surfaced what
an answer needs). Most questions have a single relevant page, so Precision@3 caps
at 0.33 — quote **Hit@1** and **MRR** as headline numbers.

### Dense vs Hybrid

The golden set includes exact-match questions (endpoints, status codes, framework
names) where BM25 helps most. `scripts/run_comparison.py` (and the **Compare
dense vs hybrid** button in the app's Evaluation tab) runs both retrievers and
shows the per-metric delta side by side.

## Scaling benchmark (dense vs hybrid at ~200 chunks)

A synthetic "Nimbus" corpus (40 near-duplicate service pages, ~200 chunks) with a
golden set that mixes **name-free exact-match** questions (opaque error codes —
dense-hard, BM25-easy) and **named semantic** questions (dense-friendly). It lives
in its OWN vector store, separate from your real Confluence index.

    make benchmark-generate     # writes data/benchmark/chunks + golden_benchmark.json
    make benchmark-build        # embeds into data/benchmark/vectorstore (needs model)
    make benchmark              # prints dense vs hybrid deltas

The corpus is shipped pre-generated; you only run build + benchmark. Expect hybrid
to roughly match dense on the semantic half and clearly beat it on the exact-match
half (error-code lookups), lifting aggregate Hit@1 / MRR. BM25 alone already scores
100% Hit@1 on the exact-match questions offline.

## Docker

    docker build -t knowledge-rag .
    docker run --rm -p 8501:8501 --env-file .env knowledge-rag   # app
    docker run --rm --env-file .env knowledge-rag python main.py # ingest

## Make targets

    make install | ingest | reindex | app | eval | test | lint | docker-build | docker-run

## Tests

    pytest -q       # 18 tests: metrics, config, sources, citations

## What changed in this pass

- Added `app/sources/` adapter layer + `MediaWikiSource` (multi-source).
- `main.py` is now source-agnostic and staged.
- `config.py` -> validated `Settings` (typed, point-of-use validation) with full
  backward compatibility; reads `OPENAI_MODEL`, `EMBEDDING_MODEL`, retrieval knobs.
- Filled `logger.py` / `exceptions.py`; added `http.py` (retries/backoff).
- Added generic `WikiMetadataExtractor` (no Confluence coupling) via subclassing.
- Added `context-sufficiency@5` to the evaluation.
- Cleaned `scripts/`: removed redundant stage-runners (now `main.py --steps`),
  moved interactive tools to `scripts/dev/repl_*.py`.
- Added `pyproject.toml`, `Dockerfile`, `Makefile`, dev deps, and pytest coverage.

## Future enhancements (scoped, not yet built)

Hybrid retrieval (BM25 + dense), a cross-encoder rerank pass, LLM-judge
faithfulness scoring, streaming answers, and incremental re-indexing of only
changed pages.
