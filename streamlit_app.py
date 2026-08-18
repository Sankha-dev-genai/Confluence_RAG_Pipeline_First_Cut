"""
Confluence RAG — Streamlit demo app.

Run from the project root:
    streamlit run streamlit_app.py

Works with or without an OpenAI key:
- With a key  -> full grounded answers + citations + metrics.
- Without one -> retrieval + citations still shown (generation disabled).
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

import app.core.config as config

# --- collection-aware paths (reset each run from the sidebar picker) ---
_def = config.collection_paths("default")
VSTORE = _def.vectorstore
META_PATH = _def.vectorstore / "chunks_metadata.json"
RESULTS_PATH = _def.evaluation / "results.json"
GOLDEN_PATH = _def.golden
ASK_HISTORY = _def.evaluation / "ask_history.json"


def set_active_collection(name):
    """Point all app paths at the chosen collection for this run."""
    global VSTORE, META_PATH, RESULTS_PATH, GOLDEN_PATH, ASK_HISTORY
    p = config.collection_paths(name)
    VSTORE = p.vectorstore
    META_PATH = p.vectorstore / "chunks_metadata.json"
    RESULTS_PATH = p.evaluation / "results.json"
    GOLDEN_PATH = p.golden
    ASK_HISTORY = p.evaluation / "ask_history.json"

st.set_page_config(
    page_title="Confluence RAG",
    page_icon="📘",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------ styling
st.markdown(
    """
    <style>
      .block-container {padding-top: 2rem;}
      .badge {display:inline-block;padding:3px 12px;border-radius:12px;
              font-size:0.8rem;font-weight:600;color:#fff;}
      .badge-HIGH{background:#1a7f37;} .badge-MEDIUM{background:#bf8700;}
      .badge-LOW{background:#b42318;}
      .cite-card{border:1px solid #e2e6ea;border-left:4px solid #4c6ef5;
                 border-radius:8px;padding:14px 16px;margin-bottom:12px;
                 background:#fafbfc;}
      .cite-title{font-weight:600;font-size:1.02rem;margin-bottom:2px;}
      .cite-crumb{color:#6b7280;font-size:0.8rem;margin-bottom:8px;}
      .cite-snip{color:#1f2937;font-size:0.9rem;}
      .small{color:#6b7280;font-size:0.82rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------------ loaders
@st.cache_data(show_spinner=False)
def load_metadata(meta_path_str):
    _mp = Path(meta_path_str)
    if not _mp.exists():
        return []
    return json.loads(_mp.read_text(encoding="utf-8"))


@st.cache_resource(show_spinner=True)
def load_pipeline(top_k: int, min_score: float, context_k: int,
                  use_llm: bool, model: str,
                  use_hybrid: bool, hybrid_alpha: float,
                  vectorstore_dir: str):
    from app.pipeline.pipeline import RAGPipeline
    return RAGPipeline(
        top_k=top_k, min_score=min_score, context_k=context_k,
        use_llm=use_llm, model=model,
        use_hybrid=use_hybrid, hybrid_alpha=hybrid_alpha,
        vectorstore_dir=vectorstore_dir,
    )


def load_golden_questions():
    if not GOLDEN_PATH.exists():
        return []
    try:
        return json.loads(GOLDEN_PATH.read_text(encoding="utf-8")).get("questions", [])
    except Exception:
        return []



def load_ask_history():
    if ASK_HISTORY.exists():
        try:
            return json.loads(ASK_HISTORY.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []


def log_ask(question, resp):
    hist = load_ask_history()
    hist.insert(0, {
        "question": question,
        "confidence": resp.get("confidence"),
        "answer_generated": resp.get("answer_generated"),
        "top_similarity": resp.get("metrics", {}).get("top_similarity"),
        "unique_pages": resp.get("metrics", {}).get("unique_pages"),
        "retrieved_pages": [c["page_id"] for c in resp.get("citations", [])],
        "retrieved_titles": [c["title"] for c in resp.get("citations", [])],
    })
    ASK_HISTORY.parent.mkdir(parents=True, exist_ok=True)
    ASK_HISTORY.write_text(json.dumps(hist[:200], indent=2, ensure_ascii=False),
                           encoding="utf-8")


def add_to_golden(question, page_id, keywords=None):
    """Append a labelled question to the golden set (dedup by question+page)."""
    data = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    qs = data.get("questions", [])
    for q in qs:
        if (q["question"].strip().lower() == question.strip().lower()
                and page_id in q.get("relevant_page_ids", [])):
            return False  # already present
    nid = f"ask{sum(1 for q in qs if str(q.get('id','')).startswith('ask'))+1:03d}"
    qs.append({
        "id": nid,
        "question": question.strip(),
        "relevant_page_ids": [page_id],
        "answer_keywords": keywords or [],
        "tag": "from_ask",
    })
    data["questions"] = qs
    GOLDEN_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False),
                           encoding="utf-8")
    return True


has_key = bool(config.OPENAI_API_KEY)

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown("## 📘 Confluence RAG")
    st.caption("Ask questions over your Confluence space with grounded, cited answers.")

    # ---- collection picker: choose which knowledge base to query ----
    from app.core import collections as _col
    _all = _col.list_collections()
    _cols = [c for c in _all if c["has_index"]] or _all
    _names = [c["name"] for c in _cols]
    _labels = {c["name"]: f"{(c.get('title') or c['name'])}  [{c['source']}]" for c in _cols}
    _sel = st.selectbox("Knowledge base", _names,
                        format_func=lambda n: _labels.get(n, n),
                        help="Each collection is an isolated index + golden set.")
    set_active_collection(_sel)
    meta = load_metadata(str(META_PATH))
    index_ready = META_PATH.exists() and (VSTORE / "faiss.index").exists()
    st.caption(f"Collection: **{_sel}**  ·  golden: "
               f"{'yes' if GOLDEN_PATH.exists() else 'none'}")
    st.divider()

    st.markdown("### ⚙️ Retrieval settings")
    top_k = st.slider("Candidates retrieved (top_k)", 5, 40, 20, 1)
    context_k = st.slider("Chunks sent to the LLM", 1, 10, 5, 1)
    min_score = st.slider("Min rerank score", 0.0, 1.0, 0.20, 0.05)

    st.markdown("### \U0001F517 Hybrid retrieval")
    use_hybrid = st.toggle("Hybrid (BM25 + dense)", value=True)
    hybrid_alpha = st.slider("Dense weight (α)", 0.0, 1.0, 0.5, 0.05,
                             disabled=not use_hybrid,
                             help="α = dense weight; (1−α) = BM25 weight")

    st.markdown("### 🤖 Generation")
    model = st.selectbox("OpenAI model",
                         ["gpt-4.1-mini", "gpt-4.1", "gpt-4o-mini", "gpt-4o"], 0)
    use_llm = st.toggle("Use LLM to write the answer", value=has_key,
                        disabled=not has_key)
    if not has_key:
        st.info("No OpenAI key found — running in **retrieval-only** mode. "
                "Citations still work.", icon="ℹ️")

    st.divider()
    st.markdown("### 📦 Status")
    st.write(f"Index loaded: {'✅' if index_ready else '❌'}")
    st.write(f"Chunks: **{len(meta)}**  |  Pages: **{len({m['page_id'] for m in meta}) if meta else 0}**")
    st.write(f"Embeddings: `all-MiniLM-L6-v2` (384-d)")
    st.write(f"OpenAI key: {'✅ detected' if has_key else '➖ not set'}")

if not index_ready:
    st.error("Vector store not found. Build it first: run the ingestion pipeline, "
             "then `python scripts/generate_embeddings.py` and "
             "`python scripts/build_vector_store.py`.")
    st.stop()

tab_ask, tab_eval, tab_corpus, tab_arch, tab_add = st.tabs(
    ["🔍 Ask", "📊 Evaluation", "📚 Corpus", "🏗️ Architecture", "➕ Add source"]
)

# ================================================================== ASK TAB
with tab_ask:
    st.markdown("### Ask a question")

    golden = load_golden_questions()
    samples = [q["question"] for q in golden[:6]] or [
        "What is the API rate limit?",
        "How do I roll back a deployment?",
    ]

    if "question" not in st.session_state:
        st.session_state.question = ""

    st.caption("Try one:")
    cols = st.columns(3)
    for i, s_q in enumerate(samples):
        if cols[i % 3].button(s_q, key=f"sample_{i}", use_container_width=True):
            st.session_state.question = s_q

    question = st.text_input("Your question", key="question",
                             placeholder="e.g. How do I authenticate with the REST API?")
    ask = st.button("Ask", type="primary")

    if ask and question.strip():
        try:
            pipe = load_pipeline(top_k, min_score, context_k, use_llm, model,
                                 use_hybrid, hybrid_alpha, str(VSTORE))
        except Exception as exc:  # noqa: BLE001
            st.exception(exc)
            st.stop()
        with st.spinner("Retrieving and generating\u2026"):
            resp = pipe.answer(question.strip())
        st.session_state.last_resp = resp
        st.session_state.last_question = question.strip()
        log_ask(question.strip(), resp)
    elif ask:
        st.warning("Please enter a question.")

    # ---- render from session so the citation buttons survive reruns ----
    resp = st.session_state.get("last_resp")
    asked_q = st.session_state.get("last_question", "")
    if resp:
        conf = resp["confidence"]
        st.markdown(
            f"#### Answer &nbsp; <span class='badge badge-{conf}'>{conf} confidence</span>",
            unsafe_allow_html=True,
        )
        if not resp["answer_generated"] and resp.get("error"):
            st.warning(f"Generation issue: {resp['error']}")
        st.markdown(resp["answer"])

        m = resp["metrics"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Top similarity", f"{m['top_similarity']:.3f}")
        c2.metric("Chunks used", m["chunks_used"])
        c3.metric("Retrieval", f"{m['retrieval_time_ms']:.0f} ms")
        c4.metric("Generation", f"{m['generation_time_ms']:.0f} ms")

        # ---- retrieval diagnostics: same vocabulary as the Evaluation page ----
        cites = resp["citations"]
        with st.expander("\U0001F52C Retrieval diagnostics (this query)", expanded=False):
            sims = [c["similarity"] for c in cites] or [0.0]
            d1, d2, d3, d4 = st.columns(4)
            d1.metric("Chunks retrieved", m["chunks_retrieved"])
            d2.metric("Unique pages", m["unique_pages"])
            d3.metric("Avg similarity", f"{m['avg_similarity']:.3f}")
            d4.metric("Min similarity", f"{min(sims):.3f}")
            if use_hybrid:
                bm = [c for c in cites if c.get("bm25_score", 0) > 0]
                st.caption(
                    f"Hybrid ON (\u03b1={hybrid_alpha:.2f}): "
                    f"{len(bm)}/{len(cites)} cited chunks had a BM25 keyword match. "
                    "Similarity shown is raw cosine; ranking uses the fused + reranked score."
                )
            else:
                st.caption("Hybrid OFF: dense-only retrieval, then metadata rerank.")
            st.dataframe(
                pd.DataFrame([{
                    "#": c["index"], "page_id": c["page_id"], "title": c["title"],
                    "cosine": c["similarity"], "bm25": c.get("bm25_score", 0.0),
                    "rerank": c["rerank_score"],
                } for c in cites]),
                hide_index=True, use_container_width=True,
            )
            st.caption("Precision@k / Hit@k / MRR need a known-correct page, so they live "
                       "on the Evaluation page. Mark a citation below as correct to add this "
                       "question to the golden set \u2014 then it becomes scorable there.")

        st.markdown("#### \U0001F4CE Citations")
        st.caption("Exact chunks the answer is grounded in \u2014 highest relevance first. "
                   "Use \u201cMark correct\u201d to feed the golden set.")
        for c in cites:
            with st.container():
                st.markdown(
                    f"<div class='cite-card'>"
                    f"<div class='cite-title'>[{c['index']}] {c['title']}"
                    f" &nbsp;\u00b7&nbsp; <span class='small'>{c['section'] or '\u2014'}</span></div>"
                    f"<div class='cite-crumb'>{c['breadcrumb'] or ''}</div>"
                    f"<div class='cite-snip'>{c['snippet']}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
                b1, b2, b3, b4 = st.columns([1, 1, 1.4, 1.4])
                b1.metric("Cosine", f"{c['similarity']:.3f}")
                b2.metric("Rerank", f"{c['rerank_score']:.3f}")
                if c.get("bm25_score", 0) > 0:
                    b3.metric("BM25", f"{c['bm25_score']:.2f}")
                elif c["source_url"]:
                    b3.markdown(
                        f"<a href='{c['source_url']}' target='_blank'>Open \u2197</a>",
                        unsafe_allow_html=True,
                    )
                if b4.button("\u2713 Mark correct", key=f"mark_{c['chunk_id']}",
                             help="Add this question with this page as the correct answer "
                                  "to the golden set"):
                    if asked_q:
                        added = add_to_golden(asked_q, c["page_id"])
                        if added:
                            st.success(f"Added to golden set \u2192 '{c['title']}' "
                                       f"is the answer for this question. "
                                       "Re-run the Evaluation to score it.")
                        else:
                            st.info("Already in the golden set.")
                with st.expander(f"Score breakdown \u2014 chunk {c['chunk_id']}"):
                    bd = c.get("score_breakdown", {})
                    if bd:
                        st.dataframe(
                            pd.DataFrame([{"signal": k, "contribution": round(v, 4)}
                                          for k, v in bd.items()]),
                            hide_index=True, use_container_width=True,
                        )
                    else:
                        st.caption("No breakdown available.")

    # ---- ask history (persisted) ----
    hist = load_ask_history()
    if hist:
        with st.expander(f"\U0001F553 Ask history ({len(hist)})", expanded=False):
            st.dataframe(
                pd.DataFrame([{
                    "question": h["question"],
                    "confidence": h.get("confidence"),
                    "top_sim": h.get("top_similarity"),
                    "pages": ", ".join(h.get("retrieved_pages", [])[:3]),
                } for h in hist[:50]]),
                hide_index=True, use_container_width=True,
            )

# ============================================================ EVALUATION TAB
with tab_eval:
    st.markdown("### Retrieval evaluation")
    _gsize = len(load_golden_questions())
    _from_ask = sum(1 for q in load_golden_questions() if q.get("tag") == "from_ask")
    gc1, gc2 = st.columns([1, 3])
    gc1.metric("Golden set: questions", _gsize)
    gc2.caption(
        f"The evaluation scores these {_gsize} labelled questions only "
        f"({_from_ask} added via \u201cMark correct\u201d on the Ask page). "
        "Asking a question on the Ask page does NOT change this number or the "
        "metrics \u2014 only marking it correct does, and then re-running."
    )

    with st.expander("\U0001F3D7\uFE0F Build / grow this collection's golden set",
                     expanded=(_gsize == 0)):
        st.caption("A new collection needs its own labelled questions before it can be "
                   "scored. Generate candidates, review, and approve them into the golden set.")
        bc1, bc2 = st.columns(2)
        if bc1.button("Auto-generate candidates (no LLM)"):
            from app.evaluation.golden_bootstrap import auto_bootstrap
            st.session_state.gold_cands = auto_bootstrap(_sel, per_page=2)
        if bc2.button("LLM-generate candidates", disabled=not has_key,
                      help=None if has_key else "Needs an OpenAI key"):
            from app.evaluation.golden_bootstrap import llm_bootstrap
            with st.spinner("Asking the model to draft questions per page\u2026"):
                try:
                    st.session_state.gold_cands = llm_bootstrap(_sel, model=model, per_page=2)
                except Exception as exc:  # noqa: BLE001
                    st.warning(f"LLM bootstrap failed: {exc}")

        cands = st.session_state.get("gold_cands")
        if cands:
            st.caption(f"{len(cands)} candidates \u2014 untick any you don't want, then approve.")
            edit_df = pd.DataFrame([{
                "approve": True, "question": c["question"],
                "page": c["relevant_page_ids"][0],
                "keywords": ", ".join(c.get("answer_keywords", [])),
            } for c in cands])
            edited = st.data_editor(edit_df, hide_index=True, use_container_width=True,
                                    disabled=["question", "page", "keywords"],
                                    key="gold_editor")
            if st.button("\u2713 Add approved to golden set", type="primary"):
                from app.evaluation.golden_bootstrap import approve_into_golden
                approved = []
                for c, keep in zip(cands, edited["approve"].tolist()):
                    if keep:
                        approved.append(c)
                n = approve_into_golden(_sel, approved)
                st.success(f"Added {n} question(s) to '{_sel}' golden set. "
                           "Re-run the evaluation to score them.")
                st.session_state.pop("gold_cands", None)
    st.caption(
        "Runs the golden question set through the live retriever and scores "
        "page-level ranking. No LLM required."
    )

    with st.expander("How to read these metrics", expanded=False):
        st.markdown(
            "- **Precision@k** — of the top-k pages retrieved, how many are relevant. "
            "Most questions here have a *single* relevant page, so the ceiling for "
            "Precision@3 is 0.33 and for Precision@5 is 0.20. High **Recall/Hit/MRR** "
            "with modest Precision@k is the *expected, healthy* pattern for this corpus.\n"
            "- **Recall@k** — of all relevant pages, how many appear in the top-k.\n"
            "- **Hit@k** — did at least one relevant page appear in the top-k.\n"
            "- **MRR** — 1/rank of the first relevant page, averaged.\n"
            "- **MAP** — mean average precision across all relevant pages."
        )

    run_col, load_col = st.columns([1, 1])
    run_eval = run_col.button("▶ Run evaluation (live)", type="primary")
    if load_col.button("📁 Load last saved results"):
        if RESULTS_PATH.exists():
            st.session_state.eval_report = json.loads(
                RESULTS_PATH.read_text(encoding="utf-8"))
        else:
            st.warning("No saved results yet — run the evaluation first.")

    if run_eval:
        try:
            pipe = load_pipeline(top_k, min_score, context_k, use_llm, model, use_hybrid, hybrid_alpha, str(VSTORE))
            from app.evaluation.evaluator import RetrievalEvaluator
            evaluator = RetrievalEvaluator(retriever=pipe.retriever, golden_path=str(GOLDEN_PATH))
            with st.spinner("Scoring golden questions…"):
                st.session_state.eval_report = evaluator.run(save=True)
        except Exception as exc:  # noqa: BLE001
            st.exception(exc)

    report = st.session_state.get("eval_report")
    if report:
        agg = report["aggregate"]
        st.markdown("#### Aggregate")
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Precision@3", f"{agg.get('precision@3', 0):.3f}")
        c2.metric("Precision@5", f"{agg.get('precision@5', 0):.3f}")
        c3.metric("Recall@5", f"{agg.get('recall@5', 0):.3f}")
        c4.metric("Hit@1", f"{agg.get('hit@1', 0):.3f}")
        c5.metric("MRR", f"{agg.get('mrr', 0):.3f}")
        c6.metric("MAP", f"{agg.get('ap', 0):.3f}")
        if "context_sufficiency@5" in agg:
            st.metric("Context sufficiency@5 (answer-groundedness proxy)",
                      f"{agg['context_sufficiency@5']:.3f}")

        ks = report["k_values"]
        chart_df = pd.DataFrame(
            {
                "Precision": [agg.get(f"precision@{k}", 0) for k in ks],
                "Recall": [agg.get(f"recall@{k}", 0) for k in ks],
                "Hit": [agg.get(f"hit@{k}", 0) for k in ks],
            },
            index=[f"@{k}" for k in ks],
        )
        st.markdown("#### Metrics by k")
        st.bar_chart(chart_df)

        st.markdown("#### Per-question results")
        rows = []
        for q in report["per_query"]:
            rows.append({
                "id": q["id"],
                "question": q["question"],
                "relevant": ", ".join(q["relevant_page_ids"]),
                "top-5 retrieved": ", ".join(q["retrieved_top5"]),
                "Hit@3": "✅" if q["metrics"].get("hit@3") else "❌",
                "P@3": round(q["metrics"].get("precision@3", 0), 2),
                "MRR": round(q["metrics"].get("mrr", 0), 2),
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    else:
        st.info("Run the evaluation to see Precision@3, Precision@5, Recall, Hit@k, MRR and MAP.")

    st.divider()
    st.markdown("#### Dense vs Hybrid comparison")
    st.caption("Runs the golden set through both retrievers and shows the delta. "
               "Hybrid adds BM25 candidates on top of dense; biggest gains are on exact-term questions.")
    if st.button("⚖️ Compare dense vs hybrid"):
        try:
            from app.retrieval.retriever import Retriever
            from app.evaluation.comparison import run_comparison
            with st.spinner("Running both retrievers over the golden set…"):
                dense_r = Retriever(top_k=top_k, min_score=0.0, use_hybrid=False, vectorstore_dir=str(VSTORE))
                hyb_r = Retriever(top_k=top_k, min_score=0.0, use_hybrid=True,
                                  hybrid_alpha=hybrid_alpha, vectorstore_dir=str(VSTORE))
                st.session_state.cmp = run_comparison(
                    dense_retriever=dense_r, hybrid_retriever=hyb_r,
                    hybrid_alpha=hybrid_alpha, golden_path=str(GOLDEN_PATH), save=True)
        except Exception as exc:  # noqa: BLE001
            st.exception(exc)

    cmp = st.session_state.get("cmp")
    if cmp:
        order = ["precision@3", "precision@5", "recall@5", "hit@1", "hit@3",
                 "mrr", "ap", "context_sufficiency@5"]
        rows = [{"metric": m, "dense": cmp["dense"].get(m, 0.0),
                 "hybrid": cmp["hybrid"].get(m, 0.0),
                 "delta": cmp["delta"].get(m, 0.0)} for m in order
                if m in cmp["dense"] or m in cmp["hybrid"]]
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        chart_df = pd.DataFrame(
            {"dense": [cmp["dense"].get(m, 0.0) for m in order],
             "hybrid": [cmp["hybrid"].get(m, 0.0) for m in order]},
            index=order)
        st.bar_chart(chart_df)
        st.caption("Per-question movement (MRR / Precision@3 / context-sufficiency):")
        st.dataframe(pd.DataFrame(cmp["per_query"]), hide_index=True,
                     use_container_width=True)

# ================================================================ CORPUS TAB
with tab_corpus:
    st.markdown("### Knowledge base")
    st.caption("The Confluence pages currently indexed.")

    by_page: dict[str, list] = {}
    for m in meta:
        by_page.setdefault(m["page_id"], []).append(m)

    rows = []
    for pid, chunks in sorted(by_page.items()):
        first = chunks[0]
        rows.append({
            "page_id": pid,
            "title": first.get("title", ""),
            "status": first.get("status", ""),
            "owner": first.get("owner", ""),
            "chunks": len(chunks),
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    st.markdown("#### Inspect chunks")
    titles = {f"{c[0].get('title','')} ({pid})": pid
              for pid, c in sorted(by_page.items(), key=lambda x: x[1][0].get('title',''))}
    choice = st.selectbox("Choose a page", list(titles.keys()))
    if choice:
        pid = titles[choice]
        for ch in by_page[pid]:
            with st.expander(f"{ch.get('chunk_id')} · {ch.get('section','')}"):
                st.markdown(ch.get("text", ""))
                if ch.get("source_url"):
                    st.markdown(f"[Open page ↗]({ch['source_url']})")

# ============================================================ ARCHITECTURE TAB
with tab_arch:
    st.markdown("### How it works")
    st.graphviz_chart(
        """
        digraph {
          rankdir=LR; node [shape=box style=rounded fontname=Helvetica fontsize=10];
          Confluence -> Ingestion -> Clean -> Chunk -> Embed -> FAISS;
          Query -> Retrieve; FAISS -> Retrieve;
          Retrieve -> Rerank -> Context -> LLM -> Answer;
          Rerank -> Citations; Citations -> Answer;
          Golden -> Evaluate; Retrieve -> Evaluate; Evaluate -> Metrics;
        }
        """,
        use_container_width=True,
    )
    st.markdown(
        "1. **Ingestion** pulls pages via the Confluence REST API and preserves the hierarchy.\n"
        "2. **Clean → Chunk** converts storage HTML to Markdown and splits it into "
        "section-aware, metadata-rich chunks.\n"
        "3. **Embed** encodes each chunk with `all-MiniLM-L6-v2` (normalized, 384-d) and "
        "stores vectors in a **FAISS** inner-product index (= cosine similarity).\n"
        "4. **Retrieve → Rerank** does semantic search, then a metadata-aware reranker "
        "boosts title/section/heading/keyword/exact-phrase matches.\n"
        "5. **LLM** answers strictly from the retrieved context; **Citations** expose the "
        "exact chunks and scores behind every answer.\n"
        "6. **Evaluation** scores retrieval against a golden question set "
        "(Precision@k, Recall@k, Hit@k, MRR, MAP)."
    )

# ============================================================ ADD SOURCE TAB
with tab_add:
    st.markdown("### Add a knowledge base")
    st.caption("Register a Confluence space or a MediaWiki source, then ingest it "
               "into its own isolated index. Credentials are read from the server's "
               ".env \u2014 they are never entered here.")

    from app.core import collections as _colmod

    existing = {c["name"] for c in _colmod.list_collections()}

    c1, c2 = st.columns(2)
    new_name = c1.text_input("Collection name (no spaces)", key="add_name",
                             placeholder="e.g. client_x")
    new_title = c2.text_input("Display title", key="add_title",
                              placeholder="e.g. Client X Space")
    src_type = st.radio("Source type", ["confluence", "mediawiki"], horizontal=True,
                        key="add_src")

    cfg_kwargs = {}
    if src_type == "confluence":
        cc1, cc2 = st.columns(2)
        cfg_kwargs["confluence_base_url"] = cc1.text_input(
            "Confluence base URL", placeholder="https://your-site.atlassian.net")
        cfg_kwargs["confluence_parent_page_id"] = cc2.text_input(
            "Parent page ID", placeholder="e.g. 65710")
        cfg_kwargs["token_env"] = st.text_input(
            "Token env-var name (read from .env)", value="CONFLUENCE_API_TOKEN")
        st.info("The API token is read from this environment variable in the "
                "server's .env \u2014 it is not entered or stored here.", icon="\U0001F512")
    else:
        wc1, wc2 = st.columns(2)
        cfg_kwargs["wiki_api_url"] = wc1.text_input(
            "Wiki API URL", value="https://en.wikipedia.org/w/api.php",
            help="Always <wiki>/w/api.php. For Wikipedia it's this value.")
        cfg_kwargs["wiki_base_url"] = wc2.text_input(
            "Wiki base URL (optional)", placeholder="https://en.wikipedia.org/wiki")

        wiki_mode = st.radio(
            "What to ingest", ["Category (a theme / many pages)",
                               "Specific page(s) by title"], horizontal=True,
            help="Category pulls all pages in a Wikipedia category. "
                 "Specific pages ingests only the exact titles you list.")
        if wiki_mode.startswith("Category"):
            wc3, wc4 = st.columns([2, 1])
            cfg_kwargs["wiki_category"] = wc3.text_input(
                "Category", placeholder="Category:Machine_learning",
                help="Find it at the bottom of any Wikipedia article under 'Categories'.")
            cfg_kwargs["wiki_page_limit"] = int(wc4.number_input("Page limit", 1, 500, 15))
        else:
            titles = st.text_area(
                "Page titles (one per line, or comma-separated)",
                placeholder="Snowflake Inc.\nData build tool\nSnapLogic",
                help="Use the exact article title as it appears on Wikipedia.")
            parsed = [t.strip() for t in titles.replace(",", "\n").splitlines() if t.strip()]
            cfg_kwargs["wiki_titles"] = ", ".join(parsed)
            cfg_kwargs["wiki_page_limit"] = max(len(parsed), 1)
            if parsed:
                st.caption(f"Will ingest {len(parsed)} page(s): {', '.join(parsed[:5])}"
                           + ("\u2026" if len(parsed) > 5 else ""))

    valid_name = bool(new_name) and new_name.replace("_", "").isalnum() and new_name != "default"

    b1, b2 = st.columns([1, 2])
    if b1.button("1\uFE0F\u20E3 Register", disabled=not valid_name):
        if new_name in existing:
            st.warning(f"'{new_name}' already exists.")
        else:
            clean = {k: v for k, v in cfg_kwargs.items() if v}
            _colmod.register_collection(new_name, src_type, title=new_title or new_name,
                                        **clean)
            st.success(f"Registered '{new_name}'. Now use ‘Ingest & prepare’ below.")

    # ---- Ingest & prepare: ONE step = ingest + golden set (no terminal) ----
    st.markdown("#### Ingest & prepare")
    ing_target = st.selectbox(
        "Collection to ingest",
        [c["name"] for c in _colmod.list_collections() if c["name"] != "default"] or ["\u2014"],
        help="Registered collections appear here. This builds the index AND a starter golden set.")
    gmethod = st.radio(
        "Starter golden set", ["auto", "llm", "none"], horizontal=True,
        format_func=lambda m: {"auto": "Auto (no LLM)", "llm": "LLM (needs key)",
                               "none": "Skip"}[m],
        help="auto = instant, generic questions \u00b7 llm = sharper, needs an OpenAI key \u00b7 skip = none")
    if gmethod == "llm" and not has_key:
        st.caption("\u26A0\uFE0F No OpenAI key detected \u2014 will fall back to auto.")

    if st.button("\u25B6 Ingest & prepare", type="primary",
                 disabled=(ing_target in ("", "\u2014"))):
        from app.core.ingest_runner import stream_ingest
        log_box = st.empty(); lines = []; ok = False
        try:
            with st.spinner(f"Ingesting '{ing_target}' \u2014 this can take a few minutes\u2026"):
                for line in stream_ingest(ing_target):
                    lines.append(line)
                    log_box.code("\n".join(lines[-18:]), language="text")
            ok = True
        except Exception as exc:  # noqa: BLE001
            log_box.code("\n".join(lines[-30:]), language="text")
            st.error(f"Ingestion failed: {exc}")

        if ok and gmethod != "none":
            try:
                from app.evaluation.golden_bootstrap import (
                    auto_bootstrap, llm_bootstrap, approve_into_golden)
                method = "auto" if (gmethod == "llm" and not has_key) else gmethod
                with st.spinner(f"Building starter golden set ({method})\u2026"):
                    cands = (llm_bootstrap(ing_target, model=model)
                             if method == "llm" else auto_bootstrap(ing_target))
                    n = approve_into_golden(ing_target, cands)
                st.success(f"\u2705 '{ing_target}' is ready \u2014 indexed and {n} golden "
                           "questions added. Select it in the sidebar to query it.")
            except Exception as exc:  # noqa: BLE001
                # ingest succeeded; golden set is optional \u2014 don't fail the whole thing
                st.success(f"\u2705 '{ing_target}' ingested and indexed. "
                           "Select it in the sidebar to query it.")
                st.warning(f"Starter golden set was skipped ({exc}). "
                           "You can build one anytime in the Evaluation tab.")
        elif ok:
            st.success(f"\u2705 '{ing_target}' ingested and indexed. "
                       "Select it in the sidebar to query it.")

    st.divider()
    st.markdown("#### Registered collections")
    st.dataframe(
        pd.DataFrame([{
            "name": c["name"], "source": c["source"],
            "indexed": "\u2705" if c["has_index"] else "\u2014",
            "golden": "\u2705" if c["has_golden"] else "\u2014",
            "title": c.get("title", ""),
        } for c in _colmod.list_collections()]),
        hide_index=True, use_container_width=True)