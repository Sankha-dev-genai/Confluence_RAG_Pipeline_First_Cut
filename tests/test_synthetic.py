import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "gen", Path("scripts/generate_synthetic_corpus.py"))
gen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gen)

REQUIRED = {"chunk_id", "page_id", "title", "section", "text", "source_url",
            "breadcrumb", "chunk_number", "total_chunks"}


def test_service_schema_and_golden():
    page_id, chunks, md, questions = gen.build_service(3, "Cobalt")
    assert len(chunks) == 5
    for c in chunks:
        assert REQUIRED <= set(c), f"missing keys: {REQUIRED - set(c)}"
        assert c["page_id"] == page_id
    # all golden questions point at this page
    for q in questions:
        assert q["relevant_page_ids"] == [page_id]
    # exact-match questions must be NAME-FREE (that's what makes them dense-hard)
    for q in questions:
        if q["tag"] == "exact_match":
            assert "cobalt" not in q["question"].lower()
            assert any("ERR-" in k for k in q["answer_keywords"])


def test_unique_error_codes_across_services():
    _, c0, _, _ = gen.build_service(0, "Aurora")
    _, c1, _, _ = gen.build_service(1, "Basalt")
    err0 = [c for c in c0 if c["section"] == "Error Codes"][0]["text"]
    err1 = [c for c in c1 if c["section"] == "Error Codes"][0]["text"]
    assert err0 != err1   # different code ranges per service
