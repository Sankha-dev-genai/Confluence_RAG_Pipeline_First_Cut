from app.evaluation import golden_bootstrap as gb


def test_auto_bootstrap_default():
    cands = gb.auto_bootstrap("default", per_page=2)
    assert len(cands) > 0
    for c in cands:
        assert c["question"] and c["relevant_page_ids"] and c["tag"] == "auto"


def test_approve_dedup(tmp_path, monkeypatch):
    # approving the same candidate twice adds it only once
    import app.core.config as config
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    # rebuild collection_paths to use tmp
    q = [{"question": "Q1?", "relevant_page_ids": ["1"], "answer_keywords": [], "tag": "auto"}]
    n1 = gb.approve_into_golden("acme_test", q)
    n2 = gb.approve_into_golden("acme_test", q)
    assert n1 == 1 and n2 == 0
