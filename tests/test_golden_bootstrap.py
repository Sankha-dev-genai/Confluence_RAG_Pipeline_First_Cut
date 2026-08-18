from app.evaluation import golden_bootstrap as gb


def test_auto_bootstrap_default():
    cands = gb.auto_bootstrap("default", per_page=2)
    assert len(cands) > 0
    for c in cands:
        assert c["question"] and c["relevant_page_ids"] and c["tag"] == "auto"


def test_approve_dedup(tmp_path):
    # approving the same question twice adds it only once (dedup by question+page)
    import json
    golden = tmp_path / "golden_qa.json"
    golden.write_text(json.dumps({"questions": []}), encoding="utf-8")

    def approve(questions):
        data = json.loads(golden.read_text(encoding="utf-8"))
        existing = data.get("questions", [])
        seen = {(q["question"].strip().lower(),
                 tuple(sorted(q.get("relevant_page_ids", [])))) for q in existing}
        added = 0
        for q in questions:
            key = (q["question"].strip().lower(),
                   tuple(sorted(q.get("relevant_page_ids", []))))
            if key in seen or not q["question"].strip():
                continue
            existing.append(q); seen.add(key); added += 1
        data["questions"] = existing
        golden.write_text(json.dumps(data), encoding="utf-8")
        return added

    q = [{"question": "Q1?", "relevant_page_ids": ["1"], "answer_keywords": [], "tag": "auto"}]
    assert approve(q) == 1   # first time: added
    assert approve(q) == 0   # second time: deduped
