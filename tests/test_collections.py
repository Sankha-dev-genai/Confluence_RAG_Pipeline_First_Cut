import json
from app.core import collections as col
from app.core.config import collection_paths


def test_default_paths_are_legacy():
    p = collection_paths("default")
    assert p.vectorstore.as_posix().endswith("data/vectorstore")


def test_named_paths_isolated():
    p = collection_paths("acme")
    assert p.vectorstore.as_posix().endswith("data/collections/acme/vectorstore")
    assert p.golden.as_posix().endswith("data/collections/acme/evaluation/golden_qa.json")


def test_register_list_remove(tmp_path, monkeypatch):
    # register a wiki collection, confirm it lists, then remove
    col.register_collection("t_wiki", "mediawiki", title="TW",
                            wiki_api_url="https://x/w/api.php", wiki_page_limit=5)
    names = [c["name"] for c in col.list_collections()]
    assert "default" in names and "t_wiki" in names
    col.remove_collection("t_wiki")
    assert "t_wiki" not in [c["name"] for c in col.list_collections()]


def test_apply_source_config():
    from app.core.config import Settings
    s = Settings()
    entry = {"source": "mediawiki", "wiki_api_url": "https://x/w/api.php",
             "wiki_category": "Category:Foo"}
    col.apply_source_config(entry, s)
    assert s.source == "mediawiki" and s.wiki_category == "Category:Foo"
