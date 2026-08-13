"""Deterministic config tests — exercise Settings directly, no ambient .env."""
import pytest

from app.core.config import Settings
from app.core.exceptions import ConfigError


def test_defaults():
    s = Settings()
    assert s.source == "confluence"
    assert s.openai_model == "gpt-4.1-mini"
    assert s.top_k == 20


def test_require_confluence_raises_when_missing():
    with pytest.raises(ConfigError):
        Settings().require_confluence()


def test_require_confluence_ok():
    Settings(
        confluence_base_url="https://x.atlassian.net",
        confluence_email="[email protected]",
        confluence_api_token="token",
        confluence_parent_page_id="1",
    ).require_confluence()  # should not raise


def test_require_wiki():
    with pytest.raises(ConfigError):
        Settings(source="mediawiki").require_wiki()
    Settings(source="mediawiki",
             wiki_api_url="https://en.wikipedia.org/w/api.php").require_wiki()


def test_require_source_dispatch():
    with pytest.raises(ConfigError):
        Settings(source="sharepoint").require_source()


def test_require_openai():
    with pytest.raises(ConfigError):
        Settings().require_openai()
    Settings(openai_api_key="sk-x").require_openai()


def test_from_env_reads_model(monkeypatch):
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o")
    assert Settings.from_env().openai_model == "gpt-4o"
