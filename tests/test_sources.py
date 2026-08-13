import pytest

from app.sources import get_source
from app.sources.base import KnowledgeSource, RawDocument
from app.core.exceptions import ConfigError


def test_factory_confluence():
    s = get_source("confluence")
    assert isinstance(s, KnowledgeSource)
    assert s.name == "confluence"


def test_factory_mediawiki():
    s = get_source("mediawiki")
    assert isinstance(s, KnowledgeSource)
    assert s.name == "mediawiki"


def test_factory_unknown():
    with pytest.raises(ConfigError):
        get_source("sharepoint")


def test_raw_document_sidecar_schema():
    doc = RawDocument(page_id="1", title="T", html="<p>x</p>",
                      url="http://x", source_type="mediawiki",
                      breadcrumb="Wiki > T", ancestor_titles=["Wiki"])
    side = doc.sidecar()
    for key in ("title", "url", "source_type", "breadcrumb",
                "ancestor_titles", "space_name"):
        assert key in side
    assert side["source_type"] == "mediawiki"
