import os
import pytest
from backend.embeddings import get_embedder


def test_default_is_stub(monkeypatch):
    monkeypatch.delenv('EMBEDDER_TYPE', raising=False)
    emb = get_embedder()
    v = emb.embed('hello')
    assert isinstance(v, list) and len(v) >= 1


def test_openai_missing(monkeypatch):
    # If OPENAI not configured, requesting openai should fall back to stub
    monkeypatch.setenv('EMBEDDER_TYPE', 'openai')
    # Ensure no OPENAI_API_KEY is set
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    emb = get_embedder()
    v = emb.embed('hello openai fallback')
    assert isinstance(v, list)

@pytest.mark.parametrize('a,b,expect_low', [
    ('same text', 'same text', True),
    ('different one', 'another very different', False),
])
def test_embedder_integration(monkeypatch, a, b, expect_low):
    # Use a deterministic stub embedder to validate embedding-based scoring paths
    monkeypatch.setenv('EMBEDDER_TYPE', 'stub')
    emb = get_embedder()
    va = emb.embed(a)
    vb = emb.embed(b)
    assert isinstance(va, list) and isinstance(vb, list)
    # identical inputs should give identical vector for stub
    if expect_low:
        assert va == vb
    else:
        assert va != vb
