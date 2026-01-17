import os
from backend.mcp.hiemdall import score_pair


def test_text_strategy():
    os.environ.pop('SCORING_STRATEGY', None)
    sa, sb = score_pair('hello world', 'goodbye world')
    assert isinstance(sa, int) and 1 <= sa <= 100


def test_embed_strategy():
    os.environ['SCORING_STRATEGY'] = 'embed'
    sa, sb = score_pair('hello world', 'hello world')
    # identical inputs should be low difference -> low scores, but at least >=1
    assert 1 <= sa <= 100


def test_hybrid_strategy():
    os.environ['SCORING_STRATEGY'] = 'hybrid'
    sa, sb = score_pair('the quick brown fox', 'the quick brown fox jumps')
    assert 1 <= sa <= 100
