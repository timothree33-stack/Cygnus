import pytest
from backend.mcp.hiemdall import score_pair

def test_score_pair_different_statements():
    a = "Cats like to climb trees and nap in the sun."
    b = "Dogs prefer fetching and running outdoors."
    sa, sb = score_pair(a, b)
    assert 1 <= sa <= 100
    assert 1 <= sb <= 100
    assert sa != sb

def test_score_pair_similar_statements():
    a = "I like apples."
    b = "I like apples."
    sa, sb = score_pair(a, b)
    assert sa != sb  # deterministic perturbation should ensure uniqueness

