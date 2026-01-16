"""Hiemdall helpers: snapshot and scoring utilities.
These are small, testable functions that persist a snapshot into memory store and compute
"difference" between two statements to derive unique scores.
"""
from typing import Tuple
import hashlib

# Basic text-difference metric (fallback if no embedder available)
def text_difference_score(a: str, b: str) -> float:
    # Simple normalized difference based on token sets
    sa = set(a.lower().split())
    sb = set(b.lower().split())
    if not sa and not sb:
        return 0.0
    inter = sa.intersection(sb)
    union = sa.union(sb)
    j = 0.0 if not union else 1.0 - (len(inter) / len(union))
    return j


def score_pair(a: str, b: str) -> Tuple[int, int]:
    """Compute two unique scores 1-100 for statements a and b, higher means more different.
    We measure difference of each from the other; normalize and map to 1-100, ensure uniqueness.
    """
    da = text_difference_score(a, b)
    db = text_difference_score(b, a)
    # normalize to 1-100
    sa = int(max(1, min(100, round(1 + da * 99))))
    sb = int(max(1, min(100, round(1 + db * 99))))
    if sa == sb:
        # deterministic perturbation: use hash of concatenation
        h = int(hashlib.sha256((a + b).encode()).hexdigest(), 16)
        if h % 2 == 0:
            sa = max(1, sa - 1)
        else:
            sb = max(1, sb - 1)
        if sa == sb:
            sa = max(1, sa - 1)
    return sa, sb


# Snapshot function: a stub that returns a synthetic snapshot id and summary
def snapshot_text(agent_id: str, text: str, timestamp: int):
    # In a full implementation this would persist into MCP and return memory id and a summary
    sid = hashlib.sha1(f"{agent_id}:{timestamp}:{text[:80]}".encode()).hexdigest()
    summary = text if len(text) < 200 else text[:197] + '...'
    return sid, summary
