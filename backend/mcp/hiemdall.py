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


def score_pair(a: str, b: str, *_) -> Tuple[int, int]:
    """Compute two unique scores 1-100 for statements a and b, higher means more different.
    Accepts extra args (debate_id, round) for deterministic tie-breaking when provided.
    """
    da = text_difference_score(a, b)
    db = text_difference_score(b, a)
    # normalize to 1-100
    sa = int(max(1, min(100, round(1 + da * 99))))
    sb = int(max(1, min(100, round(1 + db * 99))))
    if sa == sb:
        # deterministic perturbation: use hash of concatenation + optional context
        ctx = ''.join(map(str, _))
        h = int(hashlib.sha256((a + b + ctx).encode()).hexdigest(), 16)
        if h % 2 == 0:
            sa = max(1, sa - 1)
        else:
            sb = max(1, sb - 1)
        if sa == sb:
            sa = max(1, sa - 1)
    return sa, sb


# Snapshot function: a stub that returns a synthetic snapshot id and summary
def snapshot_text(agent_id: str, text: str, timestamp: int, debate_id: str = None):
    """Persist a snapshot into the SQLite store and return (id, summary)."""
    from ..db.sqlite_store import SQLiteStore
    store = SQLiteStore()
    summary = text if len(text) < 200 else text[:197] + '...'
    # Include debate id info in source so it can be queried later
    source = f"snapshot{':'+debate_id if debate_id else ''}"
    mid = store.save_memory(agent_id, summary, embedding=None, source=source)
    return mid, summary
