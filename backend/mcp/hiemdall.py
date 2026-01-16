"""Hiemdall helpers: snapshot and scoring utilities.
These are small, testable functions that persist a snapshot into memory store and compute
"difference" between two statements to derive unique scores.
"""
from typing import Tuple
import hashlib
import os
from ..embeddings import get_embedder
import math

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


def _cosine(u, v):
    if not u or not v:
        return 0.0
    num = sum(x * y for x, y in zip(u, v))
    denom = math.sqrt(sum(x * x for x in u)) * math.sqrt(sum(y * y for y in v))
    if denom == 0:
        return 0.0
    return num / denom


def embedding_difference_score(a: str, b: str) -> float:
    try:
        emb = get_embedder()
        va = emb.embed(a)
        vb = emb.embed(b)
        # similarity in [-1,1], convert to difference 0..1
        sim = _cosine(va, vb)
        diff = 1.0 - max(-1.0, min(1.0, sim))
        return diff
    except Exception:
        # fallback to text diff
        return text_difference_score(a, b)


def score_pair(a: str, b: str, *_, strategy: str = None) -> Tuple[int, int]:
    """Compute two unique scores 1-100 for statements a and b.
    strategy: 'text'|'embed'|'hybrid' or environment variable SCORING_STRATEGY.
    Accepts extra args (debate_id, round) for deterministic tie-breaking when provided.
    """
    strat = (strategy or os.environ.get('SCORING_STRATEGY') or 'text').lower()

    if strat == 'embed':
        da = embedding_difference_score(a, b)
        db = embedding_difference_score(b, a)
    elif strat == 'hybrid':
        ta = text_difference_score(a, b)
        tb = text_difference_score(b, a)
        ea = embedding_difference_score(a, b)
        eb = embedding_difference_score(b, a)
        # simple average
        da = (ta + ea) / 2.0
        db = (tb + eb) / 2.0
    else:
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
