import pytest
from backend.db.sqlite_store import SQLiteStore


def test_get_memory_by_id():
    s = SQLiteStore(':memory:')
    mid = s.save_memory('agent-x', 'snapshot text for testing', embedding=None, source='test')
    cur = s._exec("SELECT id,content,agent_id,source FROM memories WHERE id = ?", (mid,))
    row = cur.fetchone()
    assert row is not None
    assert row['content'] == 'snapshot text for testing'
