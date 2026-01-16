import pytest
import asyncio
from backend.debate.orchestrator import DebateOrchestrator
from backend.db.sqlite_store import SQLiteStore

class DummyAgent:
    def __init__(self, name):
        self.name = name
    async def respond(self, **kwargs):
        return f"{self.name} says ({kwargs.get('round')})"

@pytest.mark.asyncio
async def test_debate_persists(tmp_path):
    store = SQLiteStore(str(tmp_path / 'db.sqlite'))
    a = DummyAgent('katz')
    b = DummyAgent('dogz')
    c = DummyAgent('cygnus')
    orch = DebateOrchestrator(a, b, c, store, lambda *a, **k: ('s',), lambda *a: (50,49), lambda *a: None)

    # run one-round debate with minimal pause
    res = await orch.run_debate('db-test', 'a topic', rounds=1, pause_sec=0)
    # Check that debate record exists
    debates = store.get_debates()
    assert any(d['topic'] == 'a topic' for d in debates)

    # Check that round and arguments were persisted
    mems = store.get_memories()
    assert len(mems) >= 2  # snapshots were created
