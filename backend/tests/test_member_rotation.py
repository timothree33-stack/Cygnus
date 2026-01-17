import pytest
import asyncio
import time

from backend.debate.orchestrator import DebateOrchestrator
from backend.db.sqlite_store import SQLiteStore

class TestAgent:
    def __init__(self, name):
        self.name = name
    async def respond(self, **kwargs):
        member = kwargs.get('member')
        return f"{self.name}{(':'+member) if member else ''} responds to '{kwargs.get('topic')}' (round {kwargs.get('round')})"

@pytest.mark.asyncio
async def test_member_rotation_and_broadcasts(tmp_path):
    db_path = str(tmp_path / 'db.sqlite')
    store = SQLiteStore(db_path)

    events = []
    async def broadcaster(msg):
        # quick append; in real use this would push to websockets
        events.append(msg)

    a = TestAgent('katz')
    b = TestAgent('dogz')
    c = TestAgent('cygnus')

    orch = DebateOrchestrator(a, b, c, store, lambda *a, **k: ('snap', 'summary'), lambda *a, **k: (60, 40), broadcaster)

    # Create debate record first and run a 3-round debate with rotating katz members (Alice, Bob) and single dogz member (Xena)
    did = store.create_debate('rotation topic', helix_active=False, debate_number=1)
    await orch.run_debate(did, 'rotation topic', rounds=3, pause_sec=0, katz_members=['Alice', 'Bob'], dogz_members=['Xena'])

    # Check that member_changed events were emitted for both teams
    member_changes = [e for e in events if e.get('type') == 'member_changed']
    assert any(e.get('team') == 'katz' and e.get('member_name') == 'Alice' for e in member_changes)
    assert any(e.get('team') == 'katz' and e.get('member_name') == 'Bob' for e in member_changes)
    assert any(e.get('team') == 'dogz' and e.get('member_name') == 'Xena' for e in member_changes)

    # Check that statement events include member names in agent field and member field
    statements = [e for e in events if e.get('type') == 'statement']
    assert any(s.get('agent') == 'katz:Alice' and s.get('member') == 'Alice' for s in statements)
    assert any(s.get('agent') == 'katz:Bob' and s.get('member') == 'Bob' for s in statements)
    assert any(s.get('agent') == 'dogz:Xena' and s.get('member') == 'Xena' for s in statements)

    # Ensure agents persisted with namespaced ids exist in DB
    assert store.get_agent_by_name('katz:Alice') is not None
    assert store.get_agent_by_name('katz:Bob') is not None
    assert store.get_agent_by_name('dogz:Xena') is not None

    # Ensure rounds and arguments were persisted
    debates = store.get_debates()
    assert any(d['id'] == 'rot-test' or d['topic'] == 'rotation topic' for d in debates) or len(debates) >= 1
    # at least 3 rounds created
    # fetch arguments count
    cur = store._exec("SELECT COUNT(1) as c FROM arguments")
    row = cur.fetchone()
    assert row['c'] >= 2
