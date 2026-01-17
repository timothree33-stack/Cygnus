import pytest
import asyncio
from backend.debate.orchestrator import DebateOrchestrator

class DummyAgent:
    def __init__(self, name):
        self.name = name
    async def respond(self, **kwargs):
        return f"{self.name} response ({kwargs.get('round')})"

async def dummy_broadcast(msg):
    return

@pytest.mark.asyncio
async def test_run_allcall_once():
    a = DummyAgent('A')
    b = DummyAgent('B')
    orch = DebateOrchestrator(a, b, None, None, lambda *a: ('s1','s'), lambda *a: (50,49), dummy_broadcast)
    # Manually call run_allcall_round and assert result gets broadcast (no exceptions)
    await orch._run_allcall_round('deb-1', 'topic', 1)
    # If no exceptions, consider pass; ensure history updated if debate tracked
    orch._current_debates['deb-1'] = {'history': []}
    await orch._run_allcall_round('deb-1', 'topic', 2)
    assert len(orch._current_debates['deb-1']['history']) == 1
