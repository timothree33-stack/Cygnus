import asyncio
import pytest
from backend.debate.orchestrator import DebateOrchestrator

class DummyAgent:
    def __init__(self, name):
        self.name = name
    async def respond(self, topic=None, debate_id=None, round=None):
        return f"{self.name} on {topic} (round {round})"

async def dummy_broadcast(msg):
    # no-op
    return

def dummy_snapshot(agent_id, text, ts):
    return (f"snap-{agent_id}-{ts}", text[:50])

def dummy_score(a, b, debate_id, r):
    # trivial: length-based
    sa = min(100, max(1, len(a)%100))
    sb = min(100, max(1, len(b)%100))
    if sa==sb:
        sa = max(1, sa-1)
    return sa, sb

@pytest.mark.asyncio
async def test_run_debate_short():
    katz = DummyAgent('KatZ')
    dogz = DummyAgent('DogZ')
    cygnus = DummyAgent('Cygnus')
    # memory can be None for orchestrator unit test
    orch = DebateOrchestrator(katz, dogz, cygnus, None, dummy_snapshot, dummy_score, dummy_broadcast)
    state = await orch.run_debate('test-1', 'Is AI good?', rounds=1, pause_sec=0)
    assert state['round'] == 1
    assert len(state['history']) == 1
    assert 'katz' in state['history'][0]
