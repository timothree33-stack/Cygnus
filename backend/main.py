# Minimal backend main shim for dev use
from .debate.orchestrator import DebateOrchestrator
from .mcp.hiemdall import snapshot_text, score_pair
import asyncio

class _StubAgent:
    def __init__(self, name: str):
        self.name = name

    async def respond(self, **kwargs):
        # Simple deterministic placeholder response for dev/testing
        topic = kwargs.get('topic', '(topic)')
        round_num = kwargs.get('round', '?')
        return f"{self.name} responds to '{topic}' (round {round_num})"

async def _broadcast(msg: dict):
    # Simple broadcast stub: print to stdout (uvicorn logs capture this)
    print('BROADCAST:', msg)

# Single shared orchestrator instance used by API routes (dev-friendly)
katz = _StubAgent('Katz')
dogz = _StubAgent('Dogz')
cygnus = _StubAgent('Cygnus')
memory = None

orchestrator = DebateOrchestrator(katz, dogz, cygnus, memory, snapshot_text, score_pair, _broadcast)
