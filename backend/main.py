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

# We'll attach a WebSocketManager to the app state and pass its broadcast fn to orchestrator
from .ws_manager import WebSocketManager
ws_manager = WebSocketManager()

orchestrator = DebateOrchestrator(katz, dogz, cygnus, memory, snapshot_text, score_pair, ws_manager.broadcast)

# --- FastAPI app (so uvicorn backend.main:app works as expected) ---
from fastapi import FastAPI
from .api import admin_routes, debate_routes, ws_routes

app = FastAPI(title='Cygnus Backend (dev)')
# attach ws_manager to state for ws routes
app.state.ws_manager = ws_manager

app.include_router(admin_routes.router)
app.include_router(debate_routes.router)
app.include_router(ws_routes.router)

@app.get('/api/status')
async def status():
    return {'ok': True}
