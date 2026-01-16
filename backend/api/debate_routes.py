from fastapi import APIRouter, BackgroundTasks, HTTPException
from typing import Optional
import uuid
import os

from ..debate.orchestrator import DebateOrchestrator
from ..mcp.hiemdall import snapshot_text, score_pair

router = APIRouter(prefix="/api/debate")

# Very small in-memory map to track debates
DEBATES = {}

@router.post('/start')
async def start_debate(background_tasks: BackgroundTasks, topic: Optional[str] = None, rounds: int = 5, pause_sec: Optional[int] = None):
    if not topic:
        topic = "(random topic)"
    debate_id = str(uuid.uuid4())
    # Try to pull pause from env if not provided
    pause = int(os.environ.get('DEBATE_PAUSE_SEC', '60')) if pause_sec is None else pause_sec

    # Resolve orchestrator from main module to avoid circular imports
    try:
        from ..main import orchestrator
    except Exception:
        raise HTTPException(status_code=500, detail='orchestrator_unavailable')

    # run in background
    background_tasks.add_task(orchestrator.run_debate, debate_id, topic, rounds, pause)
    DEBATES[debate_id] = {'topic': topic, 'rounds': rounds, 'pause_sec': pause, 'state': 'started'}
    return {'debate_id': debate_id, 'status': 'started', 'topic': topic}

@router.get('/{debate_id}/state')
async def get_state(debate_id: str):
    s = DEBATES.get(debate_id)
    if not s:
        raise HTTPException(status_code=404, detail='debate not found')
    return s

@router.post('/{debate_id}/pause')
async def pause_debate(debate_id: str):
    # Not implemented: orchestrator would need pause hook
    if debate_id in DEBATES:
        DEBATES[debate_id]['state'] = 'paused'
        return {'paused': True}
    raise HTTPException(status_code=404, detail='debate not found')

@router.post('/{debate_id}/resume')
async def resume_debate(debate_id: str):
    if debate_id in DEBATES:
        DEBATES[debate_id]['state'] = 'running'
        return {'resumed': True}
    raise HTTPException(status_code=404, detail='debate not found')

@router.post('/{debate_id}/allcall')
async def allcall(debate_id: str):
    # Not implemented: switch orchestrator into AllCall mode
    if debate_id in DEBATES:
        DEBATES[debate_id]['state'] = 'allcall'
        return {'allcall': True}
    raise HTTPException(status_code=404, detail='debate not found')
