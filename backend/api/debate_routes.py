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

    # If an orchestrator instance is available, merge in live state (history, round, etc.)
    try:
        from ..main import orchestrator as orch
        if hasattr(orch, '_current_debates') and debate_id in orch._current_debates:
            merged = dict(s)
            merged.update(orch._current_debates[debate_id])
            return merged
    except Exception:
        pass

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
    # Trigger orchestrator to schedule an All Call for the given debate
    try:
        from ..main import orchestrator
        ok = orchestrator.trigger_allcall(debate_id)
        if ok:
            DEBATES.setdefault(debate_id, {})['state'] = 'allcall'
            return {'allcall': True}
        else:
            raise HTTPException(status_code=404, detail='debate not found')
    except Exception:
        raise HTTPException(status_code=500, detail='orchestrator_unavailable')


@router.post('/{debate_id}/camera-capture')
async def camera_capture(debate_id: str, round: Optional[int] = None, agent: Optional[str] = None, caption: Optional[str] = None):
    """Capture an image from a local camera (device 0), save via admin image endpoint, persist a snapshot memory and broadcast the event.

    CI-friendly: if OpenCV or camera not available, use a harmless placeholder image payload so CI can still exercise path.
    Returns: {"image_saved": <id>, "memory_id": <id>}
    """
    import base64, time

    # Try to capture from a system camera via cv2; if unavailable, fallback to placeholder bytes
    image_b64 = None
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        # tiny warm-up
        ret, frame = cap.read()
        cap.release()
        if ret:
            ok, buf = cv2.imencode('.jpg', frame)
            if ok:
                image_b64 = base64.b64encode(buf.tobytes()).decode('utf-8')
    except Exception:
        # CV2 missing or camera not accessible; fall through to placeholder
        pass

    if not image_b64:
        # Simple placeholder payload (visual embedder stub tolerates arbitrary bytes)
        image_b64 = base64.b64encode(b'no-camera-available').decode('utf-8')

    # Reuse admin route logic to persist image embedding and metadata
    try:
        from .admin_routes import save_image_bytes
        res = await save_image_bytes({'image_b64': image_b64, 'debate_id': debate_id, 'round_id': round, 'agent_id': agent, 'caption': caption})
        image_saved = res.get('saved')
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'image_save_failed: {e}')

    # Create a lightweight snapshot memory linking to the saved image (via hiemdall)
    try:
        from ..mcp.hiemdall import snapshot_text
        mem_id, summary = snapshot_text(agent or 'camera', f"camera snapshot:{image_saved}", int(time.time()), debate_id)
    except Exception:
        mem_id = None
        summary = None

    # Broadcast snapshot event to orchestrator if available
    try:
        from ..main import orchestrator
        await orchestrator._broadcast({'type': 'snapshot_taken', 'debate_id': debate_id, 'agent': agent or 'camera', 'snapshot_id': image_saved, 'memory_id': mem_id, 'summary': summary, 'ts': int(time.time())})
    except Exception:
        pass

    return {'image_saved': image_saved, 'memory_id': mem_id}
