from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Optional
from ..db.sqlite_store import SQLiteStore
from .admin_routes import verify_admin_key

router = APIRouter(dependencies=[Depends(verify_admin_key)])  # Include into admin router so paths are under /api/admin

# Compatibility conversation endpoints for the frontend
@router.post('/agents/{agent_id}/message')
async def post_agent_message_compat(agent_id: str, payload: Dict):
    text = (payload.get('text') or '').strip()
    role = payload.get('role') or 'human'
    if not text:
        raise HTTPException(status_code=400, detail='text required')
    store = SQLiteStore()
    mid = store.save_memory(agent_id, text, embedding=None, source=f'conversation:{role}')
    return {"saved": mid}

@router.get('/agents/{agent_id}/messages')
async def get_agent_messages_compat(agent_id: str):
    store = SQLiteStore()
    all_mem = store.get_memories(agent_id)
    conv = [m for m in all_mem if m.get('source', '').startswith('conversation')]
    return {"messages": conv}

@router.delete('/agents/{agent_id}/messages/{message_id}')
async def delete_agent_message_compat(agent_id: str, message_id: str):
    store = SQLiteStore()
    mems = store.get_memories(agent_id)
    if not any(m['id'] == message_id for m in mems):
        raise HTTPException(status_code=404, detail='message not found')
    ok = store.delete_memory(message_id)
    return {"deleted": bool(ok)}
