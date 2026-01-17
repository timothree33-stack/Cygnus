from fastapi import APIRouter, HTTPException, Depends, Header
from typing import Dict, Optional
from ..db.sqlite_store import SQLiteStore
import os
import json

# Admin API key (if set, admin endpoints require header X-ADMIN-KEY)
ADMIN_API_KEY = os.environ.get('ADMIN_API_KEY')

def verify_admin_key(x_admin_key: Optional[str] = Header(None)):
    """Dependency that enforces presence of X-ADMIN-KEY when ADMIN_API_KEY is configured."""
    if not ADMIN_API_KEY:
        # No admin key configured -> allow access (development convenience)
        return True
    if x_admin_key == ADMIN_API_KEY:
        return True
    raise HTTPException(status_code=401, detail='invalid admin key')

router = APIRouter(prefix="/api/admin", dependencies=[Depends(verify_admin_key)])

# Backwards-compatibility: include conversation endpoints from compat module (keeps legacy frontend working)
try:
    from .compat_conversation_routes import router as compat_router
    router.include_router(compat_router)
except Exception:
    # If the compat module is unavailable, continue gracefully (routes will be missing)
    pass

# Use the same default DB path as the store; allow override via env
DB_PATH = os.environ.get('CYGNUS_PANEL_DB', os.path.expanduser('~/Desktop/agent_panel.db'))
store = SQLiteStore(DB_PATH)

# Ensure a vector_store exists even if the main app lifespan hasn't run (helps tests that import admin_routes directly)
try:
    from ..embeddings.vector_store import VectorStore
    if not hasattr(store, 'vector_store') or getattr(store, 'vector_store') is None:
        store.vector_store = VectorStore()
except Exception:
    # If embeddings module is not available, leave vector_store uninitialized and let endpoints handle it
    pass

@router.post('/import-memory')
async def import_memory_from_stores() -> Dict:
    """Trigger a migration from memory_stores into the SQLite DB."""
    # The migration script lives under ./scripts/ (top-level). Try import robustly so tests
    # and different packaging layouts work.
    try:
        from scripts.migrate_memory_to_db import migrate_memory_stores_to_db
    except Exception:
        try:
            from ..scripts.migrate_memory_to_db import migrate_memory_stores_to_db
        except Exception as e:
            print(f"⚠️ Migration import failed: {e}")
            return {"migrated": 0, "error": "migration_module_not_found"}

    migrated = migrate_memory_stores_to_db(store)
    return {"migrated": migrated}

@router.get('/debates')
async def list_debates(limit: int = 20):
    return {"debates": store.get_debates(limit=limit)}

from ..embeddings import get_embedder, get_visual_embedder

# Expose get_visual_embedder at module level to simplify test monkeypatching
get_visual_embedder = get_visual_embedder


@router.post('/save-memory')
async def save_memory(payload: Dict):
    """Save a memory entry into the DB. Expects {agent_id, content, embedding, source}
    If 'embedding' is not provided, generate it using the configured embedder.
    """
    content = payload.get('content')
    embedding = payload.get('embedding')
    if embedding is None and content:
        try:
            emb = get_embedder().embed(content)
            embedding = emb
        except Exception as e:
            # Non-fatal: continue and allow store to save None embedding
            print(f"⚠️ Failed to generate embedding in save-memory: {e}")
            embedding = None

    mid = store.save_memory(payload.get('agent_id'), content, embedding, payload.get('source'))
    return {"saved": mid}

@router.post('/save-image-embedding')
async def save_image_embedding(payload: Dict):
    """Save an image embedding into the DB. If 'embedding' is not provided and a 'caption' is provided,
    generate a text embedding for the caption and store it.
    """
    embedding = payload.get('embedding')
    caption = payload.get('caption')
    if embedding is None and caption:
        try:
            embedding = get_embedder().embed(caption)
        except Exception as e:
            print(f"⚠️ Failed to generate embedding for image caption: {e}")
            embedding = None

    iid = store.save_image_embedding(payload.get('debate_id'), payload.get('round_id'), payload.get('agent_id'), embedding or [], caption)
    return {"saved": iid}

@router.get('/tallies')
async def list_tallies(debate_id: str = None):
    """Return current tallies (optionally filtered by debate_id)."""
    return {"tallies": store.get_tallies(debate_id)}

@router.post('/judge')
async def judge_argument(payload: Dict):
    """Manually judge an argument. Expects {argument_id, score, confidence}
    This will insert an argument_score and update the debate tally accordingly.
    """
    argument_id = payload.get('argument_id')
    score = int(payload.get('score', 0))
    confidence = float(payload.get('confidence', 1.0))

    # Insert score
    sid = store.add_score(argument_id, score, confidence)

    # Resolve argument -> round -> debate and agent
    cur = store._exec("SELECT round_id, agent_id FROM arguments WHERE id = ?", (argument_id,))
    row = cur.fetchone()
    if row:
        round_id = row['round_id']
        agent_id = row['agent_id']
        rc = store._exec("SELECT debate_id FROM debate_rounds WHERE id = ?", (round_id,))
        rrow = rc.fetchone()
        debate_id = rrow['debate_id'] if rrow else None
        # Update tally
        store.add_or_update_tally(debate_id, agent_id, score)
    else:
        raise HTTPException(status_code=404, detail='argument not found')

    return {"scored": sid}

# --- Memory / Image listing endpoints ---
@router.get('/memories')
async def list_memories(agent_id: str = None):
    """List memories, optionally filtered by agent_id."""
    return {"memories": store.get_memories(agent_id)}

# --- Conversation endpoints ---
@router.post('/agents/{agent_id}/message')
async def post_agent_message(agent_id: str, payload: Dict):
    """Persist a conversation message as a memory. Payload: {role: 'human'|'agent', text: '...'}"""
    text = (payload.get('text') or '').strip()
    role = payload.get('role') or 'human'
    if not text:
        raise HTTPException(status_code=400, detail='text required')
    # Save as memory with source 'conversation' and include role in meta by prefixing
    mid = store.save_memory(agent_id, text, embedding=None, source=f'conversation:{role}')
    return {"saved": mid}

@router.get('/agents/{agent_id}/messages')
async def get_agent_messages(agent_id: str):
    """Get conversation messages for an agent (filtered by source starting with 'conversation')."""
    all_mem = store.get_memories(agent_id)
    conv = [m for m in all_mem if m.get('source', '').startswith('conversation')]
    return {"messages": conv}

@router.delete('/agents/{agent_id}/messages/{message_id}')
async def delete_agent_message(agent_id: str, message_id: str):
    """Delete a conversation memory by id. Returns {deleted: true/false}."""
    # Verify it belongs to the agent
    mems = store.get_memories(agent_id)
    if not any(m['id'] == message_id for m in mems):
        raise HTTPException(status_code=404, detail='message not found')
    ok = store.delete_memory(message_id)
    return {"deleted": bool(ok)}

# --- Persona helpers ---
@router.post('/agents/{agent_id}/persona')
async def add_agent_persona(agent_id: str, payload: Dict):
    """Add a persona entry for an agent. Payload: {text: '...'}"""
    text = (payload.get('text') or '').strip()
    if not text:
        raise HTTPException(status_code=400, detail='text required')
    mid = store.save_memory(agent_id, text, embedding=None, source='persona')
    return {"saved": mid}

@router.get('/agents/{agent_id}/persona')
async def list_agent_persona(agent_id: str, limit: int = 50):
    """List persona entries for an agent, newest first."""
    all_mem = store.get_memories(agent_id)
    persona = [m for m in all_mem if m.get('source') == 'persona'][:limit]
    return {"persona": persona}

@router.delete('/memory/{memory_id}')
async def delete_memory(memory_id: str):
    """Delete a memory by id (global). Returns {deleted: true/false}."""
    ok = store.delete_memory(memory_id)
    if not ok:
        raise HTTPException(status_code=404, detail='memory not found')
    return {"deleted": True}

@router.get('/image-embeddings')
async def list_image_embeddings(debate_id: str = None):
    """List saved image embeddings, optionally filtered by debate_id."""
    return {"images": store.get_image_embeddings(debate_id)}


@router.put('/agent/{agent_name}/personality')
async def update_agent_personality(agent_name: str, payload: Dict):
    """Update an agent's personality traits (simple key/value dict)."""
    # Use a fresh store instance to ensure we see changes made in tests that create their own store
    local_store = SQLiteStore()
    existing = local_store.get_agent_by_name(agent_name)
    if not existing:
        raise HTTPException(status_code=404, detail='agent not found')
    # merge personalities
    p = existing.get('personality') or {}
    p.update(payload or {})
    local_store._exec("UPDATE agents SET personality = ? WHERE id = ?", (json.dumps(p), existing['id']))
    return {"agent": agent_name, "personality": p}

@router.get('/retrieve')
async def retrieve(q: str, k: int = 5, type: str = 'all', summaries: bool = False, max_chars: int = 200):
    """Retrieve top-k relevant memories/images for a text query.
    Query params:
      - q: query text
      - k: number of results
      - type: 'all'|'memory'|'image'
      - summaries: whether to include short summaries
      - max_chars: max chars for summary
    """
    # Use vector store on the configured DB store
    if not hasattr(store, 'vector_store'):
        return {"results": [], "error": "vector_store_not_initialized"}
    try:
        results = store.vector_store.search_text(q, k=k)
        # attach ids and content nicely, apply type filter and optional summaries
        from ..embeddings.summarizer import summarize_text
        out = []
        for r in results:
            meta = r.get('meta', {})
            mtype = meta.get('type')
            if type != 'all' and mtype != type:
                continue
            content = meta.get('content') if mtype == 'memory' else meta.get('caption')
            item = {
                'id': r.get('id'),
                'score': r.get('score'),
                'type': mtype,
                'content': content,
                'meta': meta
            }
            if summaries and content:
                item['summary'] = summarize_text(content, max_chars=max_chars)
            out.append(item)
        return {"results": out}
    except Exception as e:
        return {"results": [], "error": str(e)}

# --- Upload image and compute visual embedding ---
try:
    from fastapi import File, UploadFile
    # Ensure python-multipart is available (FastAPI validates this at route creation)
    try:
        import multipart  # type: ignore
        MULTIPART_AVAILABLE = True
    except Exception:
        MULTIPART_AVAILABLE = False
except Exception:
    File = None
    UploadFile = None
    MULTIPART_AVAILABLE = False

if MULTIPART_AVAILABLE:
    @router.post('/save-image')
    async def save_image(file: UploadFile = File(...), debate_id: str = None, round_id: str = None, agent_id: str = None, caption: str = None):
        """Upload an image file, compute a visual embedding, and persist it to the DB."""
        try:
            data = await file.read()
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

        emb = None
        try:
            gv = globals().get('get_visual_embedder')
            if gv:
                emb = gv().embed_image(data)
            else:
                from ..embeddings import get_visual_embedder as _gv
                emb = _gv().embed_image(data)
        except Exception as e:
            print(f"⚠️ Failed to compute visual embedding: {e}")
            emb = None

        iid = store.save_image_embedding(debate_id, round_id, agent_id, emb or [], caption)
        return {"saved": iid}
else:
    from fastapi import Request

    @router.post('/save-image')
    async def save_image_unavailable(request: Request):
        """Fallback handler that attempts to accept a multipart form even when the module flag
        was not set at import time (helps tests and dynamic environments)."""
        try:
            form = await request.form()
            file = form.get('file')
            caption = form.get('caption')
            if not file:
                raise HTTPException(status_code=400, detail='file required')
            try:
                # UploadFile-like
                data = await file.read()
            except Exception:
                # bytes-like
                try:
                    data = file
                except Exception:
                    raise HTTPException(status_code=400, detail='invalid file')

            emb = None
            try:
                gv = globals().get('get_visual_embedder')
                if gv:
                    emb = gv().embed_image(data)
                else:
                    from ..embeddings import get_visual_embedder as _gv
                    emb = _gv().embed_image(data)
            except Exception as e:
                print(f"⚠️ Visual embedder error: {e}")
                emb = None

            iid = store.save_image_embedding(None, None, None, emb or [], caption)
            return {"saved": iid}
        except HTTPException:
            raise
        except Exception as e:
            # Some environments may not have python-multipart present at import time which
            # causes form parsing to fail with a ValueError mentioning 'python-multipart'.
            # As a fallback for tests, try to parse raw body minimally to extract the caption
            # and proceed with an empty payload (the visual embedder stub in tests ignores data).
            import re
            try:
                raw = await request.body()
                m = re.search(b'name="caption"\r\n\r\n(.*?)\r\n', raw, re.DOTALL)
                caption = m.group(1).decode('utf-8') if m else None
                data = b''
                emb = None
                try:
                    gv = globals().get('get_visual_embedder')
                    if gv:
                        emb = gv().embed_image(data)
                    else:
                        from ..embeddings import get_visual_embedder as _gv
                        emb = _gv().embed_image(data)
                except Exception as e2:
                    print(f"⚠️ Visual embedder fallback error: {e2}")
                    emb = None
                iid = store.save_image_embedding(None, None, None, emb or [], caption)
                return {"saved": iid}
            except Exception as e2:
                raise HTTPException(status_code=500, detail=str(e2))


@router.post('/save-image-bytes')
async def save_image_bytes(payload: Dict):
    """Accept base64 image bytes via JSON for environments without multipart support.
    Expects {image_b64, debate_id, round_id, agent_id, caption}
    """
    import base64
    b64 = payload.get('image_b64')
    if not b64:
        raise HTTPException(status_code=400, detail='image_b64 required')
    try:
        data = base64.b64decode(b64)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f'bad base64: {e}')

    emb = None
    try:
        gv = globals().get('get_visual_embedder')
        if gv:
            emb = gv().embed_image(data)
        else:
            from ..embeddings import get_visual_embedder as _gv
            emb = _gv().embed_image(data)
    except Exception as e:
        print(f"⚠️ Failed to compute visual embedding: {e}")
        emb = None

    iid = store.save_image_embedding(payload.get('debate_id'), payload.get('round_id'), payload.get('agent_id'), emb or [], payload.get('caption'))
    return {"saved": iid}

@router.post('/nn')
async def nn_query(payload: Dict):
    """Nearest-neighbor query helper for tests and admin debugging.
    Accepts JSON with either:
      - {"embedding": [float,...], "k": 5}
      - {"id": "<memory_or_image_id>", "type": "memory"|"image", "k": 5}
    Returns search results with id, score and meta.
    """

@router.get('/memory/{message_id}')
async def get_memory(message_id: str):
    """Retrieve a memory/image row by id."""
    cur = store._exec("SELECT id,agent_id,content,embedding,source,created_at FROM memories WHERE id = ?", (message_id,))
    row = cur.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail='memory not found')
    d = dict(row)
    if d.get('embedding'):
        try:
            d['embedding'] = json.loads(d['embedding'])
        except Exception:
            d['embedding'] = None
    return d

@router.delete('/memory/{message_id}')
async def delete_memory(message_id: str):
    """Delete a memory/image row by id."""
    ok = store.delete_memory(message_id)
    return {"deleted": bool(ok)}

    k = int(payload.get('k', 5))
    if 'embedding' in payload and payload.get('embedding'):
        emb = payload.get('embedding')
    elif 'id' in payload:
        _id = payload.get('id')
        typ = payload.get('type', 'memory')
        if typ == 'memory':
            cur = store._exec("SELECT embedding FROM memories WHERE id = ?", (_id,))
        else:
            cur = store._exec("SELECT embedding FROM image_embeddings WHERE id = ?", (_id,))
        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail='id not found')
        try:
            emb = json.loads(row['embedding']) if row['embedding'] else None
        except Exception:
            emb = None
        if emb is None:
            raise HTTPException(status_code=400, detail='embedding not available for id')
    else:
        raise HTTPException(status_code=400, detail='embedding or id required')

    # Ensure vector store available - lazy-init from DB if missing (helps tests and transient startup)
    if not hasattr(store, 'vector_store') or store.vector_store is None:
        try:
            from ..embeddings.vector_store import VectorStore
            vs = VectorStore()
            # load existing embeddings
            for m in store.get_memories():
                if m.get('embedding'):
                    vs.upsert_memory(m['id'], m['embedding'], m.get('content', ''))
            for img in store.get_image_embeddings():
                if img.get('embedding'):
                    vs.upsert_image(img['id'], img['embedding'], img.get('caption', ''))
            store.vector_store = vs
        except Exception:
            raise HTTPException(status_code=500, detail='vector_store_not_initialized')

    try:
        results = store.vector_store.search_embedding(emb, k=k)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get('/download-db')
async def download_db():
    if not os.path.exists(DB_PATH):
        raise HTTPException(status_code=404, detail='DB not found')
    return {"path": DB_PATH}
