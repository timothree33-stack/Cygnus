import pytest
from fastapi.testclient import TestClient
from backend.api.admin_routes import router as admin_router
from backend.db.sqlite_store import SQLiteStore
from fastapi import FastAPI

app = FastAPI()
app.include_router(admin_router)

client = TestClient(app)

def test_add_and_list_persona(tmp_path, monkeypatch):
    # ensure store uses a temporary DB path
    monkeypatch.setenv('CYGNUS_PANEL_DB', str(tmp_path / 'test.db'))
    # create agent
    store = SQLiteStore(str(tmp_path / 'test.db'))
    aid = store.create_agent('test-agent')

    # add persona
    r = client.post(f'/api/admin/agents/{aid}/persona', json={'text': 'I love reasoning.'})
    assert r.status_code == 200
    saved = r.json().get('saved')
    assert saved

    # list persona
    r2 = client.get(f'/api/admin/agents/{aid}/persona')
    assert r2.status_code == 200
    persona = r2.json().get('persona')
    assert any(p['id'] == saved for p in persona)

def test_delete_memory(tmp_path, monkeypatch):
    monkeypatch.setenv('CYGNUS_PANEL_DB', str(tmp_path / 'test2.db'))
    store = SQLiteStore(str(tmp_path / 'test2.db'))
    aid = store.create_agent('x')
    mid = store.save_memory(aid, 'to_remove', source='persona')

    r = client.delete(f'/api/admin/memory/{mid}')
    assert r.status_code == 200
    assert r.json().get('deleted') is True

    # deleting again returns 404
    r2 = client.delete(f'/api/admin/memory/{mid}')
    assert r2.status_code == 404
