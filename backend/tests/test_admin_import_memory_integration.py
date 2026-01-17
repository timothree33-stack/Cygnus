import os
from pathlib import Path

from fastapi.testclient import TestClient

from backend.main import app
from backend.db.sqlite_store import SQLiteStore


def test_admin_import_memory_endpoint_integration(tmp_path, monkeypatch):
    # Ensure admin key is not set for this test
    monkeypatch.delenv('ADMIN_API_KEY', raising=False)

    ms = tmp_path / 'memory_stores'
    katz = ms / 'katz'
    katz.mkdir(parents=True)
    kf = katz / 'katz_mem.jsonl'
    kf.write_text('{"agent_id": "katz", "content": "integration memory"}\n')

    dbpath = tmp_path / 'panel.db'
    monkeypatch.setenv('CYGNUS_PANEL_DB', str(dbpath))
    monkeypatch.setenv('MEMORY_STORES_PATH', str(ms))

    client = TestClient(app)

    r = client.post('/api/admin/import-memory')
    assert r.status_code == 200
    body = r.json()
    assert body.get('migrated') == 1

    # Verify via admin list_memories
    r2 = client.get('/api/admin/memories', params={'agent_id': 'katz'})
    assert r2.status_code == 200
    mems = r2.json().get('memories', [])
    contents = [m['content'] for m in mems]
    assert 'integration memory' in contents
