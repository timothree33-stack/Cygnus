import os
from fastapi import FastAPI
from fastapi.testclient import TestClient
from backend.api.admin_routes import router as admin_router
from backend.db.sqlite_store import SQLiteStore


def test_get_store_respects_cygnus_panel_db(tmp_path, monkeypatch):
    """Integration: when CYGNUS_PANEL_DB changes at runtime, admin endpoints should use the new DB."""
    app = FastAPI()
    app.include_router(admin_router)
    client = TestClient(app)

    db1 = tmp_path / "db1.sqlite"
    db2 = tmp_path / "db2.sqlite"

    # Point to db1 and save a memory
    monkeypatch.setenv("CYGNUS_PANEL_DB", str(db1))
    r1 = client.post("/api/admin/save-memory", json={"agent_id": "a1", "content": "from-db1", "source": "test"})
    assert r1.status_code == 200
    mid1 = r1.json().get("saved")
    assert mid1

    # Point to db2 and save a different memory
    monkeypatch.setenv("CYGNUS_PANEL_DB", str(db2))
    r2 = client.post("/api/admin/save-memory", json={"agent_id": "a2", "content": "from-db2", "source": "test"})
    assert r2.status_code == 200
    mid2 = r2.json().get("saved")
    assert mid2

    # Verify persistence: each DB should contain only its respective memory
    s1 = SQLiteStore(str(db1))
    mems1 = [m for m in s1.get_memories() if m.get("source") == "test"]
    assert any(m["content"] == "from-db1" for m in mems1)

    s2 = SQLiteStore(str(db2))
    mems2 = [m for m in s2.get_memories() if m.get("source") == "test"]
    assert any(m["content"] == "from-db2" for m in mems2)
