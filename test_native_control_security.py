import os
import pytest
from fastapi.testclient import TestClient
from agent_panel import native_control
from agent_panel.app import app


def test_native_disabled(monkeypatch):
    # Ensure native control is disabled by default
    monkeypatch.delenv("ENABLE_NATIVE_CONTROL", raising=False)
    client = TestClient(app)

    resp = client.post("/api/native/action", json={"type": "mouse_click", "x": 10, "y": 10})
    assert resp.status_code == 403
    assert "disabled" in resp.json()["detail"].lower()


def test_native_requires_auth_key_to_be_set(monkeypatch):
    monkeypatch.setenv("ENABLE_NATIVE_CONTROL", "1")
    monkeypatch.delenv("NATIVE_CONTROL_AUTH_KEY", raising=False)
    client = TestClient(app)

    resp = client.post("/api/native/action", json={"type": "mouse_click", "x": 10, "y": 10})
    assert resp.status_code == 403
    assert "requires NATIVE_CONTROL_AUTH_KEY" in resp.json()["detail"]


def test_native_action_authorized(monkeypatch):
    monkeypatch.setenv("ENABLE_NATIVE_CONTROL", "1")
    monkeypatch.setenv("NATIVE_CONTROL_AUTH_KEY", "secret-token")

    # Stub controller to avoid interacting with the real system
    class StubController:
        async def execute_action(self, action, caller_meta=None):
            return {"success": True, "action": action.get("type")}

        class Screen:
            async def screenshot_base64(self, region=None):
                return "data:image/png;base64,AAA"

        screen = Screen()

        async def describe_screen(self):
            return {"screen_size": {"width": 100, "height": 100}, "mouse_position": {"x": 1, "y": 1}, "active_window": None}

    monkeypatch.setattr(native_control, "get_native_controller", lambda allow_create=False: StubController())

    client = TestClient(app)
    headers = {"X-NATIVE-AUTH": "secret-token"}

    resp = client.post("/api/native/action", headers=headers, json={"type": "mouse_click", "x": 10, "y": 10})
    assert resp.status_code == 200
    assert resp.json()["success"] is True

    # Screenshot and screen description endpoints
    resp2 = client.post("/api/native/screenshot", headers=headers)
    assert resp2.status_code == 200
    assert resp2.json()["image"].startswith("data:image/png;base64,")

    resp3 = client.get("/api/native/screen", headers=headers)
    assert resp3.status_code == 200
    assert "screen_size" in resp3.json()