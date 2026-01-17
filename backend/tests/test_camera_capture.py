from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

def test_camera_capture_endpoint():
    # start a debate
    r = client.post('/api/debate/start')
    assert r.status_code == 200
    did = r.json()['debate_id']

    r2 = client.post(f'/api/debate/{did}/camera-capture')
    assert r2.status_code == 200
    body = r2.json()
    assert 'image_saved' in body
    # memory may be None if snapshot persistence failed; ensure key exists
    assert 'memory_id' in body
