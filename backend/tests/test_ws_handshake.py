import json
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

def test_ws_handshake_ack():
    with client.websocket_connect('/ws/debates/handshake-test') as ws:
        # send handshake
        ws.send_text(json.dumps({'type': 'handshake', 'debate_id': 'handshake-test'}))
        data = ws.receive_text()
        msg = json.loads(data)
        assert msg.get('type') == 'handshake_ack'
        assert msg.get('debate_id') == 'handshake-test'
