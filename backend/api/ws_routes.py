from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import Optional
from ..ws_manager import WebSocketManager

router = APIRouter()

# Expect the app to have a shared ws_manager instance attached at app.state.ws_manager

@router.websocket('/ws/debates/{debate_id}')
async def debate_ws(websocket: WebSocket, debate_id: str):
    await websocket.accept()
    app = websocket.app
    manager: WebSocketManager = getattr(app.state, 'ws_manager', None)
    if manager is None:
        # If no manager, accept then close
        await websocket.close()
        return

    await manager.register(debate_id, websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # Try to parse JSON messages and honor handshake protocol
            try:
                import json
                msg = json.loads(data)
                if isinstance(msg, dict) and msg.get('type') == 'handshake':
                    # Acknowledge handshake so clients know server is ready
                    await websocket.send_json({'type': 'handshake_ack', 'debate_id': debate_id})
                    continue
            except Exception:
                pass
            # default: echo an empty string to keep connection alive
            try:
                await websocket.send_text('')
            except Exception:
                pass
    except WebSocketDisconnect:
        await manager.unregister(debate_id, websocket)
    except Exception:
        await manager.unregister(debate_id, websocket)
        try:
            await websocket.close()
        except Exception:
            pass
