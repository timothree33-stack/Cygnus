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
            # Keep connection open; we don't expect to receive messages in this simple setup
            data = await websocket.receive_text()
            # If a client sends a ping payload, we could respond; ignore for now
            await websocket.send_text('')
    except WebSocketDisconnect:
        await manager.unregister(debate_id, websocket)
    except Exception:
        await manager.unregister(debate_id, websocket)
        try:
            await websocket.close()
        except Exception:
            pass
