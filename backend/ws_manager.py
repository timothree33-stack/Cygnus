import asyncio
from typing import Dict, Set

class WebSocketManager:
    """Simple manager that tracks websocket connections per debate_id and broadcasts messages."""
    def __init__(self):
        # debate_id -> set of websocket objects
        self._conns: Dict[str, Set] = {}
        self._lock = asyncio.Lock()

    async def register(self, debate_id: str, ws):
        async with self._lock:
            if debate_id not in self._conns:
                self._conns[debate_id] = set()
            self._conns[debate_id].add(ws)

    async def unregister(self, debate_id: str, ws):
        async with self._lock:
            if debate_id in self._conns and ws in self._conns[debate_id]:
                self._conns[debate_id].remove(ws)
                if not self._conns[debate_id]:
                    del self._conns[debate_id]

    async def broadcast(self, msg: dict):
        # Broadcast only to clients listening for the given debate_id if present
        debate_id = msg.get('debate_id')
        targets = []
        async with self._lock:
            if debate_id and debate_id in self._conns:
                targets = list(self._conns[debate_id])
            else:
                # fallback: broadcast to all
                for s in self._conns.values():
                    targets.extend(list(s))

        for ws in targets:
            try:
                await ws.send_json(msg)
            except Exception:
                # swallow send errors
                pass
