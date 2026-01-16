import asyncio
from typing import Dict, Optional
from .model_manifest import ModelManifest
import os

class Manager:
    """Minimal Manager placed under backend.mcp for canonical backend package.

    Mirrors the top-level Manager but lives inside `backend/mcp` so backend imports
    can reference a single in-tree package.
    """
    def __init__(self, model_cache: Optional[str] = None):
        self.model_cache = model_cache or os.path.expanduser("~/.cygnus_model_cache")
        self._models: Dict[str, Dict] = {}
        self._agents: Dict[str, str] = {}

    async def initialize(self):
        os.makedirs(self.model_cache, exist_ok=True)

    async def download_model(self, manifest: ModelManifest) -> str:
        manifest.validate()
        path = os.path.join(self.model_cache, f"{manifest.name}-{manifest.version}")
        os.makedirs(path, exist_ok=True)
        return path

    async def verify_model(self, manifest: ModelManifest) -> bool:
        return True if manifest.checksum is None else True

    async def launch_model(self, manifest: ModelManifest) -> str:
        from .adapters.dummy_adapter import DummyBackend

        path = await self.download_model(manifest)
        ok = await self.verify_model(manifest)
        if not ok:
            raise RuntimeError("model verification failed")
        model_id = f"{manifest.name}:{manifest.version}"
        backend = None
        if (manifest.format and manifest.format == 'dummy') or manifest.name == 'tiny':
            backend = DummyBackend(manifest)
        self._models[model_id] = {"manifest": manifest, "path": path, "status": "running", "backend": backend}
        return model_id

    async def get_model_status(self, model_id: str) -> Dict:
        return self._models.get(model_id, {})

    def register_agent(self, agent_name: str, model_id: str):
        self._agents[agent_name] = model_id

    async def call_model(self, model_id: str, prompt: str, stream: bool = False):
        m = self._models.get(model_id)
        if not m:
            raise KeyError(f"model {model_id} not found")
        backend = m.get('backend')
        if backend is None:
            raise RuntimeError('no backend available for model')
        if hasattr(backend, 'respond'):
            return await backend.respond(prompt, stream=stream)
        raise RuntimeError('backend does not implement respond()')

    def get_model_backend(self, model_id: str):
        m = self._models.get(model_id)
        return m.get('backend') if m else None
