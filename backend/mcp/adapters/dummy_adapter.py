class DummyBackend:
    """A tiny in-process backend used for CI and integration tests (backend/mcp adapter)."""
    def __init__(self, manifest):
        self.manifest = manifest
        self.running = True

    async def respond(self, prompt: str, stream: bool = False):
        # deterministic reply for tests
        return f"dummy-response-to:{prompt}"

    def health(self):
        return {"ok": True, "name": self.manifest.name}
