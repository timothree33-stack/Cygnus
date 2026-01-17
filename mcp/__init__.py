"""MCP core package for Cygnus Pyramid
This package will contain the H2 manager, adapters for model backends, ZProxy, visual language model adapters,
and a thin episodic memory wrapper. Implementations are intentionally minimal here and will be filled in
by migrating tested components from the repository into these modules.
"""

# Re-export canonical backend mcp implementations from `backend.mcp` so imports like
# `import mcp` or `from mcp import Manager` continue to work while the canonical
# implementation lives under `backend/mcp`.
from backend.mcp.manager import Manager
from backend.mcp.model_manifest import ModelManifest

__all__ = [
    "Manager",
    "ModelManifest",
]
