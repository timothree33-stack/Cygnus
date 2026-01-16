# Embeddings package initializer
__all__ = ['get_embedder', 'get_visual_embedder']

from .adapters import StubEmbedder, OpenAIEmbedder
import os

class _VisualEmbedder:
    def embed_image(self, data: bytes):
        # Return a small fixed vector for images (stub)
        return [0.0]


def get_embedder():
    """Return an embedder instance based on environment variable EMBEDDER_TYPE.
    Supported types: 'openai', 'stub' (default).
    The OpenAI adapter requires OPENAI_API_KEY to be set and the `openai` package installed.
    """
    typ = os.environ.get('EMBEDDER_TYPE', 'stub').lower()
    if typ == 'openai':
        try:
            return OpenAIEmbedder()
        except Exception as e:
            # Fall back to stub to keep tests/dev working
            print(f"⚠️ OpenAI embedder unavailable, falling back to StubEmbedder: {e}")
            return StubEmbedder()
    if typ == 'onnx':
        try:
            from .onnx_adapter import ONNXEmbedder
            return ONNXEmbedder()
        except Exception as e:
            print(f"⚠️ ONNX embedder unavailable (missing runtime/model), falling back to StubEmbedder: {e}")
            return StubEmbedder()
    return StubEmbedder()


def get_visual_embedder():
    return _VisualEmbedder()
