# Embeddings package initializer
__all__ = ['get_embedder', 'get_visual_embedder']

class _TextEmbedder:
    def embed(self, text: str):
        # Simple deterministic stub embedding for local dev/tests
        return [float(abs(hash(text)) % 1000) / 1000.0]

class _VisualEmbedder:
    def embed_image(self, data: bytes):
        # Return a small fixed vector for images
        return [0.0]


def get_embedder():
    return _TextEmbedder()


def get_visual_embedder():
    return _VisualEmbedder()
