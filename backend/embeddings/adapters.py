"""Pluggable embedder adapters.

Implementations should provide an `embed(text: str) -> List[float]` method.
"""
from typing import List
import os

class EmbedderAdapter:
    def embed(self, text: str) -> List[float]:
        raise NotImplementedError()


class StubEmbedder(EmbedderAdapter):
    def embed(self, text: str) -> List[float]:
        # Deterministic small vector for testing/dev
        return [float(abs(hash(text)) % 1000) / 1000.0]


class OpenAIEmbedder(EmbedderAdapter):
    def __init__(self, api_key: str = None, model: str = 'text-embedding-3-small'):
        # Lazy import to avoid requiring openai package in CI unless used
        try:
            import openai
            self.openai = openai
        except Exception:
            raise RuntimeError('openai package is required for OpenAIEmbedder')
        self.api_key = api_key or os.environ.get('OPENAI_API_KEY')
        if not self.api_key:
            raise RuntimeError('OPENAI_API_KEY not set')
        self.model = model
        self.openai.api_key = self.api_key

    def embed(self, text: str) -> List[float]:
        # Call OpenAI embeddings API (blocking). In production, consider async client.
        resp = self.openai.Embedding.create(model=self.model, input=text)
        vec = resp['data'][0]['embedding']
        return [float(x) for x in vec]
