from typing import Any, Dict, Optional


class SampleAgent:
    """Minimal agent scaffold that matches the repo's expected interface.

    - Implement `async def respond(**kwargs)` that returns a string or serializable dict.
    - Keep the signature compatible with `_StubAgent.respond(topic, round, **kwargs)`.
    - Persist embeddings by calling `memory_store.upsert_embedding(...)` if a memory_store is passed.
    """

    def __init__(self, name: str = "sample") -> None:
        self.name = name

    async def respond(self, **kwargs: Any) -> str:
        """Return a short, deterministic response for testing and local development.

        Expected kwargs: topic (str), round (int), memory_store (optional)
        """
        topic = kwargs.get("topic", "(topic)")
        round_num = kwargs.get("round", "?")

        text = f"{self.name} responds to '{topic}' (round {round_num})"

        # Example: persist an embedding anchor if a memory store is provided.
        # if memory_store is not None:
        #     memory_store.upsert_embedding(self.name, f"anchor:{topic}", text, vector=None)

        return text
