"""Evidence embedder — generates and caches text embeddings."""

from __future__ import annotations

import hashlib

import openai
import structlog

logger = structlog.get_logger(__name__)

# In-memory embedding cache to avoid re-computing within a session.
_embedding_cache: dict[str, list[float]] = {}
_CACHE_MAX_SIZE = 5000


class EvidenceEmbedder:
    """Generates embeddings using OpenAI text-embedding-3-small."""

    def __init__(self, api_key: str, model: str = "text-embedding-3-small") -> None:
        self.client = openai.AsyncOpenAI(api_key=api_key)
        self.model = model

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts, using cache where possible."""
        results: list[list[float] | None] = [None] * len(texts)
        to_embed: list[tuple[int, str]] = []

        for i, text in enumerate(texts):
            cache_key = self._cache_key(text)
            if cache_key in _embedding_cache:
                results[i] = _embedding_cache[cache_key]
            else:
                to_embed.append((i, text))

        if to_embed:
            batch_texts = [t for _, t in to_embed]
            try:
                response = await self.client.embeddings.create(
                    model=self.model,
                    input=batch_texts,
                )
                for (idx, text), emb_data in zip(to_embed, response.data):
                    vec = emb_data.embedding
                    results[idx] = vec
                    # Cache it.
                    if len(_embedding_cache) < _CACHE_MAX_SIZE:
                        _embedding_cache[self._cache_key(text)] = vec
            except Exception:
                logger.exception("embedding_error", batch_size=len(batch_texts))
                # Return zero vectors for failed embeddings.
                for idx, _ in to_embed:
                    results[idx] = []

        return [r if r is not None else [] for r in results]

    async def embed_text(self, text: str) -> list[float]:
        """Embed a single text string."""
        result = await self.embed_texts([text])
        return result[0]

    @staticmethod
    def cosine_similarity(a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    @staticmethod
    def _cache_key(text: str) -> str:
        """SHA-256 of normalized text for cache key."""
        return hashlib.sha256(text.strip().lower().encode()).hexdigest()[:16]

    @staticmethod
    def clear_cache() -> None:
        """Clear the embedding cache."""
        _embedding_cache.clear()
