"""Embedding providers.

D08 supported four providers with hand-written retry loops. Here every
OpenAI-compatible endpoint (OpenAI, OpenRouter, a local server, ...) goes
through ``OpenAIEmbedder`` and LangChain's client handles retries. The
hashing embedder needs no network and exists for tests and offline demos;
its vectors carry no semantics.
"""

import asyncio
import hashlib
import math
import re
from typing import Protocol

from langchain_openai import OpenAIEmbeddings
from pydantic import SecretStr

_TOKEN = re.compile(r"\w+", re.UNICODE)


class Embedder(Protocol):
    """Anything that turns texts into fixed-size vectors."""

    dimension: int

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed passages for indexing."""
        ...

    async def embed_query(self, text: str) -> list[float]:
        """Embed a search query."""
        ...


class OpenAIEmbedder:
    """Embeddings from any OpenAI-compatible endpoint."""

    def __init__(self, model: str, dimension: int, api_key: str, base_url: str | None = None):
        """Create a client for ``model``; ``base_url`` selects a non-OpenAI endpoint."""
        self.dimension = dimension
        self._client = OpenAIEmbeddings(
            model=model,
            api_key=SecretStr(api_key),
            base_url=base_url or None,
            dimensions=dimension if model.startswith("text-embedding-3") else None,
        )

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed passages for indexing."""
        return await self._client.aembed_documents(texts) if texts else []

    async def embed_query(self, text: str) -> list[float]:
        """Embed a search query."""
        return await self._client.aembed_query(text)


class FastEmbedEmbedder:
    """Embeddings computed in-process on the CPU with FastEmbed (ONNX): no server, no API key.

    The model is downloaded once to ``cache_dir`` on first use; after that it runs offline.
    ``sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`` (384 dimensions,
    about 0.2 GB) covers Vietnamese; ``intfloat/multilingual-e5-large`` (1024) is more
    accurate and about ten times larger.
    """

    def __init__(self, model: str, dimension: int, cache_dir: str | None = None):
        """Load ``model``; ``dimension`` must match its output size."""
        # fastembed is the optional "offline" extra; importing it here keeps the core app importable without it.
        from fastembed import TextEmbedding

        self.dimension = dimension
        self._model = TextEmbedding(model_name=model, cache_dir=cache_dir)

    def _check(self, vectors: list[list[float]]) -> list[list[float]]:
        if vectors and len(vectors[0]) != self.dimension:
            raise ValueError(f"embedding model returns {len(vectors[0])} dimensions, configured {self.dimension}")
        return vectors

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed passages for indexing (on a worker thread: inference is CPU-bound)."""
        if not texts:
            return []
        vectors = await asyncio.to_thread(lambda: [[float(x) for x in v] for v in self._model.passage_embed(texts)])
        return self._check(vectors)

    async def embed_query(self, text: str) -> list[float]:
        """Embed a search query."""
        vectors = await asyncio.to_thread(lambda: [[float(x) for x in v] for v in self._model.query_embed(text)])
        return self._check(vectors)[0]


class HashingEmbedder:
    """Deterministic bag-of-words hashing into a unit vector. No network, no semantics."""

    def __init__(self, dimension: int = 256):
        """Create an embedder producing ``dimension``-sized vectors."""
        self.dimension = dimension

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        for token in _TOKEN.findall(text.lower()):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            vector[int.from_bytes(digest, "little") % self.dimension] += 1.0
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed passages for indexing."""
        return [self._embed(text) for text in texts]

    async def embed_query(self, text: str) -> list[float]:
        """Embed a search query."""
        return self._embed(text)
