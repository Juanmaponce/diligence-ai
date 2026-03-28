"""Shared embedding client for the RAG pipeline."""

import logging

import httpx

from app.config import settings

logger = logging.getLogger(__name__)


async def embed_text(text: str) -> list[float]:
    """Call Ollama /api/embeddings and return the vector."""
    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(
            f"{settings.ollama_base_url}/api/embeddings",
            json={"model": settings.embed_model, "prompt": text},
        )
        response.raise_for_status()
        return response.json()["embedding"]
