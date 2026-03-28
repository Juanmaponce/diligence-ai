"""
RAG Ingestion Pipeline
----------------------
Responsibilities:
  1. Chunk raw text into token-bounded segments with overlap
  2. Generate embeddings via Ollama (nomic-embed-text)
  3. Store chunks + embeddings in ChromaDB

Design notes:
  - We use a simple token-count chunker (tiktoken cl100k_base).
    For production you'd layer in semantic chunking or a document parser
    (e.g. unstructured.io) to handle PDFs, tables, etc.
  - Embeddings are requested one chunk at a time to keep memory flat.
    A production system would batch these calls.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from typing import Any, Optional

import tiktoken
import chromadb

from app.config import settings
from app.rag.embeddings import embed_text

logger = logging.getLogger(__name__)

# Shared ChromaDB client (module-level singleton)
_chroma_client: Optional[chromadb.AsyncHttpClient] = None
_chroma_lock = asyncio.Lock()


async def get_chroma_client() -> chromadb.AsyncHttpClient:
    global _chroma_client
    if _chroma_client is not None:
        return _chroma_client
    async with _chroma_lock:
        if _chroma_client is None:
            _chroma_client = await chromadb.AsyncHttpClient(
                host=settings.chroma_host,
                port=settings.chroma_port,
            )
    return _chroma_client


async def get_or_create_collection(client: chromadb.AsyncHttpClient):
    return await client.get_or_create_collection(
        name=settings.collection_name,
        metadata={"hnsw:space": "cosine"},
    )


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def _chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Split text into overlapping token windows.
    Returns a list of raw string chunks.
    """
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    chunks: list[str] = []

    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunks.append(enc.decode(chunk_tokens))
        if end == len(tokens):
            break
        start += chunk_size - overlap

    return chunks


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def ingest_document(
    content: str,
    doc_id: str,
    metadata: dict[str, Any],
) -> int:
    """
    Chunk, embed, and store a document in ChromaDB.
    Returns the number of chunks stored.
    """
    chunks = _chunk_text(content, settings.chunk_size, settings.chunk_overlap)
    logger.info("[ingest] doc_id=%s -> %d chunks", doc_id, len(chunks))

    chroma = await get_chroma_client()
    collection = await get_or_create_collection(chroma)

    ids: list[str] = []
    embeddings: list[list[float]] = []
    documents: list[str] = []
    metadatas: list[dict] = []

    for i, chunk in enumerate(chunks):
        chunk_id = hashlib.sha256(f"{doc_id}::{i}".encode()).hexdigest()
        embedding = await embed_text(chunk)

        ids.append(chunk_id)
        embeddings.append(embedding)
        documents.append(chunk)
        metadatas.append({**metadata, "doc_id": doc_id, "chunk_index": i})

    await collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )

    return len(chunks)
