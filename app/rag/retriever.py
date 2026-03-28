"""
RAG Retriever
-------------
Given a natural-language query, return the top-k most relevant
document chunks from ChromaDB using cosine similarity over embeddings.

Design notes:
  - Filtering by doc_ids is passed directly to ChromaDB's `where` clause,
    keeping the query scoped to a subset of documents when needed.
  - In production you'd add a re-ranker (e.g. Cohere Rerank) between
    the vector search results and the LLM context window.
"""

import logging
from dataclasses import dataclass

from app.rag.embeddings import embed_text
from app.rag.ingestion import get_chroma_client, get_or_create_collection

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    chunk_id: str
    doc_id: str
    content: str
    score: float  # cosine distance (lower = more similar)
    metadata: dict


async def retrieve(
    query: str,
    top_k: int = 5,
    doc_ids: list[str] | None = None,
) -> list[RetrievedChunk]:
    """
    Embed the query and search ChromaDB for the closest chunks.

    Args:
        query:   Natural-language question.
        top_k:   Number of chunks to return.
        doc_ids: If provided, restrict search to these document IDs.

    Returns:
        List of RetrievedChunk sorted by relevance (best first).
    """
    query_embedding = await embed_text(query)

    chroma = await get_chroma_client()
    collection = await get_or_create_collection(chroma)

    where_filter = None
    if doc_ids:
        where_filter = {"doc_id": {"$in": doc_ids}}

    results = await collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=where_filter,
        include=["documents", "metadatas", "distances"],
    )

    chunks: list[RetrievedChunk] = []
    for i, chunk_id in enumerate(results["ids"][0]):
        chunks.append(
            RetrievedChunk(
                chunk_id=chunk_id,
                doc_id=results["metadatas"][0][i].get("doc_id", "unknown"),
                content=results["documents"][0][i],
                score=results["distances"][0][i],
                metadata=results["metadatas"][0][i],
            )
        )

    logger.info("[retrieve] query='%.60s...' -> %d chunks returned", query, len(chunks))
    return chunks
