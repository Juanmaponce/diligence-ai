"""
Diligence AI — MCP Server
======================
Exposes the RAG + ReAct due diligence pipeline as a standard
Model Context Protocol server.

Any MCP-compatible host (Claude Desktop, Cursor, custom agents)
can connect to this server and use the tools below without knowing
anything about ChromaDB, Ollama, or the internal RAG implementation.

Transport: SSE (HTTP) on port 9000 — suitable for networked agents.
For local stdio use (e.g. Claude Desktop), see README.

Primitives exposed:
  Tools (actions):
    · retrieve_context    — RAG search, returns relevant passages
    · ingest_document     — chunk + embed + store a document
    · analyze_document    — full ReAct due diligence agent

  Resources (read-only data):
    · documents://list    — list all ingested document IDs

  Prompts (reusable templates):
    · due_diligence_brief — structured DD prompt for a given company/query
"""

import json
import logging
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import AsyncIterator

import httpx
import chromadb
from mcp.server.fastmcp import FastMCP, Context

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config (from env, with sensible defaults for local dev)
# ---------------------------------------------------------------------------

CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8001"))
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "localhost")
OLLAMA_PORT = int(os.getenv("OLLAMA_PORT", "11434"))
FASTAPI_HOST = os.getenv("FASTAPI_HOST", "localhost")
FASTAPI_PORT = int(os.getenv("FASTAPI_PORT", "8000"))
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "financial_docs")
API_KEY = os.getenv("API_KEY", "")

OLLAMA_BASE = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}"
FASTAPI_BASE = f"http://{FASTAPI_HOST}:{FASTAPI_PORT}/api/v1"


# ---------------------------------------------------------------------------
# Lifespan — shared clients initialized once at startup
# ---------------------------------------------------------------------------

@dataclass
class AppContext:
    chroma: chromadb.AsyncHttpClient
    http: httpx.AsyncClient


_app_context: AppContext | None = None


@asynccontextmanager
async def lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    """Initialize shared resources on startup, clean up on shutdown."""
    global _app_context
    logger.info("MCP server starting — connecting to ChromaDB and Ollama...")

    chroma = await chromadb.AsyncHttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    headers = {"X-API-Key": API_KEY} if API_KEY else {}
    http = httpx.AsyncClient(timeout=120, headers=headers)

    _app_context = AppContext(chroma=chroma, http=http)
    try:
        yield _app_context
    finally:
        _app_context = None
        await http.aclose()
        logger.info("MCP server shut down cleanly.")


# ---------------------------------------------------------------------------
# FastMCP server
# ---------------------------------------------------------------------------

mcp = FastMCP(
    name="diligence-ai",
    instructions=(
        "You are connected to the Diligence AI Due Diligence system. "
        "Use retrieve_context to search financial documents, "
        "ingest_document to add new documents, and analyze_document "
        "to run a full multi-step due diligence analysis."
    ),
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# TOOL 1: retrieve_context
# ---------------------------------------------------------------------------

@mcp.tool()
async def retrieve_context(
    query: str,
    top_k: int = 5,
    doc_ids: list[str] | None = None,
    ctx: Context = None,
) -> str:
    """
    Search the financial document vector store and return the most
    relevant passages for a given query.

    Use this tool when you need specific facts, figures, or sections
    from ingested financial documents before forming a conclusion.

    Args:
        query:   Natural-language search query (be specific for best results).
                 Examples: "customer churn rate Q4", "total debt maturity date"
        top_k:   Number of passages to return (1–10, default 5).
        doc_ids: Optional list of document IDs to restrict the search scope.
                 If omitted, searches across all ingested documents.

    Returns:
        Formatted string with matching passages and their source references.
    """
    app: AppContext = ctx.request_context.lifespan_context

    # Embed the query
    embed_response = await app.http.post(
        f"{OLLAMA_BASE}/api/embeddings",
        json={"model": EMBED_MODEL, "prompt": query},
    )
    embed_response.raise_for_status()
    query_vector = embed_response.json()["embedding"]

    # Query ChromaDB
    collection = await app.chroma.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    where_filter = {"doc_id": {"$in": doc_ids}} if doc_ids else None

    results = await collection.query(
        query_embeddings=[query_vector],
        n_results=min(top_k, 10),
        where=where_filter,
        include=["documents", "metadatas", "distances"],
    )

    if not results["ids"][0]:
        return "No relevant passages found. Make sure documents have been ingested first."

    # Format output for LLM consumption
    output_parts = [f"Found {len(results['ids'][0])} relevant passages:\n"]
    for i, (doc_id_result, doc, meta, dist) in enumerate(zip(
        results["ids"][0],
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    )):
        source_doc = meta.get("doc_id", "unknown")
        chunk_idx = meta.get("chunk_index", "?")
        relevance = round((1 - dist) * 100, 1)  # convert cosine distance → similarity %

        output_parts.append(
            f"--- Passage {i+1} | Source: {source_doc} (chunk {chunk_idx}) | Relevance: {relevance}% ---\n"
            f"{doc}\n"
        )

    return "\n".join(output_parts)


# ---------------------------------------------------------------------------
# TOOL 2: ingest_document
# ---------------------------------------------------------------------------

@mcp.tool()
async def ingest_document(
    doc_id: str,
    content: str,
    company: str | None = None,
    year: int | None = None,
    doc_type: str | None = None,
    ctx: Context = None,
) -> str:
    """
    Ingest a financial document into the due diligence knowledge base.

    The document will be chunked, embedded, and stored in the vector
    database so it can be searched with retrieve_context or analyzed
    with analyze_document.

    Args:
        doc_id:   Unique identifier (e.g. "acme_10k_2024"). Use consistent
                  IDs so you can scope searches to specific documents later.
        content:  Full text content of the document.
        company:  Company name (for metadata filtering).
        year:     Report year (for metadata filtering).
        doc_type: Document type, e.g. "10-K", "pitch_deck", "term_sheet".

    Returns:
        Confirmation message with number of chunks stored.
    """
    app: AppContext = ctx.request_context.lifespan_context

    metadata = {}
    if company:
        metadata["company"] = company
    if year:
        metadata["year"] = year
    if doc_type:
        metadata["doc_type"] = doc_type

    response = await app.http.post(
        f"{FASTAPI_BASE}/documents",
        json={"doc_id": doc_id, "content": content, "metadata": metadata},
    )

    if response.status_code == 201:
        data = response.json()
        return (
            f"✓ Document '{doc_id}' ingested successfully.\n"
            f"  Chunks stored: {data['chunks_stored']}\n"
            f"  Metadata: {metadata or 'none'}\n"
            f"  Ready for retrieval and analysis."
        )
    else:
        return f"✗ Ingestion failed (HTTP {response.status_code}): {response.text}"


# ---------------------------------------------------------------------------
# TOOL 3: analyze_document
# ---------------------------------------------------------------------------

@mcp.tool()
async def analyze_document(
    query: str,
    doc_ids: list[str] | None = None,
    max_reasoning_steps: int = 5,
    ctx: Context = None,
) -> str:
    """
    Run a full multi-step due diligence analysis using a ReAct reasoning agent.

    The agent will iteratively retrieve relevant passages from the document
    store, reason about the evidence, and produce a structured conclusion
    with risk assessment and key findings.

    Use this for complex due diligence questions that require synthesizing
    information from multiple sections of a document. For simple lookups,
    prefer retrieve_context (faster).

    Args:
        query:               The due diligence question to analyze.
                             Examples:
                             - "What are the main financial risks?"
                             - "Is the customer concentration a concern?"
                             - "Assess the competitive moat of this company."
        doc_ids:             Restrict analysis to specific document IDs.
                             If omitted, searches all ingested documents.
        max_reasoning_steps: How many Thought/Action/Observation cycles
                             the agent may run (1–10, default 5).

    Returns:
        Structured analysis with reasoning steps, risk level, and key findings.
    """
    app: AppContext = ctx.request_context.lifespan_context

    await ctx.info(f"Running ReAct analysis: '{query[:60]}...'")
    await ctx.report_progress(0, max_reasoning_steps)

    response = await app.http.post(
        f"{FASTAPI_BASE}/analyze",
        json={
            "query": query,
            "doc_ids": doc_ids,
            "max_reasoning_steps": max_reasoning_steps,
        },
    )

    if response.status_code != 200:
        return f"Analysis failed (HTTP {response.status_code}): {response.text}"

    data = response.json()

    await ctx.report_progress(max_reasoning_steps, max_reasoning_steps)

    # Format structured response for LLM consumption
    lines = [
        f"# Due Diligence Analysis\n",
        f"**Query:** {data['query']}\n",
        f"**Risk Level:** {data['risk_level'].upper()}\n",
        "\n## Reasoning Steps\n",
    ]

    for step in data.get("reasoning_steps", []):
        lines.append(f"**Step {step['step']}**")
        lines.append(f"- Thought: {step['thought']}")
        lines.append(f"- Action: {step['action']}")
        lines.append(f"- Observation: {step['observation'][:200]}...")
        lines.append("")

    lines.append("\n## Key Findings\n")
    for finding in data.get("key_findings", []):
        lines.append(f"• {finding}")

    lines.append(f"\n## Conclusion\n{data['final_answer']}")

    if data.get("sources_used"):
        lines.append(f"\n**Sources:** {', '.join(data['sources_used'])}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# RESOURCE: documents://list
# ---------------------------------------------------------------------------

@mcp.resource("documents://list")
async def list_documents() -> str:
    """
    List all document IDs currently stored in the knowledge base.

    Use this resource to discover what documents are available before
    calling retrieve_context or analyze_document with specific doc_ids.
    """
    if _app_context is None:
        return json.dumps({"error": "Server not initialized", "documents": []})

    try:
        collection = await _app_context.chroma.get_or_create_collection(name=COLLECTION_NAME)
        result = await collection.get(include=["metadatas"])

        if not result["metadatas"]:
            return json.dumps({"documents": [], "total_chunks": 0})

        # Aggregate unique doc_ids with chunk counts
        doc_counts: dict[str, int] = {}
        doc_meta: dict[str, dict] = {}
        for meta in result["metadatas"]:
            doc_id = meta.get("doc_id", "unknown")
            doc_counts[doc_id] = doc_counts.get(doc_id, 0) + 1
            if doc_id not in doc_meta:
                doc_meta[doc_id] = {k: v for k, v in meta.items() if k != "chunk_index"}

        docs = [
            {"doc_id": doc_id, "chunks": count, "metadata": doc_meta[doc_id]}
            for doc_id, count in doc_counts.items()
        ]

        return json.dumps({"documents": docs, "total_chunks": sum(doc_counts.values())}, indent=2)

    except Exception:
        logger.exception("Failed to list documents")
        return json.dumps({"error": "Failed to list documents", "documents": []})


# ---------------------------------------------------------------------------
# PROMPT: due_diligence_brief
# ---------------------------------------------------------------------------

@mcp.prompt()
def due_diligence_brief(company_name: str, focus_area: str = "overall") -> str:
    """
    Generate a structured due diligence prompt for a target company.

    This prompt template guides a systematic analysis covering the key
    areas that matter in financial due diligence.

    Args:
        company_name: Name of the company being analyzed.
        focus_area:   One of: "overall", "financial", "risk", "competitive",
                      "management", "market". Defaults to "overall".
    """
    focus_instructions = {
        "overall": (
            "Provide a comprehensive due diligence assessment covering: "
            "financial health, key risks, competitive position, and growth outlook."
        ),
        "financial": (
            "Focus on: revenue quality (ARR vs one-time), gross margins, burn rate, "
            "runway, unit economics (CAC, LTV, payback period), and debt obligations."
        ),
        "risk": (
            "Identify and assess: customer concentration risk, regulatory exposure, "
            "key person dependencies, competitive threats, and market timing risks."
        ),
        "competitive": (
            "Analyze: defensibility of the business model, moat (network effects, "
            "switching costs, IP), direct competitors, and big-tech encroachment risk."
        ),
        "management": (
            "Evaluate: founding team background, key hires, retention risks, "
            "board composition, and any disclosed conflicts of interest."
        ),
        "market": (
            "Assess: total addressable market size and growth rate, "
            "go-to-market strategy effectiveness, and expansion opportunities."
        ),
    }

    instruction = focus_instructions.get(focus_area, focus_instructions["overall"])

    return f"""You are a senior financial analyst conducting due diligence on **{company_name}**.

{instruction}

Use the available tools in this order:
1. Call `list_documents` resource to see what documents are available.
2. Call `retrieve_context` with specific queries to gather evidence on each dimension.
3. Call `analyze_document` with your synthesized question for the final structured assessment.

Structure your final report as:
## Executive Summary
## Financial Assessment  
## Risk Matrix (table: Risk | Likelihood | Impact | Mitigation)
## Competitive Position
## Recommendation (Proceed / Proceed with conditions / Pass)
## Open Questions for Management

Be specific, cite sources, and flag any missing information as a risk factor."""


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # SSE transport: accessible over the network (for Docker, remote agents)
    # Switch to mcp.run(transport="stdio") for Claude Desktop local mode
    mcp.run(transport="sse", port=9000)
