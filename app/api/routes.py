import json
import logging

import httpx
from fastapi import APIRouter, Depends, Form, HTTPException, UploadFile

from app.agent.analyst import run_analysis
from app.auth import require_api_key
from app.config import settings
from app.rag.extractors import FileType, detect_file_type, extract_text
from app.rag.ingestion import ingest_document
from app.schemas import (
    AnalysisRequest,
    AnalysisResponse,
    FileUploadResponse,
    HealthResponse,
    IngestRequest,
    IngestResponse,
)

# Public routes (no auth required — for load balancer probes, etc.)
public_router = APIRouter()

# Authenticated routes
router = APIRouter(dependencies=[Depends(require_api_key)])

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Health (public)
# ---------------------------------------------------------------------------

@public_router.get("/health", response_model=HealthResponse, tags=["ops"])
async def health_check():
    """
    Verify connectivity to ChromaDB and Ollama.
    Returns available LLM models so the caller can confirm the right
    model is loaded.
    """
    chroma_status = "unreachable"
    ollama_status = "unreachable"
    models: list[str] = []

    async with httpx.AsyncClient(timeout=5) as client:
        try:
            r = await client.get(f"{settings.chroma_base_url}/api/v1/heartbeat")
            chroma_status = "ok" if r.status_code == 200 else f"error:{r.status_code}"
        except Exception:
            logger.warning("ChromaDB health check failed", exc_info=True)

        try:
            r = await client.get(f"{settings.ollama_base_url}/api/tags")
            if r.status_code == 200:
                ollama_status = "ok"
                models = [m["name"] for m in r.json().get("models", [])]
            else:
                ollama_status = f"error:{r.status_code}"
        except Exception:
            logger.warning("Ollama health check failed", exc_info=True)

    overall = "healthy" if chroma_status == "ok" and ollama_status == "ok" else "degraded"
    return HealthResponse(
        status=overall,
        chromadb=chroma_status,
        ollama=ollama_status,
        models_available=models,
    )


# ---------------------------------------------------------------------------
# Document ingestion (authenticated)
# ---------------------------------------------------------------------------

@router.post("/documents", response_model=IngestResponse, status_code=201, tags=["rag"])
async def ingest(body: IngestRequest):
    """
    Ingest a financial document into the vector store.

    The document is chunked (512 tokens, 64 overlap), embedded via
    Ollama (nomic-embed-text), and stored in ChromaDB.

    Example metadata you might pass:
      { "company": "Acme Corp", "year": 2024, "type": "10-K" }
    """
    if not body.content.strip():
        raise HTTPException(status_code=422, detail="Document content cannot be empty.")

    try:
        chunks_stored = await ingest_document(
            content=body.content,
            doc_id=body.doc_id,
            metadata=body.metadata,
        )
    except httpx.HTTPError:
        logger.exception("Embedding service error during ingestion")
        raise HTTPException(status_code=502, detail="Embedding service is temporarily unavailable.")
    except Exception:
        logger.exception("Unexpected error during ingestion")
        raise HTTPException(status_code=500, detail="An internal error occurred. Please try again later.")

    return IngestResponse(
        doc_id=body.doc_id,
        chunks_stored=chunks_stored,
        message=f"Document '{body.doc_id}' ingested successfully ({chunks_stored} chunks).",
    )


# ---------------------------------------------------------------------------
# File upload (authenticated)
# ---------------------------------------------------------------------------

@router.post(
    "/documents/upload",
    response_model=FileUploadResponse,
    status_code=201,
    tags=["rag"],
)
async def upload_document(
    file: UploadFile,
    doc_id: str = Form(..., description="Unique identifier for the document"),
    metadata: str = Form(
        default="{}",
        description='JSON string with optional metadata, e.g. {"company":"Acme","year":2024}',
    ),
):
    """
    Upload a PDF or Excel file (.pdf, .xlsx, .xls) for ingestion.

    The file is parsed into text, then chunked, embedded, and stored
    in the vector database — just like the /documents endpoint but
    accepting binary files instead of raw text.
    """
    # --- validate metadata JSON ---
    try:
        meta = json.loads(metadata)
        if not isinstance(meta, dict):
            raise ValueError
    except (json.JSONDecodeError, ValueError):
        raise HTTPException(status_code=422, detail="metadata must be a valid JSON object.")

    # --- validate file size ---
    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    file_bytes = await file.read()
    if len(file_bytes) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds the {settings.max_upload_size_mb} MB limit.",
        )
    if not file_bytes:
        raise HTTPException(status_code=422, detail="Uploaded file is empty.")

    # --- detect and validate file type ---
    try:
        file_type = detect_file_type(file.filename or "", file.content_type)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    # --- extract text ---
    try:
        text = extract_text(file_bytes, file_type)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception:
        logger.exception("Failed to extract text from uploaded file")
        raise HTTPException(
            status_code=500,
            detail="Failed to extract text from the uploaded file.",
        )

    # --- ingest into vector store ---
    try:
        chunks_stored = await ingest_document(
            content=text,
            doc_id=doc_id,
            metadata=meta,
        )
    except httpx.HTTPError:
        logger.exception("Embedding service error during file upload ingestion")
        raise HTTPException(status_code=502, detail="Embedding service is temporarily unavailable.")
    except Exception:
        logger.exception("Unexpected error during file upload ingestion")
        raise HTTPException(status_code=500, detail="An internal error occurred. Please try again later.")

    return FileUploadResponse(
        doc_id=doc_id,
        file_type=file_type.value,
        chunks_stored=chunks_stored,
        message=f"File '{file.filename}' ingested as '{doc_id}' ({chunks_stored} chunks).",
    )


# ---------------------------------------------------------------------------
# Analysis (authenticated)
# ---------------------------------------------------------------------------

@router.post("/analyze", response_model=AnalysisResponse, tags=["agent"])
async def analyze(body: AnalysisRequest):
    """
    Run the ReAct due diligence agent on a natural-language query.

    The agent will:
      1. Reason about what information it needs (THOUGHT)
      2. Query the vector store for relevant passages (ACTION)
      3. Synthesize findings into a structured response with risk level

    Set `doc_ids` to limit the search scope to specific documents.
    """
    if not body.query.strip():
        raise HTTPException(status_code=422, detail="Query cannot be empty.")

    try:
        result = await run_analysis(
            query=body.query,
            doc_ids=body.doc_ids,
            max_steps=body.max_reasoning_steps,
        )
    except httpx.HTTPError:
        logger.exception("LLM service error during analysis")
        raise HTTPException(status_code=502, detail="LLM service is temporarily unavailable.")
    except Exception:
        logger.exception("Unexpected error during analysis")
        raise HTTPException(status_code=500, detail="An internal error occurred. Please try again later.")

    return result
