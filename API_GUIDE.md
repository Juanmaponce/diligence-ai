# Diligence AI — API Usage Guide

Base URL: `http://localhost:8000` (or your deployed host)

Interactive docs: `http://localhost:8000/docs` (Swagger UI)

---

## 1. Start the services

```bash
# Start all containers (ChromaDB, Ollama, API, MCP server)
docker compose up -d

# Pull the required models (first time only)
docker compose exec ollama ollama pull nomic-embed-text
docker compose exec ollama ollama pull llama3.2
```

Wait ~30 seconds for all services to initialize.

---

## 2. Check that everything is running

```bash
curl http://localhost:8000/api/v1/health
```

Expected response:

```json
{
  "status": "healthy",
  "chromadb": "ok",
  "ollama": "ok",
  "models_available": ["nomic-embed-text:latest", "llama3.2:latest"]
}
```

Both `chromadb` and `ollama` should be `"ok"`. If either shows `"unreachable"`, that service hasn't finished starting yet — wait and retry.

> This endpoint does **not** require authentication.

---

## 3. Authentication

If `API_KEY` is set in your environment, **all endpoints except `/health`** require the `X-API-Key` header:

```bash
# Start with an API key
API_KEY=my-secret-key docker compose up -d
```

Then include the header in every request:

```bash
curl -H "X-API-Key: my-secret-key" ...
```

If `API_KEY` is empty or unset (default), authentication is **disabled** (dev mode) and no header is needed.

---

## 4. Ingest a document (raw text)

Use this when you already have the document content as plain text.

```bash
curl -X POST http://localhost:8000/api/v1/documents \
  -H "Content-Type: application/json" \
  -H "X-API-Key: YOUR_KEY" \
  -d '{
    "doc_id": "acme_10k_2024",
    "content": "Acme Corp reported revenue of $12M in FY2024, up 35% YoY...",
    "metadata": {
      "company": "Acme Corp",
      "year": 2024,
      "type": "10-K"
    }
  }'
```

Response (`201 Created`):

```json
{
  "doc_id": "acme_10k_2024",
  "chunks_stored": 5,
  "message": "Document 'acme_10k_2024' ingested successfully (5 chunks)."
}
```

**Fields:**

| Field      | Required | Description                                                  |
|------------|----------|--------------------------------------------------------------|
| `doc_id`   | Yes      | Unique ID you assign (used to scope searches later)          |
| `content`  | Yes      | Full text of the document (max 2 MB)                         |
| `metadata` | No       | Free-form dict — `company`, `year`, `type` are useful fields |

---

## 5. Upload a file (PDF / Excel)

Use this when you have a `.pdf`, `.xlsx`, or `.xls` file.

```bash
curl -X POST http://localhost:8000/api/v1/documents/upload \
  -H "X-API-Key: YOUR_KEY" \
  -F "doc_id=annual_report_2024" \
  -F 'metadata={"company":"Acme Corp","year":2024,"type":"annual_report"}' \
  -F "file=@/path/to/annual_report.pdf"
```

Response (`201 Created`):

```json
{
  "doc_id": "annual_report_2024",
  "file_type": "pdf",
  "chunks_stored": 42,
  "message": "File 'annual_report.pdf' ingested as 'annual_report_2024' (42 chunks)."
}
```

**Fields (multipart form):**

| Field      | Required | Description                                                       |
|------------|----------|-------------------------------------------------------------------|
| `file`     | Yes      | The file to upload (`.pdf`, `.xlsx`, `.xls`). Max 50 MB.          |
| `doc_id`   | Yes      | Unique ID you assign to this document                             |
| `metadata` | No       | JSON string with optional metadata (defaults to `{}`)             |

**Supported formats:**

| Format | Extension | MIME type                                                              |
|--------|-----------|------------------------------------------------------------------------|
| PDF    | `.pdf`    | `application/pdf`                                                      |
| Excel  | `.xlsx`   | `application/vnd.openxmlformats-officedocument.spreadsheetml.sheet`    |
| Legacy Excel | `.xls` | `application/vnd.ms-excel`                                          |

> **Note:** Scanned/image-based PDFs are not supported (no OCR). The PDF must contain selectable text.

---

## 6. Run a due diligence analysis

Once documents are ingested, ask natural-language questions. The ReAct agent will iteratively search the vector store, reason about the evidence, and produce a structured report.

```bash
curl -X POST http://localhost:8000/api/v1/analyze \
  -H "Content-Type: application/json" \
  -H "X-API-Key: YOUR_KEY" \
  -d '{
    "query": "What are the main financial risks for this company?",
    "doc_ids": ["acme_10k_2024"],
    "max_reasoning_steps": 5
  }'
```

Response (`200 OK`):

```json
{
  "query": "What are the main financial risks for this company?",
  "reasoning_steps": [
    {
      "step": 1,
      "thought": "I need to find sections about risk factors...",
      "action": "search: risk factors revenue concentration",
      "observation": "Found passage about 60% revenue from top 3 clients..."
    }
  ],
  "final_answer": "The primary financial risks are customer concentration...",
  "risk_level": "high",
  "key_findings": [
    "60% of revenue comes from top 3 clients",
    "Burn rate suggests 14 months of runway"
  ],
  "sources_used": ["acme_10k_2024"]
}
```

**Fields:**

| Field                  | Required | Description                                                   |
|------------------------|----------|---------------------------------------------------------------|
| `query`                | Yes      | Your due diligence question (max 10,000 chars)                |
| `doc_ids`              | No       | List of document IDs to search. `null` = search all documents |
| `max_reasoning_steps`  | No       | Number of think/search/observe cycles (1–10, default 5)       |

**Risk levels returned:** `low`, `medium`, `high`, `critical`

---

## 7. Typical workflow

```
Step 1 — Health check
  GET /api/v1/health
  Make sure status is "healthy"

Step 2 — Ingest documents
  POST /api/v1/documents          (raw text)
  POST /api/v1/documents/upload   (PDF or Excel files)
  Repeat for each document. Use meaningful doc_ids.

Step 3 — Analyze
  POST /api/v1/analyze
  Ask questions scoped to specific doc_ids or across all docs.

Step 4 — Iterate
  Refine your queries, upload more documents, re-analyze.
```

---

## 8. Error reference

| HTTP Code | Meaning                                                          |
|-----------|------------------------------------------------------------------|
| `201`     | Document ingested successfully                                   |
| `200`     | Analysis completed                                               |
| `401`     | Missing `X-API-Key` header (when auth is enabled)                |
| `403`     | Invalid API key                                                  |
| `413`     | Uploaded file exceeds the 50 MB limit                            |
| `422`     | Validation error (empty content, unsupported file type, bad JSON)|
| `502`     | Embedding or LLM service temporarily unavailable                 |
| `500`     | Internal server error                                            |

---

## 9. MCP server (for AI agents)

An MCP-compatible server runs on port `9000` for integration with tools like Claude Desktop or Cursor.

**Tools exposed:**

| Tool                | Description                                      |
|---------------------|--------------------------------------------------|
| `retrieve_context`  | RAG search — returns relevant passages            |
| `ingest_document`   | Chunk + embed + store a document                  |
| `analyze_document`  | Full ReAct due diligence analysis                 |

**Resource:** `documents://list` — lists all ingested document IDs.

**Prompt template:** `due_diligence_brief` — generates a structured DD prompt for a target company.

To connect from Claude Desktop, add to your MCP config:

```json
{
  "mcpServers": {
    "diligence-ai": {
      "url": "http://localhost:9000/sse"
    }
  }
}
```

---

## 10. Configuration (environment variables)

| Variable             | Default          | Description                        |
|----------------------|------------------|------------------------------------|
| `API_KEY`            | _(empty)_        | Set to enable authentication       |
| `CHROMA_HOST`        | `localhost`      | ChromaDB hostname                  |
| `CHROMA_PORT`        | `8001`           | ChromaDB port                      |
| `OLLAMA_HOST`        | `localhost`      | Ollama hostname                    |
| `OLLAMA_PORT`        | `11434`          | Ollama port                        |
| `EMBED_MODEL`        | `nomic-embed-text` | Embedding model name             |
| `CHAT_MODEL`         | `llama3.2`       | LLM for the ReAct agent           |
| `COLLECTION_NAME`    | `financial_docs` | ChromaDB collection name           |
| `CHUNK_SIZE`         | `512`            | Chunk size in tokens               |
| `CHUNK_OVERLAP`      | `64`             | Overlap between chunks in tokens   |
| `MAX_UPLOAD_SIZE_MB` | `50`             | Max file upload size               |
