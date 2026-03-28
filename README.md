# Diligence AI — AI Due Diligence Assistant

A production-oriented RAG + ReAct agent for financial document analysis,
built to demonstrate applied AI engineering skills (MCP-ready, RAG, ReAct,
FastAPI, Docker).

## Architecture

```
POST /api/v1/documents          POST /api/v1/analyze
        │                               │
        ▼                               ▼
  [Ingestion Pipeline]         [ReAct Agent Loop]
  ┌─────────────────┐         ┌──────────────────────┐
  │ 1. Chunk text   │         │ THOUGHT: what do I   │
  │    (512 tokens) │         │ need to know?        │
  │ 2. Embed via    │◄────────│ ACTION: retrieve_    │
  │    Ollama       │         │ context("query")     │
  │    (nomic-      │         │ OBSERVATION: chunks  │
  │    embed-text)  │         │ ...repeat...         │
  │ 3. Store in     │         │ FINAL_ANSWER + risk  │
  │    ChromaDB     │         └──────────────────────┘
  └─────────────────┘
         │                              │
         └──────────┬───────────────────┘
                    ▼
              [ChromaDB]
              Vector Store
                    │
              [Ollama LLM]
              llama3.2 / nomic-embed-text
```

## Stack
| Layer | Technology |
|---|---|
| API | FastAPI + Pydantic v2 |
| Vector store | ChromaDB (cosine similarity) |
| Embeddings | Ollama — `nomic-embed-text` |
| LLM | Ollama — `llama3.2` |
| Reasoning | ReAct pattern (Thought/Action/Observation loop) |
| Infrastructure | Docker Compose |

## Quickstart

### 1. Start services
```bash
docker compose up -d
```

### 2. Pull required models (one-time)
```bash
# In a new terminal:
docker compose exec ollama ollama pull nomic-embed-text
docker compose exec ollama ollama pull llama3.2
```

### 3. Verify everything is running
```bash
curl http://localhost:8000/api/v1/health
```

Expected: `{"status":"healthy","chromadb":"ok","ollama":"ok",...}`

### 4. Run the integration test
```bash
pip install httpx
python test_pipeline.py
```

### 5. Or use the interactive docs
Open http://localhost:8000/docs in your browser.

## Example API calls

**Ingest a document:**
```bash
curl -X POST http://localhost:8000/api/v1/documents \
  -H "Content-Type: application/json" \
  -d '{
    "doc_id": "acme_2024",
    "content": "Revenue grew 38% to $12.4M...",
    "metadata": {"company": "Acme Cleantech", "year": 2024}
  }'
```

**Run due diligence analysis:**
```bash
curl -X POST http://localhost:8000/api/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the customer concentration risk?",
    "doc_ids": ["acme_2024"],
    "max_reasoning_steps": 5
  }'
```

## Why this architecture?

- **RAG over fine-tuning**: Financial docs change constantly; RAG lets us update
  the knowledge base without retraining. Embeddings in ChromaDB can be refreshed
  per document without touching the LLM.

- **ReAct over single-shot prompting**: Due diligence requires multi-step reasoning.
  ReAct makes each step auditable — you can see exactly what the model retrieved
  and why it reached its conclusion.

- **MCP-ready**: The `retrieve_context` tool in the agent is designed to be extracted
  into an MCP server, making it composable with other agents (e.g., one agent for
  financial analysis, another for legal review, a third for market research).

- **Ollama for local dev**: Zero API costs, works offline, swap to OpenAI/Gemini
  by changing two env vars (`OLLAMA_HOST` → OpenAI base URL, model names).

---

## MCP Server

The `mcp_server/` directory exposes the full RAG + ReAct pipeline as a
**Model Context Protocol server** — making it composable with any MCP-compatible
host: Claude Desktop, Cursor, custom agents, or other MCP servers.

### Architecture

```
MCP Host (Claude Desktop / custom agent)
        │
        │  SSE  (port 9000)
        ▼
┌──────────────────────────────────────┐
│       diligence-ai MCP Server        │
│                                      │
│  Tools:                              │
│    retrieve_context  ──► ChromaDB    │
│    ingest_document   ──► FastAPI     │
│    analyze_document  ──► ReAct Agent │
│                                      │
│  Resources:                          │
│    documents://list  ──► ChromaDB    │
│                                      │
│  Prompts:                            │
│    due_diligence_brief               │
└──────────────────────────────────────┘
```

### Running the MCP server

```bash
# Start all services including MCP server
docker compose up -d

# Verify MCP server is up
curl http://localhost:9000/sse
```

### Testing with the MCP client

```bash
pip install "mcp[cli]"
python test_mcp_client.py
```

### Connecting Claude Desktop (stdio mode)

For local use with Claude Desktop, switch to stdio transport.
Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "diligence-ai": {
      "command": "docker",
      "args": ["compose", "exec", "-T", "mcp", "python", "server.py"],
      "env": {
        "CHROMA_HOST": "localhost",
        "OLLAMA_HOST": "localhost"
      }
    }
  }
}
```

Then change the last line of `mcp_server/server.py` to:
```python
mcp.run(transport="stdio")
```

### Why MCP matters for this role

The job description mentions MCP explicitly. The key insight here is that
wrapping `retrieve_context` as an MCP tool makes it **composable**:

- A legal review agent can use the same tool without knowing about ChromaDB
- A market research agent can call `ingest_document` to add competitor reports
- The `due_diligence_brief` prompt becomes a reusable workflow template
- Multiple specialized agents can share one knowledge base through one protocol

---

## Next steps (production hardening)
- [ ] PDF ingestion via `unstructured.io`
- [ ] Re-ranking with Cohere Rerank before passing chunks to LLM
- [ ] Expose `retrieve_context` as an MCP server tool
- [ ] Auth middleware (API keys / OAuth)
- [ ] Async job queue (Celery/Redis) for long ingestion jobs
- [ ] Structured logging → CloudWatch / Datadog
