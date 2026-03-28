import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import public_router, router
from app.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)

app = FastAPI(
    title="Diligence AI — AI Due Diligence Assistant",
    description=(
        "Financial document analysis powered by RAG + ReAct reasoning. "
        "Ingest documents, then query them with natural language due diligence questions."
    ),
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type", "X-API-Key"],
)

app.include_router(public_router, prefix="/api/v1")
app.include_router(router, prefix="/api/v1")


@app.get("/", include_in_schema=False)
async def root():
    return {"message": "Diligence AI API — visit /docs for interactive documentation."}
