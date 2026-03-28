from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class IngestRequest(BaseModel):
    content: str = Field(..., max_length=2_000_000, description="Raw text content of the financial document (max 2MB)")
    doc_id: str = Field(..., description="Unique identifier for the document")
    metadata: dict = Field(default_factory=dict, description="Optional metadata (company, year, type...)")


class IngestResponse(BaseModel):
    doc_id: str
    chunks_stored: int
    message: str


class AnalysisRequest(BaseModel):
    query: str = Field(..., max_length=10_000, description="The due diligence question to analyze")
    doc_ids: Optional[list[str]] = Field(
        default=None,
        description="Limit search to specific document IDs. None = search all."
    )
    max_reasoning_steps: int = Field(default=5, ge=1, le=10)


class ReasoningStep(BaseModel):
    step: int
    thought: str
    action: str
    observation: str


class AnalysisResponse(BaseModel):
    query: str
    reasoning_steps: list[ReasoningStep]
    final_answer: str
    risk_level: RiskLevel
    key_findings: list[str]
    sources_used: list[str]


class HealthResponse(BaseModel):
    status: str
    chromadb: str
    ollama: str
    models_available: list[str]
