"""
Query Router - API endpoints for RAG queries
"""
from typing import List, Optional, Literal
from fastapi import APIRouter
from pydantic import BaseModel

from app.services.query import get_query_service

router = APIRouter(prefix="/query", tags=["query"])


class Source(BaseModel):
    issue_id: Optional[str]
    title: str
    date: str
    similarity: Optional[float]
    type: Optional[str] = None  # "distillation" or "chunk"
    thesis_statement: Optional[str] = None  # ADD
    topics: Optional[List[str]] = None  # ADD
    content: Optional[str] = None  # ADD (for chunks)


class QueryRequest(BaseModel):
    question: str
    limit: int = 5
    threshold: float = 0.3
    mode: Literal["auto", "distillations", "chunks", "hybrid", "reasoning"] = "auto"


class QueryResponse(BaseModel):
    answer: str
    sources: List[Source]
    query: str
    model: str
    retrieval_tier: str


@router.post("/", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Ask a question and get a Ben Thompson-style analysis."""
    service = get_query_service()

    # Use reasoning-first mode if specified
    if request.mode == "reasoning":
        result = service.query_reasoning(
            question=request.question,
            limit=request.limit,
            threshold=request.threshold
        )
    else:
        result = service.query(
            question=request.question,
            limit=request.limit,
            threshold=request.threshold,
            mode=request.mode
        )

    return {
        "answer": result.answer,
        "sources": result.sources,
        "query": result.query,
        "model": result.model,
        "retrieval_tier": result.retrieval_tier
    }


@router.get("/")
async def query_get(
    q: str, 
    limit: int = 5, 
    threshold: float = 0.3,
    mode: str = "auto"
):
    """GET endpoint for simple queries."""
    service = get_query_service()
    result = service.query(question=q, limit=limit, threshold=threshold, mode=mode)
    return {
        "answer": result.answer,
        "sources": result.sources,
        "query": result.query,
        "model": result.model,
        "retrieval_tier": result.retrieval_tier
    }