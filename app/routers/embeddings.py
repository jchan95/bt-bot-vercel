"""
Embeddings Router - API endpoints for generating and searching embeddings.
"""
from typing import Optional, List
from fastapi import APIRouter
from pydantic import BaseModel

from app.services.embeddings import get_embeddings_service

router = APIRouter(prefix="/embeddings", tags=["embeddings"])


class StatsResponse(BaseModel):
    total_issues: int
    total_chunks: int
    chunk_embeddings: int
    chunks_pending_embedding: int
    total_distillations: int
    distillation_embeddings: int
    distillations_pending_embedding: int


class SearchResult(BaseModel):
    results: List[dict]
    query: str


@router.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Get embedding statistics."""
    service = get_embeddings_service()
    return service.get_stats()


@router.post("/chunks/generate")
async def chunk_articles(limit: int = 50):
    """Chunk articles that haven't been chunked yet."""
    service = get_embeddings_service()
    return service.chunk_batch(limit=limit)


@router.post("/chunks/embed")
async def embed_chunks(limit: int = 100):
    """Generate embeddings for chunks."""
    service = get_embeddings_service()
    return service.embed_chunks(limit=limit)


@router.post("/distillations/embed")
async def embed_distillations(limit: int = 100):
    """Generate embeddings for distillations."""
    service = get_embeddings_service()
    return service.embed_distillations(limit=limit)


@router.get("/search/chunks")
async def search_chunks(q: str, limit: int = 5, threshold: float = 0.7):
    """Search chunks by semantic similarity."""
    service = get_embeddings_service()
    results = service.search_chunks(q, limit=limit, threshold=threshold)
    return {"query": q, "results": results}


@router.get("/search/distillations")
async def search_distillations(q: str, limit: int = 5, threshold: float = 0.7):
    """Search distillations by semantic similarity."""
    service = get_embeddings_service()
    results = service.search_distillations(q, limit=limit, threshold=threshold)
    return {"query": q, "results": results}