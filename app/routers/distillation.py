"""
Distillation Router - API endpoints for generating and managing distillations.
"""
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services.distillation import get_distillation_service

router = APIRouter(prefix="/distill", tags=["distillation"])


class DistillationResponse(BaseModel):
    """Response for single distillation."""
    success: bool
    distillation_id: Optional[str] = None
    issue_id: Optional[str] = None
    title: Optional[str] = None
    error: Optional[str] = None
    skipped: bool = False
    skip_reason: Optional[str] = None


class BatchDistillationResponse(BaseModel):
    """Response for batch distillation."""
    total: int
    successful: int
    skipped: int
    failed: int
    message: str


class StatsResponse(BaseModel):
    """Response for distillation stats."""
    total_issues: int
    total_distillations: int
    pending: int
    reviewed: int
    completion_rate: float


@router.post("/article/{issue_id}", response_model=DistillationResponse)
async def distill_single_article(issue_id: str, force: bool = False):
    """
    Generate a distillation for a single article.
    
    Args:
        issue_id: UUID of the article to distill
        force: If True, regenerate even if distillation exists
    """
    service = get_distillation_service()
    result = service.distill_article(issue_id, force=force)
    
    return DistillationResponse(
        success=result.success,
        distillation_id=result.distillation_id,
        issue_id=result.issue_id,
        title=result.title,
        error=result.error,
        skipped=result.skipped,
        skip_reason=result.skip_reason
    )


@router.post("/batch", response_model=BatchDistillationResponse)
async def distill_batch(
    limit: int = 10, 
    force: bool = False,
    oldest_first: bool = True
):
    """
    Distill a batch of articles that don't have distillations yet.
    
    Args:
        limit: Maximum number of articles to process (default 10)
        force: If True, regenerate all distillations
        oldest_first: If True, process oldest articles first
    
    Note: Each article costs ~$0.01-0.05 in GPT-4o API calls.
    Start with small batches to verify quality.
    """
    service = get_distillation_service()
    result = service.distill_batch(limit=limit, force=force, oldest_first=oldest_first)
    
    return BatchDistillationResponse(
        total=result.total,
        successful=result.successful,
        skipped=result.skipped,
        failed=result.failed,
        message=f"Processed {result.total} articles: {result.successful} successful, {result.skipped} skipped, {result.failed} failed"
    )


@router.get("/stats", response_model=StatsResponse)
async def get_distillation_stats():
    """Get statistics about distillation progress."""
    service = get_distillation_service()
    stats = service.get_stats()
    
    return StatsResponse(**stats)


@router.get("/article/{issue_id}")
async def get_distillation(issue_id: str):
    """
    Get the distillation for a specific article.
    """
    service = get_distillation_service()
    
    result = service.supabase.table("stratechery_distillations")\
        .select("*")\
        .eq("issue_id", issue_id)\
        .execute()
    
    if not result.data:
        raise HTTPException(status_code=404, detail="Distillation not found")
    
    return result.data[0]


@router.get("/recent")
async def get_recent_distillations(limit: int = 10):
    """
    Get the most recently created distillations.
    """
    service = get_distillation_service()
    
    result = service.supabase.table("stratechery_distillations")\
        .select("*, stratechery_issues(title, publication_date)")\
        .order("created_at", desc=True)\
        .limit(limit)\
        .execute()
    
    return result.data


@router.get("/search")
async def search_distillations(
    topic: Optional[str] = None,
    company: Optional[str] = None,
    person: Optional[str] = None,
    limit: int = 20
):
    """
    Search distillations by topic, company, or person.
    """
    service = get_distillation_service()
    
    query = service.supabase.table("stratechery_distillations")\
        .select("distillation_id, issue_id, thesis_statement, topics, entities, stratechery_issues(title, publication_date)")
    
    if topic:
        query = query.contains("topics", [topic])
    
    # Note: JSONB queries for company/person would need custom SQL or RPC
    # For now, fetch and filter in Python
    
    result = query.limit(limit).execute()
    
    data = result.data or []
    
    # Filter by company or person if specified
    if company:
        data = [
            d for d in data 
            if company.lower() in str(d.get("entities", {}).get("companies", [])).lower()
        ]
    
    if person:
        data = [
            d for d in data 
            if person.lower() in str(d.get("entities", {}).get("people", [])).lower()
        ]
    
    return data
