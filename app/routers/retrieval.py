"""
Retrieval Router - API endpoints for inspecting RAG retrieval
Exposes the two-tier retrieval process for educational/demo purposes
"""
from typing import List, Optional
from fastapi import APIRouter
from pydantic import BaseModel

from app.services.embeddings import get_embeddings_service
from app.database import get_supabase_client

router = APIRouter(prefix="/retrieval", tags=["retrieval"])


# ============== Request/Response Models ==============

class InspectRequest(BaseModel):
    query: str
    limit: int = 5
    threshold: float = 0.3


class KeyClaim(BaseModel):
    claim: str
    evidence: Optional[str] = None


class IncentiveAnalysis(BaseModel):
    who_benefits: List[str] = []
    who_loses: List[str] = []
    key_incentives: List[str] = []


class DistillationResult(BaseModel):
    distillation_id: str
    issue_id: str
    title: str
    publication_date: Optional[str]
    similarity: float
    above_threshold: bool
    # Structured distillation data
    thesis_statement: str
    key_claims: List[KeyClaim] = []
    topics: List[str] = []
    entities: dict = {}
    incentive_analysis: Optional[IncentiveAnalysis] = None
    predictions: List[dict] = []
    confidence_score: Optional[float] = None


class ChunkResult(BaseModel):
    chunk_id: str
    issue_id: str
    title: str
    publication_date: Optional[str]
    similarity: float
    above_threshold: bool
    chunk_index: int
    content: str
    token_count: Optional[int] = None


class RetrievalDecision(BaseModel):
    """Explains the retrieval routing decision"""
    query: str
    mode_requested: str
    needs_precision: bool
    distillation_max_score: Optional[float]
    chunk_max_score: Optional[float]
    tier_used: str
    reasoning: str


class InspectResponse(BaseModel):
    query: str
    threshold: float
    decision: RetrievalDecision
    distillations: List[DistillationResult]
    chunks: List[ChunkResult]
    # Summary stats
    distillation_count: int
    chunk_count: int
    distillations_above_threshold: int
    chunks_above_threshold: int


# ============== Helper Functions ==============

def needs_precision(question: str) -> bool:
    """Detect if question needs chunk-level precision."""
    precision_keywords = [
        "exact", "specific", "quote", "said", "wrote",
        "number", "percent", "statistic", "data",
        "when did", "what date", "how much"
    ]
    question_lower = question.lower()
    return any(kw in question_lower for kw in precision_keywords)


def get_article_metadata(supabase, issue_ids: List[str]) -> dict:
    """Fetch article titles and dates."""
    if not issue_ids:
        return {}

    result = supabase.table("stratechery_issues")\
        .select("issue_id, title, publication_date")\
        .in_("issue_id", list(set(issue_ids)))\
        .execute()

    return {r["issue_id"]: r for r in result.data} if result.data else {}


# ============== API Endpoints ==============

@router.post("/inspect", response_model=InspectResponse)
async def inspect_retrieval(request: InspectRequest):
    """
    Inspect the two-tier retrieval process.

    Returns both distillation and chunk results with similarity scores,
    along with the decision logic for tier selection.
    """
    embeddings_service = get_embeddings_service()
    supabase = get_supabase_client()

    # Search both tiers
    raw_distillations = embeddings_service.search_distillations(
        request.query,
        limit=request.limit,
        threshold=0.0  # Get all results, we'll filter in response
    )

    raw_chunks = embeddings_service.search_chunks(
        request.query,
        limit=request.limit,
        threshold=0.0
    )

    # Get article metadata
    issue_ids = []
    issue_ids.extend([d.get("issue_id") for d in raw_distillations if d.get("issue_id")])
    issue_ids.extend([c.get("issue_id") for c in raw_chunks if c.get("issue_id")])
    metadata = get_article_metadata(supabase, issue_ids)

    # Get full distillation data for those that matched
    distillation_ids = [d.get("distillation_id") for d in raw_distillations if d.get("distillation_id")]
    full_distillations = {}
    if distillation_ids:
        dist_result = supabase.table("stratechery_distillations")\
            .select("*")\
            .in_("distillation_id", distillation_ids)\
            .execute()
        full_distillations = {d["distillation_id"]: d for d in dist_result.data} if dist_result.data else {}

    # Get full chunk data
    chunk_ids = [c.get("chunk_id") for c in raw_chunks if c.get("chunk_id")]
    full_chunks = {}
    if chunk_ids:
        chunk_result = supabase.table("stratechery_chunks")\
            .select("*")\
            .in_("chunk_id", chunk_ids)\
            .execute()
        full_chunks = {c["chunk_id"]: c for c in chunk_result.data} if chunk_result.data else {}

    # Build distillation results
    distillations = []
    for d in raw_distillations:
        dist_id = d.get("distillation_id")
        issue_id = d.get("issue_id")
        full = full_distillations.get(dist_id, {})
        meta = metadata.get(issue_id, {})
        similarity = d.get("similarity", 0)

        # Parse key_claims
        key_claims = []
        for claim in full.get("key_claims", []) or []:
            if isinstance(claim, dict):
                key_claims.append(KeyClaim(
                    claim=claim.get("claim", ""),
                    evidence=claim.get("evidence")
                ))

        # Parse incentive_analysis
        inc = full.get("incentive_analysis", {}) or {}
        incentive_analysis = IncentiveAnalysis(
            who_benefits=inc.get("who_benefits", []) or [],
            who_loses=inc.get("who_loses", []) or [],
            key_incentives=inc.get("key_incentives", []) or []
        ) if inc else None

        distillations.append(DistillationResult(
            distillation_id=dist_id or "",
            issue_id=issue_id or "",
            title=meta.get("title", "Unknown"),
            publication_date=str(meta.get("publication_date", ""))[:10] if meta.get("publication_date") else None,
            similarity=round(similarity, 4),
            above_threshold=similarity >= request.threshold,
            thesis_statement=full.get("thesis_statement", "") or "",
            key_claims=key_claims,
            topics=full.get("topics", []) or [],
            entities=full.get("entities", {}) or {},
            incentive_analysis=incentive_analysis,
            predictions=full.get("predictions", []) or [],
            confidence_score=full.get("confidence_score")
        ))

    # Build chunk results
    chunks = []
    for c in raw_chunks:
        chunk_id = c.get("chunk_id")
        issue_id = c.get("issue_id")
        full = full_chunks.get(chunk_id, {})
        meta = metadata.get(issue_id, {})
        similarity = c.get("similarity", 0)

        chunks.append(ChunkResult(
            chunk_id=chunk_id or "",
            issue_id=issue_id or "",
            title=meta.get("title", "Unknown"),
            publication_date=str(meta.get("publication_date", ""))[:10] if meta.get("publication_date") else None,
            similarity=round(similarity, 4),
            above_threshold=similarity >= request.threshold,
            chunk_index=full.get("chunk_index", 0),
            content=full.get("content", c.get("content", ""))[:2000],  # Limit content size
            token_count=full.get("token_count")
        ))

    # Calculate decision logic
    precision_needed = needs_precision(request.query)
    dist_max = max([d.similarity for d in distillations], default=0)
    chunk_max = max([c.similarity for c in chunks], default=0)
    dist_above = sum(1 for d in distillations if d.above_threshold)
    chunks_above = sum(1 for c in chunks if c.above_threshold)

    # Determine tier used (mimicking query service logic)
    if precision_needed:
        if dist_above > 0 and chunks_above > 0:
            tier_used = "hybrid"
            reasoning = f"Query needs precision (keywords detected). Using hybrid: {dist_above} distillations + {chunks_above} chunks above threshold."
        elif chunks_above > 0:
            tier_used = "chunks"
            reasoning = f"Query needs precision but no distillations above threshold. Using {chunks_above} chunks."
        elif dist_above > 0:
            tier_used = "distillations"
            reasoning = f"Query needs precision but no chunks above threshold. Falling back to {dist_above} distillations."
        else:
            tier_used = "none"
            reasoning = "No results above threshold in either tier."
    else:
        if dist_above >= 2:
            tier_used = "distillations"
            reasoning = f"Standard query with {dist_above} distillations above threshold ({request.threshold}). Using Tier 1 (distillations)."
        elif dist_above > 0:
            tier_used = "hybrid"
            reasoning = f"Only {dist_above} distillation above threshold. Supplementing with chunks (hybrid)."
        elif chunks_above > 0:
            tier_used = "chunks"
            reasoning = f"No distillations above threshold. Falling back to Tier 2 ({chunks_above} chunks)."
        else:
            tier_used = "none"
            reasoning = "No results above threshold in either tier."

    decision = RetrievalDecision(
        query=request.query,
        mode_requested="auto",
        needs_precision=precision_needed,
        distillation_max_score=round(dist_max, 4) if dist_max > 0 else None,
        chunk_max_score=round(chunk_max, 4) if chunk_max > 0 else None,
        tier_used=tier_used,
        reasoning=reasoning
    )

    return InspectResponse(
        query=request.query,
        threshold=request.threshold,
        decision=decision,
        distillations=distillations,
        chunks=chunks,
        distillation_count=len(distillations),
        chunk_count=len(chunks),
        distillations_above_threshold=dist_above,
        chunks_above_threshold=chunks_above
    )


@router.get("/article/{issue_id}/comparison")
async def get_article_comparison(issue_id: str):
    """
    Get side-by-side comparison of raw chunks vs distillation for an article.
    Demonstrates the value of distillation.
    """
    supabase = get_supabase_client()

    # Get article
    article = supabase.table("stratechery_issues")\
        .select("issue_id, title, publication_date, word_count, cleaned_text")\
        .eq("issue_id", issue_id)\
        .single()\
        .execute()

    if not article.data:
        return {"error": "Article not found"}

    # Get chunks
    chunks = supabase.table("stratechery_chunks")\
        .select("chunk_id, chunk_index, content, token_count")\
        .eq("issue_id", issue_id)\
        .order("chunk_index")\
        .execute()

    # Get distillation
    distillation = supabase.table("stratechery_distillations")\
        .select("*")\
        .eq("issue_id", issue_id)\
        .single()\
        .execute()

    return {
        "article": {
            "issue_id": article.data["issue_id"],
            "title": article.data["title"],
            "publication_date": str(article.data.get("publication_date", ""))[:10],
            "word_count": article.data.get("word_count", 0),
            "preview": article.data.get("cleaned_text", "")[:500] + "..."
        },
        "chunks": [
            {
                "chunk_id": c["chunk_id"],
                "chunk_index": c["chunk_index"],
                "content": c["content"],
                "token_count": c.get("token_count", 0)
            }
            for c in (chunks.data or [])
        ],
        "distillation": distillation.data if distillation.data else None,
        "stats": {
            "total_chunks": len(chunks.data or []),
            "total_chunk_tokens": sum(c.get("token_count", 0) for c in (chunks.data or [])),
            "article_word_count": article.data.get("word_count", 0),
            "has_distillation": distillation.data is not None
        }
    }
