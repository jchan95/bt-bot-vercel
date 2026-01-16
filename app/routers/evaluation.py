"""
Evaluation Router - API endpoints for eval system
"""
from typing import List, Optional
from fastapi import APIRouter
from pydantic import BaseModel

from app.services.evaluation import get_eval_service

router = APIRouter(prefix="/eval", tags=["evaluation"])


class ExampleCreate(BaseModel):
    question: str
    expected_answer: Optional[str] = None
    category: Optional[str] = None
    difficulty: Optional[str] = None


class RunConfig(BaseModel):
    mode: str = "auto"
    limit: int = 5
    threshold: float = 0.3


@router.get("/examples")
async def get_examples():
    """Get all eval examples."""
    service = get_eval_service()
    return {"examples": service.get_examples()}


@router.post("/examples")
async def add_example(example: ExampleCreate):
    """Add a new eval example."""
    service = get_eval_service()
    result = service.add_example(
        question=example.question,
        expected_answer=example.expected_answer,
        category=example.category,
        difficulty=example.difficulty
    )
    return {"example": result}


@router.post("/run")
async def run_eval(config: RunConfig = None):
    """Run evaluation on all examples."""
    service = get_eval_service()
    config_dict = config.dict() if config else {}
    return service.run_eval(config_dict)


@router.get("/runs")
async def get_runs(limit: int = 10):
    """Get recent eval runs."""
    service = get_eval_service()
    return {"runs": service.get_runs(limit)}


@router.get("/runs/{run_id}")
async def get_run_details(run_id: str):
    """Get detailed results for a run."""
    service = get_eval_service()
    return service.get_run_details(run_id)


class CitationEvalRequest(BaseModel):
    question: str


@router.post("/citation-accuracy")
async def eval_citation_accuracy(request: CitationEvalRequest):
    """
    Run citation accuracy eval on a single question in reasoning-first mode.
    Tests if the LLM's citations match real articles and claims.
    """
    service = get_eval_service()
    return service.eval_citation_accuracy(request.question)


@router.post("/citation-accuracy/batch")
async def eval_citation_accuracy_batch():
    """
    Run citation accuracy eval on all examples in reasoning-first mode.
    Returns aggregate stats on citation quality. Persists to database.
    """
    service = get_eval_service()
    return service.eval_citation_accuracy_batch()


@router.get("/citation-accuracy/runs")
async def get_citation_accuracy_runs(limit: int = 10):
    """Get recent citation accuracy runs."""
    service = get_eval_service()
    return {"runs": service.get_citation_accuracy_runs(limit)}


@router.get("/citation-accuracy/runs/{run_id}")
async def get_citation_accuracy_run_details(run_id: str):
    """Get detailed results for a citation accuracy run."""
    service = get_eval_service()
    return service.get_citation_accuracy_run_details(run_id)