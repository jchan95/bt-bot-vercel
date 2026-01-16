"""
Evaluation Service - LLM-as-judge evaluation for BT Bot
"""
import logging
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from datetime import datetime
import json

from openai import OpenAI

from app.database import get_supabase_client
from app.services.query import get_query_service

logger = logging.getLogger(__name__)


JUDGE_PROMPT = """You are evaluating a RAG system that answers questions about Stratechery content.

Score the answer on three dimensions (0-5 scale):

1. **Relevance** (0-5): Does the answer address the question asked?
   - 5: Directly and completely addresses the question
   - 3: Partially addresses but misses key aspects
   - 0: Completely off-topic

2. **Faithfulness** (0-5): Is the answer grounded in the provided sources?
   - 5: All claims supported by sources, no hallucinations
   - 3: Mostly grounded but some unsupported claims
   - 0: Makes up information not in sources

3. **Completeness** (0-5): Does the answer provide sufficient depth?
   - 5: Comprehensive analysis with multiple perspectives
   - 3: Adequate but could go deeper
   - 0: Superficial or missing major points

Respond in this exact JSON format:
{
  "relevance": <score>,
  "faithfulness": <score>,
  "completeness": <score>,
  "reasoning": "<brief explanation of scores>"
}"""


CITATION_ACCURACY_PROMPT = """You are verifying citation accuracy in an AI-generated answer.

For each citation in the answer (formatted like [Article Title, Date]), determine:
1. Does the citation match a real article from the provided sources list?
2. Does the claim being cited actually align with what that article says?

SOURCES PROVIDED TO THE SYSTEM:
{sources}

ANSWER WITH CITATIONS:
{answer}

For each citation found, evaluate it:
- "valid": The article exists AND the claim aligns with the article's thesis/content
- "exists_but_misused": The article exists but the claim doesn't match
- "hallucinated": The article title doesn't match any provided source

Respond in this exact JSON format:
{{
  "citations_found": [
    {{"citation": "[Article Title, Date]", "claim": "the claim being made", "status": "valid|exists_but_misused|hallucinated", "reason": "brief explanation"}}
  ],
  "total_citations": <number>,
  "valid_count": <number>,
  "accuracy_score": <0.0-1.0>
}}"""


@dataclass
class JudgeResult:
    relevance: float
    faithfulness: float
    completeness: float
    avg_score: float
    reasoning: str


@dataclass
class CitationAccuracyResult:
    total_citations: int
    valid_count: int
    exists_but_misused: int
    hallucinated: int
    accuracy_score: float
    citations: List[Dict]


class EvaluationService:
    """Service for evaluating RAG system quality."""
    
    def __init__(self):
        self.supabase = get_supabase_client()
        self.openai = OpenAI()
        self.query_service = get_query_service()
        self.judge_model = "gpt-4o-mini"
    
    def _evaluate_citation_accuracy(self, answer: str, sources: List[Dict]) -> CitationAccuracyResult:
        """Evaluate accuracy of citations in reasoning-first answers."""

        # Build detailed source info for verification
        source_text = "\n".join([
            f"- Title: \"{s.get('title', 'Unknown')}\"\n  Date: {s.get('date', 'Unknown')}\n  Thesis: {s.get('thesis_statement', 'N/A')[:200] if s.get('thesis_statement') else 'N/A'}"
            for s in sources
        ])

        prompt = CITATION_ACCURACY_PROMPT.format(
            sources=source_text,
            answer=answer
        )

        response = self.openai.chat.completions.create(
            model=self.judge_model,
            messages=[
                {"role": "system", "content": "You verify citation accuracy. Respond only with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=1500
        )

        try:
            result_text = response.choices[0].message.content.strip()
            if result_text.startswith("```"):
                result_text = result_text.split("```")[1]
                if result_text.startswith("json"):
                    result_text = result_text[4:]

            result = json.loads(result_text)

            citations = result.get("citations_found", [])
            valid = sum(1 for c in citations if c.get("status") == "valid")
            misused = sum(1 for c in citations if c.get("status") == "exists_but_misused")
            hallucinated = sum(1 for c in citations if c.get("status") == "hallucinated")
            total = len(citations)

            return CitationAccuracyResult(
                total_citations=total,
                valid_count=valid,
                exists_but_misused=misused,
                hallucinated=hallucinated,
                accuracy_score=valid / total if total > 0 else 1.0,
                citations=citations
            )
        except Exception as e:
            logger.error(f"Error parsing citation accuracy response: {e}")
            return CitationAccuracyResult(
                total_citations=0, valid_count=0, exists_but_misused=0,
                hallucinated=0, accuracy_score=0, citations=[]
            )

    def _judge_answer(self, question: str, answer: str, sources: List[Dict]) -> JudgeResult:
        """Use LLM to judge answer quality."""

        source_text = "\n".join([
            f"- {s.get('title', 'Unknown')} ({s.get('date', 'Unknown')})"
            for s in sources
        ])
        
        user_prompt = f"""Question: {question}

Answer: {answer}

Sources cited:
{source_text}

Evaluate this answer."""

        response = self.openai.chat.completions.create(
            model=self.judge_model,
            messages=[
                {"role": "system", "content": JUDGE_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0,
            max_tokens=500
        )
        
        try:
            result_text = response.choices[0].message.content
            # Extract JSON from response
            result_text = result_text.strip()
            if result_text.startswith("```"):
                result_text = result_text.split("```")[1]
                if result_text.startswith("json"):
                    result_text = result_text[4:]
            
            result = json.loads(result_text)
            
            relevance = float(result.get("relevance", 0))
            faithfulness = float(result.get("faithfulness", 0))
            completeness = float(result.get("completeness", 0))
            avg_score = (relevance + faithfulness + completeness) / 3
            
            return JudgeResult(
                relevance=relevance,
                faithfulness=faithfulness,
                completeness=completeness,
                avg_score=round(avg_score, 2),
                reasoning=result.get("reasoning", "")
            )
        except Exception as e:
            logger.error(f"Error parsing judge response: {e}")
            return JudgeResult(
                relevance=0, faithfulness=0, completeness=0,
                avg_score=0, reasoning=f"Parse error: {e}"
            )
    
    def run_eval(self, config: Dict = None) -> Dict[str, Any]:
        """Run evaluation on all examples."""
        
        config = config or {"mode": "auto", "limit": 5, "threshold": 0.3}
        
        # Create eval run
        run_result = self.supabase.table("eval_runs").insert({
            "config": config,
            "total_examples": 0,
            "avg_score": 0
        }).execute()
        
        run_id = run_result.data[0]["run_id"]
        
        # Get all examples
        examples = self.supabase.table("eval_examples")\
            .select("*")\
            .execute()
        
        if not examples.data:
            return {"error": "No eval examples found", "run_id": run_id}
        
        results = []
        total_score = 0
        
        for ex in examples.data:
            # Get answer from system
            query_result = self.query_service.query(
                question=ex["question"],
                limit=config.get("limit", 5),
                threshold=config.get("threshold", 0.3),
                mode=config.get("mode", "auto")
            )
            
            # Judge the answer
            judge_result = self._judge_answer(
                ex["question"],
                query_result.answer,
                query_result.sources
            )
            
            # Save result
            self.supabase.table("eval_results").insert({
                "run_id": run_id,
                "example_id": ex["example_id"],
                "question": ex["question"],
                "answer": query_result.answer,
                "retrieval_tier": query_result.retrieval_tier,
                "sources": query_result.sources,
                "relevance_score": judge_result.relevance,
                "faithfulness_score": judge_result.faithfulness,
                "completeness_score": judge_result.completeness,
                "avg_score": judge_result.avg_score,
                "judge_reasoning": judge_result.reasoning
            }).execute()
            
            total_score += judge_result.avg_score
            results.append({
                "question": ex["question"][:50] + "...",
                "score": judge_result.avg_score,
                "tier": query_result.retrieval_tier
            })
        
        # Update run with final scores
        avg_score = total_score / len(examples.data) if examples.data else 0
        self.supabase.table("eval_runs").update({
            "completed_at": datetime.utcnow().isoformat(),
            "total_examples": len(examples.data),
            "avg_score": round(avg_score, 2)
        }).eq("run_id", run_id).execute()
        
        return {
            "run_id": run_id,
            "total_examples": len(examples.data),
            "avg_score": round(avg_score, 2),
            "results": results
        }
    
    def get_runs(self, limit: int = 10) -> List[Dict]:
        """Get recent eval runs."""
        result = self.supabase.table("eval_runs")\
            .select("*")\
            .order("started_at", desc=True)\
            .limit(limit)\
            .execute()
        return result.data or []
    
    def get_run_details(self, run_id: str) -> Dict[str, Any]:
        """Get detailed results for a run."""
        run = self.supabase.table("eval_runs")\
            .select("*")\
            .eq("run_id", run_id)\
            .single()\
            .execute()
        
        results = self.supabase.table("eval_results")\
            .select("*")\
            .eq("run_id", run_id)\
            .execute()
        
        return {
            "run": run.data,
            "results": results.data or []
        }
    
    def get_examples(self) -> List[Dict]:
        """Get all eval examples."""
        result = self.supabase.table("eval_examples")\
            .select("*")\
            .execute()
        return result.data or []
    
    def add_example(self, question: str, expected_answer: str = None, 
                    category: str = None, difficulty: str = None) -> Dict:
        """Add a new eval example."""
        result = self.supabase.table("eval_examples").insert({
            "question": question,
            "expected_answer": expected_answer,
            "category": category,
            "difficulty": difficulty
        }).execute()
        return result.data[0] if result.data else {}


    def eval_citation_accuracy(self, question: str) -> Dict[str, Any]:
        """
        Run a single question through reasoning-first mode and evaluate citation accuracy.
        """
        logger.info(f"Running citation accuracy eval for: {question[:50]}...")

        # Run reasoning-first query
        query_result = self.query_service.query_reasoning(
            question=question,
            limit=5,
            threshold=0.3
        )

        # Evaluate citations
        citation_result = self._evaluate_citation_accuracy(
            query_result.answer,
            query_result.sources
        )

        return {
            "question": question,
            "answer": query_result.answer,
            "sources": query_result.sources,
            "citation_eval": {
                "total_citations": citation_result.total_citations,
                "valid": citation_result.valid_count,
                "exists_but_misused": citation_result.exists_but_misused,
                "hallucinated": citation_result.hallucinated,
                "accuracy_score": citation_result.accuracy_score,
                "details": citation_result.citations
            }
        }

    def eval_citation_accuracy_batch(self) -> Dict[str, Any]:
        """
        Run citation accuracy eval on all examples in reasoning-first mode.
        Persists results to database.
        """
        examples = self.get_examples()

        if not examples:
            return {"error": "No eval examples found"}

        results = []
        total_citations = 0
        total_valid = 0
        total_misused = 0
        total_hallucinated = 0

        for ex in examples:
            logger.info(f"Evaluating: {ex['question'][:40]}...")
            result = self.eval_citation_accuracy(ex["question"])
            results.append({
                "question": ex["question"],
                "accuracy_score": result["citation_eval"]["accuracy_score"],
                "total_citations": result["citation_eval"]["total_citations"],
                "valid": result["citation_eval"]["valid"],
                "misused": result["citation_eval"]["exists_but_misused"],
                "hallucinated": result["citation_eval"]["hallucinated"],
                "details": result["citation_eval"]["details"]
            })

            total_citations += result["citation_eval"]["total_citations"]
            total_valid += result["citation_eval"]["valid"]
            total_misused += result["citation_eval"]["exists_but_misused"]
            total_hallucinated += result["citation_eval"]["hallucinated"]

        overall_accuracy = total_valid / total_citations if total_citations > 0 else 0

        # Persist to database
        run_data = {
            "completed_at": datetime.utcnow().isoformat(),
            "total_examples": len(examples),
            "total_citations": total_citations,
            "valid_citations": total_valid,
            "misused_citations": total_misused,
            "hallucinated_citations": total_hallucinated,
            "overall_accuracy": round(overall_accuracy, 4),
            "results": results
        }

        try:
            db_result = self.supabase.table("citation_accuracy_runs").insert(run_data).execute()
            run_id = db_result.data[0]["run_id"] if db_result.data else None
            logger.info(f"Saved citation accuracy run: {run_id}")
        except Exception as e:
            logger.error(f"Failed to save citation accuracy run: {e}")
            run_id = None

        return {
            "run_id": run_id,
            "total_examples": len(examples),
            "aggregate": {
                "total_citations": total_citations,
                "valid": total_valid,
                "exists_but_misused": total_misused,
                "hallucinated": total_hallucinated,
                "overall_accuracy": overall_accuracy
            },
            "results": [
                {
                    "question": r["question"][:50] + "..." if len(r["question"]) > 50 else r["question"],
                    "accuracy_score": r["accuracy_score"],
                    "total_citations": r["total_citations"],
                    "valid": r["valid"],
                    "hallucinated": r["hallucinated"]
                }
                for r in results
            ]
        }

    def get_citation_accuracy_runs(self, limit: int = 10) -> List[Dict]:
        """Get recent citation accuracy runs."""
        result = self.supabase.table("citation_accuracy_runs")\
            .select("run_id, started_at, completed_at, total_examples, total_citations, valid_citations, misused_citations, hallucinated_citations, overall_accuracy")\
            .order("started_at", desc=True)\
            .limit(limit)\
            .execute()
        return result.data or []

    def get_citation_accuracy_run_details(self, run_id: str) -> Dict[str, Any]:
        """Get detailed results for a citation accuracy run."""
        result = self.supabase.table("citation_accuracy_runs")\
            .select("*")\
            .eq("run_id", run_id)\
            .single()\
            .execute()
        return result.data if result.data else {}


# Singleton
_eval_service = None

def get_eval_service() -> EvaluationService:
    global _eval_service
    if _eval_service is None:
        _eval_service = EvaluationService()
    return _eval_service