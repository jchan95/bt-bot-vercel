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


@dataclass
class JudgeResult:
    relevance: float
    faithfulness: float
    completeness: float
    avg_score: float
    reasoning: str


class EvaluationService:
    """Service for evaluating RAG system quality."""
    
    def __init__(self):
        self.supabase = get_supabase_client()
        self.openai = OpenAI()
        self.query_service = get_query_service()
        self.judge_model = "gpt-4o-mini"
    
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


# Singleton
_eval_service = None

def get_eval_service() -> EvaluationService:
    global _eval_service
    if _eval_service is None:
        _eval_service = EvaluationService()
    return _eval_service