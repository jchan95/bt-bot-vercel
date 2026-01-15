from dotenv import load_dotenv
load_dotenv()

"""
Distillation Service - Extract structured insights from Stratechery articles using GPT-4o.
"""
import json
import re
import logging
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from datetime import datetime

from openai import OpenAI

from app.database import get_supabase_client

logger = logging.getLogger(__name__)


@dataclass
class DistillationResult:
    """Result of distilling a single article."""
    success: bool
    distillation_id: Optional[str] = None
    issue_id: Optional[str] = None
    title: Optional[str] = None
    error: Optional[str] = None
    skipped: bool = False
    skip_reason: Optional[str] = None


@dataclass
class BatchDistillationResult:
    """Result of batch distillation."""
    total: int
    successful: int
    skipped: int
    failed: int
    results: List[DistillationResult]


# The prompt for GPT-4o to extract structured insights
DISTILLATION_PROMPT = """You are an expert analyst of Ben Thompson's Stratechery newsletter. Your task is to extract structured insights from the following article.

Analyze the article and provide a JSON response with the following structure:

{
  "thesis_statement": "The main argument or thesis of the article in 1-2 sentences",
  
  "key_claims": [
    {
      "claim": "A specific claim made in the article",
      "evidence": "The evidence or reasoning provided for this claim"
    }
  ],
  
  "incentive_analysis": {
    "who_benefits": ["List of entities that benefit from the situation described"],
    "who_loses": ["List of entities that are disadvantaged"],
    "key_incentives": ["The underlying incentives driving behavior"]
  },
  
  "second_order_effects": [
    {
      "effect": "A downstream consequence or implication",
      "likelihood": "high/medium/low"
    }
  ],
  
  "counterarguments": [
    {
      "argument": "A potential counterargument or alternative view",
      "response": "How Thompson addresses or might address this (if applicable)"
    }
  ],
  
  "predictions": [
    {
      "prediction": "A prediction made in the article",
      "timeframe": "When this might occur (if specified)",
      "confidence": "high/medium/low based on how strongly stated"
    }
  ],
  
  "entities": {
    "companies": ["List of companies mentioned"],
    "people": ["List of people mentioned"],
    "products": ["List of products or services mentioned"]
  },
  
  "topics": ["List of 3-5 topic tags for this article, e.g., 'Apple', 'Aggregation Theory', 'Regulation'"],
  
  "meta_references": {
    "frameworks": ["Stratechery frameworks referenced, e.g., 'Aggregation Theory', 'Disruption Theory', 'Value Chain'"],
    "past_articles": ["References to previous Stratechery articles if any"]
  },
  
  "confidence_score": 0.85
}

Guidelines:
- Be concise but comprehensive
- If a section doesn't apply to this article, use empty arrays [] or null
- For confidence_score, rate 0.0-1.0 based on how clearly the article states its thesis
- Focus on Thompson's unique analytical insights, not just news summary
- Identify the "so what" - why this matters for tech/business strategy

ARTICLE TITLE: {title}

ARTICLE CONTENT:
{content}

Respond ONLY with valid JSON, no other text."""


class DistillationService:
    """Service for generating distillations from Stratechery articles."""
    
    def __init__(self):
        self.supabase = get_supabase_client()
        self.openai = OpenAI()
        self.model = "gpt-4o"
    
    def distill_article(self, issue_id: str, force: bool = False) -> DistillationResult:
        """
        Generate a distillation for a single article.
        
        Args:
            issue_id: UUID of the article in stratechery_issues
            force: If True, regenerate even if distillation exists
        
        Returns:
            DistillationResult
        """
        try:
            # Check if distillation already exists
            if not force:
                existing = self.supabase.table("stratechery_distillations")\
                    .select("distillation_id")\
                    .eq("issue_id", issue_id)\
                    .execute()
                
                if existing.data:
                    return DistillationResult(
                        success=False,
                        skipped=True,
                        skip_reason="Distillation already exists",
                        issue_id=issue_id
                    )
            
            # Fetch the article
            article = self.supabase.table("stratechery_issues")\
                .select("issue_id, title, cleaned_text")\
                .eq("issue_id", issue_id)\
                .single()\
                .execute()
            
            if not article.data:
                return DistillationResult(
                    success=False,
                    error=f"Article not found: {issue_id}",
                    issue_id=issue_id
                )
            
            title = article.data.get("title", "Untitled")
            content = article.data.get("cleaned_text", "")
            
            if not content or len(content) < 100:
                return DistillationResult(
                    success=False,
                    error="Article content too short",
                    issue_id=issue_id,
                    title=title
                )
            
            # Truncate content if too long (GPT-4o context limit)
            max_chars = 100000  # ~25k tokens
            if len(content) > max_chars:
                content = content[:max_chars] + "\n\n[Content truncated...]"
            
            # Generate distillation with GPT-4o
            distillation = self._generate_distillation(title, content)
            
            if not distillation:
                return DistillationResult(
                    success=False,
                    error="Failed to generate distillation",
                    issue_id=issue_id,
                    title=title
                )
            
            # Save to database
            result = self._save_distillation(issue_id, distillation, force)
            
            return DistillationResult(
                success=True,
                distillation_id=result.get("distillation_id"),
                issue_id=issue_id,
                title=title
            )
            
        except Exception as e:
            logger.error(f"Error distilling article {issue_id}: {e}")
            return DistillationResult(
                success=False,
                error=str(e),
                issue_id=issue_id
            )
    
    def _generate_distillation(self, title: str, content: str) -> Optional[Dict[str, Any]]:
        """Call GPT-4o to generate the distillation."""
        result = None
        try:
            prompt = DISTILLATION_PROMPT.format(title=title, content=content)
            
            response = self.openai.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert analyst. You must respond with valid JSON only, no other text."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=4000,
            )
            
            result = response.choices[0].message.content
            print(f"DEBUG RAW (first 300): {repr(result[:300])}")

            # Strip markdown code fences if present
            result = result.strip()
            if result.startswith('```'):
                lines = result.split('\n')
                lines = [l for l in lines if not l.startswith('```')]
                result = '\n'.join(lines)
            
            print(f"DEBUG CLEANED (first 300): {repr(result[:300])}")
            
            parsed = json.loads(result)
            print(f"DEBUG PARSED OK: {list(parsed.keys())}")
            return parsed
            
        except json.JSONDecodeError as e:
            print(f"DEBUG JSON ERROR: {e}")
            print(f"DEBUG RESULT WAS: {repr(result[:500] if result else 'None')}")
            logger.error(f"Failed to parse GPT response as JSON: {e}")
            return None
        except Exception as e:
            print(f"DEBUG EXCEPTION: {type(e).__name__}: {e}")
            logger.error(f"GPT API error: {e}")
            return None
    
    def _save_distillation(self, issue_id: str, distillation: Dict[str, Any], force: bool) -> Dict:
        """Save distillation to database."""
        print(f"DEBUG DISTILLATION KEYS: {list(distillation.keys())}")
        print(f"DEBUG THESIS: {distillation.get('thesis_statement', 'NOT FOUND')}")
        record = {
            "issue_id": issue_id,
            "thesis_statement": distillation.get("thesis_statement", ""),
            "key_claims": distillation.get("key_claims", []),
            "incentive_analysis": distillation.get("incentive_analysis", {}),
            "second_order_effects": distillation.get("second_order_effects", []),
            "counterarguments": distillation.get("counterarguments", []),
            "predictions": distillation.get("predictions", []),
            "entities": distillation.get("entities", {}),
            "topics": distillation.get("topics", []),
            "meta_references": distillation.get("meta_references", {}),
            "confidence_score": distillation.get("confidence_score", 0.5),
            "created_by": self.model,
            "reviewed": False,
        }
        
        if force:
            # Delete existing and insert new
            self.supabase.table("stratechery_distillations")\
                .delete()\
                .eq("issue_id", issue_id)\
                .execute()
        
        response = self.supabase.table("stratechery_distillations")\
            .insert(record)\
            .execute()
        
        return response.data[0] if response.data else {}
    
    def distill_batch(
        self, 
        limit: int = 10, 
        force: bool = False,
        oldest_first: bool = True
    ) -> BatchDistillationResult:
        """
        Distill a batch of articles that don't have distillations yet.
        
        Args:
            limit: Maximum number of articles to process
            force: If True, regenerate all distillations
            oldest_first: If True, process oldest articles first
        
        Returns:
            BatchDistillationResult
        """
        # Get articles without distillations
        if force:
            query = self.supabase.table("stratechery_issues")\
                .select("issue_id, title, publication_date")
        else:
            # Get issues that don't have distillations yet
            # Using a left join approach via RPC or just fetching all and filtering
            all_issues = self.supabase.table("stratechery_issues")\
                .select("issue_id, title, publication_date")\
                .execute()
            
            existing = self.supabase.table("stratechery_distillations")\
                .select("issue_id")\
                .execute()
            
            existing_ids = {d["issue_id"] for d in existing.data} if existing.data else set()
            
            pending = [
                issue for issue in all_issues.data 
                if issue["issue_id"] not in existing_ids
            ]
            
            # Sort by publication date
            pending.sort(
                key=lambda x: x.get("publication_date") or "", 
                reverse=not oldest_first
            )
            
            pending = pending[:limit]
        
        logger.info(f"Found {len(pending)} articles to distill")
        
        results = []
        for i, issue in enumerate(pending):
            issue_id = issue["issue_id"]
            title = issue.get("title", "Untitled")
            
            logger.info(f"[{i+1}/{len(pending)}] Distilling: {title[:50]}...")
            
            result = self.distill_article(issue_id, force=force)
            results.append(result)
            
            if result.success:
                logger.info(f"  ✓ Success")
            elif result.skipped:
                logger.info(f"  - Skipped: {result.skip_reason}")
            else:
                logger.warning(f"  ✗ Failed: {result.error}")
        
        successful = sum(1 for r in results if r.success)
        skipped = sum(1 for r in results if r.skipped)
        failed = sum(1 for r in results if not r.success and not r.skipped)
        
        logger.info(f"Batch complete: {successful} successful, {skipped} skipped, {failed} failed")
        
        return BatchDistillationResult(
            total=len(results),
            successful=successful,
            skipped=skipped,
            failed=failed,
            results=results
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get distillation statistics."""
        total_issues = self.supabase.table("stratechery_issues")\
            .select("issue_id", count="exact")\
            .execute()
        
        total_distillations = self.supabase.table("stratechery_distillations")\
            .select("distillation_id", count="exact")\
            .execute()
        
        reviewed = self.supabase.table("stratechery_distillations")\
            .select("distillation_id", count="exact")\
            .eq("reviewed", True)\
            .execute()
        
        return {
            "total_issues": total_issues.count or 0,
            "total_distillations": total_distillations.count or 0,
            "pending": (total_issues.count or 0) - (total_distillations.count or 0),
            "reviewed": reviewed.count or 0,
            "completion_rate": round(
                (total_distillations.count or 0) / max(total_issues.count or 1, 1) * 100, 
                1
            )
        }


# Singleton instance
_distillation_service = None

def get_distillation_service() -> DistillationService:
    """Get or create the distillation service singleton."""
    global _distillation_service
    if _distillation_service is None:
        _distillation_service = DistillationService()
    return _distillation_service
