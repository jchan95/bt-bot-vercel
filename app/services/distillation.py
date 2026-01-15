from dotenv import load_dotenv
load_dotenv()

"""
Distillation Service - DEBUGGING VERSION
"""
import json
import re
import logging
import traceback
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


def strip_markdown_fences(text: str) -> str:
    """Robustly strip markdown code fences from GPT response."""
    text = text.strip()
    
    # Handle ```json or ``` at start
    if text.startswith('```'):
        # Find first newline
        first_newline = text.find('\n')
        if first_newline != -1:
            text = text[first_newline + 1:]
    
    # Handle ``` at end
    if text.endswith('```'):
        text = text[:-3]
    
    return text.strip()


class DistillationService:
    """Service for generating distillations from Stratechery articles."""
    
    def __init__(self):
        print("DEBUG: Initializing DistillationService...")
        self.supabase = get_supabase_client()
        self.openai = OpenAI()
        self.model = "gpt-4o"
        print("DEBUG: DistillationService initialized")
    
    def distill_article(self, issue_id: str, force: bool = False) -> DistillationResult:
        """
        Generate a distillation for a single article.
        """
        print(f"\nDEBUG: === Starting distill_article for {issue_id} ===")
        try:
            # Check if distillation already exists
            print("DEBUG: Checking for existing distillation...")
            if not force:
                existing = self.supabase.table("stratechery_distillations")\
                    .select("distillation_id")\
                    .eq("issue_id", issue_id)\
                    .execute()
                
                print(f"DEBUG: Existing check result: {existing.data}")
                
                if existing.data:
                    return DistillationResult(
                        success=False,
                        skipped=True,
                        skip_reason="Distillation already exists",
                        issue_id=issue_id
                    )
            
            # Fetch the article
            print("DEBUG: Fetching article from database...")
            article = self.supabase.table("stratechery_issues")\
                .select("issue_id, title, cleaned_text")\
                .eq("issue_id", issue_id)\
                .single()\
                .execute()
            
            print(f"DEBUG: Article fetch result type: {type(article.data)}")
            print(f"DEBUG: Article data keys: {article.data.keys() if article.data else 'None'}")
            
            if not article.data:
                return DistillationResult(
                    success=False,
                    error=f"Article not found: {issue_id}",
                    issue_id=issue_id
                )
            
            title = article.data.get("title", "Untitled")
            content = article.data.get("cleaned_text", "")
            
            print(f"DEBUG: Title: {title[:50]}...")
            print(f"DEBUG: Content length: {len(content)} chars")
            
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
            print("DEBUG: Calling _generate_distillation...")
            distillation = self._generate_distillation(title, content)
            
            print(f"DEBUG: _generate_distillation returned type: {type(distillation)}")
            
            if distillation is None:
                return DistillationResult(
                    success=False,
                    error="Failed to generate distillation",
                    issue_id=issue_id,
                    title=title
                )
            
            if not isinstance(distillation, dict):
                print(f"DEBUG ERROR: distillation is not a dict! It's {type(distillation)}")
                print(f"DEBUG ERROR: value = {repr(distillation)[:500]}")
                return DistillationResult(
                    success=False,
                    error=f"Distillation is {type(distillation)}, not dict",
                    issue_id=issue_id,
                    title=title
                )
            
            # Save to database
            print("DEBUG: Calling _save_distillation...")
            result = self._save_distillation(issue_id, distillation, force)
            
            print(f"DEBUG: Save result: {result}")
            
            return DistillationResult(
                success=True,
                distillation_id=result.get("distillation_id"),
                issue_id=issue_id,
                title=title
            )
            
        except Exception as e:
            print(f"\nDEBUG EXCEPTION in distill_article:")
            print(f"  Type: {type(e).__name__}")
            print(f"  Message: {e}")
            print("  Full traceback:")
            traceback.print_exc()
            logger.error(f"Error distilling article {issue_id}: {e}")
            return DistillationResult(
                success=False,
                error=str(e),
                issue_id=issue_id
            )
    
    def _generate_distillation(self, title: str, content: str) -> Optional[Dict[str, Any]]:
        """Call GPT-4o to generate the distillation."""
        print("DEBUG: === _generate_distillation starting ===")
        raw_result = None
        try:
            prompt = DISTILLATION_PROMPT.replace("{title}", title).replace("{content}", content)
            print(f"DEBUG: Prompt length: {len(prompt)} chars")
            
            print("DEBUG: Calling OpenAI API...")
            response = self.openai.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert analyst. You must respond with valid JSON only, no other text."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=4000,
            )
            print("DEBUG: OpenAI API returned successfully")
            
            raw_result = response.choices[0].message.content
            print(f"DEBUG: Raw response type: {type(raw_result)}")
            print(f"DEBUG: Raw response (first 500): {repr(raw_result[:500])}")
            
            # Strip markdown fences robustly
            cleaned = strip_markdown_fences(raw_result)
            print(f"DEBUG: Cleaned (first 500): {repr(cleaned[:500])}")
            
            # Parse JSON
            print("DEBUG: Parsing JSON...")
            parsed = json.loads(cleaned)
            
            print(f"DEBUG: Parsed type: {type(parsed)}")
            print(f"DEBUG: Parsed keys: {list(parsed.keys()) if isinstance(parsed, dict) else 'NOT A DICT'}")
            
            return parsed
            
        except json.JSONDecodeError as e:
            print(f"DEBUG JSON ERROR: {e}")
            print(f"DEBUG: Raw was: {repr(raw_result[:1000] if raw_result else 'None')}")
            traceback.print_exc()
            return None
        except Exception as e:
            print(f"DEBUG EXCEPTION in _generate_distillation: {type(e).__name__}: {e}")
            traceback.print_exc()
            return None
    
    def _save_distillation(self, issue_id: str, distillation: Dict[str, Any], force: bool) -> Dict:
        """Save distillation to database."""
        print("DEBUG: === _save_distillation starting ===")
        print(f"DEBUG: distillation type: {type(distillation)}")
        print(f"DEBUG: distillation keys: {list(distillation.keys())}")

        # Handle null thesis_statement - use empty string if null (required by DB constraint)
        thesis = distillation.get("thesis_statement")
        if thesis is None:
            thesis = ""

        try:
            record = {
                "issue_id": issue_id,
                "thesis_statement": thesis,
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
            print(f"DEBUG: Built record with keys: {list(record.keys())}")
            
            if force:
                print("DEBUG: Force mode - deleting existing...")
                self.supabase.table("stratechery_distillations")\
                    .delete()\
                    .eq("issue_id", issue_id)\
                    .execute()
            
            print("DEBUG: Inserting record...")
            response = self.supabase.table("stratechery_distillations")\
                .insert(record)\
                .execute()
            
            print(f"DEBUG: Insert response: {response.data}")
            return response.data[0] if response.data else {}
            
        except Exception as e:
            print(f"DEBUG EXCEPTION in _save_distillation: {type(e).__name__}: {e}")
            traceback.print_exc()
            raise
    
    def distill_batch(
        self,
        limit: int = 10,
        force: bool = False,
        oldest_first: bool = True
    ) -> BatchDistillationResult:
        """Distill a batch of articles."""
        print(f"\nDEBUG: === distill_batch starting (limit={limit}, force={force}) ===")

        try:
            # Get ALL articles using pagination (Supabase has 1000 row default limit)
            print("DEBUG: Fetching all issues (with pagination)...")
            all_issues = []
            page_size = 1000
            offset = 0

            while True:
                batch = self.supabase.table("stratechery_issues")\
                    .select("issue_id, title, publication_date")\
                    .range(offset, offset + page_size - 1)\
                    .execute()

                if not batch.data:
                    break

                all_issues.extend(batch.data)

                if len(batch.data) < page_size:
                    break

                offset += page_size

            print(f"DEBUG: Found {len(all_issues)} total issues")

            # Get ALL existing distillations using pagination
            print("DEBUG: Fetching existing distillations (with pagination)...")
            existing_ids = set()
            offset = 0

            while True:
                batch = self.supabase.table("stratechery_distillations")\
                    .select("issue_id")\
                    .range(offset, offset + page_size - 1)\
                    .execute()

                if not batch.data:
                    break

                existing_ids.update(d["issue_id"] for d in batch.data)

                if len(batch.data) < page_size:
                    break

                offset += page_size

            print(f"DEBUG: Found {len(existing_ids)} existing distillations")
            
            pending = [
                issue for issue in all_issues
                if issue["issue_id"] not in existing_ids
            ]
            print(f"DEBUG: {len(pending)} pending articles")
            
            # Sort by publication date
            pending.sort(
                key=lambda x: x.get("publication_date") or "", 
                reverse=not oldest_first
            )
            
            pending = pending[:limit]
            print(f"DEBUG: Processing {len(pending)} articles")
            
            results = []
            for i, issue in enumerate(pending):
                issue_id = issue["issue_id"]
                title = issue.get("title", "Untitled")
                
                print(f"\n{'='*60}")
                print(f"DEBUG: [{i+1}/{len(pending)}] Processing: {title[:50]}...")
                print(f"DEBUG: issue_id = {issue_id}")
                
                result = self.distill_article(issue_id, force=force)
                results.append(result)
                
                if result.success:
                    print(f"DEBUG: ✓ Success")
                elif result.skipped:
                    print(f"DEBUG: - Skipped: {result.skip_reason}")
                else:
                    print(f"DEBUG: ✗ Failed: {result.error}")
            
            successful = sum(1 for r in results if r.success)
            skipped = sum(1 for r in results if r.skipped)
            failed = sum(1 for r in results if not r.success and not r.skipped)
            
            print(f"\nDEBUG: Batch complete: {successful} successful, {skipped} skipped, {failed} failed")
            
            return BatchDistillationResult(
                total=len(results),
                successful=successful,
                skipped=skipped,
                failed=failed,
                results=results
            )
            
        except Exception as e:
            print(f"\nDEBUG EXCEPTION in distill_batch:")
            print(f"  Type: {type(e).__name__}")
            print(f"  Message: {e}")
            traceback.print_exc()
            raise
    
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
