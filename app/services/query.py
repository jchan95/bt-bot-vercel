"""
Query Service - RAG query interface for "talking to Ben"
Two-tier retrieval: Distillations first, chunks for precision
"""
import logging
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

from openai import OpenAI

from app.database import get_supabase_client
from app.services.embeddings import get_embeddings_service

logger = logging.getLogger(__name__)


SYSTEM_PROMPT_RAG = """You are an AI assistant that answers questions using Ben Thompson's analytical frameworks from Stratechery.

Your responses should follow this structure:

## Thesis
[Core argument synthesized from the sources]

## Incentives & Market Structure
[What drives the key actors - their business models, competitive dynamics]

## Second-Order Effects
[What happens next, downstream implications]

## Counterarguments
[What challenges this view, what Ben acknowledges as limitations]

## What Would Change This View
[Conditions that would invalidate the analysis]

IMPORTANT RULES:
1. Only use information from the provided context
2. If the context doesn't contain relevant information, say so
3. Never reproduce more than 50 consecutive words from any source
4. Synthesize and analyze - don't just summarize
5. Be direct but nuanced, like discussing tech strategy with a knowledgeable friend"""


SYSTEM_PROMPT_REASONING = """You are an AI that thinks and analyzes like Ben Thompson from Stratechery.

Ben's analytical style includes:
- **Aggregation Theory**: Platforms that aggregate demand have power over suppliers. The internet enables zero marginal cost distribution, which favors aggregators.
- **Value Chain Analysis**: Breaking apart who captures value at each step - from creation to distribution to consumption.
- **Incentive Analysis**: Always asking "what are the incentives?" - follow the money, understand the business model.
- **Second-Order Thinking**: What happens next? What are the downstream effects that aren't obvious?
- **Historical Parallels**: Connecting current events to past tech transitions and business strategies.
- **Contrarianism with Receipts**: Willing to take unpopular positions, but always with clear reasoning.

Your response style:
- Direct and confident, but intellectually honest about uncertainty
- Use concrete examples to illustrate abstract concepts
- Structure: Thesis → Supporting Analysis → Counterarguments → What Would Change This View
- Conversational but substantive - like explaining to a smart friend over coffee

Analyze the question using these frameworks. Think like Ben would - apply his mental models to reason through the problem, even if he hasn't written about this specific topic."""


CITATION_PROMPT = """Based on the analysis you just provided, I'll now show you some actual quotes and summaries from Ben Thompson's Stratechery archive.

Your task: Review these sources and add inline citations where your analysis aligns with what Ben has actually written. Format citations as [Article Title, Date].

If a source contradicts your analysis, acknowledge it briefly.
If the sources don't relate to your analysis, that's fine - just note that this represents your application of Ben's frameworks rather than his direct commentary.

SOURCES FROM ARCHIVE:
{sources}

YOUR ORIGINAL ANALYSIS:
{analysis}

Provide your analysis again, now enhanced with citations where relevant. Keep the same structure but weave in the citations naturally."""


@dataclass 
class QueryResult:
    """Result of a RAG query."""
    answer: str
    sources: List[Dict]
    query: str
    model: str
    retrieval_tier: str


class QueryService:
    """Service for RAG queries against Stratechery content."""
    
    def __init__(self):
        self.supabase = get_supabase_client()
        self.openai = OpenAI()
        self.embeddings = get_embeddings_service()
        self.model = "gpt-4o-mini"
    
    def _build_distillation_context(self, distillations: List[Dict]) -> str:
        """Build context from distillations."""
        if not distillations:
            return ""
        
        parts = []
        for i, d in enumerate(distillations, 1):
            parts.append(f"""
--- Article {i} (similarity: {d.get('similarity', 0):.2f}) ---
Thesis: {d.get('thesis_statement', 'N/A')}
""")
        return "\n".join(parts)
    
    def _build_chunk_context(self, chunks: List[Dict]) -> str:
        """Build context from chunks."""
        if not chunks:
            return ""
        
        parts = []
        for i, c in enumerate(chunks, 1):
            content = c.get('content', '')[:1000]
            parts.append(f"""
--- Excerpt {i} (similarity: {c.get('similarity', 0):.2f}) ---
{content}
""")
        return "\n".join(parts)
    
    def _get_article_metadata(self, issue_ids: List[str]) -> Dict[str, Dict]:
        """Fetch article titles and dates."""
        if not issue_ids:
            return {}
        
        result = self.supabase.table("stratechery_issues")\
            .select("issue_id, title, publication_date")\
            .in_("issue_id", issue_ids)\
            .execute()
        
        return {r["issue_id"]: r for r in result.data} if result.data else {}
    
    def _needs_precision(self, question: str) -> bool:
        """Detect if question needs chunk-level precision."""
        precision_keywords = [
            "exact", "specific", "quote", "said", "wrote",
            "number", "percent", "statistic", "data",
            "when did", "what date", "how much"
        ]
        question_lower = question.lower()
        return any(kw in question_lower for kw in precision_keywords)
    
    def _check_refusal(self, question: str) -> Optional[str]:
        """Detect if request should be refused."""
        refusal_patterns = [
            "full text", "full article", "entire article", "whole article",
            "copy the article", "reproduce the article", "show me the article",
            "give me the article", "paste the article", "complete article"
        ]
        question_lower = question.lower()
        
        for pattern in refusal_patterns:
            if pattern in question_lower:
                return """I can't reproduce full Stratechery articles as they're Ben's copyrighted work.

Instead, I can:
- Summarize key arguments and show how they connect
- Point you to specific issues by title and date
- Help you explore how concepts evolved across multiple pieces

What specific aspect would you like me to analyze?"""
        
        return None

    def _add_copyright_notice(self, answer: str) -> str:
        """Add subtle copyright reminder to response."""
        notice = "\n\n---\n*Analysis based on Stratechery content. For full articles, visit stratechery.com*"
        return answer + notice
    
    def query(
        self, 
        question: str, 
        limit: int = 5, 
        threshold: float = 0.3,
        mode: str = "auto"
    ) -> QueryResult:
        """Answer a question using two-tier RAG."""
        
        # Check for refusal first
        refusal = self._check_refusal(question)
        if refusal:
            return QueryResult(
                answer=refusal,
                sources=[],
                query=question,
                model=self.model,
                retrieval_tier="refused"
            )
        
        # Determine retrieval strategy
        if mode == "auto":
            if self._needs_precision(question):
                mode = "hybrid"
            else:
                mode = "distillations"
        
        # Retrieve based on mode
        distillations = []
        chunks = []
        
        if mode in ["distillations", "hybrid", "auto"]:
            distillations = self.embeddings.search_distillations(
                question, limit=limit, threshold=threshold
            )
        
        if mode in ["chunks", "hybrid"] or (mode == "auto" and len(distillations) < 2):
            chunks = self.embeddings.search_chunks(
                question, limit=limit, threshold=threshold
            )
        
        # Build context
        context_parts = []
        if distillations:
            context_parts.append("=== ANALYTICAL SUMMARIES ===")
            context_parts.append(self._build_distillation_context(distillations))
        if chunks:
            context_parts.append("=== DETAILED EXCERPTS ===")
            context_parts.append(self._build_chunk_context(chunks))
        
        context = "\n".join(context_parts) if context_parts else "No relevant content found."
        
        # Determine actual tier used
        if distillations and chunks:
            retrieval_tier = "hybrid"
        elif distillations:
            retrieval_tier = "distillations"
        elif chunks:
            retrieval_tier = "chunks"
        else:
            retrieval_tier = "none"
        
        # Get metadata for citations
        issue_ids = []
        issue_ids.extend([d.get("issue_id") for d in distillations if d.get("issue_id")])
        issue_ids.extend([c.get("issue_id") for c in chunks if c.get("issue_id")])
        metadata = self._get_article_metadata(list(set(issue_ids)))
        
        # Generate response
        user_prompt = f"""Context from Stratechery:
{context}

Question: {question}

Provide a structured analysis based on the context above."""

        response = self.openai.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_RAG},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            max_tokens=1500
        )
        
        answer = response.choices[0].message.content
        
        # Add copyright notice
        answer = self._add_copyright_notice(answer)
        
        # Build sources
        sources = []
        seen_issues = set()

        for d in distillations:
            issue_id = d.get("issue_id")
            if issue_id and issue_id not in seen_issues:
                meta = metadata.get(issue_id, {})
                sources.append({
                    "issue_id": issue_id,
                    "title": meta.get("title", "Unknown"),
                    "date": str(meta.get("publication_date", ""))[:10],
                    "similarity": round(d.get("similarity", 0), 3),
                    "type": "distillation",
                    "thesis_statement": d.get("thesis_statement", ""),  # ADD THIS
                    "topics": d.get("topics", []),  # ADD THIS
                })
                seen_issues.add(issue_id)

        for c in chunks:
            issue_id = c.get("issue_id")
            if issue_id and issue_id not in seen_issues:
                meta = metadata.get(issue_id, {})
                sources.append({
                    "issue_id": issue_id,
                    "title": meta.get("title", "Unknown"),
                    "date": str(meta.get("publication_date", ""))[:10],
                    "similarity": round(c.get("similarity", 0), 3),
                    "type": "chunk",
                    "content": c.get("content", "")[:500],  # ADD THIS - excerpt
                })
                seen_issues.add(issue_id)
        
        print("DEBUG sources:", [(s.get('title'), s.get('thesis_statement', 'MISS')[:30] if s.get('thesis_statement') else 'EMPTY') for s in sources])

        return QueryResult(
            answer=answer,
            sources=sources,
            query=question,
            model=self.model,
            retrieval_tier=retrieval_tier
        )

    def query_reasoning(
        self,
        question: str,
        limit: int = 5,
        threshold: float = 0.3
    ) -> QueryResult:
        """
        Reasoning-first query mode.

        1. LLM reasons using Ben's frameworks (no RAG context)
        2. RAG finds relevant sources that support/contradict the analysis
        3. LLM adds citations where the analysis aligns with actual writings
        """

        # Check for refusal first
        refusal = self._check_refusal(question)
        if refusal:
            return QueryResult(
                answer=refusal,
                sources=[],
                query=question,
                model=self.model,
                retrieval_tier="refused"
            )

        # Step 1: Generate analysis using Ben's frameworks (no RAG)
        logger.info(f"Step 1: Reasoning with Ben's frameworks for: {question[:50]}...")

        reasoning_response = self.openai.chat.completions.create(
            model="gpt-4o",  # Use stronger model for reasoning
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_REASONING},
                {"role": "user", "content": question}
            ],
            temperature=0.7,
            max_tokens=1500
        )

        initial_analysis = reasoning_response.choices[0].message.content
        logger.info(f"Step 1 complete. Analysis length: {len(initial_analysis)} chars")

        # Step 2: Find relevant sources via RAG
        logger.info(f"Step 2: Finding supporting sources...")

        # Search both tiers for supporting evidence
        distillations = self.embeddings.search_distillations(
            question, limit=limit, threshold=threshold
        )
        chunks = self.embeddings.search_chunks(
            question, limit=limit, threshold=threshold
        )

        # Build sources context for citation step
        source_parts = []
        for i, d in enumerate(distillations, 1):
            source_parts.append(f"""
Source {i} (Distillation):
Title: {d.get('title', 'Unknown')}
Date: {str(d.get('publication_date', ''))[:10]}
Thesis: {d.get('thesis_statement', 'N/A')}
Key Arguments: {', '.join(d.get('key_arguments', [])[:3]) if d.get('key_arguments') else 'N/A'}
""")

        for i, c in enumerate(chunks, len(distillations) + 1):
            content = c.get('content', '')[:500]
            source_parts.append(f"""
Source {i} (Excerpt):
Content: {content}
""")

        sources_text = "\n".join(source_parts) if source_parts else "No directly relevant sources found in archive."

        logger.info(f"Step 2 complete. Found {len(distillations)} distillations, {len(chunks)} chunks")

        # Step 3: Add citations
        logger.info(f"Step 3: Adding citations...")

        citation_prompt = CITATION_PROMPT.format(
            sources=sources_text,
            analysis=initial_analysis
        )

        citation_response = self.openai.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You add accurate citations to analysis based on provided sources."},
                {"role": "user", "content": citation_prompt}
            ],
            temperature=0.3,  # Lower temp for more accurate citation matching
            max_tokens=2000
        )

        final_answer = citation_response.choices[0].message.content

        # Add copyright notice
        final_answer = self._add_copyright_notice(final_answer)

        # Build sources list
        issue_ids = []
        issue_ids.extend([d.get("issue_id") for d in distillations if d.get("issue_id")])
        issue_ids.extend([c.get("issue_id") for c in chunks if c.get("issue_id")])
        metadata = self._get_article_metadata(list(set(issue_ids)))

        sources = []
        seen_issues = set()

        for d in distillations:
            issue_id = d.get("issue_id")
            if issue_id and issue_id not in seen_issues:
                meta = metadata.get(issue_id, {})
                sources.append({
                    "issue_id": issue_id,
                    "title": meta.get("title", "Unknown"),
                    "date": str(meta.get("publication_date", ""))[:10],
                    "similarity": round(d.get("similarity", 0), 3),
                    "type": "distillation",
                    "thesis_statement": d.get("thesis_statement", ""),
                    "topics": d.get("topics", []),
                })
                seen_issues.add(issue_id)

        for c in chunks:
            issue_id = c.get("issue_id")
            if issue_id and issue_id not in seen_issues:
                meta = metadata.get(issue_id, {})
                sources.append({
                    "issue_id": issue_id,
                    "title": meta.get("title", "Unknown"),
                    "date": str(meta.get("publication_date", ""))[:10],
                    "similarity": round(c.get("similarity", 0), 3),
                    "type": "chunk",
                    "content": c.get("content", "")[:500],
                })
                seen_issues.add(issue_id)

        logger.info(f"Reasoning-first query complete")

        return QueryResult(
            answer=final_answer,
            sources=sources,
            query=question,
            model="gpt-4o + " + self.model,
            retrieval_tier="reasoning-first"
        )


# Singleton
_query_service = None

def get_query_service() -> QueryService:
    global _query_service
    if _query_service is None:
        _query_service = QueryService()
    return _query_service