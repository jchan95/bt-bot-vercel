"""
Embeddings Service - Generate and manage vector embeddings for Stratechery content.
"""
import logging
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import tiktoken

from openai import OpenAI

from app.database import get_supabase_client

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingResult:
    """Result of embedding operation."""
    success: bool
    embedded_count: int = 0
    error: Optional[str] = None


def count_tokens(text: str, model: str = "text-embedding-3-small") -> int:
    """Count tokens in text."""
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except Exception:
        # Fallback: rough estimate
        return len(text) // 4


def chunk_text(text: str, max_tokens: int = 500, overlap_tokens: int = 50) -> List[str]:
    """
    Split text into chunks with overlap.
    
    Args:
        text: Text to chunk
        max_tokens: Maximum tokens per chunk
        overlap_tokens: Overlap between chunks
    
    Returns:
        List of text chunks
    """
    if not text:
        return []
    
    # Split by paragraphs first
    paragraphs = text.split('\n\n')
    
    chunks = []
    current_chunk = []
    current_tokens = 0
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        
        para_tokens = count_tokens(para)
        
        # If single paragraph is too long, split by sentences
        if para_tokens > max_tokens:
            sentences = para.replace('. ', '.|').split('|')
            for sentence in sentences:
                sent_tokens = count_tokens(sentence)
                if current_tokens + sent_tokens > max_tokens and current_chunk:
                    chunks.append('\n\n'.join(current_chunk))
                    # Keep some overlap
                    overlap_text = current_chunk[-1] if current_chunk else ""
                    current_chunk = [overlap_text] if overlap_text else []
                    current_tokens = count_tokens(overlap_text) if overlap_text else 0
                
                current_chunk.append(sentence)
                current_tokens += sent_tokens
        else:
            if current_tokens + para_tokens > max_tokens and current_chunk:
                chunks.append('\n\n'.join(current_chunk))
                # Keep last paragraph for overlap
                overlap_text = current_chunk[-1] if current_chunk else ""
                current_chunk = [overlap_text] if overlap_text else []
                current_tokens = count_tokens(overlap_text) if overlap_text else 0
            
            current_chunk.append(para)
            current_tokens += para_tokens
    
    # Don't forget the last chunk
    if current_chunk:
        chunks.append('\n\n'.join(current_chunk))
    
    return chunks


class EmbeddingsService:
    """Service for generating and managing embeddings."""
    
    def __init__(self):
        self.supabase = get_supabase_client()
        self.openai = OpenAI()
        self.model = "text-embedding-3-small"
        self.dimensions = 1536
    
    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for a single text."""
        response = self.openai.embeddings.create(
            model=self.model,
            input=text
        )
        return response.data[0].embedding
    
    def _get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts in one API call."""
        if not texts:
            return []
        
        response = self.openai.embeddings.create(
            model=self.model,
            input=texts
        )
        return [item.embedding for item in response.data]
    
    # ==================== CHUNKING ====================
    
    def chunk_article(self, issue_id: str, force: bool = False) -> EmbeddingResult:
        """
        Chunk a single article and store chunks.
        
        Args:
            issue_id: UUID of the article
            force: If True, rechunk even if chunks exist
        """
        try:
            # Check if already chunked
            if not force:
                existing = self.supabase.table("stratechery_chunks")\
                    .select("chunk_id")\
                    .eq("issue_id", issue_id)\
                    .execute()
                
                if existing.data:
                    return EmbeddingResult(success=True, embedded_count=0, error="Already chunked")
            
            # Fetch article
            article = self.supabase.table("stratechery_issues")\
                .select("issue_id, cleaned_text")\
                .eq("issue_id", issue_id)\
                .single()\
                .execute()
            
            if not article.data:
                return EmbeddingResult(success=False, error="Article not found")
            
            text = article.data.get("cleaned_text", "")
            if not text:
                return EmbeddingResult(success=False, error="No content")
            
            # Chunk the text
            chunks = chunk_text(text)
            
            if not chunks:
                return EmbeddingResult(success=False, error="No chunks generated")
            
            # Delete existing chunks if force
            if force:
                self.supabase.table("stratechery_chunks")\
                    .delete()\
                    .eq("issue_id", issue_id)\
                    .execute()
            
            # Insert chunks
            records = [
                {
                    "issue_id": issue_id,
                    "chunk_index": i,
                    "content": chunk,
                    "token_count": count_tokens(chunk)
                }
                for i, chunk in enumerate(chunks)
            ]
            
            self.supabase.table("stratechery_chunks").insert(records).execute()
            
            return EmbeddingResult(success=True, embedded_count=len(chunks))
            
        except Exception as e:
            logger.error(f"Error chunking article {issue_id}: {e}")
            return EmbeddingResult(success=False, error=str(e))
    
    def chunk_batch(self, limit: int = 50) -> Dict[str, Any]:
        """Chunk articles that haven't been chunked yet."""
        # Get articles without chunks
        all_issues = self.supabase.table("stratechery_issues")\
            .select("issue_id")\
            .execute()
        
        existing_chunks = self.supabase.table("stratechery_chunks")\
            .select("issue_id")\
            .execute()
        
        chunked_ids = {c["issue_id"] for c in existing_chunks.data} if existing_chunks.data else set()
        pending = [i["issue_id"] for i in all_issues.data if i["issue_id"] not in chunked_ids][:limit]
        
        results = {"total": len(pending), "successful": 0, "failed": 0, "chunks_created": 0}
        
        for issue_id in pending:
            result = self.chunk_article(issue_id)
            if result.success:
                results["successful"] += 1
                results["chunks_created"] += result.embedded_count
            else:
                results["failed"] += 1
        
        return results
    
    # ==================== CHUNK EMBEDDINGS ====================
    
    def embed_chunks(self, limit: int = 100) -> Dict[str, Any]:
        """Generate embeddings for chunks that don't have them yet."""
        
        # Get ALL existing embedding IDs (paginated)
        embedded_ids = set()
        offset = 0
        batch_size = 1000
        
        while True:
            batch = self.supabase.table("chunk_embeddings")\
                .select("chunk_id")\
                .range(offset, offset + batch_size - 1)\
                .execute()
            
            if not batch.data:
                break
                
            embedded_ids.update(e["chunk_id"] for e in batch.data)
            offset += batch_size
            
            if len(batch.data) < batch_size:
                break
        
        # Get chunks that need embedding (paginated)
        pending = []
        offset = 0
        
        while len(pending) < limit:
            batch = self.supabase.table("stratechery_chunks")\
                .select("chunk_id, content")\
                .range(offset, offset + batch_size - 1)\
                .execute()
            
            if not batch.data:
                break
            
            for c in batch.data:
                if c["chunk_id"] not in embedded_ids:
                    pending.append(c)
                    if len(pending) >= limit:
                        break
            
            offset += batch_size
            
            if len(batch.data) < batch_size:
                break
        
        if not pending:
            return {"total": 0, "embedded": 0, "message": "No chunks to embed"}
        
        results = {"total": len(pending), "embedded": 0, "failed": 0}
        
        # Process in batches of 20
        process_batch_size = 20
        for i in range(0, len(pending), process_batch_size):
            batch = pending[i:i + process_batch_size]
            texts = [c["content"][:8000] if c["content"] else "" for c in batch]
            
            try:
                embeddings = self._get_embeddings_batch(texts)
                
                records = [
                    {
                        "chunk_id": batch[j]["chunk_id"],
                        "embedding": embeddings[j],
                        "model_name": self.model
                    }
                    for j in range(len(batch))
                ]
                
                self.supabase.table("chunk_embeddings").upsert(records, on_conflict="chunk_id").execute()
                results["embedded"] += len(batch)
                
            except Exception as e:
                logger.error(f"Error embedding batch: {e}")
                results["failed"] += len(batch)
        
        return results
    
    # ==================== DISTILLATION EMBEDDINGS ====================
    
    def embed_distillations(self, limit: int = 100) -> Dict[str, Any]:
        """Generate embeddings for distillations that don't have them yet."""
        # Get ALL distillations using pagination (Supabase 1000 row limit)
        all_distillations = []
        page_size = 1000
        offset = 0

        while True:
            batch = self.supabase.table("stratechery_distillations")\
                .select("distillation_id, thesis_statement, key_claims, topics")\
                .range(offset, offset + page_size - 1)\
                .execute()

            if not batch.data:
                break
            all_distillations.extend(batch.data)
            if len(batch.data) < page_size:
                break
            offset += page_size

        # Get ALL existing embeddings using pagination
        all_existing = []
        offset = 0

        while True:
            batch = self.supabase.table("distillation_embeddings")\
                .select("distillation_id")\
                .range(offset, offset + page_size - 1)\
                .execute()

            if not batch.data:
                break
            all_existing.extend(batch.data)
            if len(batch.data) < page_size:
                break
            offset += page_size

        embedded_ids = {e["distillation_id"] for e in all_existing} if all_existing else set()
        pending = [d for d in all_distillations if d["distillation_id"] not in embedded_ids][:limit]

        logger.info(f"Found {len(all_distillations)} total distillations, {len(embedded_ids)} already embedded, {len(pending)} pending (limit={limit})")
        
        if not pending:
            return {"total": 0, "embedded": 0, "message": "No distillations to embed"}
        
        results = {"total": len(pending), "embedded": 0, "failed": 0}
        
        # Build text to embed for each distillation
        texts_to_embed = []
        valid_pending = []  # Track which distillations have valid text

        for d in pending:
            # Combine thesis + key claims + topics for richer embedding
            parts = [d.get("thesis_statement", "")]

            claims = d.get("key_claims", [])
            if claims:
                claim_texts = [c.get("claim", "") for c in claims if isinstance(c, dict)]
                parts.extend(claim_texts)

            topics = d.get("topics", [])
            if topics:
                parts.append("Topics: " + ", ".join(topics))

            combined = "\n".join(filter(None, parts)).strip()

            # Skip empty distillations - OpenAI API rejects empty strings
            if not combined:
                logger.warning(f"Skipping distillation {d['distillation_id']} - no text to embed")
                results["failed"] += 1
                continue

            texts_to_embed.append(combined)
            valid_pending.append(d)

        # Use valid_pending instead of pending from here on
        pending = valid_pending

        if not pending:
            return {"total": results["total"], "embedded": 0, "failed": results["failed"], "message": "No valid distillations to embed"}
        
        # Process in batches
        batch_size = 100
        for i in range(0, len(pending), batch_size):
            batch_items = pending[i:i + batch_size]
            batch_texts = texts_to_embed[i:i + batch_size]
            
            try:
                embeddings = self._get_embeddings_batch(batch_texts)
                
                records = [
                    {
                        "distillation_id": batch_items[j]["distillation_id"],
                        "embedding": embeddings[j],
                        "embedded_text": batch_texts[j][:1000],  # Store truncated for reference
                        "model": self.model
                    }
                    for j in range(len(batch_items))
                ]
                
                self.supabase.table("distillation_embeddings").insert(records).execute()
                results["embedded"] += len(batch_items)
                
            except Exception as e:
                logger.error(f"Error embedding distillations batch: {e}")
                results["failed"] += len(batch_items)
        
        return results
    
    # ==================== SEARCH ====================
    
    def search_chunks(self, query: str, limit: int = 5, threshold: float = 0.3) -> List[Dict]:
        """Search for relevant chunks using vector similarity."""
        query_embedding = self._get_embedding(query)
        
        # Convert list to string format for Supabase
        embedding_str = '[' + ','.join(str(x) for x in query_embedding) + ']'
        
        result = self.supabase.rpc(
            "match_stratechery_chunks",
            {
                "query_embedding": embedding_str,
                "match_count": limit,
                "match_threshold": threshold
            }
        ).execute()
        
        return result.data or []
    
    def search_distillations(self, query: str, limit: int = 5, threshold: float = 0.3) -> List[Dict]:
        """Search for relevant distillations using vector similarity."""
        import numpy as np
        
        query_embedding = self._get_embedding(query)
        
        # Get all distillation embeddings
        result = self.supabase.table("distillation_embeddings")\
            .select("distillation_id, embedding")\
            .execute()
        
        if not result.data:
            return []
        
        # Compute cosine similarity in Python
        query_vec = np.array(query_embedding)
        
        similarities = []
        for row in result.data:
            emb = row['embedding']
            if isinstance(emb, str):
                emb = [float(x) for x in emb.strip('[]').split(',')]
            emb_vec = np.array(emb)
            
            # Cosine similarity
            sim = np.dot(query_vec, emb_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(emb_vec))
            
            if sim > threshold:
                similarities.append({
                    'distillation_id': row['distillation_id'],
                    'similarity': float(sim)
                })
        
        # Sort and limit
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        top_results = similarities[:limit]
        
        if not top_results:
            return []
        
        # Fetch full data for top results
        dist_ids = [r['distillation_id'] for r in top_results]
        
        distillations = self.supabase.table("stratechery_distillations")\
            .select("distillation_id, issue_id, thesis_statement, key_claims, topics, entities")\
            .in_("distillation_id", dist_ids)\
            .execute()
        
        issue_ids = [d['issue_id'] for d in distillations.data] if distillations.data else []
        
        issues = self.supabase.table("stratechery_issues")\
            .select("issue_id, title, publication_date")\
            .in_("issue_id", issue_ids)\
            .execute()
        
        issue_map = {i['issue_id']: i for i in issues.data} if issues.data else {}
        dist_map = {d['distillation_id']: d for d in distillations.data} if distillations.data else {}
        
        # Combine results
        # Combine results
        final_results = []
        for r in top_results:
            dist = dist_map.get(r['distillation_id'], {})
            issue = issue_map.get(dist.get('issue_id'), {})
            final_results.append({
                'distillation_id': r['distillation_id'],
                'issue_id': dist.get('issue_id'),
                'title': issue.get('title'),
                'publication_date': issue.get('publication_date'),
                'similarity': r['similarity'],
                'thesis_statement': dist.get('thesis_statement', ''),
                'topics': dist.get('topics', []),
                'key_claims': dist.get('key_claims', []),
                'entities': dist.get('entities', {}),
            })

        print("DEBUG final_results:", [(r.get('title'), r.get('thesis_statement', 'MISSING')[:50] if r.get('thesis_statement') else 'EMPTY') for r in final_results])
        return final_results
        
    
    # ==================== STATS ====================
    
    def get_stats(self) -> Dict[str, Any]:
        """Get embedding statistics and metadata for the UI."""
        issues = self.supabase.table("stratechery_issues")\
            .select("issue_id", count="exact").execute()

        chunks = self.supabase.table("stratechery_chunks")\
            .select("chunk_id", count="exact").execute()

        chunk_embeds = self.supabase.table("chunk_embeddings")\
            .select("embedding_id", count="exact").execute()

        distillations = self.supabase.table("stratechery_distillations")\
            .select("distillation_id", count="exact").execute()

        distill_embeds = self.supabase.table("distillation_embeddings")\
            .select("embedding_id", count="exact").execute()

        # Get date range of articles
        oldest = self.supabase.table("stratechery_issues")\
            .select("publication_date")\
            .order("publication_date", desc=False)\
            .limit(1).execute()

        newest = self.supabase.table("stratechery_issues")\
            .select("publication_date")\
            .order("publication_date", desc=True)\
            .limit(1).execute()

        # Get sample topics from distillations
        sample_topics = self.supabase.table("stratechery_distillations")\
            .select("topics")\
            .limit(50).execute()

        # Flatten and count topics
        topic_counts = {}
        for d in (sample_topics.data or []):
            for topic in (d.get("topics") or []):
                topic_counts[topic] = topic_counts.get(topic, 0) + 1

        # Get top 10 topics
        top_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[:10]

        # Get a few recent article titles as examples
        recent_articles = self.supabase.table("stratechery_issues")\
            .select("title, publication_date")\
            .order("publication_date", desc=True)\
            .limit(5).execute()

        return {
            "total_articles": issues.count or 0,
            "total_chunks": chunks.count or 0,
            "chunk_embeddings": chunk_embeds.count or 0,
            "chunks_pending_embedding": (chunks.count or 0) - (chunk_embeds.count or 0),
            "total_distillations": distillations.count or 0,
            "distillation_embeddings": distill_embeds.count or 0,
            "distillations_pending_embedding": (distillations.count or 0) - (distill_embeds.count or 0),
            "date_range": {
                "oldest": oldest.data[0]["publication_date"] if oldest.data else None,
                "newest": newest.data[0]["publication_date"] if newest.data else None,
            },
            "top_topics": [t[0] for t in top_topics],
            "recent_articles": [
                {"title": a["title"], "date": a["publication_date"]}
                for a in (recent_articles.data or [])
            ],
        }


# Singleton
_embeddings_service = None

def get_embeddings_service() -> EmbeddingsService:
    global _embeddings_service
    if _embeddings_service is None:
        _embeddings_service = EmbeddingsService()
    return _embeddings_service