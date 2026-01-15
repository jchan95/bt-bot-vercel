"""
Ingestion Service
Orchestrates email parsing, cleaning, and database storage.
Handles deduplication and batch processing.
"""

import os
from pathlib import Path
from typing import Optional
from datetime import datetime
from dataclasses import dataclass
import logging

from app.services.email_parser import EmailParser, ParsedEmail
from app.services.html_cleaner import StratecheryExtractor
from app.database import get_supabase_client

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class IngestionResult:
    """Result of an ingestion attempt."""
    success: bool
    issue_id: Optional[str] = None
    title: Optional[str] = None
    publication_date: Optional[str] = None
    error: Optional[str] = None
    skipped: bool = False
    skip_reason: Optional[str] = None
    
    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "issue_id": self.issue_id,
            "title": self.title,
            "publication_date": self.publication_date,
            "error": self.error,
            "skipped": self.skipped,
            "skip_reason": self.skip_reason,
        }


@dataclass
class BatchIngestionResult:
    """Result of batch ingestion."""
    total: int
    successful: int
    skipped: int
    failed: int
    results: list[IngestionResult]
    
    def to_dict(self) -> dict:
        return {
            "total": self.total,
            "successful": self.successful,
            "skipped": self.skipped,
            "failed": self.failed,
            "results": [r.to_dict() for r in self.results],
        }


class IngestionService:
    """
    Handles the full ingestion pipeline:
    1. Parse email (from file or raw content)
    2. Clean HTML to extract text
    3. Check for duplicates
    4. Store in database
    """
    
    def __init__(self):
        self.email_parser = EmailParser()
        self.html_cleaner = StratecheryExtractor()
        self.supabase = get_supabase_client()
    
    def ingest_eml_file(self, file_path: str) -> IngestionResult:
        """Ingest a single .eml file."""
        try:
            # Parse the email
            parsed = self.email_parser.parse_eml_file(file_path)
            return self._process_parsed_email(parsed)
        except FileNotFoundError as e:
            return IngestionResult(success=False, error=f"File not found: {e}")
        except Exception as e:
            logger.error(f"Error ingesting {file_path}: {e}")
            return IngestionResult(success=False, error=str(e))
    
    def ingest_raw_email(
        self, 
        html_body: str,
        subject: str,
        sender: str,
        date: str,
        message_id: Optional[str] = None,
    ) -> IngestionResult:
        """
        Ingest email from raw components (for n8n/API integration).
        
        Args:
            html_body: The HTML content of the email
            subject: Email subject line
            sender: Sender email address
            date: Publication date (ISO string or datetime)
            message_id: Optional email message ID
        """
        try:
            # Parse date if string
            if isinstance(date, str):
                date = datetime.fromisoformat(date.replace('Z', '+00:00'))
            
            # Generate content hash
            import hashlib
            content_hash = hashlib.sha256(html_body.encode('utf-8')).hexdigest()
            
            # Create a ParsedEmail-like object
            parsed = ParsedEmail(
                subject=subject,
                sender=sender,
                recipient="",
                date=date,
                html_body=html_body,
                text_body=None,
                message_id=message_id or "",
                content_hash=content_hash,
                raw_headers={},
            )
            
            return self._process_parsed_email(parsed)
        except Exception as e:
            logger.error(f"Error ingesting raw email '{subject}': {e}")
            return IngestionResult(success=False, error=str(e))
    
    def ingest_mbox(self, file_path: str, limit: int = None, batch_size: int = 50) -> BatchIngestionResult:
        """
        Ingest all emails from an .mbox file (Gmail Takeout export).
        
        Args:
            file_path: Path to the .mbox file
            limit: Optional limit on number of emails to process
            batch_size: Log progress every N emails
        
        Returns:
            BatchIngestionResult with processing stats
        """
        from app.services.email_parser import parse_mbox
        
        file_path = Path(file_path)
        if not file_path.exists():
            return BatchIngestionResult(
                total=0, successful=0, skipped=0, failed=1,
                results=[IngestionResult(success=False, error=f"File not found: {file_path}")]
            )
        
        logger.info(f"Starting .mbox ingestion from {file_path}")
        
        results = []
        count = 0
        
        for parsed in parse_mbox(str(file_path), limit=limit):
            count += 1
            
            # Log progress
            if count % batch_size == 0:
                logger.info(f"Processing email {count}...")
            
            # Process the email
            result = self._process_parsed_email(parsed)
            results.append(result)
            
            if result.success:
                logger.debug(f"  ✓ Ingested: {result.title}")
            elif result.skipped:
                logger.debug(f"  - Skipped: {result.skip_reason}")
            else:
                logger.warning(f"  ✗ Failed: {result.error}")
        
        successful = sum(1 for r in results if r.success)
        skipped = sum(1 for r in results if r.skipped)
        failed = sum(1 for r in results if not r.success and not r.skipped)
        
        logger.info(f"Completed .mbox ingestion: {successful} successful, {skipped} skipped, {failed} failed out of {len(results)} total")
        
        return BatchIngestionResult(
            total=len(results),
            successful=successful,
            skipped=skipped,
            failed=failed,
            results=results,
        )
    
    def ingest_directory(self, directory: str, pattern: str = "*.eml") -> BatchIngestionResult:
        """Ingest all .eml files from a directory."""
        directory = Path(directory)
        
        if not directory.exists():
            return BatchIngestionResult(
                total=0, successful=0, skipped=0, failed=1,
                results=[IngestionResult(success=False, error=f"Directory not found: {directory}")]
            )
        
        eml_files = list(directory.glob(pattern))
        logger.info(f"Found {len(eml_files)} email files in {directory}")
        
        results = []
        for file_path in eml_files:
            logger.info(f"Processing: {file_path.name}")
            result = self.ingest_eml_file(file_path)
            results.append(result)
            
            if result.success:
                logger.info(f"  ✓ Ingested: {result.title}")
            elif result.skipped:
                logger.info(f"  - Skipped: {result.skip_reason}")
            else:
                logger.error(f"  ✗ Failed: {result.error}")
        
        return BatchIngestionResult(
            total=len(results),
            successful=sum(1 for r in results if r.success),
            skipped=sum(1 for r in results if r.skipped),
            failed=sum(1 for r in results if not r.success and not r.skipped),
            results=results,
        )
    
    def _process_parsed_email(self, parsed: ParsedEmail) -> IngestionResult:
        """Process a parsed email and store in database."""
        
        # Check for duplicates first
        if self._is_duplicate(parsed.content_hash):
            return IngestionResult(
                success=False,
                skipped=True,
                skip_reason=f"Duplicate content (hash: {parsed.content_hash[:16]}...)",
                title=parsed.subject,
            )
        
        # Clean the HTML content
        if parsed.html_body:
            cleaned = self.html_cleaner.clean(parsed.html_body)
            clean_text = cleaned['text']
            raw_html = parsed.html_body  # cleaned['html']
            metadata = cleaned['metadata']
        else:
            clean_text = parsed.text_body or ""
            clean_html = ""
            metadata = {}
        
        # Extract title - prefer subject line, fall back to metadata
        title = self._extract_title(parsed.subject, metadata)
        
        # Build the canonical URL if available
        canonical_url = metadata.get('canonical_url', '')
        
        # Prepare the record
        record = {
            "title": title,
            "publication_date": parsed.date.isoformat() if parsed.date else None,
            "raw_html": raw_html,
            "cleaned_text": clean_text,
            "word_count": len(clean_text.split()) if clean_text else 0,
            "content_hash": parsed.content_hash,
            "canonical_url": canonical_url,
        }
        
        # Insert into database
        try:
            response = self.supabase.table("stratechery_issues").insert(record).execute()
            
            if response.data:
                issue_id = response.data[0].get('id')
                return IngestionResult(
                    success=True,
                    issue_id=issue_id,
                    title=title,
                    publication_date=parsed.date.isoformat() if parsed.date else None,
                )
            else:
                return IngestionResult(
                    success=False,
                    error="Database insert returned no data",
                    title=title,
                )
        except Exception as e:
            return IngestionResult(
                success=False,
                error=f"Database error: {str(e)}",
                title=title,
            )
    
    def _is_duplicate(self, content_hash: str) -> bool:
        """Check if an issue with this content hash already exists."""
        try:
            response = (
                self.supabase
                .table("stratechery_issues")
                .select("issue_id")
                .eq("content_hash", content_hash)
                .limit(1)
                .execute()
            )
            return len(response.data) > 0
        except Exception as e:
            logger.warning(f"Error checking for duplicates: {e}")
            return False
    
    def _extract_title(self, subject: str, metadata: dict) -> str:
        """Extract a clean title from subject line or metadata."""
        title = subject
        
        # Remove common newsletter prefixes
        prefixes_to_remove = [
            "Stratechery: ",
            "Stratechery - ",
            "[Stratechery] ",
            "Stratechery Update: ",
            "Stratechery Daily Update: ",
        ]
        
        for prefix in prefixes_to_remove:
            if title.lower().startswith(prefix.lower()):
                title = title[len(prefix):]
                break
        
        # Use metadata title if available and subject was generic
        if metadata.get('title') and len(title) < 10:
            title = metadata['title']
        
        return title.strip()
    
    def get_stats(self) -> dict:
        """Get current ingestion statistics."""
        try:
            # Count total issues
            response = self.supabase.table("stratechery_issues").select("id", count="exact").execute()
            total_issues = response.count if hasattr(response, 'count') else len(response.data)
            
            # Get date range
            earliest = (
                self.supabase
                .table("stratechery_issues")
                .select("publication_date")
                .order("publication_date", desc=False)
                .limit(1)
                .execute()
            )
            
            latest = (
                self.supabase
                .table("stratechery_issues")
                .select("publication_date")
                .order("publication_date", desc=True)
                .limit(1)
                .execute()
            )
            
            return {
                "total_issues": total_issues,
                "earliest_date": earliest.data[0]['publication_date'] if earliest.data else None,
                "latest_date": latest.data[0]['publication_date'] if latest.data else None,
            }
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {"error": str(e)}


# Create singleton instance
_ingestion_service: Optional[IngestionService] = None


def get_ingestion_service() -> IngestionService:
    """Get or create the ingestion service singleton."""
    global _ingestion_service
    if _ingestion_service is None:
        _ingestion_service = IngestionService()
    return _ingestion_service
