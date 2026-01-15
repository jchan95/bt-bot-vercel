"""
Ingestion Router
API endpoints for email ingestion - supports both file uploads and JSON payloads.
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
import tempfile
import os

from app.services.ingestion import get_ingestion_service

router = APIRouter(prefix="/ingest", tags=["ingestion"])


# Request/Response Models

class EmailPayload(BaseModel):
    """Payload for ingesting email via JSON (from n8n or direct API calls)."""
    subject: str = Field(..., description="Email subject line")
    sender: str = Field(..., description="Sender email address")
    date: str = Field(..., description="Email date in ISO format")
    html_body: str = Field(..., description="HTML content of the email")
    text_body: Optional[str] = Field(None, description="Plain text content (optional)")
    message_id: Optional[str] = Field(None, description="Email message ID (optional)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "subject": "Stratechery: Apple's AI Strategy",
                "sender": "ben@stratechery.com",
                "date": "2024-01-15T08:00:00Z",
                "html_body": "<html><body><h1>Apple's AI Strategy</h1><p>Content here...</p></body></html>",
                "message_id": "<abc123@mail.stratechery.com>"
            }
        }


class IngestionResponse(BaseModel):
    """Response from ingestion endpoint."""
    success: bool
    issue_id: Optional[str] = None
    title: Optional[str] = None
    publication_date: Optional[str] = None
    error: Optional[str] = None
    skipped: bool = False
    skip_reason: Optional[str] = None


class BatchIngestionResponse(BaseModel):
    """Response from batch ingestion."""
    total: int
    successful: int
    skipped: int
    failed: int
    message: str


class StatsResponse(BaseModel):
    """Ingestion statistics."""
    total_issues: int
    earliest_date: Optional[str] = None
    latest_date: Optional[str] = None


# Endpoints

@router.post("/email", response_model=IngestionResponse)
async def ingest_email(payload: EmailPayload):
    """
    Ingest a single email from JSON payload.
    
    This endpoint is designed for:
    - n8n webhook integration
    - Direct API calls
    - Testing
    
    The email will be processed, cleaned, and stored in the database.
    Duplicate emails (same content hash) will be skipped.
    """
    service = get_ingestion_service()
    
    result = service.ingest_raw_email(
        html_body=payload.html_body,
        subject=payload.subject,
        sender=payload.sender,
        date=payload.date,
        message_id=payload.message_id,
    )
    
    return IngestionResponse(**result.to_dict())


@router.post("/upload", response_model=IngestionResponse)
async def upload_eml_file(file: UploadFile = File(...)):
    """
    Upload and ingest a single .eml file.
    
    Accepts standard .eml email files exported from Gmail or other email clients.
    """
    if not file.filename.endswith('.eml'):
        raise HTTPException(
            status_code=400, 
            detail="File must be an .eml file"
        )
    
    # Save to temp file
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.eml') as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        # Process the file
        service = get_ingestion_service()
        result = service.ingest_eml_file(tmp_path)
        
        return IngestionResponse(**result.to_dict())
    
    finally:
        # Clean up temp file
        if 'tmp_path' in locals():
            os.unlink(tmp_path)


@router.post("/batch", response_model=BatchIngestionResponse)
async def batch_ingest(directory: str = "data/emails"):
    """
    Batch ingest all .eml files from a directory on the server.
    
    Default directory is `data/emails/`.
    
    This is useful for:
    - Initial bulk import
    - Processing Gmail Takeout exports
    - Local development testing
    """
    service = get_ingestion_service()
    result = service.ingest_directory(directory)
    
    return BatchIngestionResponse(
        total=result.total,
        successful=result.successful,
        skipped=result.skipped,
        failed=result.failed,
        message=f"Processed {result.total} files: {result.successful} successful, {result.skipped} skipped, {result.failed} failed"
    )


@router.post("/mbox", response_model=BatchIngestionResponse)
async def ingest_mbox_file(file_path: str = "data/Stratechery.mbox", limit: Optional[int] = None):
    """
    Ingest all emails from an .mbox file (Gmail Takeout export).
    
    Default file path is `data/Stratechery.mbox`.
    
    Args:
        file_path: Path to the .mbox file
        limit: Optional limit on number of emails to process (useful for testing)
    
    This is useful for:
    - Processing Gmail Takeout exports
    - Bulk import of historical emails
    """
    service = get_ingestion_service()
    result = service.ingest_mbox(file_path, limit=limit)
    
    return BatchIngestionResponse(
        total=result.total,
        successful=result.successful,
        skipped=result.skipped,
        failed=result.failed,
        message=f"Processed {result.total} emails: {result.successful} successful, {result.skipped} skipped, {result.failed} failed"
    )


@router.get("/stats", response_model=StatsResponse)
async def get_ingestion_stats():
    """
    Get current ingestion statistics.
    
    Returns the total number of issues ingested and the date range covered.
    """
    service = get_ingestion_service()
    stats = service.get_stats()
    
    if "error" in stats:
        raise HTTPException(status_code=500, detail=stats["error"])
    
    return StatsResponse(
        total_issues=stats.get("total_issues", 0),
        earliest_date=stats.get("earliest_date"),
        latest_date=stats.get("latest_date"),
    )


class WebhookResponse(BaseModel):
    """Response from webhook endpoint with full pipeline status."""
    success: bool
    issue_id: Optional[str] = None
    title: Optional[str] = None
    publication_date: Optional[str] = None
    error: Optional[str] = None
    skipped: bool = False
    skip_reason: Optional[str] = None
    pipeline: Optional[dict] = None  # Status of distillation/embedding


@router.post("/webhook", response_model=WebhookResponse)
async def webhook_ingest(payload: EmailPayload, background_tasks: BackgroundTasks):
    """
    Webhook endpoint for n8n automation.

    Ingests email AND runs the full pipeline:
    1. Ingest email → create issue
    2. Distill → extract thesis, claims, topics
    3. Embed → generate embeddings for search

    The distillation/embedding runs in the background so the webhook returns quickly.

    n8n setup:
    - Trigger: Gmail trigger on new email from ben@stratechery.com
    - HTTP Request node: POST to http://your-server:8000/ingest/webhook
    - Body: { "subject": "{{subject}}", "sender": "{{from}}", "date": "{{date}}", "html_body": "{{html}}" }
    """
    from app.services.distillation import get_distillation_service
    from app.services.embeddings import get_embeddings_service

    service = get_ingestion_service()

    # Step 1: Ingest the email
    result = service.ingest_raw_email(
        html_body=payload.html_body,
        subject=payload.subject,
        sender=payload.sender,
        date=payload.date,
        message_id=payload.message_id,
    )

    if not result.success or result.skipped:
        return WebhookResponse(**result.to_dict())

    # Step 2: Run pipeline in background
    issue_id = result.issue_id

    def run_pipeline(issue_id: str):
        """Background task to distill and embed the new article."""
        import logging
        logger = logging.getLogger(__name__)

        try:
            # Distill
            distill_service = get_distillation_service()
            distill_result = distill_service.distill_article(issue_id)
            logger.info(f"Distilled {issue_id}: {distill_result}")

            # Embed (just the new one)
            embed_service = get_embeddings_service()
            embed_result = embed_service.embed_distillations(limit=1)
            logger.info(f"Embedded distillation for {issue_id}: {embed_result}")

        except Exception as e:
            logger.error(f"Pipeline failed for {issue_id}: {e}")

    background_tasks.add_task(run_pipeline, issue_id)

    response_dict = result.to_dict()
    response_dict["pipeline"] = {"status": "started", "message": "Distillation and embedding running in background"}

    return WebhookResponse(**response_dict)


# Health check for n8n
@router.get("/health")
async def ingestion_health():
    """Health check for the ingestion service."""
    return {"status": "ok", "service": "ingestion"}
