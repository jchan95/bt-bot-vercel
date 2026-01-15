"""
BT Bot - Stratechery RAG System
Main FastAPI application with ingestion, query, and evaluation endpoints.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import get_settings
from app.routers.ingestion import router as ingestion_router
from app.routers.distillation import router as distillation_router
from app.routers.embeddings import router as embeddings_router
from app.routers.query import router as query_router
from app.routers.evaluation import router as eval_router
from app.routers.retrieval import router as retrieval_router

# Initialize settings
settings = get_settings()

# Create FastAPI app
app = FastAPI(
    title="BT Bot API",
    description="A RAG system for exploring Stratechery content with Ben Thompson's analytical frameworks.",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(ingestion_router)
app.include_router(distillation_router)
app.include_router(embeddings_router)
app.include_router(query_router)
app.include_router(eval_router)
app.include_router(retrieval_router)

# Future routers (uncomment as implemented):
# app.include_router(eval_router)


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "BT Bot API",
        "version": "0.1.0",
        "description": "Stratechery RAG System",
        "docs": "/docs",
        "endpoints": {
            "ingestion": "/ingest",
            "distillation": "/distill",
            "health": "/health",
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "bt-bot",
        "database": "connected",  # TODO: Actually check connection
    }
