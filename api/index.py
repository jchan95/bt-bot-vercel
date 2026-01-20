"""
Vercel Serverless Function Entry Point
This file exposes the FastAPI app for Vercel deployment.
"""

from app.main import app

# Export the app directly - Vercel will detect it as ASGI
# The variable must be named 'app' for Vercel to find it
