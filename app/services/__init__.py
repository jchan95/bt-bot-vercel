"""
Services module for BT Bot.
"""

from app.services.email_parser import EmailParser, ParsedEmail, parse_eml
from app.services.html_cleaner import HTMLCleaner, StratecheryExtractor, clean_html, clean_stratechery
from app.services.ingestion import IngestionService, get_ingestion_service

__all__ = [
    "EmailParser",
    "ParsedEmail", 
    "parse_eml",
    "HTMLCleaner",
    "StratecheryExtractor",
    "clean_html",
    "clean_stratechery",
    "IngestionService",
    "get_ingestion_service",
]
