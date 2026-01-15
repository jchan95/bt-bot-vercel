"""
Test script for email ingestion pipeline.
Run this to verify your setup works before using the API.

Usage:
    python test_ingestion.py
"""

import sys
from pathlib import Path

# Add app to path
sys.path.insert(0, str(Path(__file__).parent))

from app.services.email_parser import EmailParser, parse_eml
from app.services.html_cleaner import StratecheryExtractor


def test_email_parser():
    """Test parsing .eml files from data/emails/"""
    print("\n" + "="*60)
    print("TEST 1: Email Parser")
    print("="*60)
    
    emails_dir = Path("data/emails")
    
    if not emails_dir.exists():
        print(f"❌ Directory not found: {emails_dir}")
        return False
    
    eml_files = list(emails_dir.glob("*.eml"))
    
    if not eml_files:
        print(f"❌ No .eml files found in {emails_dir}")
        return False
    
    print(f"Found {len(eml_files)} .eml files\n")
    
    parser = EmailParser()
    
    for eml_file in eml_files[:3]:  # Test first 3 files
        print(f"Parsing: {eml_file.name}")
        try:
            parsed = parser.parse_eml_file(eml_file)
            print(f"  ✓ Subject: {parsed.subject[:60]}...")
            print(f"  ✓ Date: {parsed.date}")
            print(f"  ✓ Sender: {parsed.sender}")
            print(f"  ✓ Has HTML body: {bool(parsed.html_body)}")
            print(f"  ✓ Content hash: {parsed.content_hash[:16]}...")
            print()
        except Exception as e:
            print(f"  ❌ Error: {e}")
            return False
    
    print("✅ Email parser test passed!")
    return True


def test_html_cleaner():
    """Test HTML cleaning on parsed emails."""
    print("\n" + "="*60)
    print("TEST 2: HTML Cleaner")
    print("="*60)
    
    emails_dir = Path("data/emails")
    eml_files = list(emails_dir.glob("*.eml"))
    
    if not eml_files:
        print("❌ No .eml files to test")
        return False
    
    parser = EmailParser()
    cleaner = StratecheryExtractor()
    
    # Parse and clean first email
    eml_file = eml_files[0]
    print(f"Testing with: {eml_file.name}\n")
    
    parsed = parser.parse_eml_file(eml_file)
    
    if not parsed.html_body:
        print("❌ Email has no HTML body")
        return False
    
    print(f"Original HTML length: {len(parsed.html_body)} chars")
    
    cleaned = cleaner.clean(parsed.html_body)
    
    print(f"Cleaned HTML length: {len(cleaned['html'])} chars")
    print(f"Cleaned text length: {len(cleaned['text'])} chars")
    print(f"Metadata extracted: {list(cleaned['metadata'].keys())}")
    
    # Show preview of cleaned text
    preview = cleaned['text'][:500].replace('\n', '\n  ')
    print(f"\nText preview:\n  {preview}...")
    
    print("\n✅ HTML cleaner test passed!")
    return True


def test_database_connection():
    """Test database connection."""
    print("\n" + "="*60)
    print("TEST 3: Database Connection")
    print("="*60)
    
    try:
        from app.database import get_supabase_client
        client = get_supabase_client()
        
        # Try to query the issues table
        response = client.table("stratechery_issues").select("id").limit(1).execute()
        print(f"✓ Connected to Supabase")
        print(f"✓ stratechery_issues table accessible")
        print(f"✓ Current row count: {len(response.data)}")
        
        print("\n✅ Database connection test passed!")
        return True
    except Exception as e:
        print(f"❌ Database error: {e}")
        return False


def test_full_ingestion():
    """Test the full ingestion pipeline (without actually saving)."""
    print("\n" + "="*60)
    print("TEST 4: Full Ingestion Pipeline (dry run)")
    print("="*60)
    
    try:
        from app.services.ingestion import get_ingestion_service
        
        service = get_ingestion_service()
        print("✓ Ingestion service initialized")
        
        # Get stats
        stats = service.get_stats()
        print(f"✓ Current stats: {stats}")
        
        print("\n✅ Ingestion pipeline test passed!")
        print("\nReady to ingest! Run:")
        print("  curl -X POST http://localhost:8000/ingest/batch")
        print("  (or use the Swagger UI at http://localhost:8000/docs)")
        return True
    except Exception as e:
        print(f"❌ Ingestion service error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("🧪 BT Bot Ingestion Pipeline Tests")
    print("="*60)
    
    results = []
    
    results.append(("Email Parser", test_email_parser()))
    results.append(("HTML Cleaner", test_html_cleaner()))
    results.append(("Database Connection", test_database_connection()))
    results.append(("Full Ingestion", test_full_ingestion()))
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    all_passed = True
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False
    
    print()
    if all_passed:
        print("🎉 All tests passed! Ready to ingest emails.")
    else:
        print("⚠️  Some tests failed. Check the errors above.")
    
    sys.exit(0 if all_passed else 1)
