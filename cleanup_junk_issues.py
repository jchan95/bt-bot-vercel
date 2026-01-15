#!/usr/bin/env python3
"""
Cleanup Script - Remove non-article entries from the database.

These include:
- Access code emails ("XXXXXX is your Stratechery access code")
- Receipt emails ("Your receipt from Stratechery")
- Password reset emails
- Other transactional emails that aren't actual articles
"""

import os
import re
import sys
from pathlib import Path

# Load environment variables FIRST
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

from supabase import create_client

# Initialize Supabase
url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")  # Note: .env uses SUPABASE_KEY not SUPABASE_SERVICE_KEY

if not url or not key:
    print("Error: SUPABASE_URL and SUPABASE_KEY must be set in .env")
    sys.exit(1)

supabase = create_client(url, key)

# Patterns that indicate junk/non-article entries
JUNK_PATTERNS = [
    r'^\d{6} is your Stratechery access code',  # Access codes
    r'^Your receipt from Stratechery',           # Receipts
    r'^Reset your Stratechery password',         # Password resets
    r'^Confirm your Stratechery email',          # Email confirmations
    r'^Welcome to Stratechery$',                 # Welcome emails (no content)
    r'^Stratechery access code',                 # Alternative access code format
]

def fetch_all_issues():
    """Fetch ALL issues from database using pagination."""
    all_issues = []
    page_size = 1000
    offset = 0

    while True:
        response = supabase.table("stratechery_issues")\
            .select("issue_id, title, word_count")\
            .range(offset, offset + page_size - 1)\
            .execute()

        if not response.data:
            break

        all_issues.extend(response.data)
        print(f"  Fetched {len(all_issues)} issues so far...")

        if len(response.data) < page_size:
            break

        offset += page_size

    return all_issues


def find_junk_issues(dry_run=True):
    """Find all junk issues in the database."""
    print("Fetching all issues (with pagination)...")

    all_issues = fetch_all_issues()
    print(f"Found {len(all_issues)} total issues")

    junk_issues = []

    for issue in all_issues:
        title = issue.get("title", "")

        # Check against junk patterns
        for pattern in JUNK_PATTERNS:
            if re.match(pattern, title, re.IGNORECASE):
                junk_issues.append(issue)
                break
        else:
            # Also flag very short articles (< 50 words) with suspicious titles
            word_count = issue.get("word_count", 0)
            if word_count < 100 and any(keyword in title.lower() for keyword in
                ['access code', 'receipt', 'password', 'confirm', 'verify']):
                junk_issues.append(issue)

    print(f"\nFound {len(junk_issues)} junk issues:")
    print("-" * 60)

    for issue in junk_issues:
        print(f"  [{issue.get('word_count', 0):4d} words] {issue['title'][:60]}")

    return junk_issues


def delete_junk_issues(junk_issues, dry_run=True):
    """Delete junk issues and their related records."""
    if dry_run:
        print(f"\n[DRY RUN] Would delete {len(junk_issues)} issues")
        return

    print(f"\nDeleting {len(junk_issues)} junk issues...")

    for issue in junk_issues:
        issue_id = issue["issue_id"]
        title = issue.get("title", "Unknown")[:50]

        try:
            # First, get all chunk_ids for this issue
            chunks_response = supabase.table("stratechery_chunks")\
                .select("chunk_id")\
                .eq("issue_id", issue_id)\
                .execute()

            chunk_ids = [c["chunk_id"] for c in chunks_response.data] if chunks_response.data else []

            # Delete chunk_embeddings first (references chunks)
            for chunk_id in chunk_ids:
                supabase.table("chunk_embeddings")\
                    .delete()\
                    .eq("chunk_id", chunk_id)\
                    .execute()

            # Delete distillation_embeddings (references distillations)
            dist_response = supabase.table("stratechery_distillations")\
                .select("distillation_id")\
                .eq("issue_id", issue_id)\
                .execute()

            if dist_response.data:
                for dist in dist_response.data:
                    supabase.table("distillation_embeddings")\
                        .delete()\
                        .eq("distillation_id", dist["distillation_id"])\
                        .execute()

            # Now delete in order: distillations, chunks, then issue
            supabase.table("stratechery_distillations")\
                .delete()\
                .eq("issue_id", issue_id)\
                .execute()

            supabase.table("stratechery_chunks")\
                .delete()\
                .eq("issue_id", issue_id)\
                .execute()

            supabase.table("stratechery_issues")\
                .delete()\
                .eq("issue_id", issue_id)\
                .execute()

            print(f"  ✓ Deleted: {title}")

        except Exception as e:
            print(f"  ✗ Failed to delete {title}: {e}")

    print("\nCleanup complete!")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Clean up junk entries from Stratechery database')
    parser.add_argument('--delete', action='store_true',
                       help='Actually delete the entries (default is dry-run)')
    parser.add_argument('--list-only', action='store_true',
                       help='Only list junk entries, do not delete')

    args = parser.parse_args()

    dry_run = not args.delete

    if dry_run:
        print("=== DRY RUN MODE ===")
        print("Use --delete to actually remove entries\n")

    junk_issues = find_junk_issues(dry_run=dry_run)

    if not args.list_only and junk_issues:
        delete_junk_issues(junk_issues, dry_run=dry_run)


if __name__ == '__main__':
    main()
