"""
Ingest HTML articles from Stratechery website (saved pages).
These are different from email format - they're full webpage HTML.
"""

import os
import re
import hashlib
import argparse
from pathlib import Path
from datetime import datetime
from bs4 import BeautifulSoup
from supabase import create_client
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Supabase
supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)


def extract_article_data(html_content: str, filename: str) -> dict:
    """Extract article data from Stratechery webpage HTML."""
    soup = BeautifulSoup(html_content, 'html.parser')

    # Extract title from <title> tag or filename
    title_tag = soup.find('title')
    if title_tag:
        title = title_tag.get_text()
        # Clean up title - remove " – Stratechery by Ben Thompson"
        title = re.sub(r'\s*[–-]\s*Stratechery by Ben Thompson\s*$', '', title)
        title = title.strip()
    else:
        # Fall back to filename
        title = filename.replace('.html', '').replace(' – Stratechery by Ben Thompson', '')

    # Extract publication date from first <time datetime="...">
    time_tag = soup.find('time', datetime=True)
    pub_date = None
    if time_tag:
        datetime_str = time_tag.get('datetime')
        try:
            # Parse ISO format datetime
            pub_date = datetime.fromisoformat(datetime_str.replace('Z', '+00:00'))
        except (ValueError, TypeError):
            pass

    # Extract canonical URL
    canonical = soup.find('link', rel='canonical')
    canonical_url = canonical.get('href') if canonical else ''

    # Extract main content from entry-content div
    content_div = soup.find('div', class_=re.compile(r'entry-content'))

    if content_div:
        # Remove unwanted elements
        for unwanted in content_div.find_all(['script', 'style', 'nav', 'aside']):
            unwanted.decompose()

        # Remove "Related Articles" section
        for related in content_div.find_all(class_=re.compile(r'related|jp-relatedposts')):
            related.decompose()

        # Get text
        clean_text = content_div.get_text(separator='\n', strip=True)
    else:
        # Fallback: try article tag
        article = soup.find('article')
        if article:
            clean_text = article.get_text(separator='\n', strip=True)
        else:
            clean_text = soup.get_text(separator='\n', strip=True)

    # Clean up text
    # Remove excessive newlines
    clean_text = re.sub(r'\n{3,}', '\n\n', clean_text)
    # Remove common footer patterns
    clean_text = re.sub(r'(?i)share this:.*$', '', clean_text, flags=re.DOTALL)
    clean_text = clean_text.strip()

    # Generate content hash
    content_hash = hashlib.sha256(clean_text.encode('utf-8')).hexdigest()

    return {
        'title': title,
        'publication_date': pub_date.isoformat() if pub_date else None,
        'cleaned_text': clean_text,
        'raw_html': html_content,
        'canonical_url': canonical_url,
        'word_count': len(clean_text.split()),
        'content_hash': content_hash,
    }


def is_duplicate(content_hash: str) -> bool:
    """Check if article already exists in database."""
    try:
        response = supabase.table("stratechery_issues")\
            .select("issue_id")\
            .eq("content_hash", content_hash)\
            .limit(1)\
            .execute()
        return len(response.data) > 0
    except Exception as e:
        print(f"  Warning: Could not check for duplicates: {e}")
        return False


def check_title_exists(title: str) -> bool:
    """Check if an article with this title already exists."""
    try:
        response = supabase.table("stratechery_issues")\
            .select("issue_id")\
            .eq("title", title)\
            .limit(1)\
            .execute()
        return len(response.data) > 0
    except Exception as e:
        print(f"  Warning: Could not check for title: {e}")
        return False


def ingest_html_file(file_path: Path, dry_run: bool = False) -> dict:
    """Ingest a single HTML file."""
    result = {
        'file': file_path.name,
        'success': False,
        'skipped': False,
        'error': None,
    }

    try:
        # Read file
        with open(file_path, 'r', encoding='utf-8') as f:
            html_content = f.read()

        # Extract data
        data = extract_article_data(html_content, file_path.name)
        result['title'] = data['title']
        result['date'] = data['publication_date']
        result['word_count'] = data['word_count']

        # Check for duplicates
        if is_duplicate(data['content_hash']):
            result['skipped'] = True
            result['skip_reason'] = 'Duplicate content hash'
            return result

        if check_title_exists(data['title']):
            result['skipped'] = True
            result['skip_reason'] = 'Title already exists'
            return result

        if dry_run:
            result['success'] = True
            result['dry_run'] = True
            return result

        # Insert into database
        response = supabase.table("stratechery_issues").insert({
            'title': data['title'],
            'publication_date': data['publication_date'],
            'cleaned_text': data['cleaned_text'],
            'raw_html': data['raw_html'],
            'canonical_url': data['canonical_url'],
            'word_count': data['word_count'],
            'content_hash': data['content_hash'],
        }).execute()

        if response.data:
            result['success'] = True
            result['issue_id'] = response.data[0].get('issue_id')
        else:
            result['error'] = 'No data returned from insert'

    except Exception as e:
        result['error'] = str(e)

    return result


def ingest_directory(directory: Path, dry_run: bool = False, limit: int = None):
    """Ingest all HTML files from a directory."""
    html_files = sorted(directory.glob('*.html'))

    if limit:
        html_files = html_files[:limit]

    print(f"Found {len(html_files)} HTML files in {directory}")
    print(f"Mode: {'DRY RUN' if dry_run else 'LIVE'}")
    print("-" * 60)

    results = {
        'total': len(html_files),
        'successful': 0,
        'skipped': 0,
        'failed': 0,
        'details': []
    }

    for i, file_path in enumerate(html_files, 1):
        print(f"\n[{i}/{len(html_files)}] {file_path.name[:60]}...")

        result = ingest_html_file(file_path, dry_run=dry_run)
        results['details'].append(result)

        if result['success']:
            results['successful'] += 1
            print(f"  ✓ Ingested: {result.get('title', 'Unknown')[:50]}")
            print(f"    Date: {result.get('date', 'Unknown')}, Words: {result.get('word_count', 0)}")
        elif result['skipped']:
            results['skipped'] += 1
            print(f"  - Skipped: {result.get('skip_reason', 'Unknown reason')}")
        else:
            results['failed'] += 1
            print(f"  ✗ Failed: {result.get('error', 'Unknown error')}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total files:  {results['total']}")
    print(f"Successful:   {results['successful']}")
    print(f"Skipped:      {results['skipped']}")
    print(f"Failed:       {results['failed']}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Ingest HTML articles from Stratechery')
    parser.add_argument('directory', nargs='?', default='bt selected articles',
                       help='Directory containing HTML files')
    parser.add_argument('--dry-run', action='store_true',
                       help='Parse files but do not insert into database')
    parser.add_argument('--limit', type=int, default=None,
                       help='Limit number of files to process')
    parser.add_argument('--single', type=str, default=None,
                       help='Process a single file')

    args = parser.parse_args()

    if args.single:
        file_path = Path(args.single)
        if not file_path.exists():
            print(f"File not found: {file_path}")
            return
        result = ingest_html_file(file_path, dry_run=args.dry_run)
        print(result)
    else:
        directory = Path(args.directory)
        if not directory.exists():
            print(f"Directory not found: {directory}")
            return
        ingest_directory(directory, dry_run=args.dry_run, limit=args.limit)


if __name__ == '__main__':
    main()
