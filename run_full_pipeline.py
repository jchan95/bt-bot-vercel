#!/usr/bin/env python3
"""
Full Pipeline Script for BT Bot
================================
Runs the complete ingestion pipeline:
1. Ingest HTML articles → stratechery_issues
2. Generate distillations → stratechery_distillations
3. Create chunks → stratechery_chunks
4. Generate embeddings → distillation_embeddings + chunk_embeddings

Usage:
    python run_full_pipeline.py                    # Run full pipeline
    python run_full_pipeline.py --step ingest     # Run only ingestion
    python run_full_pipeline.py --step distill    # Run only distillation
    python run_full_pipeline.py --step embed      # Run only embeddings
    python run_full_pipeline.py --dry-run         # Preview without changes
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()


def run_ingestion(directory: str, dry_run: bool = False, limit: int = None):
    """Step 1: Ingest HTML articles into database."""
    print("\n" + "=" * 70)
    print("STEP 1: INGESTING HTML ARTICLES")
    print("=" * 70)

    from ingest_html_articles import ingest_directory

    dir_path = Path(directory)
    if not dir_path.exists():
        print(f"Error: Directory not found: {directory}")
        return {'success': False, 'error': 'Directory not found'}

    results = ingest_directory(dir_path, dry_run=dry_run, limit=limit)
    return results


def run_distillation(limit: int = 50, dry_run: bool = False):
    """Step 2: Generate distillations for articles without them."""
    print("\n" + "=" * 70)
    print("STEP 2: GENERATING DISTILLATIONS")
    print("=" * 70)

    from app.services.distillation import get_distillation_service

    service = get_distillation_service()

    # Get stats first
    stats = service.get_stats()
    print(f"Current status:")
    print(f"  Total articles: {stats['total_issues']}")
    print(f"  Existing distillations: {stats['total_distillations']}")
    print(f"  Pending: {stats['pending']}")

    if stats['pending'] == 0:
        print("No articles need distillation!")
        return {'success': True, 'processed': 0}

    if dry_run:
        print(f"\n[DRY RUN] Would process up to {min(limit, stats['pending'])} articles")
        return {'success': True, 'dry_run': True}

    print(f"\nProcessing up to {limit} articles...")
    result = service.distill_batch(limit=limit, force=False, oldest_first=True)

    print(f"\nDistillation complete:")
    print(f"  Successful: {result.successful}")
    print(f"  Skipped: {result.skipped}")
    print(f"  Failed: {result.failed}")

    return {
        'success': True,
        'successful': result.successful,
        'skipped': result.skipped,
        'failed': result.failed
    }


def run_embeddings(limit: int = 100, dry_run: bool = False):
    """Step 3: Generate chunks and embeddings."""
    print("\n" + "=" * 70)
    print("STEP 3: GENERATING CHUNKS AND EMBEDDINGS")
    print("=" * 70)

    from app.services.embeddings import get_embeddings_service

    service = get_embeddings_service()

    # Get current stats
    stats = service.get_stats()
    print(f"Current status:")
    print(f"  Total issues: {stats.get('total_issues', 0)}")
    print(f"  Total distillations: {stats.get('total_distillations', 0)}")
    print(f"  Total chunks: {stats.get('total_chunks', 0)}")
    print(f"  Distillation embeddings: {stats.get('distillation_embeddings', 0)}")
    print(f"  Chunk embeddings: {stats.get('chunk_embeddings', 0)}")

    if dry_run:
        print(f"\n[DRY RUN] Would process embeddings")
        return {'success': True, 'dry_run': True}

    results = {
        'chunks': None,
        'distillation_embeddings': None,
        'chunk_embeddings': None,
    }

    # Step 3a: Generate chunks for articles
    print("\n--- Generating chunks ---")
    chunk_result = service.chunk_batch(limit=limit)
    results['chunks'] = {
        'successful': chunk_result.get('successful', 0),
        'count': chunk_result.get('chunks_created', 0)
    }
    print(f"  Chunked: {chunk_result.get('successful', 0)} articles, {chunk_result.get('chunks_created', 0)} chunks created")

    # Step 3b: Generate distillation embeddings
    print("\n--- Generating distillation embeddings ---")
    dist_result = service.embed_distillations(limit=limit)
    results['distillation_embeddings'] = {
        'successful': dist_result.get('embedded', 0),
        'count': dist_result.get('embedded', 0)
    }
    print(f"  Embedded: {dist_result.get('embedded', 0)} distillations")

    # Step 3c: Generate chunk embeddings
    print("\n--- Generating chunk embeddings ---")
    chunk_embed_result = service.embed_chunks(limit=limit * 20)  # More chunks than articles
    results['chunk_embeddings'] = {
        'successful': chunk_embed_result.get('embedded', 0),
        'count': chunk_embed_result.get('embedded', 0)
    }
    print(f"  Embedded: {chunk_embed_result.get('embedded', 0)} chunks")

    return {'success': True, 'results': results}


def run_full_pipeline(
    html_directory: str = "bt selected articles",
    ingest_limit: int = None,
    distill_limit: int = 50,
    embed_limit: int = 100,
    dry_run: bool = False
):
    """Run the complete pipeline."""
    print("\n" + "#" * 70)
    print("#  BT BOT FULL PIPELINE")
    print("#  " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("#" * 70)

    if dry_run:
        print("\n*** DRY RUN MODE - No changes will be made ***\n")

    results = {}

    # Step 1: Ingest
    try:
        results['ingest'] = run_ingestion(html_directory, dry_run=dry_run, limit=ingest_limit)
    except Exception as e:
        print(f"Ingestion error: {e}")
        results['ingest'] = {'success': False, 'error': str(e)}

    # Step 2: Distill
    try:
        results['distill'] = run_distillation(limit=distill_limit, dry_run=dry_run)
    except Exception as e:
        print(f"Distillation error: {e}")
        results['distill'] = {'success': False, 'error': str(e)}

    # Step 3: Embed
    try:
        results['embed'] = run_embeddings(limit=embed_limit, dry_run=dry_run)
    except Exception as e:
        print(f"Embedding error: {e}")
        results['embed'] = {'success': False, 'error': str(e)}

    # Summary
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)

    for step, result in results.items():
        status = "✓" if result.get('success') else "✗"
        print(f"  {status} {step.upper()}: {result}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description='BT Bot Full Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_full_pipeline.py                      # Run everything
  python run_full_pipeline.py --dry-run            # Preview without changes
  python run_full_pipeline.py --step ingest        # Only ingest HTML files
  python run_full_pipeline.py --step distill       # Only generate distillations
  python run_full_pipeline.py --step embed         # Only generate embeddings
  python run_full_pipeline.py --distill-limit 10   # Limit distillations to 10
        """
    )

    parser.add_argument('--step', choices=['ingest', 'distill', 'embed', 'all'],
                       default='all', help='Which step to run')
    parser.add_argument('--html-dir', default='bt selected articles',
                       help='Directory containing HTML files')
    parser.add_argument('--ingest-limit', type=int, default=None,
                       help='Limit number of files to ingest')
    parser.add_argument('--distill-limit', type=int, default=50,
                       help='Limit number of distillations per run')
    parser.add_argument('--embed-limit', type=int, default=100,
                       help='Limit number of embeddings per run')
    parser.add_argument('--dry-run', action='store_true',
                       help='Preview without making changes')

    args = parser.parse_args()

    if args.step == 'all':
        run_full_pipeline(
            html_directory=args.html_dir,
            ingest_limit=args.ingest_limit,
            distill_limit=args.distill_limit,
            embed_limit=args.embed_limit,
            dry_run=args.dry_run
        )
    elif args.step == 'ingest':
        run_ingestion(args.html_dir, dry_run=args.dry_run, limit=args.ingest_limit)
    elif args.step == 'distill':
        run_distillation(limit=args.distill_limit, dry_run=args.dry_run)
    elif args.step == 'embed':
        run_embeddings(limit=args.embed_limit, dry_run=args.dry_run)


if __name__ == '__main__':
    main()
