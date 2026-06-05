"""
Main CLI entry point for the nAChR PubMed Data Scraper.

Usage:
    python main.py --subunits CHRNA7 CHRNA4 CHRNB2   # Search specific subunits
    python main.py --all                               # Search all 16 subunits
    python main.py --resume                            # Resume interrupted run
    python main.py --subunits CHRNA7 --dry-run         # Search only, show counts
    python main.py --reset                             # Clear checkpoints and start fresh
"""

import argparse
import sys
import time

from tqdm import tqdm

import config
from checkpoint import CheckpointManager
from pubmed_search import search_all_subunits
from paper_fetcher import PaperFetcher
from llm_extractor import setup_gemini, extract_mutations_from_paper
from excel_writer import ExcelWriter


def parse_args():
    parser = argparse.ArgumentParser(
        description="Automated PubMed data scraper for nAChR mutation studies.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --subunits CHRNA7 CHRNA4     Search for CHRNA7 and CHRNA4 papers
  python main.py --all                        Search all 16 nAChR subunits
  python main.py --resume                     Resume a previously interrupted run
  python main.py --subunits CHRNA7 --dry-run  Just count papers, don't process
  python main.py --reset                      Clear all checkpoints and output
        """,
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--subunits",
        nargs="+",
        choices=config.ALL_SUBUNITS,
        help="Specific nAChR subunits to search for.",
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="Search for all 16 nAChR subunits.",
    )
    group.add_argument(
        "--resume",
        action="store_true",
        help="Resume a previously interrupted scraping run.",
    )
    group.add_argument(
        "--reset",
        action="store_true",
        help="Clear all checkpoints and start fresh.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only search PubMed and show paper counts. Do not fetch or process.",
    )

    return parser.parse_args()


def validate_config():
    """Check that required configuration is set."""
    errors = []
    if not config.NCBI_EMAIL:
        errors.append("NCBI_EMAIL is not set. Add it to your .env file.")
    if not config.GEMINI_API_KEY:
        errors.append("GEMINI_API_KEY is not set. Add it to your .env file.")

    if errors:
        print("Configuration errors:")
        for e in errors:
            print(f"  - {e}")
        print("\nSee .env.example for required environment variables.")
        sys.exit(1)


def run_pipeline(subunits: list[str], dry_run: bool = False):
    """
    Main pipeline: search → fetch → extract → save.

    Args:
        subunits: List of subunit names to search for.
        dry_run: If True, only search and report counts.
    """
    checkpoint = CheckpointManager()
    fetcher = PaperFetcher()
    writer = ExcelWriter()

    # ── Step 1: Search PubMed ────────────────────────────────────────────
    print("=" * 60)
    print("STEP 1: Searching PubMed")
    print("=" * 60)

    papers = search_all_subunits(subunits, checkpoint=checkpoint)

    if not papers:
        print("\nNo papers found matching the search criteria.")
        return

    # Filter out already-processed papers
    unprocessed = {
        pmid: paper
        for pmid, paper in papers.items()
        if not checkpoint.is_processed(pmid)
    }

    print(f"\nTotal papers found: {len(papers)}")
    print(f"Already processed: {len(papers) - len(unprocessed)}")
    print(f"To process: {len(unprocessed)}")

    if dry_run:
        print("\n[DRY RUN] Stopping here. No papers will be fetched or processed.")
        print("\nPapers per subunit:")
        subunit_counts = {}
        for paper in papers.values():
            for sub in paper.associated_subunits:
                subunit_counts[sub] = subunit_counts.get(sub, 0) + 1
        for sub in sorted(subunit_counts.keys()):
            print(f"  {sub}: {subunit_counts[sub]}")
        return

    if not unprocessed:
        print("\nAll papers have already been processed. Use --reset to start fresh.")
        stats = checkpoint.get_stats()
        writer_stats = writer.get_stats()
        print(f"\nCheckpoint stats: {stats}")
        print(f"Excel stats: {writer_stats}")
        return

    # ── Step 2 & 3: Fetch full text and extract mutations ────────────────
    print("\n" + "=" * 60)
    print("STEP 2 & 3: Fetching papers and extracting data")
    print("=" * 60)

    # Set up Gemini
    if not dry_run:
        setup_gemini()

    processed_count = 0
    inaccessible_count = 0
    total_entries = 0
    error_count = 0

    for pmid, paper in tqdm(unprocessed.items(), desc="Processing papers"):
        try:
            # Fetch full text
            full_text, source = fetcher.fetch_full_text(
                pmid=paper.pmid,
                pmcid=paper.pmcid,
                doi=paper.doi,
            )

            if full_text is None:
                # Paper is inaccessible
                writer.append_inaccessible(
                    pmid=paper.pmid,
                    doi=paper.doi,
                    title=paper.title,
                    authors=paper.authors,
                    abstract=paper.abstract,
                    subunits=paper.associated_subunits,
                    reason=source,  # source contains the reason when text is None
                )
                checkpoint.mark_inaccessible(paper.pmid, source)
                inaccessible_count += 1
                continue

            # Extract mutations using Gemini
            print(f"\n  Processing PMID {pmid} ({source})...")
            mutations = extract_mutations_from_paper(full_text, pmid)

            if mutations:
                # Construct DOI URL if we have a DOI
                doi_url = f"https://doi.org/{paper.doi}" if paper.doi else ""
                writer.append_mutations(mutations, pmid, doi_url)
                print(f"    Extracted {len(mutations)} mutation(s)")
                total_entries += len(mutations)
            else:
                print(f"    No qualifying mutations found")

            checkpoint.mark_processed(pmid, len(mutations))
            processed_count += 1

        except KeyboardInterrupt:
            print("\n\nInterrupted by user. Progress has been saved.")
            print("Run with --resume to continue from where you left off.")
            break

        except Exception as e:
            print(f"\n  Error processing PMID {pmid}: {e}")
            checkpoint.mark_error(pmid, str(e))
            error_count += 1

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Papers processed:     {processed_count}")
    print(f"Papers inaccessible:  {inaccessible_count}")
    print(f"Papers with errors:   {error_count}")
    print(f"Total entries found:  {total_entries}")
    print(f"\nOutput files:")
    print(f"  Data:          {config.MAIN_EXCEL}")
    print(f"  Inaccessible:  {config.INACCESSIBLE_EXCEL}")
    print(f"  Checkpoint:    {config.CHECKPOINT_FILE}")

    writer_stats = writer.get_stats()
    print(f"\nTotal entries in Excel: {writer_stats['main_entries']}")
    print(f"Total inaccessible:     {writer_stats['inaccessible_papers']}")


def main():
    args = parse_args()

    # Handle reset
    if args.reset:
        checkpoint = CheckpointManager()
        checkpoint.reset()
        print("Checkpoints cleared. You can now start a fresh run.")
        return

    # Handle resume
    if args.resume:
        checkpoint = CheckpointManager()
        # Load cached search results to determine which subunits were searched
        cached_subunits = list(checkpoint.data.get("search_results", {}).keys())
        if not cached_subunits:
            print("No previous run found to resume. Start a new run with --subunits or --all.")
            sys.exit(1)
        print(f"Resuming previous run for subunits: {', '.join(cached_subunits)}")
        validate_config()
        run_pipeline(cached_subunits)
        return

    # Normal run
    subunits = config.ALL_SUBUNITS if args.all else args.subunits

    if not args.dry_run:
        validate_config()
    else:
        # For dry run, only NCBI_EMAIL is needed
        if not config.NCBI_EMAIL:
            print("NCBI_EMAIL is not set. Add it to your .env file.")
            sys.exit(1)

    print(f"nAChR PubMed Data Scraper")
    print(f"Subunits: {', '.join(subunits)}")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'FULL PROCESSING'}")
    print()

    run_pipeline(subunits, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
