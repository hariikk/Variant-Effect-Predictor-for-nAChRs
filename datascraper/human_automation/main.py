"""
Build a ranked worklist of candidate HUMAN nAChR mutation-electrophysiology
papers for manual curation, aggregated across three sources:

  * PubMed     — title/abstract + MeSH search (NCBI Entrez)
  * Europe PMC — full-text + preprint search
  * UniProt    — curated human mutagenesis/variant annotations (mutation + effect
                 + the PMIDs they were reported in)

Pipeline:  search all sources  ->  merge/dedup by PMID/DOI  ->  enrich UniProt
PMIDs with PubMed metadata  ->  keyword score (UniProt-cited papers boosted)  ->
write a two-sheet Excel (Worklist / Rejected). No LLM, no API key beyond your
NCBI email, no quota wall.

Usage:
    python main.py --all                       # all 16 subunits, all 3 sources
    python main.py --subunits CHRNA7 CHRNB2    # specific subunits
    python main.py --all --min-score 4         # tighter worklist
    python main.py --all --sources pubmed uniprot   # pick sources
    python main.py --all --refresh             # ignore caches, re-pull everything
"""

import argparse
import json
import sys

import config
import europepmc_search
import uniprot_search
from checkpoint import CheckpointManager
from pubmed_search import PaperMetadata, fetch_paper_metadata, search_all_subunits
from scoring import score_paper
from worklist_writer import write_worklist

ALL_SOURCES = ["pubmed", "europepmc", "uniprot"]


# ── metadata cache (so re-scoring / re-thresholding is instant) ──────────────

def _meta_to_dict(p: PaperMetadata) -> dict:
    return {
        "pmid": p.pmid, "title": p.title, "authors": p.authors,
        "abstract": p.abstract, "doi": p.doi, "pmcid": p.pmcid,
        "year": p.year, "journal": p.journal,
        "associated_subunits": p.associated_subunits, "sources": p.sources,
        "curated_mutations": p.curated_mutations,
        "curated_effects": p.curated_effects,
    }


def _dict_to_meta(d: dict) -> PaperMetadata:
    return PaperMetadata(
        pmid=d.get("pmid", ""), title=d.get("title", ""), authors=d.get("authors", ""),
        abstract=d.get("abstract", ""), doi=d.get("doi", ""), pmcid=d.get("pmcid", ""),
        year=d.get("year", ""), journal=d.get("journal", ""),
        associated_subunits=d.get("associated_subunits", []),
        sources=d.get("sources", []),
        curated_mutations=d.get("curated_mutations", []),
        curated_effects=d.get("curated_effects", []),
    )


def _save_cache(subunits, sources, papers):
    data = {
        "version": config.CACHE_VERSION,
        "subunits": sorted(subunits),
        "sources": sorted(sources),
        "papers": [_meta_to_dict(p) for p in papers],
    }
    with open(config.METADATA_CACHE, "w", encoding="utf-8") as f:
        json.dump(data, f)


def _load_cache(subunits, sources):
    if not config.METADATA_CACHE.exists():
        return None
    with open(config.METADATA_CACHE, "r", encoding="utf-8") as f:
        data = json.load(f)
    if (data.get("version") != config.CACHE_VERSION
            or data.get("subunits") != sorted(subunits)
            or data.get("sources") != sorted(sources)):
        return None
    return [_dict_to_meta(d) for d in data["papers"]]


# ── merge across sources ─────────────────────────────────────────────────────

def _key(m: PaperMetadata) -> str:
    if m.pmid:
        return f"pmid:{m.pmid}"
    if m.doi:
        return f"doi:{m.doi.lower()}"
    return f"ti:{(m.title or '').strip().lower()}"


def _merge_into(papers: dict, m: PaperMetadata):
    k = _key(m)
    existing = papers.get(k)
    if existing is None:
        papers[k] = m
        return
    for s in m.sources:
        if s not in existing.sources:
            existing.sources.append(s)
    for su in m.associated_subunits:
        if su not in existing.associated_subunits:
            existing.associated_subunits.append(su)
    # Backfill any field the first record was missing.
    for attr in ("title", "abstract", "authors", "year", "journal", "doi", "pmcid"):
        if not getattr(existing, attr) and getattr(m, attr):
            setattr(existing, attr, getattr(m, attr))


def collect(subunits, sources):
    """Run the selected sources and return a merged list of PaperMetadata."""
    papers = {}

    if "pubmed" in sources:
        print("\n" + "=" * 60 + "\nSOURCE: PubMed\n" + "=" * 60)
        for m in search_all_subunits(subunits, checkpoint=CheckpointManager()).values():
            _merge_into(papers, m)

    if "europepmc" in sources:
        print("\n" + "=" * 60 + "\nSOURCE: Europe PMC\n" + "=" * 60)
        for m in europepmc_search.search_all(subunits):
            _merge_into(papers, m)

    if "uniprot" in sources:
        print("\n" + "=" * 60 + "\nSOURCE: UniProt (human mutagenesis)\n" + "=" * 60)
        uni = uniprot_search.search_all(subunits)
        have = {m.pmid for m in papers.values() if m.pmid}
        missing = sorted(p for p in uni if p not in have)
        if missing:
            print(f"\nFetching PubMed metadata for {len(missing)} UniProt-only PMIDs...")
            for m in fetch_paper_metadata(missing).values():
                m.sources = ["UniProt"]
                _merge_into(papers, m)
        # Attach curated mutations/effects and ensure UniProt tagging.
        index = {m.pmid: m for m in papers.values() if m.pmid}
        for pmid, info in uni.items():
            m = index.get(pmid)
            if m is None:  # metadata fetch failed — keep a minimal record
                m = PaperMetadata(pmid=pmid, sources=["UniProt"])
                _merge_into(papers, m)
                index[pmid] = m
            if "UniProt" not in m.sources:
                m.sources.append("UniProt")
            m.curated_mutations = info["mutations"]
            m.curated_effects = info["effects"]
            for su in info["subunits"]:
                if su not in m.associated_subunits:
                    m.associated_subunits.append(su)

    return list(papers.values())


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a ranked, multi-source worklist of mouse nAChR mutation-ephys papers.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--subunits", nargs="+", choices=config.ALL_SUBUNITS,
                       help="Specific nAChR subunits to search.")
    group.add_argument("--all", action="store_true",
                       help="Search all 16 nAChR subunits.")
    parser.add_argument("--sources", nargs="+", choices=ALL_SOURCES, default=ALL_SOURCES,
                        help="Which sources to query (default: all three).")
    parser.add_argument("--min-score", type=int, default=config.DEFAULT_MIN_SCORE,
                        help=f"Worklist cutoff (default {config.DEFAULT_MIN_SCORE}).")
    parser.add_argument("--refresh", action="store_true",
                        help="Ignore the metadata cache; re-pull from all sources.")
    return parser.parse_args()


def main():
    args = parse_args()
    subunits = config.ALL_SUBUNITS if args.all else args.subunits
    sources = [s for s in ALL_SOURCES if s in args.sources]

    if "pubmed" in sources or "uniprot" in sources:
        if not config.NCBI_EMAIL:
            print("NCBI_EMAIL is not set. Copy .env.example to .env and add your email.")
            sys.exit(1)

    print("nAChR HUMAN-paper worklist builder")
    print(f"Subunits:  {', '.join(subunits)}")
    print(f"Sources:   {', '.join(sources)}")
    print(f"Min score: {args.min_score}")

    papers = None if args.refresh else _load_cache(subunits, sources)
    if papers is not None:
        print(f"\nLoaded {len(papers)} papers from metadata cache "
              f"(use --refresh to re-pull).")
    else:
        papers = collect(subunits, sources)
        if not papers:
            print("\nNo papers found.")
            return
        _save_cache(subunits, sources, papers)

    # ── Score ────────────────────────────────────────────────────────────
    print(f"\nScoring {len(papers)} unique papers (keyword/regex relevance)...")
    scored = [
        (p, score_paper(p.title, p.abstract, uniprot_curated=("UniProt" in p.sources)))
        for p in papers
    ]

    n_keep, n_drop = write_worklist(scored, args.min_score)

    # ── Summary ──────────────────────────────────────────────────────────
    src_counts, dist = {}, {}
    for p, r in scored:
        for s in p.sources:
            src_counts[s] = src_counts.get(s, 0) + 1
        dist[r["score"]] = dist.get(r["score"], 0) + 1

    print("\n" + "=" * 60 + "\nSUMMARY\n" + "=" * 60)
    print(f"Unique papers:        {len(scored)}")
    print("  by source:          " +
          ", ".join(f"{s}={n}" for s, n in sorted(src_counts.items())))
    print(f"Worklist (>= {args.min_score}):       {n_keep}")
    print(f"Rejected  (< {args.min_score}):       {n_drop}")
    print("\nScore distribution (score: count):")
    for s in sorted(dist, reverse=True):
        print(f"  {s:>3}: {dist[s]:>5}  {'#' * min(dist[s] // 20 + 1, 50)}")
    print(f"\nWorklist written to: {config.WORKLIST_EXCEL}")
    print("Open the 'Worklist' sheet, scan top-down, click DOI Link to read.")


if __name__ == "__main__":
    main()
