"""
Europe PMC source: full-text + preprint search to complement PubMed.

Europe PMC indexes PubMed/MEDLINE plus preprints, patents and (for open-access
papers) full text, so the same 4-clause query often surfaces human nAChR
mutation-electrophysiology papers that PubMed's title/abstract + MeSH search
misses. Returns PaperMetadata objects tagged source "EuropePMC".
"""

import time

import requests

import config
from pubmed_search import PaperMetadata


def _phrase_or(terms) -> str:
    return " OR ".join(f'"{t}"' for t in terms)


def _build_query(subunit: str) -> str:
    gene = subunit.upper()
    gi = config.GENE_ALIASES.get(gene)
    gene_terms = (gi["gene_symbols"] + gi["mesh_terms"] + gi["text_aliases"]
                  if gi else [gene])
    # Require the gene/subunit term in the TITLE or ABSTRACT so the paper is
    # actually ABOUT this subunit, not merely citing it somewhere in full text
    # (that full-text matching pulled in off-topic papers like AE1 / 5-HT3). The
    # species / ephys / mutation clauses stay full-text for recall.
    gene_clause = " OR ".join(
        f'(TITLE:"{t}" OR ABSTRACT:"{t}")' for t in gene_terms
    )
    return (
        f"({gene_clause}) "
        f"AND ({_phrase_or(config.SPECIES_TEXT_TERMS + [config.ORGANISM_NAME])}) "
        f"AND ({_phrase_or(config.EPHYS_TEXT_TERMS)}) "
        f"AND ({_phrase_or(config.MUTATION_TEXT_TERMS)})"
    )


def _to_meta(rec: dict, subunit: str):
    pmid = str(rec.get("pmid", "") or "")
    doi = rec.get("doi", "") or ""
    title = rec.get("title", "") or ""
    if not (pmid or doi or title):
        return None
    journal = rec.get("journalTitle", "") or (
        rec.get("journalInfo", {}).get("journal", {}).get("title", "")
    )
    return PaperMetadata(
        pmid=pmid,
        title=title,
        authors=rec.get("authorString", "") or "",
        abstract=rec.get("abstractText", "") or "",
        doi=doi,
        pmcid=rec.get("pmcid", "") or "",
        year=str(rec.get("pubYear", "") or ""),
        journal=journal,
        associated_subunits=[subunit],
        sources=["EuropePMC"],
    )


def search_subunit(subunit: str) -> list:
    query = _build_query(subunit)
    out, cursor, fetched = [], "*", 0
    while True:
        params = {
            "query": query,
            "format": "json",
            "pageSize": config.EUROPEPMC_PAGE_SIZE,
            "resultType": "core",
            "cursorMark": cursor,
        }
        resp = requests.get(config.EUROPEPMC_SEARCH_URL, params=params,
                            headers=config.HTTP_HEADERS, timeout=config.HTTP_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        results = data.get("resultList", {}).get("result", [])
        for rec in results:
            meta = _to_meta(rec, subunit)
            if meta:
                out.append(meta)
        fetched += len(results)
        nxt = data.get("nextCursorMark")
        if (not results or not nxt or nxt == cursor
                or fetched >= config.EUROPEPMC_MAX_PER_SUBUNIT):
            break
        cursor = nxt
        time.sleep(0.34)
    return out


def search_all(subunits: list) -> list:
    all_records = []
    for subunit in subunits:
        print(f"\nSearching Europe PMC for {subunit}...")
        try:
            recs = search_subunit(subunit)
        except requests.RequestException as e:
            print(f"  Europe PMC error for {subunit}: {e}")
            recs = []
        print(f"  Found {len(recs)} records for {subunit}")
        all_records.extend(recs)
    return all_records
