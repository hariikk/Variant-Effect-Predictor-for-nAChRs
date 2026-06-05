"""
PubMed search module using NCBI E-utilities (Entrez) via Biopython.
Searches for nAChR mutation papers with electrophysiology data in mouse.
"""

import time
from dataclasses import dataclass, field

from Bio import Entrez
from tqdm import tqdm

import config


@dataclass
class PaperMetadata:
    """Metadata for a single PubMed paper."""
    pmid: str
    title: str = ""
    authors: str = ""
    abstract: str = ""
    doi: str = ""
    pmcid: str = ""
    associated_subunits: list[str] = field(default_factory=list)


def _build_search_query(subunit: str) -> str:
    """
    Build a high-recall PubMed search query for a given nAChR subunit.

    4-clause AND query:
        (gene: symbols + MeSH + text aliases)
        AND (species: MeSH + text terms + Organism tag)
        AND (electrophysiology: MeSH + text terms)
        AND (mutation: MeSH + text terms)

    Each clause uses both MeSH terms (broad NLM indexing) and free-text
    (Title/Abstract) for maximum recall.
    """
    gene = subunit.upper()
    gene_info = config.GENE_ALIASES.get(gene)

    # ── Gene identity clause ────────────────────────────────────────
    if gene_info is None:
        gene_clause = f'{gene}[Title/Abstract]'
    else:
        parts = []
        for sym in gene_info["gene_symbols"]:
            parts.append(f'{sym}[Title/Abstract]')
        for mesh in gene_info["mesh_terms"]:
            parts.append(f'"{mesh}"[MeSH Terms]')
        for alias in gene_info["text_aliases"]:
            parts.append(f'"{alias}"[Title/Abstract]')
        gene_clause = " OR ".join(parts)

    # ── Species clause ──────────────────────────────────────────────
    species_parts = []
    for mesh in config.SPECIES_MESH_TERMS:
        species_parts.append(f'"{mesh}"[MeSH Terms]')
    for term in config.SPECIES_TEXT_TERMS:
        species_parts.append(f'"{term}"[Title/Abstract]')
    species_parts.append('"Mus musculus"[Organism]')
    species_clause = " OR ".join(species_parts)

    # ── Electrophysiology clause ────────────────────────────────────
    ephys_parts = []
    for mesh in config.EPHYS_MESH_TERMS:
        ephys_parts.append(f'"{mesh}"[MeSH Terms]')
    for term in config.EPHYS_TEXT_TERMS:
        ephys_parts.append(f'"{term}"[Title/Abstract]')
    ephys_clause = " OR ".join(ephys_parts)

    # ── Mutation / mutagenesis clause ───────────────────────────────
    mut_parts = []
    for mesh in config.MUTATION_MESH_TERMS:
        mut_parts.append(f'"{mesh}"[MeSH Terms]')
    for term in config.MUTATION_TEXT_TERMS:
        mut_parts.append(f'"{term}"[Title/Abstract]')
    mut_clause = " OR ".join(mut_parts)

    query = (
        f"({gene_clause}) AND ({species_clause}) "
        f"AND ({ephys_clause}) AND ({mut_clause})"
    )
    return query


def search_pubmed(subunit: str) -> list[str]:
    """
    Search PubMed for papers about a specific nAChR subunit.
    Returns a list of PMIDs.
    """
    Entrez.email = config.NCBI_EMAIL
    if config.NCBI_API_KEY:
        Entrez.api_key = config.NCBI_API_KEY

    query = _build_search_query(subunit)
    print(f"  Search query: {query[:120]}...")

    # First, get the count
    handle = Entrez.esearch(db="pubmed", term=query, retmax=0)
    result = Entrez.read(handle)
    handle.close()
    total_count = int(result["Count"])
    print(f"  Found {total_count} papers for {subunit}")

    if total_count == 0:
        return []

    # Fetch all PMIDs (in batches if needed)
    all_pmids = []
    batch_size = 500
    for start in range(0, total_count, batch_size):
        handle = Entrez.esearch(
            db="pubmed", term=query, retmax=batch_size, retstart=start
        )
        result = Entrez.read(handle)
        handle.close()
        all_pmids.extend(result["IdList"])
        time.sleep(1.0 / config.NCBI_REQUESTS_PER_SECOND)

    return all_pmids


def fetch_paper_metadata(pmids: list[str]) -> dict[str, PaperMetadata]:
    """
    Fetch metadata (title, abstract, DOI, PMCID) for a list of PMIDs.
    Returns a dict mapping PMID -> PaperMetadata.
    """
    Entrez.email = config.NCBI_EMAIL
    if config.NCBI_API_KEY:
        Entrez.api_key = config.NCBI_API_KEY

    papers = {}
    batch_size = 200

    for i in tqdm(range(0, len(pmids), batch_size), desc="Fetching metadata"):
        batch = pmids[i : i + batch_size]
        ids_str = ",".join(batch)

        # Fetch detailed records
        handle = Entrez.efetch(
            db="pubmed", id=ids_str, rettype="xml", retmode="xml"
        )
        records = Entrez.read(handle)
        handle.close()

        for article in records.get("PubmedArticle", []):
            try:
                medline = article["MedlineCitation"]
                pmid = str(medline["PMID"])
                art = medline["Article"]

                # Title
                title = str(art.get("ArticleTitle", ""))

                # Authors
                author_list = art.get("AuthorList", [])
                authors = "; ".join(
                    f"{a.get('LastName', '')} {a.get('Initials', '')}"
                    for a in author_list
                    if "LastName" in a
                )

                # Abstract
                abstract_parts = art.get("Abstract", {}).get("AbstractText", [])
                abstract = " ".join(str(part) for part in abstract_parts)

                # DOI
                doi = ""
                for id_obj in art.get("ELocationID", []):
                    if str(id_obj.attributes.get("EIdType", "")) == "doi":
                        doi = str(id_obj)
                        break

                # Also check ArticleIdList in PubmedData
                if not doi:
                    pubmed_data = article.get("PubmedData", {})
                    for id_obj in pubmed_data.get("ArticleIdList", []):
                        if str(id_obj.attributes.get("IdType", "")) == "doi":
                            doi = str(id_obj)
                            break

                # PMCID
                pmcid = ""
                pubmed_data = article.get("PubmedData", {})
                for id_obj in pubmed_data.get("ArticleIdList", []):
                    if str(id_obj.attributes.get("IdType", "")) == "pmc":
                        pmcid = str(id_obj)
                        break

                papers[pmid] = PaperMetadata(
                    pmid=pmid,
                    title=title,
                    authors=authors,
                    abstract=abstract,
                    doi=doi,
                    pmcid=pmcid,
                )
            except (KeyError, IndexError, AttributeError) as e:
                print(f"  Warning: Could not parse metadata for an article: {e}")
                continue

        time.sleep(1.0 / config.NCBI_REQUESTS_PER_SECOND)

    return papers


def search_all_subunits(subunits: list[str], checkpoint=None) -> dict[str, PaperMetadata]:
    """
    Search PubMed for all specified subunits, deduplicate PMIDs,
    and fetch metadata for all unique papers.

    Args:
        subunits: List of subunit names to search.
        checkpoint: Optional CheckpointManager for caching search results.

    Returns:
        Dict mapping PMID -> PaperMetadata (with associated_subunits populated).
    """
    # Collect PMIDs per subunit
    pmid_to_subunits: dict[str, list[str]] = {}

    for subunit in subunits:
        print(f"\nSearching PubMed for {subunit}...")

        # Check cache
        cached = checkpoint.get_cached_search(subunit) if checkpoint else None
        if cached is not None:
            print(f"  Using cached results: {len(cached)} papers")
            pmids = cached
        else:
            pmids = search_pubmed(subunit)
            if checkpoint:
                checkpoint.save_search_results(subunit, pmids)

        for pmid in pmids:
            if pmid not in pmid_to_subunits:
                pmid_to_subunits[pmid] = []
            pmid_to_subunits[pmid].append(subunit)

    unique_pmids = list(pmid_to_subunits.keys())
    print(f"\nTotal unique papers across all subunits: {len(unique_pmids)}")

    # Fetch metadata
    if not unique_pmids:
        return {}

    print("Fetching paper metadata...")
    papers = fetch_paper_metadata(unique_pmids)

    # Attach subunit associations
    for pmid, paper in papers.items():
        paper.associated_subunits = pmid_to_subunits.get(pmid, [])

    return papers
