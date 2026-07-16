"""
UniProt source: human nAChR mutagenesis / variant annotations.

For each human subunit gene (CHRNA1..CHRNG, taxid 9606) UniProt curates
"Mutagenesis" features — a residue change, a free-text functional effect, and
the PubMed IDs it was reported in. Those PMIDs are high-value candidate papers
(literature-curated mutation studies, usually electrophysiology), and the
curated mutation + effect can pre-fill part of the manual extraction.

Returns {pmid -> {"subunits": [...], "mutations": [...], "effects": [...]}}.
"""

import time

import requests

import config


def _get_json(url: str, params: dict) -> dict:
    """GET with a UA header and basic retry/backoff on 5xx / network errors."""
    last = None
    for attempt in range(3):
        try:
            r = requests.get(url, params=params, headers=config.HTTP_HEADERS,
                             timeout=config.HTTP_TIMEOUT)
            if r.status_code >= 500:
                last = requests.HTTPError(f"{r.status_code} server error", response=r)
                time.sleep(2 * (attempt + 1))
                continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException as ex:
            last = ex
            time.sleep(2 * (attempt + 1))
    raise last


_THREE_TO_ONE = {
    "Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C", "Gln": "Q",
    "Glu": "E", "Gly": "G", "His": "H", "Ile": "I", "Leu": "L", "Lys": "K",
    "Met": "M", "Phe": "F", "Pro": "P", "Ser": "S", "Thr": "T", "Trp": "W",
    "Tyr": "Y", "Val": "V",
}


def _gene_symbol(subunit: str) -> str:
    # Human gene symbols are uppercase (CHRNA7); UniProt search is case-insensitive.
    return subunit.upper()


def _aa1(seq_str: str) -> str:
    """Normalize a UniProt sequence fragment to single-letter code(s)."""
    s = (seq_str or "").strip()
    if len(s) == 3 and s.capitalize() in _THREE_TO_ONE:
        return _THREE_TO_ONE[s.capitalize()]
    return s


def _mutation_string(feat: dict) -> str:
    start = feat.get("location", {}).get("start", {}).get("value")
    if start is None:
        return ""
    alt = feat.get("alternativeSequence", {}) or {}
    orig = _aa1(alt.get("originalSequence", ""))
    alts = alt.get("alternativeSequences", []) or []
    new = _aa1(alts[0]) if alts else ""
    return f"{orig}{start}{new}".strip()


def search_subunit(subunit: str) -> dict:
    params = {
        "query": f"(gene:{_gene_symbol(subunit)}) AND (organism_id:{config.UNIPROT_TAXID})",
        "fields": "accession,gene_names,protein_name,ft_mutagen,ft_variant",
        "format": "json",
        "size": 50,
    }
    results = _get_json(config.UNIPROT_SEARCH_URL, params).get("results", [])

    by_pmid = {}
    for entry in results:
        for feat in entry.get("features", []):
            if feat.get("type") not in config.UNIPROT_FEATURE_TYPES:
                continue
            mutation = _mutation_string(feat)
            desc = (feat.get("description", "") or "").strip()
            for ev in feat.get("evidences", []) or []:
                if str(ev.get("source", "")).lower() != "pubmed":
                    continue
                pmid = str(ev.get("id", "") or "")
                if not pmid:
                    continue
                rec = by_pmid.setdefault(
                    pmid, {"subunits": set(), "mutations": [], "effects": []}
                )
                rec["subunits"].add(subunit)
                if mutation and mutation not in rec["mutations"]:
                    rec["mutations"].append(mutation)
                if desc and desc not in rec["effects"]:
                    rec["effects"].append(desc)
    return by_pmid


def search_all(subunits: list) -> dict:
    merged = {}
    for subunit in subunits:
        print(f"\nSearching UniProt for {subunit} (human)...")
        try:
            sub = search_subunit(subunit)
        except requests.RequestException as e:
            print(f"  UniProt error for {subunit}: {str(e)[:80]}")
            print("  UniProt appears unavailable; skipping it for this run "
                  "(re-run later to add it).")
            break
        print(f"  Found {len(sub)} cited PMIDs for {subunit}")
        for pmid, rec in sub.items():
            m = merged.setdefault(
                pmid, {"subunits": set(), "mutations": [], "effects": []}
            )
            m["subunits"] |= rec["subunits"]
            for x in rec["mutations"]:
                if x not in m["mutations"]:
                    m["mutations"].append(x)
            for x in rec["effects"]:
                if x not in m["effects"]:
                    m["effects"].append(x)
    for rec in merged.values():
        rec["subunits"] = sorted(rec["subunits"])
    return merged
