"""
Configuration for the nAChR PubMed Data Scraper.
All settings, constants, and API configurations are centralized here.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(Path(__file__).parent / ".env")

# ── API Keys & Credentials ──────────────────────────────────────────────────
# Only NCBI is needed — relevance ranking is a local keyword scorer (no LLM, no
# API key, no quota). NCBI_API_KEY is optional and only raises the rate limit.
NCBI_EMAIL = os.getenv("NCBI_EMAIL", "")  # Required by NCBI policy
NCBI_API_KEY = os.getenv("NCBI_API_KEY", "")  # Optional, increases rate limit

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
REPO_DIR = BASE_DIR.parent
OUTPUT_DIR = BASE_DIR / "output"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
WORKLIST_EXCEL = OUTPUT_DIR / "nachr_mouse_worklist.xlsx"
METADATA_CACHE = OUTPUT_DIR / "metadata_cache.json"
CHECKPOINT_FILE = CHECKPOINT_DIR / "checkpoint.json"

# Existing files cross-referenced to flag papers already mined / already listed.
MANUAL_DB = REPO_DIR / "nachr_db_manual.xlsx"                      # extraction target (human DB)
PRIOR_WORKLIST = REPO_DIR / "Data_Scraper" / "mouse_scraped.xlsx"  # earlier 270-paper list

# Ensure output directories exist
OUTPUT_DIR.mkdir(exist_ok=True)
CHECKPOINT_DIR.mkdir(exist_ok=True)

# ── nAChR Subunits ───────────────────────────────────────────────────────────
ALL_SUBUNITS = [
    "CHRNA1", "CHRNA2", "CHRNA3", "CHRNA4", "CHRNA5",
    "CHRNA6", "CHRNA7", "CHRNA9", "CHRNA10",
    "CHRNB1", "CHRNB2", "CHRNB3", "CHRNB4",
    "CHRND", "CHRNE", "CHRNG",
]

# ── PubMed Search Configuration ─────────────────────────────────────────────

# MeSH terms for electrophysiology (NLM indexing catches papers broadly)
EPHYS_MESH_TERMS = [
    "Electrophysiology",
    "Patch-Clamp Techniques",
    "Electrophysiological Phenomena",
    "Ion Channel Gating",
    "Membrane Potentials",
]

# Free-text electrophysiology terms (both hyphenated AND non-hyphenated variants)
EPHYS_TEXT_TERMS = [
    "electrophysiology", "electrophysiological",
    "patch-clamp", "patch clamp",
    "voltage-clamp", "voltage clamp",
    "two-electrode voltage clamp", "two electrode voltage clamp", "TEVC",
    "whole-cell recording", "whole cell recording",
    "whole-cell current", "whole cell current",
    "single-channel recording", "single channel recording",
    "ion channel recording",
    "current amplitude", "macroscopic current", "macroscopic currents",
    "agonist-evoked current", "agonist evoked current",
    "ACh-evoked current", "nicotine-evoked current",
    "peak current", "current-voltage", "current voltage", "I-V curve",
    "dose-response", "dose response",
    "concentration-response", "concentration response",
    "oocyte expression", "oocyte recording", "Xenopus oocyte",
    "ion flux", "calcium flux", "calcium imaging", "Ca2+ imaging",
    "fluorometric imaging", "FLIPR", "thallium flux",
    "86Rb efflux", "rubidium efflux",
    "desensitization", "deactivation", "channel kinetics",
    "open probability", "EC50", "IC50",
]

# MeSH terms for mouse species
SPECIES_MESH_TERMS = [
    "Mice",
    "Mice, Inbred C57BL",
    "Mice, Knockout",
    "Mice, Transgenic",
]

# Free-text species terms
SPECIES_TEXT_TERMS = [
    "mouse", "mice", "murine", "Mus musculus",
    "C57BL", "BALB/c", "ICR mouse", "Swiss mouse",
    "knock-out mouse", "knockout mouse", "transgenic mouse",
    "knock-in mouse", "knockin mouse",
]

# MeSH terms for mutations / mutagenesis
MUTATION_MESH_TERMS = [
    "Mutation",
    "Mutation, Missense",
    "Mutagenesis, Site-Directed",
    "Mutagenesis",
    "Amino Acid Substitution",
    "Loss of Function Mutation",
    "Gain of Function Mutation",
    "Sequence Deletion",
    "Frameshift Mutation",
    "Codon, Nonsense",
]

# Free-text mutation terms (broad — catches many ways papers describe mutations)
MUTATION_TEXT_TERMS = [
    "mutation", "mutations", "mutant", "mutants",
    "variant", "variants",
    "substitution", "missense", "point mutation",
    "mutagenesis", "site-directed mutagenesis",
    "alanine scanning", "alanine substitution",
    "amino acid substitution", "amino acid replacement",
    "gain-of-function", "gain of function",
    "loss-of-function", "loss of function",
    "loss-of-function mutation", "gain-of-function mutation",
    "wild-type", "wild type", "wildtype",
    "residue", "residues",
    "deletion", "frameshift", "truncation", "nonsense",
    "single nucleotide", "SNP",
    "engineered mutation", "engineered mutant",
    "chimeric receptor", "chimera",
]

# Per-subunit gene aliases: gene_symbols, mesh_terms, text_aliases
GENE_ALIASES = {
    "CHRNA1": {
        "gene_symbols": ["CHRNA1", "Chrna1", "ACHRA"],
        "mesh_terms": [],
        "text_aliases": [
            "alpha1 nicotinic", "alpha-1 nicotinic",
            "alpha1 nAChR", "alpha-1 nAChR",
            "nAChR alpha1", "nAChR alpha-1",
            "α1 nicotinic", "α1 nAChR",
            "nicotinic receptor alpha1 subunit",
            "nicotinic receptor alpha1",
            "muscle nicotinic receptor alpha",
            "muscle-type nicotinic", "muscle type nicotinic",
        ],
    },
    "CHRNA2": {
        "gene_symbols": ["CHRNA2", "Chrna2"],
        "mesh_terms": [],
        "text_aliases": [
            "alpha2 nicotinic", "alpha-2 nicotinic",
            "alpha2 nAChR", "alpha-2 nAChR",
            "nAChR alpha2", "nAChR alpha-2",
            "α2 nicotinic", "α2 nAChR",
            "nicotinic receptor alpha2 subunit",
            "nicotinic receptor alpha2",
        ],
    },
    "CHRNA3": {
        "gene_symbols": ["CHRNA3", "Chrna3"],
        "mesh_terms": [],
        "text_aliases": [
            "alpha3 nicotinic", "alpha-3 nicotinic",
            "alpha3 nAChR", "alpha-3 nAChR",
            "nAChR alpha3", "nAChR alpha-3",
            "α3 nicotinic", "α3 nAChR",
            "nicotinic receptor alpha3 subunit",
            "nicotinic receptor alpha3",
        ],
    },
    "CHRNA4": {
        "gene_symbols": ["CHRNA4", "Chrna4"],
        "mesh_terms": [],
        "text_aliases": [
            "alpha4 nicotinic", "alpha-4 nicotinic",
            "alpha4 nAChR", "alpha-4 nAChR",
            "nAChR alpha4", "nAChR alpha-4",
            "α4 nicotinic", "α4 nAChR",
            "nicotinic receptor alpha4 subunit",
            "nicotinic receptor alpha4",
            "alpha4beta2", "α4β2",
        ],
    },
    "CHRNA5": {
        "gene_symbols": ["CHRNA5", "Chrna5"],
        "mesh_terms": [],
        "text_aliases": [
            "alpha5 nicotinic", "alpha-5 nicotinic",
            "alpha5 nAChR", "alpha-5 nAChR",
            "nAChR alpha5", "nAChR alpha-5",
            "α5 nicotinic", "α5 nAChR",
            "nicotinic receptor alpha5 subunit",
            "nicotinic receptor alpha5",
        ],
    },
    "CHRNA6": {
        "gene_symbols": ["CHRNA6", "Chrna6"],
        "mesh_terms": [],
        "text_aliases": [
            "alpha6 nicotinic", "alpha-6 nicotinic",
            "alpha6 nAChR", "alpha-6 nAChR",
            "nAChR alpha6", "nAChR alpha-6",
            "α6 nicotinic", "α6 nAChR",
            "nicotinic receptor alpha6 subunit",
            "nicotinic receptor alpha6",
        ],
    },
    "CHRNA7": {
        "gene_symbols": ["CHRNA7", "Chrna7", "NACRA7"],
        "mesh_terms": ["alpha7 Nicotinic Acetylcholine Receptor"],
        "text_aliases": [
            "alpha7 nicotinic", "alpha-7 nicotinic",
            "alpha7 nAChR", "alpha-7 nAChR",
            "nAChR alpha7", "nAChR alpha-7",
            "α7 nicotinic", "α7 nAChR",
            "alpha-7 nicotinic acetylcholine receptor",
            "nicotinic receptor alpha7 subunit",
            "nicotinic receptor alpha7",
            "homomeric nicotinic receptor",
        ],
    },
    "CHRNA9": {
        "gene_symbols": ["CHRNA9", "Chrna9"],
        "mesh_terms": [],
        "text_aliases": [
            "alpha9 nicotinic", "alpha-9 nicotinic",
            "alpha9 nAChR", "alpha-9 nAChR",
            "nAChR alpha9", "nAChR alpha-9",
            "α9 nicotinic", "α9 nAChR",
            "nicotinic receptor alpha9 subunit",
            "nicotinic receptor alpha9",
        ],
    },
    "CHRNA10": {
        "gene_symbols": ["CHRNA10", "Chrna10"],
        "mesh_terms": [],
        "text_aliases": [
            "alpha10 nicotinic", "alpha-10 nicotinic",
            "alpha10 nAChR", "alpha-10 nAChR",
            "nAChR alpha10", "nAChR alpha-10",
            "α10 nicotinic", "α10 nAChR",
            "nicotinic receptor alpha10 subunit",
            "nicotinic receptor alpha10",
            "alpha9alpha10", "α9α10",
        ],
    },
    "CHRNB1": {
        "gene_symbols": ["CHRNB1", "Chrnb1"],
        "mesh_terms": [],
        "text_aliases": [
            "beta1 nicotinic", "beta-1 nicotinic",
            "beta1 nAChR", "beta-1 nAChR",
            "nAChR beta1", "nAChR beta-1",
            "β1 nicotinic", "β1 nAChR",
            "nicotinic receptor beta1 subunit",
            "nicotinic receptor beta1",
            "muscle nicotinic receptor beta",
        ],
    },
    "CHRNB2": {
        "gene_symbols": ["CHRNB2", "Chrnb2"],
        "mesh_terms": [],
        "text_aliases": [
            "beta2 nicotinic", "beta-2 nicotinic",
            "beta2 nAChR", "beta-2 nAChR",
            "nAChR beta2", "nAChR beta-2",
            "β2 nicotinic", "β2 nAChR",
            "nicotinic receptor beta2 subunit",
            "nicotinic receptor beta2",
            "alpha4beta2", "α4β2",
        ],
    },
    "CHRNB3": {
        "gene_symbols": ["CHRNB3", "Chrnb3"],
        "mesh_terms": [],
        "text_aliases": [
            "beta3 nicotinic", "beta-3 nicotinic",
            "beta3 nAChR", "beta-3 nAChR",
            "nAChR beta3", "nAChR beta-3",
            "β3 nicotinic", "β3 nAChR",
            "nicotinic receptor beta3 subunit",
            "nicotinic receptor beta3",
        ],
    },
    "CHRNB4": {
        "gene_symbols": ["CHRNB4", "Chrnb4"],
        "mesh_terms": [],
        "text_aliases": [
            "beta4 nicotinic", "beta-4 nicotinic",
            "beta4 nAChR", "beta-4 nAChR",
            "nAChR beta4", "nAChR beta-4",
            "β4 nicotinic", "β4 nAChR",
            "nicotinic receptor beta4 subunit",
            "nicotinic receptor beta4",
        ],
    },
    "CHRND": {
        "gene_symbols": ["CHRND", "Chrnd"],
        "mesh_terms": [],
        "text_aliases": [
            "delta nicotinic", "nAChR delta",
            "δ nicotinic", "δ nAChR",
            "nicotinic receptor delta subunit",
            "nicotinic receptor delta",
            "muscle nicotinic receptor delta",
        ],
    },
    "CHRNE": {
        "gene_symbols": ["CHRNE", "Chrne"],
        "mesh_terms": [],
        "text_aliases": [
            "epsilon nicotinic", "nAChR epsilon",
            "ε nicotinic", "ε nAChR",
            "nicotinic receptor epsilon subunit",
            "nicotinic receptor epsilon",
            "muscle nicotinic receptor epsilon",
        ],
    },
    "CHRNG": {
        "gene_symbols": ["CHRNG", "Chrng"],
        "mesh_terms": [],
        "text_aliases": [
            "gamma nicotinic", "nAChR gamma",
            "γ nicotinic", "γ nAChR",
            "nicotinic receptor gamma subunit",
            "nicotinic receptor gamma",
            "fetal nicotinic receptor gamma",
        ],
    },
}

# ── Rate Limiting ────────────────────────────────────────────────────────────
NCBI_REQUESTS_PER_SECOND = 3  # 10 with API key

# ── Additional sources: Europe PMC + UniProt ─────────────────────────────────
EUROPEPMC_SEARCH_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
EUROPEPMC_PAGE_SIZE = 1000           # max page size Europe PMC allows
EUROPEPMC_MAX_PER_SUBUNIT = 5000     # safety cap on full-text hits per subunit

UNIPROT_SEARCH_URL = "https://rest.uniprot.org/uniprotkb/search"
MOUSE_TAXID = 10090                  # Mus musculus
UNIPROT_FEATURE_TYPES = ["Mutagenesis", "Natural variant"]

HTTP_TIMEOUT = 30                    # seconds, for Europe PMC / UniProt requests
HTTP_HEADERS = {"User-Agent": f"nachr-worklist/1.0 (mailto:{NCBI_EMAIL})"}

# Bump to invalidate an old metadata cache after a schema change.
CACHE_VERSION = 2

# ── Relevance Scoring (free keyword/regex ranker — see scoring.py) ───────────
# Points awarded per signal group found in a paper's title + abstract. Tuned so
# the default cutoff keeps papers with one strong signal (an explicit mutation
# like "S248F") OR at least two medium signals (e.g. an effect phrase plus an
# electrophysiology method).
SCORE_WEIGHTS = {
    "mutation_notation": 3,   # per distinct hit, capped by MUTATION_NOTATION_CAP
    "effect": 2,              # LOF/GOF/no-effect wording
    "ephys": 2,               # patch/voltage clamp, TEVC, oocyte, flux ...
    "mutagenesis": 1,         # "mutant", "site-directed mutagenesis" ...
    "mouse": 1,               # mouse / murine / C57BL / knock-in ...
    "uniprot_curated": 3,     # paper is cited by a UniProt mutagenesis annotation
}
MUTATION_NOTATION_CAP = 3     # max distinct notation hits that add points
DOWNWEIGHT_REVIEW = 1         # subtracted if the paper looks like a review
DOWNWEIGHT_BINDING_ONLY = 1   # subtracted if only binding assays, no ephys

# Papers scoring at/above this go on the "Worklist" sheet; the rest go to
# "Rejected" (nothing is deleted). Override at runtime with --min-score.
DEFAULT_MIN_SCORE = 3

# ── Worklist Excel columns ───────────────────────────────────────────────────
WORKLIST_COLUMNS = [
    "PMID",
    "Title",
    "Authors",
    "Year",
    "Journal",
    "Subunits",
    "Sources",
    "Score",
    "Status",
    "Mutation hits",
    "Curated mutations",
    "Curated effects",
    "Effect signals",
    "Ephys signals",
    "DOI",
    "DOI Link",
    "PubMed Link",
    "PMCID",
    "Abstract",
]
