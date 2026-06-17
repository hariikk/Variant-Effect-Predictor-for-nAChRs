"""
Local configuration for the "a try" extraction experiment.

Self-contained: reads the worklist produced by the parent datascraper tool and
writes extracted mutation rows here, in the nachr_db_manual.xlsx format.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).parent              # .../datascraper/a try
DATASCRAPER_DIR = BASE_DIR.parent             # .../datascraper
REPO_DIR = DATASCRAPER_DIR.parent             # repo root

# Load this folder's .env, then fall back to the parent datascraper .env for
# anything not set locally (e.g. NCBI_EMAIL).
load_dotenv(BASE_DIR / ".env")
load_dotenv(DATASCRAPER_DIR / ".env")

# ── Credentials ──────────────────────────────────────────────────────────────
NCBI_EMAIL = os.getenv("NCBI_EMAIL", "")
NCBI_API_KEY = os.getenv("NCBI_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
# Normalize a display name like "Gemini 2.5 Flash" to the API id "gemini-2.5-flash".
GEMINI_MODEL = (os.getenv("GEMINI_MODEL") or "gemini-2.5-flash").strip().lower().replace(" ", "-")
UNPAYWALL_EMAIL = NCBI_EMAIL

# ── Extraction backend: "ollama" (local, free, unlimited) or "gemini" ────────
EXTRACT_BACKEND = os.getenv("EXTRACT_BACKEND", "ollama")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct")
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "32768"))      # fit full paper + JSON output
OLLAMA_NUM_PREDICT = int(os.getenv("OLLAMA_NUM_PREDICT", "8192"))  # max output tokens
OLLAMA_MAX_CHARS = 60000           # truncate paper text to fit the context window
OLLAMA_TIMEOUT = 900               # local generation can be slow (seconds)

# ── Inputs / outputs ─────────────────────────────────────────────────────────
WORKLIST_EXCEL = DATASCRAPER_DIR / "output" / "nachr_mouse_worklist.xlsx"
MANUAL_DB = REPO_DIR / "nachr_db_manual.xlsx"
EXTRACTED_EXCEL = BASE_DIR / "extracted_mutations.xlsx"
EXTRACT_INACCESSIBLE = BASE_DIR / "inaccessible_papers.xlsx"
CHECKPOINT_FILE = BASE_DIR / "extract_checkpoint.json"
EXTRACT_CHECKPOINT = CHECKPOINT_FILE

# ── nAChR subunits (for extractor validation) ────────────────────────────────
ALL_SUBUNITS = [
    "CHRNA1", "CHRNA2", "CHRNA3", "CHRNA4", "CHRNA5",
    "CHRNA6", "CHRNA7", "CHRNA9", "CHRNA10",
    "CHRNB1", "CHRNB2", "CHRNB3", "CHRNB4",
    "CHRND", "CHRNE", "CHRNG",
]

# ── Extraction schema / behaviour ────────────────────────────────────────────
VALID_MODIFICATION_TYPES = ["Substitution", "Frameshift", "Stop", "Deletion"]
VALID_EFFECTS = ["LOF", "GOF", "No net effect"]

# Output schema — matches nachr_db_manual.xlsx exactly.
EXCEL_COLUMNS = [
    "OID", "nAChR subunit", "Modification type", "AA position", "Initial AA",
    "New AA", "Effect", "Measuring Technique", "Pathology", "Reference(PMID)",
    "Entry by", "Correct?", "DOI",
]
INACCESSIBLE_COLUMNS = [
    "PMID", "DOI", "Title", "Authors", "Abstract", "Associated Subunits", "Reason",
]
# AI-extracted rows are provisional — flag them for human review, not "Yes".
DEFAULT_ENTRY_BY = "AI (Gemini)"
DEFAULT_CORRECT = "Pending"

# ── Rate limiting / retries ──────────────────────────────────────────────────
GEMINI_DELAY_SECONDS = 6            # pause between Gemini calls (pace under the rate limit)
MAX_RETRIES = 6
RETRY_BACKOFF_FACTOR = 2            # exponential backoff, capped at 60s in llm_extractor
