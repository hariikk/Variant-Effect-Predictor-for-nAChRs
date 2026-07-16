# human_automation — HUMAN nAChR mutation data pipeline

The human counterpart of the mouse pipeline in `../datascraper` (+ `../datascraper/a try`).
You already hand-curated human nAChR mutations in `../nachr_db_manual.xlsx`; this finds
candidate papers you may have **missed** and extracts their mutation rows in the same format,
so you can review and merge the genuinely new ones.

Self-contained: one `config.py` + `checkpoint.py` shared by both stages.

## Two stages

**1. Worklist (free, no LLM) — `main.py`**
Searches PubMed + Europe PMC + UniProt for **human** nAChR mutation-electrophysiology papers,
dedupes, keyword-scores, and writes `output/nachr_human_worklist.xlsx` (two sheets: `Worklist`
≥ cutoff, `Rejected` below). The `Status` column flags each paper `already in manual DB` vs
`new` — so you can focus on what you missed.

```
python main.py --all                 # all 16 subunits, all 3 sources
python main.py --all --min-score 4   # tighter worklist
python main.py --all --refresh       # re-pull (ignore caches)
```

**2. Extraction (local LLM) — `extract_main.py`**
Fetches each worklist paper's full text (PMC OA → Unpaywall; paywalled = skipped & logged) and
uses a local **Ollama** model (`qwen2.5:7b-instruct`) to extract mutation rows into
`extracted_human_mutations.xlsx`, in the `nachr_db_manual.xlsx` schema. Rows are marked
`Entry by = "AI (qwen2.5)"`, `Correct? = "Pending"` — **review before merging**. By default it
skips papers already in the manual DB (the point is to find NEW data).

```
python extract_main.py --limit 3   # quick sanity check first
python extract_main.py             # full run (resumable, local, free, ~hours)
python extract_main.py --resume    # continue after an interruption
```

## Setup

```
cd human_automation
pip install -r requirements.txt
# .env already has NCBI creds. For extraction, install Ollama + pull the model:
#   ollama pull qwen2.5:7b-instruct       (or via the API if ollama not on PATH)
# Ollama must be running (tray icon) when you run extract_main.py.
```

## Outputs (this folder)

- `output/nachr_human_worklist.xlsx` — ranked candidate papers (Worklist / Rejected sheets)
- `extracted_human_mutations.xlsx` — extracted rows (manual-DB format; provisional)
- `inaccessible_papers.xlsx` — paywalled / failed papers + reason
- `extract_checkpoint.json` — resume state
