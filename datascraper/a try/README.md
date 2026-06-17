# "a try" — automated mutation extraction

A self-contained experiment that reads the worklist produced by the parent
`datascraper` tool (`../output/nachr_mouse_worklist.xlsx`), fetches each paper's
full text, and uses Gemini to extract mutation rows in your
`../../nachr_db_manual.xlsx` format. Paywalled / no-open-access papers are logged
and skipped.

Everything for this experiment lives in this folder; the parent `datascraper`
tool is untouched.

## What it does

1. **Fetch** full text per paper: PMC Open Access (real article body required) →
   Unpaywall PDF → Unpaywall HTML. No open-access full text ⇒ logged as
   inaccessible and skipped. (`paper_fetcher.py`)
2. **Extract** with Gemini: the prompt keeps only **mouse** variants measured by
   **electrophysiology / ion flow** and classifies each as **LOF / GOF / No net
   effect**, with subunit, position, AA change, technique, pathology.
   (`llm_extractor.py`)
3. **Write** rows to `extracted_mutations.xlsx` in the exact manual-DB schema;
   OID continues from `nachr_db_manual.xlsx`. Rows are marked
   `Entry by = "AI (Gemini)"`, `Correct? = "Pending"` — **verify before trusting**.
   (`excel_writer.py`)

## Setup

```
cd "a try"
pip install -r requirements.txt
# Edit .env and add: GEMINI_API_KEY=<your key>   (enable billing for the full run)
```

## Run

```
python extract_main.py --limit 5    # small test batch first
python extract_main.py              # full run (resumable, ~1-2 h, ~$1-5)
python extract_main.py --resume     # continue after an interruption
python extract_main.py --reset      # clear the checkpoint
```

## Outputs (this folder)

- `extracted_mutations.xlsx` — extracted rows (manual-DB format; provisional).
- `inaccessible_papers.xlsx` — paywalled / failed papers + reason.
- `extract_checkpoint.json` — resume state.
