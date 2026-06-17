# nAChR mouse-paper worklist builder

Finds candidate **mouse** nAChR papers that report a **mutation** studied by
**electrophysiology** with a functional **effect** (LOF / GOF / no net effect),
aggregates them across three sources, ranks them, and writes an Excel worklist
so you can open each DOI and curate the mutation data by hand into
`../nachr_db_manual.xlsx`.

It builds the reading list; it does **not** auto-extract mutations.

## Sources

| Source     | What it adds                                                                 |
|------------|------------------------------------------------------------------------------|
| **PubMed** | Title/abstract + MeSH search (NCBI Entrez). High precision; the strict mouse query. |
| **Europe PMC** | Full-text + preprint search. Gene term required in title/abstract, other clauses full-text — finds papers PubMed's index misses while staying on-topic. |
| **UniProt** | Curated mouse mutagenesis/variant annotations: residue change + functional effect + the PMIDs they were reported in. Pre-fills part of the extraction. |

## How it works

1. **Search** each source per subunit with the 4-clause query *(gene) AND
   (mouse) AND (electrophysiology) AND (mutation)* (terms in `config.py`;
   builders in `pubmed_search.py`, `europepmc_search.py`, `uniprot_search.py`).
2. **Merge / dedup** by PMID (fallback DOI/title); union the sources and
   subunits; enrich UniProt-only PMIDs with PubMed metadata. (`main.py`.)
3. **Score** each paper's title + abstract with a transparent keyword/regex
   ranker — mutation notation (`S248F`, `L9'T`), effect wording, ephys methods,
   mouse confirmation, plus a boost for UniProt-cited papers. No LLM, no API
   key, no quota. (`scoring.py`.)
4. **Write** `output/nachr_mouse_worklist.xlsx`, best-first, two sheets:
   `Worklist` (score ≥ cutoff) and `Rejected` (everything else — nothing is
   lost). Columns include Year, Journal, Sources, clickable DOI/PubMed links,
   extracted + UniProt-curated mutations/effects, and a `Status` flag
   cross-referencing the human DB and the earlier `mouse_scraped.xlsx`.
   (`worklist_writer.py`.)

## Setup

```
cd datascraper
pip install -r requirements.txt
# .env already contains NCBI_EMAIL; edit it if you want a different email or
# add an optional NCBI_API_KEY for faster PubMed searching.
```

## Run

```
python main.py --all                          # all 16 subunits, all 3 sources
python main.py --subunits CHRNA7 CHRNB2       # a subset
python main.py --all --sources pubmed europepmc   # pick sources
python main.py --all --min-score 5            # tighter worklist (fewer papers)
python main.py --all --refresh                # re-pull everything (ignore caches)
```

Results are cached in `output/`, so re-running with a different `--min-score`
is instant. The metadata cache is keyed by subunits + sources; change either, or
pass `--refresh`, to re-pull.

## Notes

- **UniProt** occasionally returns `503` (their service); the run skips it
  gracefully and continues with the other sources. Re-run `--all` later to fold
  the curated annotations in.
- Europe PMC's species clause is full-text, so a few **human** nAChR papers that
  merely mention "mouse" can appear; confirm species when you read them, or
  raise `--min-score`. Scoring weights live in `config.py` (`SCORE_WEIGHTS`).
- `main.py` prints a score histogram and a per-source breakdown each run to help
  you tune the cutoff.
