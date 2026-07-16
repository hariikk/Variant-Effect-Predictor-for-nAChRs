# Graph Report - .  (2026-07-09)

## Corpus Check
- Large corpus: 173 files � ~1,196,565 words. Semantic extraction will be expensive (many Claude tokens). Consider running on a subfolder.

## Summary
- 488 nodes · 712 edges · 31 communities (24 shown, 7 thin omitted)
- Extraction: 94% EXTRACTED · 6% INFERRED · 0% AMBIGUOUS · INFERRED: 44 edges (avg confidence: 0.74)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Community 0|Community 0]]
- [[_COMMUNITY_Community 1|Community 1]]
- [[_COMMUNITY_Community 2|Community 2]]
- [[_COMMUNITY_Community 3|Community 3]]
- [[_COMMUNITY_Community 4|Community 4]]
- [[_COMMUNITY_Community 5|Community 5]]
- [[_COMMUNITY_Community 6|Community 6]]
- [[_COMMUNITY_Community 7|Community 7]]
- [[_COMMUNITY_Community 8|Community 8]]
- [[_COMMUNITY_Community 9|Community 9]]
- [[_COMMUNITY_Community 10|Community 10]]
- [[_COMMUNITY_Community 11|Community 11]]
- [[_COMMUNITY_Community 12|Community 12]]
- [[_COMMUNITY_Community 13|Community 13]]
- [[_COMMUNITY_Community 14|Community 14]]
- [[_COMMUNITY_Community 15|Community 15]]
- [[_COMMUNITY_Community 16|Community 16]]
- [[_COMMUNITY_Community 17|Community 17]]
- [[_COMMUNITY_Community 18|Community 18]]
- [[_COMMUNITY_Community 19|Community 19]]
- [[_COMMUNITY_Community 20|Community 20]]
- [[_COMMUNITY_Community 21|Community 21]]
- [[_COMMUNITY_Community 22|Community 22]]
- [[_COMMUNITY_Community 23|Community 23]]
- [[_COMMUNITY_Community 24|Community 24]]
- [[_COMMUNITY_Community 25|Community 25]]

## God Nodes (most connected - your core abstractions)
1. `extract_structural_features()` - 20 edges
2. `CheckpointManager` - 15 edges
3. `CheckpointManager` - 14 edges
4. `CheckpointManager` - 14 edges
5. `get_hyperparameter_space()` - 12 edges
6. `PaperFetcher` - 12 edges
7. `PaperFetcher` - 12 edges
8. `NachrFeatureEncoder` - 11 edges
9. `load_dataset()` - 10 edges
10. `nested_cross_validation()` - 10 edges

## Surprising Connections (you probably didn't know these)
- `main()` --calls--> `load_dataset()`  [INFERRED]
  VEP Nachr/scripts/run_experiment.py → VEP Nachr/vep_nachr/data/loader.py
- `main()` --calls--> `NachrFeatureEncoder`  [INFERRED]
  VEP Nachr/scripts/run_experiment.py → VEP Nachr/vep_nachr/features/encoder.py
- `_optimize_hyperparameters()` --calls--> `get_hyperparameter_space()`  [INFERRED]
  VEP Nachr/vep_nachr/training/cross_validation.py → VEP Nachr/vep_nachr/models/registry.py
- `main()` --calls--> `CheckpointManager`  [INFERRED]
  datascraper/a try/extract_main.py → datascraper/a try/checkpoint.py
- `main()` --calls--> `ExcelWriter`  [INFERRED]
  datascraper/a try/extract_main.py → datascraper/a try/excel_writer.py

## Import Cycles
- None detected.

## Communities (31 total, 7 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.06
Nodes (35): BaseEstimator, TransformerMixin, NachrFeatureEncoder, DataFrame, ndarray, Feature encoder: combines all feature groups into a single feature matrix.  Im, Compute feature names based on configuration., Get output feature names. (+27 more)

### Community 1 - "Community 1"
Cohesion: 0.08
Nodes (37): _annotate_sasa_and_hse(), _build_alignment_map(), _build_cbeta_kdtree(), _compute_bfactor(), _compute_cbeta_density(), _compute_dssp_data(), _dssp_to_flags(), extract_structural_features() (+29 more)

### Community 2 - "Community 2"
Cohesion: 0.07
Nodes (29): ExcelWriter, _make_doi_clickable(), Excel output handler for writing extracted mutation data and inaccessible paper, Add a paper to the inaccessible papers Excel file., Turn the bare DOI cells in an .xlsx into clickable https://doi.org links., Return statistics about the output files., Handles reading/writing the output Excel files., Determine the next OID by reading existing data. (+21 more)

### Community 3 - "Community 3"
Cohesion: 0.08
Nodes (31): load_worklist(), main(), paper_key(), parse_args(), Extract mutation data from the shortlisted worklist papers using the recovered f, Read the `Worklist` sheet into a list of paper dicts (already score-sorted)., Stable id for checkpointing (PMID, else DOI, else title)., _call_backend() (+23 more)

### Community 4 - "Community 4"
Cohesion: 0.09
Nodes (27): Any, main(), Main experiment runner for nAChR VEP.  Usage:     python scripts/run_experime, Run quick evaluation with default hyperparameters (no Optuna)., Run full nested CV with Optuna HP optimization., run_full_experiment(), run_quick_test(), get_model() (+19 more)

### Community 5 - "Community 5"
Cohesion: 0.11
Nodes (28): collect(), _dict_to_meta(), _key(), _load_cache(), main(), _merge_into(), _meta_to_dict(), parse_args() (+20 more)

### Community 6 - "Community 6"
Cohesion: 0.10
Nodes (14): CheckpointManager, Path, Checkpoint system for resuming interrupted scraping runs. Tracks which PMIDs hav, Clear all checkpoint data., Manages checkpoint state for the scraping pipeline., Load checkpoint from disk, or return empty state., Persist checkpoint to disk., Check if a PMID has already been processed (successfully or marked inaccessible) (+6 more)

### Community 7 - "Community 7"
Cohesion: 0.10
Nodes (14): CheckpointManager, Path, Checkpoint system for resuming interrupted scraping runs. Tracks which PMIDs hav, Clear all checkpoint data., Manages checkpoint state for the scraping pipeline., Load checkpoint from disk, or return empty state., Persist checkpoint to disk., Check if a PMID has already been processed (successfully or marked inaccessible) (+6 more)

### Community 8 - "Community 8"
Cohesion: 0.10
Nodes (14): CheckpointManager, Path, Checkpoint system for resuming interrupted scraping runs. Tracks which PMIDs hav, Clear all checkpoint data., Manages checkpoint state for the scraping pipeline., Load checkpoint from disk, or return empty state., Persist checkpoint to disk., Check if a PMID has already been processed (successfully or marked inaccessible) (+6 more)

### Community 9 - "Community 9"
Cohesion: 0.13
Nodes (12): PaperFetcher, Response, Full-text paper retrieval module. Tries multiple sources in order: PMC Open Acce, Recursively extract all text from an XML element., Try to fetch full text via Unpaywall API.         Returns (text, source_type) or, Fetches full-text content of scientific papers from legal open-access sources., Download a PDF and extract text content., Download HTML page and extract main text content. (+4 more)

### Community 10 - "Community 10"
Cohesion: 0.15
Nodes (19): _build_query(), _phrase_or(), Europe PMC source: full-text + preprint search to complement PubMed.  Europe PMC, search_all(), search_subunit(), _to_meta(), _build_search_query(), _extract_year() (+11 more)

### Community 11 - "Community 11"
Cohesion: 0.16
Nodes (19): collect(), _dict_to_meta(), _key(), _load_cache(), main(), _merge_into(), _meta_to_dict(), parse_args() (+11 more)

### Community 12 - "Community 12"
Cohesion: 0.14
Nodes (19): _build_query(), _phrase_or(), Europe PMC source: full-text + preprint search to complement PubMed.  Europe PMC, search_all(), search_subunit(), _to_meta(), _build_search_query(), _extract_year() (+11 more)

### Community 13 - "Community 13"
Cohesion: 0.15
Nodes (11): PaperFetcher, Response, Recursively extract all text from an XML element., Try to fetch full text via Unpaywall API.         Returns (text, source_type) or, Fetches full-text content of scientific papers from legal open-access sources., Download a PDF and extract text content., Download HTML page and extract main text content., Make an HTTP GET request with retry logic. (+3 more)

### Community 14 - "Community 14"
Cohesion: 0.15
Nodes (17): CVConfig, Global configuration for VEP-nAChR experiments.  Centralizes paths, biological, Configuration for cross-validation., clean_data(), encode_labels(), load_dataset(), load_raw_data(), load_wildtype_sequences() (+9 more)

### Community 15 - "Community 15"
Cohesion: 0.22
Nodes (14): get_hyperparameter_space(), _hp_gaussian_nb(), _hp_knn(), _hp_lightgbm(), _hp_logistic_regression(), _hp_mlp(), _hp_random_forest(), _hp_svm_linear() (+6 more)

### Community 16 - "Community 16"
Cohesion: 0.17
Nodes (7): ExcelWriter, Excel output handler for writing extracted mutation data and inaccessible paper, Add a paper to the inaccessible papers Excel file., Handles reading/writing the output Excel files., Return statistics about the output files., Determine the next OID by reading existing data., Append extracted mutations to the main Excel file.          Args:             mu

### Community 17 - "Community 17"
Cohesion: 0.31
Nodes (9): _load_pmid_set(), _normalize_pmid(), Write the ranked paper worklist to a two-sheet Excel workbook.  Sheet "Worklist", scored: iterable of (PaperMetadata, score_result dict), any order.      Splits i, Read a set of PMIDs from a column of an existing .xlsx (best-effort)., _row_for(), _status_for(), _write_sheet() (+1 more)

### Community 18 - "Community 18"
Cohesion: 0.31
Nodes (9): _aa1(), _gene_symbol(), _get_json(), _mutation_string(), UniProt source: human nAChR mutagenesis / variant annotations.  For each human s, GET with a UA header and basic retry/backoff on 5xx / network errors., Normalize a UniProt sequence fragment to single-letter code(s)., search_all() (+1 more)

## Knowledge Gaps
- **7 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `CheckpointManager` connect `Community 8` to `Community 2`, `Community 5`?**
  _High betweenness centrality (0.062) - this node is a cross-community bridge._
- **Why does `CheckpointManager` connect `Community 6` to `Community 3`?**
  _High betweenness centrality (0.060) - this node is a cross-community bridge._
- **Why does `CheckpointManager` connect `Community 7` to `Community 11`?**
  _High betweenness centrality (0.059) - this node is a cross-community bridge._
- **Are the 2 inferred relationships involving `CheckpointManager` (e.g. with `main()` and `collect()`) actually correct?**
  _`CheckpointManager` has 2 INFERRED edges - model-reasoned connections that need verification._
- **Are the 10 inferred relationships involving `get_hyperparameter_space()` (e.g. with `_hp_gaussian_nb()` and `_hp_knn()`) actually correct?**
  _`get_hyperparameter_space()` has 10 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Main experiment runner for nAChR VEP.  Usage:     python scripts/run_experime`, `Run quick evaluation with default hyperparameters (no Optuna).`, `Run full nested CV with Optuna HP optimization.` to the rest of the system?**
  _198 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Community 0` be split into smaller, more focused modules?**
  _Cohesion score 0.0613107822410148 - nodes in this community are weakly interconnected._