# VEP-nAChR: Variant Effect Predictor for Nicotinic Acetylcholine Receptors

A machine learning project predicting **direction of effect (GOF vs LOF)** for missense variants in human nAChR genes — not generic pathogenicity. Built as a master's thesis project at HBRS (MSc Autonomous Systems).

## What makes this project different

Most VEP tools (AlphaMissense, PolyPhen, CADD, ESM-1v) predict *pathogenicity* (damaging vs benign). This project predicts **gain-of-function vs loss-of-function** — a harder, less-explored task. A pathogenic variant can be either GOF or LOF; generic tools can't distinguish them. See `VEP Nachr/vep_research.ipynb` for the full landscape.

## Project structure

```
VEP Nachr/                    # Main VEP pipeline
├── vep_nachr/
│   ├── config.py             # Central config (paths, biological constants, CV settings)
│   ├── data/
│   │   ├── loader.py         # Dataset loading, cleaning, label encoding
│   │   └── encoders.py       # Feature encoding
│   ├── features/
│   │   ├── encoder.py        # NachrFeatureEncoder: combines all feature groups (44-52 features)
│   │   ├── physicochemical.py  # 24 AA property features
│   │   ├── substitution.py   # BLOSUM62 + Grantham scores
│   │   ├── structural.py     # DSSP, SASA, B-factor, HSE from PDB files
│   │   ├── noah_original.py  # Original VEP-ENaC feature set
│   │   └── noah_features/    # Additional feature modules
│   ├── training/
│   │   ├── cross_validation.py  # Nested CV with Optuna HP optimization
│   │   └── evaluation.py     # Evaluation metrics
│   └── models/               # Model registry + hyperparameter spaces
├── scripts/
│   ├── run_experiment.py     # Main entry point: quick test / full experiment
│   └── experiments/          # Experiment configs/results
├── MASTER.ipynb              # Master workflow notebook
├── week1.ipynb               # Feature analysis, ablation studies, open/closed conformational feature
├── vep_research.ipynb        # VEP method landscape, 5-part pipeline, 6-family map
├── VEP_reserach.ipynb        # Deep dive into papers in VEP papers/ folder
├── features.ipynb            # Concrete feature recipes (ESM-2, AlphaMissense, ConSurf, ThermoMPNN, HOLE)
├── cross_species_papers_analysis.ipynb  # Analysis of GenePlexusZoo + Nature Methods cross-species review
└── NOTES.ipynb / NOTES2.ipynb / TODO.ipynb  # Working notes

datascraper/                  # Mouse data augmentation pipeline
├── main.py                   # Orchestrator: search → score → worklist
├── pubmed_search.py          # PubMed via Entrez API
├── europepmc_search.py       # Europe PMC full-text + preprint search
├── uniprot_search.py         # UniProt variant annotations
├── scoring.py                # Keyword-based paper ranking
├── worklist_writer.py        # Writes ranked worklist to Excel
├── checkpoint.py             # Resume support across multi-day runs
├── config.py                 # Gemini model config, paths
└── a try/                    # Active extraction run (143 rows extracted, pending review)
    ├── extract_main.py       # Extraction orchestrator
    ├── llm_extractor.py      # Full-text → structured mutation rows (Ollama qwen2.5:7b)
    ├── paper_fetcher.py      # Legal full-text retrieval (PMC OA, Unpaywall)
    ├── checkpoint.py         # Extraction checkpointing
    └── excel_writer.py       # Writes extracted data to nachr_extracted_data.xlsx

human_automation/             # Human data collection & paper fetching
presentation/                 # Thesis presentation materials
assignments/                  # Scientific Writing course assignments (PSB 2027 position paper)
related_papers/               # Advisor-recommended papers (cross-species knowledge transfer)
VEP papers/                   # 19 collected VEP method papers
protein_scequences/           # Per-subunit reference sequences and READMEs
memory/                       # Persistent session memory (see MEMORY.md index)
graphify-out/                 # Knowledge graph output
```

## Key technical details

- **Data:** 351 curated human nAChR variants (218 LOF / 133 GOF) across 16 genes (CHRNA1-7, A9, A10, CHRNB1-4, CHRND, CHRNE, CHRNG). Source: `nachr_db_manual.xlsx` (human), `mouse_data_manual.xlsx`, `human_manual2.xlsx` at repo root.
- **Multi-species support:** `loader.py` has `load_multi_species_dataset()` supporting human, mouse, and rat data with species-filtering — cross-species augmentation is built into the architecture.
- **Mouse augmentation:** datascraper extracts mouse nAChR electrophysiology data to supplement human training data. 143 rows extracted via local Ollama (qwen2.5:7b). Caveat: mouse residue numbering ≠ human positions — needs alignment step.
- **Features:** 44 features (52 with structural): 24 physicochemical, 3 substitution (BLOSUM62 + Grantham), 1 normalized position + 16 subunit one-hot, 8 structural (DSSP, SASA, B-factor, HSE). Planned additions: ESM-2 embeddings, AlphaMissense scores, ThermoMPNN ddG, GenePlexusZoo network embeddings.
- **Models:** Logistic regression, SVM, Random Forest, LightGBM, XGBoost, MLP — with Optuna HP tuning
- **Evaluation:** Leave-one-subunit-out grouped CV, PR-AUC, MCC, balanced accuracy (not plain accuracy — class imbalance)
- **Novel angle:** nAChR-specific GOF/LOF + open/closed conformational structural feature (see week1.ipynb)

## SOTA comparators (direction-of-effect methods)

MissION (ion-channel GOF/LOF, 47 voltage-gated genes, ROC-AUC 0.925) does NOT cover nAChR genes — a nAChR-specific model remains a real gap. Other comparators: funNCion (Na/Ca channels), LoGoFunc (genome-wide GOF/LOF ensemble), PreMode (mode-of-action GNN).

## Paper framing

The project targets a PSB 2027 position paper (see `assignments/` for course materials using `ws-procs11x85` kit). Primary contribution: first nAChR-specific GOF/LOF predictor. Secondary contribution: principled cross-species data augmentation using embedding-based functional equivalence (agnology framework from Yuan et al. 2026).

## Critical constraints

- Small dataset (351 variants) → transfer learning over training from scratch
- User on Claude Pro plan (no API tokens) → LLM tasks use Gemini free tier or local Ollama
- Gemini free tier: 20 requests/day/model → scraping is checkpointed for multi-day resumption
- Paywalled papers listed in `inaccessible_papers.xlsx` for manual retrieval

## Key notebooks to read first

1. `VEP Nachr/vep_research.ipynb` — full VEP landscape and project identity
2. `VEP Nachr/week1.ipynb` — feature analysis, ablations, structural features
3. `VEP Nachr/features.ipynb` — concrete feature implementation recipes
4. `VEP Nachr/cross_species_papers_analysis.ipynb` — cross-species transfer papers review

## Related memory files

See `memory/MEMORY.md` for the full index. Key memories: `vep-research-direction.md`, `data-scraper-goal.md`, `datascraper-extraction.md`, `writing-course-paper.md`.
