# VEP-nAChR2: Variant Effect Predictor for Nicotinic Acetylcholine Receptors

Clean VEP predicting **GOF vs LOF vs No-net-effect** for missense variants in 16 human nAChR genes.

## Quick context

- **What:** 3-class functional direction predictor (NOT generic pathogenicity)
- **Data:** 797 curated substitution variants from `data/raw/final.xlsx` (human 542, rat 174, mouse 80)
- **Labels:** LOF→0, No net effect→1, GOF→2
- **Genes:** CHRNA1-7, A9, A10, CHRNB1-4, CHRND, CHRNE, CHRNG (16 total)
- **Features:** 66 features across 7 extractor groups (physicochemical, substitution, positional, core structural, nAChR-specific structural, conformational, embeddings placeholder)
- **Models:** 10 classical ML (LR, SVM-RBF/Linear, RF, LightGBM, KNN, MLP, GaussianNB, XGBoost, CatBoost)
- **CV:** Nested leave-one-subunit-out (gene-level StratifiedGroupKFold) + Optuna HP tuning
- **Architecture:** Modeled after VEP-ENAC (`VEP-ENAC/vep/`) but built from scratch

## How to run

```bash
cd "VEP Nachr2"
pip install -e .
python scripts/download_pdbs.py          # Download PDBs (one-time)
python scripts/download_alphafold.py      # Download AlphaFold structures
python scripts/run_experiment.py --test   # Quick test
python scripts/run_experiment.py --full   # Full experiment
```

## Project structure

```
vep_nachr2/
├── config.py               # ALL constants — PDB mapping, gene lists, AAIndex, HP spaces
├── data/
│   ├── loader.py           # load_mutation_data(): 841 → 797 filtered, standardized
│   └── reference.py        # Reference sequence loading (16 human FASTA files)
├── features/
│   ├── base.py             # FeatureExtractor ABC (name, n_features, extract(), requires_pdb())
│   ├── physicochemical.py  # 24 features: 8 AAIndex props × {wt, mt, diff}
│   ├── substitution.py     # 3 features: BLOSUM62 + Grantham
│   ├── positional.py       # 20 features: norm_pos + 16 gene OH + 3 species OH
│   ├── structural.py       # 7 features: RSA, Cβ-density, B-factor, DSSP, is_unmappable
│   ├── structural_nachr.py # 7 features: TMD helix, pore distance, ligand/interface proximity
│   ├── conformational.py   # 5 features: open/closed delta (α7 only — 7EKT vs 7EKO)
│   ├── embeddings.py       # 0 features: ESM-2 placeholder
│   └── orchestrator.py     # Runs all extractors, concatenates, caches to .pkl
├── models/
│   ├── registry.py         # 10 models + Optuna HP spaces + suggest_and_build()
│   └── imbalance.py        # Per-model strategy dispatch (cost_sensitive/brfc/ros/xgb_inverse)
└── training/
    ├── cross_validation.py # nested_cv(), homology_class_transfer_cv(), species_transfer_cv()
    ├── evaluation.py       # compute_metrics(), aggregate_metrics(), format_classification_report()
    └── runner.py           # CLI + run_single/compare/ablation/species_transfer orchestrators
```

## Key design decisions

1. **3-class prediction** (GOF/LOF/NNE) — drops 14 LOF/GOF ambiguous labels
2. **Substitutions only** — 30 indels/stops/frameshifts excluded; BLOSUM/Grantham don't apply
3. **Gene-level CV** — StratifiedGroupKFold by subunit; no gene leakage across folds
4. **Per-model imbalance** — cost_sensitive for tree models, BRFC for RF, ROS for KNN/MLP/NB, xgb_inverse for XGBoost
5. **Pipeline architecture** — each FeatureExtractor independently testable and droppable for ablation
6. **Graceful PDB degradation** — runs without PDBs; fill values for missing structures
7. **Multi-PDB architecture** — one structure per receptor type (not one PDB for all)

## PDB mapping

| PDB | Genes Covered | Chains |
|-----|--------------|--------|
| 6UW8 | CHRNA1, CHRNB1, CHRND, CHRNE, CHRNG | A,B,D,E,E |
| 7EKT | CHRNA7 (closed) | A |
| 7EKO | CHRNA7 (open, conformational) | A |
| 6CNJ | CHRNA4, CHRNB2, CHRNA2*, CHRNA6* | A,B,A,A |
| 6PV7 | CHRNA3, CHRNB4, CHRNA5*, CHRNB3* | A,B,A,B |
| AF-Q9UGM1 | CHRNA9 (AlphaFold) | A |
| AF-Q13002 | CHRNA10 (AlphaFold) | A |

*Homology-mapped (e.g., CHRNA2→α4 chain of 6CNJ)

## Feature details

- **Physicochemical (24):** EISD840101 (hydrophobicity), GRAR740102 (polarity), KRIW790103 (volume), FASG760101 (mol weight), KLEP840101 (charge), ZIMJ680104 (isoelectric), aromaticity (binary F/W/Y/H), CHOP780201 (SS preference). All min-max normalized [0,1].
- **Substitution (3):** BLOSUM62 raw + normalized + Grantham distance (composition/polarity/volume)
- **Positional (20):** position/gene_length + 16 gene one-hot + 3 species one-hot
- **Core structural (7):** RSA from DSSP, Cβ-density (KDTree 10Å), B-factor, DSSP helix/sheet/coil one-hot, is_unmappable
- **nAChR structural (7):** TM helix ID (0-4 from UniProt annotations), TM depth, pore distance, ligand proximity, interface proximity, interface contacts, subunit burial
- **Conformational (5):** α7 only — delta features between 7EKT (closed) and 7EKO (open)
- **Embeddings (0):** Placeholder for ESM-2

## Reference data

- **Source:** `merging_data/final.xlsx` (unique_variants sheet) → copied to `data/raw/final.xlsx`
- **Reference sequences:** 16 FASTA files in `data/raw/reference_sequences/human/` (copied from `protein_scequences/`)
- **PDB structures:** `data/raw/structure_files/<PDB_ID>/` containing .pdb, .cif, .dssp

## Important: what NOT to modify

- `VEP Nachr/` — old AI-generated project; ignore entirely
- The notebooks in `VEP Nachr/` (week1.ipynb, features.ipynb) are reference only — the PDB mapping and feature recipes are useful but the code is not
- `VEP-ENAC/` is the architectural reference — understand it but don't import from it
