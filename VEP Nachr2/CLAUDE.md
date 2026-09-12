# VEP-nAChR2: Variant Effect Predictor for Nicotinic Acetylcholine Receptors

Clean VEP predicting **GOF vs LOF vs No-net-effect** for missense variants in 16 human nAChR genes.

## Quick context

- **What:** 3-class functional direction predictor (NOT generic pathogenicity)
- **Data:** 797 curated substitution variants from `data/raw/final.xlsx` (human 542, rat 174, mouse 80)
- **Labels:** LOF→0, No net effect→1, GOF→2
- **Genes:** CHRNA1-7, A9, A10, CHRNB1-4, CHRND, CHRNE, CHRNG (16 total)
- **Features:** 75 features across 7 extractor groups (physicochemical 24, substitution 3, positional 20, conservation 3, core structural 7, ESM-2 2, ESM-2 embeddings 16)
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
│   ├── conservation.py     # 3 features: ortholog conservation wt/mt/delta
│   ├── embeddings.py       # 2 features: ESM-2 150M effect (esm2_wt_logprob, esm2_effect)
│   ├── esm_embeddings.py   # 16 features: ESM-2 per-residue → PCA (esm2_pc1..pc16)
│   ├── experimental/       # 11 archived null-result extractors (see experimental/README.md)
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
| 9DMG | CHRNA1, CHRNB1, CHRND, CHRNE | A, E, D, B |
| 7EKT | CHRNA7 (closed/antagonist) | A |
| 7KOX | CHRNA7 (open — conformational only) | A |
| 6CNJ | CHRNA4, CHRNB2 | A, B |
| 6PV7 | CHRNA3, CHRNB4 | A, B |
| AF-Q9UGM1 | CHRNA9 (AlphaFold) | A |
| AF-Q9GZZ6 | CHRNA10 (AlphaFold) | A |
| AF-Q15822 | CHRNA2 (AlphaFold) | A |
| AF-P30532 | CHRNA5 (AlphaFold) | A |
| AF-Q15825 | CHRNA6 (AlphaFold) | A |
| AF-Q05901 | CHRNB3 (AlphaFold) | A |
| AF-P07510 | CHRNG (AlphaFold) | A |

7 genes without a high-res experimental structure use their own full-length AlphaFold model (direct mapping, offset 0) rather than homology-mapping onto another subunit's chain (~47% identity gave wrong-residue features).

## Feature details

- **Physicochemical (24):** EISD840101 (hydrophobicity), GRAR740102 (polarity), KRIW790103 (volume), FASG760101 (mol weight), KLEP840101 (charge), ZIMJ680104 (isoelectric), aromaticity (binary F/W/Y/H), CHOP780201 (SS preference). All min-max normalized [0,1].
- **Substitution (3):** BLOSUM62 raw + normalized + Grantham distance (composition/polarity/volume)
- **Positional (20):** position/gene_length + 16 gene one-hot + 3 species one-hot
- **Core structural (7):** RSA from DSSP, Cβ-density (KDTree 10Å), B-factor, DSSP helix/sheet/coil one-hot, is_unmappable
- **Conservation (3):** position-specific ortholog conservation (conservation_wt/mt/delta)
- **ESM-2 effect (2):** esm2_wt_logprob + esm2_effect (150M model)
- **ESM-2 embeddings (16):** per-residue embeddings → PCA to 16 dims (esm2_pc1..pc16)
- **Archived (see features/experimental/):** nAChR structural (7), conformational (5) — null under gene-grouped CV

## Reference data

- **Source:** `merging_data/final.xlsx` (unique_variants sheet) → copied to `data/raw/final.xlsx`
- **Reference sequences:** 16 FASTA files in `data/raw/reference_sequences/human/` (copied from `protein_scequences/`)
- **PDB structures:** `data/raw/structure_files/<PDB_ID>/` containing .pdb, .cif, .dssp

## Important: what NOT to modify

- `VEP Nachr/` — old AI-generated project; ignore entirely
- The notebooks in `VEP Nachr/` (week1.ipynb, features.ipynb) are reference only — the PDB mapping and feature recipes are useful but the code is not
- `VEP-ENAC/` is the architectural reference — understand it but don't import from it
