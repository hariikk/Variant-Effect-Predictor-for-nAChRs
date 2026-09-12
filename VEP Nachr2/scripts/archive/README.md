# Archived experiment scripts

Null-result experiment drivers, retained for the thesis ablation record. Each
imports from `vep_nachr2.features.experimental.*`. They are not part of the active
pipeline (`scripts/run_experiment.py` + `scripts/ablate_feature_impact.py`).

| Script | Run | Hypothesis | Result |
|--------|-----|------------|--------|
| `run_alpha7_diagnostic.py` | XVI | α7-only two-state (open↔closed) | null even in homomeric α7 |
| `run_h1_functional_region.py` | XVII (H1) | functional-region partition | null / gene confound |
| `run_h2_allosteric.py` | XVIII (H2) | allosteric contact-graph position | null / gene confound |
| `run_h3_ddg.py` | XIX (H3) | ESM-IF open↔closed ΔΔG | null |
| `run_h4_h5.py` | XX–XXI (H4–H5) | coevolution + embedding delta | null / null |
| `run_ideas.py` | XIII | embeddings/stacking/ordinal ideas | null / negative |
| `lora_finetune.py` | XIII | LoRA fine-tune | negative |

Results live in `results/<run>/` and are documented in `results.ipynb`.
