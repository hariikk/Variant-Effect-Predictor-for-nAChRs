# Archived experimental feature extractors

These 11 modules encode the structural/geometric hypotheses that were tested and
found **null** under leave-one-subunit-out CV. They are retained (not deleted) as
the written record for the thesis ablation section. None is in the default feature
set — the 75-feature default is defined in `features/orchestrator.py`.

| Module | Extractor | Run | Hypothesis | Result |
|--------|-----------|-----|------------|--------|
| `structural_nachr.py` | StructuralNachrExtractor | XII | TMD helix / pore / interface proximity | ~72% zeros; neutral (−0.002 ± 0.026) |
| `conformational.py` | ConformationalExtractor | XII, XVI | open↔closed delta (α7 7EKT vs 7KOX) | null, even in α7 (−0.004 ± 0.041) |
| `stability.py` | StabilityExtractor | XV | burial × Δhydrophobicity/Δvolume | ±0.02 noise |
| `functional_motif.py` | FunctionalMotifExtractor | XV | in_tm / in_gate_ring / in_cys_loop / in_binding_site | ±0.02 noise, sign-flipped |
| `charge_context.py` | ChargeContextExtractor | XV | Δcharge × location | null |
| `functional_region.py` | FunctionalRegionExtractor | XVII (H1) | functional-region one-hot | null; pooled signal was gene/species confound |
| `allosteric.py` | AllostericNetworkExtractor | XVIII (H2) | allosteric contact-graph position | null; gate_proximity signal was gene confound |
| `ddg.py` | OpenClosedDDGExtractor | XIX (H3) | ESM-IF open↔closed ΔΔG | null (RF +0.013 / catboost −0.003) |
| `coevolution.py` | CoevolutionExtractor | XX (H4) | ESM-2 attention coupling to gate | null (RF −0.0009 / catboost −0.001) |
| `emb_delta.py` | EmbeddingDeltaExtractor | XXI (H5) | mutant−WT ESM-2 embedding delta | null (RF 0.000 / catboost −0.011) |
| `mechanism.py` | — (shared helper) | — | TM/gate/binding-site annotations | dependency of the above; archived with them |

**Conclusion across all runs:** single-residue structural/geometric context does not
carry GOF/LOF direction signal for nAChR. Direction is sequence-chemistry +
gene/mechanism-conditioned; the honest CV forbids the gene identity that structure
would need to be informative.

**Run scripts:** the corresponding experiment drivers live in `scripts/archive/`.
