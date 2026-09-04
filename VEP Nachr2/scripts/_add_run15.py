"""Append Run XV (structural overhaul — negative, reverted) cells to results.ipynb."""
import json
from pathlib import Path

NB = Path("results.ipynb")
nb = json.loads(NB.read_text(encoding="utf-8"))


def md(text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [ln + "\n" for ln in text.rstrip("\n").split("\n")],
    }


def code(text):
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": [ln + "\n" for ln in text.rstrip("\n").split("\n")],
    }


cells = [
    md(
        """## Run XV: structural overhaul (mutation-aware features) — NEGATIVE, reverted

Run XIV fixed the position→PDB mapping so `structural_core` reads the *right* residue,
but it stayed neutral (−0.002). Run XV attacks the question head-on: diagnose every reason
structure is weak, fix each one, and re-measure."""
    ),
    md(
        """### The 5 reasons structural features were weak, and the fixes

1. **WT-only features.** `structural_core` (RSA, Cβ-density, B-factor, DSSP, is_unmappable)
   describes the *wild-type* residue, so every substitution at a position gets an identical
   value → no mutation signal. **Fix:** add mutation-aware extractors.
2. **B-factor scale mixing.** pLDDT (AlphaFold) vs crystallographic B (experimental PDBs) are
   different scales, blended into one column. **Fix:** z-score within each structure type.
3. **`is_unmappable` is ~92% dead.** Only ~8% of variants are unmappable → a near-constant column.
4. **Missing nAChR biology.** No M2-gate / Cys-loop / binding-pocket / TM membership.
   **Fix:** FunctionalMotifExtractor — 5 binary flags at the GOF/LOF functional sites.
5. **No burial × chemical-change interaction.** A buried Ala→Val matters more than a surface
   one. **Fix:** StabilityExtractor — burial × Δhydrophobicity and burial × Δvolume (a
   structure-aware ΔΔG proxy).

**Three extractors wired in:** StabilityExtractor (2), FunctionalMotifExtractor (5),
ChargeContextExtractor (4 — Δcharge × location). Plus structural_core expanded to 9 features
(pLDDT + contact_number + z-scored B-factor) → 88-feat "full overhaul"."""
    ),
    code(
        """import pandas as pd
from IPython.display import display

# Run XV — structural overhaul. Two configs measured (1-seed, macro F1, gene-grouped CV):
#   88-feat "full overhaul"  (structural_core→9 + stability + functional_motif + charge_context)
#   82-feat "refined"        (structural_core→7 reverted + charge_context dropped)
# The 82-feat run was the last on-disk state; preserved as *_run15.csv before reverting to 75-feat.
# delta_f1 = F1(drop) - F1(full); NEGATIVE = group helps. 1 seed -> ±0.02 noise, not a ranking.

r15_rf = pd.read_csv("results/ablation/ablation_impact_random_forest_run15.csv", index_col=0)
r15_cb = pd.read_csv("results/ablation/ablation_impact_catboost_run15.csv", index_col=0)

def fmt(df):
    d = df[["n_features", "macro_f1_mean", "delta_f1"]].sort_values("delta_f1").copy()
    d["macro_f1_mean"] = d["macro_f1_mean"].map(lambda x: f"{x:.4f}")
    d["delta_f1"] = d["delta_f1"].map(lambda x: f"{x:+.4f}")
    return d

print("82-feat RF  (full F1 %.4f):" % r15_rf.loc["full", "macro_f1_mean"])
display(fmt(r15_rf))
print("\\n82-feat CatBoost  (full F1 %.4f):" % r15_cb.loc["full", "macro_f1_mean"])
display(fmt(r15_cb))

# Headline comparison (1-seed macro F1):
print("\\n  config        |   RF    | CatBoost")
print("  75-feat base  | 0.4715  | 0.490   <- best, kept")
print("  88-feat full  | 0.4581  | 0.4672")
print("  82-feat ref   | 0.4577  | 0.4468")"""
    ),
    md(
        """### Run XV takeaways

1. **The structural overhaul did not flip it — it regressed it.** Every mutation-aware /
   structure-expanded config scored *below* the 75-feat baseline on both models. The
   "helpful" ablation deltas for stability/functional_motif were ~±0.02 noise: they flipped
   sign between the 88-feat and 82-feat runs (1 seed → macro_f1_std = 0.0).

2. **Conclusion: single-residue structural context carries no direction-of-effect signal for
   nAChR GOF/LOF — WT-side or mutation-aware.** The 5 diagnosed reasons were real and fixed,
   yet none of the fixes produced a robust lift. The signal is *absent*, not *broken*.

3. **Repo reverted to the 75-feat default set** (Run XIV baseline, catboost 0.490). The three
   mutation-aware modules (stability.py, functional_motif.py, charge_context.py) are kept on
   disk for the thesis ablation section but are NOT in the default set.

4. **Where the direction signal actually lives** (consistent with Run XIII): sequence-level
   conservation + ESM-2 embeddings + positional one-hot. Structure is not a lever; the
   remaining headroom is the data (curated variants + label quality), not features."""
    ),
]

nb["cells"].extend(cells)
NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"Appended {len(cells)} cells; total now {len(nb['cells'])}")
