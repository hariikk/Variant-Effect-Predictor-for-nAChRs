"""Append Run XIV cells to results.ipynb (structural mapping fix + structural_nachr drop)."""
import json
from pathlib import Path

NB = Path("results.ipynb")

markdown_1 = """## Run XIV: fix structural position→PDB mapping + drop `structural_nachr`

The "structural features don't help" question had a concrete answer: the position→PDB
mapping was broken, so `structural_core` was reading features from the *wrong residues*.
Fixing it removes the harm but — consistent with Run XIII — does not turn structure into
a strong direction signal. `structural_nachr` is dropped from the default set (sparse, neutral).
"""

markdown_2 = """### What was broken

1. **5 genes homology-mapped to a different subunit** — CHRNA2/5/6, CHRNB3, CHRNG were mapped to
   another gene's chain (~47% identity), so every structural feature came from the wrong residue.
2. **CHRNA10 used the wrong isoform** — AF-Q13002 (908 aa) instead of the canonical Q9GZZ6 (450 aa).
3. **The 7 AlphaFold models had no `.dssp`** — RSA fell back to a fill value (1.0) for every position.

### Fix

- Gapped global alignment (BioPython `PairwiseAligner`, BLOSUM62, open −10 / extend −0.5) +
  **<80% identity rejection** in `structural.py::_build_alignment_map`.
- Repointed 6 genes to their **own AlphaFold model** in `config.py::PDB_MAPPING`.
- Real solvent accessibility via **Shrake-Rupley SASA** in `structural.py::_compute_dssp_dict_from_structure`
  (bypasses `mkdssp`, not installed).

Result: all 760 variants map to real RSA (91.6% overall; the previously-broken genes 100%).
"""

code = """import pandas as pd
from IPython.display import display

# Run XIV — structural_core mapping fix + structural_nachr removed from default.
# Reads results/ablation/ablation_impact_random_forest.csv (75 features, RF, 1 seed,
# gene-grouped subunit CV, 30 trials). delta_f1 = F1(drop) - F1(full); NEGATIVE = helps.
# 1 seed -> point estimate (macro_f1_std = 0.0); read deltas as ~±0.02 noise, not a ranking.

r14 = pd.read_csv("results/ablation/ablation_impact_random_forest.csv", index_col=0)
r14 = r14[["n_features", "macro_f1_mean", "delta_f1"]].sort_values("delta_f1")
r14_fmt = r14.copy()
r14_fmt["macro_f1_mean"] = r14_fmt["macro_f1_mean"].map(lambda x: f"{x:.4f}")
r14_fmt["delta_f1"] = r14_fmt["delta_f1"].map(lambda x: f"{x:+.4f}")
display(r14_fmt)

# structural_core before/after the mapping fix (broken -> fixed), RF 1-seed:
#   broken mapping (Run XII, 71-feat):  +0.007  (structural hurt)
#   fixed mapping  (Run XIV, 75-feat):  -0.002  (neutral)
print("\\nstructural_core delta:  +0.007 (broken) -> -0.002 (fixed).")
print("Full-model macro F1: 0.4597 (82-feat, with structural_nachr) -> 0.4715 (75-feat, nachr removed).")
"""

markdown_3 = """### Run XIV takeaways

1. **The mapping bug is fixed; structure is now *neutral*, not harmful.** `structural_core` flipped
   from +0.007 (broken mapping → wrong-residue features) to −0.002 (fixed → correct residues +
   real RSA). Fixing the bug removed the drag but did not create a positive signal — structure is
   still not a direction lever, consistent with Run XIII.

2. **The earlier 82-feat run overstated the structural win.** In that run `structural_core` showed
   −0.019, but that was redundancy with the (harmful) `structural_nachr` group. Once `structural_nachr`
   is dropped, `structural_core` reverts to −0.002. Read per-group deltas with the *full* feature set in mind.

3. **`structural_nachr` removed from the default set.** Its pore/interface/burial features are 0.0 for the
   7 AlphaFold-monomer genes (single chain → no interface) and its ligand feature is α-only
   (~72% of the group is zeros). Dropping it lifts full F1 0.4597 → 0.4715 and simplifies 82 → 75
   features. Module kept for the thesis ablation section.

4. **The ceiling is unchanged: data.** This is a correctness cleanup, not a new lever — the headline
   number is still catboost 0.490 (Run XIII). The structural feature space is now clean (correct mapping,
   real ASA, no degenerate group); the open lever remains curated-data expansion.
"""


def lines(s):
    return s.splitlines(keepends=True)


cells = [
    {"cell_type": "markdown", "id": "run14-header", "metadata": {}, "source": lines(markdown_1)},
    {"cell_type": "markdown", "id": "run14-fix", "metadata": {}, "source": lines(markdown_2)},
    {"cell_type": "code", "execution_count": None, "id": "run14-ablation", "metadata": {}, "outputs": [], "source": lines(code)},
    {"cell_type": "markdown", "id": "run14-takeaways", "metadata": {}, "source": lines(markdown_3)},
]

nb = json.loads(NB.read_text(encoding="utf-8"))
nb["cells"].extend(cells)
NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"Appended {len(cells)} cells -> {NB} (now {len(nb['cells'])} cells)")
