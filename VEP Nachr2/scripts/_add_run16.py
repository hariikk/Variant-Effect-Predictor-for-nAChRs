"""Append Run XVI (α7-only structural diagnostic) cells to results.ipynb."""
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
        """## Run XVI: α7-only structural diagnostic — the professor's two-state hypothesis, tested

Runs XII–XV settled that single-residue structural context is neutral/negative in the **full
16-gene model** — where most genes lack a dedicated structure and the receptor is pentameric.
The professor's counter-proposal: *run CHRNA7 (α7) alone* — it's **homomeric** (no interface
ambiguity), the most-represented gene, and the **only one with both a closed (7EKT) and open
(7KOX) experimental structure** — and if the two-state signal works there, later "map other
genes to α7 by approximation."

This run tests that hypothesis directly, in the one gene where complete two-state structure
has the best possible chance to carry signal."""
    ),
    md(
        """### Setup

- **Data:** `load_mutation_data(subunits=["CHRNA7"])` → 200 variants (human 119 / rat 80 /
  mouse 1). Labels: LOF 100 / No-net-effect 58 / GOF 42.
- **Features:** 75-feat default **+ ConformationalExtractor** (the α7-unique open↔closed
  delta group, NOT in the default set) = 80 features. The 16 dead `subunit_*` one-hots are
  masked (constant for a single gene) → 64 features.
- **CV:** `nested_cross_validation(cv_mode="standard")` — plain StratifiedKFold (gene grouping
  is meaningless for one gene), **3 seeds** so `macro_f1_std` is a real confidence interval,
  not the 1-seed `std=0` that plagued Runs XIV/XV.
- **Confound control:** the `species_*` columns carry a human↔rat↔label shortcut (human-α7 is
  84 LOF / 16 GOF; rat-α7 is 39 NNE / 25 GOF), so a `no_species` run masks them out."""
    ),
    code(
        """import pandas as pd
from IPython.display import display

# Run XVI — α7-only, 3-class, all-species (200 variants), 3 seeds.
# delta_f1 = F1(drop) - F1(full); NEGATIVE = the dropped group helps.

r16_cb = pd.read_csv("results/alpha7/alpha7_diagnostic_catboost.csv", index_col=0)
r16_rf = pd.read_csv("results/alpha7/alpha7_diagnostic_random_forest.csv", index_col=0)

def fmt(df):
    d = df[["n_features", "macro_f1_mean", "macro_f1_std", "delta_f1"]].sort_values("delta_f1").copy()
    d["macro_f1_mean"] = d["macro_f1_mean"].map(lambda x: f"{x:.4f}")
    d["macro_f1_std"]  = d["macro_f1_std"].map(lambda x: f"{x:.4f}")
    d["delta_f1"]      = d["delta_f1"].map(lambda x: f"{x:+.4f}")
    return d

print("CatBoost α7-only  (full F1 %.4f ± %.4f, 3 seeds):" % (
    r16_cb.loc["full", "macro_f1_mean"], r16_cb.loc["full", "macro_f1_std"]))
display(fmt(r16_cb))
print("\\nRandomForest α7-only  (full F1 %.4f ± %.4f, 3 seeds):" % (
    r16_rf.loc["full", "macro_f1_mean"], r16_rf.loc["full", "macro_f1_std"]))
display(fmt(r16_rf))

# The crux, both models side by side:
print("\\n  open<->closed conformational delta (drop_conformational - full):")
print("    CatBoost     %+.4f  (± %.4f)" % (
    r16_cb.loc["drop_conformational", "delta_f1"], r16_cb.loc["drop_conformational", "macro_f1_std"]))
print("    RandomForest %+.4f  (± %.4f)" % (
    r16_rf.loc["drop_conformational", "delta_f1"], r16_rf.loc["drop_conformational", "macro_f1_std"]))"""
    ),
    md(
        """### Run XVI takeaways — the hypothesis is settled

1. **α7 alone "works" at the surface level: macro F1 ≈ 0.60 / 0.58** (catboost / RF). But this
   is *not* evidence for structure — it's the single-gene + species shortcut inflating the score
   (a 3-class task on 200 variants with a human-84-LOF / rat-39-NNE split is an easier, partially
   species-leakable problem than the full 16-gene task).

2. **The open↔closed conformational group is null even in α7.** `drop_conformational` = −0.004
   (catboost, ±0.041) and −0.013 (RF, ±0.029). The std is 3–10× the effect in both models. In the
   *one* homomeric gene with *complete* two-state experimental structure (7EKT closed → 7KOX open),
   the conformational delta does not predict GOF/LOF/NNE direction. This is the direct, clean test
   of the professor's hypothesis — and it fails.

3. **structural_core is weak and sign-flips** (−0.015 catboost vs +0.011 RF) — the same noise we
   saw in Runs XIII–XV, now confirmed with real seed variance in the most structure-rich gene.

4. **The species confound is real but model-dependent** (`no_species` = −0.022 catboost vs
   +0.003 RF): catboost exploits the human↔rat↔label shortcut more than RF does. Either way it
   does not dominate — physicochemical (−0.030) is the strongest lever, and that is *sequence*
   chemistry, not structure.

**Bottom line for the professor:** the direction signal is genuinely *absent* from single-residue
structure, even in the ideal homomeric two-state case. The "map other genes to α7 by approximation"
path is therefore moot — mapping to a structure that carries no signal cannot create one. The lever
is data (more curated variants, label quality), not structure. The conformational/structural modules
stay on disk for the thesis ablation section but remain out of the default set."""
    ),
]

nb["cells"].extend(cells)
NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"Appended {len(cells)} cells; total now {len(nb['cells'])}")
