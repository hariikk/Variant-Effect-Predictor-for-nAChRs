# -*- coding: utf-8 -*-
"""Append Run XXX cells to results.ipynb (one-off recording helper)."""
import nbformat

NB = "results.ipynb"

header = """## Run XXX: data-size & class-balance ablation — data IS the anchor (steep, no plateau)

Is the model limited by the *amount* of data, the *balance* of classes, or is it at an intrinsic ceiling?
Two factors varied independently, gene-aware (`cv_mode="standard"`), 3-class, RF + CatBoost, 3 seeds × 15
trials, full 82-feat set. Delta vs full (760 = 0.6525 / 0.6499).

- **SIZE** at fixed ~50/25/25 balance: quarter (~196), half (~385), full (760, on disk).
- **BALANCE** at fixed N≈350: 50/25/25 random, 33/33/33 random, 33/33/33 gene-stratified.
"""

setup = """### Setup — size vs balance, and a "better idea" for the 33/33/33 selection

- **Per-gene fractional sampling** (quarter/half): each (gene, class) group is scaled by the fraction, so
  gene coverage AND the ~50/25/25 balance are both preserved — the fair way to shrink a gene-aware dataset.
- **Random stratified** (n350_502525, n350_333333): exact per-class counts, global random.
- **Gene-stratified 33/33/33** (the "better idea"): a *random* 350-sample can drop an entire rare gene's
  variants, which is exactly the information the gene-aware model leans on (gene identity = #1 feature).
  This variant samples each class proportionally per gene, so every gene stays represented.
- **Noise caveat:** quarter/half std is large (±0.03–0.04 for RF) — small-sample variance.
"""

code = """import pandas as pd
from IPython.display import display

d = pd.read_csv("results/data_size/data_size_summary.csv")
view = d.copy()
for c in ["macro_f1_mean", "macro_f1_std", "mcc_mean", "balanced_accuracy_mean"]:
    view[c] = view[c].map(lambda x: f"{x:.4f}")
display(view)

piv = d.pivot_table(index="model", columns="condition", values="macro_f1_mean")
FULL = {"random_forest": 0.6525, "catboost": 0.6499}
print("\\nDelta macro F1 vs full (760):")
for model in ["random_forest", "catboost"]:
    print(f"\\n[{model}] full = {FULL[model]:.4f}")
    for cond in piv.columns:
        print(f"    {cond:22s} {piv.loc[model, cond]:.4f}  Δ={piv.loc[model, cond]-FULL[model]:+.4f}")
"""

takeaways = """### Run XXX takeaways — data size is the anchor, and minority-class data is the scarce resource

1. **Data IS the anchor — the size curve is steep and still climbing.** quarter (196) 0.545/0.531 →
   half (385) 0.561/0.587 → full (760) 0.653/0.650. Halving the data costs ~0.09 (RF) / ~0.06 (CatBoost),
   and there's no plateau in sight — the curve is still rising at 760. So more data would keep helping.
2. **At fixed size, class balance matters in the minority classes' favour.** At N=350, 33/33/33
   (0.605/0.593) beats 50/25/25 (0.583/0.586). Both have 350 points, so the gain comes from taking data
   *out* of LOF and giving it to GOF/NNE — i.e. **minority-class data is worth more per point than
   majority-class data.** This is the same mechanism Run XXIX found (cross-species helps by adding NNE/GOF),
   now shown directly at constant N.
3. **Gene coverage matters for a gene-aware model (the "better idea" pays off).** Gene-stratified 33/33/33
   (0.611/0.610) > random 33/33/33 (0.605/0.593), with CatBoost gaining +0.017. A random 350-sample can
   drop a rare gene's variants, and since gene identity is the model's #1 feature, that hurts. Balanced
   sampling should preserve per-gene coverage.
4. **Caveats.** The quarter→half size step (RF +0.016) is within the ±0.03 small-sample noise; the
   half→full step (+0.09/+0.06) is the robust signal. And the "balance" comparison is confounded by class
   counts (more GOF/NNE at fixed N means fewer LOF) — which is precisely the point: minority data is the
   binding constraint.

**Verdict: data-limited, not at a ceiling.** The highest-value direction is more data — especially GOF/NNE
(minority) rows and preserving per-gene coverage — consistent with Run XXIX's cross-species conclusion.
"""

nb = nbformat.read(NB, as_version=4)
for src, kind in [(header, "markdown"), (setup, "markdown"),
                  (code, "code"), (takeaways, "markdown")]:
    nb.cells.append(nbformat.v4.new_code_cell(src) if kind == "code"
                    else nbformat.v4.new_markdown_cell(src))

nbformat.write(nb, NB)
print(f"appended 4 cells -> {len(nb.cells)} total")
