# -*- coding: utf-8 -*-
"""Append Run XXXI cells to results.ipynb (one-off recording helper)."""
import nbformat

NB = "results.ipynb"

header = """## Run XXXI: subunit-type ablation — δ/γ/ε alone beats the full model; α and β lag

Does restricting to a single subunit class change the scores? Run α alone, β alone, and δ/γ/ε alone,
gene-aware (`cv_mode="standard"`), 3-class, RF + CatBoost, 3 seeds × 15 trials, full 82-feat set.
Delta vs full (760 = 0.6525 / 0.6499, Run XXVIII).

### Results at a glance

| subset | n | genes | RF macro F1 | CatBoost macro F1 | Δ RF | Δ CatBoost |
|---|---|---|---|---|---|---|
| α subunits | 517 | 9 | 0.634 | 0.621 | −0.019 | −0.029 |
| β subunits | 128 | 4 | 0.634 | 0.613 | −0.018 | −0.037 |
| δ/γ/ε subunits | 115 | 3 | **0.688** | **0.690** | **+0.035** | **+0.040** |
| **full (760)** | 760 | 16 | **0.653** | **0.650** | — | — |
"""

setup = """### Setup — three subunit classes, one at a time

- **Groups** (by curated `subunit` name): **α** = CHRNA1-7, CHRNA9, CHRNA10 (9 genes); **β** = CHRNB1-4
  (4 genes); **δ/γ/ε** = CHRND, CHRNE, CHRNG (3 genes). α carries the principal ACh-binding face; β and
  δ/γ/ε contribute the complementary / structural faces.
- **Protocol** matches the gene-aware runs (XXVI onward): `cv_mode="standard"` (plain StratifiedKFold),
  3-class, RF + CatBoost, 3 seeds × 15 trials, full 82-feat set. Features extracted once on all 760 rows,
  then subset by group (feature dim stays 82; the unused gene one-hot columns are constant within a subset
  and are ignored by the tree models).
- **Sizes & class split**: α 517 (LOF 265 / GOF 127 / NNE 125), β 128 (39 / 44 / 45), δ/γ/ε 115
  (73 / 24 / 18). δ/γ/ε is the smallest *and* most LOF-heavy group — its comparison is the noisiest, yet it
  scores the highest.
"""

code = """import pandas as pd
from IPython.display import HTML, display

d = pd.read_csv("results/subunit_type/subunit_type_summary.csv")
FULL = {"random_forest": 0.6525, "catboost": 0.6499}

labels = {
    "alpha": "α subunits (CHRNA1-7, 9, 10)",
    "beta": "β subunits (CHRNB1-4)",
    "delta_gamma_epsilon": "δ/γ/ε subunits (CHRND, E, G)",
}

def _cell(g, m):
    r = d[(d["group"] == g) & (d["model"] == m)].iloc[0]
    val = f"{r['macro_f1_mean']:.4f} ± {r['macro_f1_std']:.3f}"
    dlt = f"{r['macro_f1_mean'] - FULL[m]:+.4f}"
    return val, dlt

css = ("<style>table{border-collapse:collapse;margin:0.4em 0 1.6em;font-size:0.92em}"
       "th,td{border:1px solid #bbb;padding:6px 12px;text-align:left}"
       "thead th{background:#eee}tbody tr:nth-child(even){background:#f8f8f8}"
       "td:nth-child(5),td:nth-child(6){color:#666}</style>")

rows = []
for g, lab in [("alpha", labels["alpha"]), ("beta", labels["beta"]),
               ("delta_gamma_epsilon", labels["delta_gamma_epsilon"]), ("full", "Full (760)")]:
    n = 760 if g == "full" else int(d[d["group"] == g]["n"].iloc[0])
    if g == "full":
        rf_v, rf_d, cb_v, cb_d = "0.6525", "—", "0.6499", "—"
    else:
        rf_v, rf_d = _cell(g, "random_forest")
        cb_v, cb_d = _cell(g, "catboost")
    rows.append(f"<tr><td>{lab}</td><td>{n}</td>"
                f"<td>{rf_v}</td><td>{cb_v}</td><td>{rf_d}</td><td>{cb_d}</td></tr>")

html = ("<h4>Subunit-type ablation — Δ vs full (760)</h4><table><thead><tr>"
        "<th>Subset</th><th>n</th>"
        "<th>RF macro F1 (mean ± std)</th><th>CatBoost macro F1 (mean ± std)</th>"
        "<th>Δ RF</th><th>Δ CatBoost</th></tr></thead><tbody>"
        + "".join(rows) + "</tbody></table>")

display(HTML(css + html))
"""

takeaways = """### Run XXXI takeaways — δ/γ/ε (muscle structural subunits) scores *above* the full model; α and β lag

1. **δ/γ/ε alone beats the full 16-gene model — on the smallest group.** δ/γ/ε (CHRND/E/G, 115 variants)
   reaches **0.688/0.690**, vs the full 760-variant model's **0.653/0.650** (Δ +0.035 RF / +0.040 CatBoost).
   That runs *opposite* to the data-size curve (Run XXX: fewer rows → lower score), so the gain here is not
   "more data" — it's *less heterogeneity*.
2. **α and β both lag the full model.** α (517, the biggest group) scores 0.634/0.621 (Δ −0.019/−0.029);
   β (128) scores 0.634/0.613 (Δ −0.018/−0.037). So the full model's 0.65 is being *held up* by the δ/γ/ε
   signal — α and β individually are each harder than the pooled model.
3. **Interpret with care — δ/γ/ε is a confound-rich group.** It is (a) the three muscle *accessory* subunits,
   which are mechanistically clean and well-characterised in disease (congenital myasthenic syndromes), and
   (b) only 3 genes, so the gene-identity feature — the model's #1 signal — collapses to a "δ vs ε vs γ" label
   that carries direction almost for free. The high score is real under gene-aware CV, but it is as much "a
   clean 3-gene identity signal" as "direction is intrinsically easier in δ/γ/ε".
4. **The decisive follow-up** is to re-run δ/γ/ε (and each group) under **leave-one-gene-out** CV
   (`cv_mode="subunit"`): if 0.69 survives when the model must generalise to an *unseen* δ/γ/ε gene, the
   signal is biological; if it collapses toward the chance floor, gene identity was doing the work.

**Verdict:** subunit type is a real axis of heterogeneity — the muscle structural subunits (δ/γ/ε) are the
most direction-predictable slice, and α/β are each harder than the pooled model. Natural next experiments:
subunit-type-conditioned models, and the leave-one-gene-out robustness check on δ/γ/ε.
"""

nb = nbformat.read(NB, as_version=4)
for src, kind in [(header, "markdown"), (setup, "markdown"),
                  (code, "code"), (takeaways, "markdown")]:
    nb.cells.append(nbformat.v4.new_code_cell(src) if kind == "code"
                    else nbformat.v4.new_markdown_cell(src))

nbformat.write(nb, NB)
print(f"appended 4 cells -> {len(nb.cells)} total")
