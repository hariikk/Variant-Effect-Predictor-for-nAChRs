# -*- coding: utf-8 -*-
"""Append Run XXXII cells to results.ipynb (one-off recording helper)."""
import nbformat

NB = "results.ipynb"

header = """## Run XXXII: δ/γ/ε's 0.69 is biological, not a gene-identity artifact

Run XXXI found δ/γ/ε (CHRND/E/G, 115) scores 0.688/0.690 *above* the full model — but with only 3 genes,
the `subunit_*` one-hot collapses to a "δ vs ε vs γ" label that could carry direction almost for free. This
run settles it: drop the 16 `subunit_*` columns and re-run each group, matched with/without identity.

### Results at a glance (Δ = with − without gene identity)

| subset | n | genes | RF with | RF without | CatBoost with | CatBoost without |
|---|---|---|---|---|---|---|
| full | 760 | 16 | 0.6525 | 0.6441 | 0.6499 | 0.6388 |
| α | 517 | 9 | 0.6340 | 0.6379 | 0.6205 | 0.6200 |
| β | 128 | 4 | 0.6342 | 0.6143 | 0.6130 | 0.5994 |
| **δ/γ/ε** | 115 | 3 | 0.6879 | **0.6947** | 0.6900 | **0.7091** |
"""

setup = """### Setup — drop gene identity, keep everything else

- **Question:** is δ/γ/ε's 0.69 a *gene-identity* artifact (the model reading "which of 3 genes" as a
  free direction cue) or *biological* (direction genuinely more predictable in the muscle accessory subunits)?
- **Test:** remove the 16 `subunit_*` one-hot columns (positions 1–16 of the positional group), keeping
  `position_normalized` + `species_*` + all 24 physicochemical + 3 substitution + 3 conservation + 7
  structural + 2 ESM-2 + 16 ESM-embedding features → 82 → 66 features. Re-run standard CV, matched with vs
  without identity (same seeds, same folds), per group + full.
- **Why not leave-one-gene-out:** it is degenerate for δ/γ/ε — 3 genes, and holding out CHRNG gives a
  single-class (7 all-LOF) test fold, while holding out CHRND leaves GOF with only 3 training rows so the
  inner StratifiedKFold(5) crashes. Dropping gene identity is the clean, decisive substitute.
- **Protocol** matches Run XXXI: `cv_mode="standard"`, 3-class, RF + CatBoost, 3 seeds × 15 trials.
- **Sanity check:** full-with-identity reproduces the Run XXVIII reference exactly (RF 0.6525 / CatBoost
  0.6499), and α-with-identity reproduces Run XXXI (0.634 / 0.621) — the matched protocol is faithful.
"""

code = """import pandas as pd
from IPython.display import HTML, display

d = pd.read_csv("results/gene_identity_drop/gene_identity_drop_summary.csv")

labels = {
    "full": "Full (760)",
    "alpha": "α subunits (CHRNA1-7, 9, 10)",
    "beta": "β subunits (CHRNB1-4)",
    "delta_gamma_epsilon": "δ/γ/ε subunits (CHRND, E, G)",
}
mlab = {"random_forest": "RF", "catboost": "CatBoost"}

def _row(g, m):
    w = d[(d["group"] == g) & (d["model"] == m) & (d["gene_identity"])].iloc[0]
    wo = d[(d["group"] == g) & (d["model"] == m) & (~d["gene_identity"])].iloc[0]
    dlt = w["macro_f1_mean"] - wo["macro_f1_mean"]
    return (f"{w['macro_f1_mean']:.4f}", f"{wo['macro_f1_mean']:.4f}", f"{dlt:+.4f}")

css = ("<style>table{border-collapse:collapse;margin:0.4em 0 1.6em;font-size:0.92em}"
       "th,td{border:1px solid #bbb;padding:6px 12px;text-align:left}"
       "thead th{background:#eee}tbody tr:nth-child(even){background:#f8f8f8}"
       "td.dn{color:#0e7490;font-weight:600}td.up{color:#b45309}</style>")

rows = []
for g in ["full", "alpha", "beta", "delta_gamma_epsilon"]:
    n = int(d[d["group"] == g]["n"].iloc[0])
    for m in ["random_forest", "catboost"]:
        wv, wov, dlt = _row(g, m)
        cls = "up" if dlt.startswith("+") else ("dn" if dlt.startswith("-") else "")
        rows.append(
            f"<tr><td>{labels[g]}</td><td>{n}</td><td>{mlab[m]}</td>"
            f"<td>{wv}</td><td>{wov}</td><td class='{cls}'>{dlt}</td></tr>")

html = ("<h4>Gene-identity drop — Δ = with − without identity (negative ⇒ identity was not helping)</h4>"
        "<table><thead><tr><th>Subset</th><th>n</th><th>Model</th>"
        "<th>With identity</th><th>Without identity</th><th>Δ</th></tr></thead><tbody>"
        + "".join(rows) + "</tbody></table>")

display(HTML(css + html))
"""

takeaways = """### Run XXXII takeaways — δ/γ/ε's edge is BIOLOGICAL; the gene-identity confound is rejected

1. **δ/γ/ε survives — and even *improves* — without gene identity.** Dropping the 16 `subunit_*` columns,
   δ/γ/ε goes **RF 0.688 → 0.695 (+0.007)** and **CatBoost 0.690 → 0.709 (+0.019)**, with MCC up too
   (CatBoost 0.546 → 0.580). If the 0.69 were a "δ vs ε vs γ" lookup, removing the lookup would crash it;
   instead it rose. **The high score is carried by real features (structural/conservation/physicochemical),
   not the gene label.**
2. **The full model barely needs gene identity** (RF −0.008, CatBoost −0.011), and **α is flat**
   (−0.004 / 0.000). Only **β leans on it** (−0.020 / −0.014): with 4 β genes, "which β subunit" carries
   ~0.02 of β's score.
3. **Why δ/γ/ε goes *up* without identity:** in a 3-gene group the gene one-hot is not informative signal
   but a nuisance split the model has to route around — 3 constant columns competing with the real 66. For
   CatBoost especially, dropping them frees it to use the physicochemical/conservation signal cleanly.
4. **Read together with Run XXXI:** δ/γ/ε (the muscle *accessory* subunits — clean, well-characterised in
   congenital myasthenic syndromes) is where direction-of-effect is genuinely most predictable, and α/β are
   each harder than the pooled model. That is a *biological heterogeneity* result, not a modelling artifact.

**Verdict:** subunit type is a real axis — the muscle structural subunits are the most direction-predictable
slice, and that edge is intrinsic (it survives, and grows, when the model is forbidden from naming the gene).
Natural next step: a subunit-type-conditioned model that exploits δ/γ/ε's cleaner signal without hand-splitting
by gene.
"""

nb = nbformat.read(NB, as_version=4)
for src, kind in [(header, "markdown"), (setup, "markdown"),
                  (code, "code"), (takeaways, "markdown")]:
    nb.cells.append(nbformat.v4.new_code_cell(src) if kind == "code"
                    else nbformat.v4.new_markdown_cell(src))

nbformat.write(nb, NB)
print(f"appended 4 cells -> {len(nb.cells)} total")
