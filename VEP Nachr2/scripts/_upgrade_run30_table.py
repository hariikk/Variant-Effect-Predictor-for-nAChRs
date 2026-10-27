# -*- coding: utf-8 -*-
"""One-off: upgrade Run XXX recording in results.ipynb to render nice tables.

Replaces the raw `display(view)` in the Run XXX code cell with two styled HTML
tables (size curve + balance), each with a Δ-vs-full column, and adds a compact
"results at a glance" markdown table to the Run XXX header cell.
"""
import nbformat

NB = "results.ipynb"

# ---------------------------------------------------------------------------
# Markdown quick-reference table appended to the header cell.
# ---------------------------------------------------------------------------
glance_table = """

### Results at a glance

| panel | condition | n | RF macro F1 | CatBoost macro F1 | Δ RF | Δ CatBoost |
|---|---|---|---|---|---|---|
| SIZE | quarter (~196, 50/25/25) | 196 | 0.545 | 0.531 | −0.108 | −0.119 |
| SIZE | half (~385, 50/25/25) | 385 | 0.561 | 0.587 | −0.091 | −0.063 |
| SIZE | **full (760, 50/25/25)** | 760 | **0.653** | **0.650** | — | — |
| BALANCE | random 50/25/25 (350) | 350 | 0.583 | 0.586 | −0.069 | −0.064 |
| BALANCE | random 33/33/33 (350) | 350 | 0.605 | 0.593 | −0.047 | −0.057 |
| BALANCE | **gene-stratified 33/33/33 (348)** | 348 | **0.611** | **0.610** | −0.042 | −0.040 |
"""

# ---------------------------------------------------------------------------
# New code cell source — two styled HTML tables with Δ vs full.
# ---------------------------------------------------------------------------
new_code = '''import pandas as pd
from IPython.display import HTML, display

d = pd.read_csv("results/data_size/data_size_summary.csv")
FULL = {"random_forest": 0.6525, "catboost": 0.6499}
ROWS = {(r["condition"], r["model"]): r for _, r in d.iterrows()}

def _cell(model, cond):
    """Return (value_with_std, delta_vs_full) for one model/condition."""
    if cond == "full":
        return f"{FULL[model]:.4f}", "—"
    r = ROWS[(cond, model)]
    val = f"{r['macro_f1_mean']:.4f} ± {r['macro_f1_std']:.3f}"
    dlt = f"{r['macro_f1_mean'] - FULL[model]:+.4f}"
    return val, dlt

def _table(title, conds, labels, n_of):
    out = [f"<h4>{title}</h4><table><thead><tr>",
           "<th>Condition</th><th>n</th>",
           "<th>RF macro F1 (mean ± std)</th><th>CatBoost macro F1 (mean ± std)</th>",
           "<th>Δ RF</th><th>Δ CatBoost</th></tr></thead><tbody>"]
    for c in conds:
        rf_v, rf_d = _cell("random_forest", c)
        cb_v, cb_d = _cell("catboost", c)
        out.append(f"<tr><td>{labels[c]}</td><td>{n_of[c]}</td>"
                   f"<td>{rf_v}</td><td>{cb_v}</td>"
                   f"<td>{rf_d}</td><td>{cb_d}</td></tr>")
    out.append("</tbody></table>")
    return "".join(out)

labels = {
    "quarter_502525": "Quarter (~196, 50/25/25)",
    "half_502525": "Half (~385, 50/25/25)",
    "full": "Full (760, 50/25/25)",
    "n350_502525": "Random 50/25/25 (350)",
    "n350_333333": "Random 33/33/33 (350)",
    "n350_333333_gene": "Gene-stratified 33/33/33 (348)",
}
n_of = {
    "quarter_502525": 196, "half_502525": 385, "full": 760,
    "n350_502525": 350, "n350_333333": 350, "n350_333333_gene": 348,
}

css = ("<style>table{border-collapse:collapse;margin:0.4em 0 1.6em;font-size:0.92em}"
       "th,td{border:1px solid #bbb;padding:6px 12px;text-align:left}"
       "thead th{background:#eee}tbody tr:nth-child(even){background:#f8f8f8}"
       "td:nth-child(5),td:nth-child(6){color:#666}</style>")

display(HTML(css))
display(HTML(_table("Size curve — fixed ~50/25/25 balance (Δ vs full)",
                    ["quarter_502525", "half_502525", "full"], labels, n_of)))
display(HTML(_table("Class balance — fixed N≈350 (Δ vs full)",
                    ["n350_502525", "n350_333333", "n350_333333_gene"], labels, n_of)))
'''

nb = nbformat.read(NB, as_version=4)

n_header = n_code = 0
for c in nb.cells:
    src = "".join(c.source)
    if c.cell_type == "markdown" and src.startswith("## Run XXX:"):
        if "Results at a glance" not in src:
            c.source = src + glance_table
            n_header += 1
    if c.cell_type == "code" and 'read_csv("results/data_size/data_size_summary.csv")' in src:
        c.source = new_code
        n_code += 1

nbformat.write(nb, NB)
print(f"updated: {n_header} header cell(s), {n_code} code cell(s)")
