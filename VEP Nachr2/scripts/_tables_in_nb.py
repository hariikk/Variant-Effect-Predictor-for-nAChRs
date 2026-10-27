# -*- coding: utf-8 -*-
"""Convert the code cells in results.ipynb that merely display CSVs/JSONs into
rendered markdown tables (no code to run). Replaces 5 code cells."""
import json
import numpy as np
import pandas as pd
import nbformat

NB = "results.ipynb"


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join([" --- "] * len(header)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out)


def f4(x):
    return f"{x:.4f}"


# ── Run XXXV: functional ────────────────────────────────────────────────
d = pd.read_csv("results/functional/functional_summary.csv")
rows = [[r.condition, r.model, int(r.n_features), f4(r.macro_f1_mean),
         f4(r.mcc_mean), f4(r.balanced_accuracy_mean)] for r in d.itertuples()]
functional_md = (md_table(["Condition", "Model", "n_feat", "Macro F1", "MCC", "Bal acc"], rows)
                 + "\n\n*(raw: `results/functional/functional_summary.csv`)*")

# ── Run XXXVI: structure gating ────────────────────────────────────────
d = pd.read_csv("results/structure_gating/structure_gating_summary.csv")
rows = [[r.condition, r.model, int(r.n), f4(r.macro_f1_mean),
         f4(r.mcc_mean), f4(r.balanced_accuracy_mean)] for r in d.itertuples()]
structure_gating_md = (md_table(["Condition", "Model", "n", "Macro F1", "MCC", "Bal acc"], rows)
                       + "\n\n*(raw: `results/structure_gating/structure_gating_summary.csv`)*")

# ── Idea A/B ───────────────────────────────────────────────────────────
pa = json.load(open("results/idea_a/part_a_am_probe.json"))
rows = []
for k in ["82|random_forest", "82|catboost", "82+AM|random_forest", "82+AM|catboost"]:
    label, model = k.split("|")
    v = pa[k]
    rows.append([label, model, f"{np.mean(v):.4f} ± {np.std(v):.4f}"])
am_md = ("**Part A — AlphaMissense probe (macro F1):**\n\n"
         + md_table(["Condition", "Model", "Macro F1"], rows))

pb = json.load(open("results/idea_a/part_b_perclass_f1.json"))
grows = []
for g, glabel in [("alpha", "α (517)"), ("beta", "β (128)"),
                  ("delta_gamma_epsilon", "δ/γ/ε (115)")]:
    for mo in ["random_forest", "catboost"]:
        d = pb[f"{g}|{mo}"]
        grows.append([glabel, mo, f"{np.mean(d['macro']):.4f}",
                      f"{np.mean(d['LOF']):.4f}", f"{np.mean(d['NNE']):.4f}",
                      f"{np.mean(d['GOF']):.4f}"])
pb_md = ("**Part B — per-class F1 per group (82-feat baseline):**\n\n"
         + md_table(["Group", "Model", "Macro", "LOF", "NNE", "GOF"], grows))
idea_md = am_md + "\n\n" + pb_md + "\n\n*(raw: `results/idea_a/`)*"

# ── α7 resting ─────────────────────────────────────────────────────────
c = json.load(open("results/a7resting/comparison.json"))
g = json.load(open("results/a7resting/geomonly.json"))
arms = [("7EKT", lambda mo: c[f"7EKT|{mo}"]),
        ("a7rest (naive)", lambda mo: c[f"a7rest|{mo}"]),
        ("geom-only", lambda mo: g[mo])]
rows = []
for arm, keyfn in arms:
    for mo in ["random_forest", "catboost"]:
        d = keyfn(mo)
        rows.append([arm, mo, f"{np.mean(d['macro']):.4f} ± {np.std(d['macro']):.4f}",
                     f"{np.mean(d['LOF']):.3f}", f"{np.mean(d['NNE']):.3f}",
                     f"{np.mean(d['GOF']):.3f}"])
delta_lines = []
for mo in ["random_forest", "catboost"]:
    base = np.mean(c[f"7EKT|{mo}"]["macro"])
    dn = np.mean(c[f"a7rest|{mo}"]["macro"]) - base
    dg = np.mean(g[mo]["macro"]) - base
    delta_lines.append(f"- **{mo}** naive Δ {dn:+.4f} · geom-only Δ {dg:+.4f}")
a7rest_md = (md_table(["Arm", "Model", "Macro F1", "LOF", "NNE", "GOF"], rows)
             + "\n\n**Δ vs 7EKT (macro F1):**\n" + "\n".join(delta_lines)
             + "\n\n*(raw: `results/a7resting/`)*")

# ── Run XXXVIII: class-wise F1 ─────────────────────────────────────────
d = pd.read_csv("results/classwise_f1/classwise_f1_summary.csv")
group_labels = {"all": "all (n=760)", "alpha": "α (n=517)", "beta": "β (n=128)",
                "delta_gamma_epsilon": "δ/γ/ε (n=115)"}
model_order = ["majority_class", "uniform_random", "stratified_random",
               "logistic_regression", "svm_rbf", "random_forest", "lightgbm",
               "svm_linear", "knn", "xgboost", "catboost", "mlp", "gaussian_nb", "ordinal"]
parts = []
for gname, glabel in group_labels.items():
    sub = d[d["group"] == gname].set_index("model")
    rows = []
    for mo in model_order:
        r = sub.loc[mo]
        rows.append([mo, f"{r['macro_f1_mean']:.3f}", f"{r['f1_LOF_mean']:.3f}",
                     f"{r['f1_No net effect_mean']:.3f}", f"{r['f1_GOF_mean']:.3f}"])
    parts.append(f"**{glabel}** — rows 0–2 are baselines (majority / uniform / stratified):\n\n"
                 + md_table(["Model", "Macro F1", "LOF", "NNE", "GOF"], rows))
classwise_md = "\n\n".join(parts) + "\n\n*(raw: `results/classwise_f1/classwise_f1_summary.csv`)*"

# ── Replace code cells with markdown-table cells ───────────────────────
replacements = [
    ("results/functional/functional_summary.csv", functional_md),
    ("results/structure_gating/structure_gating_summary.csv", structure_gating_md),
    ("results/idea_a/part_a_am_probe.json", idea_md),
    ("results/a7resting/comparison.json", a7rest_md),
    ("results/classwise_f1/classwise_f1_summary.csv", classwise_md),
]

nb = nbformat.read(NB, as_version=4)
done = 0
for i, cell in enumerate(nb.cells):
    if cell.cell_type != "code":
        continue
    src = "".join(cell.source)
    for marker, md in replacements:
        if marker in src:
            nb.cells[i] = nbformat.v4.new_markdown_cell(md)
            done += 1
            break
nbformat.write(nb, NB)
print(f"converted {done} code cells -> markdown tables ({len(nb.cells)} total cells)")
