# -*- coding: utf-8 -*-
"""Remove Run XXXVII (structural-region) from results.ipynb and insert the two
un-numbered probes (Idea A/B + α7 resting) chronologically before Run XXXV."""
import nbformat

NB = "results.ipynb"

cells = []

# ----------------------------- Idea A/B -----------------------------
cells.append(("markdown", """## Idea A/B: pathogenicity & disease axes are orthogonal to direction (AlphaMissense probe + per-class F1)

Two "free data" ideas, both dead ends on the orthogonality of direction (cf. Run VIII). *Idea A*: add
AlphaMissense pathogenicity as one more feature (82 → 83). *Idea B*: expand labels by mining disease
associations (disease → mechanism → direction). A companion per-class F1 decomposes *why* δ/γ/ε wins.

Gene-aware `cv_mode="standard"`, 3-class, RF + CatBoost, 3 seeds × 15 trials, 82-feat baseline, full 760.
Ran after Run XXXIV, before Run XXXV.
"""))

cells.append(("code", """import json
import pandas as pd
from IPython.display import display

# Part A — AlphaMissense probe (82 vs 82+AM)
pa = json.load(open("results/idea_a/part_a_am_probe.json"))
rows = []
for k, v in pa.items():
    label, model = k.split("|")
    rows.append({"condition": label, "model": model,
                 "macro_f1_mean": round(sum(v) / len(v), 4),
                 "macro_f1_std": round(pd.Series(v).std(), 4)})
a = pd.DataFrame(rows).pivot(index="model", columns="condition", values="macro_f1_mean")
print("Part A — AlphaMissense probe (macro F1):")
display(a.round(4))
print("Δ (82+AM − 82):")
for mo in a.index:
    print(f"  [{mo:15s}] {a.loc[mo, '82']:.4f} -> {a.loc[mo, '82+AM']:.4f}  Δ={a.loc[mo, '82+AM'] - a.loc[mo, '82']:+.4f}")

# Part B — per-class F1 per group (82-feat baseline)
pb = json.load(open("results/idea_a/part_b_perclass_f1.json"))
print("\\nPart B — per-class F1 per group (82-feat baseline):")
for g in ["alpha", "beta", "delta_gamma_epsilon"]:
    for mo in ["random_forest", "catboost"]:
        d = pb[f"{g}|{mo}"]
        print(f"  [{g:20s} {mo:15s}] macro {sum(d['macro'])/3:.4f} | LOF {sum(d['LOF'])/3:.4f} | "
              f"NNE {sum(d['NNE'])/3:.4f} | GOF {sum(d['GOF'])/3:.4f}")
"""))

cells.append(("markdown", """### Idea A/B takeaways — pathogenicity is orthogonal, disease is tautological, and δ/γ/ε's edge is LOF+GOF-specific

1. **Idea A (AlphaMissense as a feature): null.** 82 → 83 features: RF 0.6525→0.6445 (Δ −0.008),
   CatBoost 0.6499→0.6514 (Δ +0.0015) — both within noise. Re-confirms Run VIII: pathogenicity is
   orthogonal to direction even as a single added feature. (An earlier 1-seed smoke read "+0.024"; that was
   variance.)
2. **Idea B (disease → direction): tautological, not additive.** On the clean-label variants, 66% have NO
   pathology annotation, and the only ≥80%-pure pathology buckets are the myasthenic syndromes whose *name
   is the direction label* — CMS (86, 86% LOF), SCCMS (35, 94% GOF), FCCMS (6, 100% LOF). The abundant
   common-disease tags (nicotine dependence 41% GOF, ADNFLE 52% GOF, schizophrenia 79% LOF) are
   direction-mixed. Mining disease would only re-derive labels we already have, not expand GOF/NNE.
3. **δ/γ/ε's edge is LOF+GOF-specific, not all-class.** Per-class: δγε LOF 0.83/0.82 + GOF 0.66/0.67, but
   NNE only 0.58/0.58 — the WORST NNE of the three groups (β 0.69/0.65, α 0.61/0.60). So δγε wins because
   its two directional classes are clean, not because everything separates. Refines Runs XXXII–XXXIV.
"""))

# ----------------------------- α7 resting -----------------------------
cells.append(("markdown", """## α7 resting-structure probe: complete homomeric resting model vs 7EKT — null-to-negative

The colleague's sodium-channel precedent says a *more complete* structure improves F1. Tested by swapping
CHRNA7's structural source from 7EKT (antagonist-bound, 3.20 Å) to a new complete homomeric α7 resting model
(7KOO cryo-EM + 7RPM NMR, ligands/ions stripped, full-atom), ingested as `data/raw/structure_files/a7rest/`.
Only CHRNA7's 14 PDB features change; the other 15 genes and 68 non-structural features are identical.

Gene-aware `cv_mode="standard"`, 3-class, RF + CatBoost, 5 seeds × 15 trials, 82-feat layout, full 760.
`naive` = all 14 PDB features from a7rest; `geom-only` = geometry/dynamics (RSA/Cβ/GNM/allosteric) from
a7rest but b_factor + DSSP held at 7EKT (the spliced model has no per-atom B-factors or SS records). Ran
after Run XXXIV, before Run XXXV.
"""))

cells.append(("code", """import json
import numpy as np
import pandas as pd
from IPython.display import display

c = json.load(open("results/a7resting/comparison.json"))
g = json.load(open("results/a7resting/geomonly.json"))

def m(v): return float(np.mean(v))
def s(v): return float(np.std(v))

rows = []
for arm, keyfn in [("7EKT", lambda mo: c[f"7EKT|{mo}"]),
                   ("a7rest (naive)", lambda mo: c[f"a7rest|{mo}"]),
                   ("geom-only", lambda mo: g[mo])]:
    for mo in ["random_forest", "catboost"]:
        d = keyfn(mo)
        rows.append({"arm": arm, "model": mo,
                     "macro": f"{m(d['macro']):.4f} ± {s(d['macro']):.4f}",
                     "LOF": round(m(d["LOF"]), 3), "NNE": round(m(d["NNE"]), 3),
                     "GOF": round(m(d["GOF"]), 3)})
display(pd.DataFrame(rows).set_index(["arm", "model"]))

print("Δ vs 7EKT (macro F1):")
for mo in ["random_forest", "catboost"]:
    base = m(c[f"7EKT|{mo}"]["macro"])
    print(f"  [{mo:15s}] naive {m(c[f'a7rest|{mo}']['macro']) - base:+.4f}   "
          f"geom-only {m(g[mo]['macro']) - base:+.4f}")
"""))

cells.append(("markdown", """### α7 resting-structure takeaways — "more complete" does not transfer; keep 7EKT

1. **Null-to-negative, not a win.** Naive swap: RF −0.0114 (p=0.061), CatBoost −0.0076 (p=0.18).
   Geometry-only arm (isolating the real geometry from the B-factor/SS metadata confound): RF −0.0117
   (p=0.12), CatBoost +0.0030 (p=0.51). No sodium-channel-style improvement.
2. **The negative is RF-specific and GOF-specific.** RF GOF drops −0.026 (naive) / −0.019 (geom-only); LOF
   and NNE are flat, and CatBoost is flat throughout. So RF's penalty is a real geometry/dynamics effect,
   not the metadata collapse.
3. **Two confounds in the spliced model (not code):** b_factor → all 0.00 (no per-atom B-factors) and DSSP
   → all coil (no SS records). These explain CatBoost's naive drop (geometry-only recovers it) but not RF's
   (stays −0.012).
4. **Why geometry hurts: `gnm_slowmode` (global gating mode) nearly decorrelates between 7EKT and the
   resting state (corr −0.03).** The apo resting state's slowest mode is *less* informative about GOF
   (activated-state variants) than the antagonist-bound structure. The "completeness" gain is trivial (only
   resolves 2 residues). Keep 7EKT for CHRNA7.
"""))

nb = nbformat.read(NB, as_version=4)

# 1. Remove the three Run XXXVII (structural-region) cells.
removed = 0
for i in range(len(nb.cells) - 1, -1, -1):
    src = "".join(nb.cells[i].source)
    if src.startswith("## Run XXXVII: structural-region") or \
       src.startswith("### Run XXXVII takeaways") or \
       ("results/structural_region/structural_region_summary.csv" in src and
        nb.cells[i].cell_type == "code"):
        del nb.cells[i]
        removed += 1
print(f"removed {removed} Run XXXVII cells")

# 2. Insert the two probes immediately before the "## Run XXXV" header cell.
insert_at = None
for i, cell in enumerate(nb.cells):
    if "".join(cell.source).startswith("## Run XXXV"):
        insert_at = i
        break
assert insert_at is not None, "Run XXXV header not found"
for offset, (kind, src) in enumerate(cells):
    new = (nbformat.v4.new_code_cell(src) if kind == "code"
           else nbformat.v4.new_markdown_cell(src))
    nb.cells.insert(insert_at + offset, new)

nbformat.write(nb, NB)
print(f"inserted {len(cells)} cells at {insert_at} -> {len(nb.cells)} total")
