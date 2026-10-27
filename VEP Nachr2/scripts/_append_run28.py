# -*- coding: utf-8 -*-
"""Append Run XXVIII cells to results.ipynb (one-off recording helper)."""
import nbformat

NB = "results.ipynb"

header = """## Run XXVIII: per-feature structural ablation — gating signal lives in flexibility, not gate coupling

Run XXVII treated the structural features as whole groups. This run drops each of the 14 structural
columns **one at a time** (and re-drops the 3 whole groups as anchors) against a full 82-feature
reference (base 75 + gating 3 + allosteric 4), all under gene-aware CV (`cv_mode="standard"`), 3-class,
RF + CatBoost, 3 seeds × 15 trials.

Delta = (drop feature) − (full); **negative = the feature helps**.
"""

setup = """### Setup — 14 structural columns, 3 group anchors

- **Reference:** full 82-feat set (0.6525 RF / 0.6499 CatBoost — the carry-forward gene-aware + gating +
  interaction model, a hair above Run XXVII's +gating 0.651/0.648, within noise).
- **Per-feature drops:** structural_core {rsa, cbeta_density, b_factor, dssp_helix, dssp_sheet, dssp_coil,
  is_unmappable}, gating {gnm_msf, gnm_slowmode, gnm_gate_coupling}, allosteric {dvol, dhydro, dcharge,
  dpolarity}.
- **Group anchors:** drop whole structural_core / gating / allosteric, to read each feature against its group.
- **Noise floor:** full std ≈ ±0.007; individual-feature deltas of ±0.01–0.015 are ~1.5–2σ, so treat
  rank-order between adjacent features as suggestive, the group-level deltas as the robust signal.
"""

code = """import pandas as pd
from IPython.display import display

d = pd.read_csv("results/structural_feature_ablation/structural_feature_ablation_summary.csv")
view = d.copy()
for c in ["macro_f1_mean", "macro_f1_std", "mcc_mean", "balanced_accuracy_mean"]:
    view[c] = view[c].map(lambda x: f"{x:.4f}")
display(view)

piv = d.pivot_table(index="model", columns="ablation", values="macro_f1_mean")
print("\\nDelta macro F1 vs full (negative = feature helps):")
for model in ["random_forest", "catboost"]:
    full = piv.loc[model, "full"]
    print(f"\\n[{model}] full = {full:.4f}")
    deltas = {c: piv.loc[model, c] - full for c in piv.columns if c != "full"}
    for c in sorted(deltas, key=lambda c: deltas[c]):
        print(f"    {c:28s} {piv.loc[model, c]:.4f}  Δ={deltas[c]:+.4f}")
"""

takeaways = """### Run XXVIII takeaways — gating signal is flexibility, not gate coupling; interaction is redundant

1. **The gating group is the top structural signal — but `gnm_gate_coupling` itself is a dud.** Dropping
   the whole gating group costs −0.012 (RF) / −0.0095 (CatBoost) — the most (or near-most) important
   structural group in both models. Yet the *physically-motivated* feature, gate_coupling (Run XVIII/XXV's
   "coupling to the M2 9′ gate ring"), contributes almost nothing alone (−0.0005 / −0.0020). The group's
   value comes from `gnm_msf` (−0.0089/−0.0062) and `gnm_slowmode` (−0.0107/−0.0066) — i.e. **global
   residue flexibility** (mean-square fluctuation + slowest normal mode), not "how connected you are to
   the gate." The mechanistic story flips: what helps is *mobility* of the position, not its connectivity.

2. **The allosteric interaction terms are redundant with gating.** Run XXVII's coupling×Δchemistry added
   +0.005/+0.008 on the 75-feat baseline, but against the 82-feat reference (which already contains gating),
   dropping the whole interaction group costs only −0.0011/−0.0023 — and no individual term exceeds
   −0.008. The interaction carries no information beyond gating + physicochemical. The "indirect
   connection" hypothesis closes as a clean negative: the product term is not the mechanism.

3. **Core structural (structural_core) matters for RF, is ~neutral for CatBoost.** drop_group_structural_core
   = −0.014 (RF, its single biggest drop) vs −0.005 (CatBoost). RF leans on dssp_helix/coil, rsa, and
   is_unmappable (each −0.012 to −0.015); CatBoost shows near-zero or slightly *positive* deltas for
   dssp_helix/coil/b_factor. Structural signal is model-dependent and concentrated in RF — a robustness
   caveat for the thesis.

4. **`is_unmappable` is doing work it shouldn't.** The binary "no structure mapping" flag (a data-quality
   marker, not a physical property) is among the more important structural_core features (−0.012/−0.007).
   It correlates with which genes use AlphaFold-monomer models (CHRNA2/5/6/9/10, B3, G) vs experimental
   structures, so its contribution is plausibly **gene identity wearing a structural costume** — legitimate
   under gene-aware CV, but worth flagging so it isn't read as a structural mechanism.

**Verdict: structural features are individually weak and model-dependent; the only robust structural
signal is GNM flexibility (msf/slowmode), not gate coupling.** Carry-forward the full 82-feat gene-aware
model (0.6525/0.6499) but do not over-claim structural mechanism — the group deltas are ~1.5–2σ, and the
real lever remains more per-gene data.
"""

nb = nbformat.read(NB, as_version=4)
for src, kind in [(header, "markdown"), (setup, "markdown"),
                  (code, "code"), (takeaways, "markdown")]:
    nb.cells.append(nbformat.v4.new_code_cell(src) if kind == "code"
                    else nbformat.v4.new_markdown_cell(src))

nbformat.write(nb, NB)
print(f"appended 4 cells -> {len(nb.cells)} total")
