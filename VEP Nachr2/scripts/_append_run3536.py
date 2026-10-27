# -*- coding: utf-8 -*-
"""Append Run XXXV (functional GO embedding) + XXXVI (structural gating) to results.ipynb."""
import nbformat

NB = "results.ipynb"

cells = []

# ----------------------------- Run XXXV -----------------------------
cells.append(("markdown", """## Run XXXV: functional (GO) gene embedding — function is a partial substitute, not a free lunch

Does a *functional* representation of each subunit (α/β/δγε class + mechanistic attributes:
homomeric, high-Ca2+, muscle-type, gene length) substitute for the 16 opaque gene one-hots, and do the
mechanistic attributes add signal on top of identity? (Func2Seq/Func2Prot framing.)

Gene-aware `cv_mode="standard"`, 3-class, RF + CatBoost, 3 seeds × 15 trials, 82-feat reference
(= 75 base + gating 3 + allosteric 4). `full` = 82, `functional` = 89 (+7), `no_identity` = 66 (−16 one-hots),
`functional_no_identity` = 73 (−16 +7). Delta vs `full`.
"""))

cells.append(("code", """import pandas as pd
from IPython.display import display

d = pd.read_csv("results/functional/functional_summary.csv")
view = d.copy()
for c in ["macro_f1_mean", "macro_f1_std", "mcc_mean", "balanced_accuracy_mean"]:
    view[c] = view[c].map(lambda x: f"{x:.4f}")
display(view)

piv = d.pivot_table(index="model", columns="condition", values="macro_f1_mean")
print("\\nDelta macro F1 vs full:")
for m in ["random_forest", "catboost"]:
    b = piv.loc[m, "full"]
    print(f"\\n[{m}] full = {b:.4f}")
    for c in ["functional", "no_identity", "functional_no_identity"]:
        print(f"    {c:24s} {piv.loc[m, c]:.4f}  Δ={piv.loc[m, c]-b:+.4f}")
"""))

cells.append(("markdown", """### Run XXXV takeaways — functional encoding is null as an augment, ~50–80% substitute when identity is dropped

1. **Augment (add functional on top of identity): null.** `functional` = +0.0034 (RF) / +0.0016 (CatBoost),
   well within the ±0.006–0.007 noise. The mechanistic attributes (homomeric, high-Ca2+, muscle-type, gene
   length) carry **no direction signal beyond what the one-hots already capture** — the α/β/δγε classes are
   near-reconstructible from the one-hots.
2. **Dropping identity is surprisingly cheap.** `no_identity` = −0.0084 (RF) / −0.0111 (CatBoost) — *not* the
   −0.13 gene-agnostic gap from Run XXVI. Under gene-leaky `cv_mode="standard"`, the model reconstructs gene
   identity from gene-correlated features (structural PDB chain, ESM-2, position-normalized, gene length)
   anyway. The 0.50 → 0.63 jump was a **CV-protocol effect, not a one-hot-column effect.**
3. **Function partially substitutes for identity.** `functional_no_identity` recovers ~44% (RF: 0.6441→0.6478)
   / ~80% (CatBoost: 0.6388→0.6477) of the small identity drop. So the coarse functional classes *do* carry a
   sliver of the gene signal — but not enough to matter.
4. **Verdict: not worth adding.** Keep the 16 one-hots (they're nearly free), skip the GO encoding. Consistent
   with "direction-of-effect is gene-conditioned" (Run XXVI): identity matters, but static functional classes
   add no independent signal. 12th hypothesis that doesn't move the needle.
"""))

# ----------------------------- Run XXXVI -----------------------------
cells.append(("markdown", """## Run XXXVI: structural-alignment gating of cross-species rows — disordered helps *more*, but it's a gene confound

EnergyFingerprint predicts zero-shot cross-gene transfer **degrades at disordered regions** (ordered should
transfer better). Run XXIX says the +0.05 cross-species gain is *label rebalancing*, not species diversity.
Test it directly: split the 253 non-human (mouse/rat) rows by orderedness of the human-homologous residue
(DSSP helix/sheet = ordered, 146; coil/unmappable = disordered, 107) and compare each against human-only.

Gene-aware `cv_mode="standard"`, 3-class, RF + CatBoost, 3 seeds × 15 trials, 82-feat set. Delta vs
`human_only` (507 = 0.5987 / 0.6038).
"""))

cells.append(("code", """import pandas as pd
from IPython.display import display

d = pd.read_csv("results/structure_gating/structure_gating_summary.csv")
view = d.copy()
for c in ["macro_f1_mean", "macro_f1_std", "mcc_mean", "balanced_accuracy_mean"]:
    view[c] = view[c].map(lambda x: f"{x:.4f}")
display(view)

piv = d.pivot_table(index="model", columns="condition", values="macro_f1_mean")
print("\\nDelta macro F1 vs human_only (positive = added rows help):")
for m in ["random_forest", "catboost"]:
    b = piv.loc[m, "human_only"]
    print(f"\\n[{m}] human_only = {b:.4f}")
    for c in ["human_ordered", "human_disordered", "all_three"]:
        print(f"    {c:18s} {piv.loc[m, c]:.4f}  Δ={piv.loc[m, c]-b:+.4f}")
"""))

cells.append(("markdown", """### Run XXXVI takeaways — structure does NOT gate cross-species transfer (but the "disordered" edge is a gene confound)

1. **The augmentation gain is uneven — and *disordered* rows help ~2.5× more per row.** Ordered (146 rows)
   +0.0165/+0.0107; disordered (107 rows) +0.0453/+0.0407 (RF/CB). Fewer disordered rows, larger gain.
2. **This rejects EnergyFingerprint's structural-transfer prior.** If transfer degraded at disordered
   residues, ordered rows should have helped *more*. They didn't — alignment quality does not gate the gain.
3. **But "disordered" is confounded with gene, not structure.** Disordered non-human ≈ α4β2 family
   (CHRNA4 34, CHRNB2 14, CHRNB4 10 of 107); ordered ≈ α7/α1 (CHRNA7 58, CHRNA1 21 of 146). Both species sit
   in both bins, so species is clean — the confound is *which genes* are in each bin. So "disordered helps
   more" is really **"α4β2-family non-human rows help more"** — reinforcing Run XXIX (label rebalancing) and
   Run XXXII (the edge is biological, not identity/structure).
4. **Verdict: structure is not the transfer mechanism.** The cross-species gain tracks gene/label content,
   not structural orderedness. Closes the "EnergyFingerprint-style structural gating" line; the +0.05 stands
   as label rebalancing.
"""))

nb = nbformat.read(NB, as_version=4)
for kind, src in cells:
    nb.cells.append(nbformat.v4.new_code_cell(src) if kind == "code"
                    else nbformat.v4.new_markdown_cell(src))
nbformat.write(nb, NB)
print(f"appended {len(cells)} cells -> {len(nb.cells)} total")
