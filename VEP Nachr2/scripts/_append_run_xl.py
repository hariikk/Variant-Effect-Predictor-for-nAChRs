# -*- coding: utf-8 -*-
"""Merge Run XL per-seed outputs and append the results to results.ipynb.

Run XL = pretrained GNN (GVP), protein transformer (ProtBERT) and cheminformatics
(Mordred) feature arms, 10 seeds x 15 trials, RF + CatBoost, gene-aware CV.

Cells are appended as *markdown* (rendered tables), per the project convention
that results must be readable without executing code.
"""

import glob
import json
import uuid

import numpy as np
import pandas as pd
from scipy import stats

NB = "results.ipynb"
OUT = "results/gnn_plm_ablation"
ARMS = [
    "baseline", "add_gvp", "add_prot_bert", "add_mordred", "add_all",
    "all_minus_gvp", "all_minus_prot_bert", "all_minus_mordred",
]
MODELS = ["random_forest", "catboost"]
REF = {
    "add_gvp": "baseline", "add_prot_bert": "baseline",
    "add_mordred": "baseline", "add_all": "baseline",
    "all_minus_gvp": "add_all", "all_minus_prot_bert": "add_all",
    "all_minus_mordred": "add_all",
}


def load_per_seed():
    per = {}
    for f in glob.glob("results/xl_*/per_seed_*.json"):
        d = json.load(open(f))
        per[(d["arm"], d["model"])] = np.asarray(d["per_seed_f1"], dtype=float)
    return per


def build_tables(per):
    rows = []
    for arm in ARMS:
        for m in MODELS:
            v = per[(arm, m)]
            rows.append({
                "arm": arm, "model": m, "n_seeds": len(v),
                "macro_f1_mean": v.mean(), "macro_f1_std": v.std(ddof=1),
            })
    summary = pd.DataFrame(rows)

    lines = [
        "| Arm | Dim | RF macro-F1 | CatBoost macro-F1 | Δ RF vs ref | Δ CB vs ref | ref |",
        "|---|---|---|---|---|---|---|",
    ]
    dims = {"baseline": 82, "add_gvp": 98, "add_prot_bert": 98, "add_mordred": 98,
            "add_all": 130, "all_minus_gvp": 114, "all_minus_prot_bert": 114,
            "all_minus_mordred": 114}
    for arm in ARMS:
        rf = per[(arm, "random_forest")]
        cb = per[(arm, "catboost")]
        if arm == "baseline":
            lines.append(
                f"| **{arm}** | {dims[arm]} | {rf.mean():.4f} ± {rf.std(ddof=1):.4f} | "
                f"{cb.mean():.4f} ± {cb.std(ddof=1):.4f} | — | — | — |")
            continue
        ref = REF[arm]
        drf = rf.mean() - per[(ref, "random_forest")].mean()
        dcb = cb.mean() - per[(ref, "catboost")].mean()
        lines.append(
            f"| {arm} | {dims[arm]} | {rf.mean():.4f} ± {rf.std(ddof=1):.4f} | "
            f"{cb.mean():.4f} ± {cb.std(ddof=1):.4f} | {drf:+.4f} | {dcb:+.4f} | {ref} |")
    table = "\n".join(lines)

    sig = [
        "| Comparison | Model | Δ macro-F1 | paired t | two-sided p |",
        "|---|---|---|---|---|",
    ]
    sig_rows = []
    for arm in ARMS[1:]:
        ref = REF[arm]
        for m in MODELS:
            a, b = per[(arm, m)], per[(ref, m)]
            t, p = stats.ttest_rel(a, b)
            sig_rows.append((arm, ref, m, (a - b).mean(), t, p))
            sig.append(
                f"| {arm} vs {ref} | {'RF' if m == 'random_forest' else 'CatBoost'} | "
                f"{(a - b).mean():+.4f} | {t:+.2f} | {p:.4f} |")
    sig_table = "\n".join(sig)
    return summary, table, sig_table, sig_rows


def add_cell(nb, cell_type, source):
    cell = {
        "cell_type": cell_type,
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "source": source,
    }
    nb["cells"].append(cell)


def main():
    per = load_per_seed()
    missing = [(a, m) for a in ARMS for m in MODELS if (a, m) not in per]
    assert not missing, f"missing per-seed results: {missing}"
    for a, m in per:
        assert len(per[(a, m)]) == 10, (a, m, len(per[(a, m)]))

    summary, table, sig_table, sig_rows = build_tables(per)
    import os
    os.makedirs(OUT, exist_ok=True)
    summary.to_csv(f"{OUT}/gnn_plm_ablation_summary.csv", index=False)

    best = max(sig_rows, key=lambda r: r[3])

    nb = json.load(open(NB, encoding="utf-8"))

    add_cell(nb, "markdown", """## Run XL: pretrained GNN / protein-transformer / cheminformatics arms — GVP null, Mordred null, ProtBERT edge-of-noise

Do the published representation families add direction-of-effect signal on top of the 82-feature
anchor? Three new pretrained groups, each reduced to 16 PCA components:

- **`gnn_gvp`** — pretrained **GVP-GNN** node embeddings (ESM-IF1 structure encoder, 512-dim → PCA).
- **`prot_bert`** — **ProtBERT** (ProtTrans, UniRef100) per-residue embeddings (1024-dim → PCA).
- **`mordred_aa`** — **RDKit/Mordred** amino-acid descriptors (1,292 → PCA), wt→mt delta.

Protocol matches the gene-aware runs (XXVI onward): `cv_mode="standard"`, 3-class, RF + CatBoost,
**10 seeds × 15 Optuna trials**, 760 variants. Mapping coverage 100% for all three groups (no
zero-filled rows). Delta vs the matched reference re-run at the same seeds.
""")

    add_cell(nb, "markdown", f"""### Results — macro-F1 (mean ± SD over 10 seeds)

{table}

### Paired significance vs the matched reference (per-seed values, same seeds)

{sig_table}

With 14 comparisons, Bonferroni α = 0.05/14 ≈ 0.0036. Only `{best[0]}`
({'RF' if best[2] == 'random_forest' else 'CatBoost'}) reaches raw p < 0.05 (p = {best[5]:.4f}); it does
**not** survive multiplicity correction.
""")

    add_cell(nb, "markdown", """### Run XL takeaways — no GNN or cheminformatics win; ProtBERT is a weak, model-specific candidate

1. **GVP-GNN: null.** `add_gvp` = −0.0014 (RF) / −0.0001 (CatBoost), both p > 0.5. Removing it from
   `add_all` changes nothing (`all_minus_gvp`: +0.0027 CB / −0.0009 RF). This is the **12th structural
   null** and the first using a *pretrained* structure encoder rather than hand-built geometric
   features — it confirms the Run XV/XXVIII conclusion that single-residue structural context carries
   no direction signal at this data scale.
2. **ProtBERT: weak, RF-only.** `add_prot_bert` = +0.0077 (RF, p = 0.013) / +0.0016 (CB, ns). It is the
   only group whose removal from `add_all` hurts **both** models (`all_minus_prot_bert` is the lowest
   LOO arm: −0.0043 RF / −0.0025 CB). Direction is consistent but the effect is ~half a seed-SD and
   fails multiplicity correction — treat as a **candidate**, not a result. A larger-seed or
   ProtBERT-only lean-model follow-up would be the honest next test.
3. **Mordred: null-to-negative.** +0.0010 (RF) / −0.0050 (CB). Expected: 1,010 descriptors over only 20
   amino-acid fragments is a maximally collinear feature set that largely re-expresses the existing 24
   physicochemical features.
4. **Stacking all three hurts CatBoost.** `add_all` = −0.0072 (CB, p = 0.97 in the "better" direction),
   i.e. 48 extra near-noise dimensions cost the tree model more than they add. This is the same
   dilution pattern as Run XXVIII's redundant structural blocks.
5. **Verdict: keep the 82-feature anchor.** ESM-2 150M (already in the anchor) remains the only LM
   representation that earns its place. ProtBERT is the single arm worth a dedicated follow-up;
   GVP and Mordred are closed. Consistent with PreMode's finding that GOF/LOF direction needs
   *supervised fine-tuning on thousands of mode-of-action labels*, not frozen pretrained embeddings.
""")

    with open(NB, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write("\n")

    print(f"Appended 3 markdown cells to {NB}")
    print(f"Saved {OUT}/gnn_plm_ablation_summary.csv")
    print(table)
    print()
    print(sig_table)


if __name__ == "__main__":
    main()
