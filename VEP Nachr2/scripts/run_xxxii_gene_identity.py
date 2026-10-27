"""
Run XXXII: is δ/γ/ε's 0.69 a gene-identity artifact?

Run XXXI found δ/γ/ε (CHRND/E/G, 115 variants) scores 0.688/0.690 alone — *above* the
full 16-gene model (0.653/0.650), on the smallest group. The follow-up question is whether
that edge is biological ("direction is intrinsically easier in the muscle accessory subunits")
or an artifact of gene identity: with only 3 genes, the `subunit_*` one-hot collapses to a
"δ vs ε vs γ" label that carries direction almost for free (e.g. CHRNG is all-LOF).

The decisive test is NOT leave-one-gene-out — that is degenerate here. With 3 genes,
holding out CHRNG gives a single-class (all-LOF) test fold, and holding out CHRND leaves
GOF with only 3 training rows so the inner StratifiedKFold(5) crashes. So instead we drop
gene identity directly: remove the 16 `subunit_*` columns (keeping position_normalized +
species_* + all physicochemical/substitution/structural/embedding features) and re-run
standard CV. If δ/γ/ε's score survives without the gene one-hot, the signal is biological;
if it collapses toward the pooled floor, gene identity was doing the work.

Design: matched with/without identity for each group + full, so the delta is clean —
same seeds, same folds, same 82-feat matrix (with identity) vs the 66-feat matrix
(without the 16 subunit_* columns). Protocol matches Run XXXI exactly:
cv_mode="standard" (gene-aware), 3-class, RF + CatBoost, 3 seeds x 15 trials.

Usage:
    python scripts/run_xxxii_gene_identity.py
    python scripts/run_xxxii_gene_identity.py --models random_forest --n-seeds 1 --n-trials 10
"""

import argparse
import json
import sys
import warnings

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore", message=".*ill-defined.*")
warnings.filterwarnings("ignore", message=".*Only one class is present.*")
warnings.filterwarnings("ignore", message=".*ROC AUC score is not defined.*")

from vep_nachr2.data.loader import load_mutation_data
from vep_nachr2.data.reference import load_all_reference_sequences
from vep_nachr2.features.orchestrator import FeatureOrchestrator
from vep_nachr2.features.structural import load_all_pdb_resources
from vep_nachr2.features.gating import GatingExtractor
from vep_nachr2.features.allosteric_interaction import AllostericInteractionExtractor
from vep_nachr2.training.cross_validation import nested_cross_validation
from vep_nachr2.config import RESULTS_DIR, DEFAULT_SEEDS, LABEL_NAMES

_EFFECT_MAP = {"LOF": 0, "No net effect": 1, "GOF": 2}

# Subunit-type groups (same as Run XXXI), keyed by the curated `subunit` gene names.
GROUPS = {
    "alpha": {f"CHRNA{i}" for i in range(1, 8)} | {"CHRNA9", "CHRNA10"},
    "beta": {"CHRNB1", "CHRNB2", "CHRNB3", "CHRNB4"},
    "delta_gamma_epsilon": {"CHRND", "CHRNE", "CHRNG"},
}


def _run_cv(X, y, df, model, n_trials, n_seeds):
    cv = nested_cross_validation(
        X=X, y=y, df=df,
        model_name=model,
        n_outer_folds=5,
        n_trials=n_trials,
        seeds=DEFAULT_SEEDS[:n_seeds],
        cv_mode="standard",  # gene-aware
        verbose=False,
        n_classes=3,
        label_names=LABEL_NAMES,
    )
    return cv.get("final", {}), [float(r["aggregate"]["macro_f1_mean"]) for r in cv.get("per_seed", [])]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=15)
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--models", nargs="+", default=["random_forest", "catboost"])
    args = parser.parse_args()

    df = load_mutation_data(data_file="final_mapped.xlsx")
    y = df["effect"].map(_EFFECT_MAP).values.astype(np.int64)

    ref_seqs = load_all_reference_sequences("human")
    pdb_resources = load_all_pdb_resources()

    # Extract the full 82-feat set once. Capture names from the base orchestrator so we
    # can locate the 16 `subunit_*` one-hot columns (gating/allosteric are appended after,
    # so their presence does not shift the positional-group indices).
    base_orch = FeatureOrchestrator(verbose=False)
    X_base, names_base = base_orch.extract(df, ref_seqs=ref_seqs, use_cache=True)
    gating_feats = GatingExtractor().extract(df, ref_seqs, pdb_resources)
    inter_feats = AllostericInteractionExtractor().extract(df, ref_seqs, pdb_resources)
    X_full = np.hstack([X_base, gating_feats, inter_feats])
    assert X_full.shape == (len(df), 82), X_full.shape
    assert len(names_base) == X_base.shape[1]

    gene_idx = [i for i, n in enumerate(names_base) if n.startswith("subunit_")]
    assert len(gene_idx) == 16, f"expected 16 subunit_* cols, got {len(gene_idx)}"
    X_nogene = np.delete(X_full, gene_idx, axis=1)
    print(f"Full matrix {X_full.shape}; dropping 16 subunit_* cols -> {X_nogene.shape}",
          flush=True)

    out_dir = RESULTS_DIR / "gene_identity_drop"
    out_dir.mkdir(parents=True, exist_ok=True)

    conditions = [("full", None)] + [(g, genes) for g, genes in GROUPS.items()]
    rows = []
    perseed = {}
    for name, genes in conditions:
        if genes is None:
            mask = np.ones(len(df), dtype=bool)
        else:
            mask = df["subunit"].isin(genes).to_numpy()
        pos = np.flatnonzero(mask)
        y_sub = y[pos]
        df_sub = df[mask].reset_index(drop=True)
        n = len(y_sub)
        dist = {LABEL_NAMES[k]: int(v) for k, v in zip(*np.unique(y_sub, return_counts=True))}
        n_genes = df_sub["subunit"].nunique()
        for keep_gene, (X_sub_full, label) in [
            (True, (X_full[pos], "with_identity")),
            (False, (X_nogene[pos], "without_identity")),
        ]:
            print(f"{'=' * 70}\n{name.upper()}  {label}  (n={n}, genes={n_genes}, "
                  f"classes={dist}, dim={X_sub_full.shape[1]})", flush=True)
            for model in args.models:
                res, ps = _run_cv(X_sub_full, y_sub, df_sub, model, args.n_trials, args.n_seeds)
                perseed[f"{name}|{label}|{model}"] = ps
                rows.append({
                    "group": name,
                    "gene_identity": keep_gene,
                    "model": model,
                    "n": n,
                    "n_genes": n_genes,
                    "n_features": X_sub_full.shape[1],
                    "n_lof": dist.get("LOF", 0),
                    "n_gof": dist.get("GOF", 0),
                    "n_nne": dist.get("No net effect", 0),
                    "macro_f1_mean": res.get("macro_f1_mean", float("nan")),
                    "macro_f1_std": res.get("macro_f1_std", float("nan")),
                    "mcc_mean": res.get("mcc_mean", float("nan")),
                    "balanced_accuracy_mean": res.get("balanced_accuracy_mean", float("nan")),
                })
                print(f"  [{model:15s}] macro F1 = {res.get('macro_f1_mean', float('nan')):.4f} "
                      f"± {res.get('macro_f1_std', float('nan')):.4f}")

    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "gene_identity_drop_summary.csv", index=False)
    with open(out_dir / "gene_identity_drop_perseed.json", "w") as _f:
        json.dump(perseed, _f, indent=2)

    # Matched delta: with_identity minus without_identity, per group x model.
    print(f"\n{'=' * 70}\nDelta (with identity - without identity) macro F1:")
    for name, _ in conditions:
        sub = out[out["group"] == name]
        for model in args.models:
            m = sub[sub["model"] == model]
            w = m[m["gene_identity"]]["macro_f1_mean"].iloc[0]
            wo = m[~m["gene_identity"]]["macro_f1_mean"].iloc[0]
            print(f"  {name:22s} [{model:15s}] with={w:.4f}  without={wo:.4f}  Δ={w - wo:+.4f}")

    print(f"\nSaved {out_dir / 'gene_identity_drop_summary.csv'}")


if __name__ == "__main__":
    main()
