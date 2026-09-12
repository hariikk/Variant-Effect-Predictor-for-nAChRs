"""
Run XXIV: family-conditioned (within-family) leave-one-gene-out CV.

Question: is GOF/LOF/NNE direction more predictable *within* a receptor family than
across all 16 genes? If direction is mechanism-conditioned and family ≈ mechanism,
a model that only ever sees one family should reach a higher macro-F1 than the
pooled 16-gene model — the non-confounded version of "family matters" (the pooled
model already sees gene one-hot, but must learn 16 distinct mechanisms).

Families (16 genes total):
  muscle          CHRNA1, CHRNB1, CHRND, CHRNE, CHRNG              (5 genes)
  neuronal_hetero CHRNA2/3/4/5/6/9/10, CHRNB2/3/4                   (10 genes)
  pooled          all 16 genes (reference baseline)

Within each family: leave-one-gene-out CV (cv_mode="subunit", n_folds = n_genes,
capped at 5), RF + CatBoost, 3 seeds. The comparison is within-family F1 vs pooled
F1. Caveat: families are smaller than the pooled set, so within-family F1 is noisier
and the two are not strictly sample-matched — interpret directionally.

Usage:
    python scripts/run_xxiv_family.py                 # RF + CatBoost, 15 trials, 3 seeds
    python scripts/run_xxiv_family.py --models random_forest --n-trials 30 --n-seeds 5
"""

import argparse
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
from vep_nachr2.training.cross_validation import nested_cross_validation
from vep_nachr2.config import RESULTS_DIR, DEFAULT_SEEDS, LABEL_NAMES, NACHR_GENES

FAMILIES = {
    "muscle": ["CHRNA1", "CHRNB1", "CHRND", "CHRNE", "CHRNG"],
    "neuronal_hetero": [
        "CHRNA2", "CHRNA3", "CHRNA4", "CHRNA5", "CHRNA6",
        "CHRNA9", "CHRNA10", "CHRNB2", "CHRNB3", "CHRNB4",
    ],
}


def _run_cv(X, y, df, model, n_trials, n_seeds, cv_mode, n_folds):
    cv = nested_cross_validation(
        X=X, y=y, df=df,
        model_name=model,
        n_outer_folds=n_folds,
        n_trials=n_trials,
        seeds=DEFAULT_SEEDS[:n_seeds],
        cv_mode=cv_mode,
        verbose=False,
        n_classes=3,
        label_names=LABEL_NAMES,
    )
    return cv.get("final", {})


def _family_cv(X, y, df, mask, model, n_trials, n_seeds):
    """Run within-family CV; return (final_dict, note)."""
    df_sub = df[mask].reset_index(drop=True)
    y_sub = y[mask]
    X_sub = X[mask]
    n_genes = df_sub["subunit"].nunique()

    cv_mode = "subunit" if n_genes >= 2 else "standard"
    n_folds = min(5, n_genes) if n_genes >= 2 else 5

    try:
        res = _run_cv(X_sub, y_sub, df_sub, model, n_trials, n_seeds, cv_mode, n_folds)
        return res, f"{cv_mode}/{n_folds}"
    except Exception as e:
        print(f"    [warn] subunit CV failed ({type(e).__name__}); "
              f"falling back to standard CV", flush=True)
        res = _run_cv(X_sub, y_sub, df_sub, model, n_trials, n_seeds, "standard", 5)
        return res, "standard/5"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=15)
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--models", nargs="+", default=["random_forest", "catboost"])
    args = parser.parse_args()

    df = load_mutation_data(data_file="final_mapped.xlsx")
    y = df["effect"].map({"LOF": 0, "No net effect": 1, "GOF": 2}).values.astype(np.int64)

    ref_seqs = load_all_reference_sequences("human")
    orch = FeatureOrchestrator(verbose=False)
    X, _ = orch.extract(df, ref_seqs=ref_seqs, use_cache=True)

    out_dir = RESULTS_DIR / "family"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    # Families + pooled baseline.
    conditions = dict(FAMILIES)
    conditions["pooled"] = list(NACHR_GENES)

    for name, genes in conditions.items():
        mask = df["subunit"].isin(genes).values
        n = int(mask.sum())
        n_genes = df[mask]["subunit"].nunique()
        print(f"\n{'=' * 70}\nFAMILY: {name}  (n={n}, genes={n_genes})", flush=True)
        for model in args.models:
            res, note = _family_cv(X, y, df, mask, model, args.n_trials, args.n_seeds)
            rows.append({
                "family": name,
                "model": model,
                "cv": note,
                "n": n,
                "n_genes": n_genes,
                "macro_f1_mean": res.get("macro_f1_mean", float("nan")),
                "macro_f1_std": res.get("macro_f1_std", float("nan")),
                "mcc_mean": res.get("mcc_mean", float("nan")),
                "balanced_accuracy_mean": res.get("balanced_accuracy_mean", float("nan")),
            })
            print(f"  [{model:15s}] macro F1 = {res.get('macro_f1_mean', float('nan')):.4f} "
                  f"± {res.get('macro_f1_std', float('nan')):.4f}  (cv={note})")

    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "family_cv_summary.csv", index=False)

    # Family vs pooled delta per model.
    print(f"\n{'=' * 70}\nWithin-family vs pooled macro F1 (positive = family helps):")
    for model in args.models:
        pooled_f1 = out[(out["family"] == "pooled") & (out["model"] == model)]["macro_f1_mean"]
        pooled_f1 = pooled_f1.iloc[0] if len(pooled_f1) else float("nan")
        print(f"\n  [{model}] pooled F1 = {pooled_f1:.4f}")
        for fam in FAMILIES:
            f = out[(out["family"] == fam) & (out["model"] == model)]
            if len(f):
                f1 = f["macro_f1_mean"].iloc[0]
                print(f"      {fam:16s} F1 = {f1:.4f}  (Δ = {f1 - pooled_f1:+.4f})")

    print(f"\nSaved {out_dir / 'family_cv_summary.csv'}")


if __name__ == "__main__":
    main()
