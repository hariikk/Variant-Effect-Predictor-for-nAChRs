"""
Run XXII: subgroup / heterogeneity diagnostic.

Question: does `structural_core` (the 7 PDB-derived columns) carry GOF/LOF/NNE
direction signal WITHIN a receptor family, even though it is null pooled (Run XII,
and ablate_feature_impact's drop_structural_core delta ≈ −0.002)?

Rationale: pooled leave-one-subunit-out CV forbids gene identity, and nAChR direction
is gene/mechanism-conditioned. If structure is informative *within* a family — where
the mechanism is homogeneous — then a family-conditioned model is the next step.
If it is null within every family too, the structural hypothesis is airtight-negative.

Three subgroups (16 genes total, partitioned by receptor assembly):
  muscle            CHRNA1, CHRNB1, CHRND, CHRNE, CHRNG                (5 genes)
  neuronal_hetero   CHRNA2/3/4/5/6/9/10, CHRNB2/3/4                     (10 genes)
  neuronal_homo     CHRNA7                                             (1 gene)

Per subgroup × model: full vs drop_structural_core ablation via nested CV
(delta_f1 = F1(drop) − F1(full); negative = structural_core HELPS). CV mode is
"subunit" where ≥2 genes (leave-one-gene-out); "standard" for α7 (single gene —
gene grouping is meaningless, and this only measures the within-α7 ceiling with
position leakage — see Run XVI caveat). Plus per-subgroup permutation importance
of the 7 structural_core columns on a descriptive RF (single 80/20 split, NOT a
generalization claim).

Usage:
    python scripts/run_xxii_subgroup.py                 # RF + CatBoost, 15 trials, 1 seed
    python scripts/run_xxii_subgroup.py --models random_forest --n-trials 30 --n-seeds 3
"""

import argparse
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

# Windows console default is cp1252; the report prints α/±/−. Reconfigure stdout
# to UTF-8 so these don't crash (or mojibake) when redirected.
sys.stdout.reconfigure(encoding="utf-8")

# compute_metrics warns when a CV fold has a single class (ROC-AUC / per-class F1
# undefined). Expected here — small per-gene subgroups — so silence the noise.
warnings.filterwarnings("ignore", message=".*ill-defined.*")
warnings.filterwarnings("ignore", message=".*Only one class is present.*")
warnings.filterwarnings("ignore", message=".*ROC AUC score is not defined.*")

from vep_nachr2.data.loader import load_mutation_data
from vep_nachr2.data.reference import load_all_reference_sequences
from vep_nachr2.features.orchestrator import FeatureOrchestrator
from vep_nachr2.training.cross_validation import nested_cross_validation
from vep_nachr2.config import RESULTS_DIR, DEFAULT_SEEDS, LABEL_NAMES

SUBGROUPS = {
    "muscle": ["CHRNA1", "CHRNB1", "CHRND", "CHRNE", "CHRNG"],
    "neuronal_hetero": [
        "CHRNA2", "CHRNA3", "CHRNA4", "CHRNA5", "CHRNA6",
        "CHRNA9", "CHRNA10", "CHRNB2", "CHRNB3", "CHRNB4",
    ],
    "neuronal_homo": ["CHRNA7"],
}


def _run_cv(X, y, df, model, n_trials, n_seeds, cv_mode, n_folds):
    """Run nested CV; return the 'final' aggregate dict."""
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


def _ablation_pair(X_sub, y_sub, df_sub, model, n_trials, n_seeds, cv_mode, n_folds,
                   struct_start, struct_end):
    """Run full vs drop_structural_core; return (full_final, drop_final, note)."""
    full = _run_cv(X_sub, y_sub, df_sub, model, n_trials, n_seeds, cv_mode, n_folds)

    keep = np.ones(X_sub.shape[1], dtype=bool)
    keep[struct_start:struct_end] = False
    drop = _run_cv(X_sub[:, keep], y_sub, df_sub, model, n_trials, n_seeds, cv_mode, n_folds)

    return full, drop, None


def _subgroup_cv(X, y, df, mask, model, n_trials, n_seeds, struct_start, struct_end):
    """Run full vs drop for one subgroup, handling the α7 single-gene case.

    Returns (full_final, drop_final, note) where note records the CV mode + folds
    actually used, so the α7 standard-CV caveat is visible in the output.
    """
    df_sub = df[mask].reset_index(drop=True)
    y_sub = y[mask]
    X_sub = X[mask]
    n_genes = df_sub["subunit"].nunique()

    if n_genes >= 2:
        cv_mode = "subunit"
        n_folds = min(5, n_genes)  # leave-one-gene-out (capped at 5)
        try:
            full, drop, _ = _ablation_pair(X_sub, y_sub, df_sub, model, n_trials,
                                           n_seeds, cv_mode, n_folds, struct_start, struct_end)
            return full, drop, f"{cv_mode}/{n_folds}"
        except Exception as e:
            # Class-imbalance / group-split failure (e.g. a gene missing a class).
            # Fall back to standard stratified CV with a clear note.
            print(f"    [warn] subunit CV failed ({type(e).__name__}); "
                  f"falling back to standard CV", flush=True)
            cv_mode, n_folds = "standard", 5
    else:
        # α7: single gene → gene grouping is meaningless. Standard CV only measures
        # the within-α7 ceiling (position leakage); Run XVI already settled this.
        cv_mode, n_folds = "standard", 5

    full = _run_cv(X_sub, y_sub, df_sub, model, n_trials, n_seeds, cv_mode, n_folds)
    keep = np.ones(X_sub.shape[1], dtype=bool)
    keep[struct_start:struct_end] = False
    drop = _run_cv(X_sub[:, keep], y_sub, df_sub, model, n_trials, n_seeds, cv_mode, n_folds)
    return full, drop, f"{cv_mode}/{n_folds}"


def _struct_permutation_importance(X_sub, y_sub, struct_start, struct_end,
                                   struct_names, seed=42):
    """Descriptive permutation importance of structural_core columns (single split)."""
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_sub, y_sub, test_size=0.2, stratify=y_sub, random_state=seed)
    rf = RandomForestClassifier(
        n_estimators=300, class_weight="balanced", random_state=seed, n_jobs=-1)
    rf.fit(X_tr, y_tr)
    r = permutation_importance(
        rf, X_te, y_te, n_repeats=10, random_state=seed, scoring="f1_macro", n_jobs=-1)
    means = r.importances_mean[struct_start:struct_end]
    stds = r.importances_std[struct_start:struct_end]
    return {name: (float(m), float(s)) for name, m, s in zip(struct_names, means, stds)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=15)
    parser.add_argument("--n-seeds", type=int, default=1)
    parser.add_argument("--models", nargs="+", default=["random_forest", "catboost"])
    args = parser.parse_args()

    df = load_mutation_data(data_file="final_mapped.xlsx")
    y = df["effect"].map({"LOF": 0, "No net effect": 1, "GOF": 2}).values.astype(np.int64)

    ref_seqs = load_all_reference_sequences("human")
    orch = FeatureOrchestrator(verbose=False)
    X, feature_names = orch.extract(df, ref_seqs=ref_seqs, use_cache=True)

    struct_start, struct_end = orch.get_feature_group_indices()["structural_core"]
    struct_names = feature_names[struct_start:struct_end]
    print(f"structural_core columns ({struct_end - struct_start}): {struct_names}\n")

    out_dir = RESULTS_DIR / "subgroup"
    out_dir.mkdir(parents=True, exist_ok=True)

    ablation_rows = []
    perm_rows = []

    for name, genes in SUBGROUPS.items():
        mask = df["subunit"].isin(genes).values
        df_sub = df[mask].reset_index(drop=True)
        y_sub = y[mask]
        X_sub = X[mask]
        n = len(df_sub)

        print(f"{'=' * 70}\nSUBGROUP: {name}  (n={n}, genes={sorted(df_sub['subunit'].unique())})")
        print("species × effect crosstab (the confound to watch):")
        print(df_sub.groupby(["species", "effect"]).size().unstack(fill_value=0), "\n")

        for model in args.models:
            full, drop, note = _subgroup_cv(
                X, y, df, mask, model, args.n_trials, args.n_seeds,
                struct_start, struct_end)
            delta = drop.get("macro_f1_mean", float("nan")) - full.get("macro_f1_mean", float("nan"))
            ablation_rows.append({
                "subgroup": name,
                "model": model,
                "condition": "full",
                "cv": note,
                "n_features": X_sub.shape[1],
                "macro_f1_mean": full.get("macro_f1_mean", float("nan")),
                "macro_f1_std": full.get("macro_f1_std", float("nan")),
                "delta_f1": 0.0,
            })
            ablation_rows.append({
                "subgroup": name,
                "model": model,
                "condition": "drop_structural_core",
                "cv": note,
                "n_features": X_sub.shape[1] - (struct_end - struct_start),
                "macro_f1_mean": drop.get("macro_f1_mean", float("nan")),
                "macro_f1_std": drop.get("macro_f1_std", float("nan")),
                "delta_f1": delta,
            })
            print(f"  [{model:15s}] full F1 = {full.get('macro_f1_mean', float('nan')):.4f} "
                  f"| drop_structural_core F1 = {drop.get('macro_f1_mean', float('nan')):.4f} "
                  f"| delta = {delta:+.4f}  (cv={note})")

        # Permutation importance of structural_core within this subgroup (descriptive).
        try:
            imp = _struct_permutation_importance(
                X_sub, y_sub, struct_start, struct_end, struct_names)
            for fname, (m, s) in imp.items():
                perm_rows.append({"subgroup": name, "feature": fname,
                                  "importance_mean": m, "importance_std": s})
            top = max(imp.items(), key=lambda kv: kv[1][0])
            print(f"  top structural_core column by permutation importance: "
                  f"{top[0]} ({top[1][0]:+.4f} ± {top[1][1]:.4f})")
        except Exception as e:
            print(f"  [warn] permutation importance failed: {e}")

    # ── Save ──
    abl = pd.DataFrame(ablation_rows)
    abl.to_csv(out_dir / "ablation_summary.csv", index=False)

    # Compact delta view for the takeaway.
    delta_view = (
        abl[abl["condition"] == "drop_structural_core"]
        .pivot_table(index="subgroup", columns="model", values="delta_f1")
    )
    print(f"\n{'=' * 70}\ndelta_f1 = F1(drop_structural_core) − F1(full)  "
          f"(negative = structural_core HELPS):")
    print(delta_view.round(4).to_string())

    if perm_rows:
        perm = pd.DataFrame(perm_rows)
        perm.to_csv(out_dir / "permutation_importance.csv", index=False)
        print(f"\nPermutation importance (descriptive, single split):")
        piv = perm.pivot_table(index="feature", columns="subgroup", values="importance_mean")
        print(piv.round(4).to_string())

    print(f"\nSaved {out_dir / 'ablation_summary.csv'}"
          + (f" and {out_dir / 'permutation_importance.csv'}" if perm_rows else ""))


if __name__ == "__main__":
    main()
