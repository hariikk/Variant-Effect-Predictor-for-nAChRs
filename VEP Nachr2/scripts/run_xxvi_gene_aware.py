"""
Run XXVI: gene-aware / mechanism-conditioned model.

Research question (different from the gene-agnostic predictor):
  In diagnostic use the gene is *always known*. So instead of stripping gene
  identity out (leave-one-gene-out CV, ~0.50 ceiling), ask the defensible
  question: "given the gene, how well can we predict GOF/LOF/NNE direction?"

Four conditions, one clean decomposition:

  gene_prior     analytic majority-class-per-gene baseline (no model). This is
                 the floor: how much direction signal is in gene identity alone?
  gene_agnostic  75 features, cv_mode="subunit" (leave-one-gene-out). The honest
                 gene-agnostic predictor — the ~0.50 ceiling already documented.
  gene_aware     75 features, cv_mode="standard" (stratified K-fold, gene identity
                 is a legitimate input at test time). THE gene-aware model.
  variants_only  55 features (drop positional: gene OH + species OH + norm_pos),
                 cv_mode="standard". Per-variant biochemistry in-distribution —
                 isolates whether explicit gene identity is what carries the gain.

Comparisons:
  gene_aware  − gene_agnostic  = value of "knowing the gene" as a legitimate input
  gene_aware  − gene_prior     = per-variant contribution *within* a known gene
  gene_aware  − variants_only  = marginal value of explicit gene/species/position

Models: RF + CatBoost. 3 seeds, 15 Optuna trials, 5 outer / 5 inner folds.

Usage:
    python scripts/run_xxvi_gene_aware.py
    python scripts/run_xxvi_gene_aware.py --models random_forest --n-seeds 1 --n-trials 10
"""

import argparse
import json
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, matthews_corrcoef, balanced_accuracy_score

sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore", message=".*ill-defined.*")
warnings.filterwarnings("ignore", message=".*Only one class is present.*")
warnings.filterwarnings("ignore", message=".*ROC AUC score is not defined.*")

from vep_nachr2.data.loader import load_mutation_data
from vep_nachr2.data.reference import load_all_reference_sequences
from vep_nachr2.features.orchestrator import FeatureOrchestrator
from vep_nachr2.training.cross_validation import nested_cross_validation
from vep_nachr2.config import RESULTS_DIR, DEFAULT_SEEDS, LABEL_NAMES

_EFFECT_MAP = {"LOF": 0, "No net effect": 1, "GOF": 2}


def _run_cv(X, y, df, model, cv_mode, n_trials, n_seeds):
    cv = nested_cross_validation(
        X=X, y=y, df=df,
        model_name=model,
        n_outer_folds=5,
        n_trials=n_trials,
        seeds=DEFAULT_SEEDS[:n_seeds],
        cv_mode=cv_mode,
        verbose=False,
        n_classes=3,
        label_names=LABEL_NAMES,
    )
    return cv.get("final", {}), [float(r["aggregate"]["macro_f1_mean"]) for r in cv.get("per_seed", [])]


def _gene_prior(df, y):
    """Analytic majority-class-per-gene baseline (in-sample, optimistic upper bound)."""
    majority = df.groupby("subunit")["effect"].agg(lambda s: s.mode().iloc[0])
    y_prior = df["subunit"].map(majority).map(_EFFECT_MAP).values.astype(np.int64)
    return {
        "macro_f1": float(f1_score(y, y_prior, average="macro")),
        "mcc": float(matthews_corrcoef(y, y_prior)),
        "balanced_accuracy": float(balanced_accuracy_score(y, y_prior)),
        "accuracy": float((y == y_prior).mean()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=15)
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--models", nargs="+", default=["random_forest", "catboost"])
    args = parser.parse_args()

    df = load_mutation_data(data_file="final_mapped.xlsx")
    y = df["effect"].map(_EFFECT_MAP).values.astype(np.int64)

    # ── Features ────────────────────────────────────────────────────────────
    # The 75-feature default set is already on the disk cache (feature_cache_
    # ed58701110c5.pkl), so this is a fast load — no ESM re-extraction.
    ref_seqs = load_all_reference_sequences("human")
    orch = FeatureOrchestrator(verbose=False)
    X_full, feat_names = orch.extract(df, ref_seqs=ref_seqs, use_cache=True)

    # variants_only = full minus the positional group (16 gene OH + 3 species OH
    # + 1 norm_pos = 20 columns). Sliced in-place to avoid re-running ESM/PDB.
    idx = orch.get_feature_group_indices()
    pos_start, pos_end = idx["positional"]
    keep = np.ones(X_full.shape[1], dtype=bool)
    keep[pos_start:pos_end] = False
    X_variants = X_full[:, keep]

    print(f"  full shape={X_full.shape}  variants_only shape={X_variants.shape}\n",
          flush=True)

    out_dir = RESULTS_DIR / "gene_aware"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── gene_prior (analytic) ───────────────────────────────────────────────
    prior = _gene_prior(df, y)
    print("=" * 70)
    print("gene_prior (majority-class-per-gene, in-sample):")
    for k, v in prior.items():
        print(f"  {k:18s} {v:.4f}")

    # per-gene label distribution, for interpretation
    dist = df.groupby("subunit")["effect"].value_counts().unstack(fill_value=0)
    dist = dist.reindex(columns=["LOF", "No net effect", "GOF"], fill_value=0)
    dist["n"] = dist.sum(axis=1)
    dist["purity"] = (dist[["LOF", "No net effect", "GOF"]].max(axis=1) / dist["n"]).round(3)
    print("\nPer-gene label distribution (purity = majority share):")
    print(dist.sort_values("purity", ascending=False).to_string())
    dist.to_csv(out_dir / "gene_label_distribution.csv")

    # ── Model conditions ────────────────────────────────────────────────────
    conditions = [
        ("gene_agnostic", X_full, "subunit"),
        ("gene_aware",    X_full, "standard"),
        ("variants_only", X_variants, "standard"),
    ]

    perseed = {}
    rows = [{
        "condition": "gene_prior",
        "model": "majority_class",
        "n_features": 1,  # gene identity (categorical)
        "macro_f1_mean": prior["macro_f1"],
        "macro_f1_std": 0.0,
        "mcc_mean": prior["mcc"],
        "balanced_accuracy_mean": prior["balanced_accuracy"],
    }]

    for name, X, cv_mode in conditions:
        print(f"\n{'=' * 70}\nCONDITION: {name}  cv_mode={cv_mode}  ({X.shape[1]} feats)",
              flush=True)
        for model in args.models:
            res, ps = _run_cv(X, y, df, model, cv_mode, args.n_trials, args.n_seeds)
            perseed[f"{name}|{model}"] = ps
            rows.append({
                "condition": name,
                "model": model,
                "n_features": X.shape[1],
                "macro_f1_mean": res.get("macro_f1_mean", float("nan")),
                "macro_f1_std": res.get("macro_f1_std", float("nan")),
                "mcc_mean": res.get("mcc_mean", float("nan")),
                "balanced_accuracy_mean": res.get("balanced_accuracy_mean", float("nan")),
            })
            print(f"  [{model:15s}] macro F1 = {res.get('macro_f1_mean', float('nan')):.4f} "
                  f"± {res.get('macro_f1_std', float('nan')):.4f}")

    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "gene_aware_summary.csv", index=False)
    with open(out_dir / "gene_aware_perseed.json", "w") as _f:
        json.dump(perseed, _f, indent=2)

    # ── Delta view ──────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}\nDelta macro F1 (condition − gene_agnostic):")
    piv = out.pivot_table(index="model", columns="condition", values="macro_f1_mean")
    for model in args.models:
        if model not in piv.index:
            continue
        base = piv.loc[model, "gene_agnostic"]
        for cond in ["gene_aware", "variants_only"]:
            if cond in piv.columns:
                d = piv.loc[model, cond] - base
                print(f"  [{model:15s}] {cond:15s} {piv.loc[model, cond]:.4f}  Δ={d:+.4f}")
        # gene_prior is model-agnostic (analytic) — report the gap to it once.
        prior_f1 = out.loc[out["condition"] == "gene_prior", "macro_f1_mean"].iloc[0]
        print(f"  [{model:15s}] {'gene_prior':15s} {prior_f1:.4f}  Δ={base - prior_f1:+.4f}  (vs prior)")

    print(f"\nSaved {out_dir / 'gene_aware_summary.csv'}")


if __name__ == "__main__":
    main()
