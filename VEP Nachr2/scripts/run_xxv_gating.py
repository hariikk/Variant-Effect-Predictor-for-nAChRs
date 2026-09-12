"""
Run XXV: GNM allosteric-dynamics features (gating.py) — additive ablation.

Question: do GNM (Gaussian Network Model) derived features add direction signal
on top of the 75-feature baseline? These are the *dynamics* features the
literature (PreMode; Ersoy et al. CFTR) says static structure lacks:

  gnm_msf            theoretical mean-square fluctuation
  gnm_slowmode       participation in the 2 slowest collective (gating) modes
  gnm_gate_coupling  cross-correlation of the residue to the channel gate (9' ring)

Comparison: baseline (75 feat) vs baseline+gating (78 feat), pooled 16 genes,
leave-one-subunit-out CV, RF + CatBoost, 3 seeds, 15 Optuna trials.

Usage:
    python scripts/run_xxv_gating.py                 # RF + CatBoost, 15 trials, 3 seeds
    python scripts/run_xxv_gating.py --models random_forest --n-trials 20 --n-seeds 5
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
from vep_nachr2.features.gating import GatingExtractor
from vep_nachr2.training.cross_validation import nested_cross_validation
from vep_nachr2.config import RESULTS_DIR, DEFAULT_SEEDS, LABEL_NAMES


def _run_cv(X, y, df, model, n_trials, n_seeds):
    cv = nested_cross_validation(
        X=X, y=y, df=df,
        model_name=model,
        n_outer_folds=5,
        n_trials=n_trials,
        seeds=DEFAULT_SEEDS[:n_seeds],
        cv_mode="subunit",
        verbose=False,
        n_classes=3,
        label_names=LABEL_NAMES,
    )
    return cv.get("final", {})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=15)
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--models", nargs="+", default=["random_forest", "catboost"])
    args = parser.parse_args()

    df = load_mutation_data(data_file="final_mapped.xlsx")
    y = df["effect"].map({"LOF": 0, "No net effect": 1, "GOF": 2}).values.astype(np.int64)

    # Pre-cached matrices (written by _precache_gating.py) let the CV run without
    # re-extracting features — the extraction phase is what trips the background
    # task's memory guard on this machine (see memory: windows-bg-kill-detach).
    cache_npz = RESULTS_DIR / "gating" / "features.npz"
    if cache_npz.exists():
        print(f"Loading pre-cached features from {cache_npz}...", flush=True)
        data = np.load(cache_npz)
        X_base, X_plus = data["X_base"], data["X_plus"]
    else:
        ref_seqs = load_all_reference_sequences("human")
        base_orch = FeatureOrchestrator(verbose=False)
        plus_orch = FeatureOrchestrator(
            extractors=base_orch.extractors + [GatingExtractor()], verbose=False
        )
        print("Extracting baseline features (75)...", flush=True)
        X_base, _ = base_orch.extract(df, ref_seqs=ref_seqs, use_cache=True)
        print("Extracting baseline + gating features (78)...", flush=True)
        X_plus, _ = plus_orch.extract(df, ref_seqs=ref_seqs, use_cache=True)

    print(f"  baseline shape={X_base.shape}  +gating shape={X_plus.shape}\n", flush=True)

    out_dir = RESULTS_DIR / "gating"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for name, X in [("baseline", X_base), ("baseline_gating", X_plus)]:
        print(f"{'=' * 70}\nCONDITION: {name}  ({X.shape[1]} features)", flush=True)
        for model in args.models:
            res = _run_cv(X, y, df, model, args.n_trials, args.n_seeds)
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
    out.to_csv(out_dir / "gating_ablation_summary.csv", index=False)

    # Delta view.
    print(f"\n{'=' * 70}\nDelta macro F1 (+gating − baseline; positive = gating helps):")
    piv = out.pivot_table(index="model", columns="condition", values="macro_f1_mean")
    for model in args.models:
        if model in piv.index:
            d = piv.loc[model, "baseline_gating"] - piv.loc[model, "baseline"]
            print(f"  [{model:15s}] baseline={piv.loc[model, 'baseline']:.4f}  "
                  f"+gating={piv.loc[model, 'baseline_gating']:.4f}  Δ={d:+.4f}")

    print(f"\nSaved {out_dir / 'gating_ablation_summary.csv'}")


if __name__ == "__main__":
    main()
