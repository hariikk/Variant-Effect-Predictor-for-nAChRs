"""
1-seed screen of the new ideas against the baseline (3-class, subunit CV).

Ideas under test (all 3-class, leave-one-subunit-out):
  #3  per-residue ESM-2 embeddings  (feature: +16 PCA components)
  #4  ensemble stacking             (model: logistic meta-learner over 10 base models)
  #5  ordinal regression            (model: cumulative-link ordinal classifier)
  #2  stability ddG (RaSP)          -> BLOCKED (see notes; not pip-installable)
  #1  LoRA fine-tune                -> separate script (lora_finetune.py)

Runs a focused model set on the baseline feature set (no esm_embeddings) and the
full feature set (with esm_embeddings), plus stacking on the full set. Prints a
delta table so each idea's contribution is visible.

Usage:
    python scripts/run_ideas.py            # 1 seed
    python scripts/run_ideas.py --seeds 42 123 456 789 1011   # 5 seeds
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from vep_nachr2.config import RESULTS_DIR, DEFAULT_SEEDS
from vep_nachr2.data.loader import load_mutation_data
from vep_nachr2.data.reference import load_all_reference_sequences
from vep_nachr2.features.orchestrator import FeatureOrchestrator, clear_feature_cache
from vep_nachr2.training.cross_validation import nested_cross_validation, save_results
from vep_nachr2.training.stacking import stacked_cross_validation

# Focused model set for the feature A/B screen (headline models + ordinal).
FEATURE_MODELS = [
    "catboost", "random_forest", "lightgbm", "xgboost",
    "logistic_regression", "ordinal",
]

N_TRIALS = 30


def _macro_f1(results: dict) -> float:
    return results["final"].get("macro_f1_mean", float("nan"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="*", type=int, default=DEFAULT_SEEDS[:1],
                        help="Seeds (default: 1 seed = 42)")
    parser.add_argument("--n-trials", type=int, default=N_TRIALS)
    args = parser.parse_args()

    seeds = args.seeds
    n_trials = args.n_trials
    t0 = time.time()

    print("Loading data...")
    df = load_mutation_data()
    y = df["effect"].map({"LOF": 0, "No net effect": 1, "GOF": 2}).values.astype(np.int64)
    ref_seqs = load_all_reference_sequences("human")

    # Clear stale cache (cache key ignores feature version — known issue).
    clear_feature_cache(verbose=False)

    out_dir = RESULTS_DIR / "new_ideas"
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_tag = "5seeds" if len(seeds) > 1 else "1seed"

    print("Extracting features...")
    orch = FeatureOrchestrator(verbose=False)

    # Cache features to disk so a killed run can resume without re-extracting
    # (PDB loading + ESM-2 forward passes are the expensive part).
    feat_cache = out_dir / f"features_{seed_tag}.npz"
    if feat_cache.exists():
        print("  Loading cached features...")
        data = np.load(feat_cache)
        X_full, X_base = data["X_full"], data["X_base"]
    else:
        # Extract the full feature set ONCE (82 features), then derive the
        # baseline by dropping the esm_embeddings PCA columns.
        X_full, names_full = orch.extract(df, ref_seqs=ref_seqs, use_cache=False)
        emb_cols = [i for i, n in enumerate(names_full) if n.startswith("esm2_pc")]
        keep = [i for i in range(len(names_full)) if i not in emb_cols]
        X_base = X_full[:, keep]
        np.savez(feat_cache, X_full=X_full, X_base=X_base)

    print(f"Baseline features: {X_base.shape}, Full features: {X_full.shape}")

    rows = []

    def save_rows():
        pd.DataFrame(rows).to_csv(out_dir / f"ideas_{seed_tag}.csv", index=False)

    def run_model(model_name, X, label):
        r = nested_cross_validation(
            X=X, y=y, df=df, model_name=model_name,
            n_outer_folds=5, n_trials=n_trials, seeds=seeds,
            cv_mode="subunit", verbose=False, n_classes=3,
        )
        rows.append({"model": model_name, "condition": label,
                     "macro_f1_mean": _macro_f1(r),
                     "mcc_mean": r["final"].get("mcc_mean"),
                     "balanced_accuracy_mean": r["final"].get("balanced_accuracy_mean")})
        save_rows()
        return r

    # Baseline + full for each model in FEATURE_MODELS.
    for model_name in FEATURE_MODELS:
        print(f"\n[baseline] {model_name}")
        run_model(model_name, X_base, "baseline")
        print(f"[+embeddings] {model_name}")
        run_model(model_name, X_full, "+embeddings")

    # Stacking on the full feature set.
    print("\n[stacking] over 10 base models on full features")
    stack_r = stacked_cross_validation(
        X=X_full, y=y, df=df, seeds=seeds, cv_mode="subunit",
        verbose=False, n_classes=3,
    )
    rows.append({"model": "stacking", "condition": "+embeddings",
                 "macro_f1_mean": _macro_f1(stack_r),
                 "mcc_mean": stack_r["final"].get("mcc_mean"),
                 "balanced_accuracy_mean": stack_r["final"].get("balanced_accuracy_mean")})
    save_rows()

    # Build comparison table.
    summary = pd.DataFrame(rows)

    # Pivot: baseline vs +embeddings delta per model.
    pivot = summary.pivot_table(index="model", columns="condition",
                                values="macro_f1_mean", aggfunc="first")
    if "baseline" in pivot.columns and "+embeddings" in pivot.columns:
        pivot["delta_f1"] = pivot["+embeddings"] - pivot["baseline"]

    print("\n" + "=" * 70)
    print(f"1-seed 3-class comparison (n_trials={n_trials}, seeds={seeds})")
    print("=" * 70)
    print(summary.round(4).to_string(index=False))
    print("\nPivot (macro F1):")
    print(pivot.round(4).to_string())
    print(f"\nSaved to {out_dir / f'ideas_{seed_tag}.csv'}")
    print(f"Total time: {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
