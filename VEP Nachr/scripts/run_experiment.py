"""
Main experiment runner for nAChR VEP.

Usage:
    python scripts/run_experiment.py                    # Run all core models
    python scripts/run_experiment.py --model random_forest  # Run single model
    python scripts/run_experiment.py --quick            # Quick test (no HP opt)

This script:
1. Loads and cleans the nAChR mutation database
2. Extracts engineered features (physicochemical + substitution + positional)
3. Runs nested cross-validation for each model
4. Saves results to results/ directory
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np

from vep_nachr.config import CORE_MODELS, ALL_MODELS, RESULTS_DIR
from vep_nachr.data.loader import load_dataset
from vep_nachr.features.encoder import NachrFeatureEncoder
from vep_nachr.training.cross_validation import (
    nested_cross_validation,
    simple_cross_validation,
)


def run_quick_test(X, y, models=None):
    """Run quick evaluation with default hyperparameters (no Optuna)."""
    if models is None:
        models = CORE_MODELS

    print("\n" + "=" * 60)
    print("QUICK EVALUATION (default hyperparameters, no Optuna)")
    print("=" * 60)

    results = {}
    for model_name in models:
        try:
            result = simple_cross_validation(
                X, y, model_name=model_name, n_folds=5, n_jobs=-1
            )
            results[model_name] = result
            print(
                f"  {model_name:25s}: "
                f"F1={result['mean_f1']:.4f} +/- {result['std_f1']:.4f}  "
                f"Acc={result['mean_accuracy']:.4f} +/- {result['std_accuracy']:.4f}"
            )
        except Exception as e:
            print(f"  {model_name:25s}: FAILED - {e}")

    return results


def run_full_experiment(X, y, models=None, feature_names=None, n_trials=50):
    """Run full nested CV with Optuna HP optimization."""
    if models is None:
        models = CORE_MODELS

    print("\n" + "=" * 60)
    print("FULL NESTED CROSS-VALIDATION (with Optuna HP optimization)")
    print("=" * 60)

    all_results = {}

    for model_name in models:
        try:
            result = nested_cross_validation(
                X, y,
                model_name=model_name,
                encoding="engineered",
                n_outer_folds=5,
                n_inner_folds=5,
                n_trials=n_trials,
                n_jobs=-1,
                verbose=1,
                feature_names=feature_names,
            )
            all_results[model_name] = result

            # Save individual result
            result.save(RESULTS_DIR / f"{model_name}_engineered.json")

        except Exception as e:
            print(f"\n  {model_name}: FAILED - {e}")
            import traceback
            traceback.print_exc()

    # Print summary table
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Model':25s} {'F1':>10s} {'Std':>8s} {'Accuracy':>10s}")
    print("-" * 55)
    for model_name, result in sorted(
        all_results.items(), key=lambda x: x[1].overall_mean_f1, reverse=True
    ):
        print(
            f"{model_name:25s} {result.overall_mean_f1:10.4f} "
            f"{result.overall_std_f1:8.4f} {result.overall_mean_accuracy:10.4f}"
        )

    return all_results


def main():
    parser = argparse.ArgumentParser(description="nAChR Variant Effect Predictor")
    parser.add_argument(
        "--model", type=str, default=None,
        help="Run a specific model (e.g., random_forest). Default: all core models.",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick test with default HPs (no Optuna optimization).",
    )
    parser.add_argument(
        "--all-models", action="store_true",
        help="Run all models (core + extended) instead of just core.",
    )
    parser.add_argument(
        "--n-trials", type=int, default=50,
        help="Number of Optuna trials per fold (default: 50).",
    )
    parser.add_argument(
        "--no-structural", action="store_true",
        help="Exclude structural features (useful when no PDB files available).",
    )
    args = parser.parse_args()

    # --- Load data ---
    print("Loading data...")
    df, labels, sequences = load_dataset()

    # --- Feature engineering ---
    print("\nExtracting features...")
    encoder = NachrFeatureEncoder(
        include_structural=not args.no_structural,
        include_substitution=True,
        include_positional=True,
    )
    X = encoder.fit_transform(df)
    feature_names = encoder.get_feature_names_out()
    print(f"Feature matrix shape: {X.shape}")
    print(f"Features: {feature_names}")

    # --- Select models ---
    if args.model:
        models = [args.model]
    elif args.all_models:
        models = [m for m in ALL_MODELS if m in __import__("vep_nachr.models.registry", fromlist=["AVAILABLE_MODELS"]).AVAILABLE_MODELS]
    else:
        models = [m for m in CORE_MODELS if m in __import__("vep_nachr.models.registry", fromlist=["AVAILABLE_MODELS"]).AVAILABLE_MODELS]

    # --- Run ---
    if args.quick:
        results = run_quick_test(X, labels, models)
    else:
        results = run_full_experiment(X, labels, models, feature_names, args.n_trials)

    print("\nDone! Results saved to:", RESULTS_DIR)


if __name__ == "__main__":
    main()
