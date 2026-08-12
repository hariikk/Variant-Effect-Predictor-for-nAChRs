#!/usr/bin/env python3
"""
Main experiment entry point for VEP-nAChR2.

Quick test:
    python scripts/run_experiment.py --test

Full experiment (clears cache):
    python scripts/run_experiment.py --full --clean

Compare models:
    python scripts/run_experiment.py --compare

Ablation study (single model):
    python scripts/run_experiment.py --ablation --model catboost

Ablation study (ALL models):
    python scripts/run_experiment.py --ablation --ablation-all

Species transfer:
    python scripts/run_experiment.py --species-transfer

Clear cache only:
    python scripts/run_experiment.py --clean-cache
"""

import argparse
import sys
from pathlib import Path

# Ensure package is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from vep_nachr2.training.runner import (
    run_single,
    run_compare,
    run_ablation,
    run_ablation_all,
    run_species_transfer_experiment,
)
from vep_nachr2.features.orchestrator import clear_feature_cache


def main():
    parser = argparse.ArgumentParser(description="VEP-nAChR2 Experiment Runner")
    parser.add_argument("--test", action="store_true",
                        help="Quick test: single RF model, 2 folds, 10 trials")
    parser.add_argument("--full", action="store_true",
                        help="Full experiment: all 4 core models, 5 folds, 50 trials")
    parser.add_argument("--compare", action="store_true",
                        help="Compare all 10 models, 3 seeds, 30 trials")
    parser.add_argument("--ablation", action="store_true",
                        help="Feature group ablation study")
    parser.add_argument("--ablation-all", action="store_true",
                        help="Run ablation for ALL models (use with --ablation)")
    parser.add_argument("--species-transfer", action="store_true",
                        help="Species transfer experiment")
    parser.add_argument("--model", type=str, default="random_forest",
                        help="Model for single/ablation/species-transfer")
    parser.add_argument("--n-trials", type=int, default=None,
                        help="Override Optuna trial count")
    parser.add_argument("--clean", action="store_true",
                        help="Clear feature cache before running experiments")
    parser.add_argument("--clean-cache", action="store_true",
                        help="Clear feature cache and exit (no experiment)")
    args = parser.parse_args()

    # Standalone cache clear
    if args.clean_cache:
        clear_feature_cache(verbose=True)
        return

    if not any([args.test, args.full, args.compare, args.ablation, args.species_transfer]):
        parser.print_help()
        print("\nNo experiment selected. Use --test for a quick test.")
        return

    # Clear cache if requested
    if args.clean:
        print("\n" + "=" * 50)
        print("Clearing feature cache before run...")
        print("=" * 50)
        clear_feature_cache(verbose=True)
        print()

    # ── Quick test ──
    if args.test:
        print("Quick test: Random Forest, 2 folds, 10 Optuna trials")
        run_single(
            model_name="random_forest",
            cv_mode="subunit",
            n_outer_folds=3,
            n_trials=args.n_trials or 10,
        )

    # ── Full experiment ──
    if args.full:
        print("Full experiment: 4 core models")
        run_compare(
            models=["logistic_regression", "svm_rbf", "random_forest", "lightgbm"],
            cv_mode="subunit",
            n_trials=args.n_trials or 50,
        )

    # ── All models comparison ──
    if args.compare:
        print("Model comparison: all 10 models")
        run_compare(
            models=None,  # All models
            cv_mode="subunit",
            n_trials=args.n_trials or 30,
        )

    # ── Ablation ──
    if args.ablation:
        if args.ablation_all:
            print("Ablation study: ALL models")
            run_ablation_all(
                models=None,
                cv_mode="subunit",
                n_trials=args.n_trials or 30,
            )
        else:
            print(f"Ablation study: {args.model}")
            run_ablation(
                model_name=args.model,
                cv_mode="subunit",
                n_trials=args.n_trials or 30,
            )

    # ── Species transfer ──
    if args.species_transfer:
        print(f"Species transfer: {args.model}")
        run_species_transfer_experiment(
            model_name=args.model,
            n_trials=args.n_trials or 30,
        )


if __name__ == "__main__":
    main()
