#!/usr/bin/env python3
"""
Feature and feature-group ablation for nAChR VEP.

Two modes:
1. **Per-feature ablation** (--per-feature): Drop each feature individually,
   retrain, measure F1 delta. Ranks features by importance.
2. **Per-group ablation** (--per-group or default): Leave-one-group-out
   using FEATURE_GROUPS defined in config. Drops entire groups at once.

Usage:
    # Per-group ablation (default)
    python scripts/experiments/run_ablation.py --model random_forest

    # Per-feature ablation (ranks every feature)
    python scripts/experiments/run_ablation.py --model random_forest --per-feature

    # Both modes
    python scripts/experiments/run_ablation.py --model random_forest --all

    # All models, quick mode
    python scripts/experiments/run_ablation.py --quick --per-group
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from vep_nachr.config import (
    FEATURE_GROUPS,
    CORE_MODELS,
    RESULTS_DIR,
    DEFAULT_SEEDS,
)
from vep_nachr.data.loader import load_dataset
from vep_nachr.features.encoder import NachrFeatureEncoder
from vep_nachr.training.cross_validation import nested_cross_validation
from vep_nachr.models.registry import AVAILABLE_MODELS


# =============================================================================
# FEATURE → GROUP MAPPING
# =============================================================================

# Which features belong to which group (for engineered encoding)
# Derived from the NachrFeatureEncoder output order.
_ENG_AA_PROPS = [
    "hydrophobicity", "polarity", "volume", "molecular_weight",
    "charge", "isoelectric_point", "aromaticity", "secondary_structure_preference",
]

ENGINEERED_GROUP_MAP: dict[str, list[str]] = {
    "physicochemical": [
        f"{prefix}_{prop}"
        for prefix in ("wt", "mt", "diff")
        for prop in _ENG_AA_PROPS
    ],  # 24 features
    "substitution": [
        "blosum_raw", "blosum_normalized", "grantham_normalized",
    ],  # 3 features
    "positional": [
        "position_normalized",
        *[f"subunit_{s}" for s in [
            "CHRNA1", "CHRNA2", "CHRNA3", "CHRNA4", "CHRNA5",
            "CHRNA6", "CHRNA7", "CHRNA9", "CHRNA10",
            "CHRNB1", "CHRNB2", "CHRNB3", "CHRNB4",
            "CHRND", "CHRNE", "CHRNG",
        ]],
    ],  # 17 features
    "structural": [
        "rsa", "bfactor", "dssp_helix", "dssp_sheet", "dssp_coil",
        "cbeta_density", "hse_up", "hse_down",
    ],  # 8 features
}

# Total: 24 + 3 + 17 + 8 = 52 features


# =============================================================================
# RESULT DATACLASSES
# =============================================================================

@dataclass
class AblationResult:
    """Result from a single ablation experiment."""
    dropped: str  # Feature or group name that was dropped
    drop_type: str  # 'feature' or 'group'
    mean_f1: float
    std_f1: float
    f1_drop: float  # Baseline F1 - ablated F1 (positive = feature is important)
    n_features_remaining: int
    baseline_f1: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        return {
            "dropped": self.dropped,
            "drop_type": self.drop_type,
            "mean_f1": self.mean_f1,
            "std_f1": self.std_f1,
            "f1_drop": self.f1_drop,
            "n_features_remaining": self.n_features_remaining,
            "baseline_f1": self.baseline_f1,
            "timestamp": self.timestamp,
        }


@dataclass
class AblationExperimentResult:
    """Complete ablation results for a model."""
    model_name: str
    baseline_f1: float
    baseline_std: float
    n_features_total: int
    per_feature_results: list[AblationResult] = field(default_factory=list)
    per_group_results: list[AblationResult] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "baseline_f1": self.baseline_f1,
            "baseline_std": self.baseline_std,
            "n_features_total": self.n_features_total,
            "per_feature_results": [r.to_dict() for r in self.per_feature_results],
            "per_group_results": [r.to_dict() for r in self.per_group_results],
            "timestamp": self.timestamp,
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def summary(self) -> pd.DataFrame:
        """Return a sorted DataFrame of all ablation results."""
        all_results = self.per_feature_results + self.per_group_results
        rows = [{
            "dropped": r.dropped,
            "type": r.drop_type,
            "f1_drop": r.f1_drop,
            "ablated_f1": r.mean_f1,
            "n_features": r.n_features_remaining,
        } for r in all_results]
        return pd.DataFrame(rows).sort_values("f1_drop", ascending=False)


# =============================================================================
# ABLATION FUNCTIONS
# =============================================================================

def run_baseline(X, y, model_name, n_trials=50, n_jobs=-1, verbose=1) -> tuple[float, float]:
    """Get baseline F1 with all features."""
    result = nested_cross_validation(
        X, y,
        model_name=model_name,
        encoding="engineered",
        n_outer_folds=5, n_inner_folds=5,
        n_trials=n_trials, n_jobs=n_jobs,
        verbose=verbose,
    )
    return result.overall_mean_f1, result.overall_std_f1


def run_per_feature_ablation(
    X, y, feature_names, model_name,
    baseline_f1, n_trials=50, n_jobs=-1, verbose=1,
) -> list[AblationResult]:
    """
    Drop each feature one at a time, retrain, measure F1 delta.

    Returns results sorted by F1 drop (most important first).
    """
    results = []
    n_total = len(feature_names)

    for i, fname in enumerate(feature_names):
        if verbose:
            print(f"  [{i+1}/{n_total}] Dropping: {fname}")

        # Create feature mask (all except this one)
        mask = np.ones(X.shape[1], dtype=bool)
        mask[i] = False
        X_ablated = X[:, mask]

        try:
            result = nested_cross_validation(
                X_ablated, y,
                model_name=model_name,
                encoding=f"ablated_{fname}",
                n_outer_folds=5, n_inner_folds=5,
                seeds=[DEFAULT_SEEDS[0]],  # Single seed for speed
                n_trials=n_trials, n_jobs=n_jobs,
                verbose=0,
            )

            f1_drop = baseline_f1 - result.overall_mean_f1
            results.append(AblationResult(
                dropped=fname,
                drop_type="feature",
                mean_f1=result.overall_mean_f1,
                std_f1=result.overall_std_f1,
                f1_drop=f1_drop,
                n_features_remaining=X_ablated.shape[1],
                baseline_f1=baseline_f1,
            ))

            if verbose:
                direction = "↓" if f1_drop > 0 else "↑"
                print(f"    F1={result.overall_mean_f1:.4f} ({direction}{abs(f1_drop):.4f})")

        except Exception as e:
            if verbose:
                print(f"    FAILED: {e}")

    # Sort by importance (largest drop first)
    results.sort(key=lambda r: r.f1_drop, reverse=True)
    return results


def run_per_group_ablation(
    X, y, feature_names, model_name,
    group_map, baseline_f1,
    n_trials=50, n_jobs=-1, verbose=1,
) -> list[AblationResult]:
    """
    Leave-one-group-out ablation. Drop each feature group, retrain, measure F1 delta.
    """
    results = []
    group_names = list(group_map.keys())

    for group_name in group_names:
        if verbose:
            print(f"  Dropping group: {group_name}")

        group_features = group_map[group_name]
        # Find indices of features in this group
        drop_indices = [i for i, fn in enumerate(feature_names) if fn in group_features]

        if not drop_indices:
            if verbose:
                print(f"    WARNING: No features found for group '{group_name}'")
            continue

        mask = np.ones(X.shape[1], dtype=bool)
        mask[drop_indices] = False
        X_ablated = X[:, mask]

        try:
            result = nested_cross_validation(
                X_ablated, y,
                model_name=model_name,
                encoding=f"ablated_group_{group_name}",
                n_outer_folds=5, n_inner_folds=5,
                n_trials=n_trials, n_jobs=n_jobs,
                verbose=0,
            )

            f1_drop = baseline_f1 - result.overall_mean_f1
            results.append(AblationResult(
                dropped=group_name,
                drop_type="group",
                mean_f1=result.overall_mean_f1,
                std_f1=result.overall_std_f1,
                f1_drop=f1_drop,
                n_features_remaining=X_ablated.shape[1],
                baseline_f1=baseline_f1,
            ))

            if verbose:
                direction = "↓" if f1_drop > 0 else "↑"
                n_dropped = len(drop_indices)
                print(f"    Dropped {n_dropped} features, F1={result.overall_mean_f1:.4f} ({direction}{abs(f1_drop):.4f})")

        except Exception as e:
            if verbose:
                print(f"    FAILED: {e}")

    results.sort(key=lambda r: r.f1_drop, reverse=True)
    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="nAChR VEP — Ablation Studies")
    parser.add_argument("--model", type=str, default="random_forest",
                        help="Model to ablate (default: random_forest)")
    parser.add_argument("--per-feature", action="store_true",
                        help="Run per-feature ablation (drop each feature individually)")
    parser.add_argument("--per-group", action="store_true",
                        help="Run per-group ablation (leave-one-group-out)")
    parser.add_argument("--all", action="store_true",
                        help="Run both per-feature and per-group ablation")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: single seed, fewer trials")
    parser.add_argument("--n-trials", type=int, default=50,
                        help="Optuna trials per fold (default: 50)")
    parser.add_argument("--n-jobs", type=int, default=-1,
                        help="Parallel jobs (default: -1)")
    parser.add_argument("--verbose", type=int, default=1)
    args = parser.parse_args()

    if not args.per_feature and not args.per_group and not args.all:
        parser.error("Choose --per-feature, --per-group, or --all")

    n_trials = 20 if args.quick else args.n_trials

    # --- Load data ---
    print("Loading data...")
    df, labels, sequences = load_dataset()

    # --- Extract engineered features ---
    print("Extracting engineered features...")
    encoder = NachrFeatureEncoder(
        include_structural=True,
        include_substitution=True,
        include_positional=True,
    )
    X = encoder.fit_transform(df)
    feature_names = encoder.get_feature_names_out()
    print(f"Feature matrix: {X.shape}")

    # --- Baseline ---
    print(f"\nComputing baseline for {args.model}...")
    baseline_f1, baseline_std = run_baseline(
        X, labels, args.model, n_trials=n_trials,
        n_jobs=args.n_jobs, verbose=args.verbose,
    )
    print(f"Baseline F1: {baseline_f1:.4f} ± {baseline_std:.4f}")

    experiment = AblationExperimentResult(
        model_name=args.model,
        baseline_f1=baseline_f1,
        baseline_std=baseline_std,
        n_features_total=X.shape[1],
    )

    # --- Per-feature ablation ---
    if args.per_feature or args.all:
        print(f"\n{'='*60}")
        print("PER-FEATURE ABLATION")
        print(f"{'='*60}")
        t0 = time.time()
        experiment.per_feature_results = run_per_feature_ablation(
            X, labels, feature_names, args.model,
            baseline_f1, n_trials=n_trials,
            n_jobs=args.n_jobs, verbose=args.verbose,
        )
        elapsed = time.time() - t0
        print(f"Completed {len(experiment.per_feature_results)} features in {elapsed:.0f}s")

        # Print top-10 most important features
        print("\nTop 10 Most Important Features:")
        for i, r in enumerate(experiment.per_feature_results[:10]):
            print(f"  {i+1}. {r.dropped:40s} F1 drop: {r.f1_drop:+.4f}")

    # --- Per-group ablation ---
    if args.per_group or args.all:
        print(f"\n{'='*60}")
        print("PER-GROUP ABLATION")
        print(f"{'='*60}")
        t0 = time.time()
        experiment.per_group_results = run_per_group_ablation(
            X, labels, feature_names, args.model,
            ENGINEERED_GROUP_MAP, baseline_f1,
            n_trials=n_trials, n_jobs=args.n_jobs,
            verbose=args.verbose,
        )
        elapsed = time.time() - t0
        print(f"Completed {len(experiment.per_group_results)} groups in {elapsed:.0f}s")

        print("\nGroup Importance Ranking:")
        for i, r in enumerate(experiment.per_group_results):
            print(f"  {i+1}. {r.dropped:30s} F1 drop: {r.f1_drop:+.4f}")

    # --- Save ---
    out_dir = RESULTS_DIR / "ablation"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.model}_ablation.json"
    experiment.save(out_path)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
