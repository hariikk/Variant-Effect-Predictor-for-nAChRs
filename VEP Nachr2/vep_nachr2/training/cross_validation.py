"""
Nested cross-validation with Optuna hyperparameter optimization.

Primary: Leave-One-Subunit-Out CV (group by gene)
Secondary: Homology-class transfer CV
Supports: Species transfer CV (human_only vs mouse_only vs mixed)

Based on VEP-ENAC's cross_validation.py, adapted for 16 nAChR genes
and 3-class prediction.
"""

import json
import time
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.preprocessing import RobustScaler

from vep_nachr2.config import (
    LABEL_NAMES, NACHR_GENES, HOMOLOGY_CLASSES,
    DEFAULT_SEEDS, SPECIES_TRANSFER_SEEDS,
    get_config,
)
from vep_nachr2.models.registry import suggest_and_build, build_model, get_model_config
from vep_nachr2.models.imbalance import apply_strategy, compute_sample_weights, get_strategy
from vep_nachr2.training.evaluation import compute_metrics, aggregate_metrics


# =============================================================================
# INNER CV — OPTUNA HYPERPARAMETER OPTIMIZATION
# =============================================================================

def _inner_objective(
    trial: optuna.Trial,
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    inner_cv: StratifiedKFold,
    groups: Optional[np.ndarray] = None,
    use_scaling: bool = True,
) -> float:
    """
    Optuna objective: macro F1 on inner CV.

    Parameters
    ----------
    trial : optuna.Trial
    model_name : str
        Model identifier.
    X_train : np.ndarray
        Inner training features.
    y_train : np.ndarray
        Inner training labels.
    inner_cv : StratifiedKFold
        Inner CV splitter (pre-split to avoid randomness per trial).
    groups : np.ndarray, optional
        Group labels for stratified group splitting.
    use_scaling : bool
        Whether to apply RobustScaler (for distance-based models).

    Returns
    -------
    float
        Mean macro F1 across inner CV folds.
    """
    scores = []

    for fold_idx, (train_idx, val_idx) in enumerate(inner_cv.split(X_train, y_train, groups)):
        X_tr = X_train[train_idx]
        X_val = X_train[val_idx]
        y_tr = y_train[train_idx]
        y_val = y_train[val_idx]

        # Build model with suggested hyperparameters
        model = suggest_and_build(trial, model_name, X_tr, y_tr)

        # Apply scaling if needed
        needs_scaling = get_model_config(model_name)["needs_scaling"]
        if needs_scaling and use_scaling:
            scaler = RobustScaler()
            X_tr = scaler.fit_transform(X_tr)
            X_val = scaler.transform(X_val)

        # Apply imbalance strategy
        strategy = get_strategy(model_name)
        model = apply_strategy(model, model_name, X_tr, y_tr)

        # Fit
        try:
            if strategy == "xgb_inverse":
                sw = compute_sample_weights(y_tr)
                # Handle Pipeline wrapping
                if hasattr(model, "fit") and not hasattr(model, "named_steps"):
                    model.fit(X_tr, y_tr, sample_weight=sw)
                else:
                    model.fit(X_tr, y_tr)
            else:
                model.fit(X_tr, y_tr)

            # Predict
            y_pred = model.predict(X_val)

            from sklearn.metrics import f1_score
            f1 = f1_score(y_val, y_pred, average="macro")
            scores.append(f1)

        except Exception as e:
            # Penalize failed trials
            scores.append(0.0)
            if trial.number < 3:  # Only warn for early trials
                warnings.warn(f"Inner fold {fold_idx} failed for {model_name}: {e}")

    return np.mean(scores) if scores else 0.0


# =============================================================================
# OUTER CV — NESTED CROSS-VALIDATION
# =============================================================================

def nested_cross_validation(
    X: np.ndarray,
    y: np.ndarray,
    df: pd.DataFrame,
    model_name: str = "random_forest",
    n_outer_folds: int = 5,
    n_inner_folds: int = 5,
    n_trials: int = 50,
    seeds: Optional[list[int]] = None,
    cv_mode: str = "subunit",
    verbose: bool = True,
) -> dict:
    """
    Run nested cross-validation with Optuna HP tuning.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix (n_samples, n_features).
    y : np.ndarray
        Integer labels (0=LOF, 1=NNE, 2=GOF).
    df : pd.DataFrame
        Metadata DataFrame (needed for group keys).
    model_name : str
        Model identifier.
    n_outer_folds : int
        Number of outer CV folds.
    n_inner_folds : int
        Number of inner CV folds (for HP tuning).
    n_trials : int
        Optuna trials per outer fold.
    seeds : list[int], optional
        Seeds for reproducibility.
    cv_mode : str
        "subunit" (leave-one-gene-out) or "standard" (stratified K-fold).
    verbose : bool
        Print progress.

    Returns
    -------
    dict
        Results with per-fold metrics, aggregated stats, and best hyperparameters.
    """
    if seeds is None:
        seeds = DEFAULT_SEEDS

    if verbose:
        print(f"\n{'='*60}")
        print(f"Nested CV: {model_name}")
        print(f"  Mode: {cv_mode}, Outer folds: {n_outer_folds}, "
              f"Inner folds: {n_inner_folds}, Trials: {n_trials}")
        print(f"  Dataset: {X.shape[0]} samples, {X.shape[1]} features, {len(np.unique(y))} classes")
        print(f"  Seeds: {seeds}")
        print(f"{'='*60}\n")

    all_results = []

    for seed_idx, seed in enumerate(seeds):
        if verbose:
            print(f"--- Seed {seed} ({seed_idx + 1}/{len(seeds)}) ---")

        # Build outer CV splitter
        if cv_mode == "subunit":
            from vep_nachr2.data.loader import make_subunit_group_key
            groups = make_subunit_group_key(df)
            outer_cv = StratifiedGroupKFold(
                n_splits=n_outer_folds, shuffle=True, random_state=seed
            )
        else:
            groups = None
            outer_cv = StratifiedKFold(
                n_splits=n_outer_folds, shuffle=True, random_state=seed
            )

        fold_results = []

        for fold_idx, (train_idx, test_idx) in enumerate(
            outer_cv.split(X, y, groups)
        ):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            df_test = df.iloc[test_idx]

            if verbose:
                print(f"  Fold {fold_idx + 1}/{n_outer_folds}: "
                      f"train={len(y_train)}, test={len(y_test)} | "
                      f"classes: {dict(zip(*np.unique(y_train, return_counts=True)))}")

            # Inner CV for HP tuning
            inner_cv = StratifiedKFold(
                n_splits=n_inner_folds, shuffle=True, random_state=seed + fold_idx
            )

            # Suppress Optuna logs
            optuna.logging.set_verbosity(optuna.logging.WARNING)

            study = optuna.create_study(
                direction="maximize",
                sampler=optuna.samplers.TPESampler(seed=seed + fold_idx),
                pruner=optuna.pruners.MedianPruner(),
            )

            study.optimize(
                lambda trial: _inner_objective(
                    trial, model_name, X_train, y_train, inner_cv,
                    groups=None,
                    use_scaling=get_model_config(model_name)["needs_scaling"],
                ),
                n_trials=n_trials,
                show_progress_bar=False,
            )

            best_params = study.best_params
            best_inner_f1 = study.best_value

            if verbose:
                print(f"    Best inner F1: {best_inner_f1:.4f}")

            # Build best model and fit on full outer train
            model = build_model(model_name, best_params)

            # Apply imbalance strategy
            model = apply_strategy(model, model_name, X_train, y_train)

            # Scale if needed
            needs_scaling = get_model_config(model_name)["needs_scaling"]
            if needs_scaling:
                scaler = RobustScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
            else:
                X_train_scaled = X_train
                X_test_scaled = X_test

            # Fit
            try:
                strategy = get_strategy(model_name)
                if strategy == "xgb_inverse":
                    sw = compute_sample_weights(y_train)
                    model.fit(X_train_scaled, y_train, sample_weight=sw)
                else:
                    model.fit(X_train_scaled, y_train)

                y_pred = model.predict(X_test_scaled)
                y_proba = (
                    model.predict_proba(X_test_scaled)
                    if hasattr(model, "predict_proba")
                    else None
                )
            except Exception as e:
                warnings.warn(f"Model fitting failed: {e}")
                y_pred = np.zeros_like(y_test)
                y_proba = None

            # Compute metrics
            metrics = compute_metrics(y_test, y_pred, y_proba, labels=[0, 1, 2])

            fold_result = {
                "seed": seed,
                "fold": fold_idx,
                "train_size": len(y_train),
                "test_size": len(y_test),
                "best_params": best_params,
                "best_inner_f1": best_inner_f1,
                "test_genes": sorted(df_test["subunit"].unique().tolist()),
                "metrics": metrics,
            }
            fold_results.append(fold_result)

            if verbose:
                print(f"    Test macro F1: {metrics['macro_f1']:.4f}, "
                      f"MCC: {metrics['mcc']:.4f}")

        # Aggregate across folds for this seed
        seed_aggregate = aggregate_metrics(fold_results)
        seed_aggregate["seed"] = seed
        all_results.append({
            "seed": seed,
            "folds": fold_results,
            "aggregate": seed_aggregate,
        })

    # Final aggregate across seeds
    final = _aggregate_across_seeds(all_results)

    if verbose:
        print(f"\n{'='*60}")
        print(f"Final Results: {model_name}")
        print(f"  Macro F1: {final['macro_f1_mean']:.4f} +/- {final['macro_f1_std']:.4f}")
        print(f"  MCC:      {final['mcc_mean']:.4f} +/- {final['mcc_std']:.4f}")
        print(f"  Bal Acc:  {final['balanced_accuracy_mean']:.4f} +/- {final['balanced_accuracy_std']:.4f}")
        print(f"{'='*60}")

    return {
        "model": model_name,
        "cv_mode": cv_mode,
        "seeds": seeds,
        "per_seed": all_results,
        "final": final,
    }


def _aggregate_across_seeds(results: list[dict]) -> dict:
    """Aggregate metrics across all seeds."""
    metrics_keys = ["macro_f1", "mcc", "balanced_accuracy",
                    "accuracy", "precision_macro", "recall_macro"]

    aggregated = {}
    for key in metrics_keys:
        values = []
        for seed_result in results:
            agg = seed_result["aggregate"]
            if f"{key}_mean" in agg:
                values.append(agg[f"{key}_mean"])

        if values:
            aggregated[f"{key}_mean"] = float(np.mean(values))
            aggregated[f"{key}_std"] = float(np.std(values))

    return aggregated


# =============================================================================
# HOMOLOGY-CLASS TRANSFER CV
# =============================================================================

def homology_class_transfer_cv(
    X: np.ndarray,
    y: np.ndarray,
    df: pd.DataFrame,
    model_name: str = "random_forest",
    n_trials: int = 50,
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    """
    Cross-family transfer: train on one homology class, test on another.

    Three experiments:
      1. alpha → beta+special  (train on α subunits, test on β/δ/γ/ε)
      2. beta → alpha+special
      3. special → alpha+beta

    Parameters
    ----------
    X, y, df : standard
    model_name : str
    n_trials : int
    seed : int
    verbose : bool

    Returns
    -------
    dict with results for each transfer direction.
    """
    results = {}

    transfer_pairs = [
        ("alpha_to_rest", ["alpha"], ["beta", "special"]),
        ("beta_to_rest", ["beta"], ["alpha", "special"]),
        ("special_to_rest", ["special"], ["alpha", "beta"]),
    ]

    for name, train_classes, test_classes in transfer_pairs:
        # Map classes to genes
        train_genes = []
        for cls in train_classes:
            train_genes.extend(HOMOLOGY_CLASSES[cls])
        test_genes = []
        for cls in test_classes:
            test_genes.extend(HOMOLOGY_CLASSES[cls])

        # Split data
        train_mask = df["subunit"].isin(train_genes)
        test_mask = df["subunit"].isin(test_genes)

        if not train_mask.any() or not test_mask.any():
            warnings.warn(f"No data for {name}: train={train_mask.sum()}, test={test_mask.sum()}")
            continue

        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]

        if verbose:
            print(f"\n{name}: train={len(y_train)} (genes={train_genes}), "
                  f"test={len(y_test)} (genes={test_genes})")

        # Inner CV for HP tuning
        inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        sampler = optuna.samplers.TPESampler(seed=seed)

        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(
            lambda trial: _inner_objective(
                trial, model_name, X_train, y_train, inner_cv
            ),
            n_trials=n_trials,
            show_progress_bar=False,
        )

        # Build best model, fit, predict
        model = build_model(model_name, study.best_params)
        model = apply_strategy(model, model_name, X_train, y_train)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        metrics = compute_metrics(y_test, y_pred, None, labels=[0, 1, 2])
        metrics["best_params"] = study.best_params
        results[name] = metrics

        if verbose:
            print(f"  Macro F1: {metrics['macro_f1']:.4f}, MCC: {metrics['mcc']:.4f}")

    return results


# =============================================================================
# SPECIES TRANSFER CV
# =============================================================================

def species_transfer_cv(
    X: np.ndarray,
    y: np.ndarray,
    df: pd.DataFrame,
    model_name: str = "random_forest",
    n_trials: int = 50,
    seeds: Optional[list[int]] = None,
    verbose: bool = True,
) -> dict:
    """
    Species transfer experiment: test on human only under 3 training conditions.

    1. human_only: train on human, test on human (standard CV)
    2. mouse_only: train on mouse, test on human (cross-species)
    3. mixed: train on human + mouse + rat, test on human (augmented)

    Uses identical human test folds across all 3 conditions for paired comparison.
    """
    if seeds is None:
        seeds = SPECIES_TRANSFER_SEEDS

    # Split by species
    human_mask = df["species"].str.lower() == "human"
    mouse_mask = df["species"].str.lower() == "mouse"
    rat_mask = df["species"].str.lower() == "rat"

    X_human, y_human = X[human_mask], y[human_mask]
    X_mouse, y_mouse = X[mouse_mask], y[mouse_mask]
    X_rat, y_rat = X[rat_mask], y[rat_mask]

    if verbose:
        print(f"Species transfer: human={len(y_human)}, mouse={len(y_mouse)}, rat={len(y_rat)}")

    results = {"human_only": [], "mouse_only": [], "mixed": []}

    for seed_idx, seed in enumerate(seeds):
        # Create consistent human test folds
        outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

        for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X_human, y_human)):
            X_human_train, X_human_test = X_human[train_idx], X_human[test_idx]
            y_human_train, y_human_test = y_human[train_idx], y_human[test_idx]

            # Condition 1: human only
            f1_human = _train_and_evaluate(
                model_name, X_human_train, y_human_train, X_human_test, y_human_test,
                n_trials, seed + fold_idx
            )
            results["human_only"].append(f1_human)

            # Condition 2: mouse only → human
            if len(y_mouse) >= 10:
                f1_mouse = _train_and_evaluate(
                    model_name, X_mouse, y_mouse, X_human_test, y_human_test,
                    n_trials, seed + fold_idx
                )
                results["mouse_only"].append(f1_mouse)

            # Condition 3: mixed
            X_mixed_train = np.vstack([X_human_train, X_mouse, X_rat]) if len(y_rat) > 0 else np.vstack([X_human_train, X_mouse])
            y_mixed_train = np.hstack([y_human_train, y_mouse, y_rat]) if len(y_rat) > 0 else np.hstack([y_human_train, y_mouse])
            f1_mixed = _train_and_evaluate(
                model_name, X_mixed_train, y_mixed_train, X_human_test, y_human_test,
                n_trials, seed + fold_idx
            )
            results["mixed"].append(f1_mixed)

    # Aggregate
    aggregated = {}
    for condition, f1s in results.items():
        if f1s:
            aggregated[condition] = {
                "macro_f1_mean": float(np.mean(f1s)),
                "macro_f1_std": float(np.std(f1s)),
                "n_folds": len(f1s),
            }

    if verbose:
        print("\nSpecies Transfer Results:")
        for condition, stats in aggregated.items():
            print(f"  {condition}: F1 = {stats['macro_f1_mean']:.4f} +/- {stats['macro_f1_std']:.4f}")

    return aggregated


def _train_and_evaluate(model_name, X_tr, y_tr, X_te, y_te, n_trials, seed) -> float:
    """Quick train+evaluate, returning macro F1."""
    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )

    study.optimize(
        lambda trial: _inner_objective(trial, model_name, X_tr, y_tr, inner_cv),
        n_trials=min(n_trials, 30),  # fewer trials for speed
        show_progress_bar=False,
    )

    model = build_model(model_name, study.best_params)
    model = apply_strategy(model, model_name, X_tr, y_tr)

    try:
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        from sklearn.metrics import f1_score
        return float(f1_score(y_te, y_pred, average="macro"))
    except Exception:
        return 0.0


# =============================================================================
# RESULT SAVING
# =============================================================================

def save_results(results: dict, output_dir: Path, experiment_name: str = "cv_results"):
    """Save CV results to JSON and CSV."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save full results as JSON
    json_path = output_dir / f"{experiment_name}.json"

    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    with open(json_path, "w") as f:
        json.dump(convert(results), f, indent=2)

    # Save summary as CSV
    csv_path = output_dir / f"{experiment_name}_summary.csv"
    final = results.get("final", {})
    summary = pd.DataFrame([final])
    summary.to_csv(csv_path, index=False)

    return json_path, csv_path
