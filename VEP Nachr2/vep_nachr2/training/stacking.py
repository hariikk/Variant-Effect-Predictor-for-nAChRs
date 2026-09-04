"""
Ensemble stacking over the 10-model panel (idea #4).

A cheap logistic meta-learner is trained on the *out-of-fold* predicted class
probabilities of every base model, rather than on their raw features. This
captures the fact that CatBoost / RF / XGBoost / LightGBM / ... make different
errors — a pattern the base models individually cannot exploit.

Leakage discipline: the meta-learner's training features (the base-model
probabilities) are produced by *gene-grouped* inner CV on the training folds
only, matching the outer leave-one-subunit-out split. No test-gene variant is
ever seen by a base model during the OOF pass, so the meta-learner never peeks
at the test distribution.

Base models are built with default hyperparameters (no Optuna) for speed — this
is the 1-seed screen; if stacking shows promise the 5-seed run can tune the
base models. Each base model still gets its per-model scaling + imbalance
strategy, exactly as in the standalone pipeline.
"""

import warnings
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, train_test_split
from sklearn.preprocessing import RobustScaler

from vep_nachr2.config import ALL_MODELS, LABEL_NAMES, DEFAULT_SEEDS
from vep_nachr2.data.loader import make_subunit_group_key
from vep_nachr2.models.registry import build_model, get_model_config
from vep_nachr2.models.imbalance import apply_strategy
from vep_nachr2.training.evaluation import compute_metrics, aggregate_metrics


def _build_scaled_base(model_name, X_tr, y_tr, n_classes):
    """Build + strategy-apply a base model and scale X_tr for training."""
    model = build_model(model_name, {}, n_classes=n_classes)
    model = apply_strategy(model, model_name, X_tr, y_tr)
    needs_scaling = get_model_config(model_name)["needs_scaling"]
    if needs_scaling:
        scaler = RobustScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        return model, X_tr_s, scaler
    return model, X_tr, None


def _fit_predict_proba(model, X_tr, y_tr, X_te, scaler):
    """Fit a (possibly pre-scaled) base model and return predict_proba on X_te."""
    try:
        model.fit(X_tr, y_tr)
        if scaler is not None:
            X_te_s = scaler.transform(X_te)
        else:
            X_te_s = X_te
        return model.predict_proba(X_te_s)
    except Exception as e:  # noqa: BLE001
        warnings.warn(f"base model failed during stacking: {e}")
        # Fall back to a uniform distribution over classes.
        n_classes = len(np.unique(y_tr))
        return np.full((len(X_te), n_classes), 1.0 / n_classes)


def stacked_cross_validation(
    X: np.ndarray,
    y: np.ndarray,
    df: pd.DataFrame,
    base_models: Optional[list[str]] = None,
    n_outer_folds: int = 5,
    n_inner_folds: int = 5,
    seeds: Optional[list[int]] = None,
    cv_mode: str = "subunit",
    verbose: bool = True,
    n_classes: int = 3,
    label_names: Optional[dict] = None,
) -> dict:
    """
    Nested-CV ensemble stacking with a logistic meta-learner.

    Returns the same dict shape as `nested_cross_validation` (per_seed/final),
    so the two can be compared directly on macro F1 / MCC / balanced accuracy.
    """
    if base_models is None:
        base_models = ALL_MODELS
    if seeds is None:
        seeds = DEFAULT_SEEDS
    if label_names is None:
        label_names = LABEL_NAMES

    if verbose:
        print(f"\n{'='*60}")
        print(f"Stacked CV: {len(base_models)} base models + logistic meta-learner")
        print(f"  Mode: {cv_mode}, Outer folds: {n_outer_folds}, Inner folds: {n_inner_folds}")
        print(f"  Dataset: {X.shape[0]} samples, {X.shape[1]} features, {n_classes} classes")
        print(f"  Seeds: {seeds}")
        print(f"{'='*60}\n")

    all_results = []

    for seed_idx, seed in enumerate(seeds):
        if verbose:
            print(f"--- Seed {seed} ({seed_idx + 1}/{len(seeds)}) ---")

        # Outer CV splitter — identical to nested_cross_validation.
        if cv_mode == "holdout":
            groups = None
            train_idx, test_idx = train_test_split(
                np.arange(len(y)), test_size=0.2, stratify=y, random_state=seed
            )
            split_pairs = [(train_idx, test_idx)]
        elif cv_mode == "subunit":
            groups = make_subunit_group_key(df)
            outer_cv = StratifiedGroupKFold(
                n_splits=n_outer_folds, shuffle=True, random_state=seed
            )
            split_pairs = list(outer_cv.split(X, y, groups))
        else:
            groups = None
            outer_cv = StratifiedKFold(
                n_splits=n_outer_folds, shuffle=True, random_state=seed
            )
            split_pairs = list(outer_cv.split(X, y, groups))

        n_base = len(base_models)
        fold_results = []

        for fold_idx, (train_idx, test_idx) in enumerate(split_pairs):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            df_test = df.iloc[test_idx]

            if verbose:
                print(f"  Fold {fold_idx + 1}/{n_outer_folds}: "
                      f"train={len(y_train)}, test={len(y_test)}")

            n_train, n_test = len(y_train), len(y_test)
            oof_proba = np.zeros((n_train, n_base * n_classes))   # meta-features (train)
            test_proba = np.zeros((n_test, n_base * n_classes))   # meta-features (test)

            for bi, mname in enumerate(base_models):
                cols = slice(bi * n_classes, (bi + 1) * n_classes)

                # 1) Out-of-fold probabilities on the training set via inner CV.
                inner_cv = StratifiedKFold(
                    n_splits=n_inner_folds, shuffle=True, random_state=seed + fold_idx
                )
                for tr_idx, val_idx in inner_cv.split(X_train, y_train):
                    model, X_tr_s, scaler = _build_scaled_base(
                        mname, X_train[tr_idx], y_train[tr_idx], n_classes
                    )
                    oof_proba[val_idx, cols] = _fit_predict_proba(
                        model, X_tr_s, y_train[tr_idx], X_train[val_idx], scaler
                    )

                # 2) Refit on the full training fold, predict the test fold.
                model, X_tr_s, scaler = _build_scaled_base(
                    mname, X_train, y_train, n_classes
                )
                test_proba[:, cols] = _fit_predict_proba(
                    model, X_tr_s, y_train, X_test, scaler
                )

            # 3) Meta-learner: logistic regression on the stacked probabilities.
            meta = LogisticRegression(
                C=1.0, max_iter=2000, class_weight="balanced", random_state=seed
            )
            meta.fit(oof_proba, y_train)
            y_pred = meta.predict(test_proba)
            y_proba = meta.predict_proba(test_proba)

            metrics = compute_metrics(
                y_test, y_pred, y_proba,
                labels=list(range(n_classes)), label_names=label_names,
            )

            fold_result = {
                "seed": seed,
                "fold": fold_idx,
                "train_size": n_train,
                "test_size": n_test,
                "test_genes": sorted(df_test["subunit"].unique().tolist()),
                "metrics": metrics,
            }
            fold_results.append(fold_result)

            if verbose:
                print(f"    Test macro F1: {metrics['macro_f1']:.4f}, "
                      f"MCC: {metrics['mcc']:.4f}")

        seed_aggregate = aggregate_metrics(fold_results)
        seed_aggregate["seed"] = seed
        all_results.append({
            "seed": seed,
            "folds": fold_results,
            "aggregate": seed_aggregate,
        })

    final = _aggregate_across_seeds(all_results)

    if verbose:
        print(f"\n{'='*60}")
        print("Final Results: stacking ensemble")
        print(f"  Macro F1: {final['macro_f1_mean']:.4f} +/- {final['macro_f1_std']:.4f}")
        print(f"  MCC:      {final['mcc_mean']:.4f} +/- {final['mcc_std']:.4f}")
        print(f"  Bal Acc:  {final['balanced_accuracy_mean']:.4f} +/- {final['balanced_accuracy_std']:.4f}")
        print(f"{'='*60}")

    return {
        "model": "stacking",
        "base_models": base_models,
        "cv_mode": cv_mode,
        "seeds": seeds,
        "per_seed": all_results,
        "final": final,
    }


def _aggregate_across_seeds(results: list[dict]) -> dict:
    metrics_keys = ["macro_f1", "mcc", "balanced_accuracy",
                    "accuracy", "precision_macro", "recall_macro"]
    aggregated = {}
    for key in metrics_keys:
        values = [r["aggregate"][f"{key}_mean"]
                  for r in results if f"{key}_mean" in r["aggregate"]]
        if values:
            aggregated[f"{key}_mean"] = float(np.mean(values))
            aggregated[f"{key}_std"] = float(np.std(values))
    return aggregated
