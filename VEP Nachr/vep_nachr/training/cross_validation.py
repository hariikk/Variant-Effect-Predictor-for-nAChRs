"""
Cross-validation strategies for nAChR VEP experiments.

Provides:
- Nested cross-validation with Optuna HP optimization
- Simple cross-validation with fixed hyperparameters
- Multi-seed evaluation for robust estimates

Adapted from VEP-ENaC for binary classification (LOF vs GOF).
"""

import gc
import json
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    accuracy_score,
)
from tqdm import tqdm

from vep_nachr.config import DEFAULT_SEEDS, PRIMARY_SEED
from vep_nachr.models.registry import (
    get_model,
    get_hyperparameter_space,
    AVAILABLE_MODELS,
    OPTUNA_AVAILABLE,
)

if OPTUNA_AVAILABLE:
    import optuna
    from optuna.samplers import TPESampler

# Suppress warnings during HP search
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", message=".*X does not have valid feature names.*")


# =============================================================================
# RESULT DATACLASSES
# =============================================================================

@dataclass
class FoldResult:
    """Results from a single CV fold."""
    fold_idx: int
    y_true: np.ndarray
    y_pred: np.ndarray
    f1_score: float
    accuracy: float
    best_params: Optional[dict] = None


@dataclass
class SeedResult:
    """Results from all folds for a single seed."""
    seed: int
    fold_results: list[FoldResult]
    mean_f1: float
    std_f1: float
    mean_accuracy: float
    confusion_matrix: np.ndarray
    classification_report: dict


@dataclass
class ExperimentResult:
    """Complete experiment results across all seeds."""
    model_name: str
    encoding: str
    seed_results: list[SeedResult]
    overall_mean_f1: float
    overall_std_f1: float
    overall_mean_accuracy: float
    per_seed_f1: list[float]
    n_samples: int
    n_features: int
    n_outer_folds: int
    n_inner_folds: int
    n_seeds: int
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "model_name": self.model_name,
            "encoding": self.encoding,
            "overall_mean_f1": self.overall_mean_f1,
            "overall_std_f1": self.overall_std_f1,
            "overall_mean_accuracy": self.overall_mean_accuracy,
            "per_seed_f1": self.per_seed_f1,
            "n_samples": self.n_samples,
            "n_features": self.n_features,
            "n_outer_folds": self.n_outer_folds,
            "n_inner_folds": self.n_inner_folds,
            "n_seeds": self.n_seeds,
            "timestamp": self.timestamp,
            "seed_results": [
                {
                    "seed": sr.seed,
                    "mean_f1": sr.mean_f1,
                    "std_f1": sr.std_f1,
                    "mean_accuracy": sr.mean_accuracy,
                    "confusion_matrix": sr.confusion_matrix.tolist(),
                    "classification_report": sr.classification_report,
                }
                for sr in self.seed_results
            ],
        }

    def save(self, path: str | Path) -> None:
        """Save results to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def summary(self) -> str:
        """Get a formatted summary string."""
        return (
            f"{self.model_name} ({self.encoding}): "
            f"F1={self.overall_mean_f1:.4f} +/- {self.overall_std_f1:.4f}, "
            f"Acc={self.overall_mean_accuracy:.4f} "
            f"[{self.n_samples} samples, {self.n_features} features]"
        )


# =============================================================================
# NESTED CROSS-VALIDATION
# =============================================================================

def nested_cross_validation(
    X: np.ndarray,
    y: np.ndarray,
    model_name: str,
    encoding: str = "engineered",
    n_outer_folds: int = 5,
    n_inner_folds: int = 5,
    seeds: Optional[list[int]] = None,
    n_trials: int = 50,
    n_jobs: int = 1,
    verbose: int = 1,
    feature_names: Optional[list[str]] = None,
) -> ExperimentResult:
    """
    Perform nested cross-validation with Optuna HP optimization.

    Outer loop: evaluate model performance on held-out test folds.
    Inner loop: optimize hyperparameters using Optuna.
    Multiple seeds for robust performance estimates.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix (n_samples, n_features).
    y : np.ndarray
        Binary labels (0=LOF, 1=GOF).
    model_name : str
        Model to evaluate.
    encoding : str
        Name of encoding strategy (for logging).
    n_outer_folds : int
        Number of outer CV folds.
    n_inner_folds : int
        Number of inner CV folds for HP optimization.
    seeds : list[int], optional
        Random seeds for CV splits.
    n_trials : int
        Number of Optuna trials per fold.
    n_jobs : int
        Parallel jobs for cross_val_score.
    verbose : int
        Verbosity level (0=silent, 1=progress, 2=detailed).
    feature_names : list[str], optional
        Feature names for importance analysis.

    Returns
    -------
    ExperimentResult
        Complete results across all seeds.
    """
    if seeds is None:
        seeds = DEFAULT_SEEDS

    if not OPTUNA_AVAILABLE:
        raise ImportError("Optuna required. Install with: pip install optuna")

    n_samples, n_features = X.shape
    model_config = AVAILABLE_MODELS.get(model_name)
    needs_scaling = model_config.requires_scaling if model_config else True

    seed_results = []
    all_f1_scores = []

    if verbose:
        print(f"\n{'='*60}")
        print(f"Model: {model_name} | Encoding: {encoding}")
        print(f"Samples: {n_samples} | Features: {n_features}")
        print(f"LOF: {(y == 0).sum()} | GOF: {(y == 1).sum()}")
        print(f"{'='*60}")

    for seed in (tqdm(seeds, desc="Seeds") if verbose else seeds):
        fold_results = []

        outer_cv = StratifiedKFold(
            n_splits=n_outer_folds, shuffle=True, random_state=seed
        )

        for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Scale if needed
            if needs_scaling:
                scaler = RobustScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)

            # Inner HP optimization
            best_params = _optimize_hyperparameters(
                X_train, y_train,
                model_name=model_name,
                n_folds=n_inner_folds,
                n_trials=n_trials,
                seed=seed,
                n_jobs=n_jobs,
                verbose=verbose > 1,
            )

            # Train final model with best params
            model = get_model(model_name, params=best_params, random_state=seed, n_jobs=n_jobs)
            model.fit(X_train, y_train)

            # Predict
            y_pred = model.predict(X_test)
            fold_f1 = f1_score(y_test, y_pred, average="binary")
            fold_acc = accuracy_score(y_test, y_pred)

            fold_results.append(FoldResult(
                fold_idx=fold_idx,
                y_true=y_test,
                y_pred=y_pred,
                f1_score=fold_f1,
                accuracy=fold_acc,
                best_params=best_params,
            ))

        # Aggregate fold results for this seed
        fold_f1s = [fr.f1_score for fr in fold_results]
        fold_accs = [fr.accuracy for fr in fold_results]
        all_y_true = np.concatenate([fr.y_true for fr in fold_results])
        all_y_pred = np.concatenate([fr.y_pred for fr in fold_results])

        seed_result = SeedResult(
            seed=seed,
            fold_results=fold_results,
            mean_f1=np.mean(fold_f1s),
            std_f1=np.std(fold_f1s),
            mean_accuracy=np.mean(fold_accs),
            confusion_matrix=confusion_matrix(all_y_true, all_y_pred),
            classification_report=classification_report(
                all_y_true, all_y_pred, output_dict=True, zero_division=0
            ),
        )
        seed_results.append(seed_result)
        all_f1_scores.append(seed_result.mean_f1)

        if verbose:
            print(f"  Seed {seed}: F1={seed_result.mean_f1:.4f} +/- {seed_result.std_f1:.4f}")

    result = ExperimentResult(
        model_name=model_name,
        encoding=encoding,
        seed_results=seed_results,
        overall_mean_f1=np.mean(all_f1_scores),
        overall_std_f1=np.std(all_f1_scores),
        overall_mean_accuracy=np.mean([sr.mean_accuracy for sr in seed_results]),
        per_seed_f1=all_f1_scores,
        n_samples=n_samples,
        n_features=n_features,
        n_outer_folds=n_outer_folds,
        n_inner_folds=n_inner_folds,
        n_seeds=len(seeds),
    )

    if verbose:
        print(f"\n>>> {result.summary()}")

    return result


def _optimize_hyperparameters(
    X: np.ndarray,
    y: np.ndarray,
    model_name: str,
    n_folds: int,
    n_trials: int,
    seed: int,
    n_jobs: int = 1,
    verbose: bool = False,
) -> dict:
    """Run Optuna optimization for hyperparameters."""
    n_feats = X.shape[1]

    def objective(trial):
        params = get_hyperparameter_space(model_name, trial, n_features=n_feats)
        model = get_model(model_name, params=params, random_state=seed, n_jobs=1)
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scores = cross_val_score(
                model, X, y, cv=cv, scoring="f1", n_jobs=min(n_jobs, n_folds)
            )
        return np.mean(scores)

    sampler = TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=n_trials, n_jobs=1, show_progress_bar=verbose)

    best_params = get_hyperparameter_space(model_name, study.best_trial, n_features=n_feats)

    del study, sampler
    gc.collect()

    return best_params


# =============================================================================
# SIMPLE CROSS-VALIDATION (for quick testing)
# =============================================================================

def simple_cross_validation(
    X: np.ndarray,
    y: np.ndarray,
    model_name: str,
    params: Optional[dict] = None,
    n_folds: int = 5,
    seeds: Optional[list[int]] = None,
    n_jobs: int = -1,
) -> dict:
    """
    Simple cross-validation without HP optimization.

    Useful for quick evaluation with fixed hyperparameters.

    Returns
    -------
    dict
        Results with mean/std F1 and accuracy.
    """
    if seeds is None:
        seeds = DEFAULT_SEEDS

    from sklearn.pipeline import Pipeline

    model_config = AVAILABLE_MODELS.get(model_name)
    needs_scaling = model_config.requires_scaling if model_config else True

    all_f1 = []
    all_acc = []

    for seed in seeds:
        model = get_model(model_name, params=params, random_state=seed)
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

        if needs_scaling:
            pipeline = Pipeline([("scaler", RobustScaler()), ("model", model)])
        else:
            pipeline = model

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f1_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="f1", n_jobs=n_jobs)
            acc_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy", n_jobs=n_jobs)

        all_f1.extend(f1_scores)
        all_acc.extend(acc_scores)

    return {
        "mean_f1": np.mean(all_f1),
        "std_f1": np.std(all_f1),
        "mean_accuracy": np.mean(all_acc),
        "std_accuracy": np.std(all_acc),
        "n_seeds": len(seeds),
        "n_folds": n_folds,
    }
