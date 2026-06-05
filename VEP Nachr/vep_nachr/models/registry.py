"""
Model registry and hyperparameter spaces for nAChR VEP.

Provides a unified interface for creating models and defining Optuna
hyperparameter search spaces. Adapted from VEP-ENaC for binary
classification (LOF vs GOF).
"""

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False


# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

@dataclass
class ModelConfig:
    """Configuration for a model."""
    name: str
    display_name: str
    model_class: type
    default_params: dict = field(default_factory=dict)
    requires_scaling: bool = True
    supports_class_weight: bool = True
    n_jobs_param: Optional[str] = "n_jobs"


AVAILABLE_MODELS: dict[str, ModelConfig] = {
    "logistic_regression": ModelConfig(
        name="logistic_regression",
        display_name="Logistic Regression",
        model_class=LogisticRegression,
        default_params={
            "max_iter": 1000,
            "class_weight": "balanced",
            "random_state": 42,
            "n_jobs": -1,
        },
        requires_scaling=True,
    ),
    "svm_rbf": ModelConfig(
        name="svm_rbf",
        display_name="SVM (RBF)",
        model_class=SVC,
        default_params={
            "kernel": "rbf",
            "class_weight": "balanced",
            "random_state": 42,
            "probability": False,
            "max_iter": 5000,
        },
        requires_scaling=True,
        n_jobs_param=None,
    ),
    "svm_linear": ModelConfig(
        name="svm_linear",
        display_name="SVM (Linear)",
        model_class=SVC,
        default_params={
            "kernel": "linear",
            "class_weight": "balanced",
            "random_state": 42,
            "probability": False,
            "max_iter": 5000,
        },
        requires_scaling=True,
        n_jobs_param=None,
    ),
    "random_forest": ModelConfig(
        name="random_forest",
        display_name="Random Forest",
        model_class=RandomForestClassifier,
        default_params={
            "n_estimators": 100,
            "class_weight": "balanced",
            "random_state": 42,
            "n_jobs": -1,
        },
        requires_scaling=False,
    ),
    "knn": ModelConfig(
        name="knn",
        display_name="KNN",
        model_class=KNeighborsClassifier,
        default_params={
            "n_neighbors": 5,
            "weights": "distance",
            "n_jobs": -1,
        },
        requires_scaling=True,
        supports_class_weight=False,
    ),
    "mlp": ModelConfig(
        name="mlp",
        display_name="MLP Neural Network",
        model_class=MLPClassifier,
        default_params={
            "hidden_layer_sizes": (64, 32),
            "activation": "relu",
            "solver": "adam",
            "max_iter": 500,
            "early_stopping": True,
            "validation_fraction": 0.1,
            "random_state": 42,
        },
        requires_scaling=True,
        supports_class_weight=False,
        n_jobs_param=None,
    ),
    "gaussian_nb": ModelConfig(
        name="gaussian_nb",
        display_name="Gaussian Naive Bayes",
        model_class=GaussianNB,
        default_params={},
        requires_scaling=False,
        supports_class_weight=False,
        n_jobs_param=None,
    ),
}

# Add LightGBM if available
if LIGHTGBM_AVAILABLE:
    AVAILABLE_MODELS["lightgbm"] = ModelConfig(
        name="lightgbm",
        display_name="LightGBM",
        model_class=lgb.LGBMClassifier,
        default_params={
            "n_estimators": 100,
            "class_weight": "balanced",
            "random_state": 42,
            "verbose": -1,
            "n_jobs": -1,
        },
        requires_scaling=False,
    )

# Add XGBoost if available
if XGBOOST_AVAILABLE:
    AVAILABLE_MODELS["xgboost"] = ModelConfig(
        name="xgboost",
        display_name="XGBoost",
        model_class=xgb.XGBClassifier,
        default_params={
            "n_estimators": 100,
            "eval_metric": "logloss",
            "random_state": 42,
            "n_jobs": -1,
            "verbosity": 0,
        },
        requires_scaling=False,
    )


# =============================================================================
# MODEL FACTORY
# =============================================================================

def get_model(
    name: str,
    params: Optional[dict] = None,
    random_state: Optional[int] = None,
    n_jobs: int = -1,
) -> Any:
    """
    Create a model instance.

    Parameters
    ----------
    name : str
        Model name (e.g., 'random_forest').
    params : dict, optional
        Override default parameters.
    random_state : int, optional
        Random seed.
    n_jobs : int
        Number of parallel jobs.

    Returns
    -------
    model
        Sklearn-compatible model instance.
    """
    if name not in AVAILABLE_MODELS:
        raise ValueError(
            f"Unknown model: {name}. Available: {list(AVAILABLE_MODELS.keys())}"
        )

    config = AVAILABLE_MODELS[name]
    model_params = config.default_params.copy()

    if params:
        model_params.update(params)

    if random_state is not None and "random_state" in model_params:
        model_params["random_state"] = random_state

    if config.n_jobs_param and config.n_jobs_param in model_params:
        model_params[config.n_jobs_param] = n_jobs

    return config.model_class(**model_params)


# =============================================================================
# HYPERPARAMETER SPACES (Optuna)
# =============================================================================

def get_hyperparameter_space(
    name: str,
    trial: "optuna.Trial",
    n_features: Optional[int] = None,
) -> dict:
    """
    Get hyperparameter suggestions for Optuna optimization.

    Parameters
    ----------
    name : str
        Model name.
    trial : optuna.Trial
        Optuna trial object.
    n_features : int, optional
        Number of features (affects parameter ranges).

    Returns
    -------
    dict
        Hyperparameters for the model.
    """
    if not OPTUNA_AVAILABLE:
        raise ImportError("Optuna required for hyperparameter optimization")

    spaces = {
        "logistic_regression": _hp_logistic_regression,
        "svm_rbf": _hp_svm_rbf,
        "svm_linear": _hp_svm_linear,
        "random_forest": _hp_random_forest,
        "lightgbm": _hp_lightgbm,
        "xgboost": _hp_xgboost,
        "knn": _hp_knn,
        "mlp": _hp_mlp,
        "gaussian_nb": _hp_gaussian_nb,
    }

    if name not in spaces:
        raise ValueError(f"No hyperparameter space defined for: {name}")

    return spaces[name](trial)


def _hp_logistic_regression(trial) -> dict:
    return {
        "C": trial.suggest_float("C", 1e-4, 100.0, log=True),
        "l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1.0),
        "solver": "saga",
        "penalty": "elasticnet",
        "max_iter": 2000,
        "class_weight": "balanced",
    }


def _hp_svm_rbf(trial) -> dict:
    gamma_type = trial.suggest_categorical("gamma_type", ["scale", "auto", "float"])
    params = {
        "C": trial.suggest_float("C", 1e-4, 1e3, log=True),
        "kernel": "rbf",
        "class_weight": "balanced",
        "max_iter": 5000,
    }
    if gamma_type == "float":
        params["gamma"] = trial.suggest_float("gamma", 1e-5, 1, log=True)
    else:
        params["gamma"] = gamma_type
    return params


def _hp_svm_linear(trial) -> dict:
    return {
        "C": trial.suggest_float("C", 1e-4, 1e3, log=True),
        "kernel": "linear",
        "class_weight": "balanced",
        "max_iter": 5000,
    }


def _hp_random_forest(trial) -> dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "max_depth": trial.suggest_int("max_depth", 2, 8),
        "min_samples_split": trial.suggest_float("min_samples_split", 0.02, 0.2),
        "min_samples_leaf": trial.suggest_float("min_samples_leaf", 0.02, 0.15),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        "class_weight": "balanced",
    }


def _hp_lightgbm(trial) -> dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "max_depth": trial.suggest_int("max_depth", 2, 6),
        "num_leaves": trial.suggest_int("num_leaves", 3, 31),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 2, 20),
        "subsample": trial.suggest_float("subsample", 0.5, 0.9),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.9),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-5, 1, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-5, 1, log=True),
        "class_weight": trial.suggest_categorical("class_weight", ["balanced", None]),
        "verbose": -1,
    }


def _hp_xgboost(trial) -> dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "max_depth": trial.suggest_int("max_depth", 1, 6),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-5, 1, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-5, 1, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 5),
        "gamma": trial.suggest_float("gamma", 1e-5, 1, log=True),
        "eval_metric": "logloss",
        "verbosity": 0,
    }


def _hp_knn(trial) -> dict:
    return {
        "n_neighbors": trial.suggest_int("n_neighbors", 3, 15),
        "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
        "metric": trial.suggest_categorical("metric", ["euclidean", "manhattan"]),
    }


def _hp_mlp(trial) -> dict:
    return {
        "hidden_layer_sizes": (trial.suggest_int("n_units", 8, 64),),
        "activation": trial.suggest_categorical("activation", ["relu", "tanh"]),
        "solver": "adam",
        "alpha": trial.suggest_float("alpha", 1e-4, 1.0, log=True),
        "learning_rate_init": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
        "max_iter": 500,
        "early_stopping": True,
        "validation_fraction": 0.1,
    }


def _hp_gaussian_nb(trial) -> dict:
    return {
        "var_smoothing": trial.suggest_float("var_smoothing", 1e-12, 1e-6, log=True),
    }
