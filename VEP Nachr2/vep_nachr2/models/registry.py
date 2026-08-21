"""
Model registry with hyperparameter search spaces.

Provides 10 models:
  Core: Logistic Regression, SVM (RBF), Random Forest, LightGBM
  Extended: SVM (Linear), KNN, XGBoost, CatBoost, MLP, Gaussian NB

Each model has a hyperparameter search space defined for Optuna.
Tree models have aggressively capped complexity for small datasets (~51 samples in inner folds).
"""

from typing import Any, Callable, Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler

from vep_nachr2.config import MODEL_DISPLAY_NAMES


# =============================================================================
# MODEL FACTORIES
# =============================================================================

def make_logistic_regression(**params) -> LogisticRegression:
    defaults = {"max_iter": 1000, "solver": "lbfgs", "random_state": 42}
    defaults.update(params)
    return LogisticRegression(**defaults)


def make_svm_rbf(**params) -> SVC:
    defaults = {"kernel": "rbf", "probability": True, "random_state": 42}
    defaults.update(params)
    return SVC(**defaults)


def make_svm_linear(**params) -> SVC:
    defaults = {"kernel": "linear", "probability": True, "random_state": 42}
    defaults.update(params)
    return SVC(**defaults)


def make_random_forest(**params) -> RandomForestClassifier:
    defaults = {"random_state": 42, "n_jobs": -1}
    defaults.update(params)
    return RandomForestClassifier(**defaults)


def make_knn(**params) -> Pipeline:
    """KNN with imputation and scaling pipeline."""
    from sklearn.impute import SimpleImputer
    defaults = {"n_neighbors": 5, "weights": "distance", "metric": "euclidean"}
    defaults.update({k: v for k, v in params.items() if k in KNeighborsClassifier().get_params()})
    knn = KNeighborsClassifier(**defaults)
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler()),
        ("knn", knn),
    ])


def make_mlp(**params) -> MLPClassifier:
    defaults = {
        "hidden_layer_sizes": (8,),
        "activation": "relu",
        "solver": "adam",
        "alpha": 0.001,
        "batch_size": 16,
        "learning_rate": "adaptive",
        "max_iter": 1000,
        "early_stopping": True,
        "random_state": 42,
    }
    defaults.update(params)
    return MLPClassifier(**defaults)


def make_gaussian_nb(**params) -> GaussianNB:
    defaults = {"var_smoothing": 1e-9}
    defaults.update(params)
    return GaussianNB(**defaults)


# =============================================================================
# HYPERPARAMETER SEARCH SPACES
# =============================================================================

def _trial_suggest(trial, name: str, spec: tuple, **kwargs):
    """Suggest a hyperparameter value to Optuna based on spec type."""
    spec_type = spec[0]

    if spec_type == "float":
        return trial.suggest_float(name, spec[1], spec[2], **kwargs)
    elif spec_type == "int":
        return trial.suggest_int(name, spec[1], spec[2], **kwargs)
    elif spec_type == "categorical":
        return trial.suggest_categorical(name, spec[1])
    elif spec_type == "log_float":
        return trial.suggest_float(name, spec[1], spec[2], log=True)
    else:
        raise ValueError(f"Unknown spec type: {spec_type}")


# HP spaces: {param_name: (type, *args)}
# Types: "float"(low, high), "int"(low, high), "categorical"(choices), "log_float"(low, high)

HP_SPACES = {
    "logistic_regression": {
        "C": ("log_float", 1e-3, 1e2),
        "penalty": ("categorical", ["l2"]),
    },
    "svm_rbf": {
        "C": ("log_float", 1e-3, 1e2),
        "gamma": ("log_float", 1e-4, 1e0),
    },
    "svm_linear": {
        "C": ("log_float", 1e-3, 1e2),
    },
    "random_forest": {
        "n_estimators": ("int", 50, 500),
        "max_depth": ("int", 2, 8),
        "min_samples_split": ("int", 2, 10),
        "min_samples_leaf": ("int", 1, 5),
        "max_features": ("categorical", ["sqrt", "log2", None]),
    },
    "lightgbm": {
        "n_estimators": ("int", 50, 300),
        "learning_rate": ("log_float", 1e-3, 1e-1),
        "max_depth": ("int", 2, 5),
        "num_leaves": ("int", 4, 31),
        "min_child_samples": ("int", 2, 20),
        "reg_alpha": ("log_float", 1e-4, 1e0),
        "reg_lambda": ("log_float", 1e-4, 1e0),
        "subsample": ("float", 0.5, 1.0),
        "colsample_bytree": ("float", 0.5, 1.0),
    },
    "knn": {
        "n_neighbors": ("int", 1, 11),
        "weights": ("categorical", ["uniform", "distance"]),
        "metric": ("categorical", ["euclidean", "manhattan", "cosine"]),
    },
    "mlp": {
        "hidden_layer_sizes": ("categorical", [(2,), (4,), (8,), (16,), (32,)]),
        "activation": ("categorical", ["relu", "tanh"]),
        "alpha": ("log_float", 1e-5, 1e-1),
        "learning_rate_init": ("log_float", 1e-4, 1e-1),
        "batch_size": ("categorical", [8, 16, 32]),
    },
    "gaussian_nb": {
        "var_smoothing": ("log_float", 1e-12, 1e-7),
    },
    "xgboost": {
        "n_estimators": ("int", 50, 300),
        "learning_rate": ("log_float", 1e-3, 1e-1),
        "max_depth": ("int", 2, 5),
        "min_child_weight": ("int", 1, 10),
        "subsample": ("float", 0.5, 1.0),
        "colsample_bytree": ("float", 0.5, 1.0),
        "reg_alpha": ("log_float", 1e-4, 1e0),
        "reg_lambda": ("log_float", 1e-4, 1e0),
        "gamma": ("log_float", 1e-4, 1e0),
    },
    "catboost": {
        "iterations": ("int", 50, 300),
        "learning_rate": ("log_float", 1e-3, 1e-1),
        "depth": ("int", 2, 5),
        "l2_leaf_reg": ("log_float", 1e-2, 1e1),
        "bagging_temperature": ("float", 0.0, 2.0),
        "random_strength": ("float", 0.0, 2.0),
        "border_count": ("int", 32, 255),
    },
}


# =============================================================================
# GPU DETECTION
# =============================================================================

def _gpu_available() -> bool:
    """Check if an NVIDIA GPU is available for training."""
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5
        )
        return result.returncode == 0 and result.stdout.strip() != ""
    except Exception:
        return False


# =============================================================================
# MODEL BUILDERS
# =============================================================================

# Maps model name → (factory_function, needs_scaling, needs_imputation)
MODEL_REGISTRY: dict[str, dict[str, Any]] = {
    "logistic_regression": {
        "factory": make_logistic_regression,
        "needs_scaling": True,
        "display_name": MODEL_DISPLAY_NAMES["logistic_regression"],
        "hp_space": HP_SPACES["logistic_regression"],
        "supports_sample_weight": True,
    },
    "svm_rbf": {
        "factory": make_svm_rbf,
        "needs_scaling": True,
        "display_name": MODEL_DISPLAY_NAMES["svm_rbf"],
        "hp_space": HP_SPACES["svm_rbf"],
        "supports_sample_weight": True,
    },
    "svm_linear": {
        "factory": make_svm_linear,
        "needs_scaling": True,
        "display_name": MODEL_DISPLAY_NAMES["svm_linear"],
        "hp_space": HP_SPACES["svm_linear"],
        "supports_sample_weight": True,
    },
    "random_forest": {
        "factory": make_random_forest,
        "needs_scaling": False,
        "display_name": MODEL_DISPLAY_NAMES["random_forest"],
        "hp_space": HP_SPACES["random_forest"],
        "supports_sample_weight": True,
    },
    "lightgbm": {
        "factory": None,  # LGBM constructed directly (not sklearn API)
        "needs_scaling": False,
        "display_name": MODEL_DISPLAY_NAMES["lightgbm"],
        "hp_space": HP_SPACES["lightgbm"],
        "supports_sample_weight": True,
    },
    "knn": {
        "factory": make_knn,
        "needs_scaling": True,
        "display_name": MODEL_DISPLAY_NAMES["knn"],
        "hp_space": HP_SPACES["knn"],
        "supports_sample_weight": False,  # Pipeline handles this
    },
    "mlp": {
        "factory": make_mlp,
        "needs_scaling": True,
        "display_name": MODEL_DISPLAY_NAMES["mlp"],
        "hp_space": HP_SPACES["mlp"],
        "supports_sample_weight": False,
    },
    "gaussian_nb": {
        "factory": make_gaussian_nb,
        "needs_scaling": False,
        "display_name": MODEL_DISPLAY_NAMES["gaussian_nb"],
        "hp_space": HP_SPACES["gaussian_nb"],
        "supports_sample_weight": False,
    },
    "xgboost": {
        "factory": None,  # XGB constructed directly
        "needs_scaling": False,
        "display_name": MODEL_DISPLAY_NAMES["xgboost"],
        "hp_space": HP_SPACES["xgboost"],
        "supports_sample_weight": True,
    },
    "catboost": {
        "factory": None,  # CatBoost constructed directly
        "needs_scaling": False,
        "display_name": MODEL_DISPLAY_NAMES["catboost"],
        "hp_space": HP_SPACES["catboost"],
        "supports_sample_weight": True,
    },
}


def get_model_config(model_name: str) -> dict:
    """Get model configuration by name."""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model: {model_name}. Available: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[model_name]


def build_model(model_name: str, params: Optional[dict] = None, n_classes: int = 3) -> Any:
    """Build a model instance with given (or default) parameters.

    Parameters
    ----------
    model_name : str
        Model identifier.
    params : dict, optional
        Hyperparameter overrides.

    Returns
    -------
    model : sklearn-compatible estimator
    """
    config = get_model_config(model_name)
    params = params or {}

    if model_name == "lightgbm":
        import lightgbm as lgb
        defaults = {
            "objective": "multiclass",
            "num_class": n_classes,
            "verbosity": -1,
            "random_state": 42,
            "n_jobs": -1,
        }
        defaults.update(params)
        return lgb.LGBMClassifier(**defaults)

    elif model_name == "xgboost":
        import xgboost as xgb
        if n_classes == 2:
            # binary:logistic returns 0/1 labels from predict(); multi:softprob
            # with num_class=2 wrongly returns the (n,2) probability matrix.
            defaults = {
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "verbosity": 0,
                "random_state": 42,
                "n_jobs": 1,
                "tree_method": "hist",
            }
        else:
            defaults = {
                "objective": "multi:softprob",
                "num_class": n_classes,
                "eval_metric": "mlogloss",
                "verbosity": 0,
                "random_state": 42,
                "n_jobs": 1,
                "tree_method": "hist",  # CPU — gpu_hist broke with multi:softprob
            }
        defaults.update(params)
        return xgb.XGBClassifier(**defaults)

    elif model_name == "catboost":
        from catboost import CatBoostClassifier
        if n_classes == 2:
            defaults = {
                "loss_function": "Logloss",
                "verbose": 0,
                "random_seed": 42,
                "thread_count": -1,
                "allow_writing_files": False,
            }
        else:
            defaults = {
                "loss_function": "MultiClass",
                "classes_count": n_classes,
                "verbose": 0,
                "random_seed": 42,
                "thread_count": -1,
                "allow_writing_files": False,
            }
        if _gpu_available():
            defaults["task_type"] = "GPU"
            defaults["devices"] = "0"
        else:
            defaults["task_type"] = "CPU"
        defaults.update(params)
        return CatBoostClassifier(**defaults)

    else:
        factory = config["factory"]
        needs_scaling = config["needs_scaling"]

        if needs_scaling:
            model = factory(**params)
            return make_pipeline(RobustScaler(), model)
        else:
            return factory(**params)


def suggest_and_build(trial, model_name: str, X_train=None, y_train=None, n_classes: int = 3) -> Any:
    """Suggest hyperparameters via Optuna trial and build model.

    Parameters
    ----------
    trial : optuna.Trial
    model_name : str
    X_train : np.ndarray, optional
        Training features (for dynamic HP caps).
    y_train : np.ndarray, optional
        Training labels.

    Returns
    -------
    model : sklearn-compatible estimator
    """
    config = get_model_config(model_name)
    hp_space = config["hp_space"]
    params = {}

    # Suggest each HP
    for hp_name, hp_spec in hp_space.items():
        params[hp_name] = _trial_suggest(trial, hp_name, hp_spec)

    # Dynamic caps for small datasets
    if X_train is not None and y_train is not None:
        n_train = len(y_train)
        min_class = min(np.bincount(y_train))

        # Cap KNN neighbors to minority class count
        if model_name == "knn" and "n_neighbors" in params:
            params["n_neighbors"] = min(params["n_neighbors"], max(1, min_class - 1))

        # Cap MLP hidden units
        if model_name == "mlp" and "hidden_layer_sizes" in params:
            if isinstance(params["hidden_layer_sizes"], tuple):
                max_units = max(2, n_train // 20)
                params["hidden_layer_sizes"] = tuple(
                    min(u, max_units) for u in params["hidden_layer_sizes"]
                )

    return build_model(model_name, params, n_classes=n_classes)
