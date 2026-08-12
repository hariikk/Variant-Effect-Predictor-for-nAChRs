"""
Per-model class imbalance strategy dispatch.

Based on VEP-ENAC v3 pattern:
  - cost_sensitive: class_weight='balanced' (LR, SVM, LightGBM, CatBoost)
  - brfc: BalancedRandomForestClassifier (Random Forest)
  - ros: RandomOverSampler in Pipeline (KNN, MLP, GaussianNB)
  - xgb_inverse: sample_weight = 1/class_frequency (XGBoost)
"""

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler


# =============================================================================
# STRATEGY TABLE
# =============================================================================

IMBALANCE_STRATEGIES = {
    "logistic_regression": "cost_sensitive",
    "svm_rbf": "cost_sensitive",
    "svm_linear": "cost_sensitive",
    "random_forest": "cost_sensitive",  # class_weight='balanced_subsample' — fast, not BRFC
    "lightgbm": "cost_sensitive",
    "knn": "ros",
    "mlp": "ros",
    "gaussian_nb": "ros",
    "xgboost": "cost_sensitive",  # sample_weight strategy was broken, use class balancing
    "catboost": "cost_sensitive",
}


def get_strategy(model_name: str) -> str:
    """Get imbalance handling strategy for a model."""
    if model_name not in IMBALANCE_STRATEGIES:
        raise ValueError(f"Unknown model: {model_name}")
    return IMBALANCE_STRATEGIES[model_name]


def apply_strategy(model, model_name: str, X_train=None, y_train=None) -> object:
    """
    Apply the appropriate class imbalance strategy to a model.

    Parameters
    ----------
    model : sklearn-compatible estimator
        The base model (already built with hyperparams).
    model_name : str
        Model identifier.
    X_train : np.ndarray, optional
    y_train : np.ndarray, optional

    Returns
    -------
    model : sklearn-compatible estimator
        Model with imbalance strategy applied.
    """
    strategy = get_strategy(model_name)

    if strategy == "cost_sensitive":
        # For tree-based models with class_weight, set it here
        if model_name == "lightgbm":
            model.set_params(class_weight="balanced")
        elif model_name == "catboost":
            model.set_params(auto_class_weights="Balanced")
        elif model_name == "random_forest":
            model.set_params(class_weight="balanced_subsample")
        elif hasattr(model, "named_steps"):
            # Pipeline: set on the final estimator
            final_name = list(model.named_steps.keys())[-1]
            final_est = model.named_steps[final_name]
            if hasattr(final_est, "class_weight"):
                final_est.set_params(class_weight="balanced")
        elif hasattr(model, "class_weight"):
            model.set_params(class_weight="balanced")

    elif strategy == "brfc":
        # Replace RandomForest with BalancedRandomForest
        from imblearn.ensemble import BalancedRandomForestClassifier

        if hasattr(model, "get_params"):
            rf_params = model.get_params()
        else:
            rf_params = {}

        brfc = BalancedRandomForestClassifier(
            n_estimators=rf_params.get("n_estimators", 200),
            max_depth=rf_params.get("max_depth"),
            min_samples_split=rf_params.get("min_samples_split", 2),
            min_samples_leaf=rf_params.get("min_samples_leaf", 1),
            random_state=42,
            n_jobs=-1,
        )
        return brfc  # BRFC doesn't need scaling

    elif strategy == "ros":
        # Wrap in Pipeline with RandomOverSampler
        from imblearn.over_sampling import RandomOverSampler
        from imblearn.pipeline import Pipeline as ImbPipeline

        ros = RandomOverSampler(random_state=42)
        return ImbPipeline([
            ("scaler", RobustScaler() if "knn" in model_name or "mlp" in model_name else _Passthrough()),
            ("ros", ros),
            ("classifier", model),
        ])

    elif strategy == "xgb_inverse":
        # XGBoost handles weights at fit() time — store for later
        if y_train is not None:
            classes, counts = np.unique(y_train, return_counts=True)
            weights = 1.0 / counts
            sample_weight = np.array([weights[c] for c in y_train])
            model._sample_weight = sample_weight
        return model

    return model


class _Passthrough:
    """Transformer that does nothing (for Pipeline compatibility)."""
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X
    def fit_transform(self, X, y=None):
        return X


def compute_sample_weights(y: np.ndarray) -> np.ndarray:
    """Compute inverse-frequency sample weights."""
    classes, counts = np.unique(y, return_counts=True)
    weights = 1.0 / counts
    weight_dict = dict(zip(classes, weights))
    return np.array([weight_dict[c] for c in y])
