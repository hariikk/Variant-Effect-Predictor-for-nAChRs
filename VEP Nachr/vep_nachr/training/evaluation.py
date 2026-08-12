"""
Evaluation metrics, result aggregation, and feature importance analysis.

Provides:
- compute_metrics() / compute_per_class_metrics() — classification metrics
- aggregate_results() — scan results/ dir for JSON, build summary DataFrame
- paired_comparison_test() — Wilcoxon/t-test for model comparison
- FeatureImportanceAnalyzer — SHAP + builtin + permutation importance
- plot_feature_importance() / plot_shap_summary() — visualization
- plot_species_transfer_comparison() — box plot for species transfer

Adapted from VEP-ENaC for nAChR binary classification (LOF vs GOF).
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)

from vep_nachr.config import LABEL_NAMES

# --- Optional dependencies ---------------------------------------------------

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


# =============================================================================
# METRIC COMPUTATION
# =============================================================================

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: Optional[np.ndarray] = None,
    average: str = "binary",
) -> dict:
    """
    Compute classification metrics.

    Parameters
    ----------
    y_true : np.ndarray
        True labels (0=LOF, 1=GOF)
    y_pred : np.ndarray
        Predicted labels
    y_score : np.ndarray, optional
        Predicted probabilities for ROC-AUC (positive class)
    average : str
        Averaging strategy: 'binary', 'macro', 'micro', 'weighted'

    Returns
    -------
    dict
        Dictionary of metrics
    """
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, average=average, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, average=average, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average=average, zero_division=0)),
    }

    # Per-class F1 for binary
    per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    metrics["f1_per_class"] = [float(x) for x in per_class]
    metrics["f1_lof"] = float(per_class[0]) if len(per_class) > 0 else 0.0
    metrics["f1_gof"] = float(per_class[1]) if len(per_class) > 1 else 0.0

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics["confusion_matrix"] = cm.tolist()

    # ROC-AUC
    if y_score is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
        except ValueError:
            metrics["roc_auc"] = None
    else:
        metrics["roc_auc"] = None

    return metrics


def compute_per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> pd.DataFrame:
    """
    Compute per-class metrics as a DataFrame.

    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels

    Returns
    -------
    pd.DataFrame
        Metrics per class with columns: class, precision, recall, f1-score, support
    """
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    rows = []
    for label_int in sorted(set(y_true) | set(y_pred)):
        label_name = LABEL_NAMES.get(int(label_int), str(label_int))
        key = str(int(label_int))
        if key in report:
            rows.append({
                "class": label_name,
                "precision": report[key]["precision"],
                "recall": report[key]["recall"],
                "f1-score": report[key]["f1-score"],
                "support": report[key]["support"],
            })

    return pd.DataFrame(rows)


# =============================================================================
# RESULT AGGREGATION
# =============================================================================

def aggregate_results(
    results_dir: str | Path,
    output_file: Optional[str | Path] = None,
) -> pd.DataFrame:
    """
    Aggregate results from multiple JSON files into a summary DataFrame.

    Parameters
    ----------
    results_dir : str or Path
        Directory containing result JSON files (searched recursively)
    output_file : str or Path, optional
        Path to save aggregated CSV

    Returns
    -------
    pd.DataFrame
        Aggregated results sorted by F1 (descending)
    """
    results_dir = Path(results_dir)

    rows = []
    for json_file in results_dir.glob("**/*.json"):
        try:
            with open(json_file, "r") as f:
                data = json.load(f)

            rows.append({
                "file": str(json_file.relative_to(results_dir)),
                "model": data.get("model_name", "unknown"),
                "encoding": data.get("encoding", "unknown"),
                "dataset": data.get("dataset_name", ""),
                "mean_f1": data.get("overall_mean_f1", np.nan),
                "std_f1": data.get("overall_std_f1", np.nan),
                "mean_accuracy": data.get("overall_mean_accuracy", np.nan),
                "n_samples": data.get("n_samples", np.nan),
                "n_features": data.get("n_features", np.nan),
                "n_seeds": data.get("n_seeds", np.nan),
                "timestamp": data.get("timestamp", ""),
            })
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Could not parse {json_file}: {e}")

    df = pd.DataFrame(rows)

    if not df.empty:
        df = df.sort_values("mean_f1", ascending=False).reset_index(drop=True)

    if output_file:
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_file, index=False)

    return df


# =============================================================================
# APPROACH COMPARISON
# =============================================================================

def compare_approaches(
    domain_driven_results: pd.DataFrame,
    data_driven_results: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compare domain-driven and data-driven approaches.

    Parameters
    ----------
    domain_driven_results : pd.DataFrame
        Results from domain-driven (engineered features) approach
    data_driven_results : pd.DataFrame
        Results from data-driven (sequence encoding) approach

    Returns
    -------
    pd.DataFrame
        Pivoted comparison table
    """
    dd = domain_driven_results.copy()
    dr = data_driven_results.copy()

    dd["approach"] = "domain_driven"
    dr["approach"] = "data_driven"

    combined = pd.concat([dd, dr], ignore_index=True)

    pivot = combined.pivot_table(
        index=["model"],
        columns="approach",
        values="mean_f1",
        aggfunc="first",
    ).reset_index()

    if "domain_driven" in pivot.columns and "data_driven" in pivot.columns:
        pivot["difference"] = pivot["data_driven"] - pivot["domain_driven"]
        pivot["better_approach"] = np.where(
            pivot["difference"] > 0, "data_driven", "domain_driven"
        )

    return pivot


# =============================================================================
# STATISTICAL TESTS
# =============================================================================

def paired_comparison_test(
    scores_a: list[float],
    scores_b: list[float],
    test: str = "wilcoxon",
) -> dict:
    """
    Perform paired statistical test between two sets of scores.

    Parameters
    ----------
    scores_a : list[float]
        Scores from first model/approach
    scores_b : list[float]
        Scores from second model/approach
    test : str
        Test type: 'wilcoxon', 'ttest'

    Returns
    -------
    dict
        Test statistic and p-value
    """
    from scipy import stats

    scores_a = np.array(scores_a)
    scores_b = np.array(scores_b)

    if len(scores_a) != len(scores_b):
        raise ValueError("Score arrays must have the same length for paired test")

    if test == "wilcoxon":
        stat, p_value = stats.wilcoxon(scores_a, scores_b)
    elif test == "ttest":
        stat, p_value = stats.ttest_rel(scores_a, scores_b)
    else:
        raise ValueError(f"Unknown test: {test}")

    return {
        "test": test,
        "statistic": float(stat),
        "p_value": float(p_value),
        "mean_a": float(np.mean(scores_a)),
        "mean_b": float(np.mean(scores_b)),
        "mean_difference": float(np.mean(scores_b) - np.mean(scores_a)),
        "significant_at_05": p_value < 0.05,
        "significant_at_01": p_value < 0.01,
    }


# =============================================================================
# FEATURE IMPORTANCE (SHAP)
# =============================================================================

# Tree-based model class names that support SHAP TreeExplainer
_TREE_MODEL_CLASSES = (
    "RandomForestClassifier",
    "LGBMClassifier",
    "GradientBoostingClassifier",
    "XGBClassifier",
    "CatBoostClassifier",
)


def is_tree_model(model_name_or_obj) -> bool:
    """Check whether a model (name or object) is tree-based."""
    if isinstance(model_name_or_obj, str):
        return model_name_or_obj in (
            "random_forest", "lightgbm", "xgboost", "catboost", "gradient_boosting",
        )
    return model_name_or_obj.__class__.__name__ in _TREE_MODEL_CLASSES


@dataclass
class FeatureImportanceResult:
    """Results from feature importance analysis."""

    feature_names: list[str]
    importance_mean: np.ndarray
    importance_std: np.ndarray
    shap_values: Optional[np.ndarray] = None
    model_name: str = ""
    encoding: str = ""
    n_folds: int = 0
    method: str = "shap"

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame sorted by importance (descending)."""
        df = pd.DataFrame({
            "feature": self.feature_names,
            "importance_mean": self.importance_mean,
            "importance_std": self.importance_std,
        })
        return df.sort_values("importance_mean", ascending=False).reset_index(drop=True)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "feature_names": self.feature_names,
            "importance_mean": self.importance_mean.tolist(),
            "importance_std": self.importance_std.tolist(),
            "model_name": self.model_name,
            "encoding": self.encoding,
            "n_folds": self.n_folds,
            "method": self.method,
        }

    def save(self, path: str | Path) -> None:
        """Save to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


class FeatureImportanceAnalyzer:
    """
    Analyze feature importance for tree-based models using SHAP.

    Computes SHAP values for Random Forest, LightGBM, XGBoost, CatBoost,
    aggregates across CV folds, and provides visualization utilities.
    Falls back to builtin feature_importances_ or permutation importance.
    """

    def __init__(self, method: str = "shap"):
        if method == "shap" and not SHAP_AVAILABLE:
            raise ImportError("SHAP not available. Install with: pip install shap")
        self.method = method
        self._fold_importances: list[np.ndarray] = []
        self._fold_shap_values: list[np.ndarray] = []
        self._feature_names: Optional[list[str]] = None

    def compute_importance(
        self,
        model,
        X: np.ndarray,
        feature_names: Optional[list[str]] = None,
        y: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute feature importance for a single trained model."""
        if self._feature_names is None and feature_names is not None:
            self._feature_names = feature_names

        if self.method == "shap" and is_tree_model(model):
            return self._compute_shap_importance(model, X)
        else:
            return self._compute_builtin_importance(model, X, y)

    def _compute_shap_importance(self, model, X: np.ndarray) -> np.ndarray:
        """Compute SHAP-based feature importance."""
        shap_model = getattr(model, "_model", model)
        explainer = shap.TreeExplainer(shap_model)
        shap_values = explainer.shap_values(X)

        # Handle different SHAP output formats
        if isinstance(shap_values, list):
            # Legacy: list of (n_samples, n_features) arrays per class
            shap_values = np.array(shap_values)
            importance = np.mean(np.abs(shap_values), axis=(0, 1))
            self._fold_shap_values.append(shap_values)
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            # (n_samples, n_features, n_classes)
            importance = np.mean(np.abs(shap_values), axis=(0, 2))
            self._fold_shap_values.append(shap_values)
        else:
            # Binary/single output: (n_samples, n_features)
            importance = np.mean(np.abs(shap_values), axis=0)
            self._fold_shap_values.append(shap_values)

        self._fold_importances.append(importance)
        return importance

    def _compute_builtin_importance(
        self,
        model,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute feature importance using best available method."""
        if hasattr(model, "feature_importances_"):
            importance = model.feature_importances_
            self.method = "builtin"
        elif hasattr(model, "coef_"):
            coef = model.coef_
            importance = np.mean(np.abs(coef), axis=0)
            self.method = "coefficient"
        elif y is not None:
            from sklearn.inspection import permutation_importance
            perm = permutation_importance(
                model, X, y, n_repeats=10, random_state=0,
                scoring="f1", n_jobs=1,
            )
            importance = perm.importances_mean
            self.method = "permutation"
        else:
            importance = np.zeros(X.shape[1])
            self.method = "none"

        self._fold_importances.append(importance)
        return importance

    def add_fold_result(self, model, X_test, feature_names=None, y_test=None) -> None:
        """Add results from one CV fold."""
        self.compute_importance(model, X_test, feature_names, y=y_test)

    def aggregate(self, model_name: str = "", encoding: str = "") -> FeatureImportanceResult:
        """Aggregate importance across all folds."""
        if not self._fold_importances:
            raise ValueError("No fold results to aggregate. Call add_fold_result first.")

        importances = np.array(self._fold_importances)

        aggregated_shap = None
        if self._fold_shap_values:
            try:
                aggregated_shap = np.concatenate(self._fold_shap_values, axis=0)
            except (ValueError, IndexError):
                aggregated_shap = None

        if self._feature_names is None:
            self._feature_names = [f"feature_{i}" for i in range(importances.shape[1])]

        return FeatureImportanceResult(
            feature_names=self._feature_names,
            importance_mean=np.mean(importances, axis=0),
            importance_std=np.std(importances, axis=0),
            shap_values=aggregated_shap,
            model_name=model_name,
            encoding=encoding,
            n_folds=len(self._fold_importances),
            method=self.method,
        )

    def reset(self) -> None:
        """Reset accumulated fold results."""
        self._fold_importances = []
        self._fold_shap_values = []
        self._feature_names = None


# =============================================================================
# FEATURE IMPORTANCE VISUALIZATION
# =============================================================================

def plot_feature_importance(
    result: FeatureImportanceResult,
    top_n: int = 20,
    output_path: Optional[str | Path] = None,
    title: Optional[str] = None,
    figsize: tuple[int, int] = (10, 8),
):
    """
    Plot feature importance as horizontal bar chart.

    Parameters
    ----------
    result : FeatureImportanceResult
        Feature importance results
    top_n : int
        Number of top features to show
    output_path : str or Path, optional
        Path to save figure (saves as both PNG and PDF)
    title : str, optional
        Plot title
    figsize : tuple
        Figure size

    Returns
    -------
    plt.Figure or None
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("Matplotlib not available for plotting")

    df = result.to_dataframe().head(top_n)

    fig, ax = plt.subplots(figsize=figsize)
    y_pos = np.arange(len(df))
    ax.barh(y_pos, df["importance_mean"], xerr=df["importance_std"],
            color="steelblue", alpha=0.8, capsize=3)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["feature"])
    ax.invert_yaxis()

    xlabel_map = {
        "shap": "Mean |SHAP value|",
        "coefficient": "Mean |Coefficient|",
        "permutation": "Permutation Importance (F1 drop)",
        "builtin": "Feature Importance",
    }
    ax.set_xlabel(xlabel_map.get(result.method, "Importance"))
    ax.set_title(title or f"Top {top_n} Features — {result.model_name} / {result.encoding}")
    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        pdf_path = output_path.with_suffix(".pdf")
        fig.savefig(pdf_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return None

    return fig


def plot_shap_summary(
    result: FeatureImportanceResult,
    X: np.ndarray,
    feature_names: Optional[list[str]] = None,
    output_path: Optional[str | Path] = None,
    max_display: int = 20,
) -> None:
    """
    Plot SHAP summary (beeswarm) plot.

    Parameters
    ----------
    result : FeatureImportanceResult
        Results with SHAP values
    X : np.ndarray
        Feature data used for SHAP computation
    feature_names : list[str], optional
        Feature names (defaults to result.feature_names)
    output_path : str or Path, optional
        Path to save figure
    max_display : int
        Maximum features to display
    """
    if not SHAP_AVAILABLE:
        raise ImportError("SHAP not available for plotting")
    if result.shap_values is None:
        raise ValueError("No SHAP values in result. Use method='shap'.")

    if feature_names is None:
        feature_names = result.feature_names

    plt.figure(figsize=(10, 8))

    shap_values = result.shap_values
    if len(shap_values.shape) == 3:
        shap_values_plot = np.mean(np.abs(shap_values), axis=0)
    else:
        shap_values_plot = shap_values

    shap.summary_plot(
        shap_values_plot, X, feature_names=feature_names,
        max_display=max_display, show=False,
    )

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        pdf_path = output_path.with_suffix(".pdf")
        plt.savefig(pdf_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


# =============================================================================
# SPECIES TRANSFER VISUALIZATION
# =============================================================================

def plot_species_transfer_comparison(
    human_only_scores: list[float],
    mouse_only_scores: list[float],
    mixed_scores: list[float],
    output_path: Optional[str | Path] = None,
    title: str = "Species Transfer Comparison",
    figsize: tuple[int, int] = (8, 6),
):
    """
    Plot box plot comparing three training conditions.

    Parameters
    ----------
    human_only_scores : list[float]
        Per-seed F1 scores for human-only training
    mouse_only_scores : list[float]
        Per-seed F1 scores for mouse-only training
    mixed_scores : list[float]
        Per-seed F1 scores for mixed (human+mouse) training
    output_path : str or Path, optional
        Path to save figure
    title : str
        Plot title
    figsize : tuple
        Figure size

    Returns
    -------
    plt.Figure or None
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("Matplotlib not available for plotting")

    fig, ax = plt.subplots(figsize=figsize)

    data = [human_only_scores, mouse_only_scores, mixed_scores]
    labels = ["Human-only", "Mouse-only", "Mixed"]
    colors = ["#2ecc71", "#e74c3c", "#3498db"]

    bp = ax.boxplot(data, labels=labels, patch_artist=True)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    for i, (scores, color) in enumerate(zip(data, colors)):
        x = np.random.normal(i + 1, 0.04, size=len(scores))
        ax.scatter(x, scores, alpha=0.6, color=color, edgecolors="black", s=30)

    ax.set_ylabel("F1 Score")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)

    for i, scores in enumerate(data):
        mean_val = np.mean(scores)
        ax.annotate(f"{mean_val:.3f}", xy=(i + 1, mean_val),
                    xytext=(i + 1.2, mean_val),
                    fontsize=9, ha="left", va="center")

    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        pdf_path = output_path.with_suffix(".pdf")
        fig.savefig(pdf_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return None

    return fig


# =============================================================================
# COMPARISON HEATMAP (for models × encodings grid)
# =============================================================================

def plot_comparison_heatmap(
    summary_df: pd.DataFrame,
    output_path: Optional[str | Path] = None,
    title: str = "Model × Encoding — F1 Scores",
    figsize: tuple[int, int] = (10, 6),
):
    """
    Plot heatmap of models × encodings from aggregated results.

    Parameters
    ----------
    summary_df : pd.DataFrame
        From aggregate_results() with 'model' and 'encoding' columns
    output_path : str or Path, optional
        Path to save figure
    title : str
        Plot title
    figsize : tuple
        Figure size
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("Matplotlib not available for plotting")

    pivot = summary_df.pivot_table(
        index="model", columns="encoding", values="mean_f1", aggfunc="first"
    )

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn")

    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ax.set_yticklabels(pivot.index)
    ax.set_title(title)

    # Annotate cells
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            text_color = "white" if val < (np.nanmax(pivot.values) * 0.6) else "black"
            ax.text(j, i, f"{val:.3f}" if not np.isnan(val) else "—",
                    ha="center", va="center", color=text_color, fontsize=9)

    plt.colorbar(im, ax=ax, label="F1 Score")
    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        pdf_path = output_path.with_suffix(".pdf")
        fig.savefig(pdf_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return None

    return fig
