"""
Evaluation metrics for 3-class nAChR variant effect prediction.

Primary metrics:
  - Macro F1: balanced across GOF/LOF/NNE (Optuna objective)
  - MCC: Matthews Correlation Coefficient (most honest multi-class metric)
  - Balanced Accuracy: average recall across classes

Secondary:
  - Per-class precision, recall, F1
  - Confusion matrix
  - PR-AUC (macro, one-vs-rest)
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)
from vep_nachr2.config import LABEL_NAMES


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
    labels: list[int] | None = None,
    label_names: dict | None = None,
) -> dict:
    """
    Compute comprehensive classification metrics.

    Parameters
    ----------
    y_true : np.ndarray
        True integer labels.
    y_pred : np.ndarray
        Predicted integer labels.
    y_proba : np.ndarray, optional
        Predicted probabilities (n_samples, n_classes).
    labels : list[int], optional
        Class labels (default: [0, 1, 2]).

    Returns
    -------
    dict with all metrics.
    """
    if labels is None:
        labels = [0, 1, 2]
    if label_names is None:
        label_names = LABEL_NAMES

    metrics = {}

    # Basic metrics
    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
    metrics["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))
    metrics["macro_f1"] = float(f1_score(y_true, y_pred, average="macro", labels=labels))
    metrics["weighted_f1"] = float(f1_score(y_true, y_pred, average="weighted", labels=labels))
    metrics["mcc"] = float(matthews_corrcoef(y_true, y_pred))

    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    metrics["precision_macro"] = float(np.mean(precision))
    metrics["recall_macro"] = float(np.mean(recall))

    for i, lbl in enumerate(labels):
        name = label_names.get(lbl, f"class_{lbl}")
        metrics[f"precision_{name}"] = float(precision[i])
        metrics[f"recall_{name}"] = float(recall[i])
        metrics[f"f1_{name}"] = float(f1[i])
        metrics[f"support_{name}"] = int(support[i])

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    metrics["confusion_matrix"] = cm.tolist()

    # Per-class accuracy (recall)
    for i, lbl in enumerate(labels):
        name = label_names.get(lbl, f"class_{lbl}")
        class_mask = y_true == lbl
        if class_mask.any():
            metrics[f"accuracy_{name}"] = float(
                accuracy_score(y_true[class_mask], y_pred[class_mask])
            )
        else:
            metrics[f"accuracy_{name}"] = 0.0

    # ROC-AUC (if probabilities available)
    if y_proba is not None and y_proba.shape[1] == len(labels):
        try:
            metrics["roc_auc_macro"] = float(
                roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro", labels=labels)
            )
        except Exception:
            metrics["roc_auc_macro"] = 0.0

    return metrics


def aggregate_metrics(fold_results: list[dict]) -> dict:
    """
    Aggregate metrics across CV folds.

    Parameters
    ----------
    fold_results : list[dict]
        List of per-fold results, each containing a 'metrics' dict.

    Returns
    -------
    dict with mean and std for each metric.
    """
    if not fold_results:
        return {}

    # Collect all metric names
    metric_names = list(fold_results[0]["metrics"].keys())

    # Skip non-scalar metrics (confusion matrix)
    scalar_metrics = [m for m in metric_names if m != "confusion_matrix"]

    aggregated = {}
    for metric in scalar_metrics:
        values = [fr["metrics"][metric] for fr in fold_results]
        aggregated[f"{metric}_mean"] = float(np.mean(values))
        aggregated[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0

    # Average confusion matrix
    cm_list = [np.array(fr["metrics"]["confusion_matrix"]) for fr in fold_results]
    if cm_list:
        aggregated["confusion_matrix_sum"] = np.sum(cm_list, axis=0).tolist()

    return aggregated


def format_classification_report(metrics: dict) -> str:
    """
    Format metrics as a readable classification report.

    Parameters
    ----------
    metrics : dict
        Output from compute_metrics().

    Returns
    -------
    str
        Formatted report string.
    """
    lines = []
    lines.append("=" * 60)
    lines.append("Classification Report")
    lines.append("=" * 60)
    lines.append(f"Accuracy:          {metrics.get('accuracy', 0):.4f}")
    lines.append(f"Balanced Accuracy: {metrics.get('balanced_accuracy', 0):.4f}")
    lines.append(f"Macro F1:          {metrics.get('macro_f1', 0):.4f}")
    lines.append(f"Weighted F1:       {metrics.get('weighted_f1', 0):.4f}")
    lines.append(f"MCC:              {metrics.get('mcc', 0):.4f}")
    if "roc_auc_macro" in metrics:
        lines.append(f"ROC-AUC (macro):  {metrics['roc_auc_macro']:.4f}")
    lines.append("-" * 60)
    lines.append(f"{'Class':<20} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    lines.append("-" * 60)

    for label_key in LABEL_NAMES.values():
        name = str(label_key)
        p = metrics.get(f"precision_{name}", 0)
        r = metrics.get(f"recall_{name}", 0)
        f = metrics.get(f"f1_{name}", 0)
        s = metrics.get(f"support_{name}", 0)
        lines.append(f"{name:<20} {p:>10.4f} {r:>10.4f} {f:>10.4f} {s:>10}")

    lines.append("=" * 60)

    # Confusion matrix
    cm = metrics.get("confusion_matrix", [])
    if cm:
        lines.append("\nConfusion Matrix:")
        lines.append(f"{'':>12} " + " ".join(f"{str(LABEL_NAMES.get(i, i)):>8}" for i in range(len(cm))))
        for i, row in enumerate(cm):
            lines.append(f"{str(LABEL_NAMES.get(i, i)):>12} " + " ".join(f"{int(v):>8}" for v in row))

    return "\n".join(lines)
