"""
Statistical significance helpers for CV results.

The unit of analysis is the **per-seed macro-F1**: each seed yields one macro-F1
(the mean across that seed's 5 outer folds), so N seeds give N values per
condition. The t-tests below operate on those per-seed values.

Caveat (standard in ML): treating seeds as replicates is optimistic — the seeds
share the same data and differ only in fold / RNG shuffling, so these p-values
are "across-seed", not "across independent datasets". They are the conventional
way to ask "is this difference bigger than run-to-run noise?", which is the
question the ablation / subunit-type comparisons need answered.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def per_seed_macro_f1(cv_results: dict) -> list[float]:
    """Extract per-seed macro-F1 from ``nested_cross_validation()`` output."""
    return [float(r["aggregate"]["macro_f1_mean"]) for r in cv_results["per_seed"]]


def per_fold_macro_f1(cv_results: dict) -> list[float]:
    """Extract per-(seed, fold) macro-F1 from ``nested_cross_validation()`` output."""
    vals = []
    for r in cv_results["per_seed"]:
        for f in r["folds"]:
            vals.append(float(f["metrics"]["macro_f1"]))
    return vals


def one_sample_greater(values, null: float = 1 / 3) -> tuple[float, float]:
    """One-sample t-test, H1: mean(values) > ``null`` (e.g. chance = 1/3).

    Returns (t_statistic, one-sided p-value).
    """
    values = np.asarray(values, dtype=float)
    t, p_two = stats.ttest_1samp(values, null)
    p = p_two / 2.0 if t > 0 else 1.0 - p_two / 2.0
    return float(t), float(p)


def welch_greater(a, b) -> tuple[float, float]:
    """Welch's two-sample t-test, H1: mean(a) > mean(b). Returns (t, p)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    t, p_two = stats.ttest_ind(a, b, equal_var=False)
    p = p_two / 2.0 if t > 0 else 1.0 - p_two / 2.0
    return float(t), float(p)


def paired_greater(a, b) -> tuple[float, float]:
    """Paired t-test (matched by seed index), H1: mean(a - b) > 0. Returns (t, p)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    t, p_two = stats.ttest_rel(a, b)
    p = p_two / 2.0 if t > 0 else 1.0 - p_two / 2.0
    return float(t), float(p)


def fmt_p(p: float) -> str:
    """Human-readable p-value."""
    if p < 0.0001:
        return f"p < 0.0001"
    return f"p = {p:.4f}"
