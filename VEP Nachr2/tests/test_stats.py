"""Tests for vep_nachr2/stats.py — the significance helpers used for p-values."""

import numpy as np

from vep_nachr2.stats import (
    per_seed_macro_f1,
    one_sample_greater,
    welch_greater,
    paired_greater,
    fmt_p,
)


def _cv_result(f1s):
    """Build a nested_cross_validation()-shaped dict from a list of per-seed F1s."""
    return {
        "per_seed": [
            {"aggregate": {"macro_f1_mean": v}, "folds": [{"metrics": {"macro_f1": v}}]}
            for v in f1s
        ]
    }


def test_per_seed_macro_f1_extracts():
    res = _cv_result([0.61, 0.63, 0.59])
    assert per_seed_macro_f1(res) == [0.61, 0.63, 0.59]


def test_one_sample_greater_detects_above_chance():
    values = [0.60, 0.62, 0.58, 0.64, 0.61, 0.63]
    t, p = one_sample_greater(values, null=1 / 3)
    assert t > 0
    assert p < 0.001


def test_one_sample_greater_at_null_is_not_significant():
    values = [0.333, 0.334, 0.332, 0.334, 0.333]
    t, p = one_sample_greater(values, null=1 / 3)
    # t ≈ 0 ⇒ one-sided p ≈ 0.5
    assert p > 0.3


def test_one_sample_greater_below_null_has_large_p():
    values = [0.1, 0.12, 0.11, 0.10, 0.13]
    t, p = one_sample_greater(values, null=1 / 3)
    assert t < 0
    assert p > 0.5  # clearly not "greater than chance"


def test_welch_greater_detects_real_difference():
    a = [0.69, 0.70, 0.68, 0.71, 0.69]
    b = [0.63, 0.62, 0.64, 0.61, 0.63]
    t, p = welch_greater(a, b)
    assert t > 0
    assert p < 0.01


def test_welch_greater_equal_means_not_significant():
    # identical samples ⇒ t = 0 exactly ⇒ one-sided p = 0.5 (deterministic)
    a = [0.65, 0.66, 0.64, 0.67, 0.66]
    b = list(a)
    t, p = welch_greater(a, b)
    assert t == 0.0
    assert p == 0.5


def test_paired_greater_detects_difference():
    a = [0.65, 0.66, 0.64, 0.67, 0.66]
    b = [0.61, 0.62, 0.60, 0.63, 0.62]  # always ~0.04 below a
    t, p = paired_greater(a, b)
    assert t > 0
    assert p < 0.01


def test_fmt_p():
    assert fmt_p(0.00005) == "p < 0.0001"
    assert fmt_p(0.0321) == "p = 0.0321"
