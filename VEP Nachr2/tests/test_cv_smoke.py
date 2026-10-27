"""Smoke test: nested_cross_validation runs end-to-end and returns the
per-seed / aggregate structure the stats helpers (and all runs) rely on.

Uses synthetic data so it's fast and cache-independent — this tests the CV
*mechanics* (what the runs call), not the feature pipeline.
"""

import numpy as np
import pandas as pd

from vep_nachr2.training.cross_validation import nested_cross_validation


def test_cv_returns_per_seed_structure():
    rng = np.random.RandomState(0)
    n = 120
    X = rng.randn(n, 10)
    y = np.array([0] * 40 + [1] * 40 + [2] * 40)
    df = pd.DataFrame({"subunit": ["CHRNA1"] * n})

    res = nested_cross_validation(
        X=X, y=y, df=df,
        model_name="random_forest",
        n_outer_folds=3,
        n_inner_folds=3,
        n_trials=2,
        seeds=[42],
        cv_mode="standard",
        verbose=False,
        n_classes=3,
    )

    assert res["model"] == "random_forest"
    assert res["cv_mode"] == "standard"
    assert res["seeds"] == [42]

    # per_seed: one entry per seed, each with folds + aggregate
    assert len(res["per_seed"]) == 1
    seed0 = res["per_seed"][0]
    assert len(seed0["folds"]) == 3
    assert "aggregate" in seed0

    # the key value the stats helpers read
    f1 = seed0["aggregate"]["macro_f1_mean"]
    assert isinstance(f1, float)
    assert 0.0 <= f1 <= 1.0

    # final cross-seed aggregate
    assert "macro_f1_mean" in res["final"]
    assert 0.0 <= res["final"]["macro_f1_mean"] <= 1.0
