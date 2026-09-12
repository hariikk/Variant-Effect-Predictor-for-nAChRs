"""
H1 diagnostic: functional-region partition vs the 75-feat baseline (Run XVII).

H1 = "GOF/LOF direction is a function of *which functional element* the residue
sits in, not its static geometry." Tests the new FunctionalRegionExtractor (7
mutually-exclusive one-hot regions: agonist_binding_site, membrane_proximal_ecd,
extracellular_other, m2_m3_linker, pore_m2, other_tmd, intracellular_loop).

Three conditions, all on the full multi-species 3-class dataset, honest
subunit-grouped CV (cv_mode="subunit"):
  1. baseline              : default 75 features (no region).
  2. baseline_plus_region  : +7 region columns (82). Primary H1 test.
  3. baseline_swap         : -structural_core (7) + region (7), still 75 feats.
                            Is region a *better* use of 7 columns than geometry?

delta_f1 = F1(condition) - F1(baseline); positive = the change HELPS.

Two phases, so the GPU/ESM extraction is decoupled from the long CPU CV:
  --extract-only   build features (runs ESM-2 once) and cache to npz, then exit
  --skip-extract   load the cached npz and go straight to CV
  (default)        extract if npz missing, then run CV

Usage:
    python scripts/run_h1_functional_region.py --extract-only
    python scripts/run_h1_functional_region.py --skip-extract --n-trials 15 --n-seeds 3
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from vep_nachr2.data.loader import load_mutation_data
from vep_nachr2.data.reference import load_all_reference_sequences
from vep_nachr2.features.orchestrator import FeatureOrchestrator
from vep_nachr2.features.experimental.functional_region import FunctionalRegionExtractor
from vep_nachr2.training.cross_validation import nested_cross_validation
from vep_nachr2.config import RESULTS_DIR, DEFAULT_SEEDS, LABEL_NAMES


OUT_DIR = RESULTS_DIR / "functional_region"
FEAT_NPZ = OUT_DIR / "features.npz"


def build_features(df, ref_seqs, force=False):
    """Extract default 75 + region 7 = 82 features, cache to npz."""
    if FEAT_NPZ.exists() and not force:
        d = np.load(FEAT_NPZ, allow_pickle=True)
        return d["X"], list(d["feature_names"]), list(d["col_groups"])

    orch = FeatureOrchestrator(verbose=True)
    orch.extractors.append(FunctionalRegionExtractor())
    X, names = orch.extract(
        df, ref_seqs=ref_seqs, use_cache=False, force_recompute=True
    )

    col_groups = []
    for e in orch.extractors:
        col_groups.extend([e.name] * e.n_features)
    assert len(col_groups) == len(names) == X.shape[1], (
        f"group/name/X mismatch: {len(col_groups)} vs {len(names)} vs {X.shape[1]}"
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        FEAT_NPZ,
        X=X,
        feature_names=np.array(names, dtype=object),
        col_groups=np.array(col_groups, dtype=object),
    )
    print(f"Cached features to {FEAT_NPZ}: {X.shape}")
    return X, names, col_groups


def _run_cv(X, y, df, model, n_trials, n_seeds, label):
    print(f"\n--- {label} [{model}]: {X.shape[1]} features ---", flush=True)
    cv = nested_cross_validation(
        X=X, y=y, df=df,
        model_name=model,
        n_outer_folds=5,
        n_trials=n_trials,
        seeds=DEFAULT_SEEDS[:n_seeds],
        cv_mode="subunit",
        verbose=False,
        n_classes=3,
        label_names=LABEL_NAMES,
    )
    return cv.get("final", {})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--extract-only", action="store_true")
    p.add_argument("--skip-extract", action="store_true")
    p.add_argument("--force-extract", action="store_true")
    p.add_argument("--n-trials", type=int, default=15)
    p.add_argument("--n-seeds", type=int, default=3)
    p.add_argument("--models", nargs="+", default=["random_forest", "catboost"])
    args = p.parse_args()

    df = load_mutation_data(data_file="final_mapped.xlsx")
    y = df["effect"].map({"LOF": 0, "No net effect": 1, "GOF": 2}).values.astype(np.int64)
    ref_seqs = load_all_reference_sequences("human")

    if args.extract_only:
        X, names, groups = build_features(df, ref_seqs, force=args.force_extract)
        print("Extract-only: done.")
        return

    X, names, groups = build_features(df, ref_seqs, force=args.force_extract)
    groups = np.array(groups)

    is_region = np.array([g == "functional_region" for g in groups])
    is_struct = np.array([g == "structural_core" for g in groups])

    conditions = {
        "baseline":            ~is_region,           # 75 feats
        "baseline_plus_region": np.ones(len(groups), dtype=bool),  # 82 feats
        "baseline_swap":       ~is_struct,           # 75 feats (geometry -> region)
    }

    print(f"\nFeatures: {X.shape[1]} total. Conditions: {list(conditions)}")

    all_rows = {}
    for model in args.models:
        print(f"\n{'#' * 60}\nMODEL: {model}\n{'#' * 60}")
        for cond, keep in conditions.items():
            Xc = X[:, keep]
            res = _run_cv(Xc, y, df, model, args.n_trials, args.n_seeds, cond)
            all_rows[(model, cond)] = {"n_features": Xc.shape[1], **res}

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for model in args.models:
        rows = {k[1]: v for k, v in all_rows.items() if k[0] == model}
        out = pd.DataFrame(rows).T
        base = out.loc["baseline", "macro_f1_mean"]
        out["delta_f1"] = out["macro_f1_mean"] - base
        out = out[["n_features", "macro_f1_mean", "macro_f1_std", "delta_f1"]]

        print(f"\n{'=' * 70}\n{model}: H1 functional-region | baseline F1 = {base:.4f}")
        print("delta_f1 = F1(condition) - F1(baseline); positive = helps:\n")
        print(out.round(4).to_string())
        out.to_csv(OUT_DIR / f"h1_{model}.csv")
        print(f"Saved {OUT_DIR / f'h1_{model}.csv'}")


if __name__ == "__main__":
    main()
