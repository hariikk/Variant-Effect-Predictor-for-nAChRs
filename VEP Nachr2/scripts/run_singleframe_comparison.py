"""
Side-by-side: baseline (gene-aware) vs "map all genes onto CHRNA2" single-frame.

Answers: does collapsing all 16 genes onto one reference frame destroy the
direction-of-effect signal?

Both arms use IDENTICAL settings (cv_mode="standard", same seeds/trials/folds,
same dropped feature groups), so the delta is the effect of the single-frame
mapping. The two ESM-2 groups are dropped for speed on BOTH arms.

Usage:
    python scripts/run_singleframe_comparison.py
"""

from __future__ import annotations

import json
import time

import numpy as np

from vep_nachr2.config import RAW_DATA_DIR, RESULTS_DIR
from vep_nachr2.data.loader import load_mutation_data
from vep_nachr2.data.reference import load_all_reference_sequences
from vep_nachr2.features.orchestrator import FeatureOrchestrator
from vep_nachr2.training.cross_validation import nested_cross_validation

MODELS = ["random_forest", "catboost"]
CV_MODE = "standard"
N_OUTER = 5
N_INNER = 4
N_TRIALS = 5
SEEDS = [42, 123, 456, 789, 1011, 2024, 3142, 4269, 5555, 6789]  # 10 seeds (headline protocol)
DROP = ["esm2", "esm_embeddings"]  # speed; dropped on BOTH arms


def load_and_encode(data_file):
    df = load_mutation_data(data_file=data_file)
    y = df["effect"].map({"LOF": 0, "No net effect": 1, "GOF": 2}).values.astype(np.int64)
    return df, y


def run_arm(name, data_file):
    print(f"\n{'#' * 70}\n# ARM: {name}  ({data_file})\n{'#' * 70}")
    df, y = load_and_encode(data_file)
    ref_seqs = load_all_reference_sequences("human")
    orch = FeatureOrchestrator(verbose=False)
    X, _ = orch.extract(df, ref_seqs=ref_seqs, drop_extractors=DROP, use_cache=True)
    print(f"features: {X.shape[1]} cols, {X.shape[0]} rows")

    out = {}
    for m in MODELS:
        t0 = time.time()
        r = nested_cross_validation(
            X=X, y=y, df=df, model_name=m,
            n_outer_folds=N_OUTER, n_inner_folds=N_INNER, n_trials=N_TRIALS,
            seeds=SEEDS, cv_mode=CV_MODE, verbose=False,
        )
        out[m] = r["final"]
        f1 = r["final"]["macro_f1_mean"]
        print(f"  {m:16s} macro-F1 = {f1:.4f}  ({time.time() - t0:.0f}s)")
    return out


def _save(result) -> str:
    out_path = RESULTS_DIR / "singleframe_comparison.json"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    def convert(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, dict):
            return {k: convert(v) for k, v in o.items()}
        if isinstance(o, list):
            return [convert(v) for v in o]
        return o

    with open(out_path, "w") as f:
        json.dump(convert(result), f, indent=2)
    return str(out_path)


def main():
    settings = {"models": MODELS, "cv_mode": CV_MODE, "seeds": SEEDS,
                "n_trials": N_TRIALS, "n_inner_folds": N_INNER,
                "drop_extractors": DROP}

    baseline = run_arm("BASELINE (gene-aware, 16 genes)", "final_mapped.xlsx")
    _save({"settings": settings, "baseline": baseline, "single_frame": None})

    single = run_arm("SINGLE-FRAME (all mapped to CHRNA2)", "final_mapped_singleframe.xlsx")
    out_path = _save({"settings": settings, "baseline": baseline, "single_frame": single})

    print(f"\n{'=' * 70}\nSUMMARY (macro-F1, {CV_MODE}, {len(SEEDS)} seeds, "
          f"dropped={DROP})\n{'=' * 70}")
    print(f"{'model':16s} {'baseline':>10s} {'single-frame':>13s} {'delta':>8s}")
    for m in MODELS:
        b = baseline[m]["macro_f1_mean"]
        s = single[m]["macro_f1_mean"]
        print(f"{m:16s} {b:10.4f} {s:13.4f} {s - b:+8.4f}")

    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
