"""
Follow-up to run_a7resting_probe.py: disentangle the geometry effect from the
B-factor/SS metadata confound.

The naive swap (a7rest) recomputes ALL 14 PDB features from the new structure, but
the spliced model has no per-atom B-factors (all 0.00) and no SS annotation (all
coil), so `b_factor` and `dssp_*` collapse for CHRNA7 — a real signal loss unrelated
to the geometry. This script adds a **geometry-only** arm:

  rsa, cbeta_density, gnm_msf, gnm_slowmode, gnm_gate_coupling, allosteric_*  -> a7rest
  b_factor, dssp_helix/sheet/coil                                              -> 7EKT (held constant)

so the only thing that changes vs 7EKT is the actual geometry/dynamics. It then runs
5-seed CV on the geometry-only arm and reports paired t-tests across all three arms.

Usage:  python scripts/run_a7resting_geomonly.py --n-seeds 5 --n-trials 15
"""

import argparse
import json
import sys
import warnings

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore", message=".*ill-defined.*")
warnings.filterwarnings("ignore", message=".*Only one class is present.*")
warnings.filterwarnings("ignore", message=".*ROC AUC score is not defined.*")

from scipy import stats

import vep_nachr2.config as cfg
from vep_nachr2.config import PDB_MAPPING, RESULTS_DIR, DEFAULT_SEEDS, LABEL_NAMES
from vep_nachr2.data.loader import load_mutation_data
from vep_nachr2.data.reference import load_all_reference_sequences
from vep_nachr2.features.orchestrator import FeatureOrchestrator
from vep_nachr2.features.structural import (
    StructuralExtractor, load_all_pdb_resources, _load_pdb_resources,
)
from vep_nachr2.features.gating import GatingExtractor
from vep_nachr2.features.allosteric_interaction import AllostericInteractionExtractor
from vep_nachr2.training.cross_validation import nested_cross_validation

_EFFECT_MAP = {"LOF": 0, "No net effect": 1, "GOF": 2}

# structural_core column layout (StructuralExtractor.feature_names):
#  0 rsa, 1 cbeta_density, 2 b_factor, 3 dssp_helix, 4 dssp_sheet, 5 dssp_coil, 6 is_unmappable
_IDX_BFACTOR = 2
_IDX_SS = slice(3, 6)  # helix/sheet/coil


def _run_cv(X, y, df, model, n_trials, n_seeds):
    return nested_cross_validation(
        X=X, y=y, df=df, model_name=model,
        n_outer_folds=5, n_trials=n_trials,
        seeds=DEFAULT_SEEDS[:n_seeds], cv_mode="standard",
        verbose=False, n_classes=3, label_names=LABEL_NAMES,
    )


def _per_seed(cv, key):
    return [float(s["aggregate"][key]) for s in cv["per_seed"]]


def paired(a, b):
    a, b = np.asarray(a), np.asarray(b)
    d = b - a
    t, p = stats.ttest_rel(b, a)
    return float(np.mean(d)), float(t), float(p)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--n-trials", type=int, default=15)
    parser.add_argument("--models", nargs="+", default=["random_forest", "catboost"])
    args = parser.parse_args()

    df = load_mutation_data(data_file="final_mapped.xlsx")
    y = df["effect"].map(_EFFECT_MAP).values.astype(np.int64)
    ref_seqs = load_all_reference_sequences("human")

    orch = FeatureOrchestrator(verbose=False)
    X75, names = orch.extract(df, ref_seqs=ref_seqs, use_cache=True)
    s0, s1 = orch.get_feature_group_indices()["structural_core"]

    base_pdb = load_all_pdb_resources()

    # 7EKT structural
    struct7, gat7, int7 = StructuralExtractor().extract(df, ref_seqs, base_pdb), \
        GatingExtractor().extract(df, ref_seqs, base_pdb), \
        AllostericInteractionExtractor().extract(df, ref_seqs, base_pdb)

    # a7rest structural
    orig = PDB_MAPPING["CHRNA7"]
    PDB_MAPPING["CHRNA7"] = {"pdb_id": "a7rest", "chain": "A", "source": "experimental"}
    try:
        a7rest_res = _load_pdb_resources("a7rest")
        pdb_new = {k: v for k, v in base_pdb.items() if k != "7EKT"}
        pdb_new["a7rest"] = a7rest_res
        struct_new = StructuralExtractor().extract(df, ref_seqs, pdb_new)
        gat_new = GatingExtractor().extract(df, ref_seqs, pdb_new)
        int_new = AllostericInteractionExtractor().extract(df, ref_seqs, pdb_new)
    finally:
        PDB_MAPPING["CHRNA7"] = orig

    # geometry-only: a7rest geometry, but b_factor + SS held at 7EKT
    struct_hybrid = struct_new.copy()
    struct_hybrid[:, _IDX_BFACTOR] = struct7[:, _IDX_BFACTOR]
    struct_hybrid[:, _IDX_SS] = struct7[:, _IDX_SS]

    X75_hybrid = X75.copy()
    X75_hybrid[:, s0:s1] = struct_hybrid
    X_geom = np.hstack([X75_hybrid, gat_new, int_new])
    assert X_geom.shape == (len(df), 82), X_geom.shape

    out_dir = RESULTS_DIR / "a7resting"
    geom_file = out_dir / "geomonly.json"
    results = {}
    for model in args.models:
        cv = _run_cv(X_geom, y, df, model, args.n_trials, args.n_seeds)
        results[model] = {
            "macro": _per_seed(cv, "macro_f1_mean"),
            "LOF": _per_seed(cv, "f1_LOF_mean"),
            "NNE": _per_seed(cv, "f1_No net effect_mean"),
            "GOF": _per_seed(cv, "f1_GOF_mean"),
        }
        print(f"  [geom-only {model:15s}] macro {np.mean(results[model]['macro']):.4f} ± "
              f"{np.std(results[model]['macro']):.4f} | LOF {np.mean(results[model]['LOF']):.4f} | "
              f"NNE {np.mean(results[model]['NNE']):.4f} | GOF {np.mean(results[model]['GOF']):.4f}",
              flush=True)
    with open(geom_file, "w") as f:
        json.dump(results, f, indent=2)

    # ── paired t-tests vs the saved 7EKT / naive a7rest arms ──────────────
    naive = json.load(open(out_dir / "comparison.json"))
    print("\nPaired t-tests (n =", args.n_seeds, "seeds):")
    print(f"{'model':15s} {'a7rest(naive) - 7EKT':>26s}   {'geom-only - 7EKT':>26s}")
    for model in args.models:
        base_m = naive[f"7EKT|{model}"]["macro"]
        naive_m = naive[f"a7rest|{model}"]["macro"]
        geom_m = results[model]["macro"]
        d1, t1, p1 = paired(base_m, naive_m)
        d2, t2, p2 = paired(base_m, geom_m)
        print(f"{model:15s} Δ={d1:+.4f} t={t1:+.2f} p={p1:.3f}   "
              f"Δ={d2:+.4f} t={t2:+.2f} p={p2:.3f}")

    print(f"\nSaved -> {geom_file}")


if __name__ == "__main__":
    main()
