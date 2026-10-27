"""
Run XXVIII: PER-FEATURE ablation of the structural features — GENE-AWARE.

Runs XXVII established that GNM gating (+0.013/+0.019) and allosteric-coupling
interaction (+0.005/+0.008) features *help* under gene-aware CV, and the core
structural group (rsa/cbeta/b_factor/dssp/is_unmappable) was neutral. But those
were whole-GROUP ablations. The user now wants to see each individual structural
feature's contribution — drop one column at a time, not the whole family.

Reference = the FULL gene-aware set (82 features):
    base 75 (positional/substitution/physicochemical/conservation/structural_core/
             esm2/esm_embeddings) + gating 3 + allosteric_interaction 4

Each of the 14 structural columns is dropped one at a time:
    structural_core (7):      rsa, cbeta_density, b_factor, dssp_helix,
                              dssp_sheet, dssp_coil, is_unmappable
    gating (3):               gnm_msf, gnm_slowmode, gnm_gate_coupling
    allosteric_interaction(4):allosteric_dvol, dhydro, dcharge, dpolarity

Plus 3 group-level anchors (drop whole structural_core / gating / interaction)
so per-feature deltas can be read against whole-group deltas.

Interpretation: delta = (drop feature) − (full). Negative = the feature HELPS
(dropping it hurts); positive = the feature HURTS (dropping it helps).

Usage:
    python scripts/run_xxviii_structural_feature_ablation.py
    python scripts/run_xxviii_structural_feature_ablation.py --models random_forest --n-seeds 1 --n-trials 10
"""

import argparse
import json
import sys
import warnings

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore", message=".*ill-defined.*")
warnings.filterwarnings("ignore", message=".*Only one class is present.*")
warnings.filterwarnings("ignore", message=".*ROC AUC score is not defined.*")

from vep_nachr2.data.loader import load_mutation_data
from vep_nachr2.data.reference import load_all_reference_sequences
from vep_nachr2.features.orchestrator import FeatureOrchestrator
from vep_nachr2.features.structural import load_all_pdb_resources
from vep_nachr2.features.gating import GatingExtractor
from vep_nachr2.features.allosteric_interaction import AllostericInteractionExtractor
from vep_nachr2.training.cross_validation import nested_cross_validation
from vep_nachr2.config import RESULTS_DIR, DEFAULT_SEEDS, LABEL_NAMES

_EFFECT_MAP = {"LOF": 0, "No net effect": 1, "GOF": 2}

# The 14 structural columns, in full-matrix order (see feature_names dump).
_STRUCTURAL = [
    # structural_core (indices 50-56 in the base 75)
    "rsa", "cbeta_density", "b_factor",
    "dssp_helix", "dssp_sheet", "dssp_coil", "is_unmappable",
    # gating (appended after base 75)
    "gnm_msf", "gnm_slowmode", "gnm_gate_coupling",
    # allosteric interaction (appended after gating)
    "allosteric_dvol", "allosteric_dhydro", "allosteric_dcharge", "allosteric_dpolarity",
]

# Group anchors for per-feature vs whole-group contrast.
_GROUPS = {
    "group_structural_core": ["rsa", "cbeta_density", "b_factor",
                              "dssp_helix", "dssp_sheet", "dssp_coil", "is_unmappable"],
    "group_gating": ["gnm_msf", "gnm_slowmode", "gnm_gate_coupling"],
    "group_allosteric": ["allosteric_dvol", "allosteric_dhydro",
                         "allosteric_dcharge", "allosteric_dpolarity"],
}


def _run_cv(X, y, df, model, n_trials, n_seeds):
    cv = nested_cross_validation(
        X=X, y=y, df=df,
        model_name=model,
        n_outer_folds=5,
        n_trials=n_trials,
        seeds=DEFAULT_SEEDS[:n_seeds],
        cv_mode="standard",  # gene-aware
        verbose=False,
        n_classes=3,
        label_names=LABEL_NAMES,
    )
    return cv.get("final", {}), [float(r["aggregate"]["macro_f1_mean"]) for r in cv.get("per_seed", [])]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=15)
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--models", nargs="+", default=["random_forest", "catboost"])
    args = parser.parse_args()

    df = load_mutation_data(data_file="final_mapped.xlsx")
    y = df["effect"].map(_EFFECT_MAP).values.astype(np.int64)

    ref_seqs = load_all_reference_sequences("human")
    pdb_resources = load_all_pdb_resources()

    base_orch = FeatureOrchestrator(verbose=False)
    X_base, base_names = base_orch.extract(df, ref_seqs=ref_seqs, use_cache=True)

    gating_feats = GatingExtractor().extract(df, ref_seqs, pdb_resources)
    gating_names = GatingExtractor.feature_names
    inter_feats = AllostericInteractionExtractor().extract(df, ref_seqs, pdb_resources)
    inter_names = AllostericInteractionExtractor.feature_names

    X_full = np.hstack([X_base, gating_feats, inter_feats])
    full_names = list(base_names) + list(gating_names) + list(inter_names)
    name_to_idx = {n: i for i, n in enumerate(full_names)}

    assert X_full.shape[1] == len(full_names) == 82, (X_full.shape, len(full_names))
    for s in _STRUCTURAL:
        assert s in name_to_idx, f"missing structural feature {s}"

    print(f"  full={X_full.shape}  ({len(_STRUCTURAL)} structural cols)\n", flush=True)

    out_dir = RESULTS_DIR / "structural_feature_ablation"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Conditions: the full reference + per-feature drops + group anchors.
    conditions = [("full", None)]  # (label, cols-to-drop list or None)
    for s in _STRUCTURAL:
        conditions.append((f"drop_{s}", [s]))
    for gname, cols in _GROUPS.items():
        conditions.append((f"drop_{gname}", cols))

    rows = []
    perseed = {}
    for label, drop_cols in conditions:
        if drop_cols is None:
            X = X_full
        else:
            drop_idx = [name_to_idx[c] for c in drop_cols]
            X = np.delete(X_full, drop_idx, axis=1)
        print(f"{'=' * 70}\nCONDITION: {label}  ({X.shape[1]} feats)", flush=True)
        for model in args.models:
            res, ps = _run_cv(X, y, df, model, args.n_trials, args.n_seeds)
            perseed[f"{label}|{model}"] = ps
            rows.append({
                "ablation": label,
                "model": model,
                "n_features": X.shape[1],
                "macro_f1_mean": res.get("macro_f1_mean", float("nan")),
                "macro_f1_std": res.get("macro_f1_std", float("nan")),
                "mcc_mean": res.get("mcc_mean", float("nan")),
                "balanced_accuracy_mean": res.get("balanced_accuracy_mean", float("nan")),
            })
            print(f"  [{model:15s}] macro F1 = {res.get('macro_f1_mean', float('nan')):.4f} "
                  f"± {res.get('macro_f1_std', float('nan')):.4f}")

    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "structural_feature_ablation_summary.csv", index=False)
    with open(out_dir / "structural_feature_ablation_perseed.json", "w") as _f:
        json.dump(perseed, _f, indent=2)

    # Delta view: (drop feature) − (full). Negative = feature helps.
    piv = out.pivot_table(index="model", columns="ablation", values="macro_f1_mean")
    print(f"\n{'=' * 70}\nDelta macro F1 vs full (negative = feature helps):")
    for model in args.models:
        if model not in piv.index:
            continue
        full = piv.loc[model, "full"]
        print(f"\n[{model}] full = {full:.4f}")
        deltas = {}
        for col in piv.columns:
            if col == "full":
                continue
            deltas[col] = piv.loc[model, col] - full
        for col in sorted(deltas, key=lambda c: deltas[c]):
            print(f"    {col:28s} {piv.loc[model, col]:.4f}  Δ={deltas[col]:+.4f}")

    print(f"\nSaved {out_dir / 'structural_feature_ablation_summary.csv'}")


if __name__ == "__main__":
    main()
