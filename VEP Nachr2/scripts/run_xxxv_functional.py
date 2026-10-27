"""
Run XXXV: functional (GO) gene embedding — does function replace gene identity?

Func2Seq/Func2Prot's claim is that a *functional* (GO-style) representation lets a
model transfer structure across related proteins, instead of memorising opaque IDs.
Here: can a compact functional encoding of each subunit (α/β/δγε class + mechanistic
attributes: homomeric, high-Ca2+, muscle-type, gene length) substitute for the 16
gene one-hots, and do the mechanistic attributes add signal on top of identity?

Conditions (gene-aware `cv_mode="standard"`, 3-class, RF + CatBoost, 3 seeds × 15
trials, on the 82-feat reference = 75 base + gating 3 + allosteric 4):

  full                 82 feat   (reference)
  functional           89 feat   (82 + functional 7)              [augment]
  no_identity          66 feat   (82 − 16 gene one-hots)          [identity dropped]
  functional_no_identity 73 feat (82 − 16 + functional 7)         [function replaces identity]

Delta is vs `full`.

Usage:
    python scripts/run_xxxv_functional.py
    python scripts/run_xxxv_functional.py --models random_forest --n-seeds 1 --n-trials 10
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
from vep_nachr2.features.functional import FunctionalExtractor
from vep_nachr2.training.cross_validation import nested_cross_validation
from vep_nachr2.config import RESULTS_DIR, DEFAULT_SEEDS, LABEL_NAMES

_EFFECT_MAP = {"LOF": 0, "No net effect": 1, "GOF": 2}


def _build_matrices(df):
    """Return (X_full_82, full_names, X_func_7, func_names, gene_oh_mask)."""
    ref_seqs = load_all_reference_sequences("human")
    pdb_resources = load_all_pdb_resources()

    base_orch = FeatureOrchestrator(verbose=False)
    X_base, base_names = base_orch.extract(df, ref_seqs=ref_seqs, use_cache=True)

    gating_ext = GatingExtractor()
    inter_ext = AllostericInteractionExtractor()
    func_ext = FunctionalExtractor()

    gating = gating_ext.extract(df, ref_seqs, pdb_resources)
    inter = inter_ext.extract(df, ref_seqs, pdb_resources)
    func = func_ext.extract(df, ref_seqs, pdb_resources)

    X_full = np.hstack([X_base, gating, inter])
    full_names = list(base_names) + list(gating_ext.get_feature_names()) + list(inter_ext.get_feature_names())
    assert X_full.shape == (len(df), 82), f"expected 82, got {X_full.shape}"

    func_names = list(func_ext.get_feature_names())
    gene_oh_mask = np.array([not nm.startswith("subunit_") for nm in full_names])

    return X_full, full_names, func, func_names, gene_oh_mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=15)
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--models", nargs="+", default=["random_forest", "catboost"])
    args = parser.parse_args()

    df = load_mutation_data(data_file="final_mapped.xlsx")
    y = df["effect"].map(_EFFECT_MAP).values.astype(np.int64)

    X_full, full_names, X_func, func_names, gene_oh_mask = _build_matrices(df)
    X_full_nogene = X_full[:, gene_oh_mask]

    conditions = {
        "full": X_full,
        "functional": np.hstack([X_full, X_func]),
        "no_identity": X_full_nogene,
        "functional_no_identity": np.hstack([X_full_nogene, X_func]),
    }

    out_dir = RESULTS_DIR / "functional"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    perseed = {}
    for name, X in conditions.items():
        print(f"{'=' * 70}\nCONDITION: {name}  (n={len(y)}, n_features={X.shape[1]})", flush=True)
        for model in args.models:
            cv = nested_cross_validation(
                X=X, y=y, df=df, model_name=model,
                n_outer_folds=5, n_trials=args.n_trials,
                seeds=DEFAULT_SEEDS[:args.n_seeds],
                cv_mode="standard", verbose=False, n_classes=3, label_names=LABEL_NAMES,
            )
            res = cv.get("final", {})
            perseed[f"{name}|{model}"] = [float(r["aggregate"]["macro_f1_mean"]) for r in cv.get("per_seed", [])]
            rows.append({
                "condition": name,
                "model": model,
                "n_features": int(X.shape[1]),
                "macro_f1_mean": res.get("macro_f1_mean", float("nan")),
                "macro_f1_std": res.get("macro_f1_std", float("nan")),
                "mcc_mean": res.get("mcc_mean", float("nan")),
                "balanced_accuracy_mean": res.get("balanced_accuracy_mean", float("nan")),
            })
            print(f"  [{model:15s}] macro F1 = {res.get('macro_f1_mean', float('nan')):.4f} "
                  f"± {res.get('macro_f1_std', float('nan')):.4f}", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "functional_summary.csv", index=False)
    with open(out_dir / "functional_perseed.json", "w") as _f:
        json.dump(perseed, _f, indent=2)

    piv = out.pivot_table(index="model", columns="condition", values="macro_f1_mean")
    print(f"\n{'=' * 70}\nDelta macro F1 vs full:")
    for model in args.models:
        base = piv.loc[model, "full"]
        print(f"\n[{model}] full = {base:.4f}")
        for cond in ["functional", "no_identity", "functional_no_identity"]:
            print(f"    {cond:26s} {piv.loc[model, cond]:.4f}  Δ={piv.loc[model, cond] - base:+.4f}")

    print(f"\nSaved {out_dir / 'functional_summary.csv'}")


if __name__ == "__main__":
    main()
