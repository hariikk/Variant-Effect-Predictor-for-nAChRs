"""
Run XXVII: allosteric-coupling × mutation interaction features — GENE-AWARE.

Colleague's "indirect connection" hypothesis: a mutation's effect is transmitted
to the functional machinery through coupled residues, so the *interaction* of
allosteric coupling (GNM gate_coupling) with the mutation's chemical change
(Δvolume/Δhydrophobicity/Δcharge/Δpolarity) should carry direction signal that
neither quantity alone does.

Per the gene-aware protocol (memory: gene-aware-protocol), this runs under
cv_mode="standard" — gene identity is a legitimate input. Runs XVIII/XXV tested
raw gate proximity / raw GNM coupling gene-AGNOSTIC and were null; this re-tests
the raw GNM gating group AND the new interaction group under gene-aware CV.

Conditions (all gene-aware, 3-class, RF + CatBoost, 3 seeds × 15 trials):
  baseline                 75 features
  baseline_gating          78 features (re-test Run XXV's gnm_msf/slowmode/gate_coupling)
  baseline_allosteric      79 features (the new interaction features)

Usage:
    python scripts/run_xxvii_allosteric.py
    python scripts/run_xxvii_allosteric.py --models random_forest --n-seeds 1 --n-trials 10
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

    # ── Features ────────────────────────────────────────────────────────────
    # baseline is the cached 75-feature set; gating + interaction columns are
    # computed on top (CPU GNM only — no ESM re-extraction).
    ref_seqs = load_all_reference_sequences("human")
    pdb_resources = load_all_pdb_resources()

    base_orch = FeatureOrchestrator(verbose=False)
    X_base, _ = base_orch.extract(df, ref_seqs=ref_seqs, use_cache=True)

    gating_feats = GatingExtractor().extract(df, ref_seqs, pdb_resources)
    inter_feats = AllostericInteractionExtractor().extract(df, ref_seqs, pdb_resources)

    X_gating = np.hstack([X_base, gating_feats])
    X_inter = np.hstack([X_base, inter_feats])

    print(f"  baseline={X_base.shape}  +gating={X_gating.shape}  "
          f"+allosteric={X_inter.shape}\n", flush=True)

    out_dir = RESULTS_DIR / "allosteric_interaction"
    out_dir.mkdir(parents=True, exist_ok=True)

    conditions = [
        ("baseline", X_base),
        ("baseline_gating", X_gating),
        ("baseline_allosteric", X_inter),
    ]

    rows = []
    perseed = {}
    for name, X in conditions:
        print(f"{'=' * 70}\nCONDITION: {name}  ({X.shape[1]} feats)", flush=True)
        for model in args.models:
            res, ps = _run_cv(X, y, df, model, args.n_trials, args.n_seeds)
            perseed[f"{name}|{model}"] = ps
            rows.append({
                "condition": name,
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
    out.to_csv(out_dir / "allosteric_interaction_summary.csv", index=False)
    with open(out_dir / "allosteric_interaction_perseed.json", "w") as _f:
        json.dump(perseed, _f, indent=2)

    print(f"\n{'=' * 70}\nDelta macro F1 (condition − baseline; positive = helps):")
    piv = out.pivot_table(index="model", columns="condition", values="macro_f1_mean")
    for model in args.models:
        if model not in piv.index:
            continue
        base = piv.loc[model, "baseline"]
        for cond in ["baseline_gating", "baseline_allosteric"]:
            d = piv.loc[model, cond] - base
            print(f"  [{model:15s}] {cond:22s} {piv.loc[model, cond]:.4f}  Δ={d:+.4f}")

    print(f"\nSaved {out_dir / 'allosteric_interaction_summary.csv'}")


if __name__ == "__main__":
    main()
