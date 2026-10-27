"""
Run XXIX: species ablation — does cross-species data actually help? GENE-AWARE.

The user's question: would adding more species give us more useful data? Before
adding new species, test whether the cross-species data we ALREADY have (mouse,
rat) is helping at all. If human-only ≈ human+mouse+rat, then cross-species
pooling is neutral/negative and more species won't help.

Conditions (all gene-aware `cv_mode="standard"`, 3-class, RF + CatBoost,
3 seeds × 15 trials, full 82-feat set = base 75 + gating 3 + allosteric 4):

  human_only     507 variants   (the target distribution alone)
  human_mouse    587 variants   (does mouse add signal?)
  human_rat      680 variants   (does rat add signal?)
  all_three      760 variants   — ALREADY on disk (Run XXVIII full = 0.6525/0.6499)

Delta is vs human_only (positive = the added species helps).

Usage:
    python scripts/run_xxix_species_ablation.py
    python scripts/run_xxix_species_ablation.py --models random_forest --n-seeds 1 --n-trials 10
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

# Known reference (Run XXVIII full 82-feat) — not re-run here.
_ALL_THREE = {"random_forest": 0.6525, "catboost": 0.6499}


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

    # Extract features ONCE on the full set (cached), then subset rows.
    ref_seqs = load_all_reference_sequences("human")
    pdb_resources = load_all_pdb_resources()

    base_orch = FeatureOrchestrator(verbose=False)
    X_base, _ = base_orch.extract(df, ref_seqs=ref_seqs, use_cache=True)
    gating_feats = GatingExtractor().extract(df, ref_seqs, pdb_resources)
    inter_feats = AllostericInteractionExtractor().extract(df, ref_seqs, pdb_resources)
    X_full = np.hstack([X_base, gating_feats, inter_feats])
    assert X_full.shape == (len(df), 82), X_full.shape

    sp = df["species"].str.lower()
    human = (sp == "human").values
    mouse = (sp == "mouse").values
    rat = (sp == "rat").values

    conditions = [
        ("human_only", human),
        ("human_mouse", human | mouse),
        ("human_rat", human | rat),
    ]

    out_dir = RESULTS_DIR / "species_ablation"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    perseed = {}
    for name, mask in conditions:
        X = X_full[mask]
        y_sub = y[mask]
        df_sub = df[mask].reset_index(drop=True)
        n = len(y_sub)
        dist = dict(zip(*np.unique(y_sub, return_counts=True)))
        print(f"{'=' * 70}\nCONDITION: {name}  (n={n}, class counts={dist})", flush=True)
        for model in args.models:
            res, ps = _run_cv(X, y_sub, df_sub, model, args.n_trials, args.n_seeds)
            perseed[f"{name}|{model}"] = ps
            rows.append({
                "condition": name,
                "model": model,
                "n": n,
                "macro_f1_mean": res.get("macro_f1_mean", float("nan")),
                "macro_f1_std": res.get("macro_f1_std", float("nan")),
                "mcc_mean": res.get("mcc_mean", float("nan")),
                "balanced_accuracy_mean": res.get("balanced_accuracy_mean", float("nan")),
            })
            print(f"  [{model:15s}] macro F1 = {res.get('macro_f1_mean', float('nan')):.4f} "
                  f"± {res.get('macro_f1_std', float('nan')):.4f}")

    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "species_ablation_summary.csv", index=False)
    with open(out_dir / "species_ablation_perseed.json", "w") as _f:
        json.dump(perseed, _f, indent=2)

    piv = out.pivot_table(index="model", columns="condition", values="macro_f1_mean")
    print(f"\n{'=' * 70}\nDelta macro F1 vs human_only (positive = added species helps):")
    for model in args.models:
        if model not in piv.index:
            continue
        base = piv.loc[model, "human_only"]
        print(f"\n[{model}] human_only = {base:.4f}")
        for cond in ["human_mouse", "human_rat"]:
            d = piv.loc[model, cond] - base
            print(f"    {cond:14s} {piv.loc[model, cond]:.4f}  Δ={d:+.4f}")
        d_all = _ALL_THREE[model] - base
        print(f"    {'all_three':14s} {_ALL_THREE[model]:.4f}  Δ={d_all:+.4f}  (Run XXVIII, not re-run)")

    print(f"\nSaved {out_dir / 'species_ablation_summary.csv'}")


if __name__ == "__main__":
    main()
