"""
Run XXXVI: structural-alignment gating of cross-species rows.

EnergyFingerprint's claim: zero-shot cross-gene transfer degrades along a
structural-similarity gradient, with a hard boundary at disordered regions.
Run XXIX found the +0.05 cross-species augmentation gain is *label rebalancing*,
not species diversity. These are in tension.

Test it: split the non-human (mouse/rat) rows by *orderedness* of the
human-homologous residue (DSSP helix/sheet = ordered; coil/unmappable =
disordered), then compare whether the augmentation gain is carried by the
structurally well-aligned rows or is insensitive to alignment.

Conditions (gene-aware `cv_mode="standard"`, 3-class, RF + CatBoost, 3 seeds × 15
trials, 82-feat set = 75 base + gating 3 + allosteric 4):

  human_only            human only (the target distribution)
  human+ordered         human + ordered mouse/rat
  human+disordered      human + disordered mouse/rat
  all_three             human + mouse + rat  (Run XXIX reference ≈ +0.05)

Delta is vs human_only. If ordered ≫ disordered, EnergyFingerprint's structural
prior wins; if they're equal, the label-rebalancing story holds.

Usage:
    python scripts/run_xxxvi_structure_gating.py
    python scripts/run_xxxvi_structure_gating.py --models random_forest --n-seeds 1 --n-trials 10
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


def _build_matrices(df):
    """Return (X_full_82, full_names)."""
    ref_seqs = load_all_reference_sequences("human")
    pdb_resources = load_all_pdb_resources()

    base_orch = FeatureOrchestrator(verbose=False)
    X_base, base_names = base_orch.extract(df, ref_seqs=ref_seqs, use_cache=True)

    gating_ext = GatingExtractor()
    inter_ext = AllostericInteractionExtractor()

    gating = gating_ext.extract(df, ref_seqs, pdb_resources)
    inter = inter_ext.extract(df, ref_seqs, pdb_resources)

    X_full = np.hstack([X_base, gating, inter])
    full_names = list(base_names) + list(gating_ext.get_feature_names()) + list(inter_ext.get_feature_names())
    assert X_full.shape == (len(df), 82), f"expected 82, got {X_full.shape}"

    return X_full, full_names


def _ordered_mask(X_full, full_names):
    """Ordered = defined secondary structure (helix/sheet) at the mapped residue.

    Coil or unmappable (is_unmappable=1, which implies coil) counts as disordered.
    """
    dssp_helix = full_names.index("dssp_helix")
    dssp_sheet = full_names.index("dssp_sheet")
    return (X_full[:, dssp_helix] + X_full[:, dssp_sheet]) > 0.5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=15)
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--models", nargs="+", default=["random_forest", "catboost"])
    args = parser.parse_args()

    df = load_mutation_data(data_file="final_mapped.xlsx")
    y = df["effect"].map(_EFFECT_MAP).values.astype(np.int64)

    X_full, full_names = _build_matrices(df)
    ordered = _ordered_mask(X_full, full_names)

    sp = df["species"].str.lower()
    human = (sp == "human").values
    nonhuman = ~human
    ordered_nh = nonhuman & ordered
    disordered_nh = nonhuman & ~ordered

    print(f"Non-human rows: {int(nonhuman.sum())} "
          f"(ordered {int(ordered_nh.sum())} / disordered {int(disordered_nh.sum())})", flush=True)

    conditions = [
        ("human_only", human),
        ("human_ordered", human | ordered_nh),
        ("human_disordered", human | disordered_nh),
        ("all_three", np.ones(len(df), dtype=bool)),
    ]

    out_dir = RESULTS_DIR / "structure_gating"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    perseed = {}
    for name, mask in conditions:
        X = X_full[mask]
        y_sub = y[mask]
        df_sub = df[mask].reset_index(drop=True)
        dist = dict(zip(*np.unique(y_sub, return_counts=True)))
        print(f"{'=' * 70}\nCONDITION: {name}  (n={len(y_sub)}, class counts={dist})", flush=True)
        for model in args.models:
            cv = nested_cross_validation(
                X=X, y=y_sub, df=df_sub, model_name=model,
                n_outer_folds=5, n_trials=args.n_trials,
                seeds=DEFAULT_SEEDS[:args.n_seeds],
                cv_mode="standard", verbose=False, n_classes=3, label_names=LABEL_NAMES,
            )
            res = cv.get("final", {})
            perseed[f"{name}|{model}"] = [float(r["aggregate"]["macro_f1_mean"]) for r in cv.get("per_seed", [])]
            rows.append({
                "condition": name,
                "model": model,
                "n": int(len(y_sub)),
                "macro_f1_mean": res.get("macro_f1_mean", float("nan")),
                "macro_f1_std": res.get("macro_f1_std", float("nan")),
                "mcc_mean": res.get("mcc_mean", float("nan")),
                "balanced_accuracy_mean": res.get("balanced_accuracy_mean", float("nan")),
            })
            print(f"  [{model:15s}] macro F1 = {res.get('macro_f1_mean', float('nan')):.4f} "
                  f"± {res.get('macro_f1_std', float('nan')):.4f}", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "structure_gating_summary.csv", index=False)
    with open(out_dir / "structure_gating_perseed.json", "w") as _f:
        json.dump(perseed, _f, indent=2)

    piv = out.pivot_table(index="model", columns="condition", values="macro_f1_mean")
    print(f"\n{'=' * 70}\nDelta macro F1 vs human_only (positive = added rows help):")
    for model in args.models:
        base = piv.loc[model, "human_only"]
        print(f"\n[{model}] human_only = {base:.4f}")
        for cond in ["human_ordered", "human_disordered", "all_three"]:
            print(f"    {cond:18s} {piv.loc[model, cond]:.4f}  Δ={piv.loc[model, cond] - base:+.4f}")

    print(f"\nSaved {out_dir / 'structure_gating_summary.csv'}")


if __name__ == "__main__":
    main()
