"""
Feature-group leave-one-out ablation on the *mapped* dataset (extract once).

Unlike `runner ablation` (which re-extracts every feature group per condition and
so re-runs the ESM-2 forward pass ~11 times), this extracts the full matrix ONCE
(then caches it), and for each condition simply deletes that group's columns and
re-runs CV. Answers "which feature group has the most impact?" via
delta macro-F1 vs full — negative delta = the group helps, the more negative the
bigger the lift; near-zero/positive = candidate for removal.

Usage:
    python scripts/ablate_feature_impact.py --model random_forest
    python scripts/ablate_feature_impact.py --model random_forest --binary
"""

import argparse

import numpy as np
import pandas as pd

from vep_nachr2.data.loader import load_mutation_data
from vep_nachr2.data.reference import load_all_reference_sequences
from vep_nachr2.features.orchestrator import FeatureOrchestrator
from vep_nachr2.training.cross_validation import nested_cross_validation
from vep_nachr2.config import RESULTS_DIR, DEFAULT_SEEDS, LABEL_NAMES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="random_forest")
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--binary", action="store_true")
    args = parser.parse_args()

    print("Loading mapped data (final_mapped.xlsx, no re-remap)...", flush=True)
    if args.binary:
        df = load_mutation_data(
            effects=["GOF", "LOF"], data_file="final_mapped.xlsx", remap_nonhuman=False
        )
        y = df["effect"].map({"LOF": 0, "GOF": 1}).values.astype(np.int64)
        n_classes, label_names = 2, {0: "LOF", 1: "GOF"}
    else:
        df = load_mutation_data(data_file="final_mapped.xlsx", remap_nonhuman=False)
        y = df["effect"].map({"LOF": 0, "No net effect": 1, "GOF": 2}).values.astype(np.int64)
        n_classes, label_names = 3, LABEL_NAMES

    ref_seqs = load_all_reference_sequences("human")
    orch = FeatureOrchestrator(verbose=False)
    X, _ = orch.extract(df, ref_seqs=ref_seqs, use_cache=True)
    group_indices = orch.get_feature_group_indices()

    # Conditions: full + drop each non-empty group.
    conditions: list[tuple[str, int | None, int | None]] = [("full", None, None)]
    for name, (start, end) in group_indices.items():
        if end > start:
            conditions.append((f"drop_{name}", start, end))

    rows = {}
    for cond, start, end in conditions:
        if start is None:
            X_sub = X
        else:
            keep = np.ones(X.shape[1], dtype=bool)
            keep[start:end] = False
            X_sub = X[:, keep]

        print(f"\nAblation [{args.model}] {cond}: {X_sub.shape[1]} features", flush=True)
        cv = nested_cross_validation(
            X=X_sub, y=y, df=df,
            model_name=args.model,
            n_outer_folds=5,
            n_trials=args.n_trials,
            seeds=DEFAULT_SEEDS[:1],
            cv_mode="subunit",
            verbose=False,
            n_classes=n_classes,
            label_names=label_names,
        )
        rows[cond] = {"n_features": X_sub.shape[1], **cv.get("final", {})}

    out = pd.DataFrame(rows).T
    full_f1 = out.loc["full", "macro_f1_mean"]
    out["delta_f1"] = out["macro_f1_mean"] - full_f1
    out = out.sort_values("delta_f1")

    suffix = "_binary" if args.binary else ""
    out_dir = RESULTS_DIR / "ablation"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"ablation_impact_{args.model}{suffix}.csv"
    out.to_csv(out_path)

    print(f"\nFull-model macro F1: {full_f1:.4f}")
    print("delta_f1 = F1(drop group) - F1(full); negative = group helps:")
    print(out.to_string())
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
