"""
H3 diagnostic: open↔closed ΔΔG (energy two-state) vs the α7-only baseline (Run XIX).

Run XVI settled the α7-only GEOMETRY two-state test (ConformationalExtractor) as null.
H3 re-asks the same question with ENERGY: does a variant's inverse-folding ΔΔG
difference between the OPEN (7KOX) and CLOSED (7EKT) states predict GOF/LOF/NNE
direction? `OpenClosedDDGExtractor` emits ddg_open, ddg_closed, ddg_diff
(= ddg_open − ddg_closed; negative = open-favouring).

Scope is α7-only (the one gene with a complete open+closed pair), plain
StratifiedKFold (cv_mode="standard" — gene grouping is meaningless for 1 gene), and
the same honest protocol as Run XVI: drop the 16 dead subunit_* one-hots, keep the
3 species_* columns, and add a species-masked control to catch the species shortcut.

delta_f1 = F1(drop_ddg) − F1(full); NEGATIVE = the ΔΔG features HELP.

Usage:
    python scripts/run_h3_ddg.py --n-trials 15 --n-seeds 5
"""

import argparse

import numpy as np
import pandas as pd

from vep_nachr2.data.loader import load_mutation_data
from vep_nachr2.data.reference import load_all_reference_sequences
from vep_nachr2.features.orchestrator import FeatureOrchestrator
from vep_nachr2.features.experimental.ddg import OpenClosedDDGExtractor
from vep_nachr2.training.cross_validation import nested_cross_validation
from vep_nachr2.config import RESULTS_DIR, DEFAULT_SEEDS, LABEL_NAMES


OUT_DIR = RESULTS_DIR / "ddg"


def _group_of_each_column(orch, feature_names):
    """list[group_name] aligned to feature_names, in extractor order."""
    col = 0
    group_by_col = {}
    for e in orch.extractors:
        for _ in range(e.n_features):
            group_by_col[col] = e.name
            col += 1
    assert col == len(feature_names), f"{col} != {len(feature_names)}"
    return [group_by_col[c] for c in range(len(feature_names))]


def _run_cv(X, y, df, model_name, n_trials, n_seeds, n_folds, label):
    print(f"\n--- {label} [{model_name}]: {X.shape[1]} features ---", flush=True)
    cv = nested_cross_validation(
        X=X, y=y, df=df,
        model_name=model_name,
        n_outer_folds=n_folds,
        n_trials=n_trials,
        seeds=DEFAULT_SEEDS[:n_seeds],
        cv_mode="standard",  # plain StratifiedKFold — 1 gene, no subunit grouping
        verbose=False,
        n_classes=3,
        label_names=LABEL_NAMES,
    )
    return cv.get("final", {})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=15)
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--models", nargs="+", default=["random_forest", "catboost"])
    args = parser.parse_args()

    # ── α7-only, 3-class ──
    df = load_mutation_data(subunits=["CHRNA7"], data_file="final_mapped.xlsx")
    y = df["effect"].map({"LOF": 0, "No net effect": 1, "GOF": 2}).values.astype(np.int64)
    print(f"α7-only dataset: {len(df)} variants")
    print("species × effect crosstab (the confound to watch):")
    print(df.groupby(["species", "effect"]).size().unstack(fill_value=0), "\n")

    # ── Features: default + ddg (force_recompute — cache key ignores feature version) ──
    ref_seqs = load_all_reference_sequences("human")
    orch = FeatureOrchestrator(verbose=False)
    orch.extractors.append(OpenClosedDDGExtractor())
    X, feature_names = orch.extract(
        df, ref_seqs=ref_seqs, use_cache=False, force_recompute=True
    )
    col_groups = _group_of_each_column(orch, feature_names)
    print(f"Extracted {X.shape[1]} features.")

    is_subunit = np.array([fn.startswith("subunit_") for fn in feature_names])
    is_species = np.array([fn.startswith("species_") for fn in feature_names])
    is_ddg = np.array([g == "ddg" for g in col_groups])

    base_keep = ~is_subunit                        # drop 16 dead subunit one-hots
    no_species_keep = base_keep & ~is_species       # confound control
    drop_ddg_keep = base_keep & ~is_ddg             # the key ablation: baseline without ΔΔG

    all_rows = {}

    for model in args.models:
        print(f"\n{'#' * 60}\nMODEL: {model}\n{'#' * 60}")

        X_full = X[:, base_keep]
        full = _run_cv(X_full, y, df, model, args.n_trials, args.n_seeds, args.n_folds,
                       "full (+ddg)")
        all_rows[(model, "full")] = {"n_features": X_full.shape[1], **full}

        X_drop = X[:, drop_ddg_keep]
        drop = _run_cv(X_drop, y, df, model, args.n_trials, args.n_seeds, args.n_folds,
                       "drop_ddg (baseline)")
        all_rows[(model, "drop_ddg")] = {"n_features": X_drop.shape[1], **drop}

        X_nosp = X[:, no_species_keep]
        nosp = _run_cv(X_nosp, y, df, model, args.n_trials, args.n_seeds, args.n_folds,
                       "no_species")
        all_rows[(model, "no_species")] = {"n_features": X_nosp.shape[1], **nosp}

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for model in args.models:
        rows = {k[1]: v for k, v in all_rows.items() if k[0] == model}
        out = pd.DataFrame(rows).T
        full_f1 = out.loc["full", "macro_f1_mean"]
        out["delta_f1"] = out["macro_f1_mean"] - full_f1
        out = out[["n_features", "macro_f1_mean", "macro_f1_std", "delta_f1"]]

        print(f"\n{'=' * 70}\n{model}: α7-only 3-class | full(+ddg) F1 = {full_f1:.4f} "
              f"(± {out.loc['full','macro_f1_std']:.4f} across {args.n_seeds} seeds)")
        print("delta_f1 = F1(condition) - F1(full); drop_ddg NEGATIVE = ddg HELPS:\n")
        print(out.round(4).to_string())
        out.to_csv(OUT_DIR / f"h3_{model}.csv")
        print(f"Saved {OUT_DIR / f'h3_{model}.csv'}")


if __name__ == "__main__":
    main()
