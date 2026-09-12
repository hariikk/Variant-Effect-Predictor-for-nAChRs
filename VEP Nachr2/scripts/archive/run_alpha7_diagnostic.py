"""
α7-only structural diagnostic (Run XVI).

Settles whether complete, homomeric, two-state structure (7EKT closed / 7KOX open)
carries GOF/LOF/NNE direction signal — in the one gene where it is most likely to.

Self-contained: no changes to vep_nachr2/. Reuses the loader + orchestrator + nested
CV, adds the α7-only ConformationalExtractor (open↔closed delta, not in the default
set), and drops the 16 dead `subunit_*` one-hot columns.

Three outputs per model:
  1. α7-only 3-class macro F1 (mean ± std across seeds, cv_mode="standard").
  2. Drop-one-group ablation — delta_f1 = F1(drop) - F1(full); NEGATIVE = group helps.
     The crux: does `drop_conformational` go negative (open↔closed deltas predict direction)?
  3. Species-confound control — full model re-run with the 3 `species_*` columns masked,
     so we can see whether the "signal" is structure or the species shortcut.

Usage:
    python scripts/run_alpha7_diagnostic.py                 # defaults (20 trials, 5 seeds)
    python scripts/run_alpha7_diagnostic.py --n-trials 15 --n-seeds 3
"""

import argparse

import numpy as np
import pandas as pd

from vep_nachr2.data.loader import load_mutation_data
from vep_nachr2.data.reference import load_all_reference_sequences
from vep_nachr2.features.orchestrator import FeatureOrchestrator
from vep_nachr2.features.experimental.conformational import ConformationalExtractor
from vep_nachr2.training.cross_validation import nested_cross_validation
from vep_nachr2.config import RESULTS_DIR, DEFAULT_SEEDS, LABEL_NAMES


def _build_features(df, ref_seqs):
    """Extract the default feature set + the α7-only conformational group."""
    orch = FeatureOrchestrator(verbose=False)
    orch.extractors.append(ConformationalExtractor())  # +5 features, α7-only
    X, feature_names = orch.extract(df, ref_seqs=ref_seqs, use_cache=True)
    return orch, X, feature_names


def _group_of_each_column(orch, feature_names):
    """Return list[group_name] aligned to feature_names, in extractor order."""
    fname_to_group = {}
    col = 0
    for e in orch.extractors:
        for _ in range(e.n_features):
            fname_to_group[col] = e.name
            col += 1
    assert col == len(feature_names), f"{col} != {len(feature_names)}"
    return [fname_to_group[c] for c in range(len(feature_names))]


def _run_cv(X, y, df, model_name, n_trials, n_seeds, n_folds, label):
    print(f"\n--- {label} [{model_name}]: {X.shape[1]} features ---", flush=True)
    cv = nested_cross_validation(
        X=X, y=y, df=df,
        model_name=model_name,
        n_outer_folds=n_folds,
        n_trials=n_trials,
        seeds=DEFAULT_SEEDS[:n_seeds],
        cv_mode="standard",  # plain StratifiedKFold — gene grouping is meaningless for 1 gene
        verbose=False,
        n_classes=3,
        label_names=LABEL_NAMES,
    )
    return cv.get("final", {})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--models", nargs="+", default=["random_forest", "catboost"])
    parser.add_argument("--groups", default=None,
                        help="Comma-separated groups to ablate (default: all). "
                             "Use to scope RF to the structural groups only.")
    args = parser.parse_args()

    # ── Load α7-only, 3-class ──
    df = load_mutation_data(subunits=["CHRNA7"], data_file="final_mapped.xlsx")
    y = df["effect"].map({"LOF": 0, "No net effect": 1, "GOF": 2}).values.astype(np.int64)

    print(f"α7-only dataset: {len(df)} variants")
    print("species × effect crosstab (the confound to watch):")
    print(df.groupby(["species", "effect"]).size().unstack(fill_value=0), "\n")

    ref_seqs = load_all_reference_sequences("human")
    orch, X, feature_names = _build_features(df, ref_seqs)
    col_groups = _group_of_each_column(orch, feature_names)
    n_total = X.shape[1]
    print(f"Extracted {n_total} features ({len(set(col_groups))} groups + conformational).")

    # ── Column masks (operate on the FULL feature matrix via keep-masks) ──
    is_subunit = np.array([fn.startswith("subunit_") for fn in feature_names])  # 16 dead
    is_species = np.array([fn.startswith("species_") for fn in feature_names])   # 3 confound

    # Baseline: drop the 16 dead subunit one-hots, keep species.
    base_keep = ~is_subunit
    # Confound control: drop subunit AND species.
    no_species_keep = base_keep & ~is_species

    # Full-index columns for each group.
    groups = []
    for e in orch.extractors:
        if e.name not in groups:
            groups.append(e.name)
    if args.groups:
        requested = {g.strip() for g in args.groups.split(",")}
        ablate_groups = [g for g in groups if g in requested]
    else:
        ablate_groups = groups
    group_cols = {g: [c for c, gn in enumerate(col_groups) if gn == g] for g in groups}

    all_rows = {}

    for model in args.models:
        print(f"\n{'#' * 60}\nMODEL: {model}\n{'#' * 60}")

        # 1) Baseline (with species, subunit dropped).
        X_full = X[:, base_keep]
        full = _run_cv(X_full, y, df, model, args.n_trials, args.n_seeds, args.n_folds, "full (with species)")
        all_rows[(model, "full")] = {"n_features": X_full.shape[1], **full}

        # 2) Drop-one-group ablation.
        for g in ablate_groups:
            keep = base_keep.copy()
            keep[group_cols[g]] = False
            X_drop = X[:, keep]
            res = _run_cv(X_drop, y, df, model, args.n_trials, args.n_seeds, args.n_folds, f"drop_{g}")
            all_rows[(model, f"drop_{g}")] = {"n_features": X_drop.shape[1], **res}

        # 3) Species-confound control (species masked).
        X_nosp = X[:, no_species_keep]
        nosp = _run_cv(X_nosp, y, df, model, args.n_trials, args.n_seeds, args.n_folds, "no_species")
        all_rows[(model, "no_species")] = {"n_features": X_nosp.shape[1], **nosp}

    # ── Summarize per model ──
    out_dir = RESULTS_DIR / "alpha7"
    out_dir.mkdir(parents=True, exist_ok=True)

    for model in args.models:
        rows = {k[1]: v for k, v in all_rows.items() if k[0] == model}
        out = pd.DataFrame(rows).T
        full_f1 = out.loc["full", "macro_f1_mean"]
        out["delta_f1"] = out["macro_f1_mean"] - full_f1
        out = out[["n_features", "macro_f1_mean", "macro_f1_std", "delta_f1"]].sort_values("delta_f1")

        out_fmt = out.copy()
        for col in ["macro_f1_mean", "macro_f1_std", "delta_f1"]:
            out_fmt[col] = out_fmt[col].map(lambda x: f"{x:+.4f}")

        print(f"\n{'=' * 70}\n{model}: α7-only 3-class  |  full F1 = {full_f1:.4f} "
              f"(± {out.loc['full','macro_f1_std']:.4f} across {args.n_seeds} seeds)")
        print("delta_f1 = F1(condition) - F1(full); negative = the dropped group HELPS:\n")
        print(out_fmt.to_string())

        out.to_csv(out_dir / f"alpha7_diagnostic_{model}.csv")
        print(f"\nSaved {out_dir / f'alpha7_diagnostic_{model}.csv'}")


if __name__ == "__main__":
    main()
