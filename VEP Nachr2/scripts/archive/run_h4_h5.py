"""
H4 (coevolution) + H5 (embedding-delta) diagnostic vs the 75-feat baseline (Run XX + XXI).

H4 = "residues that coevolve with the gating hinge are functionally constrained, so a
     mutation there has a strong directional effect." CoevolutionExtractor (2 features:
     coev_gate = ESM-2 attention coupling to the M2 gate ring, coev_hub = centrality)
     from last-layer attention (need_head_weights=True, head-averaged, symmetrised).

H5 = "the mutant-minus-WT embedding delta carries direction signal." EmbeddingDeltaExtractor
     (16 PCA comps of emb(seq[mt])[pos] - emb(seq[wt])[pos]).

Both run on the full multi-species 3-class dataset (760 variants), honest subunit-grouped
CV (cv_mode="subunit"), 5 seeds, RF + catboost. delta_f1 = F1(cond) - F1(baseline);
positive = the added features HELP.

Two phases (same as H1/H2):
  --extract-only   build features (baseline + coev + emb_delta) → npz, then exit
  --skip-extract   load the cached npz and go straight to CV
  (default)        extract if npz missing, then run CV

Usage:
    python scripts/run_h4_h5.py --extract-only
    python scripts/run_h4_h5.py --skip-extract --n-trials 15 --n-seeds 5
"""

import argparse

import numpy as np
import pandas as pd

from vep_nachr2.data.loader import load_mutation_data
from vep_nachr2.data.reference import load_all_reference_sequences
from vep_nachr2.features.orchestrator import FeatureOrchestrator
from vep_nachr2.features.experimental.coevolution import CoevolutionExtractor
from vep_nachr2.features.experimental.emb_delta import EmbeddingDeltaExtractor
from vep_nachr2.training.cross_validation import nested_cross_validation
from vep_nachr2.config import RESULTS_DIR, DEFAULT_SEEDS, LABEL_NAMES


FEAT_NPZ = RESULTS_DIR / "h4_h5" / "features.npz"


def build_features(df, ref_seqs, force=False):
    """Extract default 75 + coev 2 + emb_delta 16 = 93 features, cache to npz."""
    if FEAT_NPZ.exists() and not force:
        d = np.load(FEAT_NPZ, allow_pickle=True)
        return d["X"], list(d["feature_names"]), list(d["col_groups"])

    orch = FeatureOrchestrator(verbose=True)
    orch.extractors.append(CoevolutionExtractor())
    orch.extractors.append(EmbeddingDeltaExtractor())
    X, names = orch.extract(df, ref_seqs=ref_seqs, use_cache=False, force_recompute=True)

    col_groups = []
    for e in orch.extractors:
        col_groups.extend([e.name] * e.n_features)
    assert len(col_groups) == len(names) == X.shape[1], (
        f"group/name/X mismatch: {len(col_groups)} vs {len(names)} vs {X.shape[1]}"
    )

    FEAT_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        FEAT_NPZ,
        X=X,
        feature_names=np.array(names, dtype=object),
        col_groups=np.array(col_groups, dtype=object),
    )
    print(f"Cached features to {FEAT_NPZ}: {X.shape}")
    return X, names, col_groups


def _run_cv(X, y, df, model, n_trials, n_seeds, label):
    print(f"\n--- {label} [{model}]: {X.shape[1]} features ---", flush=True)
    cv = nested_cross_validation(
        X=X, y=y, df=df,
        model_name=model,
        n_outer_folds=5,
        n_trials=n_trials,
        seeds=DEFAULT_SEEDS[:n_seeds],
        cv_mode="subunit",
        verbose=False,
        n_classes=3,
        label_names=LABEL_NAMES,
    )
    return cv.get("final", {})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--extract-only", action="store_true")
    p.add_argument("--skip-extract", action="store_true")
    p.add_argument("--force-extract", action="store_true")
    p.add_argument("--n-trials", type=int, default=15)
    p.add_argument("--n-seeds", type=int, default=5)
    p.add_argument("--models", nargs="+", default=["random_forest", "catboost"])
    args = p.parse_args()

    df = load_mutation_data(data_file="final_mapped.xlsx")
    y = df["effect"].map({"LOF": 0, "No net effect": 1, "GOF": 2}).values.astype(np.int64)
    ref_seqs = load_all_reference_sequences("human")

    if args.extract_only:
        build_features(df, ref_seqs, force=args.force_extract)
        print("Extract-only: done.")
        return

    X, names, groups = build_features(df, ref_seqs, force=args.force_extract)
    groups = np.array(groups)

    is_coev = np.array([g == "coev" for g in groups])
    is_emb = np.array([g == "emb_delta" for g in groups])

    conditions = {
        "baseline":             ~is_coev & ~is_emb,       # 75
        "baseline_plus_coev":   ~is_emb,                   # 77
        "baseline_plus_embdelta": ~is_coev,                # 91
    }

    print(f"\nFeatures: {X.shape[1]} total. Conditions: {list(conditions)}")

    all_rows = {}
    for model in args.models:
        print(f"\n{'#' * 60}\nMODEL: {model}\n{'#' * 60}")
        for cond, keep in conditions.items():
            Xc = X[:, keep]
            res = _run_cv(Xc, y, df, model, args.n_trials, args.n_seeds, cond)
            all_rows[(model, cond)] = {"n_features": Xc.shape[1], **res}

    # Save per-hypothesis tables (baseline + that hypothesis's condition).
    for model in args.models:
        rows = {k[1]: v for k, v in all_rows.items() if k[0] == model}
        full = pd.DataFrame(rows).T
        base = full.loc["baseline", "macro_f1_mean"]
        full["delta_f1"] = full["macro_f1_mean"] - base
        full = full[["n_features", "macro_f1_mean", "macro_f1_std", "delta_f1"]]

        for hyp, cond, outname in (
            ("h4", "baseline_plus_coev", "coev"),
            ("h5", "baseline_plus_embdelta", "emb_delta"),
        ):
            table = full.loc[["baseline", cond]]
            out_dir = RESULTS_DIR / outname
            out_dir.mkdir(parents=True, exist_ok=True)
            table.to_csv(out_dir / f"{hyp}_{model}.csv")
            print(f"\n{'=' * 70}\n{model}: {cond} | baseline F1 = {base:.4f}")
            print("delta_f1 = F1(cond) - F1(baseline); positive = helps:\n")
            print(table.round(4).to_string())
            print(f"Saved {out_dir / f'{hyp}_{model}.csv'}")


if __name__ == "__main__":
    main()
