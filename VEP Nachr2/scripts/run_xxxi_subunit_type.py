"""
Run XXXI: subunit-type ablation — does restricting to one subunit class change scores?

The user's question: run alpha subunits alone, beta alone, and delta/gamma/epsilon
alone, and compare against the full 16-gene model. This is a biological heterogeneity
test — alpha subunits carry the principal (ACh-binding) face, beta and delta/gamma/
epsilon contribute the complementary/structural faces — so direction-of-effect might
be more (or less) predictable within a single subunit class.

Groups (by the curated subunit names in `subunit`):
    alpha                CHRNA1-7, CHRNA9, CHRNA10   (9 genes)
    beta                 CHRNB1-4                    (4 genes)
    delta_gamma_epsilon  CHRND, CHRNE, CHRNG         (3 genes)

Protocol matches the gene-aware runs (XXVI onward): cv_mode="standard" (plain
StratifiedKFold — gene identity is a legit input), 3-class, RF + CatBoost,
3 seeds × 15 trials, full 82-feat set. Features are extracted ONCE on the full 760
rows, then rows are subset by group, so the feature dimensionality stays 82 across
all conditions (the unused gene one-hot columns are constant within a subset and
tree models ignore them). Delta is vs the full 760 reference (RF 0.6525 / CatBoost
0.6499, Run XXVIII — not re-run).

Caveat to interpret with: alpha (517) is 68% of the data, so its comparison to full
is the clean one; beta (128) and delta/gamma/epsilon (115) are small-sample and will
carry more variance.

Usage:
    python scripts/run_xxxi_subunit_type.py
    python scripts/run_xxxi_subunit_type.py --models random_forest --n-seeds 1 --n-trials 10
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

# Known full-set reference (Run XXVIII) — not re-run here.
_FULL = {"random_forest": 0.6525, "catboost": 0.6499}

# Subunit-type groups, keyed by the curated `subunit` gene names.
GROUPS = {
    "alpha": {f"CHRNA{i}" for i in range(1, 8)} | {"CHRNA9", "CHRNA10"},
    "beta": {"CHRNB1", "CHRNB2", "CHRNB3", "CHRNB4"},
    "delta_gamma_epsilon": {"CHRND", "CHRNE", "CHRNG"},
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

    # Extract the full 82-feat set once, then subset rows per group.
    base_orch = FeatureOrchestrator(verbose=False)
    X_base, _ = base_orch.extract(df, ref_seqs=ref_seqs, use_cache=True)
    gating_feats = GatingExtractor().extract(df, ref_seqs, pdb_resources)
    inter_feats = AllostericInteractionExtractor().extract(df, ref_seqs, pdb_resources)
    X_full = np.hstack([X_base, gating_feats, inter_feats])
    assert X_full.shape == (len(df), 82), X_full.shape

    out_dir = RESULTS_DIR / "subunit_type"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    perseed = {}
    for name, genes in GROUPS.items():
        mask = df["subunit"].isin(genes).to_numpy()
        pos = np.flatnonzero(mask)
        X = X_full[pos]
        y_sub = y[pos]
        df_sub = df[mask].reset_index(drop=True)
        n = len(y_sub)
        dist = {LABEL_NAMES[k]: int(v) for k, v in zip(*np.unique(y_sub, return_counts=True))}
        n_genes = df_sub["subunit"].nunique()
        print(f"{'=' * 70}\nGROUP: {name}  (n={n}, genes={n_genes}, classes={dist})",
              flush=True)
        for model in args.models:
            res, ps = _run_cv(X, y_sub, df_sub, model, args.n_trials, args.n_seeds)
            perseed[f"{name}|{model}"] = ps
            rows.append({
                "group": name,
                "model": model,
                "n": n,
                "n_genes": n_genes,
                "n_lof": dist.get("LOF", 0),
                "n_gof": dist.get("GOF", 0),
                "n_nne": dist.get("No net effect", 0),
                "macro_f1_mean": res.get("macro_f1_mean", float("nan")),
                "macro_f1_std": res.get("macro_f1_std", float("nan")),
                "mcc_mean": res.get("mcc_mean", float("nan")),
                "balanced_accuracy_mean": res.get("balanced_accuracy_mean", float("nan")),
            })
            print(f"  [{model:15s}] macro F1 = {res.get('macro_f1_mean', float('nan')):.4f} "
                  f"± {res.get('macro_f1_std', float('nan')):.4f}")

    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "subunit_type_summary.csv", index=False)
    with open(out_dir / "subunit_type_perseed.json", "w") as _f:
        json.dump(perseed, _f, indent=2)

    piv = out.pivot_table(index="model", columns="group", values="macro_f1_mean")
    print(f"\n{'=' * 70}\nDelta macro F1 vs full (760 = "
          f"{_FULL['random_forest']}/{_FULL['catboost']}):")
    for model in args.models:
        if model not in piv.index:
            continue
        print(f"\n[{model}] full = {_FULL[model]:.4f}")
        for grp in piv.columns:
            d = piv.loc[model, grp] - _FULL[model]
            print(f"    {grp:22s} {piv.loc[model, grp]:.4f}  Δ={d:+.4f}")

    print(f"\nSaved {out_dir / 'subunit_type_summary.csv'}")


if __name__ == "__main__":
    main()
