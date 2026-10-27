"""
Run XXX: data-size & class-balance ablation — is data the binding constraint?

The user's question: is the model limited by the *amount* of data, or by the
*balance* of classes, or is it at an intrinsic ceiling? Two factors are varied
independently:

  SIZE (at fixed ~50/25/25 balance):
    quarter_502525  ~190   per-gene fractional sample (0.25×)
    half_502525     ~380   per-gene fractional sample (0.50×)   [user exp 1]
    full            760    already on disk (Run XXVIII = 0.6525/0.6499)

  BALANCE (at fixed N=350):
    n350_502525     350    random, 50/25/25 (175 LOF / 88 GOF / 87 NNE)   [user exp 2]
    n350_333333     350    random, 33/33/33 (117/117/116)                 [user exp 3]
    n350_333333_gene 350   gene-stratified 33/33/33                       [better idea]

"Better idea" for the 33/33/33 selection: a *random* 350-sample can drop an entire
rare gene's variants, which is exactly the information the gene-aware model leans on
(gene identity is the #1 feature). The gene-stratified variant samples each class
proportionally per gene, so every gene stays represented — a fairer test of "what
does 350 balanced points buy" for THIS model.

All conditions gene-aware (`cv_mode="standard"`), 3-class, RF + CatBoost,
3 seeds × 15 trials, full 82-feat set. Delta is vs full (760).

Usage:
    python scripts/run_xxx_data_size.py
    python scripts/run_xxx_data_size.py --models random_forest --n-seeds 1 --n-trials 10
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


def per_gene_frac(df, frac, seed):
    """Fractional per-(gene, class) sample — preserves gene coverage AND balance."""
    parts = []
    for (gene, cls), g in df.groupby(["subunit", "effect"]):
        n_take = max(1, int(round(len(g) * frac))) if len(g) > 0 else 0
        parts.append(g.sample(n=min(n_take, len(g)), random_state=seed))
    return pd.concat(parts)


def strat_sample(df, n_per_class, seed):
    """Global random sample per class (exact counts)."""
    parts = []
    for cls, n in n_per_class.items():
        g = df[df["effect"] == cls]
        parts.append(g.sample(n=min(n, len(g)), random_state=seed))
    return pd.concat(parts)


def gene_strat_sample(df, n_per_class, seed):
    """Per-class sample, proportional per gene (keeps every gene represented)."""
    parts = []
    for cls, n in n_per_class.items():
        g = df[df["effect"] == cls]
        frac = n / len(g)
        for gene, gg in g.groupby("subunit"):
            n_take = max(1, int(round(len(gg) * frac))) if len(gg) > 0 else 0
            parts.append(gg.sample(n=min(n_take, len(gg)), random_state=seed))
    return pd.concat(parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=15)
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--models", nargs="+", default=["random_forest", "catboost"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = load_mutation_data(data_file="final_mapped.xlsx")
    y = df["effect"].map(_EFFECT_MAP).values.astype(np.int64)

    ref_seqs = load_all_reference_sequences("human")
    pdb_resources = load_all_pdb_resources()

    base_orch = FeatureOrchestrator(verbose=False)
    X_base, _ = base_orch.extract(df, ref_seqs=ref_seqs, use_cache=True)
    gating_feats = GatingExtractor().extract(df, ref_seqs, pdb_resources)
    inter_feats = AllostericInteractionExtractor().extract(df, ref_seqs, pdb_resources)
    X_full = np.hstack([X_base, gating_feats, inter_feats])
    assert X_full.shape == (len(df), 82), X_full.shape

    # Build the sub-datasets (each returns a DataFrame; .index = original row pos).
    subs = {
        "quarter_502525": per_gene_frac(df, 0.25, args.seed),
        "half_502525": per_gene_frac(df, 0.50, args.seed),
        "n350_502525": strat_sample(df, {"LOF": 175, "GOF": 88, "No net effect": 87}, args.seed),
        "n350_333333": strat_sample(df, {"LOF": 117, "GOF": 117, "No net effect": 116}, args.seed),
        "n350_333333_gene": gene_strat_sample(df, {"LOF": 117, "GOF": 117, "No net effect": 116}, args.seed),
    }

    out_dir = RESULTS_DIR / "data_size"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    perseed = {}
    for name, sub in subs.items():
        pos = sub.index.to_numpy()
        X = X_full[pos]
        y_sub = y[pos]
        df_sub = sub.reset_index(drop=True)
        n = len(y_sub)
        dist = {LABEL_NAMES[k]: int(v) for k, v in zip(*np.unique(y_sub, return_counts=True))}
        print(f"{'=' * 70}\nCONDITION: {name}  (n={n}, classes={dist})", flush=True)
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
    out.to_csv(out_dir / "data_size_summary.csv", index=False)
    with open(out_dir / "data_size_perseed.json", "w") as _f:
        json.dump(perseed, _f, indent=2)

    piv = out.pivot_table(index="model", columns="condition", values="macro_f1_mean")
    print(f"\n{'=' * 70}\nDelta macro F1 vs full (760 = {_FULL['random_forest']}/{_FULL['catboost']}):")
    for model in args.models:
        if model not in piv.index:
            continue
        print(f"\n[{model}]")
        for cond in piv.columns:
            d = piv.loc[model, cond] - _FULL[model]
            print(f"    {cond:22s} {piv.loc[model, cond]:.4f}  Δ={d:+.4f}")

    print(f"\nSaved {out_dir / 'data_size_summary.csv'}")


if __name__ == "__main__":
    main()
