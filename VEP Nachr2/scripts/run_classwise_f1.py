"""
Class-wise F1 + cheating check (Run XXXVIII — diagnostic).

Answers three questions in one sweep:

  1. *Per-class* F1 (LOF / No-net-effect / GOF) for every model, on the full data
     and on each subunit group alone (α / β / δ/γ/ε) — does a model earn its macro
     F1 by distinguishing all three directions, or by nailing one class?
  2. Baselines: majority-class ("always predict the mode"), uniform-random (chance),
     and stratified-random (priors) — so each model's per-class F1 can be judged
     against "just predict the most common effect".
  3. The δ/γ/ε cheating test explicitly: since LOF is the majority there, how much
     macro F1 does the model lose vs a classifier that *always predicts LOF*?

Protocol matches the gene-aware runs (cv_mode="standard", 3-class, 5 outer × 15
Optuna trials). The 82-feature set is extracted once and subset by row, so all
conditions share identical features. Baselines are analytic (DummyClassifier over
the full subset) — no CV needed since they don't learn.

Usage:
    python scripts/run_classwise_f1.py
    python scripts/run_classwise_f1.py --models random_forest catboost --n-seeds 1 --n-trials 10
"""

import argparse
import json
import sys
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score, balanced_accuracy_score, matthews_corrcoef

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
from vep_nachr2.training.evaluation import compute_metrics
from vep_nachr2.config import RESULTS_DIR, DEFAULT_SEEDS, LABEL_NAMES, ALL_MODELS, NACHR_GENES

_EFFECT_MAP = {"LOF": 0, "No net effect": 1, "GOF": 2}
_LABELS = [0, 1, 2]
_CLASS_NAMES = [LABEL_NAMES[l] for l in _LABELS]  # ["LOF", "No net effect", "GOF"]

GROUPS = {
    "all": set(NACHR_GENES),
    "alpha": {f"CHRNA{i}" for i in range(1, 8)} | {"CHRNA9", "CHRNA10"},
    "beta": {"CHRNB1", "CHRNB2", "CHRNB3", "CHRNB4"},
    "delta_gamma_epsilon": {"CHRND", "CHRNE", "CHRNG"},
}


# ── Baselines ────────────────────────────────────────────────────────────────

def _dummy_baseline(y, strategy, n_draws=200):
    """Return {macro_f1, f1_LOF, f1_NNE, f1_GOF, balanced_accuracy, mcc} for a
    DummyClassifier. most_frequent is deterministic; uniform/stratified are
    averaged over n_draws random draws."""
    n = len(y)
    dummy_X = np.zeros((n, 1))

    if strategy == "most_frequent":
        clf = DummyClassifier(strategy="most_frequent")
        clf.fit(dummy_X, y)
        y_pred = clf.predict(dummy_X)
        m = compute_metrics(y, y_pred, None, labels=_LABELS, label_names=LABEL_NAMES)
        return _flatten(m)

    macro, bal, mcc = [], [], []
    per_cls = {c: [] for c in _CLASS_NAMES}
    for s in range(n_draws):
        clf = DummyClassifier(strategy=strategy, random_state=s)
        clf.fit(dummy_X, y)
        y_pred = clf.predict(dummy_X)
        macro.append(f1_score(y, y_pred, average="macro", labels=_LABELS, zero_division=0))
        bal.append(balanced_accuracy_score(y, y_pred))
        mcc.append(matthews_corrcoef(y, y_pred))
        for l, c in zip(_LABELS, _CLASS_NAMES):
            per_cls[c].append(f1_score(y == l, y_pred == l, zero_division=0))
    return {
        "macro_f1_mean": float(np.mean(macro)),
        "balanced_accuracy_mean": float(np.mean(bal)),
        "mcc_mean": float(np.mean(mcc)),
        **{f"f1_{c}_mean": float(np.mean(per_cls[c])) for c in _CLASS_NAMES},
    }


def _flatten(metrics):
    return {
        "macro_f1_mean": metrics["macro_f1"],
        "balanced_accuracy_mean": metrics["balanced_accuracy"],
        "mcc_mean": metrics["mcc"],
        **{f"f1_{c}_mean": metrics[f"f1_{c}"] for c in _CLASS_NAMES},
    }


def _cv_per_class(cv, n_seeds):
    """Aggregate per-class F1 + macro across seeds from nested_cross_validation."""
    out = {}
    for key in ["macro_f1", "balanced_accuracy", "mcc"]:
        vals = [cv["per_seed"][s]["aggregate"][f"{key}_mean"] for s in range(n_seeds)]
        out[f"{key}_mean"] = float(np.mean(vals))
        out[f"{key}_std"] = float(np.std(vals))
    for c in _CLASS_NAMES:
        vals = [cv["per_seed"][s]["aggregate"][f"f1_{c}_mean"] for s in range(n_seeds)]
        out[f"f1_{c}_mean"] = float(np.mean(vals))
        out[f"f1_{c}_std"] = float(np.std(vals))
    return out


def _run_cv(X, y, df, model, n_trials, n_seeds):
    return nested_cross_validation(
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=15)
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--models", nargs="+", default=ALL_MODELS)
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

    out_dir = RESULTS_DIR / "classwise_f1"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "classwise_f1_summary.csv"
    dist_path = out_dir / "class_distribution.csv"

    def _save():
        # Incremental save so a long run survives interruption / partial monitoring.
        pd.DataFrame(rows).to_csv(summary_path, index=False)
        pd.DataFrame(dist_rows).to_csv(dist_path, index=False)

    rows = []
    dist_rows = []

    for gname, genes in GROUPS.items():
        mask = df["subunit"].isin(genes).to_numpy()
        pos = np.flatnonzero(mask)
        X = X_full[pos]
        y_sub = y[pos]
        df_sub = df[mask].reset_index(drop=True)
        n = len(y_sub)

        # Class distribution + majority class.
        counts = {LABEL_NAMES[l]: int((y_sub == l).sum()) for l in _LABELS}
        majority = max(_LABELS, key=lambda l: (y_sub == l).sum())
        majority_name = LABEL_NAMES[majority]
        purity = counts[majority_name] / n

        dist_rows.append({
            "group": gname, "n": n, "n_genes": df_sub["subunit"].nunique(),
            "n_LOF": counts["LOF"], "n_No net effect": counts["No net effect"],
            "n_GOF": counts["GOF"], "majority_class": majority_name,
            "majority_share": round(purity, 4),
        })
        print(f"\n{'=' * 70}\nGROUP: {gname}  n={n}  genes={df_sub['subunit'].nunique()}")
        print(f"  classes: {counts}  majority={majority_name} ({purity:.1%})", flush=True)

        # ── Baselines ────────────────────────────────────────────────────────
        for strat, label in [("most_frequent", "majority_class"),
                             ("uniform", "uniform_random"),
                             ("stratified", "stratified_random")]:
            b = _dummy_baseline(y_sub, strat)
            rows.append({
                "group": gname, "model": label, "n": n,
                "macro_f1_mean": b["macro_f1_mean"], "macro_f1_std": 0.0,
                **{f"f1_{c}_mean": b[f"f1_{c}_mean"] for c in _CLASS_NAMES},
                "balanced_accuracy_mean": b["balanced_accuracy_mean"],
                "mcc_mean": b["mcc_mean"],
            })
            _save()
            print(f"  [baseline {label:18s}] macro F1 = {b['macro_f1_mean']:.4f}  "
                  f"(LOF {b['f1_LOF_mean']:.3f} / NNE {b['f1_No net effect_mean']:.3f} / "
                  f"GOF {b['f1_GOF_mean']:.3f})")

        # ── Models ───────────────────────────────────────────────────────────
        for model in args.models:
            try:
                cv = _run_cv(X, y_sub, df_sub, model, args.n_trials, args.n_seeds)
                r = _cv_per_class(cv, args.n_seeds)
                rows.append({
                    "group": gname, "model": model, "n": n,
                    "macro_f1_mean": r["macro_f1_mean"],
                    "macro_f1_std": r["macro_f1_std"],
                    **{f"f1_{c}_mean": r[f"f1_{c}_mean"] for c in _CLASS_NAMES},
                    "balanced_accuracy_mean": r["balanced_accuracy_mean"],
                    "mcc_mean": r["mcc_mean"],
                })
                _save()
                print(f"  [{model:18s}] macro F1 = {r['macro_f1_mean']:.4f} ± {r['macro_f1_std']:.4f}  "
                      f"(LOF {r['f1_LOF_mean']:.3f} / NNE {r['f1_No net effect_mean']:.3f} / "
                      f"GOF {r['f1_GOF_mean']:.3f})", flush=True)
            except Exception as e:
                rows.append({
                    "group": gname, "model": model, "n": n,
                    "macro_f1_mean": float("nan"), "macro_f1_std": float("nan"),
                    "f1_LOF_mean": float("nan"), "f1_No net effect_mean": float("nan"),
                    "f1_GOF_mean": float("nan"),
                    "balanced_accuracy_mean": float("nan"), "mcc_mean": float("nan"),
                })
                _save()
                print(f"  [{model:18s}] FAILED: {e}", flush=True)

    out = pd.DataFrame(rows)
    _save()

    # ── δ/γ/ε always-LOF drop ───────────────────────────────────────────────
    print(f"\n{'=' * 70}\nδ/γ/ε always-LOF (majority) drop:")
    dge = out[out["group"] == "delta_gamma_epsilon"]
    maj = dge[dge["model"] == "majority_class"].iloc[0]
    print(f"  always-LOF macro F1 = {maj['macro_f1_mean']:.4f} "
          f"(LOF {maj['f1_LOF_mean']:.3f} / NNE {maj['f1_No net effect_mean']:.3f} / "
          f"GOF {maj['f1_GOF_mean']:.3f})")
    for model in args.models:
        row = dge[dge["model"] == model]
        if row.empty:
            continue
        row = row.iloc[0]
        drop = row["macro_f1_mean"] - maj["macro_f1_mean"]
        gof = row["f1_GOF_mean"]
        print(f"  [{model:18s}] macro {row['macro_f1_mean']:.4f}  "
              f"− always-LOF = {drop:+.4f}   (GOF-F1 {gof:.3f} vs always-LOF 0.000)")

    print(f"\nSaved {out_dir / 'classwise_f1_summary.csv'} and "
          f"{out_dir / 'class_distribution.csv'}")


if __name__ == "__main__":
    main()
