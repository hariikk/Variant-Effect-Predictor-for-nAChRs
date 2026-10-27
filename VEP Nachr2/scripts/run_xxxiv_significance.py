"""
Run XXXIV: statistical significance of the main run and the α/β/δγε comparison.

The unit of analysis is the **per-seed macro-F1** (N seeds per condition). This run
re-computes full / α / β / δγε at N seeds (gene-aware cv_mode="standard", 3-class,
RF + CatBoost, 15 trials) and answers two questions:

  1. **Is each group above chance?** one-sample t-test vs 1/3 (3-class chance).
  2. **Is δ/γ/ε genuinely better than α / β / full?** one-tailed Welch t-tests
     (unpaired — different data subsets) plus paired (matched by seed) as a check.

Run the compute step per group (parallelised by launching one process per group,
each handling both models), then summarise:

    python scripts/run_xxxiv_significance.py --group full     --n-seeds 10
    python scripts/run_xxxiv_significance.py --group alpha    --n-seeds 10
    python scripts/run_xxxiv_significance.py --group beta     --n-seeds 10
    python scripts/run_xxxiv_significance.py --group delta_gamma_epsilon --n-seeds 10
    python scripts/run_xxxiv_significance.py --summarize

Per-seed F1 values are persisted to results/significance/per_seed_{group}_{model}.json
so the summarise step reads exactly what the compute step produced (no re-derivation).
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
from vep_nachr2.config import (
    RESULTS_DIR, DEFAULT_SEEDS, SPECIES_TRANSFER_SEEDS, LABEL_NAMES,
)
from vep_nachr2.stats import (
    per_seed_macro_f1, one_sample_greater, welch_greater, paired_greater, fmt_p,
)

_EFFECT_MAP = {"LOF": 0, "No net effect": 1, "GOF": 2}

GROUPS = {
    "full": None,  # all 16 genes
    "alpha": {f"CHRNA{i}" for i in range(1, 8)} | {"CHRNA9", "CHRNA10"},
    "beta": {"CHRNB1", "CHRNB2", "CHRNB3", "CHRNB4"},
    "delta_gamma_epsilon": {"CHRND", "CHRNE", "CHRNG"},
}

GROUP_LABELS = {
    "full": "Full (760)",
    "alpha": "α (CHRNA1-7,9,10)",
    "beta": "β (CHRNB1-4)",
    "delta_gamma_epsilon": "δ/γ/ε (CHRND,E,G)",
}


def _extract_82(df):
    """Build the 82-feature matrix (75 base + 3 gating + 4 allosteric)."""
    ref_seqs = load_all_reference_sequences("human")
    pdb_resources = load_all_pdb_resources()
    base_orch = FeatureOrchestrator(verbose=False)
    X_base, _ = base_orch.extract(df, ref_seqs=ref_seqs, use_cache=True)
    gating = GatingExtractor().extract(df, ref_seqs, pdb_resources)
    inter = AllostericInteractionExtractor().extract(df, ref_seqs, pdb_resources)
    X = np.hstack([X_base, gating, inter])
    assert X.shape == (len(df), 82), X.shape
    return X


def _run_group(group_name, models, n_seeds, n_trials):
    df = load_mutation_data(data_file="final_mapped.xlsx")
    X_full = _extract_82(df)

    genes = GROUPS[group_name]
    if genes is None:
        X, y, df_sub = X_full, df["effect"].map(_EFFECT_MAP).values.astype(np.int64), df
    else:
        mask = df["subunit"].isin(genes).to_numpy()
        X, y, df_sub = X_full[mask], df["effect"].map(_EFFECT_MAP).values.astype(np.int64)[mask], df[mask].reset_index(drop=True)

    out_dir = RESULTS_DIR / "significance"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== GROUP={group_name}  n={len(y)}  seeds={n_seeds}  models={models} ===", flush=True)
    for model in models:
        cv = nested_cross_validation(
            X=X, y=y, df=df_sub, model_name=model,
            n_outer_folds=5, n_trials=n_trials,
            seeds=SPECIES_TRANSFER_SEEDS[:n_seeds],
            cv_mode="standard", verbose=False, n_classes=3, label_names=LABEL_NAMES,
        )
        f1s = per_seed_macro_f1(cv)
        payload = {
            "group": group_name,
            "model": model,
            "n": int(len(y)),
            "seeds": SPECIES_TRANSFER_SEEDS[:n_seeds],
            "per_seed_f1": f1s,
            "macro_f1_mean": float(np.mean(f1s)),
            "macro_f1_std": float(np.std(f1s)),
        }
        path = out_dir / f"per_seed_{group_name}_{model}.json"
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"  [{model:15s}] per-seed F1 = {[round(v, 4) for v in f1s]}", flush=True)
        print(f"  [{model:15s}] mean {np.mean(f1s):.4f} ± {np.std(f1s):.4f}  -> {path.name}", flush=True)


def _summarize():
    out_dir = RESULTS_DIR / "significance"
    files = sorted(out_dir.glob("per_seed_*.json"))
    if not files:
        print("No per_seed_*.json files found — run the compute step first.")
        return

    data = {}
    for p in files:
        with open(p) as f:
            rec = json.load(f)
        data[(rec["group"], rec["model"])] = rec

    print("=" * 78)
    print("1) One-sample t-test vs chance (1/3): is each group above chance?")
    print("=" * 78)
    print(f"{'group':20s} {'model':14s} {'mean F1':>8s} {'±std':>7s} {'t':>7s} {'p (one-tailed)':>15s}")
    for group in ["full", "alpha", "beta", "delta_gamma_epsilon"]:
        for model in ["random_forest", "catboost"]:
            rec = data.get((group, model))
            if rec is None:
                continue
            f1s = rec["per_seed_f1"]
            t, p = one_sample_greater(f1s, null=1 / 3)
            print(f"{GROUP_LABELS[group]:20s} {model:14s} {np.mean(f1s):8.4f} "
                  f"{np.std(f1s):7.4f} {t:7.2f} {fmt_p(p):>15s}")

    print("\n" + "=" * 78)
    print("2) One-tailed Welch t-test: is δ/γ/ε > {α, β, full}? (unpaired)")
    print("=" * 78)
    models = ["random_forest", "catboost"]
    comparators = ["alpha", "beta", "full"]
    print(f"{'comparison':28s} {'model':14s} {'Δ mean':>8s} {'t':>7s} {'p (one-tailed)':>15s}")
    for model in models:
        dge = data[("delta_gamma_epsilon", model)]["per_seed_f1"]
        for c in comparators:
            other = data[(c, model)]["per_seed_f1"]
            t, p = welch_greater(dge, other)
            d = np.mean(dge) - np.mean(other)
            print(f"δγε > {c:20s} {model:14s} {d:+8.4f} {t:7.2f} {fmt_p(p):>15s}")

    print("\n" + "=" * 78)
    print("3) Paired t-test (matched by seed) — δ/γ/ε > {α, β, full}")
    print("=" * 78)
    for model in models:
        dge = data[("delta_gamma_epsilon", model)]["per_seed_f1"]
        for c in comparators:
            other = data[(c, model)]["per_seed_f1"]
            t, p = paired_greater(dge, other)
            d = np.mean(dge) - np.mean(other)
            print(f"δγε > {c:20s} {model:14s} {d:+8.4f} {t:7.2f} {fmt_p(p):>15s}")

    print("\n" + "=" * 78)
    print("4) Full pairwise grid (one-tailed Welch, α vs β vs full)")
    print("=" * 78)
    pairs = [("alpha", "beta"), ("alpha", "full"), ("beta", "full")]
    for model in models:
        for a, b in pairs:
            va = data[(a, model)]["per_seed_f1"]
            vb = data[(b, model)]["per_seed_f1"]
            t, p = welch_greater(va, vb)
            d = np.mean(va) - np.mean(vb)
            print(f"{a} > {b:14s} {model:14s} {d:+8.4f} {t:7.2f} {fmt_p(p):>15s}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", choices=list(GROUPS.keys()), default=None,
                        help="compute per-seed F1 for this group (None = all)")
    parser.add_argument("--models", nargs="+", default=["random_forest", "catboost"])
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--n-trials", type=int, default=15)
    parser.add_argument("--summarize", action="store_true",
                        help="read saved per_seed_*.json and print p-values")
    args = parser.parse_args()

    if args.summarize:
        _summarize()
        return

    groups = [args.group] if args.group else list(GROUPS.keys())
    for g in groups:
        _run_group(g, args.models, args.n_seeds, args.n_trials)


if __name__ == "__main__":
    main()
