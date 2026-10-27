"""
Idea A probe: does adding AlphaMissense pathogenicity (1 feature) move direction macro-F1?

Run VIII concluded direction-of-effect is *orthogonal* to pathogenicity (conservation+ESM-2
moved 3-class +0.023 but binary GOF/LOF only +0.004). This re-tests it under the CURRENT
gene-aware 3-class protocol by comparing the 82-feature baseline against 82 + 1
(alphamissense_pathogenicity), on the full 760-variant dataset, RF + CatBoost, per-seed macro-F1.

Companion check (the "why δ/γ/ε wins" per-class story): per-class F1 (LOF / No net effect / GOF)
for α / β / δ/γ/ε on the 82-feature baseline, to confirm δ/γ/ε's edge is *clean labels*
(all three classes separable) rather than an artifact of LOF-dominance.

Usage:
    python scripts/run_alphamissense_probe.py --n-seeds 3 --n-trials 15
"""

import argparse
import json
import sys
import warnings

import numpy as np

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
from vep_nachr2.features.alphamissense import AlphamissenseExtractor
from vep_nachr2.training.cross_validation import nested_cross_validation
from vep_nachr2.config import RESULTS_DIR, DEFAULT_SEEDS, LABEL_NAMES

_EFFECT_MAP = {"LOF": 0, "No net effect": 1, "GOF": 2}

GROUPS = {
    "alpha": {f"CHRNA{i}" for i in range(1, 8)} | {"CHRNA9", "CHRNA10"},
    "beta": {"CHRNB1", "CHRNB2", "CHRNB3", "CHRNB4"},
    "delta_gamma_epsilon": {"CHRND", "CHRNE", "CHRNG"},
}
GROUP_LABELS = {
    "alpha": "α (517)",
    "beta": "β (128)",
    "delta_gamma_epsilon": "δ/γ/ε (115)",
}


def _extract_82(df):
    ref_seqs = load_all_reference_sequences("human")
    pdb = load_all_pdb_resources()
    X_base, _ = FeatureOrchestrator(verbose=False).extract(df, ref_seqs=ref_seqs, use_cache=True)
    gating = GatingExtractor().extract(df, ref_seqs, pdb)
    inter = AllostericInteractionExtractor().extract(df, ref_seqs, pdb)
    X = np.hstack([X_base, gating, inter])
    assert X.shape == (len(df), 82), X.shape
    return X


def _run_cv(X, y, df, model, n_trials, n_seeds):
    return nested_cross_validation(
        X=X, y=y, df=df, model_name=model,
        n_outer_folds=5, n_trials=n_trials,
        seeds=DEFAULT_SEEDS[:n_seeds], cv_mode="standard",
        verbose=False, n_classes=3, label_names=LABEL_NAMES,
    )


def _per_seed(cv, key):
    return [float(s["aggregate"][key]) for s in cv["per_seed"]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--n-trials", type=int, default=15)
    parser.add_argument("--models", nargs="+", default=["random_forest", "catboost"])
    args = parser.parse_args()

    df = load_mutation_data(data_file="final_mapped.xlsx")
    y = df["effect"].map(_EFFECT_MAP).values.astype(np.int64)
    ref_seqs = load_all_reference_sequences("human")
    pdb = load_all_pdb_resources()

    X82 = _extract_82(df)
    am = AlphamissenseExtractor().extract(df, ref_seqs=ref_seqs, pdb_resources=pdb)
    X83 = np.hstack([X82, am])
    print(f"AM feature shape: {am.shape}; coverage of non-fill values shown below (if verbose).")

    out_dir = RESULTS_DIR / "idea_a"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Part A: AlphaMissense probe (full data, 82 vs 83) ─────────────────
    print("=" * 70)
    print("PART A — AlphaMissense probe: 82 features vs 82+AM (83), full 760")
    print("=" * 70, flush=True)
    part_a = {}
    for label, X in [("82", X82), ("82+AM", X83)]:
        for model in args.models:
            cv = _run_cv(X, y, df, model, args.n_trials, args.n_seeds)
            f1s = _per_seed(cv, "macro_f1_mean")
            part_a[(label, model)] = f1s
            print(f"  [{label:5s} {model:15s}] macro F1 = {np.mean(f1s):.4f} ± {np.std(f1s):.4f}  "
                  f"per-seed {[round(v, 4) for v in f1s]}", flush=True)

    print("\n  Δ = (82+AM) − (82), per model:")
    for model in args.models:
        base = part_a[("82", model)]
        plus = part_a[("82+AM", model)]
        d = np.mean(plus) - np.mean(base)
        print(f"    [{model:15s}] {np.mean(base):.4f} -> {np.mean(plus):.4f}  Δ {d:+.4f}")

    with open(out_dir / "part_a_am_probe.json", "w") as f:
        json.dump({f"{k[0]}|{k[1]}": v for k, v in part_a.items()}, f, indent=2)

    # ── Part B: per-class F1 per group (82-feature baseline) ──────────────
    print("\n" + "=" * 70)
    print("PART B — per-class F1 (LOF / NNE / GOF) per group, 82 features")
    print("=" * 70, flush=True)
    part_b = {}
    for gname, genes in GROUPS.items():
        mask = df["subunit"].isin(genes).to_numpy()
        Xg, yg, dfg = X82[mask], y[mask], df[mask].reset_index(drop=True)
        for model in args.models:
            cv = _run_cv(Xg, yg, dfg, model, args.n_trials, args.n_seeds)
            lof = _per_seed(cv, "f1_LOF_mean")
            nne = _per_seed(cv, "f1_No net effect_mean")
            gof = _per_seed(cv, "f1_GOF_mean")
            macro = _per_seed(cv, "macro_f1_mean")
            part_b[(gname, model)] = {
                "macro": macro, "LOF": lof, "NNE": nne, "GOF": gof,
            }
            print(f"  [{GROUP_LABELS[gname]:10s} {model:15s}] "
                  f"macro {np.mean(macro):.4f} | LOF {np.mean(lof):.4f} | "
                  f"NNE {np.mean(nne):.4f} | GOF {np.mean(gof):.4f}", flush=True)

    with open(out_dir / "part_b_perclass_f1.json", "w") as f:
        json.dump({f"{g}|{m}": {k: v for k, v in d.items()}
                   for (g, m), d in part_b.items()}, f, indent=2)

    print(f"\nSaved -> {out_dir / 'part_a_am_probe.json'}, {out_dir / 'part_b_perclass_f1.json'}")


if __name__ == "__main__":
    main()
