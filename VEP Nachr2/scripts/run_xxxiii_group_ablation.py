"""
Run XXXIII: per-subunit-group feature ablation — why is δ/γ/ε the best performer?

Run XXXI found δ/γ/ε (CHRND/E/G) scores 0.69 vs α 0.63 / β 0.63, and Run XXXII proved
that edge is biological, not a gene-identity artifact. This run asks *which features* carry
each group's signal — the "why".

For each subunit group (α / β / δ/γ/ε), we drop one feature group at a time and re-run the
standard CV, comparing the macro-F1 drop against that group's own full-feature baseline.
A large drop ⇒ that feature group is load-bearing for that subunit class; a small drop ⇒ it
is redundant. Comparing the drop patterns across the three groups shows *how* the groups
differ in what they rely on.

Feature groups (82 total, column indices verified against the extractor output):
    physicochemical (24)  [0:24]     structural_core (7)  [50:57]
    substitution (3)      [24:27]    esm2 (2)             [57:59]
    positional (20)       [27:47]    esm_embeddings (16)  [59:75]
    conservation (3)      [47:50]    gating (3)           [75:78]
                                     allosteric_interaction (4) [78:82]
    (positional = position_normalized + 16 subunit_* + 3 species_*)

Protocol matches the gene-aware runs (XXVI onward): cv_mode="standard", 3-class,
RF + CatBoost. NOTE: 1 seed (vs 3 in XXXI/XXXII) to keep 60 model-CV runs tractable —
ablation *deltas* are the readout, and deltas < ~0.02 should be read as noise given the
single-seed variance (δ/γ/ε CatBoost σ ≈ 0.019). The per-group "full" baseline is re-run
here at the same 1 seed so every Δ is matched.

Usage:
    python scripts/run_xxxiii_group_ablation.py
    python scripts/run_xxxiii_group_ablation.py --models random_forest --n-seeds 1 --n-trials 10
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
from vep_nachr2.stats import per_seed_macro_f1

_EFFECT_MAP = {"LOF": 0, "No net effect": 1, "GOF": 2}

GROUPS = {
    "alpha": {f"CHRNA{i}" for i in range(1, 8)} | {"CHRNA9", "CHRNA10"},
    "beta": {"CHRNB1", "CHRNB2", "CHRNB3", "CHRNB4"},
    "delta_gamma_epsilon": {"CHRND", "CHRNE", "CHRNG"},
}

# Feature-group column slices, verified against the 82-name layout.
FEATURE_GROUPS = {
    "physicochemical": (0, 24),
    "substitution": (24, 27),
    "positional": (27, 47),
    "conservation": (47, 50),
    "structural_core": (50, 57),
    "esm2": (57, 59),
    "esm_embeddings": (59, 75),
    "gating": (75, 78),
    "allosteric_interaction": (78, 82),
}


def _run_cv(X, y, df, model, n_trials, n_seeds):
    cv = nested_cross_validation(
        X=X, y=y, df=df,
        model_name=model,
        n_outer_folds=5,
        n_trials=n_trials,
        seeds=DEFAULT_SEEDS[:n_seeds],
        cv_mode="standard",
        verbose=False,
        n_classes=3,
        label_names=LABEL_NAMES,
    )
    return cv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=15)
    parser.add_argument("--n-seeds", type=int, default=1)
    parser.add_argument("--models", nargs="+", default=["random_forest", "catboost"])
    parser.add_argument("--out-name", type=str, default="group_ablation",
                        help="subdir under results/ for this run's output")
    parser.add_argument("--groups", nargs="+", choices=sorted(GROUPS.keys()), default=None,
                        help="restrict to these subunit groups (default: all)")
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

    out_dir = RESULTS_DIR / args.out_name
    out_dir.mkdir(parents=True, exist_ok=True)

    drop_names = list(FEATURE_GROUPS.keys())
    groups = {k: v for k, v in GROUPS.items() if args.groups is None or k in args.groups}
    rows = []
    for name, genes in groups.items():
        mask = df["subunit"].isin(genes).to_numpy()
        pos = np.flatnonzero(mask)
        y_sub = y[pos]
        df_sub = df[mask].reset_index(drop=True)
        n = len(y_sub)

        # Conditions: "full" (no drop) + each single-group drop.
        conditions = [("full", None, 82)]
        for dname, (s, e) in FEATURE_GROUPS.items():
            drop_idx = list(range(s, e))
            X_drop = np.delete(X_full, drop_idx, axis=1)
            conditions.append((f"drop_{dname}", X_drop, 82 - (e - s)))

        for cond_label, X_cond, nf in conditions:
            X_sub = X_full[pos] if X_cond is None else X_cond[pos]
            print(f"{'=' * 70}\nGROUP={name}  {cond_label}  (n={n}, dim={nf})", flush=True)
            for model in args.models:
                cv = _run_cv(X_sub, y_sub, df_sub, model, args.n_trials, args.n_seeds)
                final = cv.get("final", {})
                rows.append({
                    "group": name,
                    "drop": cond_label,
                    "model": model,
                    "n": n,
                    "n_features": nf,
                    "macro_f1_mean": final.get("macro_f1_mean", float("nan")),
                    "macro_f1_std": final.get("macro_f1_std", float("nan")),
                    "mcc_mean": final.get("mcc_mean", float("nan")),
                    "balanced_accuracy_mean": final.get("balanced_accuracy_mean", float("nan")),
                })
                # Persist per-seed F1 for paired "full vs drop" significance tests.
                per_seed_f1 = per_seed_macro_f1(cv)
                rec_path = out_dir / f"per_seed_{name}_{cond_label}_{model}.json"
                with open(rec_path, "w") as f:
                    json.dump({
                        "group": name, "drop": cond_label, "model": model,
                        "seeds": DEFAULT_SEEDS[:args.n_seeds],
                        "per_seed_f1": per_seed_f1,
                    }, f, indent=2)
                print(f"  [{model:15s}] macro F1 = {final.get('macro_f1_mean', float('nan')):.4f} "
                      f"± {final.get('macro_f1_std', float('nan')):.4f}", flush=True)

    out = pd.DataFrame(rows)
    suffix = "" if args.groups is None else "_" + "+".join(sorted(args.groups))
    out.to_csv(out_dir / f"group_ablation_summary{suffix}.csv", index=False)

    # Per-group delta table: full minus each drop.
    print(f"\n{'=' * 70}\nDelta macro F1 (full - dropped), per group:")
    for name in groups:
        sub = out[out["group"] == name]
        full = {m: sub[(sub["drop"] == "full") & (sub["model"] == m)]["macro_f1_mean"].iloc[0]
                for m in args.models}
        print(f"\n  {name} (full: " + " / ".join(
            f"{m} {full[m]:.4f}" for m in args.models) + ")")
        for dname in drop_names:
            drow = sub[sub["drop"] == f"drop_{dname}"]
            if drow.empty:
                continue
            line = f"    drop_{dname:24s}"
            for m in args.models:
                v = drow[drow["model"] == m]["macro_f1_mean"].iloc[0]
                line += f"  [{m:14s}] {v:.4f} (Δ {full[m] - v:+.4f})"
            print(line)

    print(f"\nSaved {out_dir / f'group_ablation_summary{suffix}.csv'}")


if __name__ == "__main__":
    main()
