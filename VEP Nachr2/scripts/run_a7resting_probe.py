"""
α7 resting-state structure probe: 7EKT vs the new complete homomeric α7 resting model.

The new structure (homomeric_a7_resting_complete.pdb, spliced from 7KOO cryo-EM +
7RPM NMR, all ligands/ions removed) is a *complete* resting pentamer (residues
23-502, all 5 chains) versus the current 7EKT (antagonist-bound, 3.20 Å, missing a
few residues). The colleague's sodium-channel result (more complete structure → better
F1) motivates testing whether swapping CHRNA7's structural source moves direction F1.

Comparison is gene-aware 3-class CV (cv_mode="standard"), 82-feature layout
(75 base + gating 3 + allosteric 4). The ONLY difference between the two arms is the
14 PDB-derived features for CHRNA7 (structural_core 7 + gating 3 + allosteric 4);
all other genes and the 68 non-structural features are identical.

Usage:
    python scripts/run_a7resting_probe.py --n-seeds 5 --n-trials 15
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

import vep_nachr2.config as cfg
from vep_nachr2.config import PDB_MAPPING, RESULTS_DIR, DEFAULT_SEEDS, LABEL_NAMES
from vep_nachr2.data.loader import load_mutation_data
from vep_nachr2.data.reference import load_all_reference_sequences
from vep_nachr2.features.orchestrator import FeatureOrchestrator
from vep_nachr2.features.structural import StructuralExtractor, load_all_pdb_resources, _load_pdb_resources
from vep_nachr2.features.gating import GatingExtractor
from vep_nachr2.features.allosteric_interaction import AllostericInteractionExtractor
from vep_nachr2.training.cross_validation import nested_cross_validation

_EFFECT_MAP = {"LOF": 0, "No net effect": 1, "GOF": 2}


def _extract_structural(df, ref_seqs, pdb_resources):
    """The 14 PDB-derived features: structural_core 7 + gating 3 + allosteric 4."""
    struct = StructuralExtractor().extract(df, ref_seqs, pdb_resources)
    gating = GatingExtractor().extract(df, ref_seqs, pdb_resources)
    inter = AllostericInteractionExtractor().extract(df, ref_seqs, pdb_resources)
    return struct, gating, inter


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
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--n-trials", type=int, default=15)
    parser.add_argument("--models", nargs="+", default=["random_forest", "catboost"])
    args = parser.parse_args()

    df = load_mutation_data(data_file="final_mapped.xlsx")
    y = df["effect"].map(_EFFECT_MAP).values.astype(np.int64)
    ref_seqs = load_all_reference_sequences("human")

    # Base 75 features (cached) — structural_core (7EKT) sits at [s0:s1]
    orch = FeatureOrchestrator(verbose=False)
    X75, names = orch.extract(df, ref_seqs=ref_seqs, use_cache=True)
    s0, s1 = orch.get_feature_group_indices()["structural_core"]
    assert X75.shape == (len(df), 75), X75.shape

    base_pdb = load_all_pdb_resources()

    # ── 7EKT arm (current) ────────────────────────────────────────────────
    struct7, gat7, int7 = _extract_structural(df, ref_seqs, base_pdb)
    assert np.allclose(X75[:, s0:s1], struct7), "cached structural_core != fresh 7EKT"
    X_7ekt = np.hstack([X75, gat7, int7])
    assert X_7ekt.shape == (len(df), 82), X_7ekt.shape

    # ── a7rest arm (new complete resting structure) ───────────────────────
    orig = PDB_MAPPING["CHRNA7"]
    PDB_MAPPING["CHRNA7"] = {"pdb_id": "a7rest", "chain": "A", "source": "experimental"}
    try:
        a7rest_res = _load_pdb_resources("a7rest")
        pdb_new = {k: v for k, v in base_pdb.items() if k != "7EKT"}
        pdb_new["a7rest"] = a7rest_res
        struct_new, gat_new, int_new = _extract_structural(df, ref_seqs, pdb_new)
    finally:
        PDB_MAPPING["CHRNA7"] = orig

    X75_new = X75.copy()
    X75_new[:, s0:s1] = struct_new
    X_new = np.hstack([X75_new, gat_new, int_new])
    assert X_new.shape == (len(df), 82), X_new.shape

    # ── Diagnostic: how much did CHRNA7's structural features change? ─────
    mask = (df["subunit"] == "CHRNA7").to_numpy()
    print("=" * 72)
    print("CHRNA7 structural-feature change (7EKT -> a7rest), n =", int(mask.sum()))
    print("=" * 72, flush=True)
    feat_groups = {
        "structural_core": (X75[:, s0:s1], struct_new),
        "gating": (gat7, gat_new),
        "allosteric": (int7, int_new),
    }
    feat_names = (
        StructuralExtractor.feature_names
        + GatingExtractor.feature_names
        + AllostericInteractionExtractor.feature_names
    )
    old_cat = np.concatenate([feat_groups["structural_core"][0], feat_groups["gating"][0],
                              feat_groups["allosteric"][0]], axis=1)
    new_cat = np.concatenate([feat_groups["structural_core"][1], feat_groups["gating"][1],
                              feat_groups["allosteric"][1]], axis=1)
    old_m = old_cat[mask]
    new_m = new_cat[mask]
    for j, name in enumerate(feat_names):
        o, n = old_m[:, j], new_m[:, j]
        corr = np.corrcoef(o, n)[0, 1] if o.std() > 0 and n.std() > 0 else np.nan
        mad = np.mean(np.abs(o - n))
        print(f"  {name:22s} corr={corr:+.3f}  mean|Δ|={mad:.4f}", flush=True)

    # ── CV comparison ─────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("CV comparison: 7EKT vs a7rest (gene-aware standard, full 760)")
    print("=" * 72, flush=True)

    out_dir = RESULTS_DIR / "a7resting"
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for label, X in [("7EKT", X_7ekt), ("a7rest", X_new)]:
        for model in args.models:
            cv = _run_cv(X, y, df, model, args.n_trials, args.n_seeds)
            macro = _per_seed(cv, "macro_f1_mean")
            lof = _per_seed(cv, "f1_LOF_mean")
            nne = _per_seed(cv, "f1_No net effect_mean")
            gof = _per_seed(cv, "f1_GOF_mean")
            results[f"{label}|{model}"] = {
                "macro": macro, "LOF": lof, "NNE": nne, "GOF": gof,
            }
            print(f"  [{label:7s} {model:15s}] macro {np.mean(macro):.4f} ± {np.std(macro):.4f} | "
                  f"LOF {np.mean(lof):.4f} | NNE {np.mean(nne):.4f} | GOF {np.mean(gof):.4f}",
                  flush=True)

    print("\n  Δ = (a7rest) − (7EKT), per model (macro F1):")
    for model in args.models:
        base = results[f"7EKT|{model}"]["macro"]
        new = results[f"a7rest|{model}"]["macro"]
        print(f"    [{model:15s}] {np.mean(base):.4f} -> {np.mean(new):.4f}  Δ {np.mean(new)-np.mean(base):+.4f}")

    with open(out_dir / "comparison.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {out_dir / 'comparison.json'}")


if __name__ == "__main__":
    main()
