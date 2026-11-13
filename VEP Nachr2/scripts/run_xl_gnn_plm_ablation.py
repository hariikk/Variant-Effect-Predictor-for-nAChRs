"""
Run XL: pretrained GNN / protein-transformer / cheminformatics feature arms.

Adds three new pretrained feature groups one at a time and in combination, then
does a leave-one-group-out ablation, all under the gene-aware protocol used from
Run XXVI onward (cv_mode="standard", 3-class, RF + CatBoost, 10 seeds x 15
Optuna trials).

Feature groups added on top of the 82-feature anchor:
    gnn_gvp     (16)  ESM-IF1 GVP-GNN per-residue node embedding -> PCA
    prot_bert   (16)  ProtBERT (ProtTrans) per-residue embedding -> PCA
    mordred_aa  (16)  RDKit/Mordred amino-acid descriptor delta -> PCA

Arms:
    baseline           8 base groups (82)
    add_gvp            82 + 16
    add_prot_bert      82 + 16
    add_mordred        82 + 16
    add_all            82 + 48
    all_minus_gvp      82 + 32   (leave-one-out)
    all_minus_prot_bert
    all_minus_mordred

Readout: macro-F1 delta vs the matched baseline (same seeds, same model), plus
per-seed F1 files for paired significance tests.

Usage:
    python scripts/run_xl_gnn_plm_ablation.py --n-seeds 10 --n-trials 15
    python scripts/run_xl_gnn_plm_ablation.py --n-seeds 1 --n-trials 2 --arms baseline add_gvp
"""

import argparse
import json
import sys
import warnings

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

from vep_nachr2.data.loader import load_mutation_data
from vep_nachr2.data.reference import load_all_reference_sequences
from vep_nachr2.features.orchestrator import FeatureOrchestrator
from vep_nachr2.features.structural import load_all_pdb_resources
from vep_nachr2.features.gating import GatingExtractor
from vep_nachr2.features.allosteric_interaction import AllostericInteractionExtractor
from vep_nachr2.features.gnn_gvp import GVPEmbeddingExtractor
from vep_nachr2.features.prot_bert import ProtBertExtractor
from vep_nachr2.features.mordred_aa import MordredAADescriptorExtractor
from vep_nachr2.training.cross_validation import nested_cross_validation
from vep_nachr2.config import RESULTS_DIR, DEFAULT_SEEDS, LABEL_NAMES
from vep_nachr2.stats import per_seed_macro_f1

_EFFECT_MAP = {"LOF": 0, "No net effect": 1, "GOF": 2}
NEW_GROUPS = ["gnn_gvp", "prot_bert", "mordred_aa"]

ARMS = {
    "baseline": (),
    "add_gvp": ("gnn_gvp",),
    "add_prot_bert": ("prot_bert",),
    "add_mordred": ("mordred_aa",),
    "add_all": tuple(NEW_GROUPS),
    "all_minus_gvp": ("prot_bert", "mordred_aa"),
    "all_minus_prot_bert": ("gnn_gvp", "mordred_aa"),
    "all_minus_mordred": ("gnn_gvp", "prot_bert"),
}


def _run_cv(X, y, df, model, n_trials, n_seeds):
    return nested_cross_validation(
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=15)
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--models", nargs="+", default=["random_forest", "catboost"])
    parser.add_argument("--arms", nargs="+", choices=list(ARMS), default=list(ARMS))
    parser.add_argument("--out-name", type=str, default="gnn_plm_ablation")
    parser.add_argument("--extract-only", action="store_true",
                        help="build the feature-block cache and exit")
    parser.add_argument("--force-extract", action="store_true",
                        help="ignore cached feature blocks and recompute")
    args = parser.parse_args()

    df = load_mutation_data(data_file="final_mapped.xlsx")
    y = df["effect"].map(_EFFECT_MAP).values.astype(np.int64)
    ref_seqs = load_all_reference_sequences("human")
    pdb_resources = load_all_pdb_resources()

    # ---- 82-feature anchor (identical to Run XXXIII's "full") ----
    base_orch = FeatureOrchestrator(verbose=False)
    X_base, _ = base_orch.extract(df, ref_seqs=ref_seqs, use_cache=True)
    X_anchor = np.hstack([
        X_base,
        GatingExtractor().extract(df, ref_seqs, pdb_resources),
        AllostericInteractionExtractor().extract(df, ref_seqs, pdb_resources),
    ])
    assert X_anchor.shape == (len(df), 82), X_anchor.shape

    out_dir = RESULTS_DIR / args.out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    block_dir = RESULTS_DIR / "gnn_plm_ablation" / "feature_blocks"
    block_dir.mkdir(parents=True, exist_ok=True)

    # ---- new feature groups (extracted once, cached to disk for parallel jobs) ----
    blocks = {}
    extractors = {
        "gnn_gvp": GVPEmbeddingExtractor(),
        "prot_bert": ProtBertExtractor(),
        "mordred_aa": MordredAADescriptorExtractor(),
    }
    coverage = {}
    for name in NEW_GROUPS:
        cache_file = block_dir / f"{name}_{len(df)}.npy"
        if cache_file.exists() and not args.force_extract:
            X_new = np.load(cache_file)
            print(f"[features] {name}: loaded cached block {X_new.shape}", flush=True)
        else:
            print(f"[features] extracting {name} ...", flush=True)
            X_new = extractors[name].extract(df, ref_seqs, pdb_resources)
            np.save(cache_file, X_new)
        assert X_new.shape == (len(df), 16), (name, X_new.shape)
        blocks[name] = X_new
        nonzero = float(np.mean(np.abs(X_new).sum(axis=1) > 0))
        coverage[name] = nonzero
        print(f"[features] {name}: shape={X_new.shape} "
              f"non-zero-row coverage={nonzero:.1%}", flush=True)

    with open(block_dir / "feature_coverage.json", "w") as f:
        json.dump(coverage, f, indent=2)

    if args.extract_only:
        print("[features] cache built; exiting (--extract-only)")
        return

    rows = []
    for arm in args.arms:
        parts = [X_anchor] + [blocks[g] for g in ARMS[arm]]
        X = np.hstack(parts)
        nf = X.shape[1]
        print(f"\n{'=' * 72}\nARM={arm}  dim={nf}", flush=True)
        for model in args.models:
            cv = _run_cv(X, y, df, model, args.n_trials, args.n_seeds)
            final = cv.get("final", {})
            rows.append({
                "arm": arm,
                "model": model,
                "n_features": nf,
                "n_samples": len(df),
                "macro_f1_mean": final.get("macro_f1_mean", float("nan")),
                "macro_f1_std": final.get("macro_f1_std", float("nan")),
                "mcc_mean": final.get("mcc_mean", float("nan")),
                "balanced_accuracy_mean": final.get("balanced_accuracy_mean", float("nan")),
            })
            per_seed = per_seed_macro_f1(cv)
            with open(out_dir / f"per_seed_{arm}_{model}.json", "w") as f:
                json.dump({
                    "arm": arm, "model": model,
                    "seeds": DEFAULT_SEEDS[:args.n_seeds],
                    "per_seed_f1": per_seed,
                }, f, indent=2)
            print(f"  [{model:15s}] macro F1 = {final.get('macro_f1_mean', float('nan')):.4f} "
                  f"± {final.get('macro_f1_std', float('nan')):.4f}", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "gnn_plm_ablation_summary.csv", index=False)
    with open(out_dir / "feature_coverage.json", "w") as f:
        json.dump(coverage, f, indent=2)

    # ---- delta table vs matched baseline ----
    print(f"\n{'=' * 72}\nDelta macro F1 vs baseline (same seeds/models):")
    base = out[out["arm"] == "baseline"].set_index("model")["macro_f1_mean"]
    for arm in args.arms:
        if arm == "baseline":
            continue
        sub = out[out["arm"] == arm].set_index("model")
        line = f"  {arm:22s}"
        for m in args.models:
            if m in sub.index and m in base.index:
                v = sub.loc[m, "macro_f1_mean"]
                line += f"  [{m:14s}] {v:.4f} (Δ {v - base[m]:+.4f})"
        print(line)

    print(f"\nSaved {out_dir / 'gnn_plm_ablation_summary.csv'}")


if __name__ == "__main__":
    main()
