"""
Ablation significance for Run XXXIII (read-only; runs after the multi-seed re-run).

Reads the per-condition per-seed F1 files written by run_xxxiii_group_ablation.py
(``results/<out-name>/per_seed_{group}_{drop}_{model}.json``) and, for each group
× model, tests each single-feature-group drop against that group's own full
baseline with a one-tailed paired t-test (matched by seed):

    H0: full <= drop    H1: full > drop   ("dropping this group hurts")

A significant p means the dropped feature group is load-bearing. A negative Δ
(full < drop) means dropping it *helps* (the group is redundant/at worst noise).

Usage:
    python scripts/run_xxxiii_ablation_significance.py                 # group_ablation
    python scripts/run_xxxiii_ablation_significance.py --dir group_ablation_3seed
"""

import argparse
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

from vep_nachr2.config import RESULTS_DIR
from vep_nachr2.stats import paired_greater, fmt_p

DROP_ORDER = [
    "physicochemical", "substitution", "positional", "conservation",
    "structural_core", "esm2", "esm_embeddings", "gating", "allosteric_interaction",
]


def _load_all(d: Path) -> dict:
    data = {}
    for p in sorted(d.glob("per_seed_*.json")):
        with open(p) as f:
            rec = json.load(f)
        data[(rec["group"], rec["drop"], rec["model"])] = rec["per_seed_f1"]
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="group_ablation")
    args = parser.parse_args()

    d = RESULTS_DIR / args.dir
    data = _load_all(d)
    if not data:
        print(f"No per_seed_*.json found in {d}")
        return

    groups = ["alpha", "beta", "delta_gamma_epsilon"]
    models = ["random_forest", "catboost"]

    for g in groups:
        for m in models:
            full = data.get((g, "full", m))
            if full is None:
                continue
            full_mean = sum(full) / len(full)
            print(f"\n=== {g} / {m}  (full = {full_mean:.4f}, n_seeds={len(full)}) ===")
            print(f"  {'drop':24s} {'drop F1':>8s} {'Δ (full-drop)':>13s} {'t':>7s} {'p (full>drop)':>14s}")
            for dn in DROP_ORDER:
                drop = data.get((g, f"drop_{dn}", m))
                if drop is None:
                    continue
                drop_mean = sum(drop) / len(drop)
                dlt = full_mean - drop_mean
                t, p = paired_greater(full, drop)
                flag = "  ***" if p < 0.05 else ""
                print(f"  {dn:24s} {drop_mean:8.4f} {dlt:+13.4f} {t:7.2f} {fmt_p(p):>14s}{flag}")


if __name__ == "__main__":
    main()
