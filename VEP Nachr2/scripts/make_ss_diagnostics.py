"""
Secondary-structure (DSSP) diagnostics: where are the helices/sheets, and do
variants cluster relative to them?

Reuses the existing PDB pipeline (alignment maps + DSSP dicts) to assign every
canonical position a secondary-structure state (helix H/G/I, sheet E/B, coil
otherwise), then:
  - prints per-gene SS coverage + helix-run topology (TM1..TM4 etc.),
  - saves plots/ss_by_gene.json (helix/sheet runs per gene),
  - plots/plots/topology_map.png  (16 genes × position, coloured by SS),
  - plots/variants_ss_{alpha,beta,dge}.png (variants over SS-shaded background).

New files only — does not touch the existing spread-plots.
"""

import json
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

from vep_nachr2.config import PDB_MAPPING, RESULTS_DIR
from vep_nachr2.data.loader import load_mutation_data
from vep_nachr2.data.reference import load_all_reference_sequences
from vep_nachr2.features.structural import (
    load_all_pdb_resources,
    _build_all_alignment_maps,
)

PLOT_DIR = Path(__file__).resolve().parent.parent / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

GENE_ORDER = [
    "CHRNA1", "CHRNA2", "CHRNA3", "CHRNA4", "CHRNA5", "CHRNA6", "CHRNA7",
    "CHRNA9", "CHRNA10",
    "CHRNB1", "CHRNB2", "CHRNB3", "CHRNB4",
    "CHRND", "CHRNE", "CHRNG",
]
EFFECT_VALUE = {"LOF": -1, "No net effect": 0, "GOF": +1}
EFFECT_COLOR = {"LOF": "#377eb8", "No net effect": "#999999", "GOF": "#e41a1c"}
SS_COLOR = {"H": "#f6b26b", "E": "#9fc5e8", "C": "#ffffff", None: "#f0f0f0"}

_SS_HELIX = frozenset("HGI")
_SS_SHEET = frozenset("EB")


def ss_state(code):
    if code in _SS_HELIX:
        return "H"
    if code in _SS_SHEET:
        return "E"
    return "C"


def extract_ss_by_gene():
    """Return {gene: {pos: 'H'|'E'|'C'|None}} for every canonical position."""
    ref_seqs = load_all_reference_sequences("human")
    pdb = load_all_pdb_resources()
    maps = _build_all_alignment_maps(ref_seqs, pdb)

    out = {}
    for gene in GENE_ORDER:
        info = PDB_MAPPING[gene]
        resources = pdb.get(info["pdb_id"])
        dssp = resources.get("dssp_dict") if resources else None
        chain = info["chain"]
        amap = maps.get(gene, {})
        L = len(ref_seqs[gene])
        arr = {}
        for pos in range(1, L + 1):
            rid = amap.get(pos)
            if rid is None or dssp is None:
                arr[pos] = None
                continue
            key = (chain, rid)
            arr[pos] = ss_state(dssp[key][1]) if key in dssp else None
        out[gene] = arr
    return out, ref_seqs


def runs_from_array(arr):
    """Collapse a {pos: state} dict into contiguous runs [(start, end, state), ...]."""
    runs = []
    prev = None
    start = None
    for pos in sorted(arr):
        st = arr[pos]
        if st != prev:
            if prev is not None and start is not None:
                runs.append((start, pos - 1, prev))
            start = pos
            prev = st
    if prev is not None and start is not None:
        runs.append((start, sorted(arr)[-1], prev))
    return runs


def helix_runs(arr, min_len=5):
    """Contiguous helix runs (for topology reporting)."""
    runs = []
    cur = None
    for pos in sorted(arr):
        if arr[pos] == "H":
            cur = pos if cur is None else cur
        else:
            if cur is not None and pos - cur >= min_len:
                runs.append((cur, pos - 1))
            cur = None
    if cur is not None and sorted(arr)[-1] - cur + 1 >= min_len:
        runs.append((cur, sorted(arr)[-1]))
    return runs


def main():
    ss, ref_seqs = extract_ss_by_gene()

    # ── report ──────────────────────────────────────────────────────────────
    print(f"{'gene':10s} {'len':>4s} {'mapped':>6s} {'H%':>5s} {'E%':>5s} {'C%':>5s}   helix runs")
    ss_json = {}
    for gene in GENE_ORDER:
        arr = ss[gene]
        L = len(ref_seqs[gene])
        states = [s for s in arr.values() if s is not None]
        mapped = len(states)
        h = sum(1 for s in states if s == "H")
        e = sum(1 for s in states if s == "E")
        c = sum(1 for s in states if s == "C")
        hr = helix_runs(arr)
        hr_s = ", ".join(f"{a}-{b}" for a, b in hr)
        print(f"{gene:10s} {L:4d} {mapped:6d} {100*h/mapped:4.0f}% {100*e/mapped:4.0f}% {100*c/mapped:4.0f}%   {hr_s}")
        # save compact runs (H and E only; everything else = coil/unmapped)
        runs = [(a, b, s) for a, b, s in runs_from_array(arr) if s in ("H", "E")]
        ss_json[gene] = {"length": L, "runs": runs}

    (PLOT_DIR / "ss_by_gene.json").write_text(json.dumps(ss_json, indent=1), encoding="utf-8")

    df = load_mutation_data(data_file="final_mapped.xlsx")
    rng = np.random.default_rng(7)

    # ── topology map (16 genes × position, coloured ribbon) ─────────────────
    maxlen = max(len(ref_seqs[g]) for g in GENE_ORDER) + 10
    n = len(GENE_ORDER)
    fig, ax = plt.subplots(figsize=(13, 0.42 * n + 1.2))
    for i, gene in enumerate(GENE_ORDER):
        arr = ss[gene]
        for pos in sorted(arr):
            st = arr[pos]
            ax.add_patch(plt.Rectangle((pos - 0.5, i - 0.4), 1, 0.8,
                                       facecolor=SS_COLOR[st], edgecolor="none"))
    ax.set_yticks(range(n))
    ax.set_yticklabels(GENE_ORDER, fontsize=9, fontfamily="monospace")
    ax.set_xlim(0, maxlen)
    ax.set_xlabel("canonical position")
    ax.set_title("Secondary structure per gene — helix (amber) / sheet (blue) / coil (white) / unmapped (grey)")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "topology_map.png", dpi=150)
    plt.close(fig)
    print(f"\nwrote topology_map.png")

    # ── variants over SS background, faceted by family ─────────────────────
    families = {
        "alpha": [g for g in GENE_ORDER if g.startswith("CHRNA")],
        "beta": [g for g in GENE_ORDER if g.startswith("CHRNB")],
        "dge": ["CHRND", "CHRNE", "CHRNG"],
    }
    for fam, genes in families.items():
        sub = df[df["subunit"].isin(genes)]
        nrow = len(genes)
        fig, axes = plt.subplots(nrow, 1, figsize=(13, 2.0 * nrow), sharex=True,
                                 squeeze=False)
        axes = axes[:, 0]
        for j, gene in enumerate(genes):
            ax = axes[j]
            arr = ss[gene]
            # SS background bands
            for a, b, st in runs_from_array(arr):
                if st in ("H", "E"):
                    ax.axvspan(a - 0.5, b + 0.5, color=SS_COLOR[st], alpha=0.35, zorder=0)
            # variants
            gdf = sub[sub["subunit"] == gene]
            for eff in ["LOF", "No net effect", "GOF"]:
                m = gdf[gdf["effect"] == eff]
                if len(m) == 0:
                    continue
                y = EFFECT_VALUE[eff] * np.ones(len(m))
                band = 0.18 if eff != "No net effect" else 0.12
                x = m["position"].to_numpy() + rng.uniform(-1, 1, len(m))
                ax.scatter(x, y + rng.uniform(-band, band, len(m)), s=10, alpha=0.6,
                           linewidths=0, color=EFFECT_COLOR[eff], zorder=2)
            ax.axhline(0, color="#888", lw=0.7, ls="--", zorder=1)
            ax.set_yticks([-1, 0, 1])
            ax.set_yticklabels(["LOF", "NNE", "GOF"], fontsize=8)
            ax.set_ylim(-1.45, 1.45)
            ax.set_ylabel(gene, rotation=0, labelpad=30, fontsize=9, fontfamily="monospace")
            ax.set_xlim(0, maxlen)
            ax.grid(axis="x", color="#eee", lw=0.5)
        axes[-1].set_xlabel("canonical position")
        fig.suptitle(f"{fam} family — variants over secondary structure (amber=helix, blue=sheet)",
                     fontsize=11, y=1.0)
        fig.tight_layout()
        out = PLOT_DIR / f"variants_ss_{fam}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out.name}")

    print(f"\nwrote ss_by_gene.json")


if __name__ == "__main__":
    main()
