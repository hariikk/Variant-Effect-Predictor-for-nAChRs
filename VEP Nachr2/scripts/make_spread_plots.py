"""
Plot mutation spread across AA position, for the three gene families and per-gene.

Visualises where variants land along the (canonical, human-mapped) sequence and what
direction of effect they carry, so we can see what the ML model is reading:
  - Family plots  : one strip plot per family (alpha / beta / delta-gamma-epsilon),
                    y = effect value (LOF=-1, NNE=0, GOF=+1), coloured by effect.
  - Per-gene plots: alpha genes (CHRNA1-7,9,10) and beta genes (CHRNB1-4), coloured by gene.

Output -> plots/  (created under VEP Nachr2)
"""

import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

from vep_nachr2.data.loader import load_mutation_data

PLOT_DIR = Path(__file__).resolve().parent.parent / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

EFFECT_VALUE = {"LOF": -1, "No net effect": 0, "GOF": +1}
EFFECT_ORDER = ["LOF", "No net effect", "GOF"]
EFFECT_COLOR = {
    "LOF": "#377eb8",            # blue  (bottom, loss)
    "No net effect": "#999999",  # grey  (middle, neutral)
    "GOF": "#e41a1c",            # red   (top, gain)
}
EFFECT_LABEL = {"LOF": "LOF (−1)", "No net effect": "NNE (0)", "GOF": "GOF (+1)"}

GENE_COLOR = {
    "CHRNA1": "#1f77b4", "CHRNA2": "#ff7f0e", "CHRNA3": "#2ca02c",
    "CHRNA4": "#d62728", "CHRNA5": "#9467bd", "CHRNA6": "#8c564b",
    "CHRNA7": "#e377c2", "CHRNA9": "#7f7f7f", "CHRNA10": "#bcbd22",
    "CHRNB1": "#1f77b4", "CHRNB2": "#ff7f0e", "CHRNB3": "#2ca02c",
    "CHRNB4": "#d62728",
    "CHRND": "#1f77b4", "CHRNE": "#ff7f0e", "CHRNG": "#2ca02c",
}


def family(gene: str) -> str:
    if gene.startswith("CHRNA"):
        return "alpha"
    if gene.startswith("CHRNB"):
        return "beta"
    return "dge"  # delta (CHRND), gamma (CHRNG), epsilon (CHRNE)


def jitter(values: np.ndarray, band: float, rng) -> np.ndarray:
    """Spread a categorical y within [-band, +band] deterministically."""
    return values + rng.uniform(-band, band, size=len(values))


def _style_axes(ax, xlim=None):
    ax.axhline(0.0, color="#333333", lw=0.8, ls="--", zorder=1)
    ax.set_yticks([-1, 0, 1])
    ax.set_yticklabels(["LOF (−1)", "NNE (0)", "GOF (+1)"])
    ax.set_ylim(-1.45, 2.0)  # extra headroom above GOF (+1) so the legend never overlaps points
    ax.set_xlabel("AA position (canonical, human-mapped)")
    ax.grid(axis="x", color="#eeeeee", lw=0.5, zorder=0)
    if xlim is not None:
        ax.set_xlim(xlim)


def strip_effect(ax, sub, rng, x_jitter=0.0):
    """Scatter a subset with colour = effect. Returns nothing."""
    for eff in EFFECT_ORDER:
        m = sub[sub["effect"] == eff]
        if len(m) == 0:
            continue
        y = EFFECT_VALUE[eff] * np.ones(len(m))
        band = 0.18 if eff != "No net effect" else 0.12
        x = m["position"].to_numpy() + (rng.uniform(-x_jitter, x_jitter, len(m)) if x_jitter else 0)
        ax.scatter(
            x, jitter(y, band, rng), s=16, alpha=0.6, linewidths=0,
            color=EFFECT_COLOR[eff], label=EFFECT_LABEL[eff], zorder=2,
        )


def main():
    df = load_mutation_data(data_file="final_mapped.xlsx")
    df["effect_val"] = df["effect"].map(EFFECT_VALUE)
    df["family"] = df["subunit"].map(family)
    rng = np.random.default_rng(0)

    xlim = (df["position"].min() - 5, df["position"].max() + 5)

    # ── Family strip plots (colour = effect) ──────────────────────────────
    family_titles = {"alpha": "α family", "beta": "β family", "dge": "δ / γ / ε family"}
    for fam in ["alpha", "beta", "dge"]:
        sub = df[df["family"] == fam]
        fig, ax = plt.subplots(figsize=(13, 4.2))
        strip_effect(ax, sub, rng, x_jitter=1.0)
        _style_axes(ax, xlim)
        ax.set_title(f"Mutation spread — {family_titles[fam]}  (n = {len(sub)})", fontsize=12)
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc="upper right", frameon=False,
                  ncol=3, markerscale=1.5)
        fig.tight_layout()
        out = PLOT_DIR / f"family_{fam}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"wrote {out}")

    # ── Per-gene strip plots (colour = gene) ──────────────────────────────
    for fam, genes in [("alpha", ["CHRNA1", "CHRNA2", "CHRNA3", "CHRNA4", "CHRNA5",
                                  "CHRNA6", "CHRNA7", "CHRNA9", "CHRNA10"]),
                       ("beta", ["CHRNB1", "CHRNB2", "CHRNB3", "CHRNB4"]),
                       ("dge", ["CHRND", "CHRNE", "CHRNG"])]:
        sub = df[df["subunit"].isin(genes)]
        fig, ax = plt.subplots(figsize=(13, 4.6))
        for gene in genes:
            g = sub[sub["subunit"] == gene]
            if len(g) == 0:
                continue
            y = g["effect_val"].to_numpy()
            band = np.where(y == 0, 0.12, 0.18)
            x = g["position"].to_numpy() + rng.uniform(-1.0, 1.0, len(g))
            ax.scatter(x, y + rng.uniform(-band, band), s=16, alpha=0.6,
                       linewidths=0, color=GENE_COLOR[gene], label=gene, zorder=2)
        _style_axes(ax, xlim)
        ax.set_title(f"Mutation spread — {family_titles[fam]} genes, coloured by gene "
                     f"(n = {len(sub)})", fontsize=12)
        ax.legend(loc="upper right", frameon=False, ncol=3, markerscale=1.5)
        fig.tight_layout()
        out = PLOT_DIR / f"genes_{fam}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"wrote {out}")

    # ── Single-gene plot: CHRNA7 (α7) only, colour = effect ───────────────
    sub = df[df["subunit"] == "CHRNA7"]
    fig, ax = plt.subplots(figsize=(13, 4.4))
    strip_effect(ax, sub, rng, x_jitter=1.0)
    _style_axes(ax, xlim)
    ax.set_title(f"Mutation spread — CHRNA7 (α7)  (n = {len(sub)})", fontsize=12)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="upper right", frameon=False,
              ncol=3, markerscale=1.5)
    fig.tight_layout()
    out = PLOT_DIR / "alpha7.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")

    print(f"\nAll plots saved to {PLOT_DIR}")


if __name__ == "__main__":
    main()
