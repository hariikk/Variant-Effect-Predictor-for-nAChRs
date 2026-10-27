"""
Per-gene mutation-spread plots with a vertical 'range' line at each AA position.

Each gene gets its own panel (one row). At a position where the deduplicated
mutations carry two or three different effect directions, a thin vertical line
connects the lowest to the highest effect — so you can see at a glance which
positions are "sensitive": the same AA position flips direction depending on
which amino acid is substituted in.

Uses the *cleaned* variant list (one row per unique gene/position/wt->mt
mutation; mutations whose rows disagree on effect are dropped) — NOT the full
~760-row multi-species table the ML model trains on.

Output -> plots/straightline_{alpha,beta,dge}.png   (new files; existing plots untouched)
"""

import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

from vep_nachr2.data.loader import load_mutation_data
from vep_nachr2.data.viz_clean import clean_variants, conflicting_rows

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

GENE_ORDER = [
    "CHRNA1", "CHRNA2", "CHRNA3", "CHRNA4", "CHRNA5", "CHRNA6", "CHRNA7",
    "CHRNA9", "CHRNA10",
    "CHRNB1", "CHRNB2", "CHRNB3", "CHRNB4",
    "CHRND", "CHRNE", "CHRNG",
]
FAMILIES = {
    "alpha": [g for g in GENE_ORDER if g.startswith("CHRNA")],
    "beta":  [g for g in GENE_ORDER if g.startswith("CHRNB")],
    "dge":   ["CHRND", "CHRNE", "CHRNG"],
}
FAMILY_TITLE = {"alpha": "α family", "beta": "β family", "dge": "δ / γ / ε family"}

LINE_COLOR = "#555555"


def main():
    raw = load_mutation_data(data_file="final_mapped.xlsx")
    conflicts = conflicting_rows(raw)
    if len(conflicts):
        n_ambig = conflicts.groupby(
            ["subunit", "position", "wildtype_aa", "variant_aa"]
        ).ngroups
        print(f"dropping {len(conflicts)} rows spanning {n_ambig} ambiguous "
              f"(multi-effect) mutations:\n{conflicts.to_string(index=False)}\n")
    df = clean_variants(raw)
    print(f"clean (one row per mutation) variants: {len(df)}\n")
    df["effect_val"] = df["effect"].map(EFFECT_VALUE)
    rng = np.random.default_rng(0)

    xlim = (df["position"].min() - 5, df["position"].max() + 5)

    for fam, genes in FAMILIES.items():
        nrow = len(genes)
        fig, axes = plt.subplots(nrow, 1, figsize=(13, 2.0 * nrow),
                                 sharex=True, squeeze=False)
        axes = axes[:, 0]

        for j, gene in enumerate(genes):
            ax = axes[j]
            g = df[df["subunit"] == gene]
            if len(g) == 0:
                ax.set_visible(False)
                continue

            # vertical 'range' line at positions carrying >1 distinct effect
            for pos, grp in g.groupby("position"):
                vals = sorted(grp["effect_val"].unique())
                if len(vals) >= 2:
                    ax.plot([pos, pos], [vals[0], vals[-1]], color=LINE_COLOR,
                            lw=1.2, alpha=0.40, zorder=1, solid_capstyle="round")

            # points coloured by effect (exact x, y jittered only)
            for eff in EFFECT_ORDER:
                m = g[g["effect"] == eff]
                if len(m) == 0:
                    continue
                y = EFFECT_VALUE[eff] * np.ones(len(m))
                band = 0.18 if eff != "No net effect" else 0.12
                ax.scatter(m["position"].to_numpy(),
                           y + rng.uniform(-band, band, len(m)),
                           s=16, alpha=0.7, linewidths=0,
                           color=EFFECT_COLOR[eff], zorder=2)

            ax.axhline(0, color="#888", lw=0.7, ls="--", zorder=0)
            ax.set_yticks([-1, 0, 1])
            ax.set_yticklabels(["LOF", "NNE", "GOF"], fontsize=8)
            ax.set_ylim(-1.45, 1.45)
            ax.set_ylabel(f"{gene}\n(n={len(g)})", rotation=0, labelpad=38,
                          fontsize=9, fontfamily="monospace", va="center")
            ax.set_xlim(xlim)
            ax.grid(axis="x", color="#eee", lw=0.5)

        axes[-1].set_xlabel("AA position (canonical, human-mapped)")
        fig.suptitle(
            f"{FAMILY_TITLE[fam]} — per-gene mutation spread; vertical line = one position "
            f"spans multiple effect directions (n = {len(df[df['subunit'].isin(genes)])})",
            fontsize=11, y=1.0,
        )
        handles = [plt.Line2D([], [], marker="o", ls="", color=EFFECT_COLOR[e],
                              label=EFFECT_LABEL[e]) for e in EFFECT_ORDER]
        fig.legend(handles=handles, loc="upper right", frameon=False,
                   bbox_to_anchor=(0.99, 0.985), ncol=3, markerscale=1.4)
        fig.tight_layout()
        out = PLOT_DIR / f"straightline_{fam}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out}")

    print(f"\nAll straight-line plots saved to {PLOT_DIR}")


if __name__ == "__main__":
    main()
