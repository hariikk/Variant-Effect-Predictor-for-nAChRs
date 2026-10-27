# Generate 3 figures matched EXACTLY to the example poster's figure-box aspect ratios.
# Fixes this pass:
#   fig1 — vertical layout recentred so the orange "Prediction" box is no longer cut off
#   fig2 — split into two panels (3-class / binary), each sorted independently (less cluttered)
#   fig3 — whole-program (75-feature, 7-group) ablation, loaded from the real RF + CatBoost CSVs
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Patch
from matplotlib import font_manager
import matplotlib.transforms as mtransforms
from PIL import Image
import numpy as np

for f in ["Helvetica", "Arial", "DejaVu Sans"]:
    if any(f.lower() in x.name.lower() for x in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = f
        break

BLUE   = "#009FD6"
DBLUE  = "#016ABE"
ORANGE = "#C55A11"
GRAY   = "#6B7280"
LGRAY  = "#E5E7EB"
DARK   = "#1F2937"
DPI    = 200
OUT    = "poster_cytotransport/"

# Target figure-box sizes (inches) taken from the example PPTX, so the PNGs land
# at the same aspect ratio and don't get squished by replace_picture.
BOX = {
    "fig1_pipeline.png": (8.92, 9.26),
    "fig2_models.png":   (11.71, 9.70),
    "fig3_ablation.png": (9.79, 11.34),
}

def save_exact(fig, name):
    W, H = BOX[name]
    fig.savefig(OUT + name, dpi=DPI, facecolor="white",
                bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    im = Image.open(OUT + name)
    cw, ch = im.size
    target = W / H
    cur = cw / ch
    if abs(cur - target) < 0.004:
        return
    if cur > target:  # content too wide -> pad top/bottom
        nh = int(round(cw / target))
        canvas = Image.new("RGB", (cw, nh), "white")
        canvas.paste(im, (0, (nh - ch) // 2))
    else:             # content too tall -> pad left/right
        nw = int(round(ch * target))
        canvas = Image.new("RGB", (nw, ch), "white")
        canvas.paste(im, ((nw - cw) // 2, 0))
    canvas.save(OUT + name)

# ------------------------------------------------------------------
# FIG 1 — vertical pipeline flowchart (8.92 x 9.26 in)
# ------------------------------------------------------------------
def make_pipeline():
    W, H = 8.92, 9.26
    fig, ax = plt.subplots(figsize=(W, H), dpi=DPI)
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")

    steps = [
        ("1 · Curated data", "717 variants\n16 genes · 3 species", BLUE),
        ("2 · Features", "75 engineered\n7 groups", DBLUE),
        ("3 · Classifiers", "10 models\nOptuna-tuned", DBLUE),
        ("4 · Evaluation", "gene-grouped CV\n(no leakage)", BLUE),
        ("5 · Prediction", "direction: GOF / NNE / LOF", ORANGE),
    ]
    n = len(steps)
    bw, bh = 6.4, 1.28
    gap = 0.55
    total = n * bh + (n - 1) * gap            # 8.6 in
    top = H - (H - total) / 2                  # centre the stack vertically
    cx = W / 2
    ys = [top - i * (bh + gap) for i in range(n)]
    for i, ((title, sub, color), ytop) in enumerate(zip(steps, ys)):
        # highlight the "curated data" step (i == 0) with a dark outline
        edge = DARK if i == 0 else "none"
        lw = 3.0 if i == 0 else 0
        box = FancyBboxPatch((cx - bw/2, ytop - bh), bw, bh,
                             boxstyle="round,pad=0.05,rounding_size=0.20",
                             linewidth=lw, edgecolor=edge, facecolor=color, zorder=2)
        ax.add_patch(box)
        ax.text(cx, ytop - bh/2 + 0.28, title, ha="center", va="center",
                fontsize=21, fontweight="bold", color="white", zorder=3)
        ax.text(cx, ytop - bh/2 - 0.28, sub, ha="center", va="center",
                fontsize=13, color="white", zorder=3, linespacing=1.25)
        if i < n - 1:
            a = FancyArrowPatch((cx, ytop - bh - 0.05), (cx, ys[i+1] + 0.05),
                                arrowstyle="-|>", mutation_scale=26,
                                linewidth=3.0, color=GRAY, zorder=1)
            ax.add_patch(a)

    save_exact(fig, "fig1_pipeline.png")
    print("fig1_pipeline.png done")

# ------------------------------------------------------------------
# FIG 2 — model comparison, two panels (11.71 x 9.70 in)
# ------------------------------------------------------------------
def make_models():
    W, H = 11.71, 9.70
    model_names = ["CatBoost", "Random Forest", "Logistic Regression", "LightGBM",
                   "SVM-Linear", "XGBoost", "MLP", "KNN", "Gaussian NB", "SVM-RBF"]
    three  = [0.508, 0.501, 0.483, 0.475, 0.474, 0.469, 0.467, 0.465, 0.459, 0.410]
    binary = [0.637, 0.647, 0.598, 0.617, 0.579, 0.625, 0.596, 0.586, 0.585, 0.568]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(W, H), dpi=DPI, sharex=True,
                                   gridspec_kw={"hspace": 0.55})

    # --- panel 1: 3-class, sorted descending ---
    o1 = np.argsort(three)[::-1]
    m1 = [model_names[i] for i in o1]; v1 = [three[i] for i in o1]
    y1 = np.arange(len(m1))
    ax1.barh(y1, v1, height=0.60, color=BLUE, zorder=3)
    ax1.axvline(0.333, color=GRAY, ls="--", lw=1.6, alpha=0.9, zorder=4)
    ax1.set_yticks(y1); ax1.set_yticklabels(m1, fontsize=15)
    ax1.invert_yaxis(); ax1.set_xlim(0, 0.80)
    ax1.set_title("3-class (GOF / NNE / LOF)", loc="left", fontsize=17,
                  fontweight="bold", color=BLUE, pad=6)
    for yi, v in zip(y1, v1):
        ax1.text(v + 0.010, yi, f"{v:.2f}", va="center", fontsize=13, color=DARK)
    ax1.text(0.333, 1.0, "random = 0.33",
             transform=mtransforms.blended_transform_factory(ax1.transData, ax1.transAxes),
             ha="center", va="bottom", fontsize=12, color=GRAY)

    # --- panel 2: binary, sorted descending ---
    o2 = np.argsort(binary)[::-1]
    m2 = [model_names[i] for i in o2]; v2 = [binary[i] for i in o2]
    y2 = np.arange(len(m2))
    ax2.barh(y2, v2, height=0.60, color=ORANGE, zorder=3)
    ax2.axvline(0.50, color=GRAY, ls=":", lw=1.6, alpha=0.9, zorder=4)
    ax2.set_yticks(y2); ax2.set_yticklabels(m2, fontsize=15)
    ax2.invert_yaxis(); ax2.set_xlim(0, 0.80)
    ax2.set_title("Binary (GOF / LOF)", loc="left", fontsize=17,
                  fontweight="bold", color=ORANGE, pad=6)
    for yi, v in zip(y2, v2):
        ax2.text(v + 0.010, yi, f"{v:.2f}", va="center", fontsize=13, color=DARK)
    ax2.text(0.50, 1.0, "random = 0.50",
             transform=mtransforms.blended_transform_factory(ax2.transData, ax2.transAxes),
             ha="center", va="bottom", fontsize=12, color=GRAY)

    ax2.set_xlabel("Macro F1 (gene-grouped CV)", fontsize=17)
    for ax in (ax1, ax2):
        ax.tick_params(axis="x", labelsize=14)
        ax.tick_params(axis="y", length=0)
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
        ax.spines["left"].set_color(LGRAY)
        ax.spines["bottom"].set_color(LGRAY)

    plt.subplots_adjust(left=0.24, right=0.97)
    save_exact(fig, "fig2_models.png")
    print("fig2_models.png done")

# ------------------------------------------------------------------
# FIG 3 — whole-program feature ablation (9.79 x 11.34 in, portrait)
# ------------------------------------------------------------------
def load_ablation(path):
    d = {}
    with open(path, newline='') as f:
        rows = list(csv.reader(f))
    hdr = rows[0]
    di = hdr.index("delta_f1")
    for r in rows[1:]:
        d[r[0]] = float(r[di])
    return d

def make_ablation():
    W, H = 9.79, 11.34
    rf = load_ablation("VEP Nachr2/results/ablation/ablation_impact_random_forest.csv")
    cb = load_ablation("VEP Nachr2/results/ablation/ablation_impact_catboost.csv")

    LABELS = [
        ("drop_positional",      "Positional (gene identity)"),
        ("drop_substitution",    "Substitution (BLOSUM62, Grantham)"),
        ("drop_esm_embeddings",  "ESM-2 embeddings (16 PCA)"),
        ("drop_physicochemical", "Physicochemical (24 AAIndex)"),
        ("drop_conservation",    "Conservation (orthologs)"),
        ("drop_structural_core", "Structural core (RSA, B-factor)"),
        ("drop_esm2",            "ESM-2 zero-shot (effect)"),
    ]
    keys   = [k for k, _ in LABELS]
    mean   = [(rf[k] + cb[k]) / 2 for k in keys]
    lo     = [min(rf[k], cb[k])   for k in keys]
    hi     = [max(rf[k], cb[k])   for k in keys]
    labels = [l for _, l in LABELS]

    # sort most-negative (largest signal) first
    order = np.argsort(mean)
    labels = [labels[i] for i in order]
    mean   = [mean[i]   for i in order]
    lo     = [lo[i]     for i in order]
    hi     = [hi[i]     for i in order]

    fig, ax = plt.subplots(figsize=(W, H), dpi=DPI)
    colors = [BLUE if m < 0 else ORANGE for m in mean]
    y = np.arange(len(mean))
    err = [[m - l for m, l in zip(mean, lo)], [h - m for h, m in zip(hi, mean)]]
    ax.barh(y, mean, xerr=err, color=colors, height=0.58, zorder=3,
            error_kw=dict(ecolor=GRAY, elinewidth=1.4, capsize=4))
    ax.axvline(0, color=DARK, lw=1.4)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=17)
    ax.invert_yaxis()  # most-negative (largest signal) at the top
    ax.set_xlabel("Δ macro F1 when feature group dropped", fontsize=17)
    ax.tick_params(axis="x", labelsize=15)
    ax.set_xlim(-0.12, 0.03)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(LGRAY); ax.spines["bottom"].set_color(LGRAY)
    for yi, m in zip(y, mean):
        off = -0.006 if m < 0 else 0.006
        ax.text(m + off, yi, f"{m:+.3f}", va="center",
                ha="right" if m < 0 else "left", fontsize=15, color=DARK)
    ax.legend(handles=[Patch(color=BLUE,   label="dropping it hurts → signal"),
                       Patch(color=ORANGE, label="dropping it helps → noise")],
              loc="lower left", fontsize=15, frameon=False)
    ax.text(0.5, 0.98, "Mean of Random Forest & CatBoost · error bars = model range",
            transform=mtransforms.blended_transform_factory(fig.transFigure, ax.transAxes),
            ha="center", va="top", fontsize=12.5, color=GRAY)
    plt.subplots_adjust(left=0.42, right=0.97, top=0.97, bottom=0.11)
    save_exact(fig, "fig3_ablation.png")
    print("fig3_ablation.png done")

make_pipeline()
make_models()
make_ablation()
