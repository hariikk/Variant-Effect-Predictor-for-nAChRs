"""Append the gene-identity matrix + single-frame experiment to results.ipynb.

Idempotent: skips a section if its marker header already exists.

Usage:
    python scripts/_append_to_results_nb.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NB_PATH = PROJECT_ROOT / "results.ipynb"
IDENTITY_CSV = PROJECT_ROOT / "data" / "raw" / "reference_sequences" / "mapping" / "gene_identity_matrix.csv"
SINGLEFRAME_JSON = PROJECT_ROOT / "results" / "singleframe_comparison.json"

IDENTITY_MARKER = "## Gene identity matrix (16 human nAChR subunits)"
RUN_MARKER = "## Run XXXIX: single-frame mapping"


def build_identity_cells() -> list[dict]:
    with open(IDENTITY_CSV, newline="") as f:
        rows = list(csv.DictReader(f))
    genes = [c for c in rows[0].keys() if c != "gene1"]

    header = "| gene | " + " | ".join(genes) + " |"
    sep = "|" + "---|" * (len(genes) + 1)
    lines = [header, sep]
    for r in rows:
        cells = [r["gene1"]] + [f"{float(r[g]):.1f}" for g in genes]
        lines.append("| " + " | ".join(cells) + " |")

    table = "\n".join(lines)

    note = (
        "Pairwise **BLOSUM62 global-alignment % identity** between the 16 canonical human "
        "RefSeq isoforms (diagonal = self = 100%). Computed by "
        "`scripts/gene_identity_matrix.py`; saved to "
        "`data/raw/reference_sequences/mapping/gene_identity_matrix.csv`.\n\n"
        "For reference, **cross-species** identity (human ↔ mouse/rat, *same* gene) is "
        "**81–94%** — i.e. the genes are far more different from *each other* (27–63%) "
        "than the same gene is across species."
    )

    return [
        {"cell_type": "markdown", "metadata": {}, "source": [IDENTITY_MARKER]},
        {"cell_type": "markdown", "metadata": {}, "source": [note]},
        {"cell_type": "markdown", "metadata": {}, "source": [table]},
    ]


def build_run_cells() -> list[dict]:
    data = json.load(open(SINGLEFRAME_JSON, encoding="utf-8"))
    s = data["settings"]
    b = data["baseline"]
    sf = data["single_frame"]

    setup = (
        "**Setup** — same pipeline, same seeds, same feature set; the only change is the "
        "coordinate system. `cv_mode=\"standard\"` (gene-aware headline), "
        f"{len(s['seeds'])} seeds, {s['n_outer_folds'] if 'n_outer_folds' in s else 5} outer folds, "
        f"{s['n_trials']} Optuna trials, dropped extractors = {s['drop_extractors']} (for speed, "
        "identically on both arms).\n\n"
        "**Arms:** (1) *baseline* = 16 genes, real gene identity + human-numbered positions; "
        "(2) *single-frame* = every variant remapped onto CHRNA2 (the most central subunit) — "
        "gene identity collapsed to one constant column, positions rewritten to the CHRNA2-aligned "
        "residue (`scripts/map_genes_to_single_frame.py`)."
    )

    header = "| model | baseline | single-frame | delta |"
    sep = "|---|---|---|---|"
    lines = [header, sep]
    for m in s["models"]:
        bm = b[m]["macro_f1_mean"]
        sm = sf[m]["macro_f1_mean"]
        lines.append(f"| {m} | {bm:.4f} | {sm:.4f} | {sm - bm:+.4f} |")
    table = "\n".join(lines)

    takeaway = (
        "**Takeaway** — collapsing all 16 genes onto one reference frame *costs* "
        "~0.05–0.06 macro-F1. It is not a free lunch: the mapping is lossy "
        "(~45% of variants land on a non-homologous CHRNA2 residue), and the gene identity the "
        "model was relying on is gone. The remaining ~0.58 is carried by species + amino-acid "
        "features, not by a valid cross-gene position signal."
    )

    return [
        {"cell_type": "markdown", "metadata": {}, "source": [RUN_MARKER + " (all 16 genes → CHRNA2)"]},
        {"cell_type": "markdown", "metadata": {}, "source": [setup]},
        {"cell_type": "markdown", "metadata": {}, "source": [table]},
        {"cell_type": "markdown", "metadata": {}, "source": [takeaway]},
    ]


def main():
    nb = json.load(open(NB_PATH, encoding="utf-8"))
    cells = nb["cells"]

    # Find where our appended sections begin (identity is always first) and drop
    # everything from there onward, so re-running always rebuilds cleanly.
    start = None
    for i, c in enumerate(cells):
        src = "".join(c.get("source", []))
        if IDENTITY_MARKER in src or RUN_MARKER in src:
            start = i
            break
    if start is not None:
        cells = cells[:start]

    added = []
    cells.extend(build_identity_cells())
    added.append("gene-identity matrix")

    if SINGLEFRAME_JSON.exists():
        cells.extend(build_run_cells())
        added.append("single-frame run")
    else:
        print("singleframe_comparison.json not found — skipped (run not finished yet)")

    nb["cells"] = cells
    json.dump(nb, open(NB_PATH, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"results.ipynb updated (added: {added})")


if __name__ == "__main__":
    main()
