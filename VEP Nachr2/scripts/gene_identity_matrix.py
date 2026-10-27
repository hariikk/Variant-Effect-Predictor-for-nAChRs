"""
Pairwise sequence identity across the 16 human nAChR subunits.

Source of the "27-63%" claim: computed here from the project's own canonical
RefSeq sequences (data/raw/reference_sequences/human/<GENE>.fasta), using the
same BLOSUM62 global aligner as scripts/map_ortholog_positions.py.

Outputs:
  data/raw/reference_sequences/mapping/gene_identity_matrix.csv   (pairwise %)
  stdout: matrix + per-gene "centrality" (mean identity vs the other 15)

Usage:
    python scripts/gene_identity_matrix.py
"""

from __future__ import annotations

import csv
from pathlib import Path

from Bio import SeqIO
from Bio.Align import PairwiseAligner, substitution_matrices

PROJECT_ROOT = Path(__file__).resolve().parent.parent
HUMAN_DIR = PROJECT_ROOT / "data" / "raw" / "reference_sequences" / "human"
MAPPING_DIR = PROJECT_ROOT / "data" / "raw" / "reference_sequences" / "mapping"

# Canonical human RefSeq accessions (mirrors map_ortholog_positions.py / config.py).
CANONICAL_ACCESSIONS = {
    "CHRNA1": "NP_000070.1", "CHRNA2": "NP_000733.2", "CHRNA3": "NP_000734.2",
    "CHRNA4": "NP_000735.1", "CHRNA5": "NP_000736.2", "CHRNA6": "NP_004189.1",
    "CHRNA7": "NP_000737.1", "CHRNA9": "NP_060051.2", "CHRNA10": "NP_065135.2",
    "CHRNB1": "NP_000738.2", "CHRNB2": "NP_000739.1", "CHRNB3": "NP_000740.1",
    "CHRNB4": "NP_000741.1", "CHRND": "NP_000742.1", "CHRNE": "NP_000071.1",
    "CHRNG": "NP_005190.4",
}

_ALIGNER = PairwiseAligner()
_ALIGNER.mode = "global"
_ALIGNER.substitution_matrix = substitution_matrices.load("BLOSUM62")
_ALIGNER.open_gap_score = -10.0
_ALIGNER.extend_gap_score = -0.5


def load(gene: str) -> str:
    recs = {r.id: str(r.seq) for r in SeqIO.parse(str(HUMAN_DIR / f"{gene}.fasta"), "fasta")}
    acc = CANONICAL_ACCESSIONS[gene]
    return recs[acc] if acc in recs else next(iter(recs.values()))


def identity(a: str, b: str) -> float:
    al = _ALIGNER.align(a, b)[0]
    m = sum(1 for x, y in zip(al[0], al[1]) if x == y and x != "-")
    c = sum(1 for x, y in zip(al[0], al[1]) if not (x == "-" and y == "-"))
    return 100.0 * m / c if c else 0.0


def main() -> None:
    genes = list(CANONICAL_ACCESSIONS)
    seqs = {g: load(g) for g in genes}

    rows = []
    for g1 in genes:
        row = {"gene1": g1}
        for g2 in genes:
            row[g2] = round(identity(seqs[g1], seqs[g2]), 2)
        rows.append(row)

    MAPPING_DIR.mkdir(parents=True, exist_ok=True)
    out = MAPPING_DIR / "gene_identity_matrix.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["gene1"] + genes)
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {out}\n")
    print("Pairwise % identity (human nAChR subunits):")
    print("        " + "".join(f"{g[-4:]:>7s}" for g in genes))
    for g1 in genes:
        print(f"{g1:8s}" + "".join(f"{rows[genes.index(g1)][g2]:7.1f}" for g2 in genes))

    print("\nPer-gene centrality (mean identity vs the other 15):")
    cent = {}
    for g1 in genes:
        vals = [identity(seqs[g1], seqs[g2]) for g2 in genes if g2 != g1]
        cent[g1] = sum(vals) / len(vals)
    for g, v in sorted(cent.items(), key=lambda kv: -kv[1]):
        print(f"  {g:8s} {v:5.1f}%")
    print(f"\nRange: {min(cent.values()):.1f}% (most dissimilar) .. "
          f"{max(cent.values()):.1f}% (most central)")


if __name__ == "__main__":
    main()
