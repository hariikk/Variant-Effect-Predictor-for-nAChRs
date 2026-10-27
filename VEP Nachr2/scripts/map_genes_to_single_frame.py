"""
Map every variant onto ONE reference gene (the "most central" subunit), and
measure how lossy that mapping is.

This is the empirical test of the "map all genes to one gene" brainstorm.
Method mirrors scripts/map_ortholog_positions.py (BLOSUM62 global alignment),
but instead of human<->ortholog we align CHRNA2 (the most central subunit)
<-> each other subunit.

Two outputs:
  1. stdout: per-gene mapping fidelity — for every variant, does the aligned
     CHRNA2 position carry the SAME wildtype AA (cleanly mapped) or a DIFFERENT
     one (ill-defined — the variant has no well-defined CHRNA2 representation)?
  2. data/raw/final_mapped_singleframe.xlsx — a copy of final_mapped.xlsx with
     every row's subunit set to CHRNA2 and its position rewritten to the
     CHRNA2-aligned position (unmapped rows dropped; wt_mismatch rows kept).

Usage:
    python scripts/map_genes_to_single_frame.py [--central CHRNA2]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from Bio import SeqIO
from Bio.Align import PairwiseAligner, substitution_matrices

from vep_nachr2.config import RAW_DATA_DIR, CANONICAL_ACCESSIONS, NACHR_GENES

HUMAN_DIR = RAW_DATA_DIR / "reference_sequences" / "human"

_ALIGNER = PairwiseAligner()
_ALIGNER.mode = "global"
_ALIGNER.substitution_matrix = substitution_matrices.load("BLOSUM62")
_ALIGNER.open_gap_score = -10.0
_ALIGNER.extend_gap_score = -0.5


def load(gene: str) -> str:
    recs = {r.id: str(r.seq) for r in SeqIO.parse(str(HUMAN_DIR / f"{gene}.fasta"), "fasta")}
    acc = CANONICAL_ACCESSIONS[gene]
    return recs[acc] if acc in recs else next(iter(recs.values()))


def build_gene_map(central: str) -> dict[str, dict[int, int]]:
    """Return {gene: {gene_pos -> central_pos}} for every gene != central."""
    central_seq = load(central)
    maps: dict[str, dict[int, int]] = {}
    for gene in NACHR_GENES:
        if gene == central:
            continue
        seq = load(gene)
        al = _ALIGNER.align(central_seq, seq)[0]  # a=central, b=gene
        a, b = al[0], al[1]
        m: dict[int, int] = {}
        ci = gi = 0
        for ra, rb in zip(a, b):
            if ra != "-":
                ci += 1
            if rb != "-":
                gi += 1
            if rb != "-":
                if ra != "-":
                    m[gi] = ci  # gene_pos -> central_pos
                # else: insertion in gene relative to central -> unmapped
        maps[gene] = m
    return maps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--central", default="CHRNA2")
    args = ap.parse_args()
    central = args.central

    src = RAW_DATA_DIR / "final_mapped.xlsx"
    dst = RAW_DATA_DIR / "final_mapped_singleframe.xlsx"

    df = pd.read_excel(src, sheet_name="unique_variants")

    # Drop rows the loader would drop anyway (missing position / wildtype) so
    # int(row["AA position"]) never sees a NaN.
    before = len(df)
    df = df.dropna(subset=["AA position", "Initial AA", "nAChR subunit"]).copy()
    if len(df) < before:
        print(f"(dropped {before - len(df)} rows with missing position/wildtype/gene)")

    central_seq = load(central)
    maps = build_gene_map(central)

    df["orig_gene"] = df["nAChR subunit"]
    df["orig_position"] = df["AA position"]

    mapped_pos = []
    status = []
    for _, row in df.iterrows():
        gene = str(row["nAChR subunit"]).strip().upper()
        pos = int(row["AA position"])
        wt = str(row["Initial AA"]).strip().upper()

        if gene == central:
            mapped_pos.append(pos)
            status.append("central")
            continue

        gmap = maps.get(gene, {})
        if pos not in gmap:
            mapped_pos.append(pd.NA)
            status.append("unmapped")
            continue

        cpos = gmap[pos]
        central_aa = central_seq[cpos - 1].upper()
        mapped_pos.append(cpos)
        status.append("wt_match" if central_aa == wt else "wt_mismatch")

    df["singleframe_status"] = status
    df["AA position"] = mapped_pos
    df["nAChR subunit"] = central

    # Drop unmapped rows (no CHRNA2 coordinate at all).
    df = df[df["singleframe_status"] != "unmapped"].copy()

    df.to_excel(dst, sheet_name="unique_variants", index=False)

    # ---- report ----
    n = len(df)
    print(f"Central gene: {central} (len {len(central_seq)})")
    print(f"Variants: {n} (unmapped dropped from {n + status.count('unmapped')} total)\n")

    print("Mapping fidelity (does the aligned CHRNA2 position keep the variant's WT AA?):")
    per = {}
    for gene in NACHR_GENES:
        sub = df[df["orig_gene"] == gene]
        if len(sub) == 0:
            continue
        ok = (sub["singleframe_status"] == "wt_match").sum()
        bad = (sub["singleframe_status"] == "wt_mismatch").sum()
        per[gene] = (ok, bad, len(sub))
        print(f"  {gene:8s} n={len(sub):3d}  wt_match={ok:3d} ({100*ok/len(sub):4.1f}%)  wt_mismatch={bad:3d}")

    ok = (df["singleframe_status"] == "wt_match").sum()
    bad = (df["singleframe_status"] == "wt_mismatch").sum()
    cen = (df["singleframe_status"] == "central").sum()
    print(f"\n  {'TOTAL':8s} n={n:3d}  wt_match={ok:3d} ({100*ok/n:4.1f}%)  wt_mismatch={bad:3d}  central={cen}")
    print(f"\nWrote {dst}")

    # sanity: how many variants are cleanly representable in the single frame
    clean = ok + cen
    print(f"\n>>> {100*clean/n:.1f}% of variants are cleanly representable in the "
          f"{central} frame; the other {100*bad/n:.1f}% have a wildtype mismatch "
          f"(ill-defined in a single-gene reference).")


if __name__ == "__main__":
    main()
