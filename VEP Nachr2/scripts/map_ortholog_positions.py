"""
Map mouse/rat nAChR residue positions onto the human canonical isoform.

This is the "Level 2" (residue-level) cross-species mapping. "Level 1"
(gene-level) is trivial for nAChRs: gene symbols are identical across
human / mouse / rat (CHRNA1 == Chrna1 == Chrna1), so no gene mapping is
needed.

For each of the 16 nAChR subunits we:
  1. Load the human canonical RefSeq isoform (NP_ accession).
  2. Load mouse and rat orthologs and pick the canonical NP_ isoform
     (closest length to the human canonical; alignment identity as
     tiebreaker).
  3. Globally align human<->mouse and human<->rat (BLOSUM62).
  4. Emit a per-residue position mapping: source_pos -> human_pos
     (empty where the source carries an insertion absent from human).

Outputs (all under data/raw/reference_sequences/):
  mapping/mouse_to_human.csv   per-residue mapping, mouse -> human
  mapping/rat_to_human.csv     per-residue mapping, rat -> human
  mapping/gene_summary.csv     per-gene lengths / identity / gap counts
  mouse/<GENE>.fasta           canonical single-record FASTA (for reference.py)
  rat/<GENE>.fasta             canonical single-record FASTA

Usage:
    python scripts/map_ortholog_positions.py
"""

from __future__ import annotations

import csv
from pathlib import Path

from Bio import SeqIO
from Bio.Align import PairwiseAligner, substitution_matrices

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REF_SEQ_DIR = PROJECT_ROOT / "data" / "raw" / "reference_sequences"
MAPPING_DIR = REF_SEQ_DIR / "mapping"

# Canonical human RefSeq accessions (mirrors vep_nachr2/config.py CANONICAL_ACCESSIONS).
# CHRNA10 uses the full-length isoform NP_065135.2 (450 aa); NP_001289963.1 is a
# truncated splice isoform (244 aa) that cannot align to mouse/rat (447 aa).
CANONICAL_ACCESSIONS = {
    "CHRNA1":  "NP_000070.1",
    "CHRNA2":  "NP_000733.2",
    "CHRNA3":  "NP_000734.2",
    "CHRNA4":  "NP_000735.1",
    "CHRNA5":  "NP_000736.2",
    "CHRNA6":  "NP_004189.1",
    "CHRNA7":  "NP_000737.1",
    "CHRNA9":  "NP_060051.2",
    "CHRNA10": "NP_065135.2",   # full-length (see NOTE above)
    "CHRNB1":  "NP_000738.2",
    "CHRNB2":  "NP_000739.1",
    "CHRNB3":  "NP_000740.1",
    "CHRNB4":  "NP_000741.1",
    "CHRND":   "NP_000742.1",
    "CHRNE":   "NP_000071.1",
    "CHRNG":   "NP_005190.4",
}

_ALIGNER = PairwiseAligner()
_ALIGNER.mode = "global"
_ALIGNER.substitution_matrix = substitution_matrices.load("BLOSUM62")
_ALIGNER.open_gap_score = -10.0
_ALIGNER.extend_gap_score = -0.5


def parse_fasta(path: Path) -> dict[str, str]:
    """Return {accession: sequence} for every record in a FASTA file."""
    return {rec.id: str(rec.seq) for rec in SeqIO.parse(str(path), "fasta")}


def human_fasta_path(gene: str) -> Path:
    return REF_SEQ_DIR / "human" / f"{gene}.fasta"


def ortholog_fasta_path(gene: str, species: str) -> Path:
    return (
        REF_SEQ_DIR / species / gene.lower()
        / "ncbi_dataset" / "data" / "protein.faa"
    )


def _alignment_identity(a: str, b: str) -> float:
    """Fraction of aligned columns where the two sequences agree."""
    aln = _ALIGNER.align(a, b)[0]
    matches = sum(1 for ra, rb in zip(aln[0], aln[1]) if ra == rb and ra != "-")
    cols = sum(1 for ra, rb in zip(aln[0], aln[1]) if not (ra == "-" and rb == "-"))
    return matches / cols if cols else 0.0


def pick_canonical_ortholog(
    records: dict[str, str], human_seq: str
) -> tuple[str, str] | None:
    """Pick the NP_ ortholog: closest length first, then highest identity."""
    np_records = [(k, v) for k, v in records.items() if k.startswith("NP_")]
    if not np_records:
        return None

    def score(item):
        acc, seq = item
        return (-abs(len(seq) - len(human_seq)), _alignment_identity(human_seq, seq))

    return max(np_records, key=score)


def align_and_map(human_seq: str, orth_seq: str):
    """Global-align human vs ortholog; return (mapping, matches, cols, gaps)."""
    aln = _ALIGNER.align(human_seq, orth_seq)[0]
    a, b = aln[0], aln[1]  # a = human, b = ortholog

    mapping: dict[int, int | None] = {}
    human_i = orth_i = 0
    for ra, rb in zip(a, b):
        if ra != "-":
            human_i += 1
        if rb != "-":
            orth_i += 1
        if rb != "-":
            # None marks an insertion in the ortholog (no human equivalent)
            mapping[orth_i] = human_i if ra != "-" else None

    matches = sum(1 for ra, rb in zip(a, b) if ra == rb and ra != "-")
    cols = sum(1 for ra, rb in zip(a, b) if not (ra == "-" and rb == "-"))
    gaps = sum(1 for ra, rb in zip(a, b) if ra == "-" or rb == "-")
    return mapping, matches, cols, gaps


def write_flat_canonical(gene: str, species: str, acc: str, seq: str) -> None:
    """Write a single-record canonical FASTA named <GENE>.fasta (for reference.py)."""
    out_dir = REF_SEQ_DIR / species
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{gene}.fasta"
    header = f">{acc} {gene} canonical [{species}]"
    out_path.write_text(f"{header}\n{seq}\n")


def main() -> None:
    MAPPING_DIR.mkdir(parents=True, exist_ok=True)

    mouse_rows: list[dict] = []
    rat_rows: list[dict] = []
    summary_rows: list[dict] = []
    warnings: list[str] = []

    print(f"{'gene':8s} {'human':>6s} {'mouse':>6s} {'rat':>6s}  "
          f"{'H-M id%':>7s} {'H-R id%':>7s} {'H-M gaps':>8s} {'H-R gaps':>8s}")
    print("-" * 72)

    for gene, acc in CANONICAL_ACCESSIONS.items():
        human_records = parse_fasta(human_fasta_path(gene))
        if acc not in human_records:
            warnings.append(f"{gene}: canonical {acc} not found in human FASTA")
            continue
        human_seq = human_records[acc]

        # Warn if reference.py's `records[0]` (first-in-file) would disagree.
        first_human_id = next(iter(human_records))
        if first_human_id != acc:
            warnings.append(
                f"{gene}: first record in human/{gene}.fasta is {first_human_id}, "
                f"but canonical is {acc} (reference.py picks records[0])"
            )

        for species, rows in (("mouse", mouse_rows), ("rat", rat_rows)):
            orth_path = ortholog_fasta_path(gene, species)
            orth_records = parse_fasta(orth_path)
            picked = pick_canonical_ortholog(orth_records, human_seq)
            if picked is None:
                warnings.append(f"{gene}: no NP_ isoform found for {species}")
                continue
            orth_acc, orth_seq = picked
            mapping, matches, cols, gaps = align_and_map(human_seq, orth_seq)

            write_flat_canonical(gene, species, orth_acc, orth_seq)

            for orth_pos in range(1, len(orth_seq) + 1):
                human_pos = mapping[orth_pos]
                rows.append({
                    "gene": gene,
                    "species": species,
                    "source_pos": orth_pos,
                    "human_pos": human_pos if human_pos is not None else "",
                    "source_aa": orth_seq[orth_pos - 1],
                    "human_aa": human_seq[human_pos - 1] if human_pos else "",
                })

            summary_rows.append({
                "gene": gene,
                "species": species,
                "human_accession": acc,
                "human_len": len(human_seq),
                "orth_accession": orth_acc,
                "orth_len": len(orth_seq),
                "identity_pct": round(100 * matches / cols, 2) if cols else 0.0,
                "gap_columns": gaps,
                "mapped_residues": sum(1 for v in mapping.values() if v is not None),
                "insertions": sum(1 for v in mapping.values() if v is None),
            })

        # print one summary line per gene (mouse + rat stats)
        m = summary_rows[-2]
        r = summary_rows[-1]
        print(
            f"{gene:8s} {len(human_seq):>6d} {m['orth_len']:>6d} {r['orth_len']:>6d}  "
            f"{m['identity_pct']:>7.1f} {r['identity_pct']:>7.1f} {m['gap_columns']:>8d} {r['gap_columns']:>8d}"
        )

    # Write per-residue mapping CSVs.
    for species, rows, fname in (
        ("mouse", mouse_rows, "mouse_to_human.csv"),
        ("rat", rat_rows, "rat_to_human.csv"),
    ):
        with open(MAPPING_DIR / fname, "w", newline="") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=["gene", "species", "source_pos", "human_pos",
                            "source_aa", "human_aa"],
            )
            writer.writeheader()
            writer.writerows(rows)

    # Write gene summary CSV.
    with open(MAPPING_DIR / "gene_summary.csv", "w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["gene", "species", "human_accession", "human_len",
                        "orth_accession", "orth_len", "identity_pct",
                        "gap_columns", "mapped_residues", "insertions"],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print()
    print(f"Wrote {len(mouse_rows)} mouse rows and {len(rat_rows)} rat rows.")
    print(f"Outputs: {MAPPING_DIR}")
    if warnings:
        print("\nWARNINGS:")
        for w in warnings:
            print(f"  - {w}")


if __name__ == "__main__":
    main()
