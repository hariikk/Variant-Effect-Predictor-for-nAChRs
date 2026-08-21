"""
Normalize variant residue positions onto the human canonical (precursor) numbering.

Reads data/raw/final.xlsx and writes data/raw/final_mapped.xlsx — a copy in which
every row's "AA position" is rewritten to the human precursor RefSeq numbering,
regardless of the species or the numbering convention (mature vs precursor) the
source paper used.

Background
----------
The curated data is NOT in a single numbering frame:
  * some genes are recorded in *precursor* numbering (signal peptide included,
    offset 0) — these already match the RefSeq reference at the reported position;
  * other genes are recorded in *mature-protein* numbering (signal peptide
    stripped, offset = signal-peptide length, ~20-33 aa);
  * a few rows are simple data-entry typos (off by 1-5 residues).

Everything downstream keyed to position (normalized position, TM-helix, DSSP/RSA,
B-factor, pore distance) assumes precursor numbering, so the mature/typo rows were
silently ~20 residues off. This script anchors every row back onto the reference
using its wildtype amino acid, which is the one piece of ground truth the papers
report reliably.

Method (per row, in the *native* species reference):
  1. precursor : wildtype AA matches the reference at the reported position.
  2. mature    : wildtype AA matches at position + signal-peptide length.
  3. typo      : wildtype AA matches at a unique position within +/-5 (typo).
  4. isoform   : wildtype AA matches at a unique position within +15..+40
                 (data uses a non-UniProt frame, e.g. an extra pro-region).
  5. otherwise : ambiguous / no_match -> position left blank (dropped downstream).

For mouse/rat rows the resolved *native* position is then mapped onto the human
reference via the precomputed ortholog alignment
(data/raw/reference_sequences/mapping/{species}_to_human.csv).

Signal-peptide lengths are UniProt "SIGNAL" feature lengths, with two genes
(CHRNA6, CHRNB4) adjusted to the offset the curated literature actually uses.

Output columns added: "mapping_status", "original_position". The raw columns
("Species", "nAChR subunit", "AA position", ...) are preserved so the loader can
consume final_mapped.xlsx unchanged.

Usage:
    python scripts/normalize_positions.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from vep_nachr2.config import RAW_DATA_DIR, NACHR_GENES
from vep_nachr2.data.reference import (
    load_reference_sequence,
    load_ortholog_position_mapping,
)


# Signal-peptide length (mature-protein numbering offset) per gene and species.
# Source: UniProt "SIGNAL" feature (reviewed entries). Two human genes adjusted to
# the frame the curated literature actually uses:
#   CHRNA6 : UniProt signal peptide = 25 aa, but the data's mature rows cluster at
#            +33 (an extra ~8-aa pro-region "GCVGC" is stripped by the sources).
#   CHRNB4 : UniProt signal peptide = 21 aa, but the data clusters at +26.
SIGNAL_PEPTIDE_LENGTHS = {
    "human": {
        "CHRNA1": 20, "CHRNA2": 26, "CHRNA3": 31, "CHRNA4": 26, "CHRNA5": 22,
        "CHRNA6": 33, "CHRNA7": 22, "CHRNA9": 22, "CHRNA10": 24, "CHRNB1": 23,
        "CHRNB2": 25, "CHRNB3": 21, "CHRNB4": 26, "CHRND": 21, "CHRNE": 20,
        "CHRNG": 22,
    },
    "mouse": {
        "CHRNA1": 20, "CHRNA2": 27, "CHRNA3": 25, "CHRNA4": 30, "CHRNA5": 29,
        "CHRNA6": 30, "CHRNA7": 22, "CHRNA9": 22, "CHRNA10": 24, "CHRNB1": 23,
        "CHRNB2": 25, "CHRNB3": 30, "CHRNB4": 20, "CHRND": 24, "CHRNE": 20,
        "CHRNG": 22,
    },
    "rat": {
        "CHRNA1": 20, "CHRNA2": 27, "CHRNA3": 25, "CHRNA4": 30, "CHRNA5": 27,
        "CHRNA6": 30, "CHRNA7": 22, "CHRNA9": 25, "CHRNA10": 24, "CHRNB1": 23,
        "CHRNB2": 24, "CHRNB3": 30, "CHRNB4": 20, "CHRND": 21, "CHRNE": 20,
        "CHRNG": 22,
    },
}

# Raw -> standardized column names (mirrors config.COLUMN_MAPPING, kept local so
# the script doesn't need to rename back and forth for output).
_SPECIES = "Species"
_SUBUNIT = "nAChR subunit"
_POSITION = "AA position"
_WILDTYPE = "Initial AA"


def resolve_native_position(
    ref: str, position: int, wildtype: str, sigpep: int
) -> tuple[int | None, str]:
    """Find the precursor-numbered position in `ref` for a reported variant.

    Returns (position, status). position is None for ambiguous / no-match rows.
    """
    n = len(ref)

    def matches(q: int) -> bool:
        return 1 <= q <= n and ref[q - 1].upper() == wildtype

    if matches(position):
        return position, "precursor"
    if matches(position + sigpep):
        return position + sigpep, "mature"

    # typo: unique match within +/-5 (nearest single candidate)
    small = [q for q in range(position - 5, position + 6) if q != position and matches(q)]
    if len(small) == 1:
        return small[0], "typo"
    if len(small) > 1:
        return None, "ambiguous"

    # isoform / non-standard frame: unique match within +15..+40
    large = [q for q in range(position + 15, position + 41) if matches(q)]
    if len(large) == 1:
        return large[0], "isoform"
    if len(large) > 1:
        return None, "ambiguous"

    return None, "no_match"


def main() -> None:
    src = RAW_DATA_DIR / "final.xlsx"
    dst = RAW_DATA_DIR / "final_mapped.xlsx"

    df = pd.read_excel(src, sheet_name="unique_variants")
    print(f"Read {len(df)} rows from {src}")

    # Preserve the reported position before overwriting.
    df["original_position"] = df[_POSITION]

    ortholog = {
        sp: load_ortholog_position_mapping(sp) for sp in ("mouse", "rat")
    }

    statuses: list[str] = []
    final_positions: list[float] = []
    resolved = 0

    for _, row in df.iterrows():
        species = str(row[_SPECIES]).strip().lower()
        gene = str(row[_SUBUNIT]).strip().upper()
        position = pd.to_numeric(row[_POSITION], errors="coerce")
        wildtype = str(row[_WILDTYPE]).strip().upper()

        # Missing essentials -> leave blank, loader drops it downstream.
        if pd.isna(position) or not wildtype or len(wildtype) != 1:
            statuses.append("missing")
            final_positions.append(np.nan)
            continue

        position = int(position)
        try:
            native_ref = load_reference_sequence(gene, species)
        except (FileNotFoundError, ValueError):
            statuses.append("no_reference")
            final_positions.append(np.nan)
            continue

        sigpep = SIGNAL_PEPTIDE_LENGTHS.get(species, {}).get(gene)
        if sigpep is None:
            sigpep = SIGNAL_PEPTIDE_LENGTHS["human"].get(gene, 0)

        native_pos, status = resolve_native_position(
            native_ref, position, wildtype, sigpep
        )
        if native_pos is None:
            statuses.append(status)
            final_positions.append(np.nan)
            continue

        # Map non-human rows onto the human reference.
        if species != "human":
            human_pos = ortholog[species].get(gene, {}).get(native_pos)
            if human_pos is None:
                statuses.append("unmapped_ortholog")
                final_positions.append(np.nan)
                continue
            status = f"{status}+ortholog"
            final_positions.append(human_pos)
        else:
            final_positions.append(native_pos)

        statuses.append(status)
        resolved += 1

    df["mapping_status"] = statuses
    df[_POSITION] = [np.nan if pd.isna(p) else int(p) for p in final_positions]

    df.to_excel(dst, sheet_name="unique_variants", index=False)

    n = len(df)
    dropped = n - resolved
    print(f"Wrote {dst}")
    print(f"Resolved to human precursor numbering: {resolved}/{n} "
          f"({100 * resolved / n:.1f}%)")
    print(f"Dropped (position blanked): {dropped} ({100 * dropped / n:.1f}%)")
    print("\nStatus breakdown:")
    for status, count in pd.Series(statuses).value_counts().items():
        print(f"  {status:20s} {count}")


if __name__ == "__main__":
    main()
