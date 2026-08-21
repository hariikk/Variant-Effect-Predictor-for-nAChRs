"""
Reference sequence loading for nAChR subunits.

Loads wildtype FASTA sequences for human subunits (from NCBI RefSeq).
Architecture supports mouse/rat ortholog sequences when available.
"""

from pathlib import Path
from typing import Optional

from Bio import SeqIO

from vep_nachr2.config import (
    REFERENCE_SEQ_DIR,
    NACHR_GENES,
    SPECIES_LIST,
    CANONICAL_ACCESSIONS,
)


def load_reference_sequence(
    gene: str,
    species: str = "human",
) -> str:
    """
    Load the wildtype reference sequence for a single nAChR subunit.

    Parameters
    ----------
    gene : str
        Gene name (e.g., 'CHRNA7').
    species : str
        Species ('human', 'mouse', 'rat'). Currently only human is available.

    Returns
    -------
    str
        Amino acid sequence (single-letter codes).

    Raises
    ------
    FileNotFoundError
        If no FASTA file exists for the gene/species combination.
    """
    fasta_path = REFERENCE_SEQ_DIR / species / f"{gene}.fasta"

    if not fasta_path.exists():
        raise FileNotFoundError(
            f"No reference sequence for {gene} ({species}). "
            f"Expected: {fasta_path}"
        )

    records = list(SeqIO.parse(str(fasta_path), "fasta"))
    if not records:
        raise ValueError(f"Empty FASTA file: {fasta_path}")

    # Prefer the declared canonical accession (config.CANONICAL_ACCESSIONS);
    # the first record in a multi-isoform FASTA is not always the canonical.
    canonical = CANONICAL_ACCESSIONS.get(gene)
    if canonical is not None:
        for rec in records:
            if rec.id == canonical:
                return str(rec.seq)

    return str(records[0].seq)


def load_all_reference_sequences(
    species: str = "human",
) -> dict[str, str]:
    """
    Load all wildtype reference sequences for a given species.

    Parameters
    ----------
    species : str
        Species to load sequences for.

    Returns
    -------
    dict[str, str]
        Mapping from gene name to amino acid sequence.

    Raises
    ------
    FileNotFoundError
        If the species directory doesn't exist or is empty.
    """
    species_dir = REFERENCE_SEQ_DIR / species

    if not species_dir.exists():
        raise FileNotFoundError(
            f"No reference sequence directory for species '{species}'. "
            f"Expected: {species_dir}"
        )

    sequences = {}
    missing = []

    for gene in NACHR_GENES:
        try:
            sequences[gene] = load_reference_sequence(gene, species)
        except (FileNotFoundError, ValueError):
            missing.append(gene)

    if missing:
        import warnings
        warnings.warn(f"Missing reference sequences for: {missing}")

    return sequences


def load_ortholog_position_mapping(species: str) -> dict[str, dict[int, int]]:
    """
    Load the precomputed cross-species position mapping.

    Reads {species}_to_human.csv produced by scripts/map_ortholog_positions.py
    and returns {gene: {source_pos: human_pos}}. Residues that are insertions
    relative to the human reference (no human equivalent) are excluded.

    Returns an empty dict if the mapping file is absent, so callers can fall
    back gracefully (e.g. leave positions in native numbering).
    """
    import csv

    path = REFERENCE_SEQ_DIR / "mapping" / f"{species}_to_human.csv"
    if not path.exists():
        return {}

    mapping: dict[str, dict[int, int]] = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            if not row["human_pos"]:
                continue
            mapping.setdefault(row["gene"], {})[int(row["source_pos"])] = int(
                row["human_pos"]
            )
    return mapping


def validate_variant_against_reference(
    gene: str,
    position: int,
    wildtype_aa: str,
    species: str = "human",
) -> bool:
    """
    Validate that a variant's wildtype AA matches the reference sequence.

    Parameters
    ----------
    gene : str
        Gene name.
    position : int
        1-based position in the reference sequence.
    wildtype_aa : str
        Expected wildtype amino acid (single letter).
    species : str
        Species.

    Returns
    -------
    bool
        True if the reference AA at position matches wildtype_aa.
    """
    sequence = load_reference_sequence(gene, species)

    if position < 1 or position > len(sequence):
        return False

    ref_aa = sequence[position - 1]  # 0-based indexing
    return ref_aa.upper() == wildtype_aa.upper()


def get_sequence_length(gene: str, species: str = "human") -> int:
    """Get the length of a reference sequence."""
    return len(load_reference_sequence(gene, species))


def get_max_position(gene: str, species: str = "human") -> int:
    """Get the maximum position (sequence length) for normalization."""
    return get_sequence_length(gene, species)
