"""
Position-based feature extraction using MSA alignment.

Provides MSA-aligned position mapping for nAChR's 16 subunits.
If no MSA file is available, falls back to raw (unaligned) positions.

Adapted from VEP-ENaC for nAChR.
"""

import warnings
from pathlib import Path
from typing import Optional

import pandas as pd
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord

from vep_nachr.features.noah_features.config import WILDTYPE_MSA_FILE
from vep_nachr.config import SOURCE_DATA_DIR, SUBUNIT_TO_FASTA_DIR


def _find_fasta_files() -> dict[str, Path]:
    """
    Find all available wildtype FASTA files for nAChR subunits.

    Searches the NCBI dataset directory for each subunit.

    Returns
    -------
    dict[str, Path]
        Mapping from key (e.g. "human_CHRNA1") to FASTA file path
    """
    fasta_map = {}
    protein_dir = SOURCE_DATA_DIR / "protein_scequences"

    for subunit, fasta_subdir in SUBUNIT_TO_FASTA_DIR.items():
        fasta_path = protein_dir / fasta_subdir / "ncbi_dataset" / "data" / "protein.faa"
        if fasta_path.exists():
            fasta_map[f"human_{subunit}"] = fasta_path

    return fasta_map


def load_uniprot_sequences(fasta_files: Optional[dict[str, Path]] = None) -> dict[str, SeqRecord]:
    """
    Load UniProt/NCBI sequences from FASTA files.

    Args:
        fasta_files: Optional pre-computed mapping of key → FASTA path.
                     If None, auto-discovers from protein_scequences/.

    Returns:
        Dictionary mapping sequence keys (e.g. "human_CHRNA1") to SeqRecord objects
    """
    if fasta_files is None:
        fasta_files = _find_fasta_files()

    if not fasta_files:
        warnings.warn("No FASTA files found for nAChR subunits")
        return {}

    uniprot_seqs = {}
    for key, filepath in fasta_files.items():
        try:
            with open(filepath, "r") as f:
                seq_record = SeqIO.read(f, "fasta")
                uniprot_seqs[key] = seq_record
        except Exception as e:
            warnings.warn(f"Failed to load {key} from {filepath}: {e}")

    return uniprot_seqs


def get_msa(file_path: Optional[Path] = None) -> list[SeqRecord]:
    """
    Load the multiple sequence alignment (MSA) from file.

    Args:
        file_path: Path to the MSA file (FASTA format)

    Returns:
        List of SeqRecord objects representing the aligned sequences.
        Returns empty list if no MSA file exists.
    """
    if file_path is None:
        file_path = WILDTYPE_MSA_FILE

    file_path = Path(file_path)

    if not file_path.exists():
        warnings.warn(
            f"MSA file not found: {file_path}. "
            "Aligned position features will use raw (unaligned) positions as fallback. "
            "Generate an MSA to get alignment-based positions."
        )
        return []

    return list(SeqIO.parse(file_path, "fasta"))


def build_all_wildtype_msa_maps(
    uniprot_seqs: dict[str, SeqRecord]
) -> dict[str, dict[str, dict[int, int]]]:
    """
    Create alignment-based mapping from wildtype sequences to MSA positions.

    If no MSA file is available, returns an identity map (raw position → raw position).

    Args:
        uniprot_seqs: Dictionary of the original, unaligned sequence records

    Returns:
        Nested dictionary: msa_maps[species][subunit][original_position] = aligned_position
    """
    msa = get_msa()

    # If no MSA, build identity maps from uniprot sequences
    if not msa:
        return _build_identity_maps(uniprot_seqs)

    msa_maps: dict[str, dict[str, dict[int, int]]] = {}

    for seq_key, original_seq_record in uniprot_seqs.items():
        if not seq_key or not original_seq_record:
            continue

        parts = seq_key.split("_", 1)
        if len(parts) != 2:
            continue
        species, subunit = parts

        if species not in msa_maps:
            msa_maps[species] = {}

        # Find matching MSA sequence
        aligned_seq = None
        for msa_seq in msa:
            if msa_seq.id and original_seq_record.id and msa_seq.id in original_seq_record.id:
                aligned_seq = msa_seq
                break

        if not aligned_seq:
            # Fall back to identity map for this subunit
            continue

        # Build position mapping: unaligned_pos → aligned_column
        position_map: dict[int, int] = {}
        unaligned_pos = 1
        for aligned_pos, residue in enumerate(str(aligned_seq.seq), 1):
            if residue != "-":
                position_map[unaligned_pos] = aligned_pos
                unaligned_pos += 1

        msa_maps[species][subunit] = position_map

    return msa_maps


def _build_identity_maps(
    uniprot_seqs: dict[str, SeqRecord]
) -> dict[str, dict[str, dict[int, int]]]:
    """Build identity (1:1) position maps when no MSA is available."""
    identity_maps: dict[str, dict[str, dict[int, int]]] = {}

    for seq_key, seq_record in uniprot_seqs.items():
        parts = seq_key.split("_", 1)
        if len(parts) != 2:
            continue
        species, subunit = parts

        seq_len = len(seq_record.seq)
        if species not in identity_maps:
            identity_maps[species] = {}
        identity_maps[species][subunit] = {i: i for i in range(1, seq_len + 1)}

    return identity_maps


def generate_aligned_positions(
    df: pd.DataFrame,
    msa_maps: Optional[dict[str, dict[str, dict[int, int]]]] = None,
) -> pd.DataFrame:
    """
    Generate aligned positions for each mutation using MSA mapping.

    Args:
        df: DataFrame with columns: species, subunit (or protein_subunit),
            position (or mutation_position)
        msa_maps: Optional pre-computed MSA maps

    Returns:
        DataFrame with 'aligned_position' column
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"df is {type(df)}, but expected pd.DataFrame")

    # Load sequences and build MSA maps if not provided
    if msa_maps is None:
        sequences = load_uniprot_sequences()
        msa_maps = build_all_wildtype_msa_maps(sequences)

    # Determine column names (nAChR uses "subunit" and "position")
    subunit_col = "subunit" if "subunit" in df.columns else "protein_subunit"
    pos_col = "position" if "position" in df.columns else "mutation_position"

    # Species column may or may not exist
    has_species = "species" in df.columns

    aligned_positions = []
    for _, row in df.iterrows():
        species = str(row.get("species", "human")).lower() if has_species else "human"
        subunit = str(row.get(subunit_col, "")).upper()
        position = int(row.get(pos_col, 0))

        position_map = msa_maps.get(species, {}).get(subunit, {})
        aligned_pos = position_map.get(position)
        aligned_positions.append(aligned_pos)

    result = pd.DataFrame({
        "aligned_position": pd.Series(aligned_positions, index=df.index),
    })

    return result
