"""
Data loading and cleaning for nAChR mutation database.

Loads the manually curated Excel file, standardizes column names,
filters to substitution-only mutations, encodes labels, and loads
wildtype FASTA sequences.
"""

import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from Bio import SeqIO

from vep_nachr.config import (
    SOURCE_DATA_DIR,
    COLUMN_MAPPING,
    LABEL_MAPPING,
    CANONICAL_ACCESSIONS,
    SUBUNIT_TO_FASTA_DIR,
    AMINO_ACIDS,
)


def load_raw_data(path: Optional[str | Path] = None) -> pd.DataFrame:
    """
    Load the raw nAChR mutation database from Excel.

    Parameters
    ----------
    path : str or Path, optional
        Path to the Excel file. Defaults to nachr_db_cleaned.xlsx
        in the parent project directory.

    Returns
    -------
    pd.DataFrame
        Raw dataframe with original column names.
    """
    if path is None:
        path = SOURCE_DATA_DIR / "nachr_db_cleaned.xlsx"
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Database file not found: {path}")

    df = pd.read_excel(path)
    return df


def clean_data(
    df: pd.DataFrame,
    keep_only_substitutions: bool = True,
    drop_no_net_effect: bool = True,
    drop_ambiguous: bool = True,
) -> pd.DataFrame:
    """
    Clean and standardize the nAChR mutation data.

    Steps:
    1. Rename columns to standardized names
    2. Strip whitespace from string columns
    3. Filter to substitutions only (optional)
    4. Clean effect labels (strip whitespace, standardize case)
    5. Drop ambiguous LOF/GOF entries (optional)
    6. Drop "No net effect" entries (optional)
    7. Validate amino acids are standard single-letter codes

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe from load_raw_data()
    keep_only_substitutions : bool
        If True, keep only "Substitution" modification types
    drop_no_net_effect : bool
        If True, remove "No net effect" entries
    drop_ambiguous : bool
        If True, remove "LOF/GOF" ambiguous entries

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe with standardized columns.
    """
    df = df.copy()

    # Drop unnamed columns
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]

    # Rename columns
    df = df.rename(columns=COLUMN_MAPPING)

    # Strip whitespace from string columns
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip()

    # Standardize modification type
    df["modification_type"] = df["modification_type"].str.capitalize()

    # Filter to substitutions only
    if keep_only_substitutions:
        df = df[df["modification_type"] == "Substitution"].copy()

    # Clean effect labels
    df["effect"] = df["effect"].str.strip().str.upper()
    # Fix any remaining variations
    df["effect"] = df["effect"].replace({
        "LOF/GOF": "LOF/GOF",
        "NO NET EFFECT": "NO_NET_EFFECT",
    })

    # Drop ambiguous labels
    if drop_ambiguous:
        df = df[df["effect"] != "LOF/GOF"].copy()

    # Drop no-net-effect
    if drop_no_net_effect:
        df = df[df["effect"] != "NO_NET_EFFECT"].copy()
        df = df[df["effect"] != "NO NET EFFECT"].copy()

    # Validate amino acids
    valid_aa = set(AMINO_ACIDS)
    mask_wt = df["wildtype_aa"].isin(valid_aa)
    mask_mt = df["variant_aa"].isin(valid_aa)
    invalid = df[~(mask_wt & mask_mt)]
    if len(invalid) > 0:
        warnings.warn(
            f"Dropping {len(invalid)} rows with non-standard amino acids: "
            f"{invalid[['subunit', 'position', 'wildtype_aa', 'variant_aa']].to_string()}"
        )
        df = df[mask_wt & mask_mt].copy()

    # Ensure position is integer
    df["position"] = pd.to_numeric(df["position"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["position"])
    df["position"] = df["position"].astype(int)

    # Reset index
    df = df.reset_index(drop=True)

    return df


def encode_labels(df: pd.DataFrame) -> np.ndarray:
    """
    Encode effect labels to integers.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataframe with 'effect' column containing 'LOF' or 'GOF'.

    Returns
    -------
    np.ndarray
        Integer labels (0=LOF, 1=GOF).
    """
    return df["effect"].map(LABEL_MAPPING).values.astype(np.int64)


def load_wildtype_sequences(
    fasta_base_dir: Optional[str | Path] = None,
) -> dict[str, str]:
    """
    Load canonical wildtype protein sequences from FASTA files.

    Reads the protein.faa file for each subunit and extracts the
    canonical isoform sequence based on CANONICAL_ACCESSIONS.

    Parameters
    ----------
    fasta_base_dir : str or Path, optional
        Base directory containing subunit folders (chrna1/, chrna2/, etc.).
        Defaults to protein_scequences/ in the parent project directory.

    Returns
    -------
    dict[str, str]
        Mapping from subunit name (e.g., "CHRNA1") to amino acid sequence string.
    """
    if fasta_base_dir is None:
        fasta_base_dir = SOURCE_DATA_DIR / "protein_scequences"
    fasta_base_dir = Path(fasta_base_dir)

    sequences = {}

    for subunit, fasta_dir in SUBUNIT_TO_FASTA_DIR.items():
        fasta_path = fasta_base_dir / fasta_dir / "ncbi_dataset" / "data" / "protein.faa"

        if not fasta_path.exists():
            warnings.warn(f"FASTA file not found for {subunit}: {fasta_path}")
            continue

        canonical_acc = CANONICAL_ACCESSIONS.get(subunit)

        # Parse all sequences in the FASTA file
        best_record = None
        for record in SeqIO.parse(fasta_path, "fasta"):
            # Try to match canonical accession
            if canonical_acc and record.id == canonical_acc:
                best_record = record
                break
        else:
            # If canonical not found, take the first record
            for record in SeqIO.parse(fasta_path, "fasta"):
                best_record = record
                break

        if best_record is not None:
            sequences[subunit] = str(best_record.seq)
        else:
            warnings.warn(f"No sequences found for {subunit}")

    return sequences


def load_dataset(
    data_path: Optional[str | Path] = None,
    fasta_dir: Optional[str | Path] = None,
) -> tuple[pd.DataFrame, np.ndarray, dict[str, str]]:
    """
    Load and prepare the complete dataset for feature engineering.

    This is the main entry point for data loading. Returns cleaned
    mutation data, encoded labels, and wildtype sequences.

    Parameters
    ----------
    data_path : str or Path, optional
        Path to the Excel database file.
    fasta_dir : str or Path, optional
        Path to the FASTA sequence directory.

    Returns
    -------
    tuple[pd.DataFrame, np.ndarray, dict[str, str]]
        - Cleaned mutation DataFrame
        - Integer-encoded labels (0=LOF, 1=GOF)
        - Dict of wildtype sequences per subunit
    """
    # Load and clean
    raw_df = load_raw_data(data_path)
    clean_df = clean_data(raw_df)

    # Encode labels
    labels = encode_labels(clean_df)

    # Load sequences
    sequences = load_wildtype_sequences(fasta_dir)

    print(f"Loaded {len(clean_df)} mutations across {clean_df['subunit'].nunique()} subunits")
    print(f"  LOF: {(labels == 0).sum()}, GOF: {(labels == 1).sum()}")
    print(f"  Loaded {len(sequences)} wildtype sequences")

    return clean_df, labels, sequences


# =============================================================================
# MULTI-SPECIES LOADING
# =============================================================================

def load_multi_species_dataset(
    combined_path: Optional[str | Path] = None,
    human_path: Optional[str | Path] = None,
    mouse_path: Optional[str | Path] = None,
    rat_path: Optional[str | Path] = None,
    fasta_dir: Optional[str | Path] = None,
    species_filter: Optional[list[str]] = None,
) -> tuple[pd.DataFrame, np.ndarray, dict[str, str]]:
    """
    Load and combine mutation data from multiple species.

    Two modes:
    1. **Combined file** (recommended): Single Excel/CSV with a 'Species' column.
       Pass ``combined_path``.
    2. **Separate files**: One file per species. Pass ``human_path``, ``mouse_path``,
       and/or ``rat_path``.

    Parameters
    ----------
    combined_path : str or Path, optional
        Single file containing all species with a 'Species' column.
    human_path : str or Path, optional
        Human-only file. Defaults to nachr_db_cleaned.xlsx
    mouse_path : str or Path, optional
        Mouse-only file.
    rat_path : str or Path, optional
        Rat-only file.
    fasta_dir : str or Path, optional
        Path to FASTA sequence directory.
    species_filter : list[str], optional
        If provided, only keep these species (e.g. ['human', 'mouse']).

    Returns
    -------
    tuple[pd.DataFrame, np.ndarray, dict[str, str]]
        - Combined cleaned DataFrame with 'species' column
        - Integer-encoded labels (0=LOF, 1=GOF)
        - Dict of wildtype sequences per subunit
    """
    dfs = []

    # ── Mode 1: Combined file (single file with Species column) ──────
    if combined_path is not None:
        combined_path = Path(combined_path)
        if not combined_path.exists():
            raise FileNotFoundError(f"Combined data file not found: {combined_path}")

        if combined_path.suffix.lower() in (".csv",):
            raw_df = pd.read_csv(combined_path)
        else:
            raw_df = pd.read_excel(combined_path)

        # Detect and normalize species column
        species_col = None
        for col in raw_df.columns:
            if col.lower() == "species":
                species_col = col
                break

        if species_col is None:
            raise ValueError(
                "Combined file must have a 'Species' column. "
                f"Found columns: {list(raw_df.columns)}"
            )

        # Normalize species names: "Human" → "human", "Mouse" → "mouse", "Rat" → "rat"
        raw_df["species"] = raw_df[species_col].str.strip().str.lower()

        # Drop rows with missing species
        raw_df = raw_df.dropna(subset=["species"])

        # Clean (the COLUMN_MAPPING handles subunit/position/AA/effect renaming)
        clean_df = clean_data(raw_df)
        # species column was added after COLUMN_MAPPING rename; ensure it's kept
        if "species" not in clean_df.columns:
            clean_df["species"] = raw_df.loc[clean_df.index, "species"]
        dfs.append(clean_df)
        print(f"  Combined file: {len(clean_df)} mutations after cleaning")
        for sp in sorted(clean_df["species"].unique()):
            print(f"    {sp}: {(clean_df['species'] == sp).sum()}")

    # ── Mode 2: Separate files per species ────────────────────────────
    else:
        if human_path is None:
            human_path = SOURCE_DATA_DIR / "nachr_db_cleaned.xlsx"

        for species, path in [("human", human_path), ("mouse", mouse_path), ("rat", rat_path)]:
            if path is None:
                continue
            path = Path(path)
            if not path.exists():
                print(f"Warning: {species} data not found at {path} — skipping")
                continue

            if path.suffix.lower() in (".csv",):
                raw_df = pd.read_csv(path)
            else:
                raw_df = pd.read_excel(path)

            clean_df = clean_data(raw_df)
            clean_df["species"] = species
            dfs.append(clean_df)
            print(f"  {species}: {len(clean_df)} mutations")

    if not dfs:
        raise FileNotFoundError("No species data files found. Provide at least one valid path.")

    # Combine all species
    combined = pd.concat(dfs, ignore_index=True)

    # Filter species if requested
    if species_filter:
        combined = combined[combined["species"].str.lower().isin(
            [s.lower() for s in species_filter]
        )].reset_index(drop=True)

    # Encode labels
    labels = encode_labels(combined)

    # Load sequences (human sequences shared across species for position lookup)
    sequences = load_wildtype_sequences(fasta_dir)

    # Print summary
    print(f"\nTotal: {len(combined)} mutations across {combined['subunit'].nunique()} subunits")
    print(f"  LOF: {(labels == 0).sum()}, GOF: {(labels == 1).sum()}")
    if "species" in combined.columns:
        for sp in combined["species"].unique():
            sp_count = (combined["species"] == sp).sum()
            print(f"  {sp}: {sp_count}")
    print(f"  Loaded {len(sequences)} wildtype sequences")

    return combined, labels, sequences


def add_species_column(df: pd.DataFrame, species: str) -> pd.DataFrame:
    """
    Utility: add a 'species' column to an existing DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned mutation data
    species : str
        Species label to assign (e.g. 'human', 'mouse', 'rat')

    Returns
    -------
    pd.DataFrame
        DataFrame with new 'species' column
    """
    df = df.copy()
    df["species"] = species.lower()
    return df
