"""
Unified data loading for VEP-nAChR2 experiments.

Loads from merging_data/final.xlsx (unique_variants sheet), standardizes columns,
filters to substitutions only, and provides label encoding for 3-class prediction.
"""

import warnings
from pathlib import Path
from typing import Optional, Literal

import numpy as np
import pandas as pd

from vep_nachr2.config import (
    RAW_DATA_DIR,
    COLUMN_MAPPING,
    REQUIRED_COLUMNS,
    LABEL_MAPPING,
    LABEL_NAMES,
    EXCLUDED_EFFECTS,
    INCLUDED_MOD_TYPES,
    NACHR_GENES,
    SPECIES_LIST,
    AMINO_ACIDS,
)


# =============================================================================
# DATA FILE PATHS
# =============================================================================

FINAL_XLSX = RAW_DATA_DIR / "final.xlsx"


# =============================================================================
# MAIN LOADING FUNCTION
# =============================================================================

def load_mutation_data(
    source: Literal["final", "unique_variants"] = "unique_variants",
    species: Optional[list[str]] = None,
    subunits: Optional[list[str]] = None,
    effects: Optional[list[str]] = None,
    drop_ambiguous: bool = True,
    substitutions_only: bool = True,
    data_file: Optional[str] = None,
    remap_nonhuman: bool = True,
) -> pd.DataFrame:
    """
    Load nAChR mutation data from final.xlsx.

    Parameters
    ----------
    source : str
        Sheet to load: 'unique_variants' (842 variants, recommended) or 'final' (all_variants).
    species : list[str], optional
        Filter to specific species (e.g., ['human']).
    subunits : list[str], optional
        Filter to specific subunit genes (e.g., ['CHRNA7', 'CHRNA4']).
    effects : list[str], optional
        Filter to specific effects (e.g., ['GOF', 'LOF']).
    drop_ambiguous : bool
        If True, drop rows with ambiguous effect labels (LOF/GOF). Default True.
    substitutions_only : bool
        If True, keep only substitution mutations. Default True.

    Returns
    -------
    pd.DataFrame
        Cleaned mutation data with standardized column names and integer labels.

    Examples
    --------
    >>> df = load_mutation_data()  # 842 unique substitution variants
    >>> df = load_mutation_data(species=['human'], subunits=['CHRNA7'])
    >>> df = load_mutation_data(effects=['GOF', 'LOF'])  # binary subset
    """
    data_path = RAW_DATA_DIR / data_file if data_file else FINAL_XLSX
    if not data_path.exists():
        raise FileNotFoundError(
            f"Data file not found: {data_path}\n"
            "Copy merging_data/final.xlsx to VEP Nachr2/data/raw/final.xlsx"
        )

    # Select sheet
    if source == "unique_variants":
        sheet = "unique_variants"
    else:
        sheet = "all_variants"

    df = pd.read_excel(data_path, sheet_name=sheet)
    initial_n = len(df)

    # Standardize column names
    df = df.rename(columns=COLUMN_MAPPING)

    # Ensure required columns exist
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        available = list(df.columns)
        # Try to find close matches
        for req in missing:
            candidates = [c for c in available if req.lower() in c.lower()]
            if candidates:
                print(f"  Hint: '{req}' not found. Did you mean {candidates}?")
        raise ValueError(
            f"Missing required columns: {missing}\nAvailable columns: {available}"
        )

    # Report available columns
    print(f"Loaded {initial_n} rows from sheet '{sheet}'")

    # ── Clean species names ──
    df["species"] = df["species"].astype(str).str.strip().str.lower()
    # .astype(str) turns a true NaN into the literal string "nan"; restore it to
    # NaN so dropna(subset=["species"]) below actually removes the phantom row.
    df.loc[df["species"].isin(["nan", "none", "null", ""]), "species"] = np.nan

    # ── Clean subunit names (uppercase, strip whitespace) ──
    df["subunit"] = df["subunit"].astype(str).str.strip().str.upper()
    # Handle common variations
    df["subunit"] = df["subunit"].replace({
        "NACHR": "",  # remove prefix if present
    })
    # Validate against known genes
    unknown_genes = set(df["subunit"].unique()) - set(NACHR_GENES)
    if unknown_genes:
        warnings.warn(f"Unknown gene names found: {unknown_genes}")

    # ── Clean modification type ──
    if "modification_type" in df.columns:
        df["modification_type"] = (
            df["modification_type"].astype(str).str.strip()
        )

    # ── Clean AA codes ──
    for col in ["wildtype_aa", "variant_aa"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.upper()
            # Replace empty strings with NaN
            df[col] = df[col].replace("", np.nan)

    # ── Clean position ──
    if "position" in df.columns:
        df["position"] = pd.to_numeric(df["position"], errors="coerce")

    # ── Clean effect labels ──
    df["effect"] = df["effect"].astype(str).str.strip()

    # ── Drop ambiguous effect labels (LOF/GOF) ──
    if drop_ambiguous:
        before = len(df)
        df = df[~df["effect"].isin(EXCLUDED_EFFECTS)]
        dropped = before - len(df)
        if dropped:
            print(f"Dropped {dropped} rows with ambiguous effect (LOF/GOF)")

    # ── Filter to substitutions only ──
    if substitutions_only and "modification_type" in df.columns:
        before = len(df)
        df = df[df["modification_type"].isin(INCLUDED_MOD_TYPES)]
        dropped = before - len(df)
        if dropped:
            print(f"Filtered to substitutions: dropped {dropped} non-substitution rows")

    # ── Drop rows with missing essential values ──
    essential = ["species", "subunit", "position", "wildtype_aa", "variant_aa", "effect"]
    before = len(df)
    df = df.dropna(subset=essential)
    dropped = before - len(df)
    if dropped:
        warnings.warn(f"Dropped {dropped} rows with missing essential values")

    # ── Convert position to int ──
    df["position"] = df["position"].astype(int)

    # ── Remap mouse/rat positions onto the human canonical sequence ──
    # Cross-species rows are recorded in native (mouse/rat) numbering. Map each
    # non-human position to its human-homologous position so positional and
    # structural features share one reference frame with the human data.
    df["native_position"] = df["position"]
    _non_human = df["species"] != "human"
    if remap_nonhuman and _non_human.any():
        try:
            from vep_nachr2.data.reference import load_ortholog_position_mapping

            for _sp in ("mouse", "rat"):
                _mask = df["species"] == _sp
                if not _mask.any():
                    continue
                _mapping = load_ortholog_position_mapping(_sp)
                if not _mapping:
                    warnings.warn(
                        f"No ortholog position mapping for '{_sp}'; "
                        f"positions left in native numbering"
                    )
                    continue

                def _remap(row, _mapping=_mapping):
                    gene = str(row["subunit"]).upper()
                    return _mapping.get(gene, {}).get(int(row["position"]), np.nan)

                df.loc[_mask, "position"] = df.loc[_mask].apply(_remap, axis=1)

            _before = len(df)
            df = df.dropna(subset=["position"])
            _unmappable = _before - len(df)
            if _unmappable:
                warnings.warn(
                    f"Dropped {_unmappable} non-human rows at insertions with "
                    f"no human-equivalent residue"
                )
            df["position"] = df["position"].astype(int)
        except Exception:
            warnings.warn(
                "Cross-species position mapping failed; "
                "positions left in native numbering"
            )

    # ── Validate amino acid codes ──
    valid_aa = set(AMINO_ACIDS)

    def is_valid_aa(aa):
        if pd.isna(aa):
            return False
        aa = str(aa).strip().upper()
        return len(aa) == 1 and aa in valid_aa

    before = len(df)
    df = df[df["wildtype_aa"].apply(is_valid_aa) & df["variant_aa"].apply(is_valid_aa)]
    dropped = before - len(df)
    if dropped:
        warnings.warn(f"Dropped {dropped} rows with invalid amino acid codes")

    # ── Drop variants with position outside the human reference length ──
    # Some rows carry data-entry errors (e.g. CHRNA1 p=1314, CHRNE p=-8) that
    # yield garbage normalized-position values and unmappable structural features.
    # Positions are now all in human numbering (non-human rows were remapped
    # above), so this check applies uniformly to every species.
    try:
        from vep_nachr2.data.reference import load_all_reference_sequences
        _human_ref = load_all_reference_sequences("human")

        def _position_in_range(row) -> bool:
            gene = str(row["subunit"]).upper()
            ref_len = len(_human_ref.get(gene, ""))
            if ref_len == 0:
                return True  # no reference available; leave the row alone
            return 1 <= int(row["position"]) <= ref_len

        before = len(df)
        df = df[df.apply(_position_in_range, axis=1)]
        dropped = before - len(df)
        if dropped:
            warnings.warn(f"Dropped {dropped} rows with position outside human reference length")
    except Exception:
        pass

    # ── Apply user filters ──
    if species is not None:
        species_lower = [s.lower() for s in species]
        df = df[df["species"].str.lower().isin(species_lower)]

    if subunits is not None:
        subunits_upper = [s.upper() for s in subunits]
        df = df[df["subunit"].str.upper().isin(subunits_upper)]

    if effects is not None:
        df = df[df["effect"].isin(effects)]

    # ── Reset index ──
    df = df.reset_index(drop=True)

    # Report final stats
    print(f"Final dataset: {len(df)} variants")
    print(f"  Species: {dict(df['species'].value_counts())}")
    print(f"  Effects: {dict(df['effect'].value_counts())}")
    print(f"  Unique genes: {sorted(df['subunit'].unique())}")

    return df


# =============================================================================
# LABEL ENCODING
# =============================================================================

def encode_labels(effects) -> np.ndarray:
    """Convert effect strings to integer labels (0=LOF, 1=NNE, 2=GOF).

    Parameters
    ----------
    effects : pd.Series or pd.DataFrame
        Series of effect strings, or DataFrame with 'effect' column.

    Returns
    -------
    np.ndarray
        Integer label array.
    """
    if isinstance(effects, pd.DataFrame):
        effects = effects["effect"]
    return effects.replace(LABEL_MAPPING).infer_objects(copy=False).values.astype(np.int64)


def decode_labels(labels: np.ndarray) -> list[str]:
    """Convert integer labels back to effect strings."""
    return [LABEL_NAMES[int(l)] for l in labels]


def get_label_distribution(df: pd.DataFrame) -> dict:
    """Get label distribution with percentages."""
    counts = df["effect"].value_counts()
    total = len(df)
    return {
        effect: {"count": count, "pct": round(100 * count / total, 1)}
        for effect, count in counts.items()
    }


# =============================================================================
# DATASET CONVENIENCE FUNCTIONS
# =============================================================================

def get_dataset(
    name: str = "full",
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Get a ready-to-use dataset.

    Predefined configurations:
    - 'full': All species, all genes, all 3 effects
    - 'human': Human only, all genes
    - 'human_binary': Human only, GOF vs LOF (drop NNE)
    - 'alpha7': CHRNA7 only (largest single gene)
    - 'muscle': Muscle-type genes only (CHRNA1, CHRNB1, CHRND, CHRNE, CHRNG)

    Parameters
    ----------
    name : str
        Dataset configuration name.

    Returns
    -------
    X : np.ndarray (placeholder — call orchestrator separately)
    y : np.ndarray
    metadata : pd.DataFrame
    """
    configs = {
        "full": {},
        "human": {"species": ["human"]},
        "human_binary": {"species": ["human"], "effects": ["GOF", "LOF"]},
        "mouse": {"species": ["mouse"]},
        "rat": {"species": ["rat"]},
        "alpha7": {"subunits": ["CHRNA7"]},
        "muscle": {"subunits": ["CHRNA1", "CHRNB1", "CHRND", "CHRNE", "CHRNG"]},
    }

    if name not in configs:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(configs.keys())}")

    config = configs[name]
    df = load_mutation_data(**config)
    y = encode_labels(df["effect"])

    # Placeholder X — actual features computed by orchestrator
    X = np.zeros((len(df), 1))

    return X, y, df


# =============================================================================
# GROUP KEYS (for StratifiedGroupKFold)
# =============================================================================

import hashlib


def make_subunit_group_key(df: pd.DataFrame) -> np.ndarray:
    """Group key by subunit gene (for leave-one-subunit-out CV).

    This ensures no gene's variants leak across train/test folds.
    """
    keys = df["subunit"].astype(str)
    return np.array([
        hashlib.sha1(k.encode("utf-8")).hexdigest() for k in keys.values
    ])


def make_position_group_key(df: pd.DataFrame) -> np.ndarray:
    """Group key by species + subunit + position."""
    keys = (
        df["species"].astype(str) + "|"
        + df["subunit"].astype(str) + "|"
        + df["position"].astype(str)
    )
    return np.array([
        hashlib.sha1(k.encode("utf-8")).hexdigest() for k in keys.values
    ])


def make_full_group_key(df: pd.DataFrame) -> np.ndarray:
    """Group key by species + subunit + position + PMID."""
    pmid_col = "pmid" if "pmid" in df.columns else "Reference(PMID)"
    keys = (
        df["species"].astype(str) + "|"
        + df["subunit"].astype(str) + "|"
        + df["position"].astype(str) + "|"
        + df[pmid_col].astype(str)
    )
    return np.array([
        hashlib.sha1(k.encode("utf-8")).hexdigest() for k in keys.values
    ])
