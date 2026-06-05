"""
Physicochemical feature extraction from amino acid properties.

Computes 8 AAIndex-based property scales for wildtype and mutant amino acids,
plus the difference (delta). Total: 24 features per mutation.

Scales used:
1. Hydrophobicity (Eisenberg, EISD840101)
2. Polarity (Grantham, GRAR740102)
3. Volume (Krigbaum & Komoriya, KRIW790103)
4. Molecular Weight (Fasman, FASG760101)
5. Charge at pH 7 (Klein, KLEP840101)
6. Isoelectric Point (Zimmerman, ZIMJ680104)
7. Aromaticity (binary: F/W/Y/H = 1, others = 0)
8. Secondary Structure Preference (Chou & Fasman, CHOP780201)

All AAIndex scales are min-max normalized to [0, 1] across the 20 standard AAs.
"""

import numpy as np
from aaindex import aaindex1


# =============================================================================
# PROPERTY DEFINITIONS
# =============================================================================

AA_PROPERTIES = [
    "hydrophobicity",
    "polarity",
    "volume",
    "molecular_weight",
    "charge",
    "isoelectric_point",
    "aromaticity",
    "secondary_structure_preference",
]

_AA_PROPERTY_ACCESSIONS: list[str | None] = [
    "EISD840101",  # hydrophobicity
    "GRAR740102",  # polarity
    "KRIW790103",  # volume
    "FASG760101",  # molecular_weight
    "KLEP840101",  # charge
    "ZIMJ680104",  # isoelectric_point
    None,          # aromaticity (binary, computed locally)
    "CHOP780201",  # secondary_structure_preference
]

_STANDARD_AAS = list("ACDEFGHIKLMNPQRSTVWY")
_AROMATIC_AAS = frozenset("FWYH")


# =============================================================================
# BUILD LOOKUP TABLE
# =============================================================================

def _build_aa_property_table() -> dict[str, list[float]]:
    """Build the AA property lookup table, normalized to [0, 1].

    Each scale is min-max normalized across the 20 standard amino acids.
    Unknown amino acid 'X' gets the midpoint value 0.5 for every property.
    """
    normed: dict[str, dict[str, float]] = {}
    for acc in _AA_PROPERTY_ACCESSIONS:
        if acc is None:
            continue
        raw = {a: aaindex1[acc].values[a] for a in _STANDARD_AAS}
        mn, mx = min(raw.values()), max(raw.values())
        normed[acc] = {
            a: (raw[a] - mn) / (mx - mn) if mx != mn else 0.0
            for a in _STANDARD_AAS
        }

    table: dict[str, list[float]] = {}
    for aa in _STANDARD_AAS:
        row: list[float] = []
        for acc in _AA_PROPERTY_ACCESSIONS:
            if acc is None:  # aromaticity
                row.append(1.0 if aa in _AROMATIC_AAS else 0.0)
            else:
                row.append(normed[acc][aa])
        table[aa] = row

    table["X"] = [0.5] * len(AA_PROPERTIES)
    return table


# Built once at import time
AA_PROPERTY_TABLE: dict[str, list[float]] = _build_aa_property_table()


# =============================================================================
# FEATURE EXTRACTION FUNCTIONS
# =============================================================================

def get_aa_properties(aa: str) -> np.ndarray:
    """Get the 8-dimensional property vector for an amino acid.

    Parameters
    ----------
    aa : str
        Single-letter amino acid code.

    Returns
    -------
    np.ndarray
        Array of 8 normalized property values.
    """
    aa = aa.upper()
    if aa in AA_PROPERTY_TABLE:
        return np.array(AA_PROPERTY_TABLE[aa])
    return np.array(AA_PROPERTY_TABLE["X"])


def extract_physicochemical_features(wt_aa: str, mt_aa: str) -> np.ndarray:
    """
    Extract physicochemical features for a WT -> MT substitution.

    Returns 24 features:
    - 8 wildtype properties
    - 8 mutant properties
    - 8 differences (mutant - wildtype)

    Parameters
    ----------
    wt_aa : str
        Wildtype amino acid (single letter).
    mt_aa : str
        Mutant amino acid (single letter).

    Returns
    -------
    np.ndarray
        24-dimensional feature vector.
    """
    wt_props = get_aa_properties(wt_aa)
    mt_props = get_aa_properties(mt_aa)
    diff = mt_props - wt_props
    return np.concatenate([wt_props, mt_props, diff])


def get_physicochemical_feature_names() -> list[str]:
    """Get the names of all 24 physicochemical features."""
    names = []
    for prefix in ["wt", "mt", "diff"]:
        for prop in AA_PROPERTIES:
            names.append(f"{prefix}_{prop}")
    return names
