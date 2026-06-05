"""
Substitution scoring features: BLOSUM62 and Grantham distance.

BLOSUM62: Log-odds substitution score from the Henikoff & Henikoff (1992) matrix.
Higher scores = more expected substitution. Range: [-4, 11].

Grantham distance: Composite physicochemical distance between amino acid pairs,
based on composition, polarity, and molecular volume (Grantham 1974).
Range: [0, 215]. Higher = more radical substitution.
"""

import numpy as np
from Bio.Align import substitution_matrices

from vep_nachr.config import AMINO_ACIDS


# =============================================================================
# BLOSUM62
# =============================================================================

_STANDARD_AAS = list(AMINO_ACIDS)

BLOSUM62 = substitution_matrices.load("BLOSUM62")

_BLOSUM62_MIN: float = float(min(
    BLOSUM62[a, b] for a in _STANDARD_AAS for b in _STANDARD_AAS
))
_BLOSUM62_MAX: float = float(max(
    BLOSUM62[a, b] for a in _STANDARD_AAS for b in _STANDARD_AAS
))


def get_blosum_score(aa1: str, aa2: str) -> int:
    """Get raw BLOSUM62 substitution score."""
    try:
        return int(BLOSUM62[aa1.upper(), aa2.upper()])
    except (KeyError, IndexError):
        return 0


def get_blosum_normalized(aa1: str, aa2: str) -> float:
    """Get BLOSUM62 score normalized to [0, 1]."""
    score = get_blosum_score(aa1, aa2)
    return (score - _BLOSUM62_MIN) / (_BLOSUM62_MAX - _BLOSUM62_MIN)


# =============================================================================
# GRANTHAM DISTANCE
# =============================================================================

# Grantham distance matrix (Grantham 1974, Science 185:862-864)
# Measures physicochemical difference between amino acid pairs.
# This is a symmetric matrix; values from the original publication.
_GRANTHAM_DISTANCES = {
    ("A", "R"): 112, ("A", "N"): 111, ("A", "D"): 126, ("A", "C"): 195,
    ("A", "Q"): 91,  ("A", "E"): 107, ("A", "G"): 60,  ("A", "H"): 86,
    ("A", "I"): 94,  ("A", "L"): 96,  ("A", "K"): 106, ("A", "M"): 84,
    ("A", "F"): 113, ("A", "P"): 27,  ("A", "S"): 99,  ("A", "T"): 58,
    ("A", "W"): 148, ("A", "Y"): 112, ("A", "V"): 64,
    ("R", "N"): 86,  ("R", "D"): 96,  ("R", "C"): 180, ("R", "Q"): 43,
    ("R", "E"): 54,  ("R", "G"): 125, ("R", "H"): 29,  ("R", "I"): 97,
    ("R", "L"): 102, ("R", "K"): 26,  ("R", "M"): 91,  ("R", "F"): 97,
    ("R", "P"): 103, ("R", "S"): 110, ("R", "T"): 71,  ("R", "W"): 101,
    ("R", "Y"): 77,  ("R", "V"): 96,
    ("N", "D"): 23,  ("N", "C"): 139, ("N", "Q"): 46,  ("N", "E"): 42,
    ("N", "G"): 80,  ("N", "H"): 68,  ("N", "I"): 149, ("N", "L"): 153,
    ("N", "K"): 94,  ("N", "M"): 142, ("N", "F"): 158, ("N", "P"): 91,
    ("N", "S"): 46,  ("N", "T"): 65,  ("N", "W"): 174, ("N", "Y"): 143,
    ("N", "V"): 133,
    ("D", "C"): 154, ("D", "Q"): 61,  ("D", "E"): 45,  ("D", "G"): 94,
    ("D", "H"): 81,  ("D", "I"): 168, ("D", "L"): 172, ("D", "K"): 101,
    ("D", "M"): 160, ("D", "F"): 177, ("D", "P"): 108, ("D", "S"): 65,
    ("D", "T"): 85,  ("D", "W"): 181, ("D", "Y"): 160, ("D", "V"): 152,
    ("C", "Q"): 154, ("C", "E"): 170, ("C", "G"): 159, ("C", "H"): 174,
    ("C", "I"): 198, ("C", "L"): 198, ("C", "K"): 202, ("C", "M"): 196,
    ("C", "F"): 205, ("C", "P"): 169, ("C", "S"): 112, ("C", "T"): 149,
    ("C", "W"): 215, ("C", "Y"): 194, ("C", "V"): 192,
    ("Q", "E"): 29,  ("Q", "G"): 87,  ("Q", "H"): 24,  ("Q", "I"): 109,
    ("Q", "L"): 113, ("Q", "K"): 53,  ("Q", "M"): 101, ("Q", "F"): 116,
    ("Q", "P"): 76,  ("Q", "S"): 68,  ("Q", "T"): 42,  ("Q", "W"): 130,
    ("Q", "Y"): 99,  ("Q", "V"): 96,
    ("E", "G"): 98,  ("E", "H"): 40,  ("E", "I"): 134, ("E", "L"): 138,
    ("E", "K"): 56,  ("E", "M"): 126, ("E", "F"): 140, ("E", "P"): 93,
    ("E", "S"): 80,  ("E", "T"): 65,  ("E", "W"): 152, ("E", "Y"): 122,
    ("E", "V"): 121,
    ("G", "H"): 98,  ("G", "I"): 135, ("G", "L"): 138, ("G", "K"): 127,
    ("G", "M"): 127, ("G", "F"): 153, ("G", "P"): 42,  ("G", "S"): 56,
    ("G", "T"): 59,  ("G", "W"): 184, ("G", "Y"): 147, ("G", "V"): 109,
    ("H", "I"): 94,  ("H", "L"): 99,  ("H", "K"): 32,  ("H", "M"): 87,
    ("H", "F"): 100, ("H", "P"): 77,  ("H", "S"): 89,  ("H", "T"): 47,
    ("H", "W"): 115, ("H", "Y"): 83,  ("H", "V"): 84,
    ("I", "L"): 5,   ("I", "K"): 102, ("I", "M"): 10,  ("I", "F"): 21,
    ("I", "P"): 95,  ("I", "S"): 142, ("I", "T"): 89,  ("I", "W"): 61,
    ("I", "Y"): 33,  ("I", "V"): 29,
    ("L", "K"): 107, ("L", "M"): 15,  ("L", "F"): 22,  ("L", "P"): 98,
    ("L", "S"): 145, ("L", "T"): 92,  ("L", "W"): 61,  ("L", "Y"): 36,
    ("L", "V"): 32,
    ("K", "M"): 95,  ("K", "F"): 102, ("K", "P"): 103, ("K", "S"): 121,
    ("K", "T"): 78,  ("K", "W"): 110, ("K", "Y"): 85,  ("K", "V"): 97,
    ("M", "F"): 28,  ("M", "P"): 87,  ("M", "S"): 135, ("M", "T"): 81,
    ("M", "W"): 67,  ("M", "Y"): 36,  ("M", "V"): 21,
    ("F", "P"): 114, ("F", "S"): 155, ("F", "T"): 103, ("F", "W"): 40,
    ("F", "Y"): 22,  ("F", "V"): 50,
    ("P", "S"): 74,  ("P", "T"): 38,  ("P", "W"): 147, ("P", "Y"): 110,
    ("P", "V"): 68,
    ("S", "T"): 58,  ("S", "W"): 177, ("S", "Y"): 144, ("S", "V"): 124,
    ("T", "W"): 128, ("T", "Y"): 92,  ("T", "V"): 69,
    ("W", "Y"): 37,  ("W", "V"): 88,
    ("Y", "V"): 55,
}

# Max Grantham distance for normalization
_GRANTHAM_MAX = 215.0  # C-W


def get_grantham_distance(aa1: str, aa2: str) -> float:
    """Get the Grantham distance between two amino acids.

    Parameters
    ----------
    aa1, aa2 : str
        Single-letter amino acid codes.

    Returns
    -------
    float
        Grantham distance (0 for identical AAs, 0-215 for substitutions).
    """
    aa1, aa2 = aa1.upper(), aa2.upper()
    if aa1 == aa2:
        return 0.0
    key = (aa1, aa2) if (aa1, aa2) in _GRANTHAM_DISTANCES else (aa2, aa1)
    return float(_GRANTHAM_DISTANCES.get(key, 0.0))


# =============================================================================
# COMBINED EXTRACTION
# =============================================================================

def extract_substitution_features(wt_aa: str, mt_aa: str) -> np.ndarray:
    """
    Extract substitution-based features.

    Returns 3 features:
    - BLOSUM62 raw score
    - BLOSUM62 normalized [0, 1]
    - Grantham distance normalized [0, 1]

    Parameters
    ----------
    wt_aa, mt_aa : str
        Wildtype and mutant amino acid single-letter codes.

    Returns
    -------
    np.ndarray
        3-dimensional feature vector.
    """
    blosum_raw = get_blosum_score(wt_aa, mt_aa)
    blosum_norm = get_blosum_normalized(wt_aa, mt_aa)
    grantham_norm = get_grantham_distance(wt_aa, mt_aa) / _GRANTHAM_MAX
    return np.array([blosum_raw, blosum_norm, grantham_norm])


def get_substitution_feature_names() -> list[str]:
    """Get the names of substitution features."""
    return ["blosum62_raw", "blosum62_normalized", "grantham_normalized"]
