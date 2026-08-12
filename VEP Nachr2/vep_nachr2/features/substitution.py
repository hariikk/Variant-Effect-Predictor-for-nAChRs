"""
Substitution-based feature extraction.

Extracts 3 features:
  - blosum_score: Raw BLOSUM62 substitution score
  - blosum_normalized: Min-max normalized BLOSUM62 score [0, 1]
  - grantham_distance: Grantham physicochemical distance (Grantham, 1974)
"""

import numpy as np
import pandas as pd
from Bio.Align import substitution_matrices

from vep_nachr2.features.base import FeatureExtractor


# =============================================================================
# BLOSUM62
# =============================================================================

BLOSUM62 = substitution_matrices.load("BLOSUM62")

_STANDARD_AAS = list("ACDEFGHIKLMNPQRSTVWY")

_BLOSUM_MIN = float(min(
    BLOSUM62[a, b] for a in _STANDARD_AAS for b in _STANDARD_AAS
))
_BLOSUM_MAX = float(max(
    BLOSUM62[a, b] for a in _STANDARD_AAS for b in _STANDARD_AAS
))


def get_blosum_score(aa1: str, aa2: str) -> int:
    """Get raw BLOSUM62 substitution score."""
    aa1, aa2 = aa1.upper(), aa2.upper()
    try:
        return int(BLOSUM62[aa1, aa2])
    except (KeyError, IndexError):
        return 0


def get_blosum_normalized(aa1: str, aa2: str) -> float:
    """Get BLOSUM62 score normalized to [0, 1]."""
    score = get_blosum_score(aa1, aa2)
    denom = _BLOSUM_MAX - _BLOSUM_MIN
    if denom == 0:
        return 0.5
    return (score - _BLOSUM_MIN) / denom


# =============================================================================
# GRANTHAM DISTANCE
# =============================================================================

# Grantham (1974) distance = sqrt(αΔc² + βΔp² + γΔv²)
# where c=composition, p=polarity, v=volume
# α=1.833, β=0.1018, γ=0.000399

_GRANTHAM_COMPOSITION = {
    "A": 0.00, "R": 0.65, "N": 0.56, "D": 0.27, "C": 2.75,
    "Q": 1.84, "E": 1.42, "G": 0.00, "H": 1.03, "I": 0.00,
    "L": 0.00, "K": 1.87, "M": 0.78, "F": 0.00, "P": 0.24,
    "S": 0.20, "T": 0.65, "W": 0.38, "Y": 0.95, "V": 0.00,
}

_GRANTHAM_POLARITY = {
    "A": 8.1, "R": 10.5, "N": 11.6, "D": 13.0, "C": 5.5,
    "Q": 10.5, "E": 12.3, "G": 9.0, "H": 10.4, "I": 5.2,
    "L": 4.9, "K": 11.3, "M": 5.7, "F": 5.2, "P": 8.0,
    "S": 9.2, "T": 8.6, "W": 5.4, "Y": 6.2, "V": 5.9,
}

_GRANTHAM_VOLUME = {
    "A": 31.0, "R": 124.0, "N": 56.0, "D": 54.0, "C": 55.0,
    "Q": 85.0, "E": 83.0, "G": 3.0, "H": 96.0, "I": 111.0,
    "L": 111.0, "K": 119.0, "M": 105.0, "F": 132.0, "P": 32.5,
    "S": 32.0, "T": 61.0, "W": 170.0, "Y": 136.0, "V": 84.0,
}

_GRANTHAM_ALPHA = 1.833
_GRANTHAM_BETA = 0.1018
_GRANTHAM_GAMMA = 0.000399


def get_grantham_distance(aa1: str, aa2: str) -> float:
    """Calculate Grantham physicochemical distance."""
    aa1, aa2 = aa1.upper(), aa2.upper()

    try:
        dc = _GRANTHAM_COMPOSITION[aa1] - _GRANTHAM_COMPOSITION[aa2]
        dp = _GRANTHAM_POLARITY[aa1] - _GRANTHAM_POLARITY[aa2]
        dv = _GRANTHAM_VOLUME[aa1] - _GRANTHAM_VOLUME[aa2]
    except KeyError:
        return 0.0

    return np.sqrt(
        _GRANTHAM_ALPHA * dc**2
        + _GRANTHAM_BETA * dp**2
        + _GRANTHAM_GAMMA * dv**2
    )


# =============================================================================
# SUBSTITUTION EXTRACTOR
# =============================================================================

class SubstitutionExtractor(FeatureExtractor):
    """
    Extracts 3 substitution-based features:
      - blosum_score: Raw BLOSUM62 score
      - blosum_normalized: Min-max normalized [0, 1]
      - grantham_distance: Grantham physicochemical distance
    """

    name = "substitution"
    n_features = 3
    feature_names = ["blosum_score", "blosum_normalized", "grantham_distance"]

    def requires_pdb(self) -> bool:
        return False

    def requires_reference(self) -> bool:
        return False

    def extract(self, df, ref_seqs=None, pdb_resources=None):
        n = len(df)
        features = np.zeros((n, self.n_features), dtype=np.float64)

        for i, (_, row) in enumerate(df.iterrows()):
            wt_aa = str(row.get("wildtype_aa", "X"))
            mt_aa = str(row.get("variant_aa", "X"))

            features[i, 0] = get_blosum_score(wt_aa, mt_aa)
            features[i, 1] = get_blosum_normalized(wt_aa, mt_aa)
            features[i, 2] = get_grantham_distance(wt_aa, mt_aa)

        return features
