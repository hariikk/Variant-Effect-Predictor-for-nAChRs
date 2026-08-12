"""
Physicochemical feature extraction using AAIndex properties.

Extracts 24 features: 8 amino acid properties × {wildtype, variant, delta}.

AAIndex scales used (min-max normalized to [0, 1]):
  - EISD840101: Hydrophobicity (Eisenberg et al., 1984)
  - GRAR740102: Polarity (Grantham, 1974)
  - KRIW790103: Volume (Krigbaum & Komoriya, 1979)
  - FASG760101: Molecular weight (Fasman, 1976)
  - KLEP840101: Net charge (Klein et al., 1984)
  - ZIMJ680104: Isoelectric point (Zimmerman et al., 1968)
  - (binary):    Aromaticity (F/W/Y/H = 1, others = 0)
  - CHOP780201: Secondary structure preference (Chou & Fasman, 1978)
"""

import numpy as np
import pandas as pd
from aaindex import aaindex1

from vep_nachr2.features.base import FeatureExtractor


# =============================================================================
# AA PROPERTY DEFINITIONS
# =============================================================================

AA_PROPERTY_NAMES = [
    "hydrophobicity",              # EISD840101
    "polarity",                    # GRAR740102
    "volume",                      # KRIW790103
    "molecular_weight",            # FASG760101
    "charge",                      # KLEP840101
    "isoelectric_point",           # ZIMJ680104
    "aromaticity",                 # Binary: F, W, Y, H = 1
    "ss_preference",               # CHOP780201 (helix propensity)
]

_AA_PROPERTY_ACCESSIONS: list[str | None] = [
    "EISD840101",  # hydrophobicity
    "GRAR740102",  # polarity
    "KRIW790103",  # volume
    "FASG760101",  # molecular_weight
    "KLEP840101",  # charge
    "ZIMJ680104",  # isoelectric_point
    None,          # aromaticity (computed locally)
    "CHOP780201",  # ss_preference
]

_STANDARD_AAS: list[str] = list("ACDEFGHIKLMNPQRSTVWY")
_AROMATIC_AAS: frozenset[str] = frozenset("FWYH")


def _build_aa_property_table() -> dict[str, list[float]]:
    """Build AA property lookup table at import time.

    Each scale is min-max normalized to [0, 1] over the 20 standard AAs.
    'X' (unknown) receives 0.5 for every property.
    """
    # Precompute normalized scales
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

    table["X"] = [0.5] * len(AA_PROPERTY_NAMES)
    return table


# Built once, read-only thereafter
AA_PROPERTY_TABLE: dict[str, list[float]] = _build_aa_property_table()


def get_aa_properties(aa: str) -> np.ndarray:
    """Get normalized property vector for a single amino acid."""
    aa = aa.upper()
    if aa in AA_PROPERTY_TABLE:
        return np.array(AA_PROPERTY_TABLE[aa], dtype=np.float64)
    return np.array(AA_PROPERTY_TABLE["X"], dtype=np.float64)


# =============================================================================
# PHYSICOCHEMICAL EXTRACTOR
# =============================================================================

class PhysicochemicalExtractor(FeatureExtractor):
    """
    Extracts 24 AAIndex-based physicochemical features.

    Features (8 properties × 3 views):
      wt_{prop}  — wildtype amino acid property
      mt_{prop}  — variant amino acid property
      diff_{prop} — variant minus wildtype (magnitude + direction of change)
    """

    name = "physicochemical"
    n_features = 24

    def __init__(self):
        super().__init__()
        self.feature_names = []
        for prefix in ["wt", "mt", "diff"]:
            for prop in AA_PROPERTY_NAMES:
                self.feature_names.append(f"{prefix}_{prop}")

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

            wt_props = get_aa_properties(wt_aa)
            mt_props = get_aa_properties(mt_aa)
            diff = mt_props - wt_props

            features[i, 0:8] = wt_props
            features[i, 8:16] = mt_props
            features[i, 16:24] = diff

        return features
