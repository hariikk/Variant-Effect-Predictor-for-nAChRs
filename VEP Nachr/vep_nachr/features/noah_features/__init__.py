"""
Noah's Feature Extraction Package — adapted for nAChR.

Extracts 31 features combining:
- Amino acid properties from AAIndex (10 scales)
- Property changes (variant − wildtype)
- BLOSUM62 evolutionary scores
- Aligned positions from MSA (with identity-map fallback)
- Structural features from multi-PDB structures

The package leverages vep_nachr.features.structural for PDB handling
and is self-contained for AA/sequence/position features.
"""

from vep_nachr.features.noah_features.feature_extractor import NoahFeatureExtractor
from vep_nachr.features.noah_features.config import (
    FILL_VALUES,
    FEATURE_ORDER,
    MAX_ASA,
    SUBUNIT_TO_PDB_CHAIN,
)

__all__ = [
    "NoahFeatureExtractor",
    "FILL_VALUES",
    "FEATURE_ORDER",
    "MAX_ASA",
    "SUBUNIT_TO_PDB_CHAIN",
]
