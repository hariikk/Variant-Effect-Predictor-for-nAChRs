"""
Configuration for Noah's feature extraction adapted for nAChR.

16 subunits, multiple PDB structures (experimental + AlphaFold), and
binary LOF/GOF labels.  Uses PDB_MAPPING from vep_nachr.config.
"""

from pathlib import Path

from vep_nachr.config import (
    PDB_MAPPING,
    STRUCTURE_DIR,
    CANONICAL_ACCESSIONS,
    NACHR_SUBUNITS,
)

# Project paths
_NOAH_DIR = Path(__file__).parent
_VEP_NACHR_ROOT = _NOAH_DIR.parent.parent
DATA_DIR = _VEP_NACHR_ROOT / "data"

# MSA file (if available — generated separately)
WILDTYPE_MSA_FILE = DATA_DIR / "processed" / "wildtype_msa.fasta"

# Default fill values for imputation
FILL_VALUES = {
    "rsa": 1.0,           # Fully solvent-exposed (typical for disordered/loops)
    "cbeta_density": 0.0, # Low packing density (typical for loops)
    "b_factor": 0.0,      # Will be filled with chain median by structural code
}

# Theoretical maximum ASA scale (Tien et al., 2013)
MAX_ASA = {
    "A": 129.0, "R": 274.0, "N": 195.0, "D": 193.0, "C": 167.0,
    "E": 223.0, "Q": 225.0, "G": 104.0, "H": 224.0, "I": 197.0,
    "L": 201.0, "K": 236.0, "M": 224.0, "F": 240.0, "P": 159.0,
    "S": 155.0, "T": 172.0, "W": 285.0, "Y": 263.0, "V": 174.0,
}

# PDB to chain mapping for each subunit — derived from PDB_MAPPING
SUBUNIT_TO_PDB_CHAIN: dict[str, tuple[str, str]] = {
    subunit: (info["pdb_id"], info["chain"])
    for subunit, info in PDB_MAPPING.items()
}

# Feature order for consistent output (31 features for nAChR)
FEATURE_ORDER = [
    # BLOSUM62 substitution score (1)
    "blosum62_score",
    # Wildtype AA properties (10)
    "wt_hydrophobicity",
    "wt_net_charge",
    "wt_molecular_weight",
    "wt_polarity",
    "wt_bulkiness",
    "wt_flexibility",
    "wt_alpha_helix_propensity",
    "wt_beta_sheet_propensity",
    "wt_volume",
    "wt_isoelectric_point",
    # Mutant AA properties (10)
    "mut_hydrophobicity",
    "mut_net_charge",
    "mut_molecular_weight",
    "mut_polarity",
    "mut_bulkiness",
    "mut_flexibility",
    "mut_alpha_helix_propensity",
    "mut_beta_sheet_propensity",
    "mut_volume",
    "mut_isoelectric_point",
    # Property changes (10 deltas)
    "delta_hydrophobicity",
    "delta_net_charge",
    "delta_molecular_weight",
    "delta_polarity",
    "delta_bulkiness",
    "delta_flexibility",
    "delta_alpha_helix_propensity",
    "delta_beta_sheet_propensity",
    "delta_volume",
    "delta_isoelectric_point",
    # Structural features (5)
    "rsa",
    "cbeta_density",
    "b_factor",
    "ss_dssp",
    "is_unmappable",
]
