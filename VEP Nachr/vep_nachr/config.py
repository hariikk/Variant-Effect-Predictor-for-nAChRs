"""
Global configuration for VEP-nAChR experiments.

Centralizes paths, biological constants, feature groups, model settings,
and cross-validation parameters.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# =============================================================================
# PATH CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.absolute()

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
STRUCTURE_DIR = RAW_DATA_DIR / "structure_files"
RESULTS_DIR = PROJECT_ROOT / "results"

# Path to the original database (outside the VEP Nachr folder)
SOURCE_DATA_DIR = PROJECT_ROOT.parent

# Create directories if they don't exist
for _dir in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, STRUCTURE_DIR, RESULTS_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)


# =============================================================================
# RANDOM SEEDS
# =============================================================================

DEFAULT_SEEDS = [42, 123, 456, 789, 1011]
PRIMARY_SEED = 42


# =============================================================================
# BIOLOGICAL CONSTANTS
# =============================================================================

# All human nAChR subunits
NACHR_SUBUNITS = [
    "CHRNA1", "CHRNA2", "CHRNA3", "CHRNA4", "CHRNA5",
    "CHRNA6", "CHRNA7", "CHRNA9", "CHRNA10",
    "CHRNB1", "CHRNB2", "CHRNB3", "CHRNB4",
    "CHRND", "CHRNE", "CHRNG",
]

# Mapping from subunit name to FASTA folder name (lowercase)
SUBUNIT_TO_FASTA_DIR = {
    "CHRNA1": "chrna1", "CHRNA2": "chrna2", "CHRNA3": "chrna3",
    "CHRNA4": "chrna4", "CHRNA5": "chrna5", "CHRNA6": "chrna6",
    "CHRNA7": "chrna7", "CHRNA9": "chrna9", "CHRNA10": "chrna10",
    "CHRNB1": "chrnb1", "CHRNB2": "chrnb2", "CHRNB3": "chrnb3",
    "CHRNB4": "chrnb4", "CHRND": "chrnd", "CHRNE": "chrne",
    "CHRNG": "chrng",
}

# RefSeq accession for the canonical (primary) isoform of each subunit.
# These are the "isoform 1" or primary entries from NCBI.
CANONICAL_ACCESSIONS = {
    "CHRNA1": "NP_000070.1",
    "CHRNA2": "NP_000733.2",
    "CHRNA3": "NP_000734.2",
    "CHRNA4": "NP_000735.1",
    "CHRNA5": "NP_000736.2",
    "CHRNA6": "NP_004189.1",
    "CHRNA7": "NP_000737.1",
    "CHRNA9": "NP_060051.2",
    "CHRNA10": "NP_001289963.1",
    "CHRNB1": "NP_000738.2",
    "CHRNB2": "NP_000739.1",
    "CHRNB3": "NP_000740.1",
    "CHRNB4": "NP_000741.1",
    "CHRND": "NP_000742.1",
    "CHRNE": "NP_000071.1",
    "CHRNG": "NP_005190.4",
}

# PDB structures for structural feature extraction.
# Chain assignments determined by BLOSUM62 sequence alignment to canonical RefSeq.
# CIF files stored in data/raw/structure_files/
PDB_MAPPING = {
    # Muscle-type subunits — human muscle nAChR (PDB 7QKO)
    "CHRNA1": {"pdb_id": "7QKO", "chain": "A"},
    "CHRNB1": {"pdb_id": "7QKO", "chain": "B"},
    "CHRND":  {"pdb_id": "7QKO", "chain": "C"},
    "CHRNE":  {"pdb_id": "7QKO", "chain": "D"},
    "CHRNG":  {"pdb_id": "7QKO", "chain": "E"},
    # Neuronal subunits
    "CHRNA7": {"pdb_id": "7EKI", "chain": "A"},   # Homomeric — all chains identical
    "CHRNA4": {"pdb_id": "6CNJ", "chain": "A"},
    "CHRNB2": {"pdb_id": "6CNJ", "chain": "B"},
    "CHRNA3": {"pdb_id": "6PV7", "chain": "A"},
    "CHRNB4": {"pdb_id": "6PV7", "chain": "B"},
    # Subunits needing AlphaFold (no experimental structure)
    "CHRNA2":  {"pdb_id": None, "chain": None},   # UniProt: Q15822
    "CHRNA5":  {"pdb_id": None, "chain": None},   # UniProt: P30532
    "CHRNA6":  {"pdb_id": None, "chain": None},   # UniProt: Q15825
    "CHRNA9":  {"pdb_id": None, "chain": None},   # UniProt: Q9UGM1
    "CHRNA10": {"pdb_id": None, "chain": None},   # UniProt: Q13002
    "CHRNB3":  {"pdb_id": None, "chain": None},   # UniProt: Q05901
}

# Label mapping: binary classification (LOF vs GOF)
LABEL_MAPPING = {
    "LOF": 0,
    "GOF": 1,
}
LABEL_NAMES = {v: k for k, v in LABEL_MAPPING.items()}

# Standard amino acids
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_INT = {aa: i for i, aa in enumerate(AMINO_ACIDS, start=1)}
INT_TO_AA = {i: aa for aa, i in AA_TO_INT.items()}


# =============================================================================
# COLUMN NAME MAPPING (raw data -> standardized)
# =============================================================================

COLUMN_MAPPING = {
    "nAChR subunit": "subunit",
    "Modification type": "modification_type",
    "AA position": "position",
    "Initial AA": "wildtype_aa",
    "New AA": "variant_aa",
    "Effect": "effect",
    "Measuring Technique": "technique",
    "Pathology": "pathology",
    "Reference(PMID)": "pmid",
}


# =============================================================================
# FEATURE GROUPS (for ablation studies)
# =============================================================================

FEATURE_GROUPS = {
    "physicochemical": {
        "desc": "AA physicochemical properties (wt/variant/delta)",
        "n_features": 24,  # 8 AAIndex scales x {wt, mt, diff}
    },
    "substitution": {
        "desc": "BLOSUM62 substitution score + Grantham distance",
        "n_features": 3,  # blosum_raw, blosum_norm, grantham
    },
    "positional": {
        "desc": "Position and subunit encoding",
        "n_features": 17,  # 1 norm_position + 16 subunit one-hot (15 subunits)
    },
    "structural": {
        "desc": "PDB-derived structural features (RSA, B-factor, DSSP, C-beta density)",
        "n_features": 6,  # rsa, bfactor, dssp_helix, dssp_sheet, dssp_coil, cbeta_density
    },
}


# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

CORE_MODELS = ["logistic_regression", "svm_rbf", "random_forest", "lightgbm"]
EXTENDED_MODELS = ["svm_linear", "knn", "xgboost", "catboost", "mlp", "gaussian_nb"]
ALL_MODELS = CORE_MODELS + EXTENDED_MODELS


# =============================================================================
# CROSS-VALIDATION CONFIGURATION
# =============================================================================

@dataclass
class CVConfig:
    """Configuration for cross-validation."""
    n_outer_folds: int = 5
    n_inner_folds: int = 5
    seeds: list[int] = field(default_factory=lambda: DEFAULT_SEEDS.copy())
    n_trials: int = 50  # Optuna trials per fold

    @property
    def n_seeds(self) -> int:
        return len(self.seeds)
