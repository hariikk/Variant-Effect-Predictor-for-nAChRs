"""
Global configuration for VEP-nAChR experiments.

Centralizes paths, biological constants, feature groups, model settings,
and cross-validation parameters.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import yaml


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
    # AlphaFold-predicted structures (monomeric, single chain A).
    # pLDDT confidence is stored in B-factor field.
    # RSA/C-beta/HSE reflect monomer context (not pentamer assembly).
    # UniProt: Q15822, CIF: AF-Q15822.cif
    "CHRNA2":  {"pdb_id": "AF-Q15822", "chain": "A", "source": "alphafold"},
    # UniProt: P30532, CIF: AF-P30532.cif
    "CHRNA5":  {"pdb_id": "AF-P30532", "chain": "A", "source": "alphafold"},
    # UniProt: Q15825, CIF: AF-Q15825.cif
    "CHRNA6":  {"pdb_id": "AF-Q15825", "chain": "A", "source": "alphafold"},
    # UniProt: Q9UGM1, CIF: AF-Q9UGM1.cif
    "CHRNA9":  {"pdb_id": "AF-Q9UGM1", "chain": "A", "source": "alphafold"},
    # UniProt: Q13002, CIF: AF-Q13002.cif
    "CHRNA10": {"pdb_id": "AF-Q13002", "chain": "A", "source": "alphafold"},
    # UniProt: Q05901, CIF: AF-Q05901.cif
    "CHRNB3":  {"pdb_id": "AF-Q05901", "chain": "A", "source": "alphafold"},
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
        "desc": "PDB-derived structural features (RSA, B-factor, DSSP, C-beta density, half-sphere exposure)",
        "n_features": 8,  # rsa, bfactor, dssp_helix, dssp_sheet, dssp_coil, cbeta_density, hse_up, hse_down
    },
}


# =============================================================================
# SPECIES CONFIGURATION
# =============================================================================

# Supported species and their encoding
SPECIES_LIST = ["human", "mouse", "rat"]
SPECIES_MAPPING = {"human": 0, "mouse": 1, "rat": 2}

# Mouse/rat subunit name mapping (same letters as human for nAChR)
# Subunit names are identical across species for nAChR
SPECIES_TO_FASTA_DIR = {
    "human": "protein_scequences",  # Path relative to SOURCE_DATA_DIR
    "mouse": "protein_scequences_mouse",
    "rat": "protein_scequences_rat",
}


# =============================================================================
# ENCODING CONFIGURATION
# =============================================================================

# Encoding strategies for domain-driven vs data-driven comparison
ENCODING_STRATEGIES = {
    "ordinal": "Ordinal encoding (position + integer-encoded AAs + species)",
    "onehot": "One-hot encoding (position + one-hot AAs + species + subunit)",
    "fullseq": "Full sequence encoding (complete mutant sequence)",
    "engineered": "Engineered features (52 features: AAIndex + structural + substitution + positional)",
    "combined": "Deduplicated concatenation of engineered + Noah's original features",
}

# Which encodings need wildtype sequences
ENCODINGS_REQUIRING_SEQUENCES = {"fullseq", "fullsequence", "full_sequence"}

# Which encodings are domain-driven (use structural/physicochemical knowledge)
DOMAIN_DRIVEN_ENCODINGS = {"engineered", "noah_original", "combined"}

# Which encodings are data-driven (purely from sequence/position)
DATA_DRIVEN_ENCODINGS = {"ordinal", "onehot", "fullseq"}


# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

CORE_MODELS = ["logistic_regression", "svm_rbf", "random_forest", "lightgbm"]
EXTENDED_MODELS = ["svm_linear", "knn", "xgboost", "catboost", "mlp", "gaussian_nb"]
ALL_MODELS = CORE_MODELS + EXTENDED_MODELS

# Model display names
MODEL_DISPLAY_NAMES = {
    "logistic_regression": "Logistic Regression",
    "svm_rbf": "SVM (RBF)",
    "svm_linear": "SVM (Linear)",
    "random_forest": "Random Forest",
    "lightgbm": "LightGBM",
    "knn": "KNN",
    "gaussian_nb": "Gaussian Naïve Bayes",
    "mlp": "Multilayer Perceptron",
    "xgboost": "XGBoost",
    "catboost": "CatBoost",
}


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


# =============================================================================
# FEATURE CACHE
# =============================================================================

FEATURE_CACHE_DIR = RESULTS_DIR / ".cache"
FEATURE_CACHE_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# EXPERIMENT CONFIGURATION (YAML-SUPPORTED)
# =============================================================================

@dataclass
class Config:
    """Main configuration class for nAChR VEP experiments.

    Supports YAML serialization for reproducible experiment tracking.
    """

    # Experiment identification
    experiment_name: str = "default"
    approach: str = "both"  # 'domain_driven', 'data_driven', 'both'

    # Data configuration
    species: list[str] = field(default_factory=lambda: ["human"])
    subunits: list[str] = field(default_factory=lambda: NACHR_SUBUNITS.copy())

    # Model configuration
    models: list[str] = field(default_factory=lambda: CORE_MODELS.copy())
    encodings: list[str] = field(default_factory=lambda: ["engineered"])

    # CV configuration
    cv: CVConfig = field(default_factory=CVConfig)

    # Execution environment
    n_jobs: int = -1
    verbose: int = 1
    use_hpc: bool = False

    # Output
    results_dir: Path = field(default_factory=lambda: RESULTS_DIR)
    save_predictions: bool = True
    save_models: bool = False
    compute_feature_importance: bool = False

    # Feature flags
    include_structural: bool = True
    include_species: bool = True

    # Cache
    use_feature_cache: bool = True
    feature_cache_dir: Path = field(default_factory=lambda: FEATURE_CACHE_DIR)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        """Load configuration from YAML file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)

        # Handle nested dataclasses
        if "cv" in data and isinstance(data["cv"], dict):
            data["cv"] = CVConfig(**data["cv"])
        if "results_dir" in data and isinstance(data["results_dir"], str):
            data["results_dir"] = Path(data["results_dir"])
        if "feature_cache_dir" in data and isinstance(data["feature_cache_dir"], str):
            data["feature_cache_dir"] = Path(data["feature_cache_dir"])

        return cls(**data)

    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to YAML file."""
        data = self.to_dict()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    def to_dict(self) -> dict:
        """Convert config to plain dictionary."""
        return {
            "experiment_name": self.experiment_name,
            "approach": self.approach,
            "species": self.species,
            "subunits": self.subunits,
            "models": self.models,
            "encodings": self.encodings,
            "cv": {
                "n_outer_folds": self.cv.n_outer_folds,
                "n_inner_folds": self.cv.n_inner_folds,
                "seeds": self.cv.seeds,
                "n_trials": self.cv.n_trials,
            },
            "n_jobs": self.n_jobs,
            "verbose": self.verbose,
            "use_hpc": self.use_hpc,
            "results_dir": str(self.results_dir),
            "save_predictions": self.save_predictions,
            "save_models": self.save_models,
            "compute_feature_importance": self.compute_feature_importance,
            "include_structural": self.include_structural,
            "include_species": self.include_species,
            "use_feature_cache": self.use_feature_cache,
        }


# Global config instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get the global configuration instance (lazy init)."""
    global _config
    if _config is None:
        _config = Config()
    return _config


def set_config(config: Config) -> None:
    """Set the global configuration instance."""
    global _config
    _config = config


def load_config(path: str | Path) -> Config:
    """Load and set global configuration from a YAML file."""
    config = Config.from_yaml(path)
    set_config(config)
    return config


def save_config(path: str | Path) -> None:
    """Save current global configuration to YAML."""
    get_config().to_yaml(path)
