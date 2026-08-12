"""
Global configuration for VEP-nAChR2 experiments.

Centralizes paths, biological constants (16 nAChR subunits, PDB mapping,
AAIndex properties), feature groups, model settings, and CV parameters.

Modeled after VEP-ENAC's config.py but extended for nAChR's 16-subunit architecture.
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
REFERENCE_SEQ_DIR = RAW_DATA_DIR / "reference_sequences"
RESULTS_DIR = PROJECT_ROOT / "results"
CACHE_DIR = PROCESSED_DATA_DIR

# Source data — relative to repo root (VEP Nachr2 is one level down)
SOURCE_DATA_DIR = PROJECT_ROOT.parent / "merging_data"

for _dir in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, STRUCTURE_DIR,
             REFERENCE_SEQ_DIR, RESULTS_DIR, CACHE_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)


# =============================================================================
# RANDOM SEEDS
# =============================================================================

DEFAULT_SEEDS = [42, 123, 456, 789, 1011]
SPECIES_TRANSFER_SEEDS = [42, 123, 456, 789, 1011, 2024, 3142, 4269, 5555, 6789]
PRIMARY_SEED = 42


# =============================================================================
# BIOLOGICAL CONSTANTS — nAChR
# =============================================================================

# All 16 human nAChR subunit genes
NACHR_GENES = [
    "CHRNA1", "CHRNA2", "CHRNA3", "CHRNA4", "CHRNA5",
    "CHRNA6", "CHRNA7", "CHRNA9", "CHRNA10",
    "CHRNB1", "CHRNB2", "CHRNB3", "CHRNB4",
    "CHRND", "CHRNE", "CHRNG",
]

# Homology classes for cross-family transfer experiments
HOMOLOGY_CLASSES = {
    "alpha": ["CHRNA1", "CHRNA2", "CHRNA3", "CHRNA4", "CHRNA5",
              "CHRNA6", "CHRNA7", "CHRNA9", "CHRNA10"],
    "beta": ["CHRNB1", "CHRNB2", "CHRNB3", "CHRNB4"],
    "special": ["CHRND", "CHRNE", "CHRNG"],  # delta/gamma/epsilon
}

# RefSeq canonical accessions (isoform 1 for each subunit)
CANONICAL_ACCESSIONS = {
    "CHRNA1":  "NP_000070.1",
    "CHRNA2":  "NP_000733.2",
    "CHRNA3":  "NP_000734.2",
    "CHRNA4":  "NP_000735.1",
    "CHRNA5":  "NP_000736.2",
    "CHRNA6":  "NP_004189.1",
    "CHRNA7":  "NP_000737.1",
    "CHRNA9":  "NP_060051.2",
    "CHRNA10": "NP_001289963.1",
    "CHRNB1":  "NP_000738.2",
    "CHRNB2":  "NP_000739.1",
    "CHRNB3":  "NP_000740.1",
    "CHRNB4":  "NP_000741.1",
    "CHRND":   "NP_000742.1",
    "CHRNE":   "NP_000071.1",
    "CHRNG":   "NP_005190.4",
}

# =============================================================================
# PDB STRUCTURE MAPPING
# =============================================================================
# One representative experimental PDB per receptor type.
# AlphaFold structures fill gaps for subunits without dedicated experimental structures.
# Each entry: {pdb_id, chain, source ("experimental"|"alphafold"), homology_parent (optional)}

PDB_MAPPING = {
    # ── Muscle-type (9DMG: human α1β1δε apo, 2.05 Å) ──
    # Chain assignment: A/C=α1, B=ε, D=δ, E=β1
    "CHRNA1": {"pdb_id": "9DMG", "chain": "A", "source": "experimental"},
    "CHRNB1": {"pdb_id": "9DMG", "chain": "E", "source": "experimental"},
    "CHRND":  {"pdb_id": "9DMG", "chain": "D", "source": "experimental"},
    "CHRNE":  {"pdb_id": "9DMG", "chain": "B", "source": "experimental"},
    "CHRNG":  {"pdb_id": "9DMG", "chain": "B", "source": "experimental",
               "homology_parent": "CHRNE"},  # fetal γ ≈ adult ε chain

    # ── α7 homopentamer (7EKT: closed/antagonist, 3.20 Å) ──
    "CHRNA7": {"pdb_id": "7EKT", "chain": "A", "source": "experimental"},

    # ── α4β2 neuronal (6CNJ: 3.30 Å) ──
    "CHRNA4": {"pdb_id": "6CNJ", "chain": "A", "source": "experimental"},
    "CHRNB2": {"pdb_id": "6CNJ", "chain": "B", "source": "experimental"},
    "CHRNA2": {"pdb_id": "6CNJ", "chain": "A", "source": "experimental",
               "homology_parent": "CHRNA4"},  # α2 ≈ α4
    "CHRNA6": {"pdb_id": "6CNJ", "chain": "A", "source": "experimental",
               "homology_parent": "CHRNA4"},  # α6 ≈ α4

    # ── α3β4 neuronal (6PV7: 2.80 Å) ──
    "CHRNA3": {"pdb_id": "6PV7", "chain": "A", "source": "experimental"},
    "CHRNB4": {"pdb_id": "6PV7", "chain": "B", "source": "experimental"},
    "CHRNA5": {"pdb_id": "6PV7", "chain": "A", "source": "experimental",
               "homology_parent": "CHRNA3"},  # α5 ≈ α3
    "CHRNB3": {"pdb_id": "6PV7", "chain": "B", "source": "experimental",
               "homology_parent": "CHRNB4"},  # β3 ≈ β4

    # ── α9α10 (AlphaFold fallback — no high-res experimental structure) ──
    "CHRNA9":  {"pdb_id": "AF-Q9UGM1", "chain": "A", "source": "alphafold"},
    "CHRNA10": {"pdb_id": "AF-Q13002", "chain": "A", "source": "alphafold"},
}

# Conformational state pairs (closed/resting → open/activated)
# Used by ConformationalExtractor for delta features.
# Only α7 has both states experimentally; architecture ready for others.
CONFORMATIONAL_PAIRS = {
    "CHRNA7": {"closed": "7EKT", "open": "7KOX"},
}

# Unique PDB IDs to download (deduplicated from PDB_MAPPING)
REQUIRED_PDB_IDS = sorted(set(
    info["pdb_id"] for info in PDB_MAPPING.values()
    if info["source"] == "experimental"
))
# → ['6CNJ', '6PV7', '7EKT', '9DMG']

REQUIRED_ALPHAFOLD_IDS = sorted(set(
    info["pdb_id"] for info in PDB_MAPPING.values()
    if info["source"] == "alphafold"
))
# → ['AF-Q13002', 'AF-Q9UGM1']

# Additional: open/activated-state α7 for conformational delta
CONFORMATIONAL_PDB_IDS = ["7KOX"]


# =============================================================================
# LABEL CONFIGURATION
# =============================================================================

# 3-class label mapping (sklearn-compatible: 0, 1, 2)
LABEL_MAPPING = {
    "LOF": 0,
    "loss-of-function": 0,
    "No net effect": 1,
    "no-net-effect": 1,
    "no net effect": 1,
    "GOF": 2,
    "gain-of-function": 2,
}
LABEL_NAMES = {0: "LOF", 1: "No net effect", 2: "GOF"}
LABEL_ORDER = ["LOF", "No net effect", "GOF"]

# Effects to EXCLUDE (ambiguous dual-effect labels)
EXCLUDED_EFFECTS = {"LOF/GOF", "lof/gof", "GOF/LOF", "gof/lof"}

# Modification types to INCLUDE (substitutions only for clean feature space)
INCLUDED_MOD_TYPES = {"Substitution", "substitution"}


# =============================================================================
# AMINO ACID CONSTANTS
# =============================================================================

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_INT = {aa: i for i, aa in enumerate(AMINO_ACIDS, start=1)}
INT_TO_AA = {i: aa for aa, i in AA_TO_INT.items()}

# Maximum ASA scale (Tien et al., 2013) — for RSA computation
MAX_ASA = {
    "A": 129.0, "R": 274.0, "N": 195.0, "D": 193.0, "C": 167.0,
    "E": 223.0, "Q": 225.0, "G": 104.0, "H": 224.0, "I": 197.0,
    "L": 201.0, "K": 236.0, "M": 224.0, "F": 240.0, "P": 159.0,
    "S": 155.0, "T": 172.0, "W": 285.0, "Y": 263.0, "V": 174.0,
}


# =============================================================================
# FEATURE GROUPS (for ablation studies)
# =============================================================================

@dataclass
class FeatureGroup:
    """Definition of a feature group for ablation tracking."""
    name: str
    description: str
    extractor_class: str  # module path for dynamic import
    n_features_expected: int
    requires_pdb: bool = False
    requires_reference: bool = False


FEATURE_GROUPS = {
    "physicochemical": FeatureGroup(
        name="physicochemical",
        description="AAIndex physicochemical properties (8 props × {wt, mt, diff})",
        extractor_class="PhysicochemicalExtractor",
        n_features_expected=24,
    ),
    "substitution": FeatureGroup(
        name="substitution",
        description="BLOSUM62 substitution score + Grantham distance",
        extractor_class="SubstitutionExtractor",
        n_features_expected=3,
    ),
    "positional": FeatureGroup(
        name="positional",
        description="Normalized position + subunit one-hot encoding (16 subunits) + species one-hot (3 species)",
        extractor_class="PositionalExtractor",
        n_features_expected=20,  # 1 position + 16 subunit OH + 3 species OH
    ),
    "structural_core": FeatureGroup(
        name="structural_core",
        description="Core PDB structural features (RSA, Cβ-density, B-factor, DSSP, is_unmappable)",
        extractor_class="StructuralExtractor",
        n_features_expected=5,
        requires_pdb=True,
    ),
    "structural_nachr": FeatureGroup(
        name="structural_nachr",
        description="nAChR-specific structural features (TMD helix, ligand proximity, interface proximity, pore distance)",
        extractor_class="StructuralNachrExtractor",
        n_features_expected=7,
        requires_pdb=True,
    ),
    "conformational": FeatureGroup(
        name="conformational",
        description="Open/closed conformational delta features (α7 only initially)",
        extractor_class="ConformationalExtractor",
        n_features_expected=0,  # 0 for non-α7, up to 5 for α7
        requires_pdb=True,
    ),
    "embeddings": FeatureGroup(
        name="embeddings",
        description="ESM-2 protein language model embeddings (future)",
        extractor_class="EmbeddingExtractor",
        n_features_expected=0,  # placeholder
    ),
}

# Fill values for missing/imputed structural features
STRUCTURAL_FILL_VALUES = {
    "rsa": 1.0,            # Fully solvent-exposed (disordered default)
    "cbeta_density": 0,     # No local packing (disordered default)
    "b_factor": 0.0,        # Will be replaced by chain median
    "ss_dssp": "-",         # Coil/loop (disordered default)
    "is_unmappable": 1,     # Unmapped until pseudo-resolved
}


# =============================================================================
# SPECIES CONFIGURATION
# =============================================================================

SPECIES_LIST = ["human", "mouse", "rat"]
SPECIES_MAPPING = {"human": 0, "mouse": 1, "rat": 2}


# =============================================================================
# COLUMN NAME MAPPING (raw → standardized)
# =============================================================================

# Maps raw column names from final.xlsx to internal standard names
COLUMN_MAPPING = {
    "Species": "species",
    "nAChR subunit": "subunit",
    "Modification type": "modification_type",
    "AA position": "position",
    "Initial AA": "wildtype_aa",
    "New AA": "variant_aa",
    "Effect": "effect",
    "Measuring Technique": "technique",
    "Pathology": "pathology",
    "Reference(PMID)": "pmid",
    "DOI": "doi",
    "Entry by": "entry_by",
    "Correct?": "is_correct",
    "review_flag": "review_flag",
    "dedup_note": "dedup_note",
}

REQUIRED_COLUMNS = [
    "species", "subunit", "position", "wildtype_aa", "variant_aa", "effect"
]


# =============================================================================
# ENCODING CONFIGURATION
# =============================================================================

ENCODING_STRATEGIES = {
    "ordinal": "Ordinal encoding (position + integer-encoded AAs + species + subunit)",
    "onehot": "One-hot encoding (position + one-hot AAs + species + subunit)",
    "engineered": "Engineered features (AAIndex + BLOSUM + structural + positional)",
    "combined": "Engineered + nAChR-specific structural extensions",
}

DOMAIN_DRIVEN_ENCODINGS = {"engineered", "combined"}
DATA_DRIVEN_ENCODINGS = {"ordinal", "onehot"}


# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

CORE_MODELS = ["logistic_regression", "svm_rbf", "random_forest", "lightgbm"]
EXTENDED_MODELS = ["svm_linear", "knn", "xgboost", "catboost", "mlp", "gaussian_nb"]
ALL_MODELS = CORE_MODELS + EXTENDED_MODELS

MODEL_DISPLAY_NAMES = {
    "logistic_regression": "Logistic Regression",
    "svm_rbf": "SVM (RBF)",
    "svm_linear": "SVM (Linear)",
    "random_forest": "Random Forest",
    "lightgbm": "LightGBM",
    "knn": "K-Nearest Neighbors",
    "gaussian_nb": "Gaussian Naïve Bayes",
    "mlp": "Multi-Layer Perceptron",
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
    stratified: bool = True
    shuffle: bool = True

    # CV mode: "subunit" (leave-one-gene-out), "homology" (cross-family), or "standard" (stratified K-fold)
    mode: str = "subunit"

    @property
    def n_seeds(self) -> int:
        return len(self.seeds)


@dataclass
class HyperoptConfig:
    """Configuration for hyperparameter optimization."""
    n_trials: int = 50
    timeout: Optional[int] = None
    sampler: str = "tpe"
    pruner: str = "median"
    n_jobs: int = 1


# =============================================================================
# EXPERIMENT CONFIGURATION
# =============================================================================

@dataclass
class Config:
    """Main configuration class for nAChR VEP experiments.

    Supports YAML serialization for reproducible experiment tracking.
    """

    # Experiment identification
    experiment_name: str = "default"

    # Data configuration
    species: list[str] = field(default_factory=lambda: ["human"])
    subunits: list[str] = field(default_factory=lambda: NACHR_GENES.copy())
    effects: list[str] = field(default_factory=lambda: ["LOF", "No net effect", "GOF"])

    # Model & encoding
    models: list[str] = field(default_factory=lambda: CORE_MODELS.copy())
    encodings: list[str] = field(default_factory=lambda: ["engineered"])

    # CV & hyperopt
    cv: CVConfig = field(default_factory=CVConfig)
    hyperopt: HyperoptConfig = field(default_factory=HyperoptConfig)

    # Execution
    n_jobs: int = -1
    verbose: int = 1

    # Output
    results_dir: Path = field(default_factory=lambda: RESULTS_DIR)
    save_predictions: bool = True
    save_models: bool = False
    compute_feature_importance: bool = True

    # Feature flags
    include_structural: bool = True
    include_nachr_specific: bool = True
    include_conformational: bool = False  # α7 only for now

    # Cache
    use_feature_cache: bool = True
    feature_cache_path: Path = field(
        default_factory=lambda: CACHE_DIR / "feature_cache.pkl"
    )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        """Load configuration from YAML file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)

        if "cv" in data and isinstance(data["cv"], dict):
            data["cv"] = CVConfig(**data["cv"])
        if "hyperopt" in data and isinstance(data["hyperopt"], dict):
            data["hyperopt"] = HyperoptConfig(**data["hyperopt"])
        for path_field in ["results_dir", "feature_cache_path"]:
            if path_field in data and isinstance(data[path_field], str):
                data[path_field] = Path(data[path_field])

        return cls(**data)

    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    def to_dict(self) -> dict:
        """Convert config to plain dictionary for serialization."""
        return {
            "experiment_name": self.experiment_name,
            "species": self.species,
            "subunits": self.subunits,
            "effects": self.effects,
            "models": self.models,
            "encodings": self.encodings,
            "cv": {
                "n_outer_folds": self.cv.n_outer_folds,
                "n_inner_folds": self.cv.n_inner_folds,
                "seeds": self.cv.seeds,
                "stratified": self.cv.stratified,
                "shuffle": self.cv.shuffle,
                "mode": self.cv.mode,
            },
            "hyperopt": {
                "n_trials": self.hyperopt.n_trials,
                "timeout": self.hyperopt.timeout,
                "sampler": self.hyperopt.sampler,
                "pruner": self.hyperopt.pruner,
                "n_jobs": self.hyperopt.n_jobs,
            },
            "n_jobs": self.n_jobs,
            "verbose": self.verbose,
            "results_dir": str(self.results_dir),
            "save_predictions": self.save_predictions,
            "save_models": self.save_models,
            "compute_feature_importance": self.compute_feature_importance,
            "include_structural": self.include_structural,
            "include_nachr_specific": self.include_nachr_specific,
            "include_conformational": self.include_conformational,
            "use_feature_cache": self.use_feature_cache,
        }


# =============================================================================
# GLOBAL CONFIG INSTANCE (singleton pattern)
# =============================================================================

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
