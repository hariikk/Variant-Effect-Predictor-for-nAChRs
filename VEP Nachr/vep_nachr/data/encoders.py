"""
Encoding strategies for nAChR mutation data.

Provides multiple encoding approaches for the domain-driven vs data-driven
comparison framework:

1. OrdinalEncoder   — 3-4 features (position + integer AAs + optional species)
2. OneHotEncoder     — ~43 features (position + one-hot AAs + species)
3. FullSequenceEncoder — full mutant sequence as integer vector (~500-900 dims)

All encoders follow a sklearn-like API with fit/transform methods.
Adapted from VEP-ENaC for nAChR's 16-subunit, binary-label setting.
"""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import (
    StandardScaler,
    RobustScaler,
    OneHotEncoder as SklearnOneHotEncoder,
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from vep_nachr.config import AMINO_ACIDS, NACHR_SUBUNITS
from vep_nachr.data.loader import load_wildtype_sequences


# =============================================================================
# AMINO ACID CONSTANTS
# =============================================================================

AA_TO_INT: dict[str, int] = {aa: i for i, aa in enumerate(AMINO_ACIDS, start=1)}
INT_TO_AA: dict[int, str] = {i: aa for aa, i in AA_TO_INT.items()}


# =============================================================================
# BASE ENCODER
# =============================================================================

class BaseEncoder(ABC):
    """Abstract base class for all data-driven encoders."""

    def __init__(self, scale: bool = True, scaler_type: str = "robust"):
        self.scale = scale
        self.scaler_type = scaler_type
        self._fitted = False
        self._scaler = None

    @abstractmethod
    def fit(self, df: pd.DataFrame, wildtype_sequences: Optional[dict] = None) -> "BaseEncoder":
        """Fit the encoder to data."""
        pass

    @abstractmethod
    def transform(self, df: pd.DataFrame, wildtype_sequences: Optional[dict] = None) -> np.ndarray:
        """Transform data using the fitted encoder."""
        pass

    def fit_transform(
        self,
        df: pd.DataFrame,
        wildtype_sequences: Optional[dict] = None,
    ) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(df, wildtype_sequences)
        return self.transform(df, wildtype_sequences)

    def _get_scaler(self):
        """Get scaler instance based on configuration."""
        if self.scaler_type == "standard":
            return StandardScaler()
        elif self.scaler_type == "robust":
            return RobustScaler()
        else:
            raise ValueError(f"Unknown scaler type: {self.scaler_type}")

    @property
    @abstractmethod
    def n_features(self) -> int:
        """Number of output features."""
        ...

    @property
    @abstractmethod
    def feature_names(self) -> list[str]:
        """Names of output features."""
        ...


# =============================================================================
# ORDINAL ENCODER
# =============================================================================

class OrdinalEncoder(BaseEncoder):
    """
    Ordinal encoding: position + integer-encoded amino acids.

    Output features (3-4 depending on species column):
    - mutation_position (scaled)
    - wildtype_aa (integer 1-20)
    - variant_aa (integer 1-20)
    - species (0=human, 1=mouse) — only if 'species' column present

    Total: 3-4 features
    """

    def __init__(self, scale: bool = True, scaler_type: str = "robust"):
        super().__init__(scale, scaler_type)
        self._species_mapping: dict[str, int] = {"human": 0, "mouse": 1, "rat": 2}
        self._has_species: bool = False

    def fit(self, df: pd.DataFrame, wildtype_sequences: Optional[dict] = None) -> "OrdinalEncoder":
        """Fit the encoder (learns scaling parameters)."""
        self._has_species = "species" in df.columns
        if self.scale:
            X = self._encode_raw(df)
            self._scaler = self._get_scaler()
            self._scaler.fit(X)
        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame, wildtype_sequences: Optional[dict] = None) -> np.ndarray:
        """Transform data to ordinal encoding."""
        if not self._fitted:
            raise RuntimeError("Encoder not fitted. Call fit() first.")
        X = self._encode_raw(df)
        if self.scale and self._scaler is not None:
            X = self._scaler.transform(X)
        return X

    def _encode_raw(self, df: pd.DataFrame) -> np.ndarray:
        n_samples = len(df)
        n_cols = 4 if self._has_species else 3
        X = np.zeros((n_samples, n_cols), dtype=np.float32)

        # Position: handle both "position" (nAChR) and "mutation_position" (ENaC-style)
        pos_col = "position" if "position" in df.columns else "mutation_position"
        X[:, 0] = df[pos_col].values

        # Wildtype AA
        X[:, 1] = df["wildtype_aa"].apply(lambda x: AA_TO_INT.get(str(x).strip().upper(), 0)).values

        # Variant AA
        X[:, 2] = df["variant_aa"].apply(
            lambda x: AA_TO_INT.get(str(x).strip().upper(), 0) if pd.notna(x) and x != "-" else 0
        ).values

        # Species (if present)
        if self._has_species:
            X[:, 3] = df["species"].str.lower().map(self._species_mapping).fillna(0).values

        return X

    @property
    def n_features(self) -> int:
        return 4 if self._has_species else 3

    @property
    def feature_names(self) -> list[str]:
        names = ["mutation_position", "wildtype_aa", "variant_aa"]
        if self._has_species:
            names.append("species")
        return names


# =============================================================================
# ONE-HOT ENCODER
# =============================================================================

class OneHotEncoder(BaseEncoder):
    """
    One-hot encoding: position + one-hot amino acids + species.

    Output features:
    - mutation_position (scaled): 1
    - wildtype_aa (one-hot): 20
    - variant_aa (one-hot): 20 (+1 for unknown)
    - species (one-hot): 2-3 (depends on data)

    Total: ~43-44 features
    """

    def __init__(
        self,
        scale: bool = True,
        scaler_type: str = "robust",
        include_unknown_aa: bool = True,
    ):
        super().__init__(scale, scaler_type)
        self.include_unknown_aa = include_unknown_aa
        self._aa_encoder: Optional[SklearnOneHotEncoder] = None
        self._species_encoder: Optional[SklearnOneHotEncoder] = None
        self._position_scaler = None
        self._has_species: bool = False

    def fit(self, df: pd.DataFrame, wildtype_sequences: Optional[dict] = None) -> "OneHotEncoder":
        """Fit the encoder."""
        self._has_species = "species" in df.columns

        # Position scaler
        if self.scale:
            pos_col = "position" if "position" in df.columns else "mutation_position"
            self._position_scaler = self._get_scaler()
            self._position_scaler.fit(df[[pos_col]].values)

        # AA encoder
        aa_categories = list(AMINO_ACIDS)
        if self.include_unknown_aa:
            aa_categories = aa_categories + ["-"]

        self._aa_encoder = SklearnOneHotEncoder(
            categories=[aa_categories, aa_categories],
            sparse_output=False,
            handle_unknown="ignore",
        )

        wt_aa = df["wildtype_aa"].fillna("-").values.reshape(-1, 1)
        var_aa = df["variant_aa"].fillna("-").values.reshape(-1, 1)
        aa_data = np.hstack([wt_aa, var_aa])
        self._aa_encoder.fit(aa_data)

        # Species encoder (if present)
        if self._has_species:
            species_list = sorted(df["species"].str.lower().unique())
            self._species_encoder = SklearnOneHotEncoder(
                categories=[species_list],
                sparse_output=False,
                handle_unknown="ignore",
            )
            self._species_encoder.fit(df[["species"]].values)

        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame, wildtype_sequences: Optional[dict] = None) -> np.ndarray:
        """Transform data to one-hot encoding."""
        if not self._fitted:
            raise RuntimeError("Encoder not fitted. Call fit() first.")

        features = []

        # Position
        pos_col = "position" if "position" in df.columns else "mutation_position"
        pos = df[[pos_col]].values
        if self.scale and self._position_scaler is not None:
            pos = self._position_scaler.transform(pos)
        features.append(pos)

        # AAs
        wt_aa = df["wildtype_aa"].fillna("-").values.reshape(-1, 1)
        var_aa = df["variant_aa"].fillna("-").values.reshape(-1, 1)
        aa_data = np.hstack([wt_aa, var_aa])
        aa_encoded = self._aa_encoder.transform(aa_data)
        features.append(aa_encoded)

        # Species
        if self._has_species and self._species_encoder is not None:
            species_encoded = self._species_encoder.transform(df[["species"]].values)
            features.append(species_encoded)

        return np.hstack(features).astype(np.float32)

    @property
    def n_features(self) -> int:
        aa_size = (20 + int(self.include_unknown_aa)) * 2
        species_size = (
            len(self._species_encoder.categories_[0]) if self._species_encoder else 0
        )
        return 1 + aa_size + species_size

    @property
    def feature_names(self) -> list[str]:
        names = ["mutation_position"]
        aa_cats = list(AMINO_ACIDS) + (["-"] if self.include_unknown_aa else [])
        names.extend(f"wt_{a}" for a in aa_cats)
        names.extend(f"mt_{a}" for a in aa_cats)
        if self._species_encoder is not None:
            names.extend(f"species_{s}" for s in self._species_encoder.categories_[0])
        return names


# =============================================================================
# FULL SEQUENCE ENCODER
# =============================================================================

class FullSequenceEncoder(BaseEncoder):
    """
    Full sequence encoding: complete mutant sequence as integers.

    The mutation is applied to the wildtype sequence and the entire
    sequence is encoded as integers (1-20 for amino acids, 0 for padding).

    Output: ~500-900 features depending on subunit sequence length.
    """

    def __init__(
        self,
        scale: bool = False,
        scaler_type: str = "robust",
        max_length: Optional[int] = None,
        pad_value: int = 0,
    ):
        super().__init__(scale, scaler_type)
        self.max_length = max_length
        self.pad_value = pad_value
        self._fitted_max_length: Optional[int] = None

    def fit(self, df: pd.DataFrame, wildtype_sequences: dict) -> "FullSequenceEncoder":
        """Fit the encoder (determines max sequence length from wildtype seqs)."""
        if wildtype_sequences is None:
            raise ValueError("wildtype_sequences required for FullSequenceEncoder")

        # Determine max length from wildtype sequences
        max_len = 0
        for seq in wildtype_sequences.values():
            max_len = max(max_len, len(seq))

        self._fitted_max_length = self.max_length or max_len

        if self.scale:
            X = self._encode_all(df, wildtype_sequences)
            self._scaler = self._get_scaler()
            self._scaler.fit(X)

        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame, wildtype_sequences: dict) -> np.ndarray:
        """Transform data to full sequence encoding."""
        if not self._fitted:
            raise RuntimeError("Encoder not fitted. Call fit() first.")
        if wildtype_sequences is None:
            raise ValueError("wildtype_sequences required for FullSequenceEncoder")

        X = self._encode_all(df, wildtype_sequences)
        if self.scale and self._scaler is not None:
            X = self._scaler.transform(X)
        return X

    def _encode_all(self, df: pd.DataFrame, wildtype_sequences: dict) -> np.ndarray:
        encoded = []
        for _, row in df.iterrows():
            seq = self._apply_mutation(row, wildtype_sequences)
            enc = self._encode_sequence(seq)
            encoded.append(enc)
        return np.array(encoded, dtype=np.float32)

    def _apply_mutation(self, row: pd.Series, wildtype_sequences: dict) -> str:
        """Apply mutation to wildtype sequence."""
        subunit = row.get("subunit") or row.get("protein_subunit")
        if subunit not in wildtype_sequences:
            raise ValueError(f"Unknown subunit: {subunit}")

        sequence = wildtype_sequences[subunit]
        pos_col = "position" if "position" in row.index else "mutation_position"
        position = int(row[pos_col])
        variant_aa = row.get("variant_aa", "")

        if pd.isna(variant_aa):
            variant_aa = ""

        # Apply substitution (1-based indexing)
        if variant_aa:
            sequence = sequence[: position - 1] + variant_aa + sequence[position:]

        return sequence

    def _encode_sequence(self, sequence: str) -> np.ndarray:
        """Encode a single sequence to integers with padding."""
        encoded = [AA_TO_INT.get(aa, 0) for aa in sequence]
        if len(encoded) < self._fitted_max_length:
            encoded.extend([self.pad_value] * (self._fitted_max_length - len(encoded)))
        elif len(encoded) > self._fitted_max_length:
            encoded = encoded[: self._fitted_max_length]
        return np.array(encoded, dtype=np.float32)

    @property
    def n_features(self) -> int:
        if self._fitted_max_length is None:
            raise RuntimeError("Encoder not fitted")
        return self._fitted_max_length

    @property
    def feature_names(self) -> list[str]:
        if self._fitted_max_length is None:
            raise RuntimeError("Encoder not fitted")
        return [f"seq_pos_{i}" for i in range(self._fitted_max_length)]


# =============================================================================
# SUBUNIT ONE-HOT ENCODER (for nAChR's 16 subunits)
# =============================================================================

class SubunitOneHotEncoder:
    """
    Utility: encode nAChR subunit names as one-hot vectors.

    Not a full encoder — used by NachrFeatureEncoder and available
    as a standalone utility for data-driven encodings that need subunit info.
    """

    def __init__(self, subunits: Optional[list[str]] = None):
        self.subunits = subunits or NACHR_SUBUNITS

    def encode(self, df: pd.DataFrame) -> np.ndarray:
        """Encode subunit column to one-hot."""
        subunit_col = "subunit" if "subunit" in df.columns else "protein_subunit"
        oh = np.zeros((len(df), len(self.subunits)))
        for i, (_, row) in enumerate(df.iterrows()):
            subunit = row[subunit_col]
            if subunit in self.subunits:
                oh[i, self.subunits.index(subunit)] = 1.0
        return oh

    @property
    def feature_names(self) -> list[str]:
        return [f"subunit_{s}" for s in self.subunits]


# =============================================================================
# ENCODER FACTORY
# =============================================================================

def get_encoder(
    encoding: str,
    scale: bool = True,
    scaler_type: str = "robust",
    **kwargs,
) -> BaseEncoder:
    """
    Get encoder instance by name.

    Parameters
    ----------
    encoding : str
        Encoding type: 'ordinal', 'onehot', 'fullseq', 'engineered',
        'noah_original', 'combined'
    scale : bool
        Whether to scale features
    scaler_type : str
        Scaler type: 'standard' or 'robust'
    **kwargs
        Additional encoder-specific arguments

    Returns
    -------
    BaseEncoder or sklearn-compatible encoder
        Encoder instance
    """
    # Normalize encoding name
    enc_lower = encoding.lower().replace("-", "").replace("_", "").replace(" ", "")

    # Data-driven encoders
    data_driven = {
        "ordinal": OrdinalEncoder,
        "onehot": OneHotEncoder,
        "fullseq": FullSequenceEncoder,
        "fullsequence": FullSequenceEncoder,
    }

    if enc_lower in data_driven:
        return data_driven[enc_lower](scale=scale, scaler_type=scaler_type, **kwargs)

    # Domain-driven encoders — lazy imports to avoid circular dependencies
    if enc_lower == "engineered":
        from vep_nachr.features.encoder import NachrFeatureEncoder
        return NachrFeatureEncoder(**kwargs)

    if enc_lower in ("noahoriginal", "noah", "noahoriginal"):
        try:
            from vep_nachr.features.noah_original import NoahOriginalFeatureEncoder
            return NoahOriginalFeatureEncoder(**kwargs)
        except ImportError:
            raise ImportError(
                "NoahOriginalFeatureEncoder not available. "
                "Ensure vep_nachr/features/noah_original.py exists."
            )

    if enc_lower == "combined":
        try:
            from vep_nachr.features.noah_original import CombinedFeatureEncoder
            return CombinedFeatureEncoder(**kwargs)
        except ImportError:
            raise ImportError(
                "CombinedFeatureEncoder not available. "
                "Ensure vep_nachr/features/noah_original.py exists."
            )

    raise ValueError(
        f"Unknown encoding: '{encoding}'. "
        f"Available: {list(data_driven.keys()) + ['engineered', 'noah_original', 'combined']}"
    )
