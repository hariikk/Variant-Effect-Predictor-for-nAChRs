"""
Feature encoder: combines all feature groups into a single feature matrix.

Implements an sklearn-compatible transformer that extracts:
- Physicochemical features (24)
- Substitution scores (3: BLOSUM62 + Grantham)
- Positional features (1 normalized position + 16 subunit one-hot)
- Structural features (8, from PDB or imputed)

Total: 44 features (without structural) or 52 features (with structural).
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from vep_nachr.config import NACHR_SUBUNITS, SPECIES_LIST
from vep_nachr.features.physicochemical import (
    extract_physicochemical_features,
    get_physicochemical_feature_names,
)
from vep_nachr.features.substitution import (
    extract_substitution_features,
    get_substitution_feature_names,
)
from vep_nachr.features.structural import (
    extract_structural_features,
    get_structural_feature_names,
    structural_features_available,
)


class NachrFeatureEncoder(BaseEstimator, TransformerMixin):
    """
    Sklearn-compatible feature encoder for nAChR mutations.

    Encodes a mutation DataFrame into a numerical feature matrix
    suitable for ML models.

    Parameters
    ----------
    include_structural : bool
        Whether to include structural features (requires PDB files).
    include_substitution : bool
        Whether to include BLOSUM62 + Grantham features.
    include_positional : bool
        Whether to include position + subunit encoding.
    include_species : bool
        Whether to include species one-hot features (auto-detected if 'species'
        column is present in the DataFrame).
    """

    def __init__(
        self,
        include_structural: bool = True,
        include_substitution: bool = True,
        include_positional: bool = True,
        include_species: bool = True,
    ):
        self.include_structural = include_structural
        self.include_substitution = include_substitution
        self.include_positional = include_positional
        self.include_species = include_species

        # Set during fit
        self.max_position_ = 700
        self.feature_names_ = None
        self.n_features_ = None
        self.subunit_list_ = NACHR_SUBUNITS
        self._has_species: bool = False
        self._species_list: list[str] = SPECIES_LIST

    def fit(self, X: pd.DataFrame, y=None) -> "NachrFeatureEncoder":
        """
        Fit the encoder (learn max position for normalization).

        Parameters
        ----------
        X : pd.DataFrame
            Must have columns: wildtype_aa, variant_aa, position, subunit.
            Optional: species (auto-detected if present).
        """
        if "position" in X.columns:
            self.max_position_ = X["position"].max()

        # Auto-detect species column
        self._has_species = (
            self.include_species
            and "species" in X.columns
            and X["species"].nunique() > 0
        )
        if self._has_species:
            # Build species list from data
            self._species_list = sorted(X["species"].str.lower().unique())

        self._compute_feature_names()
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform mutation data into feature matrix.

        Parameters
        ----------
        X : pd.DataFrame
            Mutation data with required columns.

        Returns
        -------
        np.ndarray
            Feature matrix (n_samples, n_features).
        """
        features_list = []

        # --- Physicochemical (per-row) ---
        physchem = np.array([
            extract_physicochemical_features(row["wildtype_aa"], row["variant_aa"])
            for _, row in X.iterrows()
        ])
        features_list.append(physchem)

        # --- Substitution scores (per-row) ---
        if self.include_substitution:
            subst = np.array([
                extract_substitution_features(row["wildtype_aa"], row["variant_aa"])
                for _, row in X.iterrows()
            ])
            features_list.append(subst)

        # --- Positional features ---
        if self.include_positional:
            # Normalized position
            pos_norm = (X["position"].values / self.max_position_).reshape(-1, 1)

            # One-hot subunit encoding
            subunit_oh = np.zeros((len(X), len(self.subunit_list_)))
            for i, (_, row) in enumerate(X.iterrows()):
                subunit = row["subunit"]
                if subunit in self.subunit_list_:
                    idx = self.subunit_list_.index(subunit)
                    subunit_oh[i, idx] = 1.0

            features_list.append(pos_norm)
            features_list.append(subunit_oh)

        # --- Structural features (batch) ---
        if self.include_structural:
            struct = extract_structural_features(X)
            features_list.append(struct)

        # --- Species one-hot ---
        if self._has_species:
            species_oh = np.zeros((len(X), len(self._species_list)))
            for i, (_, row) in enumerate(X.iterrows()):
                sp = str(row["species"]).lower()
                if sp in self._species_list:
                    species_oh[i, self._species_list.index(sp)] = 1.0
            features_list.append(species_oh)

        return np.hstack(features_list)

    def fit_transform(self, X: pd.DataFrame, y=None) -> np.ndarray:
        return self.fit(X, y).transform(X)

    def _compute_feature_names(self) -> None:
        """Compute feature names based on configuration."""
        names = get_physicochemical_feature_names()

        if self.include_substitution:
            names.extend(get_substitution_feature_names())

        if self.include_positional:
            names.append("position_normalized")
            names.extend([f"subunit_{s}" for s in self.subunit_list_])

        if self.include_structural:
            names.extend(get_structural_feature_names())

        if self._has_species:
            names.extend([f"species_{s}" for s in self._species_list])

        self.feature_names_ = names
        self.n_features_ = len(names)

    def get_feature_names_out(self, input_features=None) -> list[str]:
        """Get output feature names."""
        if self.feature_names_ is None:
            self._compute_feature_names()
        return self.feature_names_
