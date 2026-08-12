"""
Noah's Original Feature Encoder + CombinedFeatureEncoder — adapted for nAChR.

NoahOriginalFeatureEncoder (31 features):
    Noah Plingen's thesis feature set:
    - 10 wildtype AA properties
    - 10 mutant AA properties
    - 10 property deltas
    - 1 BLOSUM62 score
    - 1 MSA aligned position
    - 5 structural features (rsa, cbeta_density, b_factor, ss_dssp, is_unmappable)

    After dropping metadata columns: 31 numeric features.

CombinedFeatureEncoder:
    Deduplicated merge of EngineeredFeatureEncoder (up to 52 features) +
    NoahOriginalFeatureEncoder (31 features) → ~50 unique features after
    removing 18 overlapping eng_ features.

Reference:
    Noah Plingen (2024). "Variant Effect Prediction for ENaC Mutations"
    Master's Thesis, University of Amsterdam
"""

import warnings
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from vep_nachr.features.noah_features import NoahFeatureExtractor


# =============================================================================
# NOAH'S ORIGINAL FEATURE ENCODER
# =============================================================================

class NoahOriginalFeatureEncoder(BaseEstimator, TransformerMixin):
    """
    Sklearn-compatible encoder for Noah's thesis features (31 numeric features).

    Parameters
    ----------
    drop_metadata : bool, default=True
        Drop non-feature columns (ss_mmcif, ss_mmcif_sheet, pathology).

    Attributes
    ----------
    feature_names_ : list[str]
        Names of extracted features
    n_features_ : int
        Number of output features (typically 31)
    """

    def __init__(self, drop_metadata: bool = True):
        self.drop_metadata = drop_metadata
        self.extractor: Optional[NoahFeatureExtractor] = None
        self.feature_names_: Optional[list[str]] = None
        self.n_features_: Optional[int] = None
        self._fitted: bool = False

    def fit(self, X: pd.DataFrame, y=None) -> "NoahOriginalFeatureEncoder":
        """Initialize Noah's feature extractor.

        Parameters
        ----------
        X : pd.DataFrame
            Mutation data with columns: subunit, position, wildtype_aa, variant_aa
            (species column optional — defaults to "human")
        y : ignored
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"X must be pd.DataFrame, got {type(X)}")

        required_cols = ["wildtype_aa"]
        if "variant_aa" not in X.columns and "mutant_aa" not in X.columns:
            raise ValueError("Missing column: 'variant_aa' or 'mutant_aa'")
        missing = [c for c in required_cols if c not in X.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        self.extractor = NoahFeatureExtractor(verbose=False)
        self._fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Extract features using Noah's pipeline.

        Returns
        -------
        np.ndarray
            Feature matrix of shape (n_samples, n_features)
        """
        if not self._fitted:
            raise RuntimeError("Encoder must be fitted before transform. Call fit() first.")
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"X must be pd.DataFrame, got {type(X)}")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            features_df = self.extractor.fit_transform(X.copy())

        # Impute any remaining NaNs
        if features_df.isnull().any().any():
            if "rsa" in features_df.columns:
                features_df["rsa"] = features_df["rsa"].fillna(1.0)
            if "cbeta_density" in features_df.columns:
                features_df["cbeta_density"] = features_df["cbeta_density"].fillna(0.0)
            for col in features_df.columns:
                if features_df[col].isnull().any():
                    if pd.api.types.is_numeric_dtype(features_df[col]):
                        features_df[col] = features_df[col].fillna(0.0)

        # Metadata columns to exclude from features
        metadata_cols = [
            "species", "subunit", "protein_subunit", "position", "mutation_position",
            "wildtype_aa", "variant_aa", "mutant_aa", "effect",
            "modification_type", "technique", "pathology", "pmid",
            "oid", "measuring_technique", "entry_by", "is_correct",
        ]
        # Columns that are already in the DataFrame
        drop_cols = [c for c in metadata_cols if c in features_df.columns]

        if self.drop_metadata:
            extra_drops = ["pathology", "ss_mmcif", "ss_mmcif_sheet"]
            drop_cols.extend([c for c in extra_drops if c in features_df.columns])

        feature_cols = [c for c in features_df.columns if c not in drop_cols]

        self.feature_names_ = feature_cols
        self.n_features_ = len(feature_cols)

        return features_df[feature_cols].values

    def fit_transform(self, X: pd.DataFrame, y=None, **kwargs) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)

    def get_feature_names_out(self, input_features=None) -> list[str]:
        """Get output feature names."""
        if self.feature_names_ is None:
            raise RuntimeError("Encoder not fitted. Call fit() first.")
        return self.feature_names_

    def get_feature_info(self) -> dict:
        """Get feature group breakdown."""
        if not self._fitted or self.feature_names_ is None:
            return {"total_features": 0, "groups": {}, "message": "Not fitted yet."}

        groups = {
            "aa_properties_wildtype": [f for f in self.feature_names_ if f.startswith("wildtype_aa_")],
            "aa_properties_variant": [f for f in self.feature_names_ if f.startswith("variant_aa_")],
            "aa_property_changes": [f for f in self.feature_names_ if f.startswith("aa_") and f.endswith("_change")],
            "evolutionary": [f for f in self.feature_names_ if "blosum" in f],
            "positional": [f for f in self.feature_names_ if "position" in f],
            "structural": [f for f in self.feature_names_ if f in ("rsa", "cbeta_density", "b_factor", "ss_dssp", "is_unmappable")],
        }

        return {
            "total_features": self.n_features_,
            "groups": {k: {"count": len(v), "features": v} for k, v in groups.items() if v},
        }


# =============================================================================
# COMBINED FEATURE ENCODER
# =============================================================================

class CombinedFeatureEncoder(BaseEstimator, TransformerMixin):
    """
    Combined Feature Encoder — deduplicated merge of engineered + Noah features.

    Concatenates:
    - EngineeredFeatureEncoder (up to 52 features)
    - NoahOriginalFeatureEncoder (31 features)

    Then deduplicates by dropping 18 ``eng_`` features that overlap with
    ``noah_`` equivalents (shared AAIndex scales, RSA, B-factor, BLOSUM62).

    Final: ~65 features (52 − 18 + 31), subject to column alignment.
    """

    # 18 eng_ features to drop (duplicated by noah_ equivalents)
    _DUPLICATE_ENG_FEATURES: list[str] = [
        # RSA, B-factor, BLOSUM62 raw
        "eng_rsa",
        "eng_bfactor",
        "eng_blosum_score",
        # 5 shared AA scales × {wt, mt, diff} = 15
        *[f"eng_{prefix}_{prop}"
          for prefix in ("wt", "mt", "diff")
          for prop in ("hydrophobicity", "polarity", "molecular_weight",
                       "charge", "secondary_structure_preference")],
    ]

    def __init__(self):
        from vep_nachr.features.encoder import NachrFeatureEncoder

        self._engineered_encoder = NachrFeatureEncoder()
        self._noah_encoder = NoahOriginalFeatureEncoder()
        self._fitted: bool = False
        self.n_features_: Optional[int] = None
        self.feature_names_: Optional[list[str]] = None

    def fit(self, X: pd.DataFrame, y=None) -> "CombinedFeatureEncoder":
        """Fit both encoders to the data."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"X must be pd.DataFrame, got {type(X)}")

        # Fit both encoders
        _ = self._engineered_encoder.fit_transform(X, y)
        _ = self._noah_encoder.fit_transform(X, y)

        # Build combined feature names with prefixes
        eng_names = [f"eng_{n}" for n in self._engineered_encoder.get_feature_names_out()]
        noah_names = [f"noah_{n}" for n in self._noah_encoder.feature_names_]
        all_names = eng_names + noah_names

        # Drop duplicate eng_ features
        drop_set = set(self._DUPLICATE_ENG_FEATURES)
        self.feature_names_ = [n for n in all_names if n not in drop_set]
        self.n_features_ = len(self.feature_names_)

        self._fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform data using both encoders and deduplicate."""
        if not self._fitted:
            raise RuntimeError("Encoder not fitted. Call fit() first.")
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"X must be pd.DataFrame, got {type(X)}")

        X_eng = self._engineered_encoder.transform(X)
        X_noah = self._noah_encoder.transform(X)
        X_combined = np.hstack([X_eng, X_noah])

        # Build full name list to find column indices to drop
        eng_names = [f"eng_{n}" for n in self._engineered_encoder.get_feature_names_out()]
        noah_names = [f"noah_{n}" for n in self._noah_encoder.feature_names_]
        all_names = eng_names + noah_names

        drop_set = set(self._DUPLICATE_ENG_FEATURES)
        keep_idx = [i for i, n in enumerate(all_names) if n not in drop_set]
        return X_combined[:, keep_idx]

    def fit_transform(self, X: pd.DataFrame, y=None, **kwargs) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)

    def get_feature_names_out(self, input_features=None) -> list[str]:
        """Get output feature names."""
        if self.feature_names_ is None:
            raise RuntimeError("Encoder not fitted. Call fit() first.")
        return self.feature_names_

    def get_feature_info(self) -> dict:
        """Get feature breakdown."""
        if not self._fitted:
            return {"total_features": 0, "message": "Not fitted yet."}

        return {
            "total_features": self.n_features_,
            "deduplicated": True,
            "dropped_duplicates": len(self._DUPLICATE_ENG_FEATURES),
            "engineered_features_kept": sum(1 for n in self.feature_names_ if n.startswith("eng_")),
            "noah_features": sum(1 for n in self.feature_names_ if n.startswith("noah_")),
        }
