"""
Noah Feature Extractor — main orchestrator for nAChR.

Extracts Noah's thesis feature set adapted for nAChR:
- BLOSUM62 substitution score (1)
- Wildtype AA properties (10)
- Mutant AA properties (10)
- Amino acid property changes (10 deltas)
- MSA aligned positions (1)
- Structural features: RSA, cbeta_density, b_factor, ss_dssp (4)
- Unmappable flag (1)

Total: 37 raw features → 31 after dropping structural metadata columns.
"""

import pandas as pd

from vep_nachr.features.noah_features.config import FILL_VALUES, FEATURE_ORDER
from vep_nachr.features.noah_features.aa_features import (
    generate_amino_acid_properties,
    generate_amino_acid_property_changes,
)
from vep_nachr.features.noah_features.sequence_features import generate_matrix_score
from vep_nachr.features.noah_features.position_features import (
    load_uniprot_sequences,
    build_all_wildtype_msa_maps,
    generate_aligned_positions,
)
from vep_nachr.features.noah_features.structural_features import get_structural_features


class NoahFeatureExtractor:
    """
    Extract Noah's thesis feature set adapted for nAChR.

    Features (31 after processing):
    - BLOSUM62 score (1)
    - Wildtype AA properties (10)
    - Mutant AA properties (10)
    - AA property deltas (10)
    - MSA aligned position (1)
    - Structural: rsa, cbeta_density, b_factor, ss_dssp, is_unmappable (5)

    Downsamples to 31 by keeping only the structural features that
    overlap with Noah's original set (dropping hse_up, hse_down,
    dssp_helix/sheet/coil booleans in favor of ss_dssp integer).
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self._msa_maps = None
        self._fitted = False

    def _log(self, message: str) -> None:
        if self.verbose:
            print(f"[NoahFeatureExtractor] {message}")

    def fit(self, df: pd.DataFrame) -> "NoahFeatureExtractor":
        """Fit the extractor (precompute MSA maps and structural resources)."""
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"df is {type(df)}, but expected pd.DataFrame")

        self._log("Loading UniProt sequences...")
        uniprot_seqs = load_uniprot_sequences()

        self._log("Building MSA alignment maps...")
        self._msa_maps = build_all_wildtype_msa_maps(uniprot_seqs)

        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract all features from variant data."""
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"df is {type(df)}, but expected pd.DataFrame")

        if not self._fitted:
            self.fit(df)

        # Handle variant_aa vs mutant_aa column naming
        work_df = df.copy()
        if "variant_aa" in work_df.columns and "mutant_aa" not in work_df.columns:
            work_df["mutant_aa"] = work_df["variant_aa"]
        elif "mutant_aa" in work_df.columns and "variant_aa" not in work_df.columns:
            work_df["variant_aa"] = work_df["mutant_aa"]

        # Ensure wildtype_aa column exists
        if "wildtype_aa" not in work_df.columns:
            raise ValueError("DataFrame must contain 'wildtype_aa' column")

        features = pd.DataFrame(index=df.index)

        # 1. BLOSUM62 substitution score
        self._log("Generating BLOSUM62 scores...")
        features["blosum62_score"] = generate_matrix_score(work_df)

        # 2-3. Wildtype and Variant AA properties (20 features)
        self._log("Generating AA properties...")
        aa_props = generate_amino_acid_properties(work_df)
        features = pd.concat([features, aa_props], axis=1)

        # 4. Property changes (10 delta features)
        self._log("Generating property changes...")
        delta_props = generate_amino_acid_property_changes(work_df)
        features = pd.concat([features, delta_props], axis=1)

        # 5. MSA aligned positions
        self._log("Generating aligned positions...")
        aligned_pos = generate_aligned_positions(df, self._msa_maps)
        features = pd.concat([features, aligned_pos], axis=1)

        # 6. Structural features
        self._log("Extracting structural features...")
        struct_features = get_structural_features(df)
        for col in struct_features.columns:
            features[col] = struct_features[col]

        # Fill missing values
        self._log("Filling missing values...")
        for col in features.columns:
            if col in FILL_VALUES:
                features[col] = features[col].fillna(FILL_VALUES[col])
            else:
                features[col] = features[col].fillna(0.0)

        # Reorder columns if FEATURE_ORDER is defined
        if FEATURE_ORDER:
            existing_cols = [c for c in FEATURE_ORDER if c in features.columns]
            extra_cols = [c for c in features.columns if c not in FEATURE_ORDER]
            features = features[existing_cols + extra_cols]

        self._log(f"Extracted {len(features.columns)} features")
        return features

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(df).transform(df)

    @property
    def feature_names_(self) -> list[str]:
        """Return the list of feature names."""
        return list(FEATURE_ORDER) if FEATURE_ORDER else []
