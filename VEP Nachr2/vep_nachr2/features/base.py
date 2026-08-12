"""
Base class for all feature extractors.

Each extractor is independently testable, cacheable, and can be dropped
for ablation studies by name. The orchestrator discovers and runs them.
"""

from abc import ABC, abstractmethod
from typing import List, Optional
import numpy as np
import pandas as pd


class FeatureExtractor(ABC):
    """
    Abstract base class for feature extractors.

    Subclasses must implement:
    - extract(): compute features from variant DataFrame
    - requires_pdb(): whether PDB structures are needed
    - requires_reference(): whether reference sequences are needed
    """

    # Set by subclasses
    name: str = "base"
    n_features: int = 0
    feature_names: List[str] = []

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    @abstractmethod
    def extract(
        self,
        df: pd.DataFrame,
        ref_seqs: Optional[dict[str, str]] = None,
        pdb_resources: Optional[dict] = None,
    ) -> np.ndarray:
        """
        Extract features from variant DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Standardized variant data with columns:
            species, subunit, position, wildtype_aa, variant_aa, effect.
        ref_seqs : dict, optional
            Reference sequences: {gene: sequence_str}.
        pdb_resources : dict, optional
            Pre-loaded PDB resources (structure, DSSP, etc.).

        Returns
        -------
        np.ndarray
            Feature matrix of shape (n_samples, n_features).
        """
        ...

    @abstractmethod
    def requires_pdb(self) -> bool:
        """Whether this extractor needs PDB structures loaded."""
        ...

    @abstractmethod
    def requires_reference(self) -> bool:
        """Whether this extractor needs reference sequences."""
        ...

    def get_feature_names(self) -> List[str]:
        """Return human-readable feature names."""
        if not self.feature_names:
            return [f"{self.name}_{i}" for i in range(self.n_features)]
        return self.feature_names

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(n_features={self.n_features})"


class CompositeFeatureExtractor(FeatureExtractor):
    """
    A feature extractor that combines multiple sub-extractors.

    Useful for defining feature groups that can themselves be composed
    of smaller extractors (e.g., structural = core_structural + nachr_specific).
    """

    def __init__(self, extractors: List[FeatureExtractor], name: str = "composite"):
        super().__init__()
        self.extractors = extractors
        self.name = name
        self._compute_meta()

    def _compute_meta(self):
        self.n_features = sum(e.n_features for e in self.extractors)
        self.feature_names = []
        for e in self.extractors:
            self.feature_names.extend(e.get_feature_names())

    def extract(self, df, ref_seqs=None, pdb_resources=None):
        parts = []
        for extractor in self.extractors:
            if self.verbose:
                print(f"  Running {extractor.name}...")
            feats = extractor.extract(df, ref_seqs, pdb_resources)
            parts.append(feats)
        return np.hstack(parts)

    def requires_pdb(self) -> bool:
        return any(e.requires_pdb() for e in self.extractors)

    def requires_reference(self) -> bool:
        return any(e.requires_reference() for e in self.extractors)
