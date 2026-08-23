"""
Feature orchestrator: runs all extractors, concatenates outputs, caches results.

This is the main entry point for feature extraction. The orchestrator:
1. Runs each registered FeatureExtractor independently
2. Concatenates feature matrices (n_samples, total_n_features)
3. Caches results to disk for fast re-runs
4. Supports dropping extractors by name (for ablation studies)

Usage:
    orchestrator = FeatureOrchestrator()
    X, feature_names = orchestrator.extract(df, ref_seqs, pdb_resources)

    # Ablation: drop a feature group
    X_no_struct = orchestrator.extract(df, ref_seqs, pdb_resources,
                                       drop_extractors=["structural_core"])
"""

import pickle
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from vep_nachr2.features.base import FeatureExtractor
from vep_nachr2.config import CACHE_DIR, FEATURE_GROUPS


class FeatureOrchestrator:
    """
    Orchestrates feature extraction across all registered extractors.

    Each extractor runs independently and produces a 2D array.
    Results are concatenated horizontally and optionally cached.
    """

    def __init__(
        self,
        extractors: Optional[list[FeatureExtractor]] = None,
        verbose: bool = True,
    ):
        """
        Parameters
        ----------
        extractors : list[FeatureExtractor], optional
            Custom extractor list. If None, uses all default extractors.
        verbose : bool
            Print progress info.
        """
        self.verbose = verbose
        self.extractors = extractors if extractors is not None else self._get_default_extractors()

        # Metadata
        self._feature_names: list[str] = []
        self._n_total_features: int = 0
        self._last_hash: Optional[str] = None

    @staticmethod
    def _get_default_extractors() -> list[FeatureExtractor]:
        """Create the default set of feature extractors."""
        from vep_nachr2.features.physicochemical import PhysicochemicalExtractor
        from vep_nachr2.features.substitution import SubstitutionExtractor
        from vep_nachr2.features.positional import PositionalExtractor
        from vep_nachr2.features.structural import StructuralExtractor
        from vep_nachr2.features.structural_nachr import StructuralNachrExtractor
        from vep_nachr2.features.conformational import ConformationalExtractor
        from vep_nachr2.features.conservation import ConservationExtractor
        from vep_nachr2.features.embeddings import EmbeddingExtractor
        from vep_nachr2.features.alphamissense import AlphamissenseExtractor

        return [
            PhysicochemicalExtractor(),
            SubstitutionExtractor(),
            PositionalExtractor(),
            ConservationExtractor(),
            StructuralExtractor(),
            StructuralNachrExtractor(),
            ConformationalExtractor(),
            EmbeddingExtractor(),
            AlphamissenseExtractor(),
        ]

    def get_extractor_names(self) -> list[str]:
        """Return list of registered extractor names."""
        return [e.name for e in self.extractors]

    def get_feature_names(self) -> list[str]:
        """Return full list of feature names (all extractors concatenated)."""
        if not self._feature_names:
            self._feature_names = []
            for extractor in self.extractors:
                self._feature_names.extend(extractor.get_feature_names())
        return self._feature_names

    @property
    def n_features(self) -> int:
        """Total number of features across all extractors."""
        if not self._n_total_features:
            self._n_total_features = sum(e.n_features for e in self.extractors)
        return self._n_total_features

    def extract(
        self,
        df: pd.DataFrame,
        ref_seqs: Optional[dict[str, str]] = None,
        pdb_resources: Optional[dict] = None,
        drop_extractors: Optional[list[str]] = None,
        use_cache: bool = True,
        force_recompute: bool = False,
    ) -> tuple[np.ndarray, list[str]]:
        """
        Extract all features from a variant DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Standardized variant data.
        ref_seqs : dict, optional
            Reference sequences: {gene: sequence}.
        pdb_resources : dict, optional
            Pre-loaded PDB resources.
        drop_extractors : list[str], optional
            Extractor names to skip (for ablation studies).
        use_cache : bool
            If True, check disk cache before recomputing.
        force_recompute : bool
            If True, bypass cache and recompute.

        Returns
        -------
        X : np.ndarray
            Feature matrix (n_samples, total_n_features).
        feature_names : list[str]
            Human-readable feature names.
        """
        # Check cache
        if use_cache and not force_recompute:
            cache_path = self._get_cache_path(df)
            if cache_path.exists():
                try:
                    with open(cache_path, "rb") as f:
                        cached = pickle.load(f)
                    if self.verbose:
                        print(f"Loaded features from cache: {cache_path}")
                        print(f"  Shape: {cached['X'].shape}")
                        print(f"  Features: {len(cached['feature_names'])}")
                    return cached["X"], cached["feature_names"]
                except Exception as e:
                    warnings.warn(f"Cache load failed: {e}, recomputing...")

        # Determine which extractors to run
        drop_set = set(drop_extractors or [])
        active_extractors = [e for e in self.extractors if e.name not in drop_set]

        if self.verbose:
            print(f"Running {len(active_extractors)} feature extractors "
                  f"(dropped: {list(drop_set) if drop_set else 'none'}):")

        # Load PDB resources if needed
        if pdb_resources is None and any(e.requires_pdb() for e in active_extractors):
            from vep_nachr2.features.structural import load_all_pdb_resources
            if self.verbose:
                print("  Loading PDB resources...")
            pdb_resources = load_all_pdb_resources()
            available = sum(1 for r in pdb_resources.values() if r is not None)
            if self.verbose:
                print(f"  PDB availability: {available}/{len(pdb_resources)} structures loaded")

        # Load reference sequences if needed
        if ref_seqs is None and any(e.requires_reference() for e in active_extractors):
            from vep_nachr2.data.reference import load_all_reference_sequences
            if self.verbose:
                print("  Loading reference sequences...")
            ref_seqs = load_all_reference_sequences("human")

        # Run each extractor
        feature_parts = []
        feature_names = []

        for extractor in active_extractors:
            if self.verbose:
                status = ""
                if extractor.requires_pdb():
                    status += " [PDB]"
                if extractor.requires_reference():
                    status += " [REF]"
                print(f"  -> {extractor.name} ({extractor.n_features} features){status}...", end=" ")

            try:
                feats = extractor.extract(df, ref_seqs, pdb_resources)
                if feats.shape[0] != len(df):
                    warnings.warn(
                        f"{extractor.name}: expected {len(df)} rows, got {feats.shape[0]}"
                    )
                if feats.shape[1] != extractor.n_features:
                    warnings.warn(
                        f"{extractor.name}: expected {extractor.n_features} cols, got {feats.shape[1]}"
                    )

                feature_parts.append(feats)
                feature_names.extend(extractor.get_feature_names())

                if self.verbose:
                    print(f"ok shape={feats.shape}")

            except Exception as e:
                warnings.warn(f"{extractor.name} failed: {e}")
                # Produce zero-filled fallback
                fallback = np.zeros((len(df), extractor.n_features), dtype=np.float64)
                feature_parts.append(fallback)
                feature_names.extend(extractor.get_feature_names())
                if self.verbose:
                    print(f"FALLBACK shape={fallback.shape}")

        # Concatenate
        X = np.hstack(feature_parts) if feature_parts else np.zeros((len(df), 0))
        self._feature_names = feature_names
        self._n_total_features = X.shape[1]

        if self.verbose:
            print(f"\nFinal feature matrix: {X.shape}")
            print(f"Total features: {self._n_total_features}")

        # Cache
        if use_cache:
            cache_path = self._get_cache_path(df)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "wb") as f:
                pickle.dump({"X": X, "feature_names": feature_names}, f)
            if self.verbose:
                print(f"Cached to: {cache_path}")

        return X, feature_names

    def _get_cache_path(self, df: pd.DataFrame) -> Path:
        """Generate cache path from DataFrame hash."""
        import hashlib

        # Hash based on row identities (species + subunit + position + wt + mt)
        hash_str = "|".join(
            df["species"].astype(str) + "|"
            + df["subunit"].astype(str) + "|"
            + df["position"].astype(str) + "|"
            + df["wildtype_aa"].astype(str) + "|"
            + df["variant_aa"].astype(str)
        )
        df_hash = hashlib.sha1(hash_str.encode()).hexdigest()[:12]
        self._last_hash = df_hash

        return CACHE_DIR / f"feature_cache_{df_hash}.pkl"

    def get_feature_group_indices(self) -> dict[str, tuple[int, int]]:
        """Return (start, end) column indices for each extractor.

        Useful for feature-group ablation at the model level.
        """
        indices = {}
        col = 0
        for extractor in self.extractors:
            n = extractor.n_features
            indices[extractor.name] = (col, col + n)
            col += n
        return indices

    def summary(self) -> str:
        """Return a formatted summary of the feature extraction pipeline."""
        lines = ["FeatureOrchestrator Summary", "=" * 35]
        lines.append(f"Total features: {self.n_features}")
        lines.append(f"Extractors: {len(self.extractors)}")
        lines.append("-" * 35)
        for extractor in self.extractors:
            lines.append(
                f"  {extractor.name:30s} {extractor.n_features:4d} features"
                f"  [PDB: {'Y' if extractor.requires_pdb() else 'N'}]"
                f"  [REF: {'Y' if extractor.requires_reference() else 'N'}]"
            )
        lines.append("=" * 35)
        return "\n".join(lines)


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def extract_all_features(
    df: pd.DataFrame,
    drop_extractors: Optional[list[str]] = None,
    verbose: bool = True,
) -> tuple[np.ndarray, list[str]]:
    """
    Convenience function: load data, extract features, return X + names.

    Parameters
    ----------
    df : pd.DataFrame
        Loaded variant data.
    drop_extractors : list[str], optional
        Extractor names to skip.
    verbose : bool
        Print progress.

    Returns
    -------
    X, feature_names
    """
    orchestrator = FeatureOrchestrator(verbose=verbose)
    return orchestrator.extract(df, drop_extractors=drop_extractors)


def clear_feature_cache(verbose: bool = True) -> int:
    """
    Delete all cached feature pickle files.

    Parameters
    ----------
    verbose : bool
        Print progress info.

    Returns
    -------
    int
        Number of cache files deleted.
    """
    from vep_nachr2.config import CACHE_DIR

    cache_dir = Path(CACHE_DIR)
    if not cache_dir.exists():
        if verbose:
            print(f"Cache directory does not exist: {cache_dir}")
        return 0

    deleted = 0
    for cache_file in cache_dir.glob("feature_cache_*.pkl"):
        try:
            cache_file.unlink()
            deleted += 1
            if verbose:
                print(f"  Deleted: {cache_file.name}")
        except OSError as e:
            if verbose:
                print(f"  Failed to delete {cache_file.name}: {e}")

    if verbose:
        print(f"Cleared {deleted} feature cache file(s) from {cache_dir}")

    return deleted
