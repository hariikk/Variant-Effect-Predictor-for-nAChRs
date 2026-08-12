"""
Placeholder for ESM-2 protein language model embeddings.

Future: extract per-residue embeddings from ESM-2 and use variant-position
embeddings as features. Requires facebook/esm library and pre-trained model weights.

Architecture ready — register in orchestrator when implemented.
"""

import numpy as np
import pandas as pd

from vep_nachr2.features.base import FeatureExtractor


class EmbeddingExtractor(FeatureExtractor):
    """
    Placeholder extractor for ESM-2 embeddings.

    When implemented, will:
    1. Load ESM-2 model (esm2_t33_650M_UR50D or esm2_t6_8M_UR50D)
    2. Run each wildtype sequence through the model
    3. Extract per-position embeddings (layer 33 or final)
    4. Return embedding vector at variant position

    For now, returns empty feature matrix (0 features).
    """

    name = "embeddings"
    n_features = 0  # 0 until implemented
    feature_names = []  # Will be ['esm2_dim_0', ..., 'esm2_dim_N'] when implemented

    def requires_pdb(self) -> bool:
        return False

    def requires_reference(self) -> bool:
        return True  # Needs wildtype sequences for embedding

    def extract(self, df, ref_seqs=None, pdb_resources=None):
        """Placeholder: returns empty feature matrix."""
        n = len(df)
        return np.zeros((n, 0), dtype=np.float64)
