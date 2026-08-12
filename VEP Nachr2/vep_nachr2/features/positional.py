"""
Positional and metadata feature extraction.

Extracts:
  - position_normalized: Position divided by gene-specific max position
  - subunit one-hot: 16 binary features (one per nAChR gene)
  - species one-hot: 3 binary features (human/mouse/rat)
"""

import numpy as np
import pandas as pd

from vep_nachr2.features.base import FeatureExtractor
from vep_nachr2.config import NACHR_GENES, SPECIES_LIST


class PositionalExtractor(FeatureExtractor):
    """
    Extracts positional + subunit + species features.

    Features (20 total):
      - position_normalized (1)
      - subunit_{gene} one-hot (16)
      - species_{species} one-hot (3)
    """

    name = "positional"
    n_features = 1 + len(NACHR_GENES) + len(SPECIES_LIST)  # 20

    def __init__(self):
        super().__init__()
        self.feature_names = ["position_normalized"]
        for gene in NACHR_GENES:
            self.feature_names.append(f"subunit_{gene}")
        for sp in SPECIES_LIST:
            self.feature_names.append(f"species_{sp}")

        # Cache gene max positions (set during extract)
        self._gene_max_positions: dict[str, int] = {}

    def requires_pdb(self) -> bool:
        return False

    def requires_reference(self) -> bool:
        return True  # For gene sequence lengths

    def extract(self, df, ref_seqs=None, pdb_resources=None):
        n = len(df)
        features = np.zeros((n, self.n_features), dtype=np.float64)

        # Compute per-gene max positions for normalization
        if ref_seqs:
            self._gene_max_positions = {
                gene: len(seq) for gene, seq in ref_seqs.items()
            }
        else:
            # Fallback: compute from data
            for gene in df["subunit"].unique():
                gene_df = df[df["subunit"] == gene]
                self._gene_max_positions[gene] = gene_df["position"].max()

        # Build subunit index map
        gene_to_idx = {gene: i for i, gene in enumerate(NACHR_GENES)}
        species_to_idx = {sp: i for i, sp in enumerate(SPECIES_LIST)}

        for i, (_, row) in enumerate(df.iterrows()):
            gene = str(row.get("subunit", "")).upper()
            position = int(row.get("position", 0))
            species = str(row.get("species", "")).lower()

            # Position normalized by gene length
            max_pos = self._gene_max_positions.get(gene, position)
            features[i, 0] = position / max(max_pos, 1)

            # Subunit one-hot
            col_offset = 1
            if gene in gene_to_idx:
                features[i, col_offset + gene_to_idx[gene]] = 1.0

            # Species one-hot
            col_offset = 1 + len(NACHR_GENES)
            if species in species_to_idx:
                features[i, col_offset + species_to_idx[species]] = 1.0

        return features
