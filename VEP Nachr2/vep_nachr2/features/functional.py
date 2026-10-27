"""
Functional (GO-style) gene-annotation features for nAChR subunits.

The positional extractor encodes each gene as a 16-way one-hot — an opaque
identifier. This extractor instead encodes each subunit by its *function*, so a
model can share structure across subunits that are biologically related
(δ/γ/ε muscle complements, the high-Ca2+ α7/α9/α10 set, the muscle set) rather
than memorising 16 disjoint IDs.

Motivated by Func2Seq/Func2Prot (GO-semantic conditioning for cross-family
transfer). Attributes are static, well-established nAChR biology — no new
downloads or aligners.

Features (7):
  func_is_alpha           principal / agonist-binding subunit (α1-7, α9, α10)
  func_is_beta            β subunit (β1-4)
  func_is_muscle_special  δ/γ/ε (the interchangeable muscle complement subunits)
  func_is_muscle          muscle-type (α1, β1, δ, γ, ε) vs neuronal
  func_homomeric          capable of forming functional homopentamers (α7, α9)
  func_high_ca2           high Ca2+ permeability (α7, α9, α10)
  func_gene_length        reference sequence length, normalised [0, 1]
"""

import numpy as np
import pandas as pd

from vep_nachr2.features.base import FeatureExtractor


# Per-subunit functional annotation, ordered as _ATTR_NAMES.
#   (is_alpha, is_beta, is_muscle_special, is_muscle, homomeric, high_ca2)
_FUNCTIONAL_ATTRS = {
    "CHRNA1":  (1, 0, 0, 1, 0, 0),
    "CHRNA2":  (1, 0, 0, 0, 0, 0),
    "CHRNA3":  (1, 0, 0, 0, 0, 0),
    "CHRNA4":  (1, 0, 0, 0, 0, 0),
    "CHRNA5":  (1, 0, 0, 0, 0, 0),
    "CHRNA6":  (1, 0, 0, 0, 0, 0),
    "CHRNA7":  (1, 0, 0, 0, 1, 1),
    "CHRNA9":  (1, 0, 0, 0, 1, 1),
    "CHRNA10": (1, 0, 0, 0, 0, 1),
    "CHRNB1":  (0, 1, 0, 1, 0, 0),
    "CHRNB2":  (0, 1, 0, 0, 0, 0),
    "CHRNB3":  (0, 1, 0, 0, 0, 0),
    "CHRNB4":  (0, 1, 0, 0, 0, 0),
    "CHRND":   (0, 0, 1, 1, 0, 0),
    "CHRNE":   (0, 0, 1, 1, 0, 0),
    "CHRNG":   (0, 0, 1, 1, 0, 0),
}

_ATTR_NAMES = [
    "func_is_alpha",
    "func_is_beta",
    "func_is_muscle_special",
    "func_is_muscle",
    "func_homomeric",
    "func_high_ca2",
]


class FunctionalExtractor(FeatureExtractor):
    """Extracts 7 static functional-annotation features per variant (by subunit)."""

    name = "functional"
    n_features = 7
    feature_names = _ATTR_NAMES + ["func_gene_length"]

    def __init__(self, verbose: bool = False):
        super().__init__(verbose=verbose)

    def requires_pdb(self) -> bool:
        return False

    def requires_reference(self) -> bool:
        return True  # gene length comes from the reference sequence

    def extract(self, df, ref_seqs=None, pdb_resources=None):
        n = len(df)
        features = np.zeros((n, self.n_features), dtype=np.float64)

        ref_seqs = ref_seqs or {}
        max_len = max((len(s) for s in ref_seqs.values() if s), default=1)

        for i, (_, row) in enumerate(df.iterrows()):
            gene = str(row.get("subunit", "")).upper()

            attrs = _FUNCTIONAL_ATTRS.get(gene)
            if attrs is not None:
                features[i, :6] = attrs

            seq = ref_seqs.get(gene)
            if seq:
                features[i, 6] = len(seq) / max(max_len, 1)

        return features
