"""
nAChR functional-motif membership features.

Five binary flags marking whether a variant residue sits in a functionally
critical region of the receptor. These are *sharp* where the existing
`structural_nachr` features are continuous (e.g. "is this residue IN the
channel gate" vs "how far is it from the pore axis").

Features (5):
  - in_tm           : in any transmembrane helix
  - in_tm2          : in M2 (the pore-lining helix)
  - in_gate_ring    : in the M2 channel-gate constriction (middle of M2)
  - in_cys_loop     : in the extracellular Cys-loop region
  - in_binding_site : in the principal-face agonist-binding site

Annotation-only — no PDB or reference needed. Genes without annotations
degrade to 0.0 (graceful, same convention as structural_nachr).
"""

import numpy as np
import pandas as pd

from vep_nachr2.features.base import FeatureExtractor
from vep_nachr2.features import mechanism as mech


class FunctionalMotifExtractor(FeatureExtractor):
    name = "functional_motif"
    n_features = 5
    feature_names = [
        "in_tm", "in_tm2", "in_gate_ring", "in_cys_loop", "in_binding_site",
    ]

    def requires_pdb(self) -> bool:
        return False

    def requires_reference(self) -> bool:
        return False

    def extract(self, df, ref_seqs=None, pdb_resources=None):
        n = len(df)
        features = np.zeros((n, self.n_features), dtype=np.float64)
        for i, (_, row) in enumerate(df.iterrows()):
            gene = str(row.get("subunit", "")).upper()
            pos = int(row.get("position", 0))
            features[i, 0] = mech.in_tm(gene, pos)
            features[i, 1] = mech.in_tm2(gene, pos)
            features[i, 2] = mech.in_gate_ring(gene, pos)
            features[i, 3] = mech.in_cys_loop(gene, pos)
            features[i, 4] = mech.in_binding_site(gene, pos)
        return features
