"""
Signed charge-change × location features.

GOF vs LOF in ligand-gated ion channels often depends on *where* a charge
change lands: a charge gain/loss in the pore (M2) or the binding pocket has a
different functional consequence than the same change on the solvent-exposed
surface. These features capture that interaction.

Features (4):
  - charge_change            : net charge(mutant) - net charge(wildtype)
  - charge_change_in_tm      : charge_change × in_tm
  - charge_change_in_tm2     : charge_change × in_tm2 (pore-lining)
  - charge_change_in_binding : charge_change × in_binding_site

Charge is a simplified net charge at pH ~7.4 (D/E = -1, K/R = +1, else 0),
distinct from the continuous KLEP840101 charge already in `physicochemical`.
Annotation-only — no PDB or reference needed.
"""

import numpy as np
import pandas as pd

from vep_nachr2.features.base import FeatureExtractor
from vep_nachr2.features import mechanism as mech


class ChargeContextExtractor(FeatureExtractor):
    name = "charge_context"
    n_features = 4
    feature_names = [
        "charge_change", "charge_change_in_tm",
        "charge_change_in_tm2", "charge_change_in_binding",
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
            wt = str(row.get("wildtype_aa", "X")).upper()
            mt = str(row.get("variant_aa", "X")).upper()

            dc = mech.aa_charge(mt) - mech.aa_charge(wt)
            features[i, 0] = dc
            features[i, 1] = dc * mech.in_tm(gene, pos)
            features[i, 2] = dc * mech.in_tm2(gene, pos)
            features[i, 3] = dc * mech.in_binding_site(gene, pos)
        return features
