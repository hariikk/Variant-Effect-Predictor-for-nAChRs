"""
Allosteric-coupling × mutation interaction features (Run XXVII).

Motivation (colleague's "indirect connection" idea):
  A mutation's effect on channel function is *indirect* — it is transmitted to
  the functional machinery through a network of coupled residues. The GNM gate
  coupling (gating.py) measures how strongly each position is allosterically
  coupled to the M2 9' gate; the physicochemical delta measures how much the
  mutation changes local chemistry. Neither alone captures the *interaction*:
  the same chemical change matters more at a position that is strongly coupled
  to the gate than at one that is not.

These features are the product, per variant:

  allosteric_dvol      gate_coupling × Δvolume
  allosteric_dhydro    gate_coupling × Δhydrophobicity
  allosteric_dcharge   gate_coupling × Δcharge
  allosteric_dpolarity gate_coupling × Δpolarity

gate_coupling ∈ [0,1] is the |mean correlation| of the residue to the gate ring
(0 for the 7 AlphaFold-monomer genes, which have no assembled pore); Δprop ∈
[-1,1] is the variant-minus-wildtype normalized AAIndex change. The product is
signed: a big volume *increase* at a gate-coupled residue gets a large positive
value, a volume *decrease* a large negative one, and any change at an uncoupled
residue ~0.

Runs XVIII/XXV tested raw gate proximity and raw GNM coupling — both gene-AGNOSTIC
null. Run XXVII tests the interaction under gene-AWARE CV: direction is
gene-conditioned, so "coupled to the gate" may mean GOF in one subunit and LOF in
another, and only the gene-aware model can learn that rule. This is NOT in the
default 75-feature set — added as an additive group.
"""

import numpy as np

from vep_nachr2.features.base import FeatureExtractor
from vep_nachr2.features.structural import _build_all_alignment_maps
from vep_nachr2.features.gating import _build_gnm_tables
from vep_nachr2.features.physicochemical import get_aa_properties
from vep_nachr2.config import PDB_MAPPING

# Indices into the AAIndex property vector (AA_PROPERTY_NAMES order):
#   hydrophobicity=0, polarity=1, volume=2, molecular_weight=3, charge=4,
#   isoelectric_point=5, aromaticity=6, ss_preference=7
_IDX_VOLUME = 2
_IDX_HYDROPHOBICITY = 0
_IDX_CHARGE = 4
_IDX_POLARITY = 1


class AllostericInteractionExtractor(FeatureExtractor):
    """Allosteric-coupling × mutation-delta interaction features (Run XXVII)."""

    name = "allosteric_interaction"
    n_features = 4
    feature_names = [
        "allosteric_dvol",
        "allosteric_dhydro",
        "allosteric_dcharge",
        "allosteric_dpolarity",
    ]

    def requires_pdb(self) -> bool:
        return True

    def requires_reference(self) -> bool:
        return True

    def extract(self, df, ref_seqs=None, pdb_resources=None):
        n = len(df)
        feats = np.zeros((n, self.n_features), dtype=np.float64)

        if pdb_resources is None or ref_seqs is None:
            # Unmappable default: no coupling, no interaction signal.
            return feats

        alignment_maps = _build_all_alignment_maps(ref_seqs, pdb_resources)
        gnm_tables = _build_gnm_tables(pdb_resources)

        for i, (_, row) in enumerate(df.iterrows()):
            gene = str(row.get("subunit", "")).upper()
            position = int(row.get("position", 0))

            if gene not in PDB_MAPPING or gene not in alignment_maps:
                continue
            pdb_id = PDB_MAPPING[gene]["pdb_id"]
            chain_id = PDB_MAPPING[gene]["chain"]
            pdb_res_id = alignment_maps[gene].get(position)
            if pdb_res_id is None or pdb_id not in gnm_tables:
                continue

            table = gnm_tables[pdb_id]
            idx = table["index_map"].get((chain_id, pdb_res_id))
            if idx is None:
                continue

            coupling = table["gate_coupling"][idx]

            wt = get_aa_properties(str(row.get("wildtype_aa", "X")))
            mt = get_aa_properties(str(row.get("variant_aa", "X")))
            d = mt - wt  # Δproperty, signed, in [-1, 1]

            feats[i, 0] = coupling * d[_IDX_VOLUME]
            feats[i, 1] = coupling * d[_IDX_HYDROPHOBICITY]
            feats[i, 2] = coupling * d[_IDX_CHARGE]
            feats[i, 3] = coupling * d[_IDX_POLARITY]

        return feats
