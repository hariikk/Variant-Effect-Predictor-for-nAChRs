"""
Structure-aware stability proxy (burial-weighted physicochemical change).

This is the "structure-aware ΔΔG" slot. The originally-planned ESM-IF inverse-
folding model was dropped: fair-esm 2.0.0's ESM-IF is pinned to a 2022
dependency stack (old `biotite` with `filter_backbone`, `torch_geometric` ~2.0,
`torch_scatter`) that is incompatible with the installed torch 2.6 — three
dependencies deep it breaks on a removed biotite API, and fixing it would
require downgrading torch (which would break the working ESM-2). ThermoMPNN has
the same problem (separate repo + torch_geometric).

Instead we use the classic pre-ML stability signal: the *buried* physicochemical
change. A substitution's effect on folding stability is dominated by (a) the
hydrophobic effect and (b) packing/sterics, and both matter only where the
residue is buried (low solvent accessibility). These are interaction terms —
"does the chemical change land in the buried core" — which are not present in
the existing physicochemical (wt/mt/diff only) or structural (RSA only) groups.

Features (2, positive = destabilizing):
  - ddg_hydrophobic : (hyd_wt - hyd_mt) * burial
  - ddg_packing     : |vol_mt - vol_wt| * burial

where hyd = EISD840101 hydrophobicity, vol = KRIW790103 volume (both
min-max-normalized [0,1]), burial = (1 - RSA). Unmappable residues get burial 0.
"""

import numpy as np
import pandas as pd

from vep_nachr2.features.base import FeatureExtractor
from vep_nachr2.features.physicochemical import get_aa_properties
from vep_nachr2.features.structural import _build_all_alignment_maps, _compute_rsa
from vep_nachr2.config import PDB_MAPPING, AMINO_ACIDS

# Column indices into get_aa_properties() output.
_HYD = 0  # EISD840101 hydrophobicity
_VOL = 2  # KRIW790103 volume


class StabilityExtractor(FeatureExtractor):
    name = "stability"
    n_features = 2
    feature_names = ["ddg_hydrophobic", "ddg_packing"]

    def requires_pdb(self) -> bool:
        return True

    def requires_reference(self) -> bool:
        return True  # alignment maps need the reference sequence

    def extract(self, df, ref_seqs=None, pdb_resources=None):
        n = len(df)
        features = np.zeros((n, self.n_features), dtype=np.float64)

        if pdb_resources is None or ref_seqs is None:
            return features

        alignment_maps = _build_all_alignment_maps(ref_seqs, pdb_resources)

        for i, (_, row) in enumerate(df.iterrows()):
            gene = str(row.get("subunit", "")).upper()
            position = int(row.get("position", 0))
            wt = str(row.get("wildtype_aa", "X")).upper()
            mt = str(row.get("variant_aa", "X")).upper()

            if gene not in PDB_MAPPING or gene not in alignment_maps:
                continue
            pdb_info = PDB_MAPPING[gene]
            pdb_id = pdb_info["pdb_id"]
            chain_id = pdb_info["chain"]
            amap = alignment_maps[gene]

            resources = pdb_resources.get(pdb_id)
            if resources is None:
                continue

            rsa = _compute_rsa(
                gene, position, amap, resources.get("dssp_dict"), chain_id
            )
            burial = 1.0 - rsa if not np.isnan(rsa) else 0.0
            burial = float(np.clip(burial, 0.0, 1.0))

            if wt not in AMINO_ACIDS or mt not in AMINO_ACIDS:
                continue  # non-standard residue — leave stability at 0

            props_wt = get_aa_properties(wt)
            props_mt = get_aa_properties(mt)

            hyd_wt, hyd_mt = props_wt[_HYD], props_mt[_HYD]
            vol_wt, vol_mt = props_wt[_VOL], props_mt[_VOL]

            features[i, 0] = (hyd_wt - hyd_mt) * burial
            features[i, 1] = abs(vol_mt - vol_wt) * burial

        return features
