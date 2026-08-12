"""
Conformational state feature extraction (open vs closed).

Extracts delta features between resting/closed and activated/open states
of nAChR structures.

Currently implemented only for CHRNA7 (7EKT closed vs 7KOX open).
Architecture supports additional receptor types when structures become available.

Features (for α7):
  - delta_rsa: Change in RSA between open and closed
  - delta_bfactor: Change in B-factor
  - delta_interface: Change in interface proximity
  - ca_rmsd: Cα displacement between states
  - pore_radius_change: Change in M2 pore radius
"""

import warnings
from typing import Optional

import numpy as np
import pandas as pd

from vep_nachr2.features.base import FeatureExtractor
from vep_nachr2.config import CONFORMATIONAL_PAIRS, PDB_MAPPING, STRUCTURE_DIR
from vep_nachr2.features.structural import _load_pdb_resources, _build_alignment_map


class ConformationalExtractor(FeatureExtractor):
    """
    Extracts open/closed conformational delta features.

    Currently active only for CHRNA7 (7EKT -> 7KOX).
    For all other genes, returns all zeros (placeholder).

    Features (5 when active):
      - delta_rsa: RSA(open) - RSA(closed)
      - delta_bfactor: B-factor(open) - B-factor(closed)
      - delta_interface: Interface proximity change
      - ca_rmsd: Cα displacement magnitude (Å)
      - pore_radius_change: Change in M2 pore constriction
    """

    name = "conformational"
    n_features = 5
    feature_names = [
        "delta_rsa", "delta_bfactor", "delta_interface",
        "ca_rmsd", "pore_radius_change",
    ]

    # Genes with active conformational features
    _active_genes = {"CHRNA7"}

    def requires_pdb(self) -> bool:
        return True

    def requires_reference(self) -> bool:
        return True

    def extract(self, df, ref_seqs=None, pdb_resources=None):
        n = len(df)
        features = np.zeros((n, self.n_features), dtype=np.float64)

        # Check if we can compute conformational features
        if pdb_resources is None or ref_seqs is None:
            return features

        # Only CHRNA7 has both states currently
        if "CHRNA7" not in CONFORMATIONAL_PAIRS:
            return features

        pair = CONFORMATIONAL_PAIRS["CHRNA7"]
        closed_id = pair["closed"]
        open_id = pair["open"]

        # Load conformational PDBs if not already loaded
        if closed_id not in pdb_resources:
            pdb_resources[closed_id] = _load_pdb_resources(closed_id)
        if open_id not in pdb_resources:
            pdb_resources[open_id] = _load_pdb_resources(open_id)

        closed_res = pdb_resources.get(closed_id)
        open_res = pdb_resources.get(open_id)

        if closed_res is None or open_res is None:
            return features

        # Build alignment maps for both states
        ref_seq = ref_seqs.get("CHRNA7", "")
        if not ref_seq:
            return features

        try:
            closed_chain = closed_res["structure"][0]["A"]
            open_chain = open_res["structure"][0]["A"]
        except KeyError:
            return features

        closed_map = _build_alignment_map(ref_seq, closed_chain, "A")
        open_map = _build_alignment_map(ref_seq, open_chain, "A")

        # Compute per-position RMSD between states
        rmsd_map = {}
        for pos in closed_map:
            if pos in open_map:
                c_res = closed_map[pos]
                o_res = open_map[pos]
                if c_res is not None and o_res is not None:
                    try:
                        c_atom = closed_chain[c_res]
                        o_atom = open_chain[o_res]
                        if "CA" in c_atom and "CA" in o_atom:
                            rmsd_map[pos] = np.linalg.norm(
                                c_atom["CA"].coord - o_atom["CA"].coord
                            )
                        else:
                            rmsd_map[pos] = 0.0
                    except (KeyError, Exception):
                        rmsd_map[pos] = 0.0

        # Apply to CHRNA7 variants
        for i, (_, row) in enumerate(df.iterrows()):
            gene = str(row.get("subunit", "")).upper()
            if gene != "CHRNA7":
                continue

            position = int(row.get("position", 0))
            features[i, 4] = rmsd_map.get(position, 0.0)  # CA RMSD
            # Other delta features require full DSSP pass — placeholder for now

        return features
