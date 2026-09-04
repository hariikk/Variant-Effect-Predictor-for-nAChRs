"""
Conformational state feature extraction (open vs closed).

Extracts delta features between resting/closed and activated/open states
of nAChR structures.

Currently implemented only for CHRNA7 (7EKT closed vs 7KOX open).
Architecture supports additional receptor types when structures become available.

Features (for α7):
  - delta_rsa      : RSA(open) - RSA(closed)
  - delta_bfactor  : B-factor(open) - B-factor(closed)
  - delta_interface: nearest-neighbor chain CA distance change (open - closed)
  - ca_rmsd        : Cα displacement magnitude between states (Å)
  - pore_radius_change: change in distance to the M2 pore axis (open - closed)
"""

import numpy as np
import pandas as pd

from vep_nachr2.features.base import FeatureExtractor
from vep_nachr2.config import CONFORMATIONAL_PAIRS
from vep_nachr2.features.structural import (
    _load_pdb_resources,
    _build_alignment_map,
    _compute_rsa,
    _compute_b_factor,
)

# α7 M2 (pore-lining) helix boundaries, used to build the pore axis.
_A7_M2_RANGE = (243, 268)


def _point_line_distance(point, axis_point, axis_dir):
    """Perpendicular distance from `point` to the line through `axis_point` along `axis_dir`."""
    v = point - axis_point
    proj = np.dot(v, axis_dir) * axis_dir
    return float(np.linalg.norm(v - proj))


def _pore_axis(structure, m2_ids):
    """Centroid + first PCA axis of the M2 CA atoms across all chains."""
    coords = []
    for chain in structure[0]:
        for rid in m2_ids:
            try:
                if "CA" in chain[rid]:
                    coords.append(chain[rid]["CA"].coord)
            except KeyError:
                continue
    if len(coords) < 3:
        return None
    coords = np.asarray(coords)
    centroid = coords.mean(axis=0)
    _, _, vh = np.linalg.svd(coords - centroid, full_matrices=False)
    axis_dir = vh[0]
    axis_dir = axis_dir / np.linalg.norm(axis_dir)
    return centroid, axis_dir


def _nearest_chain_distance(structure, chain_id, ca_coord):
    """Nearest CA distance from a residue to any atom in the other chains."""
    best = np.inf
    for other in structure[0]:
        if other.id == chain_id:
            continue
        for r in other:
            if "CA" in r:
                d = np.linalg.norm(r["CA"].coord - ca_coord)
                if d < best:
                    best = d
    return best if best < np.inf else np.nan


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
      - pore_radius_change: Change in distance to the M2 pore axis
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

    def _state_metrics(self, chain, structure, dssp_dict, chain_id, ref_seq):
        """Per-position metrics for one conformational state.

        Returns a dict keyed by metric name, each a {ref_pos: value} dict:
        rsa, bfactor, interface, pore_dist, ca (CA coordinate).
        """
        amap = _build_alignment_map(ref_seq, chain, chain_id)

        m2_ids = [amap.get(p) for p in range(_A7_M2_RANGE[0], _A7_M2_RANGE[1] + 1)]
        m2_ids = [r for r in m2_ids if r is not None]
        axis = _pore_axis(structure, m2_ids)

        metrics = {"rsa": {}, "bfactor": {}, "interface": {}, "pore_dist": {}, "ca": {}}
        for pos, res_id in amap.items():
            if res_id is None:
                continue
            metrics["rsa"][pos] = _compute_rsa("CHRNA7", pos, amap, dssp_dict, chain_id)
            metrics["bfactor"][pos] = _compute_b_factor(
                "CHRNA7", pos, amap, structure, chain_id, {}
            )
            try:
                ca = chain[res_id]["CA"].coord if "CA" in chain[res_id] else None
            except KeyError:
                ca = None
            if ca is None:
                continue
            metrics["ca"][pos] = ca
            metrics["interface"][pos] = _nearest_chain_distance(structure, chain_id, ca)
            if axis is not None:
                metrics["pore_dist"][pos] = _point_line_distance(ca, axis[0], axis[1])
        return metrics

    def extract(self, df, ref_seqs=None, pdb_resources=None):
        n = len(df)
        features = np.zeros((n, self.n_features), dtype=np.float64)

        # Check if we can compute conformational features
        if pdb_resources is None or ref_seqs is None:
            return features
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

        ref_seq = ref_seqs.get("CHRNA7", "")
        if not ref_seq:
            return features

        try:
            closed_chain = closed_res["structure"][0]["A"]
            open_chain = open_res["structure"][0]["A"]
        except KeyError:
            return features

        closed = self._state_metrics(
            closed_chain, closed_res["structure"], closed_res.get("dssp_dict"), "A", ref_seq
        )
        open_ = self._state_metrics(
            open_chain, open_res["structure"], open_res.get("dssp_dict"), "A", ref_seq
        )

        for i, (_, row) in enumerate(df.iterrows()):
            gene = str(row.get("subunit", "")).upper()
            if gene != "CHRNA7":
                continue

            position = int(row.get("position", 0))

            # delta_rsa
            ro = open_["rsa"].get(position, np.nan)
            rc = closed["rsa"].get(position, np.nan)
            if not np.isnan(ro) and not np.isnan(rc):
                features[i, 0] = ro - rc

            # delta_bfactor
            bo = open_["bfactor"].get(position, np.nan)
            bc = closed["bfactor"].get(position, np.nan)
            if not np.isnan(bo) and not np.isnan(bc):
                features[i, 1] = bo - bc

            # delta_interface
            io = open_["interface"].get(position, np.nan)
            ic = closed["interface"].get(position, np.nan)
            if not np.isnan(io) and not np.isnan(ic):
                features[i, 2] = io - ic

            # ca_rmsd
            ca_o = open_["ca"].get(position)
            ca_c = closed["ca"].get(position)
            if ca_o is not None and ca_c is not None:
                features[i, 3] = float(np.linalg.norm(ca_o - ca_c))

            # pore_radius_change
            po = open_["pore_dist"].get(position, np.nan)
            pc = closed["pore_dist"].get(position, np.nan)
            if not np.isnan(po) and not np.isnan(pc):
                features[i, 4] = po - pc

        return features
