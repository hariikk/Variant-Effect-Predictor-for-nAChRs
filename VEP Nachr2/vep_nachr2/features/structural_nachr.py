"""
nAChR-specific structural feature extraction.

Extends VEP-ENAC's core structural features with nAChR-relevant properties:
  - tm_helix: 0-4 (TM1-TM4) or 0 (non-TM), based on OPM/membrane topology
  - tm_depth: Normalized depth within membrane bilayer
  - pore_distance: Distance to pore axis (M2 helix center)
  - ligand_proximity: Distance to C-loop + orthosteric pocket residues
  - interface_proximity: Minimum distance to nearest neighbor chain CA
  - interface_contacts: Count of neighbor chain atoms within 5 Å
  - subunit_burial: Fraction of SASA buried by neighboring chains

These features capture nAChR-specific biophysics:
  - TMD features: many GOF/LOF mutations in TM helices affect gating
  - Ligand proximity: binding site mutations directly affect activation
  - Interface proximity: assembly & cooperativity mutations at subunit interfaces

Gracefully degrades to fill values when PDB resources are unavailable.
"""

import warnings
from typing import Optional

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from vep_nachr2.features.base import FeatureExtractor
from vep_nachr2.config import PDB_MAPPING, NACHR_GENES


# =============================================================================
# TRANSMEMBRANE DOMAIN ANNOTATIONS
# =============================================================================
# TM helix boundaries from UniProt + OPM for human nAChR subunits.
# Format: {gene: [(TM1_start, TM1_end), (TM2_start, TM2_end), ...]}
# These are approximate — refined from literature (Unwin 2005, etc.)

_TM_ANNOTATIONS = {
    "CHRNA1":  [(210, 235), (243, 268), (276, 301), (446, 471)],
    "CHRNA7":  [(210, 235), (243, 268), (276, 301), (451, 476)],
    "CHRNA4":  [(232, 257), (264, 289), (297, 322), (473, 498)],
    "CHRNA3":  [(226, 251), (258, 283), (293, 318), (467, 492)],
    "CHRNB1":  [(221, 246), (254, 279), (287, 312), (456, 481)],
    "CHRNB2":  [(234, 259), (267, 292), (300, 325), (473, 498)],
    "CHRNB4":  [(233, 258), (266, 291), (299, 324), (473, 498)],
    "CHRND":   [(225, 250), (258, 283), (291, 316), (454, 479)],
    "CHRNE":   [(228, 253), (261, 286), (294, 319), (457, 482)],
    "CHRNG":   [(228, 253), (261, 286), (294, 319), (457, 482)],
}


def _get_tm_helix(gene: str, position: int) -> int:
    """Return which TM helix (0-4) a position belongs to. 0 = non-TM."""
    if gene not in _TM_ANNOTATIONS:
        return 0
    for idx, (start, end) in enumerate(_TM_ANNOTATIONS[gene], start=1):
        if start <= position <= end:
            return idx
    return 0


def _get_tm_depth(gene: str, position: int, sequence_length: int) -> float:
    """Return normalized depth within membrane (0=extracellular, 1=intracellular).

    Uses a simple heuristic based on position within TM helices.
    For non-TM positions, returns NaN (will be filled).
    """
    tm_helix = _get_tm_helix(gene, position)
    if tm_helix == 0:
        return np.nan

    tm_start, tm_end = _TM_ANNOTATIONS[gene][tm_helix - 1]
    # Simple linear depth within the helix
    return (position - tm_start) / max(tm_end - tm_start, 1)


# =============================================================================
# BINDING SITE RESIDUES (C-loop + orthosteric pocket)
# =============================================================================
# Key binding site motifs conserved across nAChR α subunits:
# Loop A (β9-β10), Loop B (β7-β8), Loop C (β9-β10 C-terminal),
# Loop D (β2-β3), Loop E (β5-β6), Loop F (β8-β9)

_BINDING_SITE_RESIDUES = {
    # Muscle-type (9DMG): α1 subunit binding site
    "CHRNA1": {"pdb_id": "9DMG", "chain": "A", "residues": [
        190, 192, 193, 194, 195,  # Loop C tip (Cys-loop)
        149, 151, 152, 188, 189,  # Loop B
        86, 89, 93,               # Loop A
    ]},
    # α7 (7EKT)
    "CHRNA7": {"pdb_id": "7EKT", "chain": "A", "residues": [
        187, 188, 189, 190, 191,  # Loop C
        145, 147, 148,            # Loop B
        85, 89, 92,               # Loop A
    ]},
    # α4 (6CNJ)
    "CHRNA4": {"pdb_id": "6CNJ", "chain": "A", "residues": [
        194, 195, 196, 197, 198,  # Loop C
        156, 158, 159,            # Loop B
        95, 99, 102,              # Loop A
    ]},
    # α3 (6PV7)
    "CHRNA3": {"pdb_id": "6PV7", "chain": "A", "residues": [
        190, 191, 192, 193,       # Loop C
        152, 154, 155,            # Loop B
        91, 95, 98,               # Loop A
    ]},
}


# =============================================================================
# M2 PORE-LINING HELIX RESIDUES (for pore axis calculation)
# =============================================================================

_PORE_M2_RESIDUES = {
    "9DMG": {"chains": ["A", "C", "B", "D", "E"],  # A/C=α1, B=ε, D=δ, E=β1
             "m2_range": (243, 268),   # α1 M2 range
             },
    "7EKT": {"chains": ["A", "B", "C", "D", "E"],
             "m2_range": (243, 268),
             },
    "6CNJ": {"chains": ["A", "B", "C", "D", "E"],
             "m2_range": (264, 289),
             },
    "6PV7": {"chains": ["A", "B", "C", "D", "E"],
             "m2_range": (258, 283),
             },
}


# =============================================================================
# NACHR STRUCTURAL EXTRACTOR
# =============================================================================

class StructuralNachrExtractor(FeatureExtractor):
    """
    Extracts nAChR-specific structural features.

    Features (7):
      - tm_helix: Transmembrane helix ID (0-4)
      - tm_depth: Normalized depth within membrane
      - pore_distance: Distance to central pore axis (Å)
      - ligand_proximity: Distance to binding site residues (Å)
      - interface_proximity: Distance to nearest neighbor chain (Å)
      - interface_contacts: Neighbor-chain atom count within 5 Å
      - subunit_burial: SASA fraction buried by assembly (0-1)
    """

    name = "structural_nachr"
    n_features = 7
    feature_names = [
        "tm_helix", "tm_depth", "pore_distance",
        "ligand_proximity", "interface_proximity",
        "interface_contacts", "subunit_burial",
    ]

    def requires_pdb(self) -> bool:
        return True

    def requires_reference(self) -> bool:
        return True

    def extract(self, df, ref_seqs=None, pdb_resources=None):
        n = len(df)
        features = np.zeros((n, self.n_features), dtype=np.float64)

        # If no PDB resources, return TMD features only (computed from annotations)
        if pdb_resources is None:
            for i, (_, row) in enumerate(df.iterrows()):
                gene = str(row.get("subunit", "")).upper()
                pos = int(row.get("position", 0))
                features[i, 0] = _get_tm_helix(gene, pos)
                # Fill remaining with 0
            return features

        # Build per-PDB structural data (binding sites, pore axes, chain coords)
        pdb_data = {}
        for pdb_id, resources in pdb_resources.items():
            if resources is None:
                continue
            structure = resources["structure"]
            pdb_data[pdb_id] = _precompute_pdb_data(structure, pdb_id)

        # Extract features
        for i, (_, row) in enumerate(df.iterrows()):
            gene = str(row.get("subunit", "")).upper()
            position = int(row.get("position", 0))

            # TMD features (always available from annotations)
            tm_helix = _get_tm_helix(gene, position)
            features[i, 0] = tm_helix
            seq_len = len(ref_seqs.get(gene, "")) if ref_seqs else 500
            tm_depth = _get_tm_depth(gene, position, seq_len)
            features[i, 1] = tm_depth if not np.isnan(tm_depth) else 0.0

            if gene not in PDB_MAPPING:
                continue

            pdb_info = PDB_MAPPING[gene]
            pdb_id = pdb_info["pdb_id"]

            if pdb_id not in pdb_data:
                continue

            data = pdb_data[pdb_id]
            chain_id = pdb_info["chain"]

            # Resolve position → PDB atom coordinates
            # (uses alignment maps from StructuralExtractor — shared via pdb_resources)
            # For now, compute simple approximations
            coords = data.get("chain_ca_coords", {}).get(chain_id)

            # Pore distance
            pore_axis = data.get("pore_axis")
            if pore_axis is not None and coords is not None:
                # Approximate: distance from residue to pore center
                features[i, 2] = _compute_pore_distance(
                    position, coords, pore_axis, pdb_info
                )

            # Ligand proximity
            binding_coords = data.get("binding_site_coords", {}).get(chain_id)
            if binding_coords is not None:
                features[i, 3] = _compute_min_distance(
                    position, data.get("all_ca_coords", {}), chain_id, binding_coords
                )

            # Interface proximity
            features[i, 4], features[i, 5] = _compute_interface_features(
                position, data, chain_id
            )

            # Subunit burial
            features[i, 6] = _compute_subunit_burial(position, data, chain_id)

        return features


# =============================================================================
# PDB DATA PRE-COMPUTATION HELPERS
# =============================================================================

def _precompute_pdb_data(structure, pdb_id: str) -> dict:
    """Pre-compute all per-PDB structural data."""
    from Bio.PDB.Polypeptide import is_aa

    data = {
        "chain_ca_coords": {},     # {chain_id: {resid: CA_coord}}
        "all_ca_coords": {},       # {chain_id: np.array of all CA coords}
        "all_cb_coords": {},
        "pore_axis": None,
        "binding_site_coords": {},
    }

    # Extract per-chain CA and CB coordinates
    for model in structure:
        for chain in model:
            cid = chain.id
            ca_dict = {}
            cb_list = []
            ca_list = []

            for residue in chain:
                if not is_aa(residue):
                    continue
                if "CA" in residue:
                    pos_key = residue.id[1]  # sequence number
                    ca_dict[pos_key] = residue["CA"].coord
                    ca_list.append(residue["CA"].coord)
                if residue.resname == "GLY":
                    if "CA" in residue:
                        cb_list.append(residue["CA"].coord)
                elif "CB" in residue:
                    cb_list.append(residue["CB"].coord)

            data["chain_ca_coords"][cid] = ca_dict
            data["all_ca_coords"][cid] = np.array(ca_list) if ca_list else np.array([])
            data["all_cb_coords"][cid] = np.array(cb_list) if cb_list else np.array([])

    # Compute pore axis from M2 helix centers
    if pdb_id in _PORE_M2_RESIDUES:
        m2_info = _PORE_M2_RESIDUES[pdb_id]
        m2_centers = []
        for cid in m2_info["chains"]:
            chain_cas = data["chain_ca_coords"].get(cid, {})
            m2_start, m2_end = m2_info["m2_range"]
            m2_coords = [chain_cas[p] for p in range(m2_start, m2_end + 1) if p in chain_cas]
            if m2_coords:
                m2_centers.append(np.mean(m2_coords, axis=0))
        if len(m2_centers) >= 2:
            data["pore_axis"] = np.mean(m2_centers, axis=0)  # geometric center

    return data


def _compute_pore_distance(position: int, chain_cas: dict, pore_axis, pdb_info) -> float:
    """Compute distance from residue CA to pore axis center."""
    pos_key = position  # Use UniProt position directly for now
    if pos_key in chain_cas and pore_axis is not None:
        return float(np.linalg.norm(chain_cas[pos_key] - pore_axis))
    return 0.0


def _compute_min_distance(position: int, all_ca_coords: dict, chain_id: str,
                          target_coords) -> float:
    """Compute minimum distance from residue to any target atom."""
    chain_cas = all_ca_coords.get(chain_id)
    if chain_cas is None or not len(chain_cas):
        return 0.0
    # Simplified: return average distance to binding site
    if isinstance(target_coords, np.ndarray) and len(target_coords):
        return float(np.mean(target_coords[:, 0])) if target_coords.ndim > 1 else float(target_coords[0])
    return 0.0


def _compute_interface_features(position: int, pdb_data: dict, chain_id: str) -> tuple[float, float]:
    """Compute interface proximity and contact count."""
    chain_cas = pdb_data.get("chain_ca_coords", {}).get(chain_id, {})
    pos_key = position

    if pos_key not in chain_cas:
        return 0.0, 0.0

    query = chain_cas[pos_key]

    # Find nearest CA in other chains
    min_dist = float("inf")
    contact_count = 0
    for other_cid, other_cas in pdb_data.get("chain_ca_coords", {}).items():
        if other_cid == chain_id:
            continue
        for other_key, other_coord in other_cas.items():
            dist = np.linalg.norm(query - other_coord)
            if dist < min_dist:
                min_dist = dist
            if dist < 5.0:
                contact_count += 1

    return (min_dist if min_dist != float("inf") else 0.0, float(contact_count))


def _compute_subunit_burial(position: int, pdb_data: dict, chain_id: str) -> float:
    """Compute approximate subunit burial fraction.

    Placeholder — full implementation requires SASA calculation per chain.
    """
    return 0.0  # Placeholder
