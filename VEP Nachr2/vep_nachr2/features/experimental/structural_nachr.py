"""
nAChR-specific structural feature extraction.

Extends VEP-ENAC's core structural features with nAChR-relevant properties:
  - tm_helix: 0-4 (TM1-TM4) or 0 (non-TM), based on OPM/membrane topology
  - tm_depth: Normalized depth within membrane bilayer
  - pore_distance: Perpendicular distance to the M2 pore axis (Å)
  - ligand_proximity: Min distance to the orthosteric agonist-binding site (Å)
  - interface_proximity: Minimum distance to nearest neighbor chain CA
  - interface_contacts: Count of neighbor chain atoms within 5 Å
  - subunit_burial: Fraction of residue atoms buried by neighboring chains

These features capture nAChR-specific biophysics:
  - TMD features: many GOF/LOF mutations in TM helices affect gating
  - Ligand proximity: binding site mutations directly affect activation
  - Interface proximity: assembly & cooperativity mutations at subunit interfaces

Position → PDB mapping uses the SAME alignment maps as StructuralExtractor
(`_build_all_alignment_maps`). The previous version used the raw UniProt
position as a PDB residue key, which silently left pore/interface/burial/ligand
features at 0.0 for most variants. TM annotations and binding-site residues now
come from `mechanism.py` (single source of truth, all 16 genes — previously only
10 were annotated here).

Gracefully degrades to fill values (0.0) when PDB resources are unavailable.
"""

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from vep_nachr2.features.base import FeatureExtractor
from vep_nachr2.features.structural import _build_all_alignment_maps
from vep_nachr2.features.experimental import mechanism as mech
from vep_nachr2.config import PDB_MAPPING


# =============================================================================
# ANNOTATION HELPERS (delegated to mechanism.py — 16 genes)
# =============================================================================

def _get_tm_helix(gene: str, position: int) -> int:
    """Return which TM helix (0-4) a position belongs to. 0 = non-TM."""
    return mech.get_tm_helix(gene, position)


def _get_tm_depth(gene: str, position: int) -> float:
    """Return normalized depth within membrane (0=extracellular, 1=intracellular)."""
    idx = mech.get_tm_helix(gene, position)
    if idx == 0:
        return np.nan
    s, e = mech.get_tm(gene, idx)
    return (position - s) / max(e - s, 1)


def _point_line_distance(point, axis_point, axis_dir) -> float:
    """Perpendicular distance from `point` to the line through `axis_point` along `axis_dir`."""
    v = point - axis_point
    proj = float(np.dot(v, axis_dir)) * axis_dir
    return float(np.linalg.norm(v - proj))


# =============================================================================
# PDB DATA PRE-COMPUTATION
# =============================================================================

def _precompute_pdb_data(structure) -> dict:
    """Precompute per-PDB coordinate tables for fast feature lookup."""
    from Bio.PDB.Polypeptide import is_aa

    data = {
        "chain_ca_coords": {},       # {chain_id: {resseq: CA_coord}}
        "chain_ca_arrays": {},       # {chain_id: (N,3) CA coords}
        "chain_heavy": {},           # {chain_id: {resseq: (N,3) heavy-atom coords}}
        "heavy_kdtree": None,        # cKDTree over all heavy atoms (all chains)
        "heavy_chain_codes": None,   # int chain code per heavy atom
        "chain_to_code": {},         # {chain_id: int}
    }

    heavy_all = []
    heavy_chain = []
    chain_to_code = {}
    next_code = 0

    for model in structure:
        for chain in model:
            cid = chain.id
            if cid not in chain_to_code:
                chain_to_code[cid] = next_code
                next_code += 1
            code = chain_to_code[cid]

            ca_dict = {}
            ca_list = []
            heavy_dict = {}
            for residue in chain:
                if not is_aa(residue):
                    continue
                resseq = residue.id[1]
                if "CA" in residue:
                    ca_dict[resseq] = residue["CA"].coord
                    ca_list.append(residue["CA"].coord)
                atoms = [a.coord for a in residue if a.element != "H"]
                if atoms:
                    arr = np.asarray(atoms, dtype=np.float64)
                    heavy_dict[resseq] = arr
                    heavy_all.append(arr)
                    heavy_chain.extend([code] * len(arr))

            data["chain_ca_coords"][cid] = ca_dict
            data["chain_ca_arrays"][cid] = np.asarray(ca_list, dtype=np.float64) if ca_list else np.zeros((0, 3))
            data["chain_heavy"][cid] = heavy_dict

    data["chain_to_code"] = chain_to_code
    if heavy_all:
        data["heavy_kdtree"] = cKDTree(np.vstack(heavy_all))
        data["heavy_chain_codes"] = np.asarray(heavy_chain, dtype=np.int64)
    return data


def _build_pore_axis(pdb_id: str, data: dict, alignment_maps: dict):
    """Pore axis = centroid + principal (PCA) axis of the M2-helix CAs.

    M2 residues are taken from each subunit's TM2 range (precursor numbering,
    mechanism.py) and mapped to PDB coordinates via the alignment map — robust
    to PDB vs precursor numbering differences.
    """
    coords = []
    for gene, info in PDB_MAPPING.items():
        if info["pdb_id"] != pdb_id:
            continue
        tm2 = mech.get_tm(gene, 2)
        if tm2 is None:
            continue
        amap = alignment_maps.get(gene, {})
        chain_cas = data["chain_ca_coords"].get(info["chain"], {})
        s, e = tm2
        for pos in range(s, e + 1):
            r = amap.get(pos)
            if r is not None and r[1] in chain_cas:
                coords.append(chain_cas[r[1]])

    if len(coords) < 3:
        return None

    coords = np.asarray(coords, dtype=np.float64)
    centroid = coords.mean(axis=0)
    _, _, vh = np.linalg.svd(coords - centroid, full_matrices=False)
    axis_dir = vh[0] / np.linalg.norm(vh[0])
    return centroid, axis_dir


# =============================================================================
# NACHR STRUCTURAL EXTRACTOR
# =============================================================================

class StructuralNachrExtractor(FeatureExtractor):
    """
    Extracts nAChR-specific structural features.

    Features (7):
      - tm_helix: Transmembrane helix ID (0-4)
      - tm_depth: Normalized depth within membrane
      - pore_distance: Perpendicular distance to the M2 pore axis (Å)
      - ligand_proximity: Min distance to binding site residues (Å)
      - interface_proximity: Distance to nearest neighbor chain CA (Å)
      - interface_contacts: Neighbor-chain CA count within 5 Å
      - subunit_burial: Fraction of residue atoms buried by neighboring chains
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

        # TM helix + depth are annotation-only and available for all 16 genes.
        for i, (_, row) in enumerate(df.iterrows()):
            gene = str(row.get("subunit", "")).upper()
            pos = int(row.get("position", 0))
            features[i, 0] = _get_tm_helix(gene, pos)
            depth = _get_tm_depth(gene, pos)
            features[i, 1] = 0.0 if np.isnan(depth) else depth

        if pdb_resources is None or ref_seqs is None:
            return features

        alignment_maps = _build_all_alignment_maps(ref_seqs, pdb_resources)

        pdb_data = {}
        for pdb_id, resources in pdb_resources.items():
            if resources is None:
                continue
            pdb_data[pdb_id] = _precompute_pdb_data(resources["structure"])

        # Pore axes per PDB (computed once).
        pore_axes = {}
        for pdb_id, data in pdb_data.items():
            axis = _build_pore_axis(pdb_id, data, alignment_maps)
            if axis is not None:
                pore_axes[pdb_id] = axis

        # Binding-site CA coordinates per gene (orthosteric pocket, α subunits).
        binding_coords = {}
        for gene, info in PDB_MAPPING.items():
            residues = mech._BINDING_SITE_RESIDUES.get(gene, [])
            if not residues:
                continue
            data = pdb_data.get(info["pdb_id"])
            if data is None:
                continue
            amap = alignment_maps.get(gene, {})
            chain_cas = data["chain_ca_coords"].get(info["chain"], {})
            coords = []
            for pos in residues:
                r = amap.get(pos)
                if r is not None and r[1] in chain_cas:
                    coords.append(chain_cas[r[1]])
            if coords:
                binding_coords[gene] = np.asarray(coords, dtype=np.float64)

        for i, (_, row) in enumerate(df.iterrows()):
            gene = str(row.get("subunit", "")).upper()
            position = int(row.get("position", 0))

            if gene not in PDB_MAPPING:
                continue
            info = PDB_MAPPING[gene]
            pdb_id = info["pdb_id"]
            data = pdb_data.get(pdb_id)
            if data is None:
                continue
            chain_id = info["chain"]
            amap = alignment_maps.get(gene, {})
            pdb_res_id = amap.get(position)
            if pdb_res_id is None:
                continue
            resseq = pdb_res_id[1]

            ca = data["chain_ca_coords"].get(chain_id, {}).get(resseq)
            if ca is None:
                continue

            # pore_distance: perpendicular distance to the M2 pore axis.
            if pdb_id in pore_axes:
                centroid, axis_dir = pore_axes[pdb_id]
                features[i, 2] = _point_line_distance(ca, centroid, axis_dir)

            # ligand_proximity: min distance to the agonist-binding site.
            bc = binding_coords.get(gene)
            if bc is not None and len(bc):
                features[i, 3] = float(np.min(np.linalg.norm(bc - ca, axis=1)))

            # interface_proximity: min cross-chain CA distance (vectorized).
            min_dist = float("inf")
            for other_cid, other_cas in data["chain_ca_arrays"].items():
                if other_cid == chain_id or len(other_cas) == 0:
                    continue
                d = np.linalg.norm(other_cas - ca, axis=1)
                min_dist = min(min_dist, float(d.min()))
            features[i, 4] = min_dist if min_dist != float("inf") else 0.0

            # interface_contacts + subunit_burial (heavy-atom based, cross-chain).
            heavy = data["chain_heavy"].get(chain_id, {}).get(resseq)
            kdtree = data["heavy_kdtree"]
            if heavy is not None and len(heavy) and kdtree is not None:
                code = data["chain_to_code"][chain_id]
                chain_codes = data["heavy_chain_codes"]
                buried_atoms = 0
                contact_atoms = set()
                for atom_coord in heavy:
                    nbrs = kdtree.query_ball_point(atom_coord, 5.0)
                    cross = [j for j in nbrs if chain_codes[j] != code]
                    if cross:
                        buried_atoms += 1
                        contact_atoms.update(cross)
                features[i, 5] = float(len(contact_atoms))   # interface_contacts
                features[i, 6] = buried_atoms / len(heavy)   # subunit_burial

        return features
