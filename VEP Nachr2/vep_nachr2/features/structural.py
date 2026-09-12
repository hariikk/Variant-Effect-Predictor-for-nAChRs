"""
Core structural feature extraction from PDB structures.

Extracts 5 features per variant by mapping UniProt positions to PDB chains:
  - rsa: Relative Solvent Accessibility (DSSP ASA / MAX_ASA)
  - cbeta_density: Number of Cβ atoms within 10 Å (local packing)
  - b_factor: Average B-factor of residue heavy atoms
  - dssp_onehot: 3 one-hot flags (helix, sheet, coil) from DSSP
  - is_unmappable: Binary flag (1 = position not in PDB, 0 = mapped or pseudo-resolved)

Cloned from VEP-ENAC's structural features with adaptations for nAChR's
multi-PDB architecture (one PDB per receptor type instead of a single PDB).

Graceful degradation: returns fill values when PDB resources are unavailable.
"""

import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from Bio import PDB
from Bio.PDB import MMCIFParser, MMCIF2Dict
from Bio.PDB.DSSP import make_dssp_dict
from Bio.PDB.Polypeptide import is_aa, protein_letters_3to1
from scipy.spatial import cKDTree

from vep_nachr2.features.base import FeatureExtractor
from vep_nachr2.config import (
    PDB_MAPPING,
    NACHR_GENES,
    MAX_ASA,
    STRUCTURE_DIR,
    STRUCTURAL_FILL_VALUES,
)


# =============================================================================
# DSSP CODE MAPPING
# =============================================================================

_DSSP_HELIX_CODES = frozenset("HGI")   # α-helix, 3-10, π-helix
_DSSP_SHEET_CODES = frozenset("EB")    # extended strand, β-bridge


def _dssp_to_flags(ss: object) -> tuple[float, float, float]:
    """Convert DSSP single-letter code to (helix, sheet, coil) one-hot."""
    if ss is None or not isinstance(ss, str):
        return 0.0, 0.0, 1.0  # default: coil
    helix = 1.0 if ss in _DSSP_HELIX_CODES else 0.0
    sheet = 1.0 if ss in _DSSP_SHEET_CODES else 0.0
    coil = 0.0 if (helix or sheet) else 1.0
    return helix, sheet, coil


# =============================================================================
# PDB RESOURCE LOADING
# =============================================================================

def _extract_ss_from_cif(cif_dict: dict) -> dict:
    """Build a DSSP-like secondary structure dict from mmCIF annotations.

    Parses _struct_conf (helices) and _struct_sheet_range (strands).
    Returns {(chainid, res_id): (aa, ss, acc, ...)} dict mimic of make_dssp_dict,
    where res_id = (" ", resseq, " ").

    DSSP data (ASA, phi, psi, H-bonds) is not available from CIF — filled with
    defaults so downstream code falls back to STRUCTURAL_FILL_VALUES for RSA.
    """
    ss_map = {}

    def _add_range(chain_id, start_seq, end_seq, ss_code):
        try:
            for r in range(int(start_seq), int(end_seq) + 1):
                key = (chain_id, (" ", r, " "))
                if key not in ss_map:
                    # (aa, ss, acc, phi, psi, dssp_index, ...)
                    # Fill with defaults: aa='X', acc=0, phi=360, psi=360
                    ss_map[key] = ("X", ss_code, 0, 360.0, 360.0,
                                   0, 0, 0.0, 0, 0.0, 0, 0.0, 0, 0.0)
        except (ValueError, TypeError):
            pass

    # Helices
    try:
        n_helices = len(cif_dict.get("_struct_conf.conf_type_id", []))
        for i in range(n_helices):
            chain = cif_dict["_struct_conf.beg_label_asym_id"][i]
            start = cif_dict["_struct_conf.beg_auth_seq_id"][i]
            end = cif_dict["_struct_conf.end_auth_seq_id"][i]
            _add_range(chain, start, end, "H")
    except (KeyError, IndexError):
        pass

    # Sheets
    try:
        n_strands = len(cif_dict.get("_struct_sheet_range.sheet_id", []))
        for i in range(n_strands):
            chain = cif_dict["_struct_sheet_range.beg_label_asym_id"][i]
            start = cif_dict["_struct_sheet_range.beg_auth_seq_id"][i]
            end = cif_dict["_struct_sheet_range.end_auth_seq_id"][i]
            _add_range(chain, start, end, "E")
    except (KeyError, IndexError):
        pass

    return ss_map


def _compute_dssp_dict_from_structure(structure, cif_dict: dict) -> dict:
    """Build a DSSP-like dict with REAL solvent accessibility (SASA) + SS codes.

    For AlphaFold models that have no precomputed .dssp file, computes per-residue
    solvent-accessible surface area via Shrake-Rupley (probe radius 1.4 Å, matching
    DSSP's "acc" field) and reads secondary structure from mmCIF _struct_conf /
    _struct_sheet_range annotations. Returns the same {(chainid, res_id):
    (aa, ss, acc, phi, psi, dssp_index, ...)} 14-tuple shape as
    Bio.PDB.DSSP.make_dssp_dict so downstream RSA/SS code is unchanged.
    """
    from Bio.PDB.SASA import ShrakeRupley

    ss_map = _extract_ss_from_cif(cif_dict)  # {(chainid, res_id): (..., ss, ...)}

    sr = ShrakeRupley()
    sr.compute(structure, level="R")

    dssp_dict = {}
    index = 1
    for model in structure:
        for chain in model:
            for residue in chain:
                if not is_aa(residue) or residue.resname not in protein_letters_3to1:
                    continue
                key = (chain.id, residue.id)
                aa = protein_letters_3to1[residue.resname]
                ss = ss_map[key][1] if key in ss_map else "-"
                acc = float(getattr(residue, "sasa", 0.0))
                dssp_dict[key] = (
                    aa,       # [0] amino acid
                    ss,       # [1] secondary structure
                    acc,      # [2] accessible surface area (Å²)
                    360.0,    # [3] phi (placeholder)
                    360.0,    # [4] psi (placeholder)
                    index,    # [5] dssp index
                    0, 0.0, 0, 0.0, 0, 0.0, 0, 0.0,  # [6..13] H-bond placeholders
                )
                index += 1
    return dssp_dict


def _load_pdb_resources(pdb_id: str) -> Optional[dict]:
    """Load PDB structure + DSSP for a single PDB ID.

    Returns None if files are missing (graceful degradation).
    DSSP is loaded from .dssp file if available; falls back to
    CIF _struct_conf / _struct_sheet_range annotations for SS codes.
    ASA (RSA) will use fill values in the fallback case.
    """
    pdb_dir = STRUCTURE_DIR / pdb_id
    cif_path = pdb_dir / f"{pdb_id}.cif"
    dssp_path = pdb_dir / f"{pdb_id}.dssp"

    if not cif_path.exists():
        return None

    try:
        # Load mmCIF structure
        parser = MMCIFParser(QUIET=True)
        structure = parser.get_structure(pdb_id, str(cif_path))

        resources = {"structure": structure}

        # Load DSSP — try file first, then CIF extraction
        dssp_loaded = False
        if dssp_path.exists():
            try:
                dssp_dict, _ = make_dssp_dict(str(dssp_path))
                if dssp_dict:
                    resources["dssp_dict"] = dssp_dict
                    dssp_loaded = True
            except Exception:
                pass

        if not dssp_loaded:
            # Fallback: compute SASA (Shrake-Rupley) + SS (CIF annotations)
            try:
                cif_dict = MMCIF2Dict.MMCIF2Dict(str(cif_path))
                ss_dict = _compute_dssp_dict_from_structure(structure, cif_dict)
                resources["dssp_dict"] = ss_dict if ss_dict else None
            except Exception:
                resources["dssp_dict"] = None

        return resources

    except Exception as e:
        warnings.warn(f"Failed to load PDB {pdb_id}: {e}")
        return None


def _build_alignment_map(
    ref_seq: str,
    pdb_chain,
    chain_id: str,
) -> dict[int, tuple]:
    """Map 1-based precursor positions → PDB residue IDs via gapped alignment.

    The reference is the full precursor (signal peptide included); the PDB chain
    is the mature protein (experimental) or full-length (AlphaFold). A global
    Needleman-Wunsch alignment makes signal-peptide trimming, M3-M4 loop
    deletions (common in crystallised constructs), and terminal truncations into
    explicit gaps — rather than shifting the whole mapping — so positions aligned
    to a gap are simply left unmapped.

    If the aligned (non-gap) identity is below 80%, the chain is a *different*
    subunit/isoform (homology-mapped or wrong accession); an empty map is
    returned so the gene falls back to fill values instead of silently emitting
    wrong-residue features. Returns {precursor_pos: residue.id}.
    """
    from Bio import Align

    pdb_residues = []
    pdb_res_ids = []
    for residue in pdb_chain:
        if is_aa(residue):
            three_letter = residue.resname
            if three_letter in protein_letters_3to1:
                pdb_residues.append(protein_letters_3to1[three_letter])
                pdb_res_ids.append(residue.id)

    pdb_sequence = "".join(pdb_residues)
    if not pdb_sequence:
        return {}

    aligner = Align.PairwiseAligner()
    aligner.mode = "global"
    aligner.substitution_matrix = Align.substitution_matrices.load("BLOSUM62")
    aligner.open_gap_score = -10.0
    aligner.extend_gap_score = -0.5

    try:
        alignments = aligner.align(ref_seq, pdb_sequence)
        if not alignments:
            return {}
        best = next(iter(alignments))
    except (OverflowError, ValueError, MemoryError):
        return {}

    # Identity over aligned (non-gap) columns; reject different subunits/isoforms.
    aligned_ref, aligned_pdb = best[0], best[1]
    cols = [(a, b) for a, b in zip(aligned_ref, aligned_pdb) if a != "-" and b != "-"]
    if not cols:
        return {}
    identity = sum(1 for a, b in cols if a == b) / len(cols)
    if identity < 0.80:
        warnings.warn(
            f"Low-identity mapping ({identity:.1%}) for chain {chain_id}; "
            f"gene is homology-mapped to a different subunit or uses a wrong isoform."
        )
        return {}

    alignment_map = {}
    for uniprot_block, pdb_block in zip(best.aligned[0], best.aligned[1]):
        uniprot_start, uniprot_end = uniprot_block
        pdb_start, pdb_end = pdb_block
        for i, pos in enumerate(range(uniprot_start, uniprot_end)):
            pdb_idx = pdb_start + i
            if pdb_idx < len(pdb_res_ids):
                alignment_map[pos + 1] = pdb_res_ids[pdb_idx]
    return alignment_map


def _build_all_alignment_maps(
    ref_seqs: dict[str, str],
    pdb_resources: dict[str, dict],
) -> dict[str, dict[int, tuple]]:
    """
    Build alignment maps for all genes.

    Returns: {gene: {uniprot_position: pdb_residue_id}}
    """
    maps = {}

    for gene in NACHR_GENES:
        if gene not in PDB_MAPPING or gene not in ref_seqs:
            continue

        pdb_info = PDB_MAPPING[gene]
        pdb_id = pdb_info["pdb_id"]
        chain_id = pdb_info["chain"]

        if pdb_id not in pdb_resources:
            continue

        resources = pdb_resources[pdb_id]
        if resources is None:
            continue

        try:
            structure = resources["structure"]
            pdb_chain = structure[0][chain_id]
            ref_seq = ref_seqs[gene]
            maps[gene] = _build_alignment_map(ref_seq, pdb_chain, chain_id)
        except (KeyError, Exception) as e:
            warnings.warn(f"Failed to build alignment for {gene} → {pdb_id}:{chain_id}: {e}")
            maps[gene] = {}

    return maps


# =============================================================================
# FEATURE COMPUTATION
# =============================================================================

def _compute_rsa(
    gene: str, position: int, alignment_map: dict,
    dssp_dict: Optional[dict], chain_id: str,
) -> float:
    """Compute Relative Solvent Accessibility."""
    pdb_res_id = alignment_map.get(position)
    if pdb_res_id is None or dssp_dict is None:
        return np.nan

    dssp_key = (chain_id, pdb_res_id)
    if dssp_key not in dssp_dict:
        return np.nan

    dssp_data = dssp_dict[dssp_key]
    aa_code = dssp_data[0]
    if aa_code.islower():
        aa_code = "C"

    if aa_code not in MAX_ASA:
        return np.nan

    try:
        return float(dssp_data[2]) / MAX_ASA[aa_code]
    except (ValueError, ZeroDivisionError):
        return np.nan


def _compute_cbeta_density(
    gene: str, position: int, alignment_map: dict,
    structure, chain_id: str, kdtree, cb_coords: np.ndarray,
) -> float:
    """Compute C-beta density (atoms within 10 Å)."""
    pdb_res_id = alignment_map.get(position)
    if pdb_res_id is None or cb_coords.size == 0:
        return np.nan

    try:
        pdb_chain = structure[0][chain_id]
        residue = pdb_chain[pdb_res_id]

        if residue.resname == "GLY":
            if "CA" not in residue:
                return np.nan
            query_coord = residue["CA"].coord
        else:
            if "CB" not in residue:
                return np.nan
            query_coord = residue["CB"].coord

        neighbors = kdtree.query_ball_point(query_coord, r=10.0)
        return len(neighbors) - 1  # exclude self
    except (KeyError, Exception):
        return np.nan


def _compute_b_factor(
    gene: str, position: int, alignment_map: dict,
    structure, chain_id: str, chain_medians: dict,
) -> float:
    """Compute average B-factor for a residue."""
    pdb_res_id = alignment_map.get(position)
    if pdb_res_id is None:
        return np.nan

    try:
        pdb_chain = structure[0][chain_id]
        residue = pdb_chain[pdb_res_id]
        atom_bfs = [atom.bfactor for atom in residue if atom.element != "H"]
        if atom_bfs:
            return float(np.mean(atom_bfs))
    except (KeyError, Exception):
        pass

    return chain_medians.get(chain_id, np.nan)


def _compute_dssp_ss(
    gene: str, position: int, alignment_map: dict,
    dssp_dict: Optional[dict], chain_id: str,
) -> str:
    """Get DSSP secondary structure code."""
    pdb_res_id = alignment_map.get(position)
    if pdb_res_id is None or dssp_dict is None:
        return "-"

    dssp_key = (chain_id, pdb_res_id)
    if dssp_key in dssp_dict:
        return dssp_dict[dssp_key][1]
    return "-"


# =============================================================================
# STRUCTURAL EXTRACTOR
# =============================================================================

class StructuralExtractor(FeatureExtractor):
    """
    Extracts 5 core structural features from PDB structures.

    Features:
      - rsa: Relative Solvent Accessibility
      - cbeta_density: C-beta packing density (atoms within 10 Å)
      - b_factor: Average residue B-factor
      - dssp_helix: 1 if DSSP helix (H/G/I)
      - dssp_sheet: 1 if DSSP sheet (E/B)
      - dssp_coil: 1 if DSSP coil/loop/turn (everything else)
      - is_unmappable: 1 if position can't be mapped to PDB

    Note: dssp_helix + dssp_sheet + dssp_coil sum to 1 (one-hot).
    """

    name = "structural_core"
    # 7 columns: rsa, cbeta_density, b_factor, dssp_helix, dssp_sheet, dssp_coil, is_unmappable
    n_features = 7
    feature_names = [
        "rsa", "cbeta_density", "b_factor",
        "dssp_helix", "dssp_sheet", "dssp_coil",
        "is_unmappable",
    ]

    def requires_pdb(self) -> bool:
        return True

    def requires_reference(self) -> bool:
        return True

    def extract(self, df, ref_seqs=None, pdb_resources=None):
        n = len(df)
        features = np.zeros((n, self.n_features), dtype=np.float64)

        # If no PDB resources, return fill values
        if pdb_resources is None:
            features[:, 0] = STRUCTURAL_FILL_VALUES["rsa"]       # rsa = 1.0
            features[:, 1] = STRUCTURAL_FILL_VALUES["cbeta_density"]  # 0
            features[:, 2] = 0.0                                   # b_factor
            features[:, 5] = 1.0                                   # dssp_coil
            features[:, 6] = 1.0                                   # is_unmappable
            return features

        # Build alignment maps
        if ref_seqs is None:
            features[:, 6] = 1.0
            return features

        alignment_maps = _build_all_alignment_maps(ref_seqs, pdb_resources)

        # Precompute KD trees and chain medians per PDB
        pdb_kdtrees = {}
        pdb_chain_medians = {}
        for pdb_id, resources in pdb_resources.items():
            if resources is None:
                continue

            structure = resources["structure"]

            # Build KDTree for C-beta atoms
            cb_coords = []
            for model in structure:
                for chain in model:
                    for residue in chain:
                        if is_aa(residue):
                            if residue.resname == "GLY":
                                if "CA" in residue:
                                    cb_coords.append(residue["CA"].coord)
                            else:
                                if "CB" in residue:
                                    cb_coords.append(residue["CB"].coord)

            if cb_coords:
                pdb_kdtrees[pdb_id] = (cKDTree(np.array(cb_coords)), np.array(cb_coords))
            else:
                pdb_kdtrees[pdb_id] = (None, np.array([]))

            # Compute chain median B-factors
            chain_medians = {}
            for model in structure:
                for chain in model:
                    chain_id = chain.id
                    bf_vals = []
                    for residue in chain:
                        if is_aa(residue):
                            heavy_bfs = [atom.bfactor for atom in residue if atom.element != "H"]
                            if heavy_bfs:
                                bf_vals.append(float(np.mean(heavy_bfs)))
                    chain_medians[chain_id] = float(np.median(bf_vals)) if bf_vals else np.nan

            pdb_chain_medians[pdb_id] = chain_medians

        # Extract features row by row
        for i, (_, row) in enumerate(df.iterrows()):
            gene = str(row.get("subunit", "")).upper()
            position = int(row.get("position", 0))

            if gene not in PDB_MAPPING or gene not in alignment_maps:
                features[i, 0] = STRUCTURAL_FILL_VALUES["rsa"]
                features[i, 1] = STRUCTURAL_FILL_VALUES["cbeta_density"]
                features[i, 5] = 1.0  # coil
                features[i, 6] = 1.0  # unmappable
                continue

            pdb_info = PDB_MAPPING[gene]
            pdb_id = pdb_info["pdb_id"]
            chain_id = pdb_info["chain"]
            alignment_map = alignment_maps.get(gene, {})

            resources = pdb_resources.get(pdb_id)
            if resources is None:
                features[i, 6] = 1.0
                continue

            # RSA
            rsa = _compute_rsa(gene, position, alignment_map,
                              resources.get("dssp_dict"), chain_id)
            features[i, 0] = rsa if not np.isnan(rsa) else STRUCTURAL_FILL_VALUES["rsa"]

            # C-beta density
            kdtree, cb_coords_arr = pdb_kdtrees.get(pdb_id, (None, np.array([])))
            if kdtree is not None:
                cb_density = _compute_cbeta_density(
                    gene, position, alignment_map,
                    resources["structure"], chain_id, kdtree, cb_coords_arr
                )
                features[i, 1] = cb_density if not np.isnan(cb_density) else STRUCTURAL_FILL_VALUES["cbeta_density"]
            else:
                features[i, 1] = STRUCTURAL_FILL_VALUES["cbeta_density"]

            # B-factor
            bf = _compute_b_factor(gene, position, alignment_map,
                                   resources["structure"], chain_id,
                                   pdb_chain_medians.get(pdb_id, {}))
            features[i, 2] = bf if not np.isnan(bf) else 0.0

            # DSSP → helix/sheet/coil one-hot
            ss = _compute_dssp_ss(gene, position, alignment_map,
                                  resources.get("dssp_dict"), chain_id)
            h, s, c = _dssp_to_flags(ss)
            features[i, 3] = h
            features[i, 4] = s
            features[i, 5] = c

            # Unmappable
            features[i, 6] = 0.0 if alignment_map.get(position) is not None else 1.0

        return features


# =============================================================================
# PDB RESOURCE MANAGEMENT
# =============================================================================

# Module-level cache for PDB resources
_pdb_resource_cache: dict[str, Optional[dict]] = {}


def load_all_pdb_resources(
    pdb_ids: Optional[list[str]] = None,
    force_reload: bool = False,
) -> dict[str, Optional[dict]]:
    """
    Load PDB resources for all required structures.

    Parameters
    ----------
    pdb_ids : list[str], optional
        Specific PDB IDs to load. If None, loads all from PDB_MAPPING.
    force_reload : bool
        If True, bypass cache and reload.

    Returns
    -------
    dict[str, Optional[dict]]
        {pdb_id: resources_dict or None if unavailable}
    """
    global _pdb_resource_cache

    if pdb_ids is None:
        pdb_ids = sorted(set(
            info["pdb_id"] for info in PDB_MAPPING.values()
        ))

    if not force_reload and _pdb_resource_cache:
        # Return cached, loading any missing
        for pdb_id in pdb_ids:
            if pdb_id not in _pdb_resource_cache:
                _pdb_resource_cache[pdb_id] = _load_pdb_resources(pdb_id)
        return _pdb_resource_cache

    _pdb_resource_cache = {
        pdb_id: _load_pdb_resources(pdb_id) for pdb_id in pdb_ids
    }
    return _pdb_resource_cache


def get_pdb_availability() -> dict[str, bool]:
    """Check which PDB structures are available on disk."""
    availability = {}
    pdb_ids = sorted(set(info["pdb_id"] for info in PDB_MAPPING.values()))
    for pdb_id in pdb_ids:
        cif_path = STRUCTURE_DIR / pdb_id / f"{pdb_id}.cif"
        availability[pdb_id] = cif_path.exists()
    return availability
