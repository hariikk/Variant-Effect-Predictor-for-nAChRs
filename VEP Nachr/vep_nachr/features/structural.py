"""
Structural feature extraction from PDB/CIF files.

Extracts per-residue structural features for nAChR mutations:
- RSA (Relative Solvent Accessibility) from DSSP algorithm
- B-factor (temperature factor / atomic mobility)
- DSSP secondary structure (helix, sheet, coil binary flags)
- C-beta density (packing density within 10A radius)

Uses BioPython for PDB/CIF parsing, pairwise alignment for
mapping UniProt positions to PDB residue numbers, and scipy
KDTree for efficient neighbor searches.

Adapted from VEP-Enac/vep/features/noah_features/structural_features.py
with support for multiple PDB structures (ENaC used a single PDB).
"""

import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from Bio import Align, SeqIO
from Bio.PDB import MMCIFParser
from Bio.PDB.Polypeptide import is_aa, protein_letters_3to1
from scipy.spatial import cKDTree

from vep_nachr.config import (
    CANONICAL_ACCESSIONS,
    PDB_MAPPING,
    STRUCTURE_DIR,
    SUBUNIT_TO_FASTA_DIR,
)


# Max ASA values for RSA normalization (Tien et al. 2013, empirical-GXG tripeptide)
MAX_ASA = {
    "A": 129.0, "R": 274.0, "N": 195.0, "D": 193.0, "C": 167.0,
    "E": 223.0, "Q": 225.0, "G": 104.0, "H": 224.0, "I": 197.0,
    "L": 201.0, "K": 236.0, "M": 224.0, "F": 240.0, "P": 159.0,
    "S": 155.0, "T": 172.0, "W": 285.0, "Y": 263.0, "V": 174.0,
}

# DSSP secondary structure code groups
_DSSP_HELIX_CODES = frozenset("HGI")  # alpha, 3-10, pi helix
_DSSP_SHEET_CODES = frozenset("EB")   # extended strand, beta-bridge


def _dssp_to_flags(ss_code: object) -> tuple[float, float, float]:
    """Convert DSSP code to (helix, sheet, coil) binary flags."""
    if ss_code is None or not isinstance(ss_code, str):
        return 0.0, 0.0, 1.0
    helix = 1.0 if ss_code in _DSSP_HELIX_CODES else 0.0
    sheet = 1.0 if ss_code in _DSSP_SHEET_CODES else 0.0
    coil = 0.0 if (helix or sheet) else 1.0
    return helix, sheet, coil


# ============================================================================
# STRUCTURE LOADING
# ============================================================================

def _load_structure(pdb_id: str) -> object:
    """Load a PDB/CIF structure file."""
    cif_path = STRUCTURE_DIR / f"{pdb_id}.cif"
    pdb_path = STRUCTURE_DIR / f"{pdb_id}.pdb"

    if cif_path.exists():
        parser = MMCIFParser(QUIET=True)
        return parser.get_structure(pdb_id, str(cif_path))

    if pdb_path.exists():
        from Bio.PDB import PDBParser
        pdb_parser = PDBParser(QUIET=True)
        return pdb_parser.get_structure(pdb_id, str(pdb_path))

    raise FileNotFoundError(
        f"No structure file found for {pdb_id}. "
        f"Expected {cif_path} or {pdb_path}"
    )


def _get_chain_sequence_and_residues(structure, chain_id: str):
    """Extract amino acid sequence and residue IDs from a PDB chain."""
    try:
        chain = structure[0][chain_id]
    except KeyError:
        raise ValueError(f"Chain {chain_id} not found in structure")

    residues = []
    res_ids = []
    for residue in chain:
        if is_aa(residue) and residue.resname in protein_letters_3to1:
            residues.append(protein_letters_3to1[residue.resname])
            res_ids.append(residue.id)

    return "".join(residues), res_ids


def _load_canonical_sequence(subunit: str) -> str:
    """Load the canonical FASTA sequence for a subunit."""
    fasta_dir_name = SUBUNIT_TO_FASTA_DIR.get(subunit)
    if fasta_dir_name is None:
        raise ValueError(f"Unknown subunit: {subunit}")

    # Look for FASTA in the protein_scequences folder (parent of project)
    project_root = Path(__file__).parent.parent.parent
    source_dir = project_root.parent

    fasta_path = source_dir / "protein_scequences" / fasta_dir_name / "ncbi_dataset" / "data" / "protein.faa"
    if not fasta_path.exists():
        raise FileNotFoundError(f"FASTA file not found: {fasta_path}")

    accession = CANONICAL_ACCESSIONS.get(subunit)
    for record in SeqIO.parse(str(fasta_path), "fasta"):
        if accession and accession in record.id:
            return str(record.seq)

    # Fallback: return the first sequence
    for record in SeqIO.parse(str(fasta_path), "fasta"):
        return str(record.seq)

    raise ValueError(f"No sequences found in {fasta_path}")


# ============================================================================
# ALIGNMENT: UniProt position -> PDB residue mapping
# ============================================================================

def _build_alignment_map(
    uniprot_seq: str,
    pdb_seq: str,
    pdb_res_ids: list,
) -> dict[int, tuple | None]:
    """
    Build mapping from 1-based UniProt positions to PDB residue IDs
    using BLOSUM62 global pairwise alignment.

    Returns dict: {uniprot_pos: pdb_residue_id or None}
    """
    aligner = Align.PairwiseAligner()
    aligner.mode = "global"
    aligner.substitution_matrix = Align.substitution_matrices.load("BLOSUM62")

    alignments = aligner.align(uniprot_seq, pdb_seq)
    best = alignments[0]

    alignment_map = {}
    aligned_uniprot = best.aligned[0]
    aligned_pdb = best.aligned[1]

    for uniprot_block, pdb_block in zip(aligned_uniprot, aligned_pdb):
        uni_start, uni_end = uniprot_block
        pdb_start, pdb_end = pdb_block

        if pdb_start == pdb_end:
            # UniProt positions with no PDB match (gap in PDB)
            for pos in range(uni_start, uni_end):
                alignment_map[pos + 1] = None
        else:
            for i, pos in enumerate(range(uni_start, uni_end)):
                pdb_idx = pdb_start + i
                if pdb_idx < len(pdb_res_ids):
                    alignment_map[pos + 1] = pdb_res_ids[pdb_idx]
                else:
                    alignment_map[pos + 1] = None

    return alignment_map


# ============================================================================
# SECONDARY STRUCTURE FROM mmCIF
# ============================================================================

def _parse_ss_from_mmcif(cif_path: str, chain_id: str) -> dict[int, str]:
    """
    Parse secondary structure annotations from mmCIF file.
    Returns dict: {residue_number: ss_code} where ss_code is 'H', 'E', or '-'.
    """
    from Bio.PDB import MMCIF2Dict

    mmcif_dict = MMCIF2Dict.MMCIF2Dict(cif_path)
    ss_assignments = {}

    # Parse helices from _struct_conf
    if "_struct_conf.conf_type_id" in mmcif_dict:
        conf_types = mmcif_dict["_struct_conf.conf_type_id"]
        beg_chains = mmcif_dict["_struct_conf.beg_auth_asym_id"]
        beg_seqs = mmcif_dict["_struct_conf.beg_auth_seq_id"]
        end_chains = mmcif_dict["_struct_conf.end_auth_asym_id"]
        end_seqs = mmcif_dict["_struct_conf.end_auth_seq_id"]

        if not isinstance(conf_types, list):
            conf_types = [conf_types]
            beg_chains = [beg_chains]
            beg_seqs = [beg_seqs]
            end_chains = [end_chains]
            end_seqs = [end_seqs]

        for conf_type, bc, bs, ec, es in zip(
            conf_types, beg_chains, beg_seqs, end_chains, end_seqs
        ):
            if bc == chain_id and conf_type.startswith("HELX"):
                for res_num in range(int(bs), int(es) + 1):
                    ss_assignments[res_num] = "H"

    # Parse sheets from _struct_sheet_range
    if "_struct_sheet_range.sheet_id" in mmcif_dict:
        beg_chains = mmcif_dict["_struct_sheet_range.beg_auth_asym_id"]
        beg_seqs = mmcif_dict["_struct_sheet_range.beg_auth_seq_id"]
        end_chains = mmcif_dict["_struct_sheet_range.end_auth_asym_id"]
        end_seqs = mmcif_dict["_struct_sheet_range.end_auth_seq_id"]

        if not isinstance(beg_chains, list):
            beg_chains = [beg_chains]
            beg_seqs = [beg_seqs]
            end_chains = [end_chains]
            end_seqs = [end_seqs]

        for bc, bs, ec, es in zip(beg_chains, beg_seqs, end_chains, end_seqs):
            if bc == chain_id:
                for res_num in range(int(bs), int(es) + 1):
                    ss_assignments[res_num] = "E"

    return ss_assignments


# ============================================================================
# FEATURE EXTRACTION FUNCTIONS
# ============================================================================

def _compute_bfactor(residue) -> float:
    """Compute average B-factor for a residue (heavy atoms only)."""
    b_factors = [atom.bfactor for atom in residue if atom.element != "H"]
    if not b_factors:
        return np.nan
    return float(np.mean(b_factors))


def _build_cbeta_kdtree(structure, chain_id: str):
    """
    Build a KDTree of C-beta atom coordinates for a specific chain.
    Returns (kdtree, coords_array) or (None, None) if no atoms found.
    """
    coords = []
    try:
        chain = structure[0][chain_id]
    except KeyError:
        return None, None

    for residue in chain:
        if not is_aa(residue):
            continue
        if residue.resname == "GLY":
            if "CA" in residue:
                coords.append(residue["CA"].coord)
        else:
            if "CB" in residue:
                coords.append(residue["CB"].coord)

    if not coords:
        return None, None

    coords = np.array(coords)
    return cKDTree(coords), coords


def _compute_cbeta_density(residue, kdtree, radius: float = 10.0) -> float:
    """Count C-beta atoms within radius of this residue."""
    if kdtree is None:
        return np.nan

    if residue.resname == "GLY":
        atom_name = "CA"
    else:
        atom_name = "CB"

    if atom_name not in residue:
        return np.nan

    query_coord = residue[atom_name].coord
    neighbors = kdtree.query_ball_point(query_coord, r=radius)
    return float(len(neighbors) - 1)  # exclude self


# ============================================================================
# DSSP COMPUTATION
# ============================================================================

def _compute_dssp_data(structure, chain_id: str) -> dict:
    """
    Compute DSSP-like data for a chain. Tries BioPython DSSP first,
    falls back to mmCIF secondary structure annotations.

    Returns dict: {residue_id: {'ss': str, 'asa': float_or_None}}
    """
    # Try BioPython DSSP (requires mkdssp binary installed)
    try:
        from Bio.PDB.DSSP import DSSP
        model = structure[0]
        cif_path = STRUCTURE_DIR / f"{structure.id}.cif"
        if cif_path.exists():
            dssp = DSSP(model, str(cif_path), dssp="mkdssp", acc_array="Wilke")
            result = {}
            for key in dssp:
                res_chain, res_id = key[0], key[1]
                if res_chain == chain_id:
                    dssp_data = dssp[key]
                    result[res_id] = {
                        "ss": dssp_data[2],
                        "asa": dssp_data[3],
                    }
            if result:
                return result
    except Exception:
        pass  # mkdssp not installed or failed — use fallback

    # Fallback: parse SS from mmCIF annotations
    cif_path = STRUCTURE_DIR / f"{structure.id}.cif"
    if not cif_path.exists():
        return {}

    ss_map = _parse_ss_from_mmcif(str(cif_path), chain_id)

    result = {}
    try:
        chain = structure[0][chain_id]
    except KeyError:
        return {}

    for residue in chain:
        if not is_aa(residue):
            continue
        res_id = residue.id
        res_num = res_id[1]
        result[res_id] = {
            "ss": ss_map.get(res_num, "-"),
            "asa": None,  # ASA not available without DSSP binary
        }

    return result


# ============================================================================
# MAIN INTERFACE
# ============================================================================

def structural_features_available() -> bool:
    """Check if any PDB structures are configured and files exist."""
    for subunit, mapping in PDB_MAPPING.items():
        if mapping["pdb_id"] is not None:
            pdb_id = mapping["pdb_id"]
            cif_path = STRUCTURE_DIR / f"{pdb_id}.cif"
            pdb_path = STRUCTURE_DIR / f"{pdb_id}.pdb"
            if cif_path.exists() or pdb_path.exists():
                return True
    return False


def extract_structural_features(
    df: pd.DataFrame,
    wildtype_sequences: Optional[dict[str, str]] = None,
) -> np.ndarray:
    """
    Extract structural features for each mutation in the DataFrame.

    For each mutation, maps the UniProt position to the PDB residue
    using BLOSUM62 global alignment, then extracts:
    - RSA (relative solvent accessibility)
    - B-factor (temperature factor)
    - DSSP helix/sheet/coil (binary flags)
    - C-beta density (packing within 10A)

    Mutations in subunits without PDB structures get imputed defaults.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned mutation data with 'subunit' and 'position' columns.
    wildtype_sequences : dict, optional
        Pre-loaded wildtype sequences {subunit: sequence}.

    Returns
    -------
    np.ndarray
        Feature matrix (n_samples, 6).
    """
    if not structural_features_available():
        warnings.warn(
            "No PDB structures found. Using imputed structural features. "
            "Download CIF/PDB files and update config.PDB_MAPPING."
        )
        return _imputed_structural_features(len(df))

    # Caches to avoid reloading/recomputing for each row
    _structure_cache = {}    # pdb_id -> Structure
    _alignment_cache = {}    # subunit -> alignment_map
    _dssp_cache = {}         # (pdb_id, chain) -> dssp_data
    _kdtree_cache = {}       # (pdb_id, chain) -> (kdtree, coords)
    _bfactor_cache = {}      # (pdb_id, chain) -> {res_id: bfactor}
    _chain_median_bf = {}    # (pdb_id, chain) -> median bfactor
    _chain_residue_list = {} # (pdb_id, chain) -> [(res_id, bfactor), ...]

    features = np.zeros((len(df), 6))
    n_mapped = 0
    n_imputed = 0

    for i, (idx, row) in enumerate(df.iterrows()):
        subunit = row.get("subunit", row.get("protein_subunit"))
        position = int(row.get("position", row.get("mutation_position", 0)))

        mapping = PDB_MAPPING.get(subunit)
        if mapping is None or mapping["pdb_id"] is None:
            features[i] = _imputed_row()
            n_imputed += 1
            continue

        pdb_id = mapping["pdb_id"]
        chain_id = mapping["chain"]

        # Load structure (cached)
        if pdb_id not in _structure_cache:
            try:
                _structure_cache[pdb_id] = _load_structure(pdb_id)
            except FileNotFoundError:
                features[i] = _imputed_row()
                n_imputed += 1
                continue

        structure = _structure_cache[pdb_id]

        # Build alignment map (cached per subunit)
        if subunit not in _alignment_cache:
            try:
                if wildtype_sequences and subunit in wildtype_sequences:
                    uniprot_seq = wildtype_sequences[subunit]
                else:
                    uniprot_seq = _load_canonical_sequence(subunit)

                pdb_seq, pdb_res_ids = _get_chain_sequence_and_residues(
                    structure, chain_id
                )
                _alignment_cache[subunit] = _build_alignment_map(
                    uniprot_seq, pdb_seq, pdb_res_ids
                )
            except Exception as e:
                warnings.warn(f"Alignment failed for {subunit}: {e}")
                _alignment_cache[subunit] = {}

        alignment_map = _alignment_cache[subunit]

        # Look up PDB residue for this UniProt position
        pdb_res_id = alignment_map.get(position)
        if pdb_res_id is None:
            features[i] = _imputed_row()
            n_imputed += 1
            continue

        # DSSP data (cached per pdb_id + chain)
        dssp_key = (pdb_id, chain_id)
        if dssp_key not in _dssp_cache:
            _dssp_cache[dssp_key] = _compute_dssp_data(structure, chain_id)

        dssp_data = _dssp_cache[dssp_key]

        # Get PDB residue object
        try:
            pdb_chain = structure[0][chain_id]
            residue = pdb_chain[pdb_res_id]
        except (KeyError, Exception):
            features[i] = _imputed_row()
            n_imputed += 1
            continue

        # --- RSA ---
        rsa = np.nan
        dssp_entry = dssp_data.get(pdb_res_id, {})
        asa = dssp_entry.get("asa")
        if asa is not None:
            aa_code = protein_letters_3to1.get(residue.resname, "X")
            if aa_code.islower():
                aa_code = "C"
            max_asa = MAX_ASA.get(aa_code)
            if max_asa and max_asa > 0:
                rsa = float(asa) / max_asa
        if np.isnan(rsa):
            rsa = 1.0  # impute as fully exposed

        # --- B-factor (with fallback) ---
        if dssp_key not in _bfactor_cache:
            bf_dict = {}
            res_list = []
            for r in pdb_chain:
                if is_aa(r):
                    bf = _compute_bfactor(r)
                    bf_dict[r.id] = bf
                    if not np.isnan(bf):
                        res_list.append((r.id, bf))
            _bfactor_cache[dssp_key] = bf_dict
            _chain_residue_list[dssp_key] = res_list
            bf_values = [bf for _, bf in res_list]
            _chain_median_bf[dssp_key] = (
                float(np.median(bf_values)) if bf_values else 0.0
            )

        bfactor = _bfactor_cache[dssp_key].get(pdb_res_id, np.nan)
        if np.isnan(bfactor):
            # Fallback 1: average of adjacent residues
            res_list = _chain_residue_list[dssp_key]
            res_id_list = [rid for rid, _ in res_list]
            try:
                idx_in_list = res_id_list.index(pdb_res_id)
                neighbor_vals = []
                if idx_in_list > 0:
                    v = res_list[idx_in_list - 1][1]
                    if not np.isnan(v):
                        neighbor_vals.append(v)
                if idx_in_list < len(res_list) - 1:
                    v = res_list[idx_in_list + 1][1]
                    if not np.isnan(v):
                        neighbor_vals.append(v)
                if neighbor_vals:
                    bfactor = float(np.mean(neighbor_vals))
            except ValueError:
                pass

            if np.isnan(bfactor):
                bfactor = _chain_median_bf[dssp_key]

        # --- DSSP secondary structure ---
        ss_code = dssp_entry.get("ss")
        helix, sheet, coil = _dssp_to_flags(ss_code)

        # --- C-beta density ---
        if dssp_key not in _kdtree_cache:
            _kdtree_cache[dssp_key] = _build_cbeta_kdtree(structure, chain_id)

        kdtree, _ = _kdtree_cache[dssp_key]
        cbeta = _compute_cbeta_density(residue, kdtree)
        if np.isnan(cbeta):
            cbeta = 0.0

        features[i] = [rsa, bfactor, helix, sheet, coil, cbeta]
        n_mapped += 1

    total = len(df)
    if n_imputed > 0:
        warnings.warn(
            f"Structural features: {n_mapped}/{total} mapped to PDB, "
            f"{n_imputed}/{total} imputed (no structure or unmappable position)."
        )

    return features


def _imputed_row() -> np.ndarray:
    """Default values for a single unmappable residue."""
    return np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])


def _imputed_structural_features(n_samples: int) -> np.ndarray:
    """Return imputed structural features when PDB data is unavailable."""
    features = np.zeros((n_samples, 6))
    features[:, 0] = 1.0  # RSA = fully exposed
    features[:, 4] = 1.0  # coil = 1
    return features


def get_structural_feature_names() -> list[str]:
    """Get the names of structural features."""
    return [
        "rsa",
        "bfactor",
        "dssp_helix",
        "dssp_sheet",
        "dssp_coil",
        "cbeta_density",
    ]
