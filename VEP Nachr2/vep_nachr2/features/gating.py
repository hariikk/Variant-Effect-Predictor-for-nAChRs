"""
Gating / allosteric-dynamics structural features (Run XXV).

Literature basis
----------------
- The M2 9' Leu gate is the canonical direction hotspot in nAChR: Leu9' ->
  polar/smaller = slow-channel GOF (prolonged openings, constitutive activation);
  fast-channel LOF also localises to M2. Direction is a property of the *gating
  equilibrium*, not of a single static residue.
- PreMode (Zhong et al., Nat. Comms. 2025) found GoF variants enrich at the
  inner-helix gate but states that static structure is insufficient for ion
  channels and protein *dynamics* is the missing signal.
- Ersoy et al. (eLife 2023) computed per-residue allosteric coupling to the CFTR
  gate with a Gaussian Network Model and found the coupling hotspots overlap
  disease-causing mutations.

This extractor computes GNM (Gaussian Network Model, Bahar/Atilgan) derived
features on the C-alpha backbone of each structure:

  gnm_msf            theoretical mean-square fluctuation (thermal flexibility —
                     complements the *experimental* b_factor already in core)
  gnm_slowmode       participation in the 2 slowest collective modes (the global
                     hinge / gating motions)
  gnm_gate_coupling  cross-correlation of the residue to the channel gate (the
                     9' constriction ring). 0 for the 7 AlphaFold-monomer genes
                     (no assembled pore to define a gate).

NOT in the default 75-feature set — added as an additive group in Run XXV
(baseline 75 vs 75+gating). The GNM coupling is the genuinely-new quantity:
Runs XV/XVII/XVIII used boolean membership / static contact graphs / spatial
gate proximity, and Run XX used an ESM-2 *attention* proxy — none computed the
mechanical fluctuation correlation that defines allosteric coupling.
"""

import warnings
from typing import Optional

import numpy as np
import pandas as pd
from Bio.PDB import is_aa

from vep_nachr2.features.base import FeatureExtractor
from vep_nachr2.features.structural import _build_all_alignment_maps
from vep_nachr2.config import PDB_MAPPING, NACHR_GENES

# GNM interaction cutoff (Å) — Bahar default.
_GNM_CUTOFF = 7.3
# Gate ring = residues within this radial distance of the innermost constriction.
_GATE_RING_TOL = 4.0


def _gnm_decompose(coords: np.ndarray, cutoff: float = _GNM_CUTOFF):
    """Gaussian Network Model eigendecomposition of C-alpha coordinates.

    Returns
    -------
    msf : (N,) mean-square fluctuation (diagonal of Gamma pseudoinverse)
    slowmode : (N,) participation in the 2 slowest nonzero modes
    v : (N, N) eigenvectors (ascending eigenvalues)
    inv_w : (N,) inverse eigenvalues (0 for the rigid-body zero mode)
    """
    n = len(coords)
    diff = coords[:, None, :] - coords[None, :, :]
    dist2 = (diff ** 2).sum(-1)

    gamma = np.zeros((n, n), dtype=np.float64)
    contact = (dist2 < cutoff ** 2) & ~np.eye(n, dtype=bool)
    gamma[contact] = -1.0
    np.fill_diagonal(gamma, -gamma.sum(axis=1))

    w, v = np.linalg.eigh(gamma)  # ascending
    nonzero = w > 1e-8
    inv_w = np.where(nonzero, 1.0 / np.maximum(w, 1e-8), 0.0)

    # MSF = diagonal of pinv(Gamma) = sum_k v_ik^2 / w_k
    msf = (v ** 2) @ inv_w

    # 2 slowest nonzero modes (skip index 0 = zero rigid-translation mode)
    slow_idx = np.where(nonzero)[0][:2]
    slowmode = (v[:, slow_idx] ** 2).mean(axis=1) if len(slow_idx) else np.zeros(n)

    return msf, slowmode, v, inv_w


def _collect_ca(structure):
    """Collect (chain_id, res_id, coord) for all protein C-alpha atoms."""
    refs, coords = [], []
    for model in structure:
        for chain in model:
            for res in chain:
                if is_aa(res) and "CA" in res:
                    refs.append((chain.id, res.id))
                    coords.append(res["CA"].coord)
    return refs, np.array(coords, dtype=np.float64)


def _identify_gate(coords: np.ndarray, tol: float = _GATE_RING_TOL):
    """Identify the channel-gate (9' constriction) residue indices.

    The pore axis is the principal axis of the C-alpha cloud (for a pentameric
    pLGIC the longest dimension is the ECD->ICD pore axis). Pore-lining M2
    residues are the innermost; the gate ring is the set within `tol` Å of the
    minimum radial distance.
    """
    centroid = coords.mean(axis=0)
    centered = coords - centroid
    cov = centered.T @ centered / len(coords)
    _, eigvecs = np.linalg.eigh(cov)
    axis = eigvecs[:, -1]  # largest variance = pore axis

    proj = (centered @ axis)[:, None] * axis[None, :]
    radial = np.linalg.norm(centered - proj, axis=1)

    min_radial = radial.min()
    gate = np.where(radial < min_radial + tol)[0]
    return gate, radial


def _build_gnm_tables(pdb_resources: dict):
    """Precompute GNM quantities per PDB id.

    Returns {pdb_id: dict} with:
      index_map : {(chain_id, res_id): index into the C-alpha list}
      msf, slowmode : per-residue GNM quantities
      gate_idx : indices of gate-ring residues (empty for monomers)
      gate_coupling : per-residue |cross-correlation| to the gate (0 for monomers)
    """
    tables = {}
    for pdb_id, resources in pdb_resources.items():
        if resources is None:
            continue
        structure = resources.get("structure")
        if structure is None:
            continue

        refs, coords = _collect_ca(structure)
        if len(coords) < 2:
            continue

        msf, slowmode, v, inv_w = _gnm_decompose(coords)

        # Index map (chain_id, res_id) -> array index
        index_map = {(c, r): i for i, (c, r) in enumerate(refs)}

        n_chains = len({c for c, _ in refs})
        gate_idx = np.array([], dtype=int)
        gate_coupling = np.zeros(len(coords), dtype=np.float64)

        if n_chains >= 2:
            try:
                gate_idx, _ = _identify_gate(coords)
            except Exception:
                gate_idx = np.array([], dtype=int)

        if len(gate_idx):
            # pinv[:, gate] = sum_k v_ik inv_w_k v_gate_k  (N x n_gate)
            pinv_cols_gate = (v * inv_w) @ v[gate_idx, :].T
            pinv_diag = (v ** 2) @ inv_w
            gate_diag = pinv_diag[gate_idx]
            denom = np.sqrt(np.maximum(pinv_diag[:, None] * gate_diag[None, :], 1e-12))
            corr = pinv_cols_gate / denom
            gate_coupling = np.abs(corr).mean(axis=1)

        tables[pdb_id] = {
            "index_map": index_map,
            "msf": msf,
            "slowmode": slowmode,
            "gate_coupling": gate_coupling,
            "n_residues": len(coords),
            "n_gate": int(len(gate_idx)),
        }
    return tables


class GatingExtractor(FeatureExtractor):
    """GNM allosteric-dynamics features (Run XXV, additive — not default)."""

    name = "gating"
    n_features = 3
    feature_names = ["gnm_msf", "gnm_slowmode", "gnm_gate_coupling"]

    def requires_pdb(self) -> bool:
        return True

    def requires_reference(self) -> bool:
        return True

    def extract(self, df, ref_seqs=None, pdb_resources=None):
        n = len(df)
        feats = np.zeros((n, self.n_features), dtype=np.float64)

        if pdb_resources is None or ref_seqs is None:
            # Unmappable default: neutral coupling, zero flexibility signal.
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

            feats[i, 0] = table["msf"][idx]
            feats[i, 1] = table["slowmode"][idx]
            feats[i, 2] = table["gate_coupling"][idx]

        return feats
