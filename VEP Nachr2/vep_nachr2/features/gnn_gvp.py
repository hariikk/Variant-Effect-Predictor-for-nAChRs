"""
Pretrained GVP-GNN node embeddings (structure-based, GNN feature arm).

Uses the *already cached* ESM-IF1 checkpoint (`esm_if1_gvp4_t16_142M_UR50`),
whose encoder is a 4-layer Geometric Vector Perceptron graph neural network
over backbone atoms. That encoder is a pretrained protein structure GNN — the
same family as GVP-GNN / GearNet / ProNet used in the literature — so we can
read per-residue node embeddings without installing torchdrug (which does not
build on Windows/Python 3.11).

For each variant we take the 512-dim node embedding of the mutated residue,
project to N_PCA_COMPONENTS with PCA (unsupervised, fitted on variants), and
return those components as features. If the checkpoint or structure is
unavailable, the extractor degrades to zero-filled features.

Note: Run XIX tested ESM-IF *ΔΔG* (a scalar energy difference). This extractor
uses the raw GVP node representation, which is a different signal.
"""

import warnings

import numpy as np
import pandas as pd

from vep_nachr2.config import PDB_MAPPING, STRUCTURE_DIR
from vep_nachr2.features.base import FeatureExtractor

N_PCA_COMPONENTS = 16
EMB_DIM = 512

_MODEL = None
_MODEL_ERROR = None


def _load_esmif():
    """Lazy-load the ESM-IF1 model (torch.load needs weights_only=False)."""
    global _MODEL, _MODEL_ERROR
    if _MODEL is not None or _MODEL_ERROR is not None:
        return _MODEL
    try:
        import torch

        # torch >= 2.6 defaults weights_only=True; fair-esm checkpoints need False.
        _orig_load = torch.load

        def _patched_load(*args, **kwargs):
            kwargs.setdefault("weights_only", False)
            return _orig_load(*args, **kwargs)

        torch.load = _patched_load
        try:
            from esm.pretrained import esm_if1_gvp4_t16_142M_UR50

            model, alphabet = esm_if1_gvp4_t16_142M_UR50()
        finally:
            torch.load = _orig_load

        model.eval()
        _MODEL = (model, alphabet)
    except Exception as exc:  # pragma: no cover - environment dependent
        _MODEL_ERROR = f"{type(exc).__name__}: {exc}"
        warnings.warn(f"ESM-IF GVP unavailable: {_MODEL_ERROR}")
    return _MODEL


def _map_ref_to_chain(ref_seq: str, chain_seq: str, min_identity: float = 0.80):
    """1-based reference position -> 0-based index in the ESM-IF chain sequence.

    Global BLOSUM62 alignment (same gate as structural.py): below
    `min_identity` the chain is a different subunit/isoform and we return {}.
    """
    from Bio import Align

    if not ref_seq or not chain_seq:
        return {}
    aligner = Align.PairwiseAligner()
    aligner.mode = "global"
    aligner.substitution_matrix = Align.substitution_matrices.load("BLOSUM62")
    aligner.open_gap_score = -10.0
    aligner.extend_gap_score = -0.5

    try:
        best = next(iter(aligner.align(ref_seq, chain_seq)))
    except (OverflowError, ValueError, MemoryError):
        return {}

    cols = [(a, b) for a, b in zip(best[0], best[1]) if a != "-" and b != "-"]
    if not cols:
        return {}
    identity = sum(1 for a, b in cols if a == b) / len(cols)
    if identity < min_identity:
        return {}

    mapping = {}
    for ref_block, chain_block in zip(best.aligned[0], best.aligned[1]):
        r0, r1 = ref_block
        c0, c1 = chain_block
        for i in range(min(r1 - r0, c1 - c0)):
            mapping[r0 + i + 1] = c0 + i
    return mapping


class GVPEmbeddingExtractor(FeatureExtractor):
    """N_PCA_COMPONENTS PCA components of ESM-IF1 GVP node embeddings."""

    name = "gnn_gvp"
    n_features = N_PCA_COMPONENTS
    feature_names = [f"gvp_pc{i + 1}" for i in range(N_PCA_COMPONENTS)]

    def requires_pdb(self) -> bool:
        return True

    def requires_reference(self) -> bool:
        return True

    def extract(self, df, ref_seqs=None, pdb_resources=None):
        n = len(df)
        features = np.zeros((n, self.n_features), dtype=np.float64)
        ref_seqs = ref_seqs or {}
        loaded = _load_esmif()
        if loaded is None:
            return features
        model, alphabet = loaded

        from esm.inverse_folding import util

        raw = np.full((n, EMB_DIM), np.nan, dtype=np.float64)

        for gene, idxs in df.groupby("subunit").groups.items():
            gene = str(gene).upper()
            if gene not in PDB_MAPPING or gene not in ref_seqs:
                continue
            info = PDB_MAPPING[gene]
            path = STRUCTURE_DIR / info["pdb_id"] / f"{info['pdb_id']}.cif"
            if not path.exists():
                continue
            try:
                structure = util.load_structure(str(path), chain=info["chain"])
                coords, chain_seq = util.extract_coords_from_structure(structure)
                pos_map = _map_ref_to_chain(ref_seqs[gene], chain_seq)
                if not pos_map:
                    continue
                emb = util.get_encoder_output(model, alphabet, coords)
                emb = np.asarray(emb.detach().cpu())
            except Exception as exc:
                warnings.warn(f"GVP extraction failed for {gene}: {exc}")
                continue

            for idx in idxs:
                pos = int(df.iloc[idx]["position"])
                chain_idx = pos_map.get(pos)
                if chain_idx is not None and 0 <= chain_idx < emb.shape[0]:
                    raw[idx, :] = emb[chain_idx, :]

        valid = ~np.isnan(raw).any(axis=1)
        if not valid.any():
            return features
        raw[~valid] = raw[valid].mean(axis=0)

        from sklearn.decomposition import PCA

        features = PCA(n_components=self.n_features, random_state=42).fit_transform(raw)
        return features.astype(np.float64)
