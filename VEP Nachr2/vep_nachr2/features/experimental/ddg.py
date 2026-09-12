"""
Open↔closed ΔΔG features (H3) — the two-state test done with *energy*, not geometry.

Run XVI (alpha7-two-state-null) tested open↔closed GEOMETRY deltas (delta_rsa,
delta_bfactor, delta_interface, ca_rmsd, pore_radius_change) between 7EKT (closed)
and 7KOX (open) and found nothing. H3 re-asks the SAME hypothesis with free energy:
a GOF mutation should be one that the OPEN state accommodates *better* than the CLOSED
state does, and a LOF mutation the reverse. Energy here is ESM-IF inverse-folding
log-likelihood — log P(sequence | backbone structure) — a zero-shot proxy for
sequence-structure compatibility (not a true MD ΔΔG).

For each α7 variant position p (the only gene with an open+closed structure pair,
config.CONFORMATIONAL_PAIRS), we score the WT and MT residue against each state:

    ddg_state = logP(WT | state) − logP(MT | state)      # mutation destabilization

    ddg_open   : destabilization in the OPEN state
    ddg_closed : destabilization in the CLOSED state
    ddg_diff   : ddg_open − ddg_closed                    # <0 = open-stabilizing (GOF)

A positive ddg_diff means the mutation destabilizes open more than closed → the
closed state is favoured → LOF; negative → open favoured → GOF. (Sign is empirical;
the model learns it either way.)

Only CHRNA7 variants get non-zero values (the rest are 0.0, mirroring the α7-only
scope of Run XVI). Alignment of the human precursor reference sequence to the ESM-IF
backbone sequence reuses the same PairwiseAligner params as StructuralExtractor, so
numbering stays in the validated human-precursor frame.
"""

import numpy as np
import pandas as pd

from vep_nachr2.features.base import FeatureExtractor
from vep_nachr2.config import CONFORMATIONAL_PAIRS, STRUCTURE_DIR


_AA20 = "ACDEFGHIKLMNPQRSTVWY"

# Lazy singletons (model load is ~20s; avoid at import time).
_ESMIF = None
def _get_esmif():
    global _ESMIF
    if _ESMIF is None:
        # biotite 1.6 renamed filter_backbone -> filter_peptide_backbone; esm 2.0
        # still imports the old name, so shim it before importing esm.
        import biotite.structure as bs
        if not hasattr(bs, "filter_backbone"):
            bs.filter_backbone = bs.filter_peptide_backbone
        import esm
        model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
        model = model.eval()
        _ESMIF = (model, alphabet)
    return _ESMIF


def _backbone_coords_and_seq(pdb_id, chain):
    """(coords Lx3x3, seq) for one chain via ESM-IF's loader."""
    from esm.inverse_folding.util import load_coords

    pdb_path = STRUCTURE_DIR / pdb_id / f"{pdb_id}.pdb"
    if not pdb_path.exists():
        return None, None
    return load_coords(str(pdb_path), chain)


def _align_ref_to_seq(ref_seq, seq):
    """Map 1-based precursor position -> 0-based backbone index (same aligner as
    StructuralExtractor). Returns {} on low identity / failure."""
    from Bio import Align
    from Bio.Data import CodonTable

    if not ref_seq or not seq:
        return {}

    aligner = Align.PairwiseAligner()
    aligner.mode = "global"
    aligner.substitution_matrix = Align.substitution_matrices.load("BLOSUM62")
    aligner.open_gap_score = -10.0
    aligner.extend_gap_score = -0.5

    try:
        alignments = aligner.align(ref_seq, seq)
        if not alignments:
            return {}
        best = next(iter(alignments))
    except (OverflowError, ValueError, MemoryError):
        return {}

    aligned_ref, aligned_seq = best[0], best[1]
    cols = [(a, b) for a, b in zip(aligned_ref, aligned_seq) if a != "-" and b != "-"]
    if not cols:
        return {}
    identity = sum(1 for a, b in cols if a == b) / len(cols)
    if identity < 0.80:
        return {}

    pos_to_index = {}
    for ref_block, seq_block in zip(best.aligned[0], best.aligned[1]):
        for i, pos in enumerate(range(ref_block[0], ref_block[1])):
            idx = seq_block[0] + i
            if idx < len(seq):
                pos_to_index[pos + 1] = idx
    return pos_to_index


def _state_log_probs(model, alphabet, coords, seq):
    """Per-residue log P(aa | structure): (L, vocab) log-softmaxed logits."""
    import torch
    from esm.inverse_folding.util import CoordBatchConverter

    batch_converter = CoordBatchConverter(alphabet)
    batch = [(coords, None, seq)]
    coords_t, confidence, strs, tokens, padding_mask = batch_converter(batch)
    logits, _ = model.forward(
        coords_t, padding_mask, confidence, tokens[:, :-1]
    )
    # logits is (batch, vocab, seq_len); softmax over vocab (dim 0 of logits[0]).
    # Returns (vocab, L): row = alphabet token, column = residue index.
    return torch.log_softmax(logits[0], dim=0).detach().cpu().numpy()


class OpenClosedDDGExtractor(FeatureExtractor):
    """Open↔closed inverse-folding ΔΔG features (H3). α7-only."""

    name = "ddg"
    n_features = 3
    feature_names = ["ddg_open", "ddg_closed", "ddg_diff"]

    def requires_pdb(self) -> bool:
        return True

    def requires_reference(self) -> bool:
        return True

    def extract(self, df, ref_seqs=None, pdb_resources=None):
        n = len(df)
        features = np.zeros((n, self.n_features), dtype=np.float64)

        # Only CHRNA7 has an open+closed pair; every other gene is 0.0 (out of scope).
        if ref_seqs is None or "CHRNA7" not in ref_seqs:
            return features

        ref_seq = ref_seqs["CHRNA7"]
        pair = CONFORMATIONAL_PAIRS.get("CHRNA7")
        if pair is None:
            return features

        model, alphabet = _get_esmif()
        aa_idx = {aa: alphabet.get_idx(aa) for aa in _AA20}

        # Per-state: backbone sequence, ref->index map, per-residue log-probs.
        states = {}
        for state, pdb_id in pair.items():
            coords, seq = _backbone_coords_and_seq(pdb_id, "A")
            if coords is None:
                continue
            logp = _state_log_probs(model, alphabet, coords, seq)
            pos_to_index = _align_ref_to_seq(ref_seq, seq)
            states[state] = {"logp": logp, "pos_to_index": pos_to_index}
        if len(states) != 2:
            return features

        def _scored(st, pos, aa):
            """logP(aa | state) at 1-based precursor pos, or None if unmapped/unknown."""
            idx = st["pos_to_index"].get(pos)
            if idx is None or idx >= st["logp"].shape[1]:
                return None
            j = aa_idx.get(aa)
            if j is None:
                return None
            return float(st["logp"][j, idx])

        for i, (_, row) in enumerate(df.iterrows()):
            gene = str(row.get("subunit", "")).upper()
            if gene != "CHRNA7":
                continue
            pos = int(row.get("position", 0))
            wt = str(row.get("wildtype_aa", "")).upper()
            mt = str(row.get("variant_aa", "")).upper()
            if not wt or not mt:
                continue

            open_ = states.get("open")
            closed = states.get("closed")
            if open_ is None or closed is None:
                continue

            wp_o = _scored(open_, pos, wt)
            mp_o = _scored(open_, pos, mt)
            wp_c = _scored(closed, pos, wt)
            mp_c = _scored(closed, pos, mt)

            if None in (wp_o, mp_o):
                wp_o = mp_o = None
            if None in (wp_c, mp_c):
                wp_c = mp_c = None

            if wp_o is not None and mp_o is not None:
                features[i, 0] = wp_o - mp_o
            if wp_c is not None and mp_c is not None:
                features[i, 1] = wp_c - mp_c
            if features[i, 0] != 0.0 and features[i, 1] != 0.0:
                features[i, 2] = features[i, 0] - features[i, 1]

        return features
