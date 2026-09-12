"""
Coevolution features from ESM-2 attention maps (H4).

Hypothesis: residues that coevolve with the gating hinge are functionally
constrained, so a missense mutation there has a strong *directional* effect
(GOF or LOF) rather than none. The idea is to measure "coevolution" without an
explicit MSA — ESM-2's learned attention patterns are a proxy for statistical
coupling, since the model was trained to predict residues from the rest of the
sequence and therefore learns which positions are jointly informative.

For each gene we run the wildtype reference sequence once, take the LAST layer's
attention (averaged over heads), and symmetrise it into a coupling matrix

    C = (A + Aᵀ) / 2          A[i,j] = attention token i gives token j

and read off two per-position features:

  * `coev_gate` — mean coupling of the variant position to the M2 gate-ring
    residues (the middle 40% of TM2). High = the residue is tightly linked to
    the gating hinge.
  * `coev_hub`  — total coupling (column-sum of C over real residues), a
    coevolutionary "centrality": how strongly this position is coupled to the
    rest of the protein.

These are POSITION features (computed on the WT sequence, no mutant sequence
needed), so extraction is one forward pass per gene (16 total) — cheap. We use
`need_head_weights=True` rather than the (absent) contact-regression head; the
base model's `contact_head` is untrained random noise, whereas the attention
maps are genuinely informative.

Shares the lazy 150M ESM-2 instance from `embeddings._load_esm2`.
"""

import warnings

import numpy as np
import pandas as pd

from vep_nachr2.features.base import FeatureExtractor
from vep_nachr2.features.embeddings import _load_esm2
from vep_nachr2.features.experimental.mechanism import get_tm


class CoevolutionExtractor(FeatureExtractor):
    """ESM-2 attention-derived coevolution coupling (gate + hub), per position."""

    name = "coev"
    n_features = 2
    feature_names = ["coev_gate", "coev_hub"]

    def requires_pdb(self) -> bool:
        return False

    def requires_reference(self) -> bool:
        return True

    def extract(self, df, ref_seqs=None, pdb_resources=None):
        n = len(df)
        features = np.zeros((n, self.n_features), dtype=np.float64)
        ref_seqs = ref_seqs or {}

        model, alphabet = _load_esm2()
        if model is None:
            warnings.warn("ESM-2 unavailable; returning zero-filled coev features.")
            return features

        import torch

        device = next(model.parameters()).device
        batch_converter = alphabet.get_batch_converter()

        by_gene: dict[str, list[int]] = {}
        for i, (_, row) in enumerate(df.iterrows()):
            gene = str(row.get("subunit", "")).upper()
            by_gene.setdefault(gene, []).append(i)

        for gene, idxs in by_gene.items():
            seq = ref_seqs.get(gene)
            if not seq:
                continue

            # One forward pass per gene (WT sequence only).
            _, _, tokens = batch_converter([(gene, seq)])
            tokens = tokens.to(device)
            with torch.no_grad():
                out = model(tokens, repr_layers=[], need_head_weights=True)
            # attentions: (num_layers, B, num_heads, L+2, L+2)
            attn = out["attentions"][-1].mean(dim=1)[0].cpu().numpy()  # (L+2, L+2)
            A = np.asarray(attn, dtype=np.float64)
            C = (A + A.T) / 2.0
            L = len(seq)

            # Gate-ring residue indices (1-based, middle 40% of TM2).
            tm2 = get_tm(gene, 2)
            gate_positions = []
            if tm2 is not None:
                s, e = tm2
                lo = s + int((e - s) * 0.30)
                hi = s + int((e - s) * 0.70)
                gate_positions = list(range(lo, hi + 1))

            for idx in idxs:
                pos = int(df.iloc[idx]["position"])
                if pos < 1 or pos > L:
                    continue
                # token 0 = BOS, so 1-based residue p -> coupling index p.
                if gate_positions:
                    features[idx, 0] = float(np.mean([C[pos, g] for g in gate_positions]))
                # hub: total coupling to real residues (exclude BOS/EOS at 0 and L+1).
                features[idx, 1] = float(C[1 : L + 1, pos].sum())

        return features
