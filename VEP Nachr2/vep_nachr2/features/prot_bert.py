"""
ProtBERT per-residue embedding features (transformer feature arm).

`Rostlab/prot_bert` (ProtTrans, 420M params, 1024-dim hidden) is a BERT model
pretrained on UniRef100. This extractor runs each gene's wildtype reference
sequence through ProtBERT and reads the final-layer embedding at the variant
position, then projects to N_PCA_COMPONENTS (unsupervised PCA).

This is a *different pretraining corpus and architecture* from ESM-2 (already in
the baseline as `esm2` + `esm_embeddings`), so it is a genuine held-out
transformer representation rather than a re-run of the same model.

Degrades to zero-filled features if transformers or the checkpoint are
unavailable (e.g. offline).
"""

import warnings

import numpy as np
import pandas as pd

from vep_nachr2.features.base import FeatureExtractor

N_PCA_COMPONENTS = 16
EMB_DIM = 1024
MODEL_NAME = "Rostlab/prot_bert"

_MODEL = None
_MODEL_ERROR = None


def _load_prot_bert():
    global _MODEL, _MODEL_ERROR
    if _MODEL is not None or _MODEL_ERROR is not None:
        return _MODEL
    try:
        import torch
        from transformers import BertModel, BertTokenizer

        tokenizer = BertTokenizer.from_pretrained(MODEL_NAME, do_lower_case=False)
        model = BertModel.from_pretrained(MODEL_NAME)
        model.eval()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        _MODEL = (model, tokenizer, device)
    except Exception as exc:  # pragma: no cover - environment dependent
        _MODEL_ERROR = f"{type(exc).__name__}: {exc}"
        warnings.warn(f"ProtBERT unavailable: {_MODEL_ERROR}")
    return _MODEL


class ProtBertExtractor(FeatureExtractor):
    """N_PCA_COMPONENTS PCA components of ProtBERT per-residue embeddings."""

    name = "prot_bert"
    n_features = N_PCA_COMPONENTS
    feature_names = [f"protbert_pc{i + 1}" for i in range(N_PCA_COMPONENTS)]

    def requires_pdb(self) -> bool:
        return False

    def requires_reference(self) -> bool:
        return True

    def extract(self, df, ref_seqs=None, pdb_resources=None):
        n = len(df)
        features = np.zeros((n, self.n_features), dtype=np.float64)
        ref_seqs = ref_seqs or {}
        loaded = _load_prot_bert()
        if loaded is None:
            return features
        model, tokenizer, device = loaded

        import torch

        raw = np.full((n, EMB_DIM), np.nan, dtype=np.float64)

        for gene, idxs in df.groupby("subunit").groups.items():
            gene = str(gene).upper()
            seq = ref_seqs.get(gene)
            if not seq:
                continue
            # ProtBERT expects single-letter residues separated by spaces.
            seq = "".join(a if a in "ACDEFGHIKLMNPQRSTVWY" else "X" for a in seq.upper())
            spaced = " ".join(seq)
            try:
                enc = tokenizer(spaced, return_tensors="pt", add_special_tokens=True)
                enc = {k: v.to(device) for k, v in enc.items()}
                with torch.no_grad():
                    out = model(**enc)
                rep = out.last_hidden_state[0].detach().cpu().numpy()  # [L+2, 1024]
            except Exception as exc:
                warnings.warn(f"ProtBERT forward failed for {gene}: {exc}")
                continue

            for idx in idxs:
                pos = int(df.iloc[idx]["position"])
                token_idx = pos  # [CLS] at 0, so 1-based position p -> token p
                if 1 <= pos <= len(seq) and token_idx < rep.shape[0]:
                    raw[idx, :] = rep[token_idx, :]

        valid = ~np.isnan(raw).any(axis=1)
        if not valid.any():
            return features
        raw[~valid] = raw[valid].mean(axis=0)

        from sklearn.decomposition import PCA

        features = PCA(n_components=self.n_features, random_state=42).fit_transform(raw)
        return features.astype(np.float64)
