"""
Mutant-minus-WT embedding delta features (H5).

Run XIII used ESM-2 embeddings of the WT sequence as *position* features
(`esm_embeddings`, 16 PCA comps) and it was null. H5 tests the *difference*
between WT and mutant embeddings as an *effect* feature — a different quantity:
not "what does this position look like" but "how far, and in which direction, does
this specific mutation move the learned representation".

For a variant (pos, wt->mt), we run the reference sequence with the residue at
pos swapped to `variant_aa`, read the final-layer per-position embedding at pos,
and subtract the WT embedding at pos:

    delta = emb(seq[mt])[pos] - emb(seq[wt])[pos]     # 640-dim

then PCA to a handful of components (unsupervised, same convention as
`esm_embeddings`). The magnitude (L2 norm) and direction of `delta` encode how
surprising and how directionally-loaded the substitution is in the model's
representation — orthogonal to the scalar MLM zero-shot score already in `esm2`.

Shares the lazy-loaded 150M ESM-2 instance from `embeddings._load_esm2` (moved to
CUDA once). Degrades to zero features if ESM-2 is unavailable.
"""

import warnings

import numpy as np
import pandas as pd

from vep_nachr2.features.base import FeatureExtractor
from vep_nachr2.features.embeddings import _load_esm2


N_PCA_COMPONENTS = 16


class EmbeddingDeltaExtractor(FeatureExtractor):
    """Mutant-minus-WT ESM-2 per-position embedding delta, PCA-reduced (H5)."""

    name = "emb_delta"
    n_features = N_PCA_COMPONENTS
    feature_names = [f"delta_pc{i + 1}" for i in range(N_PCA_COMPONENTS)]

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
            warnings.warn(
                "ESM-2 unavailable; returning zero-filled emb_delta features. "
                f"Reason: {getattr(_load_esm2, '__dict__', {}).get('error', 'unknown')}"
            )
            return features

        import torch

        device = next(model.parameters()).device
        batch_converter = alphabet.get_batch_converter()

        raw = np.full((n, 640), np.nan, dtype=np.float64)

        by_gene: dict[str, list[int]] = {}
        for i, (_, row) in enumerate(df.iterrows()):
            gene = str(row.get("subunit", "")).upper()
            by_gene.setdefault(gene, []).append(i)

        for gene, idxs in by_gene.items():
            seq = ref_seqs.get(gene)
            if not seq:
                continue

            # WT per-position embedding (one forward pass per gene).
            _, _, wt_tokens = batch_converter([("wt", seq)])
            wt_tokens = wt_tokens.to(device)
            with torch.no_grad():
                out_wt = model(wt_tokens, repr_layers=[model.num_layers])
            wt_rep = out_wt["representations"][model.num_layers][0].cpu().numpy()  # [L+2, 640]

            # Mutant sequences (one per variant), batched per gene.
            data = []
            valid = []  # (row_index, 1-based pos) in batch order
            for idx in idxs:
                pos = int(df.iloc[idx]["position"])
                mt = str(df.iloc[idx]["variant_aa"]).upper()
                if pos < 1 or pos > len(seq):
                    continue
                if mt not in alphabet.all_toks:
                    continue
                data.append((str(idx), seq[: pos - 1] + mt + seq[pos:]))
                valid.append((idx, pos))
            if not data:
                continue

            _, _, mt_tokens = batch_converter(data)
            mt_tokens = mt_tokens.to(device)
            with torch.no_grad():
                out_mt = model(mt_tokens, repr_layers=[model.num_layers])
            mt_rep = out_mt["representations"][model.num_layers].cpu().numpy()  # [B, L+2, 640]

            for b, (idx, pos) in enumerate(valid):
                # token 0 = BOS, so 1-based pos p -> token index p.
                raw[idx, :] = mt_rep[b, pos, :] - wt_rep[pos, :]

        # Impute missing with the mean of available deltas, then PCA.
        valid_mask = ~np.isnan(raw).any(axis=1)
        if valid_mask.any():
            mean_delta = raw[valid_mask].mean(axis=0)
        else:
            mean_delta = np.zeros(640)
        raw[~valid_mask] = mean_delta

        from sklearn.decomposition import PCA

        pca = PCA(n_components=self.n_features, random_state=42)
        features = pca.fit_transform(raw).astype(np.float64)

        return features
