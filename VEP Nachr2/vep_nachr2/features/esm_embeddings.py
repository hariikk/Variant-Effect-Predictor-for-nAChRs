"""
Per-residue ESM-2 embedding features (idea #3).

The zero-shot scorer in `embeddings.py` reduces ESM-2 to two scalars
(wt log-prob and mutation effect) and throws away the model's real
representation. This extractor keeps the *local-context* signal: it runs the
wildtype reference sequence through ESM-2 and reads off the final-layer
embedding at the variant position, then projects the ~640-dim vector down to a
handful of PCA components.

Why this is expected to help: the scalar zero-shot score is an evolutionary
"how surprising is this substitution" proxy (a pathogenicity signal, orthogonal
to GOF/LOF direction — see Run VIII). The per-residue embedding instead encodes
"what does this position look like in the folded/sequence context" (local
biochemical neighbourhood), which is exactly the direction-relevant signal the
scalar discards.

The model is the same lazy-loaded 150M ESM-2 instance shared with
`embeddings.py` (`_load_esm2`), so no extra weights are downloaded. If ESM-2 is
unavailable, the extractor degrades to zero-filled features so the pipeline
still runs (shape stays consistent at n_features=N_PCA_COMPONENTS).
"""

import warnings

import numpy as np
import pandas as pd

from vep_nachr2.features.base import FeatureExtractor
from vep_nachr2.features.embeddings import _load_esm2

# Number of PCA components kept from the 640-dim per-residue embedding. Chosen
# low relative to n_samples (~760) to avoid overfitting; the first ~16 PCs
# capture the dominant local-context structure of ESM-2.
N_PCA_COMPONENTS = 16


class PerResidueEmbeddingExtractor(FeatureExtractor):
    """
    Extracts N_PCA_COMPONENTS PCA-projected ESM-2 per-residue embeddings.

    Requires reference sequences (the wildtype context for the model); no PDB.
    Degrades gracefully to zero features if ESM-2 is unavailable.
    """

    name = "esm_embeddings"
    n_features = N_PCA_COMPONENTS
    feature_names = [f"esm2_pc{i + 1}" for i in range(N_PCA_COMPONENTS)]

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
                "ESM-2 unavailable; returning zero-filled esm_embeddings features. "
                f"Reason: {getattr(_load_esm2, '__dict__', {}).get('error', 'unknown')}"
            )
            return features

        import torch

        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = model.to(device)
        except Exception:
            device = "cpu"
        model.eval()
        batch_converter = alphabet.get_batch_converter()

        # Raw per-variant embedding buffer (filled below); rows with no reference
        # stay as NaN until imputed with the global mean before PCA.
        raw = np.full((n, 640), np.nan, dtype=np.float64)

        # Group variant rows by gene — one forward pass per gene (the wildtype
        # context is the same for every variant in that gene).
        by_gene: dict[str, list[int]] = {}
        for i, (_, row) in enumerate(df.iterrows()):
            gene = str(row.get("subunit", "")).upper()
            by_gene.setdefault(gene, []).append(i)

        for gene, idxs in by_gene.items():
            seq = ref_seqs.get(gene)
            if not seq:
                continue

            _, _, batch_tokens = batch_converter([("wt", seq)])
            batch_tokens = batch_tokens.to(device)

            with torch.no_grad():
                out = model(batch_tokens, repr_layers=[model.num_layers],
                            return_contacts=False)

            rep = out["representations"][model.num_layers][0].cpu().numpy()  # [L+2, 640]

            for idx in idxs:
                pos = int(df.iloc[idx]["position"])
                # token 0 = BOS, so 1-based position p -> token index p
                if 1 <= pos <= len(seq):
                    raw[idx, :] = rep[pos, :]

        # Impute rows with no reference/out-of-range position with the global mean
        # of the *available* embeddings, so PCA sees finite input.
        valid_mask = ~np.isnan(raw).any(axis=1)
        if valid_mask.any():
            mean_emb = raw[valid_mask].mean(axis=0)
        else:
            mean_emb = np.zeros(640)
        raw[~valid_mask] = mean_emb

        # PCA projection: fit on all variants (unsupervised — no label leakage),
        # then transform to the top-N components.
        from sklearn.decomposition import PCA

        pca = PCA(n_components=self.n_features, random_state=42)
        features = pca.fit_transform(raw).astype(np.float64)

        return features
