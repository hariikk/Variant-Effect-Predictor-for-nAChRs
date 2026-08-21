"""
ESM-2 protein language model zero-shot variant-effect features.

Implements the masked-marginal scoring from Meier et al. 2021 ("Language models
enable zero-shot prediction of the effects of mutations on protein function").
For a variant at position p (wildtype w -> mutant m), the model is run with
position p masked and the conditional log-probabilities of w and m are read off:

    esm2_wt_logprob = log P(w | context)
    esm2_effect     = log P(m | context) - log P(w | context)

The first captures how "surprising"/constrained the wildtype residue is at that
position (an ESM-based conservation proxy); the second is the zero-shot mutation
effect (negative = mutation disfavoured = likely damaging).

The model is loaded lazily and cached module-wide (load once, reuse across
ablation conditions). If the `esm` package is missing or weights cannot be
downloaded, the extractor warns and returns zero-filled features so the rest of
the pipeline still runs (shape stays consistent at n_features=2).
"""

import numpy as np
import pandas as pd

from vep_nachr2.features.base import FeatureExtractor


# ESM-2 checkpoint. 150M is the CPU-friendly sweet spot; swap for
# esm2_t33_650M_UR50D if more RAM/time is available.
ESM2_MODEL_NAME = "esm2_t30_150M_UR50D"

# Module-level lazy cache: {loaded, model, alphabet, error}.
_model_state: dict = {"loaded": False, "model": None, "alphabet": None, "error": None}


def _load_esm2():
    """Load (and cache) the ESM-2 model + alphabet. Returns (model, alphabet)."""
    if _model_state["loaded"]:
        return _model_state["model"], _model_state["alphabet"]

    _model_state["loaded"] = True
    try:
        import torch  # noqa: F401
        import esm

        load_fn = getattr(esm.pretrained, ESM2_MODEL_NAME, None)
        if load_fn is None:
            raise ImportError(
                f"fair-esm has no checkpoint '{ESM2_MODEL_NAME}'"
            )
        model, alphabet = load_fn()
        model.eval()
        _model_state["model"] = model
        _model_state["alphabet"] = alphabet
    except Exception as e:  # noqa: BLE001
        _model_state["error"] = e

    return _model_state["model"], _model_state["alphabet"]


class EmbeddingExtractor(FeatureExtractor):
    """
    Extracts 2 ESM-2 zero-shot features per variant.

    Requires reference sequences (wildtype context for the model); no PDB access.
    Degrades gracefully to zero features if ESM-2 is unavailable.
    """

    name = "esm2"
    n_features = 2
    feature_names = ["esm2_wt_logprob", "esm2_effect"]

    def requires_pdb(self) -> bool:
        return False

    def requires_reference(self) -> bool:
        return True  # Wildtype sequences are the model context

    def extract(self, df, ref_seqs=None, pdb_resources=None):
        import warnings

        n = len(df)
        features = np.zeros((n, self.n_features), dtype=np.float64)
        ref_seqs = ref_seqs or {}

        model, alphabet = _load_esm2()
        if model is None:
            warnings.warn(
                "ESM-2 unavailable; returning zero-filled esm2 features. "
                f"Reason: {_model_state.get('error')}"
            )
            return features

        import torch

        batch_converter = alphabet.get_batch_converter()

        # Group variant row indices by gene (same wildtype context per gene).
        by_gene: dict[str, list[int]] = {}
        for i, (_, row) in enumerate(df.iterrows()):
            gene = str(row.get("subunit", "")).upper()
            by_gene.setdefault(gene, []).append(i)

        for gene, idxs in by_gene.items():
            seq = ref_seqs.get(gene)
            if not seq:
                continue

            # Build one masked sequence per variant (batch per gene).
            data = []
            valid_idx_pos = []  # (row_index, 1-based position) in batch order
            for idx in idxs:
                pos = int(df.iloc[idx]["position"])
                if pos < 1 or pos > len(seq):
                    continue
                masked = seq[: pos - 1] + "<mask>" + seq[pos:]
                data.append((str(idx), masked))
                valid_idx_pos.append((idx, pos))

            if not data:
                continue

            _, _, batch_tokens = batch_converter(data)

            with torch.no_grad():
                out = model(batch_tokens)

            # fair-esm returns a dict with "logits"; handle alternative shapes.
            if isinstance(out, dict):
                logits = out["logits"]
            elif isinstance(out, (tuple, list)):
                logits = out[0]
            else:
                logits = out

            log_probs = torch.log_softmax(logits, dim=-1)  # [B, L+2, vocab]

            for b, (idx, pos) in enumerate(valid_idx_pos):
                wt = str(df.iloc[idx]["wildtype_aa"]).upper()
                mt = str(df.iloc[idx]["variant_aa"]).upper()
                wt_idx = alphabet.get_idx(wt)
                mt_idx = alphabet.get_idx(mt)
                if wt_idx < 0 or mt_idx < 0:
                    continue

                # token 0 = <cls>/BOS, so 1-based position p -> token index p
                lp_wt = float(log_probs[b, pos, wt_idx].item())
                lp_mt = float(log_probs[b, pos, mt_idx].item())

                features[idx, 0] = lp_wt
                features[idx, 1] = lp_mt - lp_wt

        return features
