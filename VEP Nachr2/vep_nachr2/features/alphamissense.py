"""
AlphaMissense pathogenicity feature extraction.

AlphaMissense (Cheng et al. 2023, Science) is a deep-learning model (AlphaFold
+ protein language model) that predicts the *pathogenicity* of every possible
single-amino-acid substitution in the human proteome, producing a continuous
score `am_pathogenicity` in [0, 1] and a three-way class (likely_benign < 0.34,
likely_pathogenic > 0.564, else ambiguous).

This extractor reads the pre-filtered per-variant AlphaMissense table for the 16
nAChR genes (data/raw/alphamissense/nachr_16_genes.tsv) and attaches the
am_pathogenicity score for each (gene, position, variant_aa) as a single feature.

Important nuances
-----------------
1. AlphaMissense measures *pathogenicity* (damaging vs benign), which — as
   established in Run VIII — is a different axis from *direction of effect*
   (GOF vs LOF). This feature is therefore expected to help less than a
   nAChR-mechanism feature would; the Run X ablation quantifies exactly how much.

2. AlphaMissense is keyed by the UniProt canonical sequence, whereas this
   project's positions use the RefSeq isoform numbering. For 15/16 genes the two
   sequences are identical, but CHRNA1's UniProt entry (P02708, 482 aa) carries a
   25-aa P3A alternative-splice exon that the RefSeq isoform (NP_000070.1, 457 aa)
   lacks. We therefore build a per-gene RefSeq->UniProt position map via a
   sequence alignment (difflib) and translate positions before lookup. A clean
   single insertion (the case here) becomes a fixed +25 offset after the exon.

Features (1):
  - alphamissense_pathogenicity : am_pathogenicity for the specific substitution.
                                  Variants whose UniProt wildtype does not match the
                                  project's recorded wildtype (a data-entry error)
                                  or with no score are imputed with the in-batch
                                  mean of the non-missing scores ("no signal").
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from vep_nachr2.features.base import FeatureExtractor
from vep_nachr2.config import UNIPROT_ACCESSIONS, ALPHAMISSENSE_TSV

# Fallback fill if no score is available in the whole batch (midpoint of the
# AlphaMissense "ambiguous" band [0.34, 0.564]).
_AM_AMBIGUOUS_CENTER = 0.45


def _load_scores() -> tuple[dict, dict]:
    """
    Load {(uniprot, pos, mt): score} and {(uniprot, pos): wt} from the compact TSV.

    The AlphaMissense `protein_variant` column ("WtPosMt", e.g. "M1A") encodes
    the wildtype AA, 1-based position, and mutant AA in the UniProt sequence, so
    the lookup is self-contained (no reference sequence required).
    """
    scores: dict[tuple[str, int, str], float] = {}
    wts: dict[tuple[str, int], str] = {}
    path = Path(ALPHAMISSENSE_TSV)
    if not path.exists():
        return scores, wts

    with open(path, newline="") as fh:
        fh.readline()  # header: uniprot_id, protein_variant, am_pathogenicity, am_class
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            uniprot, variant, score_str = parts[0], parts[1], parts[2]
            if len(variant) < 3:
                continue
            wt = variant[0]
            mt = variant[-1]
            pos_str = variant[1:-1]
            if not pos_str.isdigit():
                continue
            pos = int(pos_str)
            try:
                score = float(score_str)
            except ValueError:
                continue
            scores[(uniprot, pos, mt)] = score
            wts[(uniprot, pos)] = wt

    return scores, wts


class AlphamissenseExtractor(FeatureExtractor):
    """
    Extracts 1 AlphaMissense pathogenicity feature per variant.

    Requires the reference sequences so it can build the RefSeq->UniProt position
    map (see module docstring). If the compact TSV has not been generated, the
    feature degrades to a constant fill so the rest of the pipeline still runs.
    """

    name = "alphamissense"
    n_features = 1
    feature_names = ["alphamissense_pathogenicity"]

    def __init__(self, verbose: bool = False):
        super().__init__(verbose=verbose)
        self._scores: Optional[dict] = None
        self._wts: Optional[dict] = None
        self._maps: Optional[dict[str, dict[int, int]]] = None

    def requires_pdb(self) -> bool:
        return False

    def requires_reference(self) -> bool:
        return True  # needed to map RefSeq numbering -> UniProt numbering

    def _get_scores(self):
        if self._scores is None:
            self._scores, self._wts = _load_scores()
        return self._scores, self._wts

    def _get_maps(self, ref_seqs) -> dict[str, dict[int, int]]:
        """
        Build {gene: {ref_index_0based: uni_index_0based}} position maps.

        Aligns each reference sequence against the UniProt sequence reconstructed
        from the AlphaMissense wildtype column, then records the ref->uni mapping
        for every aligned ('equal') column. For 15/16 genes this is the identity;
        for CHRNA1 it is identity up to the P3A exon then a fixed +25 offset.
        """
        if self._maps is None:
            import difflib

            _, wts = self._get_scores()
            self._maps = {}
            for gene, uniprot in UNIPROT_ACCESSIONS.items():
                rs = str(ref_seqs.get(gene, "")) if ref_seqs else ""
                max_pos = max(
                    (p for (u, p) in wts if u == uniprot), default=0
                )
                us = "".join(wts.get((uniprot, p), "-") for p in range(1, max_pos + 1))
                if not rs or not us:
                    self._maps[gene] = {}
                    continue
                sm = difflib.SequenceMatcher(a=rs, b=us, autojunk=False)
                ref_to_uni: dict[int, int] = {}
                for tag, i1, i2, j1, j2 in sm.get_opcodes():
                    if tag == "equal":
                        for k in range(i2 - i1):
                            ref_to_uni[i1 + k] = j1 + k
                self._maps[gene] = ref_to_uni
        return self._maps

    def extract(self, df, ref_seqs=None, pdb_resources=None):
        scores, wts = self._get_scores()
        maps = self._get_maps(ref_seqs)
        n = len(df)
        features = np.full((n, self.n_features), np.nan, dtype=np.float64)

        n_matched = 0
        n_missing = 0
        n_wt_mismatch = 0

        for i, (_, row) in enumerate(df.iterrows()):
            gene = str(row.get("subunit", "")).upper()
            position = int(row.get("position", 0))
            wt = str(row.get("wildtype_aa", "X")).upper()
            mt = str(row.get("variant_aa", "X")).upper()
            uniprot = UNIPROT_ACCESSIONS.get(gene)

            if uniprot is None:
                n_missing += 1
                continue

            # Translate RefSeq 1-based position -> UniProt 1-based position.
            ref_idx = position - 1
            uni_idx = maps.get(gene, {}).get(ref_idx)
            if uni_idx is None:
                # Position not covered by the alignment (should not happen).
                n_missing += 1
                continue
            uni_pos = uni_idx + 1

            # Cross-check the AlphaMissense wildtype against ours; a mismatch
            # after correct mapping flags a data-entry error in the source table,
            # so the score would be the wrong substitution — impute instead.
            expected_wt = wts.get((uniprot, uni_pos))
            if expected_wt is not None and expected_wt != wt:
                n_wt_mismatch += 1
                continue

            score = scores.get((uniprot, uni_pos, mt))
            if score is None:
                n_missing += 1
                continue

            features[i, 0] = score
            n_matched += 1

        # Impute missing values with the in-batch mean of matched scores (a
        # "no signal" value centred on this dataset's distribution).
        matched_vals = features[~np.isnan(features[:, 0]), 0]
        fill = float(matched_vals.mean()) if matched_vals.size else _AM_AMBIGUOUS_CENTER
        features[np.isnan(features[:, 0]), 0] = fill

        if self.verbose:
            print(
                f"    matched={n_matched}/{n} missing={n_missing} "
                f"wt_mismatch={n_wt_mismatch}"
            )

        return features
