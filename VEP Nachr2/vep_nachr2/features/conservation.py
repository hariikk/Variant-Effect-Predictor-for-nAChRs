"""
Position-specific evolutionary conservation feature extraction.

Conservation answers "how constrained is THIS residue across homologs" — a
position-specific signal that BLOSUM/Grantham (position-independent "how big is
the change") cannot capture. A variant at a residue conserved across species is
far more likely to alter function than one at a variable residue.

Computed from the precomputed ortholog alignments already on disk
(data/raw/reference_sequences/mapping/{mouse,rat}_to_human.csv), which map every
ortholog residue to its human-homologous position with the aligned amino acid.
No new downloads or aligners required.

Features (3):
  - conservation_wt   : fraction of homologs (human + mouse + rat) carrying the
                        wildtype AA at this position. 1.0 = fully conserved.
  - conservation_mt   : fraction of homologs carrying the *mutant* AA at this
                        position. A mutation to a naturally-observed state
                        (e.g. the mouse AA) is more likely tolerated.
  - conservation_delta: conservation_wt - conservation_mt (net loss of constraint).
"""

import csv
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from vep_nachr2.features.base import FeatureExtractor
from vep_nachr2.config import REFERENCE_SEQ_DIR, AMINO_ACIDS


# Species whose ortholog alignments are available on disk (order irrelevant;
# only the aggregate per-position AA list matters).
_ORTHOLOG_SPECIES = ("mouse", "rat")


def _load_ortholog_columns() -> dict[str, dict[int, list[str]]]:
    """
    Build {gene: {human_pos: [ortholog AA, ...]}} from the mapping CSVs.

    Only aligned residues with a single-letter AA are kept; ortholog insertions
    (empty human_pos) and gap columns are skipped, so a human position absent
    from the dict has no ortholog evidence (handled as "no homolog" upstream).
    """
    lookup: dict[str, dict[int, list[str]]] = {}

    for species in _ORTHOLOG_SPECIES:
        path = REFERENCE_SEQ_DIR / "mapping" / f"{species}_to_human.csv"
        if not path.exists():
            continue
        with open(path, newline="") as fh:
            for row in csv.DictReader(fh):
                if not row.get("human_pos") or not row.get("source_aa"):
                    continue
                aa = str(row["source_aa"]).strip().upper()
                if len(aa) != 1 or aa not in AMINO_ACIDS:
                    continue
                gene = str(row["gene"]).strip().upper()
                pos = int(row["human_pos"])
                lookup.setdefault(gene, {}).setdefault(pos, []).append(aa)

    return lookup


class ConservationExtractor(FeatureExtractor):
    """
    Extracts 3 ortholog-based conservation features per variant.

    Requires reference sequences (to confirm the human wildtype AA at each
    position); no PDB access needed.
    """

    name = "conservation"
    n_features = 3
    feature_names = ["conservation_wt", "conservation_mt", "conservation_delta"]

    def __init__(self, verbose: bool = False):
        super().__init__(verbose=verbose)
        self._lookup: Optional[dict[str, dict[int, list[str]]]] = None

    def requires_pdb(self) -> bool:
        return False

    def requires_reference(self) -> bool:
        return True  # Human reference sequences confirm the wildtype AA

    def _get_lookup(self) -> dict[str, dict[int, list[str]]]:
        if self._lookup is None:
            self._lookup = _load_ortholog_columns()
        return self._lookup

    def extract(self, df, ref_seqs=None, pdb_resources=None):
        lookup = self._get_lookup()
        n = len(df)
        features = np.zeros((n, self.n_features), dtype=np.float64)

        ref_seqs = ref_seqs or {}

        for i, (_, row) in enumerate(df.iterrows()):
            gene = str(row.get("subunit", "")).upper()
            position = int(row.get("position", 0))
            wt = str(row.get("wildtype_aa", "X")).upper()
            mt = str(row.get("variant_aa", "X")).upper()

            orthologs = lookup.get(gene, {}).get(position, [])

            # Human contribution: does the reference AA at this position match
            # the reported wildtype? (True by construction on mapped data.)
            seq = ref_seqs.get(gene, "")
            human_matches = True
            if seq and 1 <= position <= len(seq):
                human_matches = seq[position - 1].upper() == wt

            total = len(orthologs) + 1  # human + available orthologs
            wt_count = sum(1 for a in orthologs if a == wt)
            mt_count = sum(1 for a in orthologs if a == mt)

            conservation_wt = (wt_count + (1 if human_matches else 0)) / total
            conservation_mt = mt_count / total

            features[i, 0] = conservation_wt
            features[i, 1] = conservation_mt
            features[i, 2] = conservation_wt - conservation_mt

        return features
