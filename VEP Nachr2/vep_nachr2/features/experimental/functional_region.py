"""
nAChR functional-region partition features (H1).

A *mutually-exclusive* classification of each mutated residue into ONE functional
element of the receptor, one-hot encoded (7 named regions + an implicit "other"
reference). This is the nAChR analogue of the "region" signal that carries ENaC
GOF/LOF (Liddle GOF in the C-terminal PY motif vs LOF in the pore/ECD): direction
correlates with *which element* the residue sits in, not its static geometry
(RSA/DSSP/B-factor) — which Runs XIII–XVI showed carries no direction signal.

Regions (N→C along the subunit; each residue maps to exactly one, priority = the
most specific element first):

  1. agonist_binding_site : principal-face orthosteric pocket (α genes only)
  2. membrane_proximal_ecd: ~30 residues N-terminal to TM1 — the ECD–TMD
                            transduction region (Cys-loop, β10, pre-M1)
  3. extracellular_other  : rest of the N-terminal ECD (incl. signal peptide)
  4. m2_m3_linker         : M2→M3 linker (gating-coupling)
  5. pore_m2              : M2 pore-lining helix (channel gate)
  6. other_tmd            : M1 / M3 / M4 non-pore transmembrane helices
  7. intracellular_loop   : M3→M4 intracellular loop

"other" (C-terminal tail after M4, or a gene with no TM annotation) = all-zero
row, the implicit reference category.

Annotation-only — no PDB or reference sequences needed. Built on the TM
boundaries + binding-site residues in `mechanism.py` (single source of truth).
"""

import numpy as np
import pandas as pd

from vep_nachr2.features.base import FeatureExtractor
from vep_nachr2.features.experimental import mechanism as mech


# Ordered, mutually-exclusive functional regions. "other" is the implicit
# reference category (all-zero row) and is NOT a column.
REGION_NAMES = [
    "agonist_binding_site",
    "membrane_proximal_ecd",
    "extracellular_other",
    "m2_m3_linker",
    "pore_m2",
    "other_tmd",
    "intracellular_loop",
]

_REGION_TO_INDEX = {name: i for i, name in enumerate(REGION_NAMES)}


def get_functional_region(gene: str, pos: int) -> str:
    """Return the functional region of a residue, or "other" if unclassifiable.

    Priority order (most specific first): binding site > pore M2 > M2-M3 linker >
    other TMD > intracellular loop > membrane-proximal ECD > extracellular other.
    """
    gene = str(gene).upper()
    pos = int(pos)

    # 1. Agonist-binding site (principal face, α genes only).
    binding = mech._BINDING_SITE_RESIDUES.get(gene)
    if binding and pos in binding:
        return "agonist_binding_site"

    helices = mech.get_tm_helices(gene)
    if not helices:
        return "other"

    m1, m2, m3, m4 = helices
    m1s, _ = m1
    m2s, m2e = m2
    m3s, m3e = m3
    m4s, m4e = m4

    # 2. M2 pore-lining helix.
    if m2s <= pos <= m2e:
        return "pore_m2"

    # 3. M2→M3 linker.
    if m2e < pos < m3s:
        return "m2_m3_linker"

    # 4. Non-pore TMD (M1 / M3 / M4).
    if (m1s <= pos <= m1[1]) or (m3s <= pos <= m3e) or (m4s <= pos <= m4e):
        return "other_tmd"

    # 5. Intracellular M3→M4 loop.
    if m3e < pos < m4s:
        return "intracellular_loop"

    # 6. Membrane-proximal ECD (the ECD–TMD transduction region, ~30 aa N-terminal
    #    to TM1 — includes the Cys-loop, β10, and pre-M1).
    if (m1s - 30) <= pos < m1s:
        return "membrane_proximal_ecd"

    # 7. Rest of the N-terminal ECD.
    if pos < (m1s - 30):
        return "extracellular_other"

    # C-terminal tail after M4 (extracellular), or anything else.
    return "other"


def get_functional_region_index(gene: str, pos: int) -> int:
    """Return the one-hot column index for a residue's region (-1 = "other")."""
    region = get_functional_region(gene, pos)
    return _REGION_TO_INDEX.get(region, -1)


class FunctionalRegionExtractor(FeatureExtractor):
    """One-hot functional-region membership (7 features, "other" = all-zero)."""

    name = "functional_region"
    n_features = len(REGION_NAMES)
    feature_names = list(REGION_NAMES)

    def requires_pdb(self) -> bool:
        return False

    def requires_reference(self) -> bool:
        return False

    def extract(self, df, ref_seqs=None, pdb_resources=None):
        n = len(df)
        features = np.zeros((n, self.n_features), dtype=np.float64)
        for i, (_, row) in enumerate(df.iterrows()):
            gene = str(row.get("subunit", "")).upper()
            pos = int(row.get("position", 0))
            idx = get_functional_region_index(gene, pos)
            if idx >= 0:
                features[i, idx] = 1.0
        return features
