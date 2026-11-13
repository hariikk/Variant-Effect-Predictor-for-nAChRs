"""
RDKit/Mordred amino-acid descriptor features (cheminformatics feature arm).

Mordred computes ~1,300 2D molecular descriptors from a SMILES string. It is
built for small molecules, so it does not apply to a whole protein — but it
*does* apply to an amino acid side chain if we encode each of the 20 standard
residues as its free amino-acid SMILES.

For a substitution wt -> mt we compute the descriptor vector of both residues,
keep the descriptors that actually vary across the 20 amino acids, PCA them to
N_PCA_COMPONENTS, and use the *difference* (mt - wt) as the feature — the
cheminformatics analogue of the `esm2_effect` delta and the physicochemical
diff block.

This tests whether a much larger, automatically generated descriptor set
carries substitution signal that the hand-picked 24 physicochemical features
miss. Degrades to zero-filled features if rdkit/mordred are unavailable.
"""

import warnings

import numpy as np
import pandas as pd

from vep_nachr2.features.base import FeatureExtractor

N_PCA_COMPONENTS = 16
N_DESCRIPTORS = 1292  # before dropping constant columns (mordred, ignore_3D=True)

# Free amino-acid SMILES for the 20 standard residues.
AA_SMILES = {
    "A": "CC(N)C(=O)O",
    "R": "NC(CCCNC(=N)N)C(=O)O",
    "N": "NC(CC(N)C(=O)O)C(=O)O",
    "D": "OC(=O)CC(N)C(=O)O",
    "C": "NC(CS)C(=O)O",
    "E": "OC(=O)CCC(N)C(=O)O",
    "Q": "NC(=O)CCC(N)C(=O)O",
    "G": "NCC(=O)O",
    "H": "NC(Cc1c[nH]cn1)C(=O)O",
    "I": "CCC(C)C(N)C(=O)O",
    "L": "CC(C)CC(N)C(=O)O",
    "K": "NCCCCC(N)C(=O)O",
    "M": "CSCCC(N)C(=O)O",
    "F": "NC(Cc1ccccc1)C(=O)O",
    "P": "O=C(O)C1CCCN1",
    "S": "NC(CO)C(=O)O",
    "T": "CC(O)C(N)C(=O)O",
    "W": "NC(Cc1c[nH]c2ccccc12)C(=O)O",
    "Y": "NC(Cc1ccc(O)cc1)C(=O)O",
    "V": "CC(C)C(N)C(=O)O",
}

_AA_MATRIX = None
_AA_ERROR = None


def _build_aa_matrix():
    """Return (20 x n_varying) Mordred descriptor matrix, PCA-reduced later."""
    global _AA_MATRIX, _AA_ERROR
    if _AA_MATRIX is not None or _AA_ERROR is not None:
        return _AA_MATRIX
    try:
        from rdkit import Chem
        from mordred import Calculator, descriptors

        calc = Calculator(descriptors, ignore_3D=True)
        mols = [Chem.MolFromSmiles(s) for s in AA_SMILES.values()]
        table = calc.pandas(mols)
        numeric = table.select_dtypes("number").astype(float)
        numeric = numeric.replace([np.inf, -np.inf], np.nan)
        # Keep descriptors that vary across residues; impute rare NaNs with column mean.
        varies = numeric.nunique(dropna=True) > 1
        numeric = numeric.loc[:, varies]
        numeric = numeric.fillna(numeric.mean())
        matrix = numeric.to_numpy(dtype=np.float64)
        index = {aa: i for i, aa in enumerate(AA_SMILES)}
        _AA_MATRIX = (matrix, index)
    except Exception as exc:  # pragma: no cover - environment dependent
        _AA_ERROR = f"{type(exc).__name__}: {exc}"
        warnings.warn(f"Mordred unavailable: {_AA_ERROR}")
    return _AA_MATRIX


class MordredAADescriptorExtractor(FeatureExtractor):
    """N_PCA_COMPONENTS PCA components of the wt->mt Mordred descriptor delta."""

    name = "mordred_aa"
    n_features = N_PCA_COMPONENTS
    feature_names = [f"mordred_pc{i + 1}" for i in range(N_PCA_COMPONENTS)]

    def requires_pdb(self) -> bool:
        return False

    def requires_reference(self) -> bool:
        return True

    def extract(self, df, ref_seqs=None, pdb_resources=None):
        n = len(df)
        features = np.zeros((n, self.n_features), dtype=np.float64)
        built = _build_aa_matrix()
        if built is None:
            return features
        aa_matrix, aa_index = built

        from sklearn.decomposition import PCA

        # PCA fitted on the 20 amino-acid descriptor rows (unsupervised, label-free).
        pca = PCA(n_components=min(self.n_features, aa_matrix.shape[0] - 1), random_state=42)
        aa_pca = pca.fit_transform(aa_matrix)  # 20 x k

        delta = np.zeros((n, aa_pca.shape[1]), dtype=np.float64)
        for i, (_, row) in enumerate(df.iterrows()):
            wt = str(row.get("wildtype_aa", "")).strip().upper()
            mt = str(row.get("variant_aa", "")).strip().upper()
            if wt in aa_index and mt in aa_index:
                delta[i, :] = aa_pca[aa_index[mt]] - aa_pca[aa_index[wt]]

        # Pad/truncate to the declared feature count.
        out = np.zeros((n, self.n_features), dtype=np.float64)
        k = min(self.n_features, delta.shape[1])
        out[:, :k] = delta[:, :k]
        return out
