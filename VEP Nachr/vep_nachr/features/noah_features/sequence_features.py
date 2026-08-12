"""
Sequence-based feature extraction (BLOSUM62 substitution scores).

Adapted from VEP-ENaC for nAChR.
"""

import numpy as np
import pandas as pd
from Bio.Align import substitution_matrices


def generate_matrix_score(
    df: pd.DataFrame,
    matrix_name: str = "BLOSUM62",
    score_col_name: str = "blosum62_score",
) -> pd.Series:
    """
    Generate substitution matrix scores for amino acid changes.

    Args:
        df: DataFrame containing 'wildtype_aa' and 'variant_aa' columns
        matrix_name: Name of the substitution matrix (default: BLOSUM62)
        score_col_name: Name for the resulting score column

    Returns:
        Series containing substitution scores
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"df is {type(df)}, but expected pd.DataFrame")
    if "wildtype_aa" not in df.columns or "variant_aa" not in df.columns:
        raise ValueError("DataFrame must contain 'wildtype_aa' and 'variant_aa' columns")

    try:
        matrix = substitution_matrices.load(matrix_name)
    except Exception as e:
        raise ValueError(f"Failed to load substitution matrix '{matrix_name}': {e}")

    def validate_and_score(row) -> float:
        wt = row["wildtype_aa"]
        mut = row["variant_aa"]
        if not (isinstance(wt, str) and isinstance(mut, str) and len(wt) == 1 and len(mut) == 1):
            return np.nan
        try:
            return float(matrix[(wt, mut)])
        except KeyError:
            return np.nan

    result = df.apply(validate_and_score, axis=1)
    result.name = score_col_name
    return result
