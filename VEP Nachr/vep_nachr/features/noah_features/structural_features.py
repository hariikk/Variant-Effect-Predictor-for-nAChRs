"""
Structural feature extraction — lightweight wrapper around vep_nachr.features.structural.

Adapts the existing multi-PDB structural pipeline to Noah's feature API
(returns a DataFrame with named columns instead of raw numpy arrays).

Columns returned:
    rsa, cbeta_density, b_factor, ss_dssp, is_unmappable
"""

import numpy as np
import pandas as pd

from vep_nachr.features.structural import (
    extract_structural_features,
    get_structural_feature_names,
)


def get_structural_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract structural features for all mutations in the DataFrame.

    Delegates to the existing vep_nachr.features.structural pipeline which
    handles multi-PDB mapping, Shrake-Rupley SASA, HSE, and AlphaFold
    structures automatically.

    Parameters
    ----------
    df : pd.DataFrame
        Mutation data with columns: subunit, position, wildtype_aa

    Returns
    -------
    pd.DataFrame
        Structural features with columns:
        rsa, bfactor, dssp_helix, dssp_sheet, dssp_coil,
        cbeta_density, hse_up, hse_down
    """
    if df.empty:
        return pd.DataFrame(columns=get_structural_feature_names())

    # Extract features using the existing pipeline
    try:
        struct_array = extract_structural_features(df)
    except Exception:
        # If structural extraction fails entirely (e.g., no PDB files),
        # return imputed features
        n_samples = len(df)
        struct_array = np.full((n_samples, 8), np.nan)

    feature_names = get_structural_feature_names()

    result = pd.DataFrame(struct_array, index=df.index, columns=feature_names)

    # Add is_unmappable flag (True if RSA is NaN — structural code imputes
    # unmappable residues, so check the original extraction)
    result["is_unmappable"] = result["rsa"].isna().astype(float)

    # Fill NaN with defaults
    result["rsa"] = result["rsa"].fillna(1.0)
    result["bfactor"] = result["bfactor"].fillna(0.0)
    result["cbeta_density"] = result["cbeta_density"].fillna(0.0)

    # Convert DSSP booleans to a single ss_dssp code for Noah compatibility
    # Noah's encoding: H=0, B=1, E=2, G=3, I=4, T=5, S=6, -=7
    # We only have helix/sheet/coil flags from our pipeline.
    # Map: helix→H(0), sheet→E(2), coil→-(7)
    ss_codes = np.full(len(result), 7, dtype=int)  # Default: coil
    if "dssp_helix" in result.columns and "dssp_sheet" in result.columns:
        is_helix = result["dssp_helix"].values > 0.5
        is_sheet = result["dssp_sheet"].values > 0.5
        ss_codes[is_helix] = 0  # H
        ss_codes[is_sheet] = 2  # E
    result["ss_dssp"] = ss_codes

    # Select Noah-compatible columns
    noah_cols = ["rsa", "cbeta_density", "bfactor", "ss_dssp", "is_unmappable"]
    return result[noah_cols]
