"""
Amino acid feature extraction using AAIndex properties.

Adapted from VEP-ENaC for nAChR.  Uses 10 AAIndex scales
(2 more than the engineered encoder — adds bulkiness, flexibility,
beta_sheet_propensity; drops aromaticity, secondary_structure_preference).
"""

import pandas as pd
from aaindex import aaindex1


def get_aa_properties(aa: str, expected_property: str | set | None = "all") -> dict:
    """
    Get properties of a given amino acid from AAIndex.

    Args:
        aa: Single-letter amino acid code
        expected_property: Property or set of properties to retrieve

    Returns:
        Dictionary of property → value

    Properties (10 AAIndex scales):
        hydrophobicity        — Eisenberg (EISD840101)
        net_charge            — Klein (KLEP840101)
        molecular_weight      — Fasman (FASG760101)
        bulkiness             — Zimmerman (ZIMJ680101)
        polarity              — Grantham (GRAR740102)
        flexibility           — Bhaskaran & Ponnuswamy (BHAR880101)
        alpha_helix_propensity  — Chou & Fasman (CHOP780201)
        beta_sheet_propensity   — Chou & Fasman (CHOP780202)
        volume                — Krigbaum & Komoriya (KRIW790103)
        isoelectric_point     — Zimmerman (ZIMJ680104)
    """
    if not isinstance(aa, str) or len(aa) != 1:
        raise TypeError("Invalid amino acid. Must be a single character string.")

    if expected_property == "all" or expected_property is None:
        expected_property = {
            "hydrophobicity", "net_charge", "molecular_weight", "polarity",
            "bulkiness", "flexibility", "alpha_helix_propensity",
            "beta_sheet_propensity", "volume", "isoelectric_point",
        }
    elif isinstance(expected_property, str):
        expected_property = {expected_property}

    result = {}

    if "hydrophobicity" in expected_property:
        result["hydrophobicity"] = aaindex1["EISD840101"].values[aa]
    if "net_charge" in expected_property:
        result["net_charge"] = aaindex1["KLEP840101"].values[aa]
    if "molecular_weight" in expected_property:
        result["molecular_weight"] = aaindex1["FASG760101"].values[aa]
    if "bulkiness" in expected_property:
        result["bulkiness"] = aaindex1["ZIMJ680101"].values[aa]
    if "polarity" in expected_property:
        result["polarity"] = aaindex1["GRAR740102"].values[aa]
    if "flexibility" in expected_property:
        result["flexibility"] = aaindex1["BHAR880101"].values[aa]
    if "alpha_helix_propensity" in expected_property:
        result["alpha_helix_propensity"] = aaindex1["CHOP780201"].values[aa]
    if "beta_sheet_propensity" in expected_property:
        result["beta_sheet_propensity"] = aaindex1["CHOP780202"].values[aa]
    if "volume" in expected_property:
        result["volume"] = aaindex1["KRIW790103"].values[aa]
    if "isoelectric_point" in expected_property:
        result["isoelectric_point"] = aaindex1["ZIMJ680104"].values[aa]

    return result


_DEFAULT_PROPERTIES = [
    "hydrophobicity", "net_charge", "molecular_weight", "polarity",
    "bulkiness", "flexibility", "alpha_helix_propensity",
    "beta_sheet_propensity", "volume", "isoelectric_point",
]


def generate_amino_acid_properties(
    df: pd.DataFrame,
    expected_properties: str | list | None = "all",
) -> pd.DataFrame:
    """
    Generate amino acid property columns for wildtype and variant residues.

    Args:
        df: DataFrame containing 'wildtype_aa' and 'variant_aa' columns
        expected_properties: Properties to compute (default: all 10)

    Returns:
        DataFrame with columns: wildtype_aa_{prop}, variant_aa_{prop}
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"df is {type(df)}, but expected pd.DataFrame")

    required_columns = ["wildtype_aa", "variant_aa"]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"DataFrame must contain '{col}' column")

    if expected_properties == "all" or expected_properties is None:
        properties = _DEFAULT_PROPERTIES
    elif isinstance(expected_properties, str):
        properties = [expected_properties]
    else:
        properties = expected_properties

    result_dict = {}
    for prop in properties:
        result_dict[f"wildtype_aa_{prop}"] = pd.Series([0.0] * len(df), index=df.index)
        result_dict[f"variant_aa_{prop}"] = pd.Series([0.0] * len(df), index=df.index)

    for idx, row in df.iterrows():
        try:
            wildtype_aa = row["wildtype_aa"]
            variant_aa = row["variant_aa"]

            if pd.isna(wildtype_aa) or wildtype_aa == "":
                wildtype_props = {prop: 0 for prop in properties}
            else:
                wildtype_props = get_aa_properties(wildtype_aa)

            if pd.isna(variant_aa) or variant_aa == "":
                mutant_props = {prop: 0 for prop in properties}
            else:
                mutant_props = get_aa_properties(variant_aa)

            for prop in properties:
                result_dict[f"wildtype_aa_{prop}"][idx] = wildtype_props.get(prop, 0)
                result_dict[f"variant_aa_{prop}"][idx] = mutant_props.get(prop, 0)

        except (TypeError, ValueError):
            for prop in properties:
                result_dict[f"wildtype_aa_{prop}"][idx] = 0.0
                result_dict[f"variant_aa_{prop}"][idx] = 0.0

    return pd.DataFrame(result_dict, index=df.index)


def generate_amino_acid_property_changes(
    df: pd.DataFrame,
    expected_properties: str | list | None = "all",
) -> pd.DataFrame:
    """
    Calculate property changes for each AA substitution (variant − wildtype).

    Args:
        df: DataFrame containing 'wildtype_aa' and 'variant_aa' columns
        expected_properties: Properties to compute deltas for

    Returns:
        DataFrame with columns: aa_{prop}_change
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"df is {type(df)}, but expected pd.DataFrame")

    required_columns = ["wildtype_aa", "variant_aa"]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"DataFrame must contain '{col}' column")

    if expected_properties == "all" or expected_properties is None:
        properties = _DEFAULT_PROPERTIES
    elif isinstance(expected_properties, str):
        properties = [expected_properties]
    else:
        properties = expected_properties

    result_dict = {}
    for prop in properties:
        result_dict[f"aa_{prop}_change"] = pd.Series([0.0] * len(df), index=df.index)

    for idx, row in df.iterrows():
        try:
            wildtype_aa = row["wildtype_aa"]
            variant_aa = row["variant_aa"]

            if pd.isna(wildtype_aa) or wildtype_aa == "":
                initial_props = {prop: 0 for prop in properties}
            else:
                initial_props = get_aa_properties(wildtype_aa)

            if pd.isna(variant_aa) or variant_aa == "":
                new_props = {prop: 0 for prop in properties}
            else:
                new_props = get_aa_properties(variant_aa)

            for prop in properties:
                result_dict[f"aa_{prop}_change"][idx] = new_props[prop] - initial_props[prop]

        except (TypeError, ValueError):
            for prop in properties:
                result_dict[f"aa_{prop}_change"][idx] = 0.0

    return pd.DataFrame(result_dict, index=df.index)
