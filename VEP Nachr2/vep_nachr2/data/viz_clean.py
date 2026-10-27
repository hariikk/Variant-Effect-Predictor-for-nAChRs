"""
Deduplication helper for *visualization* exports (plots + dashboard).

The ML pipeline trains on the full multi-species table (~760 rows), where the
same mutation can legitimately appear once per species (and once per paper) and
is treated as an independent data point. For visualization we want ONE point per
mutation, so this collapses (subunit, position, wt->mt) to a single row and
drops mutations whose rows disagree on effect (ambiguous — a mutation reported
as both LOF and GOF by different sources).

This is deliberately separate from loader.load_mutation_data() so that changing
how we *draw* the data can never silently change what the model *trains* on.
"""

from __future__ import annotations

import pandas as pd

from vep_nachr2.data.loader import load_mutation_data

_SPECIES_RANK = {"human": 0, "rat": 1, "mouse": 2}
_KEY_COLS = ["subunit", "position", "wildtype_aa", "variant_aa"]


def _key(df: pd.DataFrame) -> pd.Series:
    return (
        df["subunit"] + "|" + df["position"].astype(str) + "|"
        + df["wildtype_aa"] + "|" + df["variant_aa"]
    )


def conflicting_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Rows whose (gene, position, wt->mt) appears with more than one effect label."""
    n_eff = df.groupby(_key(df))["effect"].transform("nunique")
    cols = ["subunit", "position", "wildtype_aa", "variant_aa", "effect", "species", "pmid"]
    return df[n_eff > 1][cols].sort_values(["subunit", "position", "variant_aa"])


def clean_variants(df: pd.DataFrame) -> pd.DataFrame:
    """One row per unique mutation, dropping ambiguous (multi-effect) mutations.

    Keeps the human observation when a mutation is seen in several species.
    """
    df = df.copy()
    n_eff = df.groupby(_key(df))["effect"].transform("nunique")
    df = df[n_eff == 1]                                   # drop ambiguous conflicts
    df["_sp"] = df["species"].map(_SPECIES_RANK)
    df = df.sort_values("_sp").drop_duplicates(
        [c for c in _KEY_COLS], keep="first"
    )
    return df.drop(columns=["_sp"]).reset_index(drop=True)


def load_clean_variants(data_file: str = "final_mapped.xlsx") -> pd.DataFrame:
    """load_mutation_data() then collapse to one row per unique mutation."""
    return clean_variants(load_mutation_data(data_file=data_file))
