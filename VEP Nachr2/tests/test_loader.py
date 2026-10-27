"""Verify the loader produces exactly the dataset the runs claim: 760 variants,
16 genes, 3 classes, 3 species, with the correct class and species counts."""

from vep_nachr2.config import NACHR_GENES
from vep_nachr2.data.loader import encode_labels


def test_total_rows(loaded_df):
    assert len(loaded_df) == 760


def test_16_genes(loaded_df):
    assert set(loaded_df["subunit"].unique()) == set(NACHR_GENES)


def test_three_classes(loaded_df):
    assert sorted(loaded_df["effect"].unique()) == ["GOF", "LOF", "No net effect"]


def test_class_counts(loaded_df):
    vc = loaded_df["effect"].value_counts().to_dict()
    assert vc["LOF"] == 377
    assert vc["GOF"] == 195
    assert vc["No net effect"] == 188


def test_species_counts(loaded_df):
    sc = loaded_df["species"].value_counts().to_dict()
    assert sc["human"] == 507
    assert sc["rat"] == 173
    assert sc["mouse"] == 80


def test_encode_labels_mapping(loaded_df):
    y = encode_labels(loaded_df["effect"])
    assert set(y) == {0, 1, 2}
    assert (y[loaded_df["effect"].to_numpy() == "LOF"] == 0).all()
    assert (y[loaded_df["effect"].to_numpy() == "No net effect"] == 1).all()
    assert (y[loaded_df["effect"].to_numpy() == "GOF"] == 2).all()


def test_no_ambiguous_and_substitutions_only(loaded_df):
    # loader drops LOF/GOF (drop_ambiguous) and non-substitution rows
    assert not loaded_df["effect"].str.contains("LOF/GOF", na=False).any()
    assert set(loaded_df["modification_type"].unique()) <= {"Substitution"}
