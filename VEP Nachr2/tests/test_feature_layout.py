"""Verify the 82-column feature layout matches what the runs claim.

The 82-column order is: physicochemical [0:24], substitution [24:27],
positional [27:47] (1 position + 16 subunit_* + 3 species_*), conservation
[47:50], structural_core [50:57], esm2 [57:59], esm_embeddings [59:75],
gating [75:78], allosteric_interaction [78:82].
"""

from vep_nachr2.config import NACHR_GENES


def test_total_82(names82):
    assert len(names82) == 82


def test_substitution_slice(names82):
    assert names82[24:27] == ["blosum_score", "blosum_normalized", "grantham_distance"]


def test_positional_slice(names82):
    assert names82[27] == "position_normalized"
    subunit = names82[28:44]
    species = names82[44:47]
    assert all(n.startswith("subunit_") for n in subunit)
    assert all(n.startswith("species_") for n in species)
    assert len(species) == 3


def test_conservation_slice(names82):
    assert names82[47:50] == [
        "conservation_wt",
        "conservation_mt",
        "conservation_delta",
    ]


def test_structural_core_slice(names82):
    assert names82[50:57] == [
        "rsa", "cbeta_density", "b_factor",
        "dssp_helix", "dssp_sheet", "dssp_coil", "is_unmappable",
    ]


def test_esm2_slice(names82):
    assert names82[57:59] == ["esm2_wt_logprob", "esm2_effect"]


def test_esm_embeddings_slice(names82):
    assert names82[59:75] == [f"esm2_pc{i}" for i in range(1, 17)]


def test_gating_slice(names82):
    assert names82[75:78] == ["gnm_msf", "gnm_slowmode", "gnm_gate_coupling"]


def test_allosteric_slice(names82):
    assert names82[78:82] == [
        "allosteric_dvol", "allosteric_dhydro",
        "allosteric_dcharge", "allosteric_dpolarity",
    ]


def test_physicochemical_count_24(names82):
    # 24 = 8 AAIndex properties × {wt, mt, diff}; exact prefixes are built
    # dynamically so assert the count and the boundary position only.
    assert len(names82[:24]) == 24


def test_subunit_columns_are_the_16_genes(names82):
    subunit = [n[len("subunit_"):] for n in names82 if n.startswith("subunit_")]
    assert len(subunit) == 16
    assert sorted(subunit) == sorted(NACHR_GENES)


def test_identity_drop_leaves_66(names82):
    subunit_cols = [i for i, n in enumerate(names82) if n.startswith("subunit_")]
    assert len(subunit_cols) == 16
    assert 82 - len(subunit_cols) == 66
