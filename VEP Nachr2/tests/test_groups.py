"""Verify the subunit GROUPS and FEATURE_GROUPS slices in the ablation script.

These tests read the *actual* constants from scripts/run_xxxiii_group_ablation.py
(via the ``ablation_script`` fixture), so a wrong slice or a wrong gene set in the
script fails here rather than silently producing a wrong ablation.
"""

from vep_nachr2.config import NACHR_GENES, HOMOLOGY_CLASSES


# ── GROUPS partition ────────────────────────────────────────────────────────

def test_groups_keys_and_sizes(ablation_script):
    g = ablation_script.GROUPS
    assert set(g) == {"alpha", "beta", "delta_gamma_epsilon"}
    assert len(g["alpha"]) == 9
    assert len(g["beta"]) == 4
    assert len(g["delta_gamma_epsilon"]) == 3


def test_groups_disjoint_and_cover_all_16(ablation_script):
    g = ablation_script.GROUPS
    a, b, d = g["alpha"], g["beta"], g["delta_gamma_epsilon"]
    assert not (a & b)
    assert not (a & d)
    assert not (b & d)
    assert a | b | d == set(NACHR_GENES)


def test_groups_match_config_homology(ablation_script):
    g = ablation_script.GROUPS
    assert g["alpha"] == set(HOMOLOGY_CLASSES["alpha"])
    assert g["beta"] == set(HOMOLOGY_CLASSES["beta"])
    assert g["delta_gamma_epsilon"] == set(HOMOLOGY_CLASSES["special"])


# ── FEATURE_GROUPS slices ───────────────────────────────────────────────────

_EXPECTED = {
    "physicochemical": (0, 24),
    "substitution": (24, 27),
    "positional": (27, 47),
    "conservation": (47, 50),
    "structural_core": (50, 57),
    "esm2": (57, 59),
    "esm_embeddings": (59, 75),
    "gating": (75, 78),
    "allosteric_interaction": (78, 82),
}


def test_feature_groups_slices_cover_82_contiguously(ablation_script):
    fg = ablation_script.FEATURE_GROUPS
    assert set(fg) == set(_EXPECTED)
    slices = sorted(fg.values())
    assert slices[0][0] == 0
    assert slices[-1][1] == 82
    for (s1, e1), (s2, e2) in zip(slices, slices[1:]):
        assert e1 == s2, f"gap/overlap between {slices}"


def test_feature_groups_match_expected_slices(ablation_script):
    assert ablation_script.FEATURE_GROUPS == _EXPECTED


def test_feature_group_slices_match_names(ablation_script, names82):
    fg = ablation_script.FEATURE_GROUPS
    # spot-check the exactly-known slices land on the right names
    assert names82[fg["substitution"][0]:fg["substitution"][1]] == [
        "blosum_score", "blosum_normalized", "grantham_distance",
    ]
    assert names82[fg["conservation"][0]:fg["conservation"][1]] == [
        "conservation_wt", "conservation_mt", "conservation_delta",
    ]
    assert names82[fg["esm2"][0]:fg["esm2"][1]] == ["esm2_wt_logprob", "esm2_effect"]
    assert names82[fg["gating"][0]:fg["gating"][1]] == [
        "gnm_msf", "gnm_slowmode", "gnm_gate_coupling",
    ]
