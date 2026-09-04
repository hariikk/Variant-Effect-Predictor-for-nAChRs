"""
nAChR functional-mechanism annotations and helpers.

Central source of truth for the *direction-relevant* functional sites used by
the `functional_motif` and `charge_context` extractors:

  - Transmembrane (TM1-TM4) boundaries. 10 genes are exact (UniProt/OPM);
    6 low-variant genes (CHRNA2/5/6/9/10, CHRNB3) are homology-estimated
    (marked ESTIMATED) and approximate to ~±3-5 residues.
  - The M2 channel-gate constriction ring (middle of TM2).
  - The extracellular Cys-loop region (just N-terminal to TM1).
  - The agonist-binding-site residues (principal face, α subunits).

These are the sites where GOF vs LOF mutations cluster: gate mutations change
gating, binding-pocket mutations change agonist sensitivity, Cys-loop
mutations change folding/assembly. Flags degrade to 0.0 where a gene has no
annotation (same graceful convention as structural_nachr).
"""

# Transmembrane helix boundaries, 1-based precursor numbering.
# TM1..TM4 as (start, end) inclusive.
_TM_ANNOTATIONS = {
    # ── exact (UniProt/OPM) ──
    "CHRNA1": [(210, 235), (243, 268), (276, 301), (446, 471)],
    "CHRNA7": [(210, 235), (243, 268), (276, 301), (451, 476)],
    "CHRNA4": [(232, 257), (264, 289), (297, 322), (473, 498)],
    "CHRNA3": [(226, 251), (258, 283), (293, 318), (467, 492)],
    "CHRNB1": [(221, 246), (254, 279), (287, 312), (456, 481)],
    "CHRNB2": [(234, 259), (267, 292), (300, 325), (473, 498)],
    "CHRNB4": [(233, 258), (266, 291), (299, 324), (473, 498)],
    "CHRND":  [(225, 250), (258, 283), (291, 316), (454, 479)],
    "CHRNE":  [(228, 253), (261, 286), (294, 319), (457, 482)],
    "CHRNG":  [(228, 253), (261, 286), (294, 319), (457, 482)],
    # ── ESTIMATED (homology to closest annotated subunit) ──
    "CHRNA2":  [(222, 247), (254, 279), (287, 312), (466, 491)],
    "CHRNA5":  [(216, 241), (248, 273), (283, 308), (456, 481)],
    "CHRNA6":  [(225, 250), (257, 282), (290, 315), (459, 484)],
    "CHRNA9":  [(225, 250), (257, 282), (290, 315), (450, 475)],
    "CHRNA10": [(225, 250), (257, 282), (290, 315), (447, 472)],
    "CHRNB3":  [(221, 246), (253, 278), (286, 311), (456, 481)],
}

# Principal-face (α subunit) agonist-binding residues — the orthosteric aromatic
# box + loops A/B/C. The complementary face (β/δ/ε/γ loops D/E/F) is not
# annotated here, so non-α subunits degrade to 0.0 for in_binding_site.
_BINDING_SITE_RESIDUES = {
    "CHRNA1": [86, 89, 93, 149, 151, 152, 188, 189, 190, 192, 193, 194, 195],
    "CHRNA7": [85, 89, 92, 145, 147, 148, 187, 188, 189, 190, 191],
    "CHRNA4": [95, 99, 102, 156, 158, 159, 194, 195, 196, 197, 198],
    "CHRNA3": [91, 95, 98, 152, 154, 155, 190, 191, 192, 193],
}


def get_tm_helices(gene: str):
    """Return TM1..TM4 boundaries for a gene, or None if unannotated."""
    return _TM_ANNOTATIONS.get(gene)


def get_tm_helix(gene: str, pos: int) -> int:
    """Return which TM helix (1-4) a position belongs to; 0 = non-TM."""
    helices = get_tm_helices(gene)
    if not helices:
        return 0
    for idx, (s, e) in enumerate(helices, start=1):
        if s <= pos <= e:
            return idx
    return 0


def get_tm(gene: str, idx: int):
    """Return the (start, end) of the idx-th TM helix (1-based), or None."""
    helices = get_tm_helices(gene)
    if not helices or idx < 1 or idx > len(helices):
        return None
    return helices[idx - 1]


def in_tm(gene: str, pos: int) -> float:
    return 1.0 if get_tm_helix(gene, pos) > 0 else 0.0


def in_tm2(gene: str, pos: int) -> float:
    """1.0 if the position is in M2 (the pore-lining helix)."""
    tm2 = get_tm(gene, 2)
    if tm2 is None:
        return 0.0
    return 1.0 if tm2[0] <= pos <= tm2[1] else 0.0


def in_gate_ring(gene: str, pos: int) -> float:
    """1.0 if in the middle 40% of M2 — the channel-gate constriction ring."""
    tm2 = get_tm(gene, 2)
    if tm2 is None:
        return 0.0
    s, e = tm2
    lo = s + int((e - s) * 0.30)
    hi = s + int((e - s) * 0.70)
    return 1.0 if lo <= pos <= hi else 0.0


def in_cys_loop(gene: str, pos: int) -> float:
    """1.0 if in the extracellular Cys-loop region (~5-30 residues before TM1)."""
    tm1 = get_tm(gene, 1)
    if tm1 is None:
        return 0.0
    s = tm1[0]
    return 1.0 if (s - 30) <= pos <= (s - 5) else 0.0


def in_binding_site(gene: str, pos: int) -> float:
    """1.0 if in the principal-face agonist-binding residues."""
    residues = _BINDING_SITE_RESIDUES.get(gene)
    if residues is None:
        return 0.0
    return 1.0 if pos in residues else 0.0


# Net charge at pH ~7.4 (simplified): acidic = -1, basic = +1, else 0.
# Distinct from the continuous KLEP840101 charge in `physicochemical`.
_AA_CHARGE = {"D": -1.0, "E": -1.0, "K": 1.0, "R": 1.0}


def aa_charge(aa: str) -> float:
    return _AA_CHARGE.get(aa.upper(), 0.0)
