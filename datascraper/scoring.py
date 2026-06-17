"""
Free, transparent relevance scoring for candidate papers.

No LLM / API / quota: each paper's title + abstract is scanned for keyword and
regex signals that indicate it reports a MOUSE nAChR MUTATION studied by
ELECTROPHYSIOLOGY with a functional EFFECT (LOF / GOF / no net effect). Every
paper gets a numeric score (used to rank) plus a breakdown of which signals
fired (shown in the worklist so a human can see *why* it ranked where it did).

Nothing is dropped here — main.py / worklist_writer.py only use the score to
decide which sheet a paper lands on, and the lowest sheet keeps everything.
"""

import re

import config


# ── Signal vocabularies (lowercased substring match on title + abstract) ─────

EFFECT_TERMS = [
    "loss of function", "loss-of-function",
    "gain of function", "gain-of-function",
    "no net effect", "no significant difference", "no significant change",
    "no effect on", "without affecting",
    "reduced current", "decreased current", "increased current",
    "reduced response", "decreased response", "increased response",
    "reduced amplitude", "increased amplitude", "abolished",
    "loss of channel function", "non-functional", "nonfunctional",
    "constitutive", "spontaneous opening", "spontaneously open",
    "potentiat",          # potentiate / potentiation
    "increased sensitivity", "reduced sensitivity", "decreased sensitivity",
    "hypersensitiv", "enhanced response", "diminished",
    "shift in ec50", "ec50 shift", "altered ec50",
    "fold increase", "fold decrease", "fold reduction",
]

EPHYS_TERMS = [
    "patch-clamp", "patch clamp",
    "voltage-clamp", "voltage clamp",
    "two-electrode voltage clamp", "two electrode voltage clamp", "tevc",
    "whole-cell", "whole cell",
    "single-channel", "single channel",
    "xenopus oocyte", "oocyte",
    "macroscopic current", "agonist-evoked", "agonist evoked",
    "ach-evoked", "nicotine-evoked", "acetylcholine-evoked",
    "current amplitude", "peak current", "current-voltage",
    "dose-response", "dose response", "concentration-response",
    "electrophysiolog",
    "rb efflux", "rubidium efflux", "86rb",
    "calcium flux", "ca2+ flux", "calcium imaging", "ca2+ imaging",
    "thallium flux", "flipr",
    "open probability", "channel gating", "channel conductance",
    "desensitiz",         # desensitisation kinetics — a functional readout
]

MUTAGENESIS_TERMS = [
    "mutation", "mutations", "mutant", "mutants",
    "missense", "substitution", "point mutation",
    "site-directed mutagenesis", "site directed mutagenesis",
    "alanine scanning", "alanine substitution",
    "amino acid substitution", "engineered mutant",
    "knock-in", "knockin", "knock in",
    "variant", "naturally occurring mutation",
]

MOUSE_TERMS = [
    "mouse", "mice", "murine", "mus musculus",
    "c57bl", "balb/c", "knock-in mouse", "knockout mouse",
    "transgenic mouse",
]

REVIEW_TERMS = [
    "this review", "we review", "a review of", "review article",
    "systematic review", "meta-analysis", "meta analysis",
    "in this overview",
]

BINDING_TERMS = [
    "radioligand binding", "binding assay", "binding affinity",
    "[3h]", "competition binding", "saturation binding",
]


# ── Mutation-notation regexes (run on ORIGINAL-case text) ────────────────────

_AA = "ACDEFGHIKLMNPQRSTVWY"
_THREE = ("Ala|Arg|Asn|Asp|Cys|Gln|Glu|Gly|His|Ile|Leu|"
          "Lys|Met|Phe|Pro|Ser|Thr|Trp|Tyr|Val")

# e.g. S248F, E97A, L250T  (wild-type AA, position, mutant AA). Word boundaries
# keep cell-line tokens out: "HEK293T" has no boundary before "K293T".
_SUBSTITUTION_RE = re.compile(r"\b[" + _AA + r"]\d{1,4}[" + _AA + r"]\b")

# nAChR pore "prime" notation, e.g. L9'T, Leu9', T6', V13'. Requires an AA-letter
# / 3-letter-code prefix so it never matches 5'/3' nucleic-acid directions.
# Accepts straight, curly and prime unicode apostrophes.
_PRIME_RE = re.compile(
    r"\b(?:[" + _AA + r"]|" + _THREE + r")\d{1,2}['’′]"
)

# Common false positives for the substitution pattern (water, peroxide, ...).
_NOTATION_STOPLIST = {"H2O", "H2O2", "D2O"}


def _find_mutation_hits(text: str) -> list[str]:
    """Distinct mutation-notation strings found in the original-case text."""
    hits, seen = [], set()
    for rx in (_SUBSTITUTION_RE, _PRIME_RE):
        for m in rx.finditer(text):
            s = m.group(0)
            if s in _NOTATION_STOPLIST or s in seen:
                continue
            seen.add(s)
            hits.append(s)
    return hits


def _matched_terms(haystack_lower: str, terms: list[str]) -> list[str]:
    """Distinct terms (from `terms`) that appear as substrings in the text."""
    return [t for t in terms if t in haystack_lower]


def score_paper(title: str, abstract: str, uniprot_curated: bool = False) -> dict:
    """
    Score one paper. Returns the numeric `score` plus the signal breakdown used
    to build the worklist columns. `uniprot_curated` flags papers that a UniProt
    mutagenesis/variant annotation cites (a strong relevance signal).
    """
    original = f"{title} . {abstract}"
    lower = original.lower()

    mutation_hits = _find_mutation_hits(original)
    effect = _matched_terms(lower, EFFECT_TERMS)
    ephys = _matched_terms(lower, EPHYS_TERMS)
    mutagenesis = _matched_terms(lower, MUTAGENESIS_TERMS)
    mouse = _matched_terms(lower, MOUSE_TERMS)

    w = config.SCORE_WEIGHTS
    score = w["mutation_notation"] * min(len(mutation_hits), config.MUTATION_NOTATION_CAP)
    score += w["effect"] if effect else 0
    score += w["ephys"] if ephys else 0
    score += w["mutagenesis"] if mutagenesis else 0
    score += w["mouse"] if mouse else 0
    score += w["uniprot_curated"] if uniprot_curated else 0

    # Down-weights (never exclude — just nudge ranking).
    looks_review = any(t in lower for t in REVIEW_TERMS)
    binding_only = (not ephys) and any(t in lower for t in BINDING_TERMS)
    if looks_review:
        score -= config.DOWNWEIGHT_REVIEW
    if binding_only:
        score -= config.DOWNWEIGHT_BINDING_ONLY

    return {
        "score": score,
        "uniprot_curated": uniprot_curated,
        "mutation_hits": mutation_hits,
        "effect": effect,
        "ephys": ephys,
        "mutagenesis": mutagenesis,
        "mouse": mouse,
        "looks_review": looks_review,
        "binding_only": binding_only,
    }
