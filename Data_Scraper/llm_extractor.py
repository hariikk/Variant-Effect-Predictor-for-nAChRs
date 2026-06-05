"""
LLM-based mutation data extraction using Google Gemini API.
Sends full paper text to Gemini and gets structured mutation data back.
"""

import json
import re
import time

import google.generativeai as genai

import config

# The extraction prompt sent to Gemini along with the paper text
EXTRACTION_PROMPT = """\
You are an expert in molecular biology and nicotinic acetylcholine receptors (nAChRs).
I will give you the full text of a scientific paper. Your job is to extract ALL mutations
of nicotinic acetylcholine receptor (nAChR) subunits that were studied using ion flow
measurement techniques (electrophysiology, patch-clamp, voltage-clamp, TEVC, calcium flux,
thallium flux, FLIPR, etc.).

IMPORTANT RULES:
1. ONLY extract mutations of MOUSE (Mus musculus) nAChR variants. Skip human, rat, or
   other species variants unless the paper explicitly states the mouse gene/protein was used.
   Note: Many papers express mouse nAChR subunits in Xenopus oocytes or HEK293 cells for
   electrophysiology - these COUNT as mouse variants (the gene is from mouse, the expression
   system is just the host).
2. ONLY include entries where ion flow was actually measured (electrophysiology, patch-clamp,
   voltage-clamp, TEVC, whole-cell recording, single-channel recording, calcium imaging,
   thallium flux, FLIPR, etc.). Do NOT include binding assays, Western blots, or other
   non-functional assays.
3. For the Effect field, classify as:
   - "LOF" (Loss of Function): if current/response decreased, channel failed to open,
     reduced efficacy, reduced potency with functional impact, decreased Imax, etc.
   - "GOF" (Gain of Function): if current/response increased, enhanced opening,
     increased sensitivity, increased Imax, spontaneous openings, etc.
   - "No net effect": if no significant change in function was observed.
   If the paper doesn't clearly state LOF/GOF but describes changes like "current reduced
   by 80%" → that's LOF. "Current increased 3-fold" → that's GOF. "No significant
   difference" → that's "No net effect".
   If you CANNOT determine the effect from the paper, skip that entry entirely.
4. Modification types allowed: "Substitution", "Frameshift", "Stop", "Deletion".
   - Substitution: single amino acid change (e.g., E97A)
   - Frameshift: insertion/deletion causing a reading frame shift
   - Stop: premature stop codon / nonsense mutation
   - Deletion: removal of one or more amino acids (not frameshift)
5. nAChR subunit names must be one of: CHRNA1, CHRNA2, CHRNA3, CHRNA4, CHRNA5, CHRNA6,
   CHRNA7, CHRNA9, CHRNA10, CHRNB1, CHRNB2, CHRNB3, CHRNB4, CHRND, CHRNE, CHRNG.
   Map common names: alpha7 → CHRNA7, beta2 → CHRNB2, delta → CHRND, epsilon → CHRNE,
   gamma → CHRNG, etc.
6. For amino acid positions, use the position number from the paper. Use standard single-letter
   amino acid codes (e.g., A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y).
   For Stop mutations, use "*" as the New AA.
   For Deletions, put "del" as the New AA.
   For Frameshifts, put "fs" as the New AA.

Return your results as a JSON array. Each entry must have these exact fields:
{
  "subunit": "CHRNA7",
  "modification_type": "Substitution",
  "aa_position": 97,
  "initial_aa": "E",
  "new_aa": "A",
  "effect": "LOF",
  "measuring_technique": "Two-electrode voltage-clamp",
  "pathology": ""
}

- "pathology": the disease or condition associated with this mutation, if mentioned. Leave as
  empty string "" if no pathology is stated.
- If you find NO qualifying mutations in this paper, return an empty array: []

PAPER TEXT:
"""


def setup_gemini():
    """Configure the Gemini API client."""
    if not config.GEMINI_API_KEY:
        raise ValueError(
            "GEMINI_API_KEY not set. Add it to your .env file or set it as an "
            "environment variable."
        )
    genai.configure(api_key=config.GEMINI_API_KEY)


def extract_mutations_from_paper(
    paper_text: str, pmid: str, max_text_length: int = 900000
) -> list[dict]:
    """
    Send paper text to Gemini and extract structured mutation data.

    Args:
        paper_text: Full text content of the paper.
        pmid: PubMed ID (for logging).
        max_text_length: Maximum characters to send (Gemini has token limits).

    Returns:
        List of validated mutation dictionaries.
    """
    # Truncate very long papers to stay within token limits
    if len(paper_text) > max_text_length:
        paper_text = paper_text[:max_text_length] + "\n\n[TEXT TRUNCATED]"

    prompt = EXTRACTION_PROMPT + paper_text

    model = genai.GenerativeModel(config.GEMINI_MODEL)

    for attempt in range(config.MAX_RETRIES):
        try:
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,  # Low temperature for factual extraction
                    max_output_tokens=8192,
                ),
            )

            # Parse the response
            raw_text = response.text
            mutations = _parse_gemini_response(raw_text, pmid)

            # Rate limit
            time.sleep(config.GEMINI_DELAY_SECONDS)

            return mutations

        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "quota" in error_msg.lower():
                wait_time = config.RETRY_BACKOFF_FACTOR ** (attempt + 2)  # Longer wait for quota
                print(f"    Gemini rate limit hit, waiting {wait_time}s...")
                time.sleep(wait_time)
            elif attempt < config.MAX_RETRIES - 1:
                wait_time = config.RETRY_BACKOFF_FACTOR ** (attempt + 1)
                print(f"    Gemini error ({e}), retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"    Gemini extraction failed for PMID {pmid}: {e}")
                return []

    return []


def _parse_gemini_response(raw_text: str, pmid: str) -> list[dict]:
    """
    Parse Gemini's response text to extract the JSON array of mutations.
    Handles cases where the model wraps JSON in markdown code blocks.
    """
    # Try to find JSON array in the response
    # Handle markdown code blocks
    json_match = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", raw_text, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
    else:
        # Try to find raw JSON array
        json_match = re.search(r"\[.*\]", raw_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
        else:
            # Check if the response says no mutations found
            if any(
                phrase in raw_text.lower()
                for phrase in ["no qualifying", "no mutations", "empty array", "[]"]
            ):
                return []
            print(f"    Warning: Could not find JSON array in Gemini response for PMID {pmid}")
            print(f"    Response preview: {raw_text[:200]}...")
            return []

    try:
        mutations = json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"    Warning: JSON parse error for PMID {pmid}: {e}")
        print(f"    JSON preview: {json_str[:200]}...")
        return []

    if not isinstance(mutations, list):
        print(f"    Warning: Expected list, got {type(mutations)} for PMID {pmid}")
        return []

    # Validate each mutation
    validated = []
    for mut in mutations:
        valid = _validate_mutation(mut, pmid)
        if valid:
            validated.append(valid)

    return validated


def _validate_mutation(mut: dict, pmid: str) -> dict | None:
    """
    Validate and normalize a single mutation entry from Gemini.
    Returns the cleaned dict or None if invalid.
    """
    required_fields = [
        "subunit", "modification_type", "aa_position",
        "initial_aa", "new_aa", "effect",
    ]

    # Check required fields exist
    for field in required_fields:
        if field not in mut or mut[field] is None or str(mut[field]).strip() == "":
            return None

    # Normalize subunit name
    subunit = str(mut["subunit"]).upper().strip()
    if subunit not in config.ALL_SUBUNITS:
        # Try common mappings
        subunit = _normalize_subunit(subunit)
        if subunit not in config.ALL_SUBUNITS:
            return None

    # Validate modification type
    mod_type = str(mut["modification_type"]).strip()
    # Normalize capitalization
    mod_type_map = {m.lower(): m for m in config.VALID_MODIFICATION_TYPES}
    if mod_type.lower() not in mod_type_map:
        return None
    mod_type = mod_type_map[mod_type.lower()]

    # Validate effect
    effect = str(mut["effect"]).strip()
    effect_map = {e.lower(): e for e in config.VALID_EFFECTS}
    if effect.lower() not in effect_map:
        # Try partial matches
        effect_lower = effect.lower()
        if "loss" in effect_lower or "lof" in effect_lower:
            effect = "LOF"
        elif "gain" in effect_lower or "gof" in effect_lower:
            effect = "GOF"
        elif "no" in effect_lower and ("effect" in effect_lower or "change" in effect_lower):
            effect = "No net effect"
        else:
            return None
    else:
        effect = effect_map[effect.lower()]

    # Validate AA position
    try:
        aa_pos = int(mut["aa_position"])
        if aa_pos < 1:
            return None
    except (ValueError, TypeError):
        return None

    # Normalize amino acids
    initial_aa = str(mut["initial_aa"]).upper().strip()
    new_aa = str(mut["new_aa"]).upper().strip()

    valid_aas = set("ACDEFGHIKLMNPQRSTVWY")

    # For substitutions, both must be standard AAs
    if mod_type == "Substitution":
        if initial_aa not in valid_aas or new_aa not in valid_aas:
            return None
        if initial_aa == new_aa:
            return None

    # For Stop mutations
    if mod_type == "Stop":
        new_aa = "*"

    # For Deletion
    if mod_type == "Deletion":
        new_aa = "del"

    # For Frameshift
    if mod_type == "Frameshift":
        new_aa = "fs"

    return {
        "subunit": subunit,
        "modification_type": mod_type,
        "aa_position": aa_pos,
        "initial_aa": initial_aa,
        "new_aa": new_aa,
        "effect": effect,
        "measuring_technique": str(mut.get("measuring_technique", "")).strip(),
        "pathology": str(mut.get("pathology", "")).strip(),
    }


def _normalize_subunit(name: str) -> str:
    """Try to map common subunit name variants to standard gene symbols."""
    name = name.upper().strip()

    # Direct mappings for common formats
    mappings = {
        "ALPHA1": "CHRNA1", "ALPHA2": "CHRNA2", "ALPHA3": "CHRNA3",
        "ALPHA4": "CHRNA4", "ALPHA5": "CHRNA5", "ALPHA6": "CHRNA6",
        "ALPHA7": "CHRNA7", "ALPHA9": "CHRNA9", "ALPHA10": "CHRNA10",
        "BETA1": "CHRNB1", "BETA2": "CHRNB2", "BETA3": "CHRNB3",
        "BETA4": "CHRNB4",
        "DELTA": "CHRND", "EPSILON": "CHRNE", "GAMMA": "CHRNG",
        "A1": "CHRNA1", "A2": "CHRNA2", "A3": "CHRNA3", "A4": "CHRNA4",
        "A5": "CHRNA5", "A6": "CHRNA6", "A7": "CHRNA7", "A9": "CHRNA9",
        "A10": "CHRNA10",
        "B1": "CHRNB1", "B2": "CHRNB2", "B3": "CHRNB3", "B4": "CHRNB4",
    }

    # Try direct match
    if name in mappings:
        return mappings[name]

    # Try removing common prefixes
    for prefix in ["NACHR-", "NACHR_", "NACHR ", "ACHR"]:
        if name.startswith(prefix):
            suffix = name[len(prefix):]
            if suffix in mappings:
                return mappings[suffix]

    return name
