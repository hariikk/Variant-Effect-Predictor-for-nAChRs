"""
Download AlphaFold-predicted structures for nAChR subunits lacking
experimental PDB structures.

Downloads CIF files from AlphaFold DB (v6) to data/raw/structure_files/.

Usage:
    python scripts/download_alphafold_structures.py

Subunits downloaded (6 total):
    CHRNA2  -> Q15822  (AF-Q15822-F1-model_v6.cif)
    CHRNA5  -> P30532  (AF-P30532-F1-model_v6.cif)
    CHRNA6  -> Q15825  (AF-Q15825-F1-model_v6.cif)
    CHRNA9  -> Q9UGM1  (AF-Q9UGM1-F1-model_v6.cif)
    CHRNA10 -> Q13002  (AF-Q13002-F1-model_v6.cif)
    CHRNB3  -> Q05901  (AF-Q05901-F1-model_v6.cif)

Notes:
    - AlphaFold stores pLDDT (confidence) in the B-factor field.
      This is used as a feature -- low pLDDT = disordered/flexible.
    - Models are monomeric (single chain A).
      RSA/C-beta/HSE reflect monomer context, not pentamer assembly.
    - Downloaded CIFs are renamed to AF-{UniProt}.cif for clean naming.
"""

import sys
import time
from pathlib import Path

import requests

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from vep_nachr.config import STRUCTURE_DIR

# Subunits needing AlphaFold structures
ALPHAFOLD_SUBUNITS = {
    "CHRNA2":  "Q15822",
    "CHRNA5":  "P30532",
    "CHRNA6":  "Q15825",
    "CHRNA9":  "Q9UGM1",
    "CHRNA10": "Q13002",
    "CHRNB3":  "Q05901",
}

ALPHAFOLD_BASE = "https://alphafold.ebi.ac.uk/files"
VERSION = 6  # Latest as of 2026-07


def download_cif(uniprot_id: str, dest_path: Path) -> str:
    """
    Download AlphaFold CIF for a UniProt ID.

    Returns
    -------
    str
        "ok" if success, error message if failed.
    """
    url = f"{ALPHAFOLD_BASE}/AF-{uniprot_id}-F1-model_v{VERSION}.cif"

    try:
        response = requests.get(url, stream=True, timeout=120)
    except requests.RequestException as e:
        return f"request failed: {e}"

    if response.status_code != 200:
        return f"HTTP {response.status_code}"

    total_size = int(response.headers.get("content-length", 0))

    try:
        with open(dest_path, "wb") as f:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)

        if dest_path.stat().st_size == 0:
            dest_path.unlink()
            return "empty file"
    except IOError as e:
        return f"write error: {e}"

    size_kb = downloaded / 1024
    return f"ok ({size_kb:.0f} KB)"


def main():
    """Download all missing AlphaFold structures."""
    STRUCTURE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"AlphaFold DB v{VERSION}")
    print(f"Target: {STRUCTURE_DIR}")
    print(f"Downloading {len(ALPHAFOLD_SUBUNITS)} structures...")
    print("-" * 50)

    ok_count = 0
    skip_count = 0
    fail_count = 0

    for subunit, uniprot_id in ALPHAFOLD_SUBUNITS.items():
        dest_name = f"AF-{uniprot_id}.cif"
        dest_path = STRUCTURE_DIR / dest_name

        if dest_path.exists():
            size_kb = dest_path.stat().st_size / 1024
            print(f"  [SKIP] {subunit:12s} ({uniprot_id}): already exists ({size_kb:.0f} KB)")
            skip_count += 1
            continue

        result = download_cif(uniprot_id, dest_path)
        if result.startswith("ok"):
            print(f"  [OK]   {subunit:12s} ({uniprot_id}): {result}")
            ok_count += 1
        else:
            print(f"  [FAIL] {subunit:12s} ({uniprot_id}): {result}")
            fail_count += 1
            # Clean up partial download
            if dest_path.exists():
                dest_path.unlink()

        time.sleep(0.5)

    print("-" * 50)
    print(f"Done: {ok_count} downloaded, {skip_count} skipped, {fail_count} failed")

    if fail_count > 0:
        print("\nSome downloads failed. Check network and retry.")
        sys.exit(1)

    print(f"\nNext: update PDB_MAPPING in vep_nachr/config.py")
    print(f"  AF-* entries -> chain='A' (monomeric AlphaFold models)")


if __name__ == "__main__":
    main()
