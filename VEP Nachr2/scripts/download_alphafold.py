#!/usr/bin/env python3
"""
Download AlphaFold-predicted structures for nAChR subunits without experimental PDBs.

AlphaFold DB URLs:
  https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v4.pdb
  https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v4.cif

Usage:
    python scripts/download_alphafold.py              # Download all needed
    python scripts/download_alphafold.py --dry-run    # List what would be downloaded

Structures needed:
    AF-Q9UGM1  — CHRNA9  (UniProt: Q9UGM1)
    AF-Q13002  — CHRNA10 (UniProt: Q13002)
"""

import argparse
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
STRUCTURE_DIR = PROJECT_ROOT / "data" / "raw" / "structure_files"

ALPHAFOLD_STRUCTURES = {
    "AF-Q9UGM1": {
        "gene": "CHRNA9",
        "uniprot": "Q9UGM1",
        "url_pdb": "https://alphafold.ebi.ac.uk/files/AF-Q9UGM1-F1-model_v6.pdb",
        "url_cif": "https://alphafold.ebi.ac.uk/files/AF-Q9UGM1-F1-model_v6.cif",
    },
    "AF-Q13002": {
        "gene": "CHRNA10",
        "uniprot": "Q13002",
        "url_pdb": "https://alphafold.ebi.ac.uk/files/AF-Q13002-F1-model_v6.pdb",
        "url_cif": "https://alphafold.ebi.ac.uk/files/AF-Q13002-F1-model_v6.cif",
    },
}


def download_file(url: str, dest: Path) -> bool:
    """Download a file. Returns True on success."""
    if dest.exists():
        print(f"  Already exists: {dest.name} ({dest.stat().st_size / 1024:.0f} KB)")
        return True

    print(f"  Downloading {dest.name}...", end=" ", flush=True)
    try:
        urllib.request.urlretrieve(url, str(dest))
        size_kb = dest.stat().st_size / 1024
        print(f"OK ({size_kb:.0f} KB)")
        return True
    except Exception as e:
        print(f"FAILED: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Download AlphaFold structures for VEP-nAChR2")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    STRUCTURE_DIR.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        print("Would download:")
        for af_id, info in ALPHAFOLD_STRUCTURES.items():
            print(f"  {af_id}/ ({info['gene']})")
            print(f"    {af_id}.pdb")
            print(f"    {af_id}.cif")
        return

    for af_id, info in ALPHAFOLD_STRUCTURES.items():
        pdb_dir = STRUCTURE_DIR / af_id
        pdb_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{info['gene']} ({af_id}, UniProt: {info['uniprot']})")
        print(f"  Directory: {pdb_dir}")

        ok_pdb = download_file(info["url_pdb"], pdb_dir / f"{af_id}.pdb")
        ok_cif = download_file(info["url_cif"], pdb_dir / f"{af_id}.cif")

        if ok_pdb and ok_cif:
            print(f"  -> OK. No DSSP needed (AlphaFold confidence = B-factor field)")
        else:
            print(f"  -> Some files failed. Check URLs manually.")

    print(f"\nNote: AlphaFold structures are monomeric (single chain A).")
    print(f"Features like interface_proximity and subunit_burial will use")
    print(f"fallback values since there are no neighbor chains.")


if __name__ == "__main__":
    main()
