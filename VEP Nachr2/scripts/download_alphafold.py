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
    AF-Q9GZZ6  — CHRNA10 (UniProt: Q9GZZ6)  [correct canonical accession]
    AF-Q15822  — CHRNA2  (UniProt: Q15822)  [was homology-mapped to CHRNA4]
    AF-P30532  — CHRNA5  (UniProt: P30532)  [was homology-mapped to CHRNA3]
    AF-Q15825  — CHRNA6  (UniProt: Q15825)  [was homology-mapped to CHRNA4]
    AF-Q05901  — CHRNB3  (UniProt: Q05901)  [was homology-mapped to CHRNB4]
    AF-P07510  — CHRNG   (UniProt: P07510)  [was homology-mapped to CHRNE]
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
    "AF-Q9GZZ6": {
        "gene": "CHRNA10",
        "uniprot": "Q9GZZ6",
        "url_pdb": "https://alphafold.ebi.ac.uk/files/AF-Q9GZZ6-F1-model_v6.pdb",
        "url_cif": "https://alphafold.ebi.ac.uk/files/AF-Q9GZZ6-F1-model_v6.cif",
    },
    "AF-Q15822": {
        "gene": "CHRNA2",
        "uniprot": "Q15822",
        "url_pdb": "https://alphafold.ebi.ac.uk/files/AF-Q15822-F1-model_v6.pdb",
        "url_cif": "https://alphafold.ebi.ac.uk/files/AF-Q15822-F1-model_v6.cif",
    },
    "AF-P30532": {
        "gene": "CHRNA5",
        "uniprot": "P30532",
        "url_pdb": "https://alphafold.ebi.ac.uk/files/AF-P30532-F1-model_v6.pdb",
        "url_cif": "https://alphafold.ebi.ac.uk/files/AF-P30532-F1-model_v6.cif",
    },
    "AF-Q15825": {
        "gene": "CHRNA6",
        "uniprot": "Q15825",
        "url_pdb": "https://alphafold.ebi.ac.uk/files/AF-Q15825-F1-model_v6.pdb",
        "url_cif": "https://alphafold.ebi.ac.uk/files/AF-Q15825-F1-model_v6.cif",
    },
    "AF-Q05901": {
        "gene": "CHRNB3",
        "uniprot": "Q05901",
        "url_pdb": "https://alphafold.ebi.ac.uk/files/AF-Q05901-F1-model_v6.pdb",
        "url_cif": "https://alphafold.ebi.ac.uk/files/AF-Q05901-F1-model_v6.cif",
    },
    "AF-P07510": {
        "gene": "CHRNG",
        "uniprot": "P07510",
        "url_pdb": "https://alphafold.ebi.ac.uk/files/AF-P07510-F1-model_v6.pdb",
        "url_cif": "https://alphafold.ebi.ac.uk/files/AF-P07510-F1-model_v6.cif",
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
