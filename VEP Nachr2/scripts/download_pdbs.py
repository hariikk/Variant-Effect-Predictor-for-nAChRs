#!/usr/bin/env python3
"""
Download PDB structures and DSSP files for VEP-nAChR2.

Downloads .pdb and .cif from RCSB PDB. DSSP files are obtained by:
  1. Trying mkdssp (if installed: conda install -c salilab dssp)
  2. Falling back to PDBe pre-computed DSSP (no external tools needed)

Usage:
    python scripts/download_pdbs.py              # Download all required PDBs
    python scripts/download_pdbs.py --pdb 9DMG   # Download a specific PDB
    python scripts/download_pdbs.py --dry-run    # List what would be downloaded
    python scripts/download_pdbs.py --no-dssp    # Skip DSSP entirely

PDBs needed:
    9DMG  — Human muscle a1b1de (2.05 A, apo) -> CHRNA1, CHRNB1, CHRND, CHRNE, CHRNG
    7EKT  — Human a7 homopentamer (3.20 A, closed) -> CHRNA7
    7KOX  — Human a7 activated state (epibatidine+PNU) -> CHRNA7 (conformational)
    6CNJ  — Human a4b2 neuronal (3.30 A) -> CHRNA4, CHRNB2, CHRNA2, CHRNA6
    6PV7  — Human a3b4 neuronal (2.80 A) -> CHRNA3, CHRNB4, CHRNA5, CHRNB3
"""

import argparse
import subprocess
import sys
import urllib.request
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent.parent
STRUCTURE_DIR = PROJECT_ROOT / "data" / "raw" / "structure_files"

# PDBs to download
PDB_IDS = ["9DMG", "7EKT", "7KOX", "6CNJ", "6PV7"]

# RCSB URLs
PDB_URL = "https://files.rcsb.org/download/{pdb_id}.pdb"
CIF_URL = "https://files.rcsb.org/download/{pdb_id}.cif"

# PDBe DSSP URL (pre-computed, no mkdssp needed)
# Source: PDBe knowledge base secondary structure files
PDBE_DSSP_URL = "https://www.ebi.ac.uk/pdbe/static/entry/pdb{pdb_id_lower}.ent"
# Fallback: RCSB PDB secondary structure via 1D SIFTS mapping
RCSB_SSE_URL = "https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"


def download_file(url: str, dest: Path) -> bool:
    """Download a file from URL to destination. Returns True on success."""
    if dest.exists():
        print(f"  Already exists: {dest.name}")
        return True

    print(f"  Downloading {dest.name}...", end=" ", flush=True)
    try:
        urllib.request.urlretrieve(url, str(dest))
        size_mb = dest.stat().st_size / (1024 * 1024)
        print(f"OK ({size_mb:.1f} MB)")
        return True
    except Exception as e:
        print(f"FAILED: {e}")
        return False


def run_dssp(pdb_dir: Path, pdb_id: str) -> bool:
    """Generate DSSP file: try mkdssp first, then CIF extraction, then PDBe API."""
    dssp_path = pdb_dir / f"{pdb_id}.dssp"

    if dssp_path.exists():
        # Check if it has actual residue data (not just header)
        content = dssp_path.read_text()
        if len(content.splitlines()) > 2:
            print(f"  DSSP already exists: {dssp_path.name}")
            return True
        # Empty DSSP from previous run — regenerate

    cif_path = pdb_dir / f"{pdb_id}.cif"
    pdb_path = pdb_dir / f"{pdb_id}.pdb"

    # ── Method 1: mkdssp (gives full DSSP with ASA) ──
    input_path = cif_path if cif_path.exists() else pdb_path
    try:
        result = subprocess.run(
            ["mkdssp", "-i", str(input_path), "-o", str(dssp_path)],
            capture_output=True, text=True, timeout=300
        )
        if result.returncode == 0 and dssp_path.exists():
            print(f"  DSSP generated via mkdssp")
            return True
    except FileNotFoundError:
        pass
    except Exception:
        pass

    # ── Method 2: Extract SS from CIF file (most reliable, no network needed) ──
    if cif_path.exists():
        try:
            from Bio.PDB import MMCIF2Dict
            print(f"  Extracting SS from CIF...", end=" ", flush=True)
            cif_dict = MMCIF2Dict.MMCIF2Dict(str(cif_path))
            ss_map = _extract_ss_from_cif(cif_dict)
            if ss_map:
                _write_dssp_from_map(dssp_path, pdb_id, ss_map)
                print(f"OK ({len(ss_map)} residues)")
                return True
            else:
                print("no SS annotations found")
        except ImportError:
            print("Bio.PDB not available")
        except Exception as e:
            print(f"failed ({e})")

    # ── Method 3: PDBe API (tries both old and new response formats) ──
    pdb_lower = pdb_id.lower()
    pdbe_url = f"https://www.ebi.ac.uk/pdbe/api/pdb/entry/secondary_structure/{pdb_lower}"

    try:
        print(f"  Trying PDBe DSSP API...", end=" ", flush=True)
        import json
        req = urllib.request.Request(pdbe_url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())

        if pdb_lower in data:
            ss_map = _extract_ss_from_pdbe(data[pdb_lower])
            if ss_map:
                _write_dssp_from_map(dssp_path, pdb_id, ss_map)
                print(f"OK ({len(ss_map)} residues from PDBe)")
                return True
            else:
                print("no SS annotations in PDBe response")
        else:
            print("not found in PDBe")
    except Exception as e:
        print(f"unavailable ({e})")

    print(f"  NOTE: DSSP unavailable for {pdb_id}. Structural features will use fill values.")
    print(f"  To get full DSSP: install mkdssp and re-run.")
    return False


def _extract_ss_from_cif(cif_dict: dict) -> dict:
    """Extract per-residue secondary structure from mmCIF dict.

    Parses _struct_conf (helices) and _struct_sheet_range (strands) categories.
    Returns {(chain_id, resnum): ss_code} where ss_code is 'H' or 'E'.
    """
    ss_map = {}

    # Helices from _struct_conf
    try:
        n_helices = len(cif_dict.get("_struct_conf.conf_type_id", []))
        for i in range(n_helices):
            chain = cif_dict["_struct_conf.beg_label_asym_id"][i]
            start = int(cif_dict["_struct_conf.beg_auth_seq_id"][i])
            end = int(cif_dict["_struct_conf.end_auth_seq_id"][i])
            for r in range(start, end + 1):
                ss_map[(chain, r)] = "H"
    except (KeyError, ValueError, IndexError):
        pass

    # Sheets from _struct_sheet_range
    try:
        n_strands = len(cif_dict.get("_struct_sheet_range.sheet_id", []))
        for i in range(n_strands):
            chain = cif_dict["_struct_sheet_range.beg_label_asym_id"][i]
            start = int(cif_dict["_struct_sheet_range.beg_auth_seq_id"][i])
            end = int(cif_dict["_struct_sheet_range.end_auth_seq_id"][i])
            for r in range(start, end + 1):
                if (chain, r) not in ss_map:
                    ss_map[(chain, r)] = "E"
    except (KeyError, ValueError, IndexError):
        pass

    return ss_map


def _extract_ss_from_pdbe(ss_data: dict) -> dict:
    """Extract per-residue secondary structure from PDBe API response.

    Handles both the old format (top-level helices/sheets) and the new
    format (nested under molecules -> chains -> secondary_structure).
    """
    ss_map = {}

    def _add_range(chain, start, end, code):
        try:
            for r in range(int(start), int(end) + 1):
                if (chain, r) not in ss_map:
                    ss_map[(chain, r)] = code
        except (ValueError, TypeError):
            pass

    # Try new format: molecules -> chains -> secondary_structure
    for molecule in ss_data.get("molecules", []):
        for chain_data in molecule.get("chains", []):
            chain_id = chain_data.get("chain_id", "A")
            ss = chain_data.get("secondary_structure", {})
            for helix in ss.get("helices", []):
                _add_range(
                    chain_id,
                    helix.get("start", {}).get("author_residue_number"),
                    helix.get("end", {}).get("author_residue_number"),
                    "H"
                )
            for sheet in ss.get("sheets", []):
                for strand in sheet.get("strands", []):
                    _add_range(
                        chain_id,
                        strand.get("start", {}).get("author_residue_number"),
                        strand.get("end", {}).get("author_residue_number"),
                        "E"
                    )

    # Fallback: old format (top-level helices/sheets)
    if not ss_map:
        for helix in ss_data.get("helices", []):
            _add_range(
                helix.get("start", {}).get("author_asym_id", "A"),
                helix.get("start", {}).get("author_residue_number"),
                helix.get("end", {}).get("author_residue_number"),
                "H"
            )
        for sheet in ss_data.get("sheets", []):
            for strand in sheet.get("strands", []):
                _add_range(
                    strand.get("start", {}).get("author_asym_id", "A"),
                    strand.get("start", {}).get("author_residue_number"),
                    strand.get("end", {}).get("author_residue_number"),
                    "E"
                )

    return ss_map


def _write_dssp_from_map(dssp_path: Path, pdb_id: str, ss_map: dict):
    """Write a DSSP-format file from a {(chain, resnum): ss_code} mapping.

    Column positions must match Bio.PDB.DSSP._make_dssp_dict():
      col 0-4   : dssp_index (5 chars, int)
      col 5-9   : resseq (5 chars, int)
      col 10    : icode (insertion code, space)
      col 11    : chainid (1 char)
      col 13    : aa (1-letter AA code, 'X' for unknown)
      col 16    : ss (H/E/space for coil)
      col 34-37 : acc (4 chars, int — 0 = unknown)
      col 103-108 : phi (float, 360.0 = unknown)
      col 109-114 : psi (float, 360.0 = unknown)
    """
    with open(dssp_path, "w") as f:
        f.write(f"==== DSSP file for {pdb_id} (simplified, SS only) ====\n")
        f.write("  #  RESIDUE AA STRUCTURE BP1 BP2 ACC    N-H-->O    O-->H-N    N-H-->O    O-->H-N    TCO  KAPPA ALPHA PHI   PSI    X-CA   Y-CA   Z-CA\n")
        for idx, ((chain, resnum), ss) in enumerate(sorted(ss_map.items()), 1):
            # Build line with exact column positions
            idx_str = f"{idx:>5}"           # col 0-4
            res_str = f"{resnum:>5}"        # col 5-9
            icode = " "                      # col 10
            chain_str = chain                # col 11
            aa = "X"                         # col 13 (unknown AA)
            ss_code = ss                     # col 16
            acc_str = "   0"                 # col 34-37 (unknown ASA)
            phi_str = " 360.0"               # col 103-108
            psi_str = " 360.0"               # col 109-114
            # Compose: col0-4, 5-9, 10, 11, 12, 13, 14, 15, 16, then padding, then 34-37
            line = (
                f"{idx_str}{res_str}{icode}{chain_str} X{aa}   {ss_code}"
                f"   0    0    0"                    # BP1/BP2 placeholders
                f"{acc_str}"
                f"      0, 0.0     0, 0.0     0, 0.0     0, 0.0"  # H-bond placeholders
                f"   0.000 360.0 360.0"
                f"{phi_str}{psi_str}"
                f"   0.0   0.0   0.0"               # XYZ placeholders
                f"\n"
            )
            f.write(line)


def download_pdb(pdb_id: str, run_dssp_flag: bool = True) -> bool:
    """Download all files for a single PDB ID."""
    pdb_dir = STRUCTURE_DIR / pdb_id
    pdb_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*50}")
    print(f"PDB: {pdb_id}")
    print(f"  Directory: {pdb_dir}")
    print(f"{'='*50}")

    success = True

    # Download PDB
    if not download_file(PDB_URL.format(pdb_id=pdb_id), pdb_dir / f"{pdb_id}.pdb"):
        success = False

    # Download CIF
    if not download_file(CIF_URL.format(pdb_id=pdb_id), pdb_dir / f"{pdb_id}.cif"):
        success = False

    # Run DSSP
    if run_dssp_flag and success:
        run_dssp(pdb_dir, pdb_id)

    return success


def main():
    parser = argparse.ArgumentParser(description="Download PDB structures for VEP-nAChR2")
    parser.add_argument("--pdb", type=str, nargs="*", default=None,
                        help=f"Specific PDB IDs to download (default: all). Available: {PDB_IDS}")
    parser.add_argument("--no-dssp", action="store_true",
                        help="Skip DSSP computation")
    parser.add_argument("--dry-run", action="store_true",
                        help="List what would be downloaded without downloading")
    args = parser.parse_args()

    pdb_ids = args.pdb if args.pdb else PDB_IDS

    # Validate
    for pdb_id in pdb_ids:
        if pdb_id not in PDB_IDS:
            print(f"Warning: {pdb_id} not in known PDB list. Known: {PDB_IDS}")

    STRUCTURE_DIR.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        print("Dry run — would download:")
        for pdb_id in pdb_ids:
            print(f"  {pdb_id}/")
            print(f"    {pdb_id}.pdb")
            print(f"    {pdb_id}.cif")
            if not args.no_dssp:
                print(f"    {pdb_id}.dssp")
        return

    # Download each PDB
    results = {}
    for pdb_id in pdb_ids:
        results[pdb_id] = download_pdb(pdb_id, run_dssp_flag=not args.no_dssp)

    # Summary
    print(f"\n{'='*50}")
    print("Summary")
    print(f"{'='*50}")
    for pdb_id, ok in results.items():
        status = "OK" if ok else "FAILED"
        print(f"  {pdb_id}: {status}")

    print(f"\nNext steps:")
    print(f"  1. If DSSP was unavailable for any PDB, install mkdssp and re-run:")
    print(f"       conda install -c salilab dssp   (or: pip install pdb-tools)")
    print(f"       python scripts/download_pdbs.py")
    print(f"  2. Run: python scripts/run_experiment.py --test")


if __name__ == "__main__":
    main()
