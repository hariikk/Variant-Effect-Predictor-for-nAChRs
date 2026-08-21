#!/usr/bin/env python3
"""
Generate real DSSP files (with solvent-accessible surface area) using mkdssp.

mkdssp is the canonical DSSP implementation (Kabsch & Sander, 1983).
It computes both secondary structure AND per-residue solvent accessibility,
which is what the structural_core RSA feature depends on.

Why WSL: mkdssp has no official Windows binary that runs without admin
elevation, and the user does not use conda. WSL (Ubuntu 22.04) ships the
`dssp` package (v4.0.4) in its apt universe repo. This script extracts that
package's binaries + shared libraries into ~/.local/mkdssp (no sudo needed)
and invokes mkdssp through WSL.

Input: <pdb_id>.pdb  (PDB format parses cleanly; .cif triggers harmless
       EM-metadata validation warnings on some cryo-EM entries)
Output: <pdb_id>.dssp (classic DSSP format, parsed by Bio.PDB.DSSP)

Usage:
    python scripts/generate_dssp.py                # all 7 required structures
    python scripts/generate_dssp.py --pdb 9DMG     # a specific structure
"""

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
STRUCTURE_DIR = PROJECT_ROOT / "data" / "raw" / "structure_files"

# The 7 structures actually used by PDB_MAPPING (5 experimental + 2 AlphaFold).
# 6UW8 and 7EKO are the old replaced PDBs — not in PDB_MAPPING, skipped.
REQUIRED_PDB_IDS = ["9DMG", "7EKT", "7KOX", "6CNJ", "6PV7"]
REQUIRED_ALPHAFOLD_IDS = ["AF-Q9UGM1", "AF-Q13002"]

# WSL locations for mkdssp (persist across WSL restarts — NOT /tmp)
MKDSSP_BIN = "$HOME/.local/mkdssp/bin/mkdssp"
MKDSSP_LIB = "$HOME/.local/mkdssp/lib"


def windows_to_wsl(path: Path) -> str:
    """Convert a Windows path to a WSL /mnt/... path."""
    s = str(path.resolve())
    s = s.replace("\\", "/")
    # C:/Users/... -> /mnt/c/Users/...
    if len(s) >= 2 and s[1] == ":":
        drive = s[0].lower()
        s = f"/mnt/{drive}{s[2:]}"
    return s


def setup_mkdssp() -> bool:
    """Extract mkdssp + deps into ~/.local/mkdssp (idempotent, no sudo)."""
    script = (
        f"if [ -x {MKDSSP_BIN} ]; then exit 0; fi; "
        "DST=$HOME/.local/mkdssp; mkdir -p $DST/bin $DST/lib; cd $DST; "
        "for pkg in dssp libcifpp2 libboost-iostreams1.74.0 "
        "libboost-program-options1.74.0; do apt-get download $pkg "
        ">/dev/null 2>&1; done; "
        "for f in *.deb; do dpkg -x $f .; done; "
        "cp usr/bin/mkdssp bin/ 2>/dev/null; "
        "cp usr/bin/dssp bin/ 2>/dev/null; "
        "cp usr/lib/x86_64-linux-gnu/libcifpp.so* lib/ 2>/dev/null; "
        "cp usr/lib/x86_64-linux-gnu/libboost_iostreams.so* lib/ 2>/dev/null; "
        "cp usr/lib/x86_64-linux-gnu/libboost_program_options.so* lib/ 2>/dev/null; "
        "rm -f *.deb; rm -rf usr; "
        f"test -x {MKDSSP_BIN}"
    )
    result = subprocess.run(
        ["wsl", "-e", "bash", "-lc", script],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"ERROR: mkdssp setup failed:\n{result.stderr}")
        return False
    return True


def generate_dssp(pdb_id: str) -> bool:
    """Generate <pdb_id>.dssp from <pdb_id>.pdb via mkdssp in WSL."""
    pdb_dir = STRUCTURE_DIR / pdb_id
    pdb_path = pdb_dir / f"{pdb_id}.pdb"
    dssp_path = pdb_dir / f"{pdb_id}.dssp"

    if not pdb_path.exists():
        print(f"  SKIP {pdb_id}: {pdb_path.name} missing")
        return False

    # Back up existing (simplified) DSSP if present
    if dssp_path.exists():
        dssp_path.replace(dssp_path.with_suffix(".dssp.old"))

    wsl_pdb = windows_to_wsl(pdb_path)
    wsl_dssp = windows_to_wsl(dssp_path)

    script = (
        f"export LD_LIBRARY_PATH={MKDSSP_LIB}:$LD_LIBRARY_PATH; "
        f"{MKDSSP_BIN} --output-format dssp "
        f"-i '{wsl_pdb}' -o '{wsl_dssp}'"
    )
    result = subprocess.run(
        ["wsl", "-e", "bash", "-lc", script],
        capture_output=True, text=True, timeout=600,
    )

    if result.returncode != 0 or not dssp_path.exists():
        print(f"  FAIL {pdb_id}: {result.stderr.strip()[:200]}")
        return False

    # Validate: count residues with ASA > 0
    try:
        from Bio.PDB.DSSP import make_dssp_dict
        d, _ = make_dssp_dict(str(dssp_path))
        accs = [float(v[2]) for v in d.values()]
        nz = sum(1 for a in accs if a > 0)
        pct = 100 * nz / len(accs) if accs else 0
        print(f"  OK {pdb_id}: {len(accs)} residues, {nz} with ASA>0 ({pct:.1f}%)")
        # Remove backup on success
        backup = dssp_path.with_suffix(".dssp.old")
        if backup.exists():
            backup.unlink()
        return True
    except Exception as e:
        print(f"  WARN {pdb_id}: generated but parse failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Generate DSSP files via mkdssp (WSL)")
    parser.add_argument("--pdb", type=str, nargs="*", default=None,
                        help="Specific PDB IDs (default: all 7 required)")
    args = parser.parse_args()

    pdb_ids = args.pdb if args.pdb else (REQUIRED_PDB_IDS + REQUIRED_ALPHAFOLD_IDS)

    print("Checking mkdssp setup in WSL...")
    if not setup_mkdssp():
        sys.exit(1)

    print(f"\nGenerating DSSP for {len(pdb_ids)} structures...")
    results = {}
    for pdb_id in pdb_ids:
        results[pdb_id] = generate_dssp(pdb_id)

    print(f"\n{'='*50}")
    print("Summary")
    print(f"{'='*50}")
    for pdb_id, ok in results.items():
        print(f"  {'[OK]' if ok else '[FAIL]'} {pdb_id}")


if __name__ == "__main__":
    main()
