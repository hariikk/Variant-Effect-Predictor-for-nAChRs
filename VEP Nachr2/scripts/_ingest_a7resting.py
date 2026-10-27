"""Ingest the new homomeric α7 resting-state structure (7KOO cryo-EM + 7RPM NMR).

Converts homomeric_a7_resting_complete.pdb -> data/raw/structure_files/a7rest/a7rest.cif
(stripping explicit hydrogens so SASA/Cβ-density/B-factor are consistent with the
other heavy-atom-only structures), then loads it through the existing pipeline and
reports how many CHRNA7 variants map vs the current 7EKT structure.

Run once:  python scripts/_ingest_a7resting.py
"""

import sys
import warnings

sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

from pathlib import Path

from Bio import PDB
from Bio.PDB import PDBParser, MMCIFIO, is_aa

from vep_nachr2.config import STRUCTURE_DIR, PDB_MAPPING
from vep_nachr2.data.loader import load_mutation_data
from vep_nachr2.data.reference import load_all_reference_sequences
from vep_nachr2.features.structural import (
    _load_pdb_resources,
    _build_all_alignment_maps,
)

SRC = Path("homomeric_a7_resting_complete.pdb")
DST_DIR = STRUCTURE_DIR / "a7rest"
DST_CIF = DST_DIR / "a7rest.cif"


def convert(src: Path, dst: Path) -> None:
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("a7rest", str(src))

    n_h = 0
    n_atom = 0
    for model in structure:
        for chain in model:
            for residue in list(chain):
                if not is_aa(residue):
                    continue
                for atom in list(residue):
                    n_atom += 1
                    if atom.element == "H":
                        residue.detach_child(atom.id)
                        n_h += 1

    io = MMCIFIO()
    io.set_structure(structure)
    dst.parent.mkdir(parents=True, exist_ok=True)
    io.save(str(dst))
    print(f"stripped {n_h}/{n_atom} hydrogens; wrote {dst}")


def main() -> None:
    if not DST_CIF.exists():
        convert(SRC, DST_CIF)
    else:
        print(f"{DST_CIF} already exists; skipping conversion")

    # Load through the standard pipeline path
    resources = _load_pdb_resources("a7rest")
    if resources is None:
        print("ERROR: failed to load a7rest resources")
        return
    structure = resources["structure"]
    print(f"\na7rest structure: {len(list(structure.get_chains()))} chains")
    chainA = structure[0]["A"]
    n_res = sum(1 for r in chainA if is_aa(r))
    print(f"chain A: {n_res} residues")

    # Alignment map for CHRNA7 (mature 23..502 vs precursor reference)
    ref_seqs = load_all_reference_sequences("human")
    maps = _build_all_alignment_maps(ref_seqs, {"a7rest": resources})
    a7map = maps.get("CHRNA7", {})
    print(f"CHRNA7 a7rest alignment map: {len(a7map)} positions mapped")

    # Compare against 7EKT on the actual variant set
    df = load_mutation_data(data_file="final_mapped.xlsx")
    chrna7 = df[df["subunit"] == "CHRNA7"]
    print(f"\nCHRNA7 variants in dataset: {len(chrna7)}")

    base = _load_pdb_resources("7EKT")
    maps_7ekt = _build_all_alignment_maps(ref_seqs, {"7EKT": base})
    a7_7ekt = maps_7ekt.get("CHRNA7", {})

    positions = chrna7["position"].astype(int).tolist()
    mapped_new = sum(1 for p in positions if p in a7map)
    mapped_old = sum(1 for p in positions if p in a7_7ekt)
    print(f"  mapped under 7EKT (current): {mapped_old}/{len(positions)}")
    print(f"  mapped under a7rest (new):   {mapped_new}/{len(positions)}")
    print(f"  newly-mapped positions (a7rest only): {sorted(set(positions) & (set(a7map) - set(a7_7ekt)))}")


if __name__ == "__main__":
    main()
