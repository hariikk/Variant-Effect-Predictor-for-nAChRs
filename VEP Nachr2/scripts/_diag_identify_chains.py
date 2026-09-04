"""Identify what each PDB chain actually is: for every chain in 9DMG and 6PV7,
report its best-matching reference subunit + identity rate. Resolves chain
misassignments (e.g. is 9DMG chain E really beta1?).
"""
from Bio.PDB.Polypeptide import protein_letters_3to1, is_aa
from vep_nachr2.data.reference import load_all_reference_sequences
from vep_nachr2.features.structural import load_all_pdb_resources

refs = load_all_reference_sequences("human")
pdb_res = load_all_pdb_resources()


def best_ref_match(chain_seq):
    best = None
    for gene, ref in refs.items():
        L = len(chain_seq)
        if L > len(ref):
            continue
        best_m = 0
        for o in range(0, len(ref) - L + 1):
            m = sum(1 for j in range(L) if ref[o + j] == chain_seq[j])
            if m > best_m:
                best_m = m
        rate = best_m / L
        if best is None or rate > best[0]:
            best = (rate, gene)
    return best


for pdb_id in ["9DMG", "6PV7", "6CNJ", "7EKT"]:
    resources = pdb_res.get(pdb_id)
    if resources is None:
        print(f"{pdb_id}: no resources")
        continue
    print(f"\n=== {pdb_id} chain composition ===")
    for chain in resources["structure"][0]:
        seq = "".join(
            protein_letters_3to1[r.resname]
            for r in chain if is_aa(r) and r.resname in protein_letters_3to1
        )
        if not seq:
            continue
        rate, gene = best_ref_match(seq)
        print(f"  chain {chain.id}: L={len(seq):3d}  -> {gene:9s} ({rate:.1%})")
