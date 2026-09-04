"""Find the optimal precursor->PDB-chain offset per gene (whole-sequence match).

For each gene, scan every offset o and report the one maximizing AA matches
between ref_seq[o:o+L] and the PDB chain sequence, plus the match rate at the
signal-peptide offset (from normalize_positions.py). A direct-mapped gene should
show best_offset == signal_peptide_len with ~100% match.
"""
from Bio.PDB.Polypeptide import protein_letters_3to1, is_aa
from vep_nachr2.data.reference import load_all_reference_sequences
from vep_nachr2.features.structural import load_all_pdb_resources
from vep_nachr2.config import PDB_MAPPING

refs = load_all_reference_sequences("human")
pdb_res = load_all_pdb_resources()

# Human signal-peptide lengths (from scripts/normalize_positions.py).
sig = {
    "CHRNA1": 20, "CHRNA2": 26, "CHRNA3": 31, "CHRNA4": 26, "CHRNA5": 22,
    "CHRNA6": 33, "CHRNA7": 22, "CHRNA9": 22, "CHRNA10": 24, "CHRNB1": 23,
    "CHRNB2": 25, "CHRNB3": 21, "CHRNB4": 26, "CHRND": 21, "CHRNE": 20,
    "CHRNG": 22,
}

print(f"{'gene':9s} {'pdb':9s} {'chain':5s} {'refL':5s} {'pdbL':5s} {'sig':4s} "
      f"{'bestOff':8s} {'bestMatch':10s} {'match@sig':10s}")
for gene, info in PDB_MAPPING.items():
    ref = refs.get(gene)
    pdb_id = info["pdb_id"]
    chain_id = info["chain"]
    resources = pdb_res.get(pdb_id)
    if ref is None or resources is None:
        continue
    chain = resources["structure"][0][chain_id]
    pdb_seq = "".join(
        protein_letters_3to1[r.resname]
        for r in chain if is_aa(r) and r.resname in protein_letters_3to1
    )
    L = len(pdb_seq)
    s = sig.get(gene, 0)

    def match_rate(o):
        if o < 0 or o + L > len(ref):
            return 0.0
        m = sum(1 for j in range(L) if ref[o + j] == pdb_seq[j])
        return m / L

    best_off = max(range(0, len(ref) - L + 1), key=match_rate)
    print(f"{gene:9s} {pdb_id:9s} {chain_id:5s} {len(ref):5d} {L:5d} {s:4d} "
          f"{best_off:8d} {match_rate(best_off):9.2%} {match_rate(s):9.2%}")
