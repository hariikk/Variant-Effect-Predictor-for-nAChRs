"""Find the best (chain, offset) mapping for each gene across ALL chains in its PDB.

Reveals misassigned chains and the true precursor->PDB offset. Reports the best
chain+offset+match rate, and flags genes whose best match is <85% (homology or
wrong isoform -> needs an AlphaFold model or chain reassignment).
"""
from Bio.PDB.Polypeptide import protein_letters_3to1, is_aa
from vep_nachr2.data.reference import load_all_reference_sequences
from vep_nachr2.features.structural import load_all_pdb_resources
from vep_nachr2.config import PDB_MAPPING

refs = load_all_reference_sequences("human")
pdb_res = load_all_pdb_resources()

print(f"{'gene':9s} {'assigned':8s} {'bestChain':9s} {'refL':5s} {'pdbL':5s} "
      f"{'bestOff':8s} {'bestMatch':9s}  flag")
for gene, info in PDB_MAPPING.items():
    ref = refs.get(gene)
    pdb_id = info["pdb_id"]
    resources = pdb_res.get(pdb_id)
    if ref is None or resources is None:
        print(f"{gene:9s} {info['chain']:8s} {'-':9s}  (no ref/resources)")
        continue

    best = None  # (match, chain_id, offset, L)
    for chain in resources["structure"][0]:
        pdb_seq = "".join(
            protein_letters_3to1[r.resname]
            for r in chain if is_aa(r) and r.resname in protein_letters_3to1
        )
        L = len(pdb_seq)
        if L == 0:
            continue
        for o in range(0, max(0, len(ref) - L + 1)):
            m = sum(1 for j in range(L) if ref[o + j] == pdb_seq[j])
            rate = m / L
            if best is None or rate > best[0]:
                best = (rate, chain.id, o, L)

    if best is None:
        chain_lens = ", ".join(
            f"{c.id}:{sum(1 for r in c if is_aa(r) and r.resname in protein_letters_3to1)}"
            for c in resources["structure"][0]
        )
        print(f"{gene:9s} {info['chain']:8s} {'-':9s} {len(ref):5d} {'-':5s} "
              f"{'-':8s} {'-':9s}  ref shorter than all chains -> wrong isoform "
              f"(chainLens: {chain_lens})")
        continue

    rate, cid, off, L = best
    flag = ""
    if cid != info["chain"]:
        flag += " CHAIN-MISMATCH"
    if rate < 0.85:
        flag += " LOW-MATCH(homology/isoform)"
    print(f"{gene:9s} {info['chain']:8s} {cid:9s} {len(ref):5d} {L:5d} "
          f"{off:8d} {rate:8.2%} {flag}")
