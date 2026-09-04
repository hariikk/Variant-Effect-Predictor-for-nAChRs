"""Diagnose structural mapping quality: does the aligned PDB residue carry the
wildtype AA? Direct-match genes should be ~100%; homology-mapped genes will show
the expected sequence-identity gap. Also reports RSA fill / unmappable rates."""
import numpy as np
import pandas as pd
from Bio.PDB.Polypeptide import protein_letters_3to1, is_aa

from vep_nachr2.data.loader import load_mutation_data
from vep_nachr2.data.reference import load_all_reference_sequences
from vep_nachr2.features.structural import load_all_pdb_resources, _build_all_alignment_maps
from vep_nachr2.config import PDB_MAPPING

df = load_mutation_data()
refs = load_all_reference_sequences("human")
pdb_res = load_all_pdb_resources()
maps = _build_all_alignment_maps(refs, pdb_res)

# PDB chain residue AA: (pdb_id, chain_id) -> {resseq: one-letter AA}
chain_aa = {}
for pdb_id, resources in pdb_res.items():
    if resources is None:
        continue
    structure = resources["structure"]
    model = structure[0]
    for chain in model:
        key = (pdb_id, chain.id)
        chain_aa[key] = {}
        for res in chain:
            if is_aa(res) and res.resname in protein_letters_3to1:
                chain_aa[key][res.id[1]] = protein_letters_3to1[res.resname]

rows = []
for _, row in df.iterrows():
    gene = str(row["subunit"]).upper()
    pos = int(row["position"])
    wt = str(row["wildtype_aa"]).upper()
    info = PDB_MAPPING.get(gene)
    if info is None:
        continue
    pdb_id, chain_id = info["pdb_id"], info["chain"]
    amap = maps.get(gene, {})
    pdb_res_id = amap.get(pos)
    mapped = pdb_res_id is not None
    aa_match = None
    if mapped:
        pdb_aa = chain_aa.get((pdb_id, chain_id), {}).get(pdb_res_id[1])
        aa_match = (pdb_aa == wt) if pdb_aa else None
    rows.append({
        "gene": gene, "pdb_id": pdb_id, "chain": chain_id,
        "homology": "homology" in str(info.get("homology_parent", "")),
        "source": info.get("source"),
        "mapped": mapped, "aa_match": aa_match,
    })

r = pd.DataFrame(rows)
r["homology"] = r["gene"].isin(
    [g for g, i in PDB_MAPPING.items() if "homology_parent" in i])

print("=== per-gene mapping quality ===")
g = r.groupby("gene").agg(
    n=("mapped", "size"),
    pct_mapped=("mapped", "mean"),
    pct_aa_match=("aa_match", "mean"),
    n_aa_known=("aa_match", "count"),
    homology=("homology", "first"),
    pdb=("pdb_id", "first"),
)
g["pct_mapped"] = (g["pct_mapped"] * 100).round(0).astype(int)
g["pct_aa_match"] = (g["pct_aa_match"] * 100).round(0)
print(g.to_string())

print("\n=== summary: mapped-variant AA identity ===")
mapped = r[r["mapped"]]
for label, sub in [("direct (16 genes, non-homology)", mapped[~mapped["homology"]]),
                   ("homology-mapped genes", mapped[mapped["homology"]])]:
    n = len(sub)
    match = sub["aa_match"].sum()
    print(f"  {label:35s} {match}/{n} AA-identical "
          f"({100*match/max(n,1):.1f}%)")
