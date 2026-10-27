# -*- coding: utf-8 -*-
"""One-off diagnostic: quantify structural-feature coverage per gene."""
import sys
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

from vep_nachr2.data.loader import load_mutation_data
from vep_nachr2.data.reference import load_all_reference_sequences
from vep_nachr2.features.structural import load_all_pdb_resources, StructuralExtractor
from vep_nachr2.features.gating import GatingExtractor
from vep_nachr2.config import PDB_MAPPING

df = load_mutation_data(data_file="final_mapped.xlsx")
ref_seqs = load_all_reference_sequences("human")
pdb_resources = load_all_pdb_resources()

struct = StructuralExtractor().extract(df, ref_seqs, pdb_resources)
is_unmappable = struct[:, 6]          # 1 = no PDB residue mapped
has_gate = GatingExtractor().extract(df, ref_seqs, pdb_resources)[:, 2]  # gate_coupling

df = df.copy()
df["unmappable"] = is_unmappable
df["gate_coupling"] = has_gate
df["source"] = df["subunit"].map(lambda g: PDB_MAPPING.get(g, {}).get("source", "?"))

print("=" * 78)
print("Structural coverage — per gene")
print("=" * 78)
rows = []
for gene, g in df.groupby("subunit", sort=True):
    n = len(g)
    n_un = int(g["unmappable"].sum())
    n_map = n - n_un
    n_zero_coupling = int((g["gate_coupling"] == 0).sum())
    src = g["source"].iloc[0]
    pdb = PDB_MAPPING.get(gene, {}).get("pdb_id", "?")
    rows.append([gene, src, pdb, n, n_map, n_un, f"{100*n_un/n:.0f}%",
                 f"{100*n_zero_coupling/n:.0f}%"])

t = pd.DataFrame(rows, columns=["gene", "source", "pdb_id", "n", "mapped", "unmappable",
                                "%unmappable", "%zero_coupling"])
print(t.to_string(index=False))

print()
print("=" * 78)
print("Aggregate")
print("=" * 78)
n_total = len(df)
n_un = int(df["unmappable"].sum())
n_map = n_total - n_un
print(f"total variants            : {n_total}")
print(f"mapped to a PDB residue   : {n_map}  ({100*n_map/n_total:.1f}%)")
print(f"unmappable (fill values)  : {n_un}  ({100*n_un/n_total:.1f}%)")

print()
print("by structure source:")
for src, g in df.groupby("source", sort=False):
    n = len(g)
    n_un_s = int(g["unmappable"].sum())
    print(f"  {src:14s}: n={n:4d}  unmappable={n_un_s:4d} ({100*n_un_s/n:.1f}%)  "
          f"zero_coupling={100*int((g['gate_coupling']==0).sum())/n:.1f}%")

print()
print("experimental genes only (mapping imperfection within real structures):")
exp = df[df["source"] == "experimental"]
for gene, g in exp.groupby("subunit", sort=True):
    n = len(g)
    n_un_s = int(g["unmappable"].sum())
    print(f"  {gene:8s}: n={n:4d}  unmappable={n_un_s:4d} ({100*n_un_s/n:.1f}%)")

print()
print("by species:")
for sp, g in df.groupby("species", sort=False):
    n = len(g)
    n_un_s = int(g["unmappable"].sum())
    print(f"  {sp:8s}: n={n:4d}  unmappable={n_un_s:4d} ({100*n_un_s/n:.1f}%)")
