"""Smoke-test the structural-family extractors (no ESM) before the full run."""
import numpy as np

from vep_nachr2.data.loader import load_mutation_data
from vep_nachr2.data.reference import load_all_reference_sequences
from vep_nachr2.features.structural import load_all_pdb_resources, StructuralExtractor
from vep_nachr2.features.stability import StabilityExtractor
from vep_nachr2.features.charge_context import ChargeContextExtractor
from vep_nachr2.features.functional_motif import FunctionalMotifExtractor

df = load_mutation_data(data_file="final_mapped.xlsx")
ref_seqs = load_all_reference_sequences("human")
pdb = load_all_pdb_resources()

for ex in [StructuralExtractor(), StabilityExtractor(),
           ChargeContextExtractor(), FunctionalMotifExtractor()]:
    X = ex.extract(df, ref_seqs=ref_seqs, pdb_resources=pdb)
    assert X.shape == (len(df), ex.n_features), f"{ex.name} shape {X.shape}"
    print(f"\n{ex.name}: {X.shape} (expected {ex.n_features})")
    for j, nm in enumerate(ex.get_feature_names()):
        col = X[:, j]
        print(f"   {nm:22s} min={col.min():7.3f} max={col.max():7.3f} "
              f"mean={col.mean():7.3f} frac_zero={(col == 0).mean():.3f}")

print("\nOK: all structural-family extractors ran without error.")
