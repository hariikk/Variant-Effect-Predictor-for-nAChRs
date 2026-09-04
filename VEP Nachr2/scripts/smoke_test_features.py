"""Smoke test: extract full feature matrix and report per-group sanity stats."""
import numpy as np

from vep_nachr2.data.loader import load_mutation_data
from vep_nachr2.data.reference import load_all_reference_sequences
from vep_nachr2.features.orchestrator import FeatureOrchestrator

df = load_mutation_data(data_file="final_mapped.xlsx")
print(f"Loaded {len(df)} variants")

ref_seqs = load_all_reference_sequences("human")
orch = FeatureOrchestrator(verbose=True)
X, names = orch.extract(df, ref_seqs=ref_seqs, use_cache=True)

print(f"\nX shape: {X.shape}, n_features={len(names)}")

group_idx = orch.get_feature_group_indices()
print("\nPer-group sanity (nan / inf / %non-zero):")
for gname, (s, e) in group_idx.items():
    if e <= s:
        continue
    block = X[:, s:e]
    n_nan = int(np.isnan(block).sum())
    n_inf = int(np.isinf(block).sum())
    nz = float(np.count_nonzero(block)) / block.size * 100.0
    rng = f"min={np.nanmin(block):.4g} max={np.nanmax(block):.4g}" if block.size else "-"
    print(f"  {gname:20s} nan={n_nan:5d} inf={n_inf:5d} nz={nz:5.1f}%  {rng}")
