"""Clear the feature cache and rebuild the full default feature matrix (foreground).

The ESM-2 forward pass is the GPU-heavy step that gets killed when run in the
background on Windows; this runs it in the foreground and writes the cache so the
later ablation can run CPU-only against the cache.
"""
from vep_nachr2.data.loader import load_mutation_data
from vep_nachr2.data.reference import load_all_reference_sequences
from vep_nachr2.features.orchestrator import FeatureOrchestrator, clear_feature_cache

clear_feature_cache()

df = load_mutation_data(data_file="final_mapped.xlsx")
ref_seqs = load_all_reference_sequences("human")
orch = FeatureOrchestrator(verbose=True)
X, names = orch.extract(df, ref_seqs=ref_seqs, use_cache=True)

print("\nX shape:", X.shape, "| n_features:", len(names))
groups = orch.get_feature_group_indices()
print("Feature groups:")
for name in orch.get_extractor_names():
    s, e = groups[name]
    print(f"  {name:20s} cols [{s}:{e}) = {e - s} features")
print("Total expected:", sum(e - s for s, e in groups.values()))
