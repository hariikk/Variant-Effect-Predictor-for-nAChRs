"""Training, cross-validation, and evaluation modules."""
from vep_nachr.training.cross_validation import (
    nested_cross_validation,
    simple_cross_validation,
    ExperimentResult,
    SeedResult,
    FoldResult,
)
from vep_nachr.training.evaluation import (
    compute_metrics,
    compute_per_class_metrics,
    aggregate_results,
    compare_approaches,
    paired_comparison_test,
    FeatureImportanceAnalyzer,
    FeatureImportanceResult,
    is_tree_model,
    plot_feature_importance,
    plot_shap_summary,
    plot_species_transfer_comparison,
    plot_comparison_heatmap,
)
