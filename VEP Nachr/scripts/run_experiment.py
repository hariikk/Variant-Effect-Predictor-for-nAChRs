"""
Main experiment runner for nAChR VEP.

Usage:
    # Quick test with default engineered features
    python scripts/run_experiment.py --quick

    # Full nested CV on a single model
    python scripts/run_experiment.py --model random_forest

    # Compare all encodings across all core models
    python scripts/run_experiment.py --all-encodings

    # Run with a specific encoding (ordinal, onehot, fullseq, engineered)
    python scripts/run_experiment.py --encoding onehot

    # Multi-species (when mouse/rat data is available)
    python scripts/run_experiment.py --species human mouse

This script:
1. Loads and cleans the nAChR mutation database
2. Extracts features for the requested encoding(s)
3. Runs nested cross-validation for each model × encoding combination
4. Saves results to results/ directory
5. Prints comparison summary
"""

import argparse
import hashlib
import json
import pickle
import sys
import warnings
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np

from vep_nachr.config import (
    CORE_MODELS,
    ALL_MODELS,
    RESULTS_DIR,
    ENCODING_STRATEGIES,
    DATA_DRIVEN_ENCODINGS,
    DOMAIN_DRIVEN_ENCODINGS,
    FEATURE_CACHE_DIR,
)
from vep_nachr.data.loader import load_dataset, load_multi_species_dataset

from vep_nachr.training.cross_validation import (
    nested_cross_validation,
    simple_cross_validation,
)
from vep_nachr.models.registry import AVAILABLE_MODELS

# Suppress convergence warnings during grid search
warnings.filterwarnings("ignore", message=".*X does not have valid feature names.*")


# =============================================================================
# ENCODING HELPERS
# =============================================================================

def encode_features(df, encoding: str, sequences=None) -> tuple[np.ndarray, list[str]]:
    """
    Encode a DataFrame using the specified encoding strategy.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned mutation data
    encoding : str
        Encoding strategy name
    sequences : dict, optional
        Wildtype sequences (required for fullseq encoding)

    Returns
    -------
    tuple[np.ndarray, list[str]]
        Feature matrix and feature names
    """
    enc_lower = encoding.lower().replace("-", "").replace("_", "").replace(" ", "")

    if enc_lower in ("ordinal", "onehot", "fullseq", "fullsequence"):
        from vep_nachr.data.encoders import get_encoder
        encoder = get_encoder(encoding)
        if encoding in ("fullseq", "fullsequence"):
            X = encoder.fit_transform(df, sequences)
        else:
            X = encoder.fit_transform(df)
        feature_names = encoder.feature_names

    elif enc_lower == "engineered":
        from vep_nachr.features.encoder import NachrFeatureEncoder
        encoder = NachrFeatureEncoder()
        X = encoder.fit_transform(df)
        feature_names = encoder.get_feature_names_out()

    elif enc_lower in ("noahoriginal", "noah"):
        from vep_nachr.features.noah_original import NoahOriginalFeatureEncoder
        encoder = NoahOriginalFeatureEncoder()
        X = encoder.fit_transform(df)
        feature_names = encoder.get_feature_names_out()

    elif enc_lower == "combined":
        from vep_nachr.features.noah_original import CombinedFeatureEncoder
        encoder = CombinedFeatureEncoder()
        X = encoder.fit_transform(df)
        feature_names = encoder.get_feature_names_out()

    else:
        raise ValueError(f"Unknown encoding: {encoding}")

    return X, feature_names


# =============================================================================
# FEATURE CACHING
# =============================================================================

def _cache_key(df, encoding: str) -> str:
    """Generate a deterministic cache key from DataFrame + encoding."""
    h = hashlib.sha256()
    h.update(encoding.encode())
    h.update(str(len(df)).encode())
    for _, row in df.iterrows():
        ident = f"{row.get('subunit','')}_{row.get('position','')}_{row.get('wildtype_aa','')}_{row.get('variant_aa','')}"
        h.update(ident.encode())
    return h.hexdigest()[:16]


def _cache_path(df, encoding: str) -> Path:
    """Get the cache file path."""
    key = _cache_key(df, encoding)
    return FEATURE_CACHE_DIR / f"{encoding}_{key}.npz"


def load_cached_features(df, encoding: str) -> tuple | None:
    """Try to load from cache. Returns None on miss."""
    cache_file = _cache_path(df, encoding)
    if not cache_file.exists():
        return None
    try:
        data = np.load(cache_file, allow_pickle=True)
        return data["X"], list(data["feature_names"])
    except Exception:
        return None


def save_cached_features(df, encoding: str, X, feature_names) -> None:
    """Save features to cache."""
    cache_file = _cache_path(df, encoding)
    np.savez_compressed(cache_file, X=X,
                         feature_names=np.array(feature_names, dtype=object))


def encode_features_cached(df, encoding: str, sequences=None) -> tuple:
    """Encode features with caching layer (cache-first)."""
    cached = load_cached_features(df, encoding)
    if cached is not None:
        return cached
    X, feature_names = encode_features_cached(df, encoding, sequences)
    save_cached_features(df, encoding, X, feature_names)
    return X, feature_names


# =============================================================================
# QUICK TEST
# =============================================================================

def run_quick_test(X, y, models=None, encoding="engineered"):
    """Run quick evaluation with default hyperparameters (no Optuna)."""
    if models is None:
        models = [m for m in CORE_MODELS if m in AVAILABLE_MODELS]

    print("\n" + "=" * 60)
    print(f"QUICK EVALUATION — {encoding} (default HPs, no Optuna)")
    print("=" * 60)

    results = {}
    for model_name in models:
        try:
            result = simple_cross_validation(
                X, y, model_name=model_name, n_folds=5, n_jobs=-1
            )
            results[f"{model_name}_{encoding}"] = result
            print(
                f"  {model_name:25s}: "
                f"F1={result['mean_f1']:.4f} +/- {result['std_f1']:.4f}  "
                f"Acc={result['mean_accuracy']:.4f} +/- {result['std_accuracy']:.4f}"
            )
        except Exception as e:
            print(f"  {model_name:25s}: FAILED — {e}")

    return results


# =============================================================================
# FULL EXPERIMENT
# =============================================================================

def run_full_experiment(
    X, y, models=None, encoding="engineered",
    feature_names=None, n_trials=50,
):
    """Run full nested CV with Optuna HP optimization."""
    if models is None:
        models = [m for m in CORE_MODELS if m in AVAILABLE_MODELS]

    print("\n" + "=" * 60)
    print(f"FULL NESTED CV — {encoding} (with Optuna HP optimization)")
    print("=" * 60)

    all_results = {}

    for model_name in models:
        try:
            result = nested_cross_validation(
                X, y,
                model_name=model_name,
                encoding=encoding,
                n_outer_folds=5,
                n_inner_folds=5,
                n_trials=n_trials,
                n_jobs=-1,
                verbose=1,
                feature_names=feature_names,
            )
            all_results[model_name] = result

            # Save with encoding in filename
            safe_enc = encoding.replace("-", "_")
            result.save(RESULTS_DIR / f"{model_name}_{safe_enc}.json")

        except Exception as e:
            print(f"\n  {model_name}: FAILED — {e}")
            import traceback
            traceback.print_exc()

    return all_results


# =============================================================================
# COMPARISON MODE
# =============================================================================

def run_comparison(
    df, labels, sequences,
    models=None, encodings=None, n_trials=50,
):
    """
    Run full grid search: all models × all encodings.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned mutation data
    labels : np.ndarray
        Binary labels
    sequences : dict
        Wildtype sequences
    models : list[str], optional
        Models to evaluate (default: all core)
    encodings : list[str], optional
        Encodings to evaluate (default: all available)
    n_trials : int
        Optuna trials per fold
    """
    if models is None:
        models = [m for m in CORE_MODELS if m in AVAILABLE_MODELS]
    if encodings is None:
        encodings = list(ENCODING_STRATEGIES.keys())
        # Only include encodings whose dependencies are available
        available = []
        for enc in encodings:
            if enc in ("noah_original", "combined"):
                try:
                    from vep_nachr.features.noah_original import NoahOriginalFeatureEncoder
                    available.append(enc)
                except ImportError:
                    print(f"  Skipping '{enc}' — Noah's features not available")
                    continue
            else:
                available.append(enc)
        encodings = available

    all_results = {}

    for encoding in encodings:
        print(f"\n{'=' * 60}")
        print(f"ENCODING: {encoding}")
        print(f"{'=' * 60}")

        try:
            print("  Extracting features...")
            X, feature_names = encode_features_cached(df, encoding, sequences)
            print(f"  Feature matrix: {X.shape}")

            enc_results = run_full_experiment(
                X, labels,
                models=models,
                encoding=encoding,
                feature_names=feature_names,
                n_trials=n_trials,
            )
            all_results.update(enc_results)

        except Exception as e:
            print(f"  Encoding '{encoding}': FAILED — {e}")
            import traceback
            traceback.print_exc()

    # Print grand summary
    _print_comparison_summary(all_results)

    return all_results


def _print_comparison_summary(all_results):
    """Print a models × encodings comparison table."""
    if not all_results:
        return

    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)

    # Organize by (model, encoding)
    rows = []
    for key, result in all_results.items():
        if hasattr(result, "overall_mean_f1"):
            rows.append({
                "model": result.model_name,
                "encoding": result.encoding,
                "f1": result.overall_mean_f1,
                "std": result.overall_std_f1,
                "acc": result.overall_mean_accuracy,
                "n_features": result.n_features,
            })

    if not rows:
        print("  No valid results to summarize.")
        return

    # Sort by F1
    rows.sort(key=lambda r: r["f1"], reverse=True)

    header = f"{'Model':25s} {'Encoding':20s} {'F1':>8s} {'Std':>8s} {'Acc':>8s} {'Feats':>6s}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['model']:25s} {r['encoding']:20s} "
            f"{r['f1']:8.4f} {r['std']:8.4f} {r['acc']:8.4f} {r['n_features']:6d}"
        )

    # Best overall
    best = rows[0]
    print(f"\nBest: {best['model']} × {best['encoding']} — F1={best['f1']:.4f}")


# =============================================================================
# MAIN CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="nAChR Variant Effect Predictor — Experiment Runner"
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Run a specific model (e.g., random_forest). Default: all core models.",
    )
    parser.add_argument(
        "--encoding", type=str, default="engineered",
        help="Encoding strategy: ordinal, onehot, fullseq, engineered, combined. "
             "Default: engineered.",
    )
    parser.add_argument(
        "--all-encodings", action="store_true",
        help="Run comparison across ALL encoding strategies (grid search).",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick test with default HPs (no Optuna optimization).",
    )
    parser.add_argument(
        "--all-models", action="store_true",
        help="Run all models (core + extended) instead of just core.",
    )
    parser.add_argument(
        "--n-trials", type=int, default=50,
        help="Number of Optuna trials per fold (default: 50).",
    )
    parser.add_argument(
        "--no-structural", action="store_true",
        help="Exclude structural features from engineered encoding.",
    )
    parser.add_argument(
        "--species", nargs="+", default=None,
        help="Species to include (e.g., 'human mouse'). Requires multi-species data files.",
    )
    parser.add_argument(
        "--data", type=str, default=None,
        help="Path to data file (overrides default).",
    )
    parser.add_argument(
        "--mouse-data", type=str, default=None,
        help="Path to mouse data file (enables multi-species mode).",
    )
    parser.add_argument(
        "--rat-data", type=str, default=None,
        help="Path to rat data file (enables multi-species mode).",
    )
    args = parser.parse_args()

    # --- Load data ---
    print("Loading data...")
    has_multi_species = args.mouse_data or args.rat_data or args.species

    # Auto-detect: if --data points to a file with a Species column, treat as
    # combined multi-species; otherwise load as single-species human-only.
    if has_multi_species:
        # User explicitly requested multi-species mode
        df, labels, sequences = load_multi_species_dataset(
            combined_path=args.data if args.data and not (args.mouse_data or args.rat_data) else None,
            human_path=args.data if (args.mouse_data or args.rat_data) else None,
            mouse_path=args.mouse_data,
            rat_path=args.rat_data,
            species_filter=args.species,
        )
    elif args.data:
        # Auto-detect: check if file has Species column
        import pandas as _pd
        data_path = Path(args.data)
        if data_path.suffix.lower() in (".csv",):
            peek = _pd.read_csv(data_path, nrows=0)
        else:
            peek = _pd.read_excel(data_path, nrows=0)
        has_species_col = any(c.lower() == "species" for c in peek.columns)

        if has_species_col:
            print(f"  Auto-detected Species column in {args.data} — loading as multi-species")
            df, labels, sequences = load_multi_species_dataset(combined_path=args.data)
        else:
            df, labels, sequences = load_dataset(data_path=args.data)
    else:
        # Default: load single-species human data
        df, labels, sequences = load_dataset()

    # --- Determine models ---
    if args.model:
        models = [args.model]
    elif args.all_models:
        models = [m for m in ALL_MODELS if m in AVAILABLE_MODELS]
    else:
        models = [m for m in CORE_MODELS if m in AVAILABLE_MODELS]

    # --- Run ---
    if args.all_encodings:
        # Full grid search
        encodings = list(ENCODING_STRATEGIES.keys())
        if args.quick:
            # Quick test across encodings
            for enc in encodings:
                try:
                    X, fnames = encode_features(df, enc, sequences)
                    run_quick_test(X, labels, models, encoding=enc)
                except Exception as e:
                    print(f"  Encoding '{enc}': SKIPPED — {e}")
        else:
            run_comparison(df, labels, sequences, models, encodings, args.n_trials)

    elif args.encoding != "engineered" or args.quick:
        # Single encoding run
        print(f"\nEncoding: {args.encoding}")
        X, feature_names = encode_features(df, args.encoding, sequences)
        print(f"Feature matrix shape: {X.shape}")

        if args.quick:
            run_quick_test(X, labels, models, encoding=args.encoding)
        else:
            run_full_experiment(
                X, labels, models=models,
                encoding=args.encoding,
                feature_names=feature_names,
                n_trials=args.n_trials,
            )

    else:
        # Default: engineered encoding, full CV
        print("\nExtracting engineered features...")
        from vep_nachr.features.encoder import NachrFeatureEncoder
        encoder = NachrFeatureEncoder(
            include_structural=not args.no_structural,
            include_substitution=True,
            include_positional=True,
        )
        X = encoder.fit_transform(df)
        feature_names = encoder.get_feature_names_out()
        print(f"Feature matrix shape: {X.shape}")
        print(f"Features: {feature_names}")

        if args.quick:
            run_quick_test(X, labels, models, encoding="engineered")
        else:
            run_full_experiment(
                X, labels, models=models,
                encoding="engineered",
                feature_names=feature_names,
                n_trials=args.n_trials,
            )

    print("\nDone! Results saved to:", RESULTS_DIR)


if __name__ == "__main__":
    main()
