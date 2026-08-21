"""
Experiment runner: CLI and high-level orchestrators.

Usage:
    python -m vep_nachr2.training.runner single --model random_forest
    python -m vep_nachr2.training.runner compare --models rf lgbm xgb
    python -m vep_nachr2.training.runner ablation --model random_forest
    python -m vep_nachr2.training.runner species-transfer --model random_forest
"""

import argparse
import concurrent.futures
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from vep_nachr2.config import (
    ALL_MODELS, CORE_MODELS, MODEL_DISPLAY_NAMES,
    FEATURE_GROUPS, RESULTS_DIR, DEFAULT_SEEDS, PROJECT_ROOT,
    LABEL_NAMES, Config, CVConfig, HyperoptConfig, get_config,
)
from vep_nachr2.data.loader import load_mutation_data
from vep_nachr2.data.reference import load_all_reference_sequences
from vep_nachr2.features.orchestrator import FeatureOrchestrator, clear_feature_cache
from vep_nachr2.training.cross_validation import (
    nested_cross_validation,
    homology_class_transfer_cv,
    species_transfer_cv,
    save_results,
)
from vep_nachr2.training.evaluation import format_classification_report


# =============================================================================
# HIGH-LEVEL ORCHESTRATORS
# =============================================================================

def run_single(
    model_name: str = "random_forest",
    cv_mode: str = "subunit",
    n_outer_folds: int = 5,
    n_trials: int = 50,
    seeds: Optional[list[int]] = None,
    drop_extractors: Optional[list[str]] = None,
    verbose: bool = True,
) -> dict:
    """
    Run a single model through nested CV.

    Parameters
    ----------
    model_name : str
        Model identifier.
    cv_mode : str
        "subunit" (leave-one-gene-out) or "standard".
    n_outer_folds : int
        Number of outer CV folds.
    n_trials : int
        Optuna trials per fold.
    seeds : list[int], optional
        Random seeds.
    drop_extractors : list[str], optional
        Feature extractors to skip (for quick ablation tests).
    verbose : bool

    Returns
    -------
    dict with CV results.
    """
    # Load data
    if verbose:
        print("Loading data...")
    df = load_mutation_data()
    y = df["effect"].map({"LOF": 0, "No net effect": 1, "GOF": 2}).values.astype(np.int64)

    # Extract features
    if verbose:
        print("Extracting features...")
    ref_seqs = load_all_reference_sequences("human")
    orchestrator = FeatureOrchestrator(verbose=verbose)
    X, feature_names = orchestrator.extract(
        df, ref_seqs=ref_seqs,
        drop_extractors=drop_extractors,
        use_cache=True,
    )

    # Run CV
    results = nested_cross_validation(
        X=X, y=y, df=df,
        model_name=model_name,
        n_outer_folds=n_outer_folds,
        n_trials=n_trials,
        seeds=seeds or DEFAULT_SEEDS,
        cv_mode=cv_mode,
        verbose=verbose,
    )

    # Save
    output_dir = RESULTS_DIR / "baseline"
    json_path, csv_path = save_results(results, output_dir, f"{model_name}_{cv_mode}")

    if verbose:
        print(f"Results saved to: {json_path}")

    return results


def run_compare(
    models: Optional[list[str]] = None,
    cv_mode: str = "subunit",
    n_trials: int = 50,
    verbose: bool = True,
    binary: bool = False,
    data_file: Optional[str] = None,
    remap_nonhuman: bool = True,
    output_suffix: str = "",
) -> dict[str, dict]:
    """
    Run multiple models and compare results.

    Parameters
    ----------
    models : list[str], optional
        Models to compare. Default: CORE_MODELS.
    cv_mode : str
    n_trials : int
        Optuna trials per fold. Lower = faster comparison.
    verbose : bool

    Returns
    -------
    dict mapping model_name → results.
    """
    if models is None:
        models = ALL_MODELS  # All 10 models for full comparison

    # Load data once
    if verbose:
        print("Loading data...")
    if binary:
        df = load_mutation_data(effects=["GOF", "LOF"], data_file=data_file, remap_nonhuman=remap_nonhuman)
        y = df["effect"].map({"LOF": 0, "GOF": 1}).values.astype(np.int64)
        n_classes = 2
        label_names = {0: "LOF", 1: "GOF"}
    else:
        df = load_mutation_data(data_file=data_file, remap_nonhuman=remap_nonhuman)
        y = df["effect"].map({"LOF": 0, "No net effect": 1, "GOF": 2}).values.astype(np.int64)
        n_classes = 3
        label_names = LABEL_NAMES

    if verbose:
        print("Extracting features...")
    ref_seqs = load_all_reference_sequences("human")
    orchestrator = FeatureOrchestrator(verbose=False)
    X, feature_names = orchestrator.extract(df, ref_seqs=ref_seqs, use_cache=True)

    all_results = {}
    comparison_rows = []

    for i, model_name in enumerate(models):
        if verbose:
            print(f"\n{'#'*60}")
            print(f"Model {i + 1}/{len(models)}: {MODEL_DISPLAY_NAMES.get(model_name, model_name)}")
            print(f"{'#'*60}")

        results = nested_cross_validation(
            X=X, y=y, df=df,
            model_name=model_name,
            n_outer_folds=5,
            n_trials=n_trials,
            seeds=DEFAULT_SEEDS[:3],  # Fewer seeds for speed
            cv_mode=cv_mode,
            verbose=verbose,
            n_classes=n_classes,
            label_names=label_names,
        )

        all_results[model_name] = results
        comparison_rows.append({
            "model": model_name,
            **results.get("final", {}),
        })

    # Save comparison table
    comparison_df = pd.DataFrame(comparison_rows)
    output_dir = RESULTS_DIR / "comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_binary" if binary else ""
    comparison_df.to_csv(output_dir / f"comparison_{cv_mode}{suffix}{output_suffix}.csv", index=False)

    if verbose:
        print(f"\n{'='*60}")
        print("Model Comparison")
        print(f"{'='*60}")
        print(comparison_df.to_string())

    return all_results


def run_ablation(
    model_name: str = "random_forest",
    cv_mode: str = "subunit",
    n_trials: int = 30,
    verbose: bool = True,
) -> dict:
    """
    Feature-group leave-one-out ablation study for a single model.

    Runs CV with each feature group dropped, plus the full model.
    """
    # Load data once
    if verbose:
        print("Loading data...")
    df = load_mutation_data()
    y = df["effect"].map({"LOF": 0, "No net effect": 1, "GOF": 2}).values.astype(np.int64)

    ref_seqs = load_all_reference_sequences("human")

    # Get extractor names
    orchestrator = FeatureOrchestrator(verbose=False)
    extractor_names = orchestrator.get_extractor_names()

    # Conditions: full + drop each group
    conditions = {"full": None}
    for name in extractor_names:
        if orchestrator.extractors[extractor_names.index(name)].n_features > 0:
            conditions[f"drop_{name}"] = [name]

    results = {}

    for cond_name, drop_list in conditions.items():
        if verbose:
            print(f"\n{'='*50}")
            print(f"Ablation [{model_name}]: {cond_name}")
            print(f"{'='*50}")

        X, _ = orchestrator.extract(
            df, ref_seqs=ref_seqs,
            drop_extractors=drop_list,
            use_cache=False,  # Different feature sets, skip cache
        )

        cv_results = nested_cross_validation(
            X=X, y=y, df=df,
            model_name=model_name,
            n_outer_folds=5,
            n_trials=n_trials,
            seeds=DEFAULT_SEEDS[:1],  # 1 seed for ablation (feature ranking doesn't need multi-seed)
            cv_mode=cv_mode,
            verbose=verbose,
        )

        results[cond_name] = {
            "n_features": X.shape[1],
            **cv_results.get("final", {}),
        }

    # Save
    output_dir = RESULTS_DIR / "ablation"
    output_dir.mkdir(parents=True, exist_ok=True)
    ablation_df = pd.DataFrame(results).T
    ablation_df.to_csv(output_dir / f"ablation_{model_name}.csv")

    if verbose:
        print(f"\nAblation Results ({model_name}):")
        print(ablation_df.to_string())

    return results


def run_ablation_all(
    models: Optional[list[str]] = None,
    cv_mode: str = "subunit",
    n_trials: int = 30,
    max_workers: int = 4,
    verbose: bool = True,
) -> dict[str, dict]:
    """
    Feature-group ablation study for ALL models — PARALLEL.

    Launches each model's ablation as a separate subprocess so independent
    models run concurrently. A 9-model ablation that takes 3+ hours
    sequentially finishes in ~1 hour with 4 workers.

    Parameters
    ----------
    models : list[str], optional
        Models to ablate. Default: ALL_MODELS.
    cv_mode : str
    n_trials : int
        Optuna trials per inner fold.
    max_workers : int
        Max concurrent models (default 4 — balances CPU vs contention).
    verbose : bool

    Returns
    -------
    dict mapping model_name → ablation results dict.
    """
    if models is None:
        models = ALL_MODELS

    output_dir = RESULTS_DIR / "ablation"
    output_dir.mkdir(parents=True, exist_ok=True)

    n_models = len(models)
    if verbose:
        print(f"\n{'#'*70}")
        print(f"PARALLEL Ablation: {n_models} models x up to 7 conds, {max_workers} workers")
        print(f"Models: {[MODEL_DISPLAY_NAMES.get(m, m) for m in models]}")
        print(f"{'#'*70}\n")

    # Worker function: runs a single model's ablation via subprocess
    def _run_one_model(model_name: str) -> tuple[str, Optional[dict], float, str]:
        t0 = time.time()
        display = MODEL_DISPLAY_NAMES.get(model_name, model_name)
        cmd = [
            sys.executable, "-u", "-m", "vep_nachr2.training.runner",
            "ablation",
            "--model", model_name,
            "--n-trials", str(n_trials),
        ]
        try:
            result = subprocess.run(
                cmd,
                capture_output=True, text=True,
                timeout=1800,  # 30min per model (1 seed, fewer trials)
                cwd=str(PROJECT_ROOT),
            )
            elapsed = time.time() - t0
            stdout = result.stdout
            stderr = result.stderr

            # Try to load per-model CSV for structured results
            csv_path = output_dir / f"ablation_{model_name}.csv"
            if csv_path.exists():
                try:
                    df = pd.read_csv(csv_path, index_col=0)
                    results_dict = df.to_dict(orient="index")
                except Exception:
                    results_dict = None
            else:
                results_dict = None

            if result.returncode != 0:
                return (model_name, results_dict, elapsed,
                        f"FAILED (rc={result.returncode}): {stderr[-500:] if stderr else 'no stderr'}")

            return (model_name, results_dict, elapsed, "OK")

        except subprocess.TimeoutExpired:
            elapsed = time.time() - t0
            return (model_name, None, elapsed, "TIMEOUT (>2h)")
        except Exception as e:
            elapsed = time.time() - t0
            return (model_name, None, elapsed, f"ERROR: {e}")

    # Launch all models in parallel
    all_ablations = {}
    t_total_start = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(_run_one_model, m): m
            for m in models
        }

        completed = 0
        for future in concurrent.futures.as_completed(future_map):
            model_name = future_map[future]
            model_name, results_dict, elapsed, status = future.result()
            completed += 1

            if verbose:
                icon = "[OK]" if status == "OK" else "[FAIL]"
                f1_val = "?"
                if results_dict and "full" in results_dict:
                    f1_val = f"{results_dict['full'].get('macro_f1_mean', '?'):.3f}"
                print(f"[{completed}/{n_models}] {icon} {MODEL_DISPLAY_NAMES.get(model_name, model_name):25s} "
                      f"F1={f1_val}  {elapsed:.0f}s  ({status})")

            if results_dict:
                all_ablations[model_name] = results_dict

    t_total = time.time() - t_total_start

    # Build combined summary from individual CSVs
    summary_rows = []
    for model_name in models:
        csv_path = output_dir / f"ablation_{model_name}.csv"
        if not csv_path.exists():
            continue
        try:
            df = pd.read_csv(csv_path, index_col=0)
            full_f1 = df.loc["full", "macro_f1_mean"] if "full" in df.index else 0
            for cond in df.index:
                if cond == "full":
                    continue
                row = df.loc[cond]
                summary_rows.append({
                    "model": model_name,
                    "display_name": MODEL_DISPLAY_NAMES.get(model_name, model_name),
                    "condition": cond,
                    "n_features": row.get("n_features", 0),
                    "macro_f1_mean": row.get("macro_f1_mean"),
                    "delta_f1": row.get("macro_f1_mean", 0) - full_f1,
                    "mcc_mean": row.get("mcc_mean"),
                })
        except Exception:
            continue

    # Save combined outputs
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(output_dir / "ablation_ALL_summary.csv", index=False)

        pivot = summary_df.pivot_table(
            index="condition", columns="display_name",
            values="delta_f1", aggfunc="first"
        )
        pivot.to_csv(output_dir / "ablation_ALL_delta_pivot.csv")

        if verbose:
            print(f"\n{'='*70}")
            print(f"Total wall-clock time: {t_total:.0f}s ({t_total/60:.1f}m)")
            print(f"Ablation delta-F1 vs Full (negative = feature helps):")
            print(f"{'='*70}")
            print(pivot.to_string())

    return all_ablations


def run_species_transfer_experiment(
    model_name: str = "random_forest",
    n_trials: int = 30,
    verbose: bool = True,
) -> dict:
    """Run species transfer experiment."""
    if verbose:
        print("Loading data...")
    df = load_mutation_data()
    y = df["effect"].map({"LOF": 0, "No net effect": 1, "GOF": 2}).values.astype(np.int64)

    ref_seqs = load_all_reference_sequences("human")
    orchestrator = FeatureOrchestrator(verbose=False)
    X, _ = orchestrator.extract(df, ref_seqs=ref_seqs, use_cache=True)

    results = species_transfer_cv(
        X=X, y=y, df=df,
        model_name=model_name,
        n_trials=n_trials,
        seeds=DEFAULT_SEEDS,
        verbose=verbose,
    )

    # Save
    output_dir = RESULTS_DIR / "species_transfer"
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).T.to_csv(output_dir / f"species_transfer_{model_name}.csv")

    return results


# =============================================================================
# CLI
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    """Build argument parser for CLI."""
    parser = argparse.ArgumentParser(
        description="VEP-nAChR2: Variant Effect Predictor for nAChR receptors"
    )
    parser.add_argument("--clean", action="store_true", default=False,
                        help="Clear feature cache before running experiment")
    subparsers = parser.add_subparsers(dest="command", help="Experiment type")

    # clean — standalone cache clearing (no experiment run)
    subparsers.add_parser("clean", help="Clear feature cache only")

    # single
    single_parser = subparsers.add_parser("single", help="Run a single model")
    single_parser.add_argument("--model", type=str, default="random_forest",
                               choices=ALL_MODELS, help="Model name")
    single_parser.add_argument("--cv-mode", type=str, default="subunit",
                               choices=["subunit", "standard", "holdout"], help="CV mode")
    single_parser.add_argument("--n-folds", type=int, default=5, help="Outer CV folds")
    single_parser.add_argument("--n-trials", type=int, default=50, help="Optuna trials")
    single_parser.add_argument("--drop", nargs="*", default=None,
                               help="Feature extractors to drop")

    # compare
    compare_parser = subparsers.add_parser("compare", help="Compare multiple models")
    compare_parser.add_argument("--models", nargs="+", default=CORE_MODELS,
                                choices=ALL_MODELS, help="Models to compare")
    compare_parser.add_argument("--cv-mode", type=str, default="subunit")
    compare_parser.add_argument("--n-trials", type=int, default=30)
    compare_parser.add_argument("--binary", action="store_true", default=False,
                                help="Binary GOF/LOF prediction (drop 'No net effect')")
    compare_parser.add_argument("--data-file", type=str, default=None,
                                help="Override data file (e.g. final_mapped.xlsx) in data/raw/")
    compare_parser.add_argument("--no-remap", action="store_true", default=False,
                                help="Disable ortholog remapping of mouse/rat positions")
    compare_parser.add_argument("--output-suffix", type=str, default="",
                                help="Suffix appended to the comparison CSV filename")

    # ablation
    ablation_parser = subparsers.add_parser("ablation", help="Feature ablation study")
    ablation_parser.add_argument("--model", type=str, default=None,
                                 choices=ALL_MODELS,
                                 help="Single model (default: all models if --all, else random_forest)")
    ablation_parser.add_argument("--all", action="store_true", default=False,
                                 help="Run ablation for ALL models")
    ablation_parser.add_argument("--models", nargs="+", default=None,
                                 choices=ALL_MODELS, help="Specific models for ablation")
    ablation_parser.add_argument("--cv-mode", type=str, default="subunit")
    ablation_parser.add_argument("--n-trials", type=int, default=30)
    ablation_parser.add_argument("--workers", type=int, default=4,
                                 help="Max parallel models (for --all / --models)")

    # species-transfer
    st_parser = subparsers.add_parser("species-transfer", help="Species transfer experiment")
    st_parser.add_argument("--model", type=str, default="random_forest",
                           choices=ALL_MODELS)
    st_parser.add_argument("--n-trials", type=int, default=30)

    return parser


def main():
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args()

    # Handle standalone clean command (no experiment run)
    if args.command == "clean":
        clear_feature_cache(verbose=True)
        return

    if args.command is None:
        parser.print_help()
        return

    # Clear cache if --clean flag is set (for any experiment type)
    if args.clean:
        print("\n" + "=" * 50)
        print("Clearing feature cache before run...")
        print("=" * 50)
        clear_feature_cache(verbose=True)
        print()

    t_start = time.time()

    if args.command == "single":
        run_single(
            model_name=args.model,
            cv_mode=args.cv_mode,
            n_outer_folds=args.n_folds,
            n_trials=args.n_trials,
            drop_extractors=args.drop,
        )

    elif args.command == "compare":
        run_compare(
            models=args.models,
            cv_mode=args.cv_mode,
            n_trials=args.n_trials,
            binary=args.binary,
            data_file=args.data_file,
            remap_nonhuman=not args.no_remap,
            output_suffix=args.output_suffix,
        )

    elif args.command == "ablation":
        if args.all or args.models:
            models = args.models if args.models else ALL_MODELS
            run_ablation_all(
                models=models,
                cv_mode=args.cv_mode,
                n_trials=args.n_trials,
                max_workers=args.workers,
            )
        else:
            model = args.model if args.model else "random_forest"
            run_ablation(
                model_name=model,
                cv_mode=args.cv_mode,
                n_trials=args.n_trials,
            )

    elif args.command == "species-transfer":
        run_species_transfer_experiment(
            model_name=args.model,
            n_trials=args.n_trials,
        )

    elapsed = time.time() - t_start
    print(f"\nTotal time: {elapsed:.1f}s ({elapsed / 60:.1f}m)")


if __name__ == "__main__":
    main()
