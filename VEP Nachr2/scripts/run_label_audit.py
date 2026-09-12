"""
Run XXIII: label-quality audit.

The 3-class metric inherits every label error in `final_mapped.xlsx`. This script
does a pandas-only audit of label provenance and flags the rows most likely to be
mislabeled, so the thesis can quantify (rather than assume) label noise. It does
NOT train a model (that is the `--sensitivity` mode, run separately).

Flags (one `flag_reason` per row, `;`-separated):
  nne_single_source       No-net-effect label backed by a single paper (weakest
                          evidence class — 186/194 NNE are single-source).
  nne_am_pathogenic       NNE that AlphaMissense scores > 0.564 (likely pathogenic):
                          a "no effect" label that a strong pathogenicity model
                          contradicts → candidate mislabel.
  ambiguous_effect        LOF/GOF rows (dropped by the loader).
  low_conf_mapping        position mapping_status in {ambiguous, no_match,
                          no_reference, typo*} — the residue/position may be wrong.
  wt_mismatch             mapping_status ends in "wt_mismatch" — recorded WT does
                          not match the reference sequence (data-entry error).
  manual_review           Correct? is CHECK or blank.
  multi_row_merged        n_rows_merged > 1 (potential hidden effect conflict).

Usage:
    python scripts/run_label_audit.py               # pandas-only audit (fast)
    python scripts/run_label_audit.py --sensitivity # CV relabeling check (slow)
"""

import argparse
import sys
import warnings

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore", message=".*ill-defined.*")
warnings.filterwarnings("ignore", message=".*Only one class is present.*")

from vep_nachr2.config import RAW_DATA_DIR, RESULTS_DIR, DEFAULT_SEEDS, LABEL_NAMES

AM_PATHOGENIC = 0.564   # AlphaMissense "likely pathogenic" threshold
AM_BENIGN = 0.34        # AlphaMissense "likely benign" threshold
LOW_CONF_MAPPING = {"ambiguous", "no_match", "no_reference"}


def _is_low_conf(status: str) -> bool:
    s = str(status)
    if s in LOW_CONF_MAPPING:
        return True
    if s.startswith("typo"):
        return True
    return False


def _is_wt_mismatch(status: str) -> bool:
    return str(status).endswith("wt_mismatch")


def audit(raw: pd.DataFrame, am_map: dict[str, float]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (summary_df, flags_df)."""
    # ── Effect distribution ──
    effect_counts = raw["Effect"].value_counts(dropna=False)
    effect_stats = []
    for effect, n in effect_counts.items():
        sub = raw[raw["Effect"] == effect]
        n_single = int((sub["n_papers"] <= 1).sum())
        n_multi = int((sub["n_papers"] > 1).sum())
        n_low_conf = int(sub["mapping_status"].apply(_is_low_conf).sum())
        effect_stats.append({
            "effect": effect,
            "count": int(n),
            "n_single_source": n_single,
            "n_multi_source": n_multi,
            "single_source_pct": round(100 * n_single / max(n, 1), 1),
            "n_low_conf_mapping": n_low_conf,
            "n_wt_mismatch": int(sub["mapping_status"].apply(_is_wt_mismatch).sum()),
            "mean_n_papers": round(float(sub["n_papers"].mean()), 2),
        })
    summary = pd.DataFrame(effect_stats)

    # ── Build a join key + attach AlphaMissense ──
    # Position is normalized to int so float "100.0" matches the loader's int "100".
    raw["_key"] = (
        raw["Species"].astype(str).str.strip().str.lower() + "|"
        + raw["nAChR subunit"].astype(str).str.strip().str.upper() + "|"
        + pd.to_numeric(raw["AA position"], errors="coerce").fillna(-1).astype(int).astype(str) + "|"
        + raw["Initial AA"].astype(str).str.strip().str.upper() + "|"
        + raw["New AA"].astype(str).str.strip().str.upper()
    )
    raw["am_path"] = raw["_key"].map(am_map)

    # ── Flag rows ──
    reasons = [[] for _ in range(len(raw))]
    for i, (_, row) in enumerate(raw.iterrows()):
        eff = str(row["Effect"])
        if eff == "No net effect" and (pd.isna(row["n_papers"]) or row["n_papers"] <= 1):
            reasons[i].append("nne_single_source")
        if eff == "No net effect" and pd.notna(row["am_path"]) and row["am_path"] > AM_PATHOGENIC:
            reasons[i].append("nne_am_pathogenic")
        if eff in {"LOF/GOF", "GOF/LOF"}:
            reasons[i].append("ambiguous_effect")
        if _is_low_conf(row["mapping_status"]):
            reasons[i].append("low_conf_mapping")
        if _is_wt_mismatch(row["mapping_status"]):
            reasons[i].append("wt_mismatch")
        if str(row["Correct?"]).strip().upper() in {"CHECK", "NAN", ""}:
            reasons[i].append("manual_review")
        if pd.notna(row["n_rows_merged"]) and row["n_rows_merged"] > 1:
            reasons[i].append("multi_row_merged")

    flags = raw[[
        "Species", "nAChR subunit", "AA position", "Initial AA", "New AA", "Effect",
        "Reference(PMID)", "DOI", "n_papers", "n_rows_merged", "mapping_status",
        "Measuring Technique", "Pathology", "Correct?", "am_path",
    ]].copy()
    flags["flag_reason"] = [";".join(r) for r in reasons]
    flags = flags[flags["flag_reason"].astype(bool)].reset_index(drop=True)

    return summary, flags


def sensitivity_check():
    """Relabeling sensitivity: does dropping lowest-confidence NNE rows move F1?"""
    from vep_nachr2.data.loader import load_mutation_data
    from vep_nachr2.data.reference import load_all_reference_sequences
    from vep_nachr2.features.orchestrator import FeatureOrchestrator
    from vep_nachr2.training.cross_validation import nested_cross_validation

    df = load_mutation_data(data_file="final_mapped.xlsx")
    y = df["effect"].map({"LOF": 0, "No net effect": 1, "GOF": 2}).values.astype(np.int64)

    ref_seqs = load_all_reference_sequences("human")
    orch = FeatureOrchestrator(verbose=False)
    X, _ = orch.extract(df, ref_seqs=ref_seqs, use_cache=True)

    # Lowest-confidence NNE rows: "no effect" contradicted by AlphaMissense
    # (pathogenicity > 0.564). These are the labels a strong pathogenicity model
    # calls likely-damaging — the most probable mislabels. Dropping them keeps
    # ~110 NNE (vs single-source, which would collapse the class to 2 rows).
    am_map = _load_am_map()
    key = (df["species"] + "|" + df["subunit"] + "|" + df["position"].astype(str)
           + "|" + df["wildtype_aa"] + "|" + df["variant_aa"])
    am_path = key.map(am_map)
    drop_mask = ((df["effect"] == "No net effect") & (am_path > AM_PATHOGENIC)).values
    keep = ~drop_mask
    print(f"Relabeling sensitivity: dropping {int(drop_mask.sum())} "
          f"AlphaMissense-pathogenic NNE rows ({100 * drop_mask.mean():.1f}% of data)")

    def run(X_sub, y_sub, df_sub):
        cv = nested_cross_validation(
            X=X_sub, y=y_sub, df=df_sub,
            model_name="random_forest", n_outer_folds=5, n_trials=15,
            seeds=DEFAULT_SEEDS[:1], cv_mode="subunit", verbose=False,
            n_classes=3, label_names=LABEL_NAMES,
        )
        return cv.get("final", {})

    full = run(X, y, df)
    sub = run(X[keep], y[keep], df[keep].reset_index(drop=True))
    print(f"full F1 = {full.get('macro_f1_mean', float('nan')):.4f} | "
          f"drop AM-pathogenic NNE F1 = {sub.get('macro_f1_mean', float('nan')):.4f}")
    metric_keys = ["macro_f1_mean", "macro_f1_std", "mcc_mean", "balanced_accuracy_mean"]
    out = pd.DataFrame([
        {"condition": "full", "n": len(y),
         **{k: full.get(k, np.nan) for k in metric_keys}},
        {"condition": "drop_am_pathogenic_nne", "n": int(keep.sum()),
         **{k: sub.get(k, np.nan) for k in metric_keys}},
    ])
    out_dir = RESULTS_DIR / "label_audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_dir / "relabel_sensitivity.csv", index=False)
    print(f"Saved {out_dir / 'relabel_sensitivity.csv'}")


def _load_am_map() -> dict[str, float]:
    """Build {species|subunit|pos|wt|mt: am_path} from the AlphaMissense extractor."""
    from vep_nachr2.data.loader import load_mutation_data
    from vep_nachr2.data.reference import load_all_reference_sequences
    from vep_nachr2.features.alphamissense import AlphamissenseExtractor

    df = load_mutation_data(data_file="final_mapped.xlsx")
    ref_seqs = load_all_reference_sequences("human")
    am = AlphamissenseExtractor()
    scores = am.extract(df, ref_seqs=ref_seqs).ravel()
    key = (
        df["species"] + "|" + df["subunit"] + "|" + df["position"].astype(str)
        + "|" + df["wildtype_aa"] + "|" + df["variant_aa"]
    )
    return dict(zip(key, scores))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sensitivity", action="store_true",
                        help="Run the CV relabeling-sensitivity check (slow).")
    args = parser.parse_args()

    if args.sensitivity:
        sensitivity_check()
        return

    out_dir = RESULTS_DIR / "label_audit"
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_excel(RAW_DATA_DIR / "final_mapped.xlsx", sheet_name="unique_variants")
    am_map = _load_am_map()
    summary, flags = audit(raw, am_map)

    summary.to_csv(out_dir / "label_audit_summary.csv", index=False)
    flags.to_csv(out_dir / "label_audit_flags.csv", index=False)

    print("Effect × provenance summary:")
    print(summary.to_string(index=False))
    print(f"\nFlagged rows: {len(flags)} / {len(raw)}")
    print("Flags by reason:")
    exploded = flags["flag_reason"].str.split(";").explode()
    print(exploded.value_counts().to_string())
    print(f"\nNNE flagged as AlphaMissense-pathogenic: "
          f"{flags['flag_reason'].str.contains('nne_am_pathogenic').sum()}")
    print(f"Saved {out_dir / 'label_audit_summary.csv'}")
    print(f"Saved {out_dir / 'label_audit_flags.csv'}")


if __name__ == "__main__":
    main()
