# -*- coding: utf-8 -*-
"""Add per-seed macro-F1 capture to the 9 gene-aware run scripts that currently
discard it (they only save the across-seed mean/std). Per-seed values are needed
for paired t-tests on the deltas.

Each patched script gains:
  * `import json`
  * `_run_cv(...)` returning `(final, per_seed_list)` (or inline capture)
  * a `perseed` dict keyed by `condition|model` (or `group|identity|model`)
  * a `<name>_perseed.json` dump next to the existing summary CSV.
"""
import io

P = "scripts/"

# Per-script: (summary_basename, [ (old, new), ... ])
# old strings are matched exactly (with surrounding newlines where needed).
PATCHES = {
    "run_xxvii_allosteric.py": [
        ("import argparse\nimport sys\n", "import argparse\nimport json\nimport sys\n"),
        ("    return cv.get(\"final\", {})",
         "    return cv.get(\"final\", {}), [float(r[\"aggregate\"][\"macro_f1_mean\"]) for r in cv.get(\"per_seed\", [])]"),
        ("    rows = []\n", "    rows = []\n    perseed = {}\n"),
        ("            res = _run_cv(X, y, df, model, args.n_trials, args.n_seeds)",
         "            res, ps = _run_cv(X, y, df, model, args.n_trials, args.n_seeds)\n"
         "            perseed[f\"{name}|{model}\"] = ps"),
        ("    out.to_csv(out_dir / \"allosteric_interaction_summary.csv\", index=False)",
         "    out.to_csv(out_dir / \"allosteric_interaction_summary.csv\", index=False)\n"
         "    with open(out_dir / \"allosteric_interaction_perseed.json\", \"w\") as _f:\n"
         "        json.dump(perseed, _f, indent=2)"),
    ],
    "run_xxviii_structural_feature_ablation.py": [
        ("import argparse\nimport sys\n", "import argparse\nimport json\nimport sys\n"),
        ("    return cv.get(\"final\", {})",
         "    return cv.get(\"final\", {}), [float(r[\"aggregate\"][\"macro_f1_mean\"]) for r in cv.get(\"per_seed\", [])]"),
        ("    rows = []\n", "    rows = []\n    perseed = {}\n"),
        ("            res = _run_cv(X, y, df, model, args.n_trials, args.n_seeds)",
         "            res, ps = _run_cv(X, y, df, model, args.n_trials, args.n_seeds)\n"
         "            perseed[f\"{label}|{model}\"] = ps"),
        ("    out.to_csv(out_dir / \"structural_feature_ablation_summary.csv\", index=False)",
         "    out.to_csv(out_dir / \"structural_feature_ablation_summary.csv\", index=False)\n"
         "    with open(out_dir / \"structural_feature_ablation_perseed.json\", \"w\") as _f:\n"
         "        json.dump(perseed, _f, indent=2)"),
    ],
    "run_xxix_species_ablation.py": [
        ("import argparse\nimport sys\n", "import argparse\nimport json\nimport sys\n"),
        ("    return cv.get(\"final\", {})",
         "    return cv.get(\"final\", {}), [float(r[\"aggregate\"][\"macro_f1_mean\"]) for r in cv.get(\"per_seed\", [])]"),
        ("    rows = []\n", "    rows = []\n    perseed = {}\n"),
        ("            res = _run_cv(X, y_sub, df_sub, model, args.n_trials, args.n_seeds)",
         "            res, ps = _run_cv(X, y_sub, df_sub, model, args.n_trials, args.n_seeds)\n"
         "            perseed[f\"{name}|{model}\"] = ps"),
        ("    out.to_csv(out_dir / \"species_ablation_summary.csv\", index=False)",
         "    out.to_csv(out_dir / \"species_ablation_summary.csv\", index=False)\n"
         "    with open(out_dir / \"species_ablation_perseed.json\", \"w\") as _f:\n"
         "        json.dump(perseed, _f, indent=2)"),
    ],
    "run_xxx_data_size.py": [
        ("import argparse\nimport sys\n", "import argparse\nimport json\nimport sys\n"),
        ("    return cv.get(\"final\", {})",
         "    return cv.get(\"final\", {}), [float(r[\"aggregate\"][\"macro_f1_mean\"]) for r in cv.get(\"per_seed\", [])]"),
        ("    rows = []\n", "    rows = []\n    perseed = {}\n"),
        ("            res = _run_cv(X, y_sub, df_sub, model, args.n_trials, args.n_seeds)",
         "            res, ps = _run_cv(X, y_sub, df_sub, model, args.n_trials, args.n_seeds)\n"
         "            perseed[f\"{name}|{model}\"] = ps"),
        ("    out.to_csv(out_dir / \"data_size_summary.csv\", index=False)",
         "    out.to_csv(out_dir / \"data_size_summary.csv\", index=False)\n"
         "    with open(out_dir / \"data_size_perseed.json\", \"w\") as _f:\n"
         "        json.dump(perseed, _f, indent=2)"),
    ],
    "run_xxxi_subunit_type.py": [
        ("import argparse\nimport sys\n", "import argparse\nimport json\nimport sys\n"),
        ("    return cv.get(\"final\", {})",
         "    return cv.get(\"final\", {}), [float(r[\"aggregate\"][\"macro_f1_mean\"]) for r in cv.get(\"per_seed\", [])]"),
        ("    rows = []\n", "    rows = []\n    perseed = {}\n"),
        ("            res = _run_cv(X, y_sub, df_sub, model, args.n_trials, args.n_seeds)",
         "            res, ps = _run_cv(X, y_sub, df_sub, model, args.n_trials, args.n_seeds)\n"
         "            perseed[f\"{name}|{model}\"] = ps"),
        ("    out.to_csv(out_dir / \"subunit_type_summary.csv\", index=False)",
         "    out.to_csv(out_dir / \"subunit_type_summary.csv\", index=False)\n"
         "    with open(out_dir / \"subunit_type_perseed.json\", \"w\") as _f:\n"
         "        json.dump(perseed, _f, indent=2)"),
    ],
    "run_xxxii_gene_identity.py": [
        ("import argparse\nimport sys\n", "import argparse\nimport json\nimport sys\n"),
        ("    return cv.get(\"final\", {})",
         "    return cv.get(\"final\", {}), [float(r[\"aggregate\"][\"macro_f1_mean\"]) for r in cv.get(\"per_seed\", [])]"),
        ("    rows = []\n", "    rows = []\n    perseed = {}\n"),
        ("                res = _run_cv(X_sub_full, y_sub, df_sub, model, args.n_trials, args.n_seeds)",
         "                res, ps = _run_cv(X_sub_full, y_sub, df_sub, model, args.n_trials, args.n_seeds)\n"
         "                perseed[f\"{name}|{label}|{model}\"] = ps"),
        ("    out.to_csv(out_dir / \"gene_identity_drop_summary.csv\", index=False)",
         "    out.to_csv(out_dir / \"gene_identity_drop_summary.csv\", index=False)\n"
         "    with open(out_dir / \"gene_identity_drop_perseed.json\", \"w\") as _f:\n"
         "        json.dump(perseed, _f, indent=2)"),
    ],
    "run_xxxv_functional.py": [
        ("import argparse\nimport sys\n", "import argparse\nimport json\nimport sys\n"),
        ("    rows = []\n", "    rows = []\n    perseed = {}\n"),
        ("            res = nested_cross_validation(\n"
         "                X=X, y=y, df=df, model_name=model,\n"
         "                n_outer_folds=5, n_trials=args.n_trials,\n"
         "                seeds=DEFAULT_SEEDS[:args.n_seeds],\n"
         "                cv_mode=\"standard\", verbose=False, n_classes=3, label_names=LABEL_NAMES,\n"
         "            ).get(\"final\", {})",
         "            cv = nested_cross_validation(\n"
         "                X=X, y=y, df=df, model_name=model,\n"
         "                n_outer_folds=5, n_trials=args.n_trials,\n"
         "                seeds=DEFAULT_SEEDS[:args.n_seeds],\n"
         "                cv_mode=\"standard\", verbose=False, n_classes=3, label_names=LABEL_NAMES,\n"
         "            )\n"
         "            res = cv.get(\"final\", {})\n"
         "            perseed[f\"{name}|{model}\"] = [float(r[\"aggregate\"][\"macro_f1_mean\"]) for r in cv.get(\"per_seed\", [])]"),
        ("    out.to_csv(out_dir / \"functional_summary.csv\", index=False)",
         "    out.to_csv(out_dir / \"functional_summary.csv\", index=False)\n"
         "    with open(out_dir / \"functional_perseed.json\", \"w\") as _f:\n"
         "        json.dump(perseed, _f, indent=2)"),
    ],
    "run_xxxvi_structure_gating.py": [
        ("import argparse\nimport sys\n", "import argparse\nimport json\nimport sys\n"),
        ("    rows = []\n", "    rows = []\n    perseed = {}\n"),
        ("            res = nested_cross_validation(\n"
         "                X=X, y=y_sub, df=df_sub, model_name=model,\n"
         "                n_outer_folds=5, n_trials=args.n_trials,\n"
         "                seeds=DEFAULT_SEEDS[:args.n_seeds],\n"
         "                cv_mode=\"standard\", verbose=False, n_classes=3, label_names=LABEL_NAMES,\n"
         "            ).get(\"final\", {})",
         "            cv = nested_cross_validation(\n"
         "                X=X, y=y_sub, df=df_sub, model_name=model,\n"
         "                n_outer_folds=5, n_trials=args.n_trials,\n"
         "                seeds=DEFAULT_SEEDS[:args.n_seeds],\n"
         "                cv_mode=\"standard\", verbose=False, n_classes=3, label_names=LABEL_NAMES,\n"
         "            )\n"
         "            res = cv.get(\"final\", {})\n"
         "            perseed[f\"{name}|{model}\"] = [float(r[\"aggregate\"][\"macro_f1_mean\"]) for r in cv.get(\"per_seed\", [])]"),
        ("    out.to_csv(out_dir / \"structure_gating_summary.csv\", index=False)",
         "    out.to_csv(out_dir / \"structure_gating_summary.csv\", index=False)\n"
         "    with open(out_dir / \"structure_gating_perseed.json\", \"w\") as _f:\n"
         "        json.dump(perseed, _f, indent=2)"),
    ],
}


def apply(script, pairs):
    with io.open(P + script, "r", encoding="utf-8") as f:
        src = f.read()
    for old, new in pairs:
        n = src.count(old)
        if n != 1:
            raise SystemExit(f"{script}: pattern matched {n}x (expected 1): {old[:60]!r}")
        src = src.replace(old, new, 1)
    with io.open(P + script, "w", encoding="utf-8") as f:
        f.write(src)
    print(f"patched {script} ({len(pairs)} edits)")


if __name__ == "__main__":
    for s, pairs in PATCHES.items():
        apply(s, pairs)
    print("\nall patched.")
