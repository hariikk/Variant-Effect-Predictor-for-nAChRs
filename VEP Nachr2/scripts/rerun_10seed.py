# -*- coding: utf-8 -*-
"""
Re-run the gene-aware experiments (Run XXVI → XXXVIII, minus the removed XXXVII)
at 10 seeds × 15 trials, CONCURRENTLY (default 3 jobs), resumable via markers.

Run XXXIV (significance) is already 10-seed and is intentionally skipped.

Each command writes its output to logs/rerun_10seed/<label>.log, and a
`<label>.done` marker is written only after the command exits 0. Re-running this
script skips any command whose marker exists, so a killed run resumes where it
left off. The MASTER log records start/end/duration/status per command.

Concurrency: commands run independently (each is its own subprocess), so up to
`--jobs` of them run at once. A single command averages ~4-5 cores (ESM-2
extraction is single-threaded; model fitting is short bursts), so running ~3 in
parallel roughly triples throughput without oversubscribing the 16-core box.
Results are bit-identical to sequential (fixed seeds; n_jobs/thread count does
not change the trained model).

Launch fully detached (survives the ~3-min background-kill on this Windows box):
    powershell -Command "Start-Process -FilePath python -ArgumentList '-u','scripts/rerun_10seed.py','--jobs','3' -WorkingDirectory '<cwd>' -RedirectStandardOutput logs/rerun_10seed/MASTER.console.log -RedirectStandardError logs/rerun_10seed/MASTER.console.err -WindowStyle Hidden -PassThru"
"""
import argparse
import concurrent.futures
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT / "logs" / "rerun_10seed"
LOG_DIR.mkdir(parents=True, exist_ok=True)

PY = sys.executable

# (label, argv) — chronological. XXXVII removed; XXXIV already 10-seed (skipped).
COMMANDS = [
    ("xxvi_gene_aware",        ["scripts/run_xxvi_gene_aware.py", "--n-seeds", "10", "--n-trials", "15"]),
    ("xxvii_allosteric",       ["scripts/run_xxvii_allosteric.py", "--n-seeds", "10", "--n-trials", "15"]),
    ("xxviii_structural",      ["scripts/run_xxviii_structural_feature_ablation.py", "--n-seeds", "10", "--n-trials", "15"]),
    ("xxix_species",           ["scripts/run_xxix_species_ablation.py", "--n-seeds", "10", "--n-trials", "15"]),
    ("xxx_data_size",          ["scripts/run_xxx_data_size.py", "--n-seeds", "10", "--n-trials", "15"]),
    ("xxxi_subunit",           ["scripts/run_xxxi_subunit_type.py", "--n-seeds", "10", "--n-trials", "15"]),
    ("xxxii_gene_identity",    ["scripts/run_xxxii_gene_identity.py", "--n-seeds", "10", "--n-trials", "15"]),
    ("xxxiii_group_ablation",  ["scripts/run_xxxiii_group_ablation.py", "--n-seeds", "10", "--n-trials", "15", "--out-name", "group_ablation_10seed"]),
    ("xxxv_functional",        ["scripts/run_xxxv_functional.py", "--n-seeds", "10", "--n-trials", "15"]),
    ("xxxvi_structure_gating", ["scripts/run_xxxvi_structure_gating.py", "--n-seeds", "10", "--n-trials", "15"]),
    ("idea_a",                 ["scripts/run_alphamissense_probe.py", "--n-seeds", "10", "--n-trials", "15"]),
    ("a7resting_probe",        ["scripts/run_a7resting_probe.py", "--n-seeds", "10", "--n-trials", "15"]),
    ("a7resting_geomonly",     ["scripts/run_a7resting_geomonly.py", "--n-seeds", "10", "--n-trials", "15"]),
    ("classwise_f1",           ["scripts/run_classwise_f1.py", "--n-seeds", "10", "--n-trials", "15"]),
]

_lock = threading.Lock()


def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with _lock:
        with open(LOG_DIR / "MASTER.log", "a", encoding="utf-8") as f:
            f.write(line + "\n")


def run_one(label, argv):
    log(f"START {label}: {PY} -u {' '.join(argv)}")
    t0 = time.time()
    log_path = LOG_DIR / f"{label}.log"
    with open(log_path, "wb") as lf:
        proc = subprocess.run(
            [PY, "-u"] + argv,
            cwd=str(ROOT),
            stdout=lf,
            stderr=subprocess.STDOUT,
        )
    dt = time.time() - t0
    return label, proc.returncode, dt, log_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=3, help="max concurrent commands")
    args = ap.parse_args()

    todo = [(label, argv) for label, argv in COMMANDS
            if not (LOG_DIR / f"{label}.done").exists()]
    for label, _ in COMMANDS:
        if (LOG_DIR / f"{label}.done").exists():
            log(f"SKIP  {label} (already done)")
    log(f"=== rerun_10seed start — {len(todo)} to run, jobs={args.jobs} ===")

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futs = {ex.submit(run_one, label, argv): label for label, argv in todo}
        for fut in concurrent.futures.as_completed(futs):
            label = futs[fut]
            label, rc, dt, log_path = fut.result()
            if rc == 0:
                (LOG_DIR / f"{label}.done").write_text(
                    f"done rc=0 in {dt/60:.1f} min\n", encoding="utf-8")
                log(f"DONE  {label} rc=0 in {dt/60:.1f} min")
            else:
                log(f"FAIL  {label} rc={rc} in {dt/60:.1f} min (log: {log_path})")

    log("=== rerun_10seed finished ===")


if __name__ == "__main__":
    main()
