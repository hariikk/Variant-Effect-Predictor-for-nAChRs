"""
Download + filter the AlphaMissense aa-substitutions table to the 16 nAChR genes.

Streams the ~1.2 GB gzipped table from Google Cloud Storage, keeps only rows whose
UniProt accession is one of the 16 nAChR genes (from config.UNIPROT_ACCESSIONS),
and writes a compact TSV (data/raw/alphamissense/nachr_16_genes.tsv) consumed by
AlphamissenseExtractor.

Data: Cheng et al. 2023 (AlphaMissense), CC BY-NC-SA 4.0 (non-commercial).
Source file schema: uniprot_id \t protein_variant \t am_pathogenicity \t am_class
"""

import gzip
import sys
from pathlib import Path

from vep_nachr2.config import UNIPROT_ACCESSIONS, ALPHAMISSENSE_TSV

GCS_URL = (
    "https://storage.googleapis.com/dm_alphamissense/"
    "AlphaMissense_aa_substitutions.tsv.gz"
)


def main():
    accessions = set(UNIPROT_ACCESSIONS.values())
    out_path = Path(ALPHAMISSENSE_TSV)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Streaming {GCS_URL}", flush=True)
    print(f"Keeping {len(accessions)} accessions: {sorted(accessions)}", flush=True)

    try:
        import requests  # type: ignore
        resp = requests.get(GCS_URL, stream=True, timeout=120)
        resp.raise_for_status()
        reader = gzip.GzipFile(fileobj=resp.raw)
    except Exception as e:  # fall back to stdlib
        print(f"  requests unavailable/failed ({e}); using urllib", flush=True)
        import urllib.request
        raw = urllib.request.urlopen(GCS_URL, timeout=120)
        reader = gzip.GzipFile(fileobj=raw)

    kept = 0
    total = 0
    with open(out_path, "w", newline="") as out:
        # The source file begins with a '#' copyright comment before the real
        # column header; skip comments and write the first real line as header.
        header_written = False
        for raw_line in reader:
            line = raw_line.decode()
            if not header_written:
                if line.startswith("#"):
                    print(f"Skipping comment: {line.strip()[:50]}...", flush=True)
                    continue
                out.write(line)
                print(f"Header: {line.strip()}", flush=True)
                header_written = True
                continue
            total += 1
            uniprot = line.split("\t", 1)[0]
            if uniprot in accessions:
                out.write(line)
                kept += 1
            if total % 10_000_000 == 0:
                print(f"  ... scanned {total / 1e6:.0f}M rows, kept {kept}", flush=True)

    size_kb = out_path.stat().st_size / 1024
    print(f"\nDone: scanned {total:,} rows, kept {kept:,}", flush=True)
    print(f"Wrote {out_path} ({size_kb:.0f} KB)", flush=True)

    if kept == 0:
        print("WARNING: kept 0 rows — accessions likely wrong or schema mismatch.",
              file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
