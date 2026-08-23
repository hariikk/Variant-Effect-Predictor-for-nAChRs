# -*- coding: utf-8 -*-
"""Map the 16 nAChR genes -> canonical UniProt accessions via the search API.

Synchronous, one lightweight GET per gene (no async job / poll / stream, which
was flaky). Uses the reviewed (Swiss-Prot) canonical entry for each gene, which
is what AlphaMissense keys on.
"""
import json
import time

try:
    import requests  # type: ignore
except Exception:
    requests = None

GENES = [
    "CHRNA1", "CHRNA2", "CHRNA3", "CHRNA4", "CHRNA5", "CHRNA6", "CHRNA7",
    "CHRNA9", "CHRNA10", "CHRNB1", "CHRNB2", "CHRNB3", "CHRNB4",
    "CHRND", "CHRNE", "CHRNG",
]


def search(gene):
    url = "https://rest.uniprot.org/uniprotkb/search"
    params = {
        "query": f"gene:{gene} AND organism_id:9606 AND reviewed:true",
        "fields": "accession,id,protein_name",
        "format": "tsv",
    }
    for attempt in range(4):
        try:
            if requests is not None:
                r = requests.get(url, params=params, timeout=30)
                return r.status_code, r.text
            import urllib.request
            import urllib.parse
            q = urllib.parse.urlencode(params)
            with urllib.request.urlopen(f"{url}?{q}", timeout=30) as resp:
                return resp.status, resp.read().decode()
        except Exception as e:
            print(f"  {gene}: retry {attempt+1}: {e}", flush=True)
            time.sleep(2 * (attempt + 1))
    return None, ""


out = {}
for gene in GENES:
    code, text = search(gene)
    if code != 200 or not text:
        print(f"{gene:9s} -> ERROR (code {code})", flush=True)
        out[gene] = "MISSING"
        continue
    lines = text.strip().splitlines()
    if len(lines) < 2:
        print(f"{gene:9s} -> NO RESULT", flush=True)
        out[gene] = "MISSING"
        continue
    # First data row is the canonical entry.
    fields = lines[1].split("\t")
    acc = fields[0]
    entry = fields[1] if len(fields) > 1 else ""
    name = fields[2] if len(fields) > 2 else ""
    out[gene] = acc
    print(f"{gene:9s} -> {acc:10s} {entry:16s} {name[:60]}", flush=True)

print("\nUNIPROT_ACCESSIONS = {")
for gene in GENES:
    print(f'    "{gene}": "{out[gene]}",')
print("}")

# Also dump JSON for downstream use
with open("_uniprot_mapping.json", "w") as fh:
    json.dump(out, fh, indent=2)
print("\nWrote _uniprot_mapping.json")
