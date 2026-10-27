#!/usr/bin/env python3
"""
simple_collector.py — turn a folder of text files into a spreadsheet using Ollama.

The idea (in one sentence):
  You drop text files (papers, abstracts, lab notes) into the "input" folder;
  this script asks a local AI model (Ollama) to pull out the facts you care about;
  and it writes them as rows in a spreadsheet you can open in Excel.

Why this is nice for data collection:
  - It runs on YOUR computer (Ollama). No cloud, no API key, free, and your
    text never leaves the machine.
  - You tell it exactly which columns you want, and it returns a table — not an
    essay — so the result drops straight into Excel.

What YOU need to edit: only the lines between "EDIT THESE" and "STOP EDITING".
Then run:  python simple_collector.py

Requirements:
  - Python 3 (any version)
  - Ollama installed (https://ollama.com) and running, with your model pulled:
        ollama pull qwen2.5:7b-instruct
  - No `pip install` needed — this uses only Python's built-in libraries.
"""

import csv
import json
import re
import sys
import urllib.request
from pathlib import Path

# ==================== EDIT THESE ====================

MODEL      = "qwen2.5:7b-instruct"    # the model you pulled in Ollama
OLLAMA_URL = "http://localhost:11434"  # where Ollama listens (default)

INPUT_DIR  = "input"                  # folder of .txt files (one per paper/record)
OUTPUT_CSV = "output/results.csv"      # where the spreadsheet is written
DONE_FILE  = "output/done.txt"         # remembers which files are already done

# What to extract, in plain English. Tell the model what a "record" is and any rules.
INSTRUCTION = """\
You are a careful research assistant. Read the TEXT below and extract every
mutation of a nicotinic acetylcholine receptor (nAChR) subunit that was studied.

Rules:
- Only include mutations where the effect on channel function was measured or stated.
- Map common names to official symbols: alpha7 -> CHRNA7, beta2 -> CHRNB2, etc.
- If no qualifying mutations are described, return an empty list [].
"""

# The columns you want in your spreadsheet. Each line is:
#   ("column name", "kind" [, [allowed values]])
#   kinds:  "text"   -> any text
#           "number" -> a whole number
#           "choice" -> the model must pick one of the allowed values you list
FIELDS = [
    ("subunit",     "text"),
    ("mutation",    "text"),
    ("position",    "number"),
    ("effect",      "choice", ["GOF", "LOF", "No effect"]),
    ("technique",   "text"),
]

# ==================== STOP EDITING ====================

BASE_DIR = Path(__file__).resolve().parent


def build_schema(fields):
    """Turn the FIELDS list into the JSON schema Ollama uses for structured output."""
    properties = {}
    for entry in fields:
        name, kind = entry[0], entry[1]
        if kind == "number":
            properties[name] = {"type": "integer"}
        elif kind == "choice":
            properties[name] = {"type": "string", "enum": entry[2]}
        else:  # "text"
            properties[name] = {"type": "string"}
    return {
        "type": "array",
        "items": {
            "type": "object",
            "properties": properties,
            "required": [e[0] for e in fields],
        },
    }


def field_prompt(fields):
    """Describe the columns in the prompt so the model knows what each one means."""
    lines = []
    for entry in fields:
        name, kind = entry[0], entry[1]
        if kind == "number":
            lines.append(f'  - "{name}": a whole number')
        elif kind == "choice":
            lines.append(f'  - "{name}": one of {", ".join(entry[2])}')
        else:
            lines.append(f'  - "{name}": short text')
    return "\n".join(lines)


def build_prompt(instruction, fields, text):
    return (
        instruction.strip()
        + "\n\nFor each entry, return a JSON object with exactly these fields:\n"
        + field_prompt(fields)
        + "\n\nReturn ONLY a JSON array (no prose, no markdown fences). "
        + "If nothing qualifies, return []."
        + "\n\nTEXT:\n" + text
    )


def check_ollama():
    """Friendly early exit if Ollama isn't running or the model isn't pulled."""
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=5) as resp:
            tags = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        print(f"Can't reach Ollama at {OLLAMA_URL}. Is it installed and running? ({e})")
        sys.exit(1)

    names = [m.get("name", "") for m in tags.get("models", [])]
    base = MODEL.split(":")[0]
    if not any(base in n for n in names):
        print(f"Model '{MODEL}' isn't downloaded yet. Pull it with:\n    ollama pull {MODEL}")
        print("Models you already have: " + (", ".join(names) or "(none)"))
        sys.exit(1)


def call_ollama(prompt, schema):
    """Send the prompt to Ollama and return the raw text answer."""
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "format": schema,               # <-- the trick that forces a table, not prose
        "options": {
            "temperature": 0.1,         # low = consistent, predictable answers
            "num_ctx": 32768,           # how much text the model can read at once
            "num_predict": 8192,        # max length of the answer
        },
    }
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=900) as resp:
        return json.loads(resp.read().decode("utf-8")).get("response", "")


def parse_response(raw):
    """Pull the JSON array out of the model's answer (it sometimes adds markdown)."""
    # 1) the model returned clean JSON
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass

    # 2) it wrapped the JSON in ```json ... ``` fences
    m = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    # 3) it stuck the JSON somewhere in prose
    m = re.search(r"\[.*\]", raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass

    return []  # gave up — treat as "nothing extracted"


def clean(value):
    """Make a value safe to write into a CSV cell (no None, no nested lists)."""
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return ", ".join(str(v) for v in value)
    return value


def load_done(done_path):
    if not done_path.exists():
        return set()
    return {line.strip() for line in done_path.read_text(encoding="utf-8").splitlines() if line.strip()}


def main():
    input_dir = BASE_DIR / INPUT_DIR
    output_path = BASE_DIR / OUTPUT_CSV
    done_path = BASE_DIR / DONE_FILE

    check_ollama()

    files = sorted(input_dir.glob("*.txt"))
    if not files:
        print(f"No .txt files in '{input_dir}'. Put your text files there and run again.")
        sys.exit(1)

    done = load_done(done_path)
    todo = [f for f in files if f.name not in done]
    if not todo:
        print("Nothing new to do - every file is already in the done list.")
        print(f"To redo them all, delete {done_path}.")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    schema = build_schema(FIELDS)
    fieldnames = ["source_file"] + [e[0] for e in FIELDS]

    print(f"Found {len(files)} file(s); {len(todo)} still to process.")

    # Append as we go, so a crash never loses earlier results.
    new_file = not output_path.exists() or output_path.stat().st_size == 0
    csv_encoding = "utf-8-sig" if new_file else "utf-8"  # BOM only on first write
    with open(output_path, "a", newline="", encoding=csv_encoding) as csv_f, \
         open(done_path, "a", encoding="utf-8") as done_f:
        writer = csv.DictWriter(csv_f, fieldnames=fieldnames)
        if new_file:
            writer.writeheader()

        for path in todo:
            text = path.read_text(encoding="utf-8", errors="ignore")
            print(f"\n[{path.name}] {len(text)} characters -> asking {MODEL} ...")
            try:
                raw = call_ollama(build_prompt(INSTRUCTION, FIELDS, text), schema)
                rows = parse_response(raw)
            except Exception as e:
                print(f"  !! error: {e}  (skipped; re-run to retry)")
                continue

            rows = [r for r in rows if isinstance(r, dict)]
            for row in rows:
                row["source_file"] = path.name
                writer.writerow({k: clean(row.get(k, "")) for k in fieldnames})

            done_f.write(path.name + "\n")
            done_f.flush()

            if rows:
                print(f"  -> wrote {len(rows)} row(s)")
            else:
                print("  -> no matching entries (nothing to write)")

    print(f"\nDone. Open '{OUTPUT_CSV}' in Excel.")


if __name__ == "__main__":
    main()
