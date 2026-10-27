# Simple Ollama Data Collector

Turn a folder of text files (papers, abstracts, notes) into a spreadsheet — using a
local AI model that runs on **your own computer**, for free.

## Setup (one time, ~2 minutes)

1. Install **Ollama** from https://ollama.com (double-click the installer).
2. Open a terminal and download the model:
   ```
   ollama pull qwen2.5:7b-instruct
   ```
3. Done. No Python packages to install — the script uses only Python's built-in libraries.

## Use it

1. Put your text files (`.txt`, one per paper/record) into the `input/` folder.
2. Open `simple_collector.py` and edit the lines between `EDIT THESE` and
   `STOP EDITING`: describe what to extract (`INSTRUCTION`) and name your columns
   (`FIELDS`).
3. Run it:
   ```
   python simple_collector.py
   ```
4. Open `output/results.csv` in Excel.

## Why it's nice

- **Free & private** — the model runs on your machine; your text never leaves it.
- **Remembers what it did** — it keeps a list in `output/done.txt`, so you can add more
  files and re-run without redoing old work. Delete `done.txt` to start over.
- **No matches?** It just writes nothing and marks the file done — nothing breaks.

## The sample files

`input/chrna1_example.txt` and `input/chrnb2_example.txt` are examples so you can try it
immediately. Replace them with your own files.
