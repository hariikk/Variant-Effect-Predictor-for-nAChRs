# Slide outline — "Data collection, but faster"

Audience: biology peers (non-technical). Tone: plain language, no ML jargon.
4 slides. Swap in your own real numbers where noted.

---

## Slide 1 — The problem: manual data collection is slow

**On screen:** left, someone typing rows into Excel one paper at a time; right, the
punchline — "143 mutation rows extracted automatically" (your real number) or
"one folder + one command → a full spreadsheet."

**Say out loud:**
- "You all know this: read a paper, find the numbers, type them into Excel, repeat.
  It's slow, and it's easy to make mistakes."
- "I did exactly this for my thesis — extracting mutation data from papers. By hand,
  each paper took me several minutes. I turned it into: drop the text in a folder,
  run one command, get a spreadsheet in seconds."
- "That's what this talk is about — a free tool that does the boring typing for you."

---

## Slide 2 — The idea: a helper that lives on your computer

**On screen:** the analogy in one line + a tiny diagram
`[ your papers ]  ->  [ Ollama (local AI) ]  ->  [ spreadsheet ]`

**Say out loud:**
- "Ollama is like a very diligent student who lives on **your** computer — not in the cloud."
- "You hand it a stack of papers and a blank table with the column names you want.
  It reads each paper and fills in the rows."
- "Three reasons it's perfect for research: it's **free**, it's **private** (your data
  never leaves your machine), and it needs no special hardware — a normal laptop works."

---

## Slide 3 — The 3 tricks that make it actually work

**On screen:** three short bullets
1. Runs locally — free and private
2. "Structured output" — you name the columns, it returns a table, not an essay
3. It's a template — change the column names to reuse it for **any** data task

**Say out loud:**
- "The magic word is 'structured output'. Normally an AI writes prose. Here we tell it:
  'give me only a table with these exact columns' — and it does."
- "That means the output drops straight into Excel — no copy-pasting, no reformatting."
- "And it's not just for mutations. Whatever *you* collect — patient data, species
  traits, drug doses — you just rename the columns."

---

## Slide 4 — Live demo + how to adapt it to your own work

**On screen:** the `FIELDS` block from the script
```python
FIELDS = [
    ("subunit",   "text"),
    ("mutation",  "text"),
    ("effect",    "choice", ["GOF", "LOF", "No effect"]),
]
```

**Say out loud:**
- *[run it live]* "I'll drop these two text files in and run one command... here's the
  spreadsheet it produced."
- "To use this on your own data, you change exactly **one** place: this list of column
  names. That's it. You don't write any code."
- "I've put the script and a 5-line setup guide in the shared folder — grab it, and I'm
  happy to set it up with you."

---

### Notes for you
- Slide 1's "several minutes per paper" — replace with your actual manual timing if you
  have it (e.g. "~4 min/paper by hand vs seconds with the tool").
- The "143 rows" figure is your real extraction count; feel free to cite it as "my thesis
  used this to extract 143 mutation rows".
- The whole pitch leans on one contrast: **manual = slow & error-prone; tool = free,
  private, and you only rename the columns.**
