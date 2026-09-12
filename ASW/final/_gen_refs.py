# -*- coding: utf-8 -*-
"""Parse nachr mutations.bib and emit LaTeX \\bibitem entries for main.tex."""
import re

SRC = "nachr mutations.bib"
OUT = "_nachr_refs.tex"

raw = open(SRC, encoding="utf-8").read()

# Split into @Article blocks (non-greedy up to matching closing brace at line start)
blocks = re.findall(r"@Article\{([^,]+),\s*(.*?)\n\}", raw, re.S)
print("found", len(blocks), "entries")

def get_field(body, name):
    m = re.search(r"^\s*" + name + r"\s*=\s*\{(.*?)\}\s*,?\s*$", body, re.M | re.S)
    return m.group(1).strip() if m else None

# unicode -> latex
def latex(s):
    if s is None:
        return ""
    repl = {
        "\\textsuperscript{2+}": r"$^{2+}$",
        "&amp;": r"\&",
        "α": r"$\alpha$", "β": r"$\beta$", "γ": r"$\gamma$",
        "δ": r"$\delta$", "ε": r"$\varepsilon$", "ɛ": r"$\varepsilon$", "σ": r"$\sigma$",
        "μ": r"$\mu$", "κ": r"$\kappa$", "π": r"$\pi$", "λ": r"$\lambda$",
        "ß": r"{\ss}", "é": r"{\'e}", "è": r"{\`e}", "ê": r"{\^e}",
        "ë": r"{\"e}", "á": r"{\'a}", "à": r"{\`a}", "â": r"{\^a}",
        "ä": r"{\"a}", "í": r"{\'i}", "ì": r"{\`i}", "î": r"{\^i}",
        "ï": r"{\"i}", "ó": r"{\'o}", "ò": r"{\`o}", "ô": r"{\^o}",
        "ö": r"{\"o}", "ú": r"{\'u}", "ù": r"{\`u}", "û": r"{\^u}",
        "ü": r"{\"u}", "ñ": r"{\~n}", "ç": r"{\c c}", "ã": r"{\~a}",
        "õ": r"{\~o}", "å": r"{\r a}", "ø": r"{\o}", "ć": r"{\'c}",
        "č": r"{\v c}", "š": r"{\v s}", "ž": r"{\v z}", "Ž": r"{\v Z}",
        "ě": r"{\v e}", "ř": r"{\v r}", "ń": r"{\'n}", "ł": r"{\l}",
        "ō": r"{\=o}", "ū": r"{\=u}", "ę": r"{\k e}", "ą": r"{\k a}",
        "’": "'", "‘": "`", "”": "''", "“": "``", "–": "--", "—": "---",
        "…": r"\dots", "·": r"$\cdot$", "×": r"$\times$",
        "≥": r"$\geq$", "≤": r"$\leq$", "±": r"$\pm$", "≈": r"$\approx$",
        "°": r"$^\circ$", "†": r"$^\dagger$", "−": "-",
        "‐": "-", "‑": "-", " ": " ", "​": "", "\xa0": " ",
        "₂": r"$_2$", "₃": r"$_3$", "₄": r"$_4$", "₅": r"$_5$",
        "⁺": r"$^+$", "⁻": r"$^-$", "²": r"$^2$", "³": r"$^3$",
        "⅐": "", "⅒": "",
    }
    for k, v in repl.items():
        s = s.replace(k, v)
    return s

def fmt_author(author):
    """First author 'initials surname', plus 'et al.' if >1 author."""
    if not author:
        return ""
    # split on ' and ' (case-sensitive, also handle 'AND')
    parts = re.split(r"\s+and\s+", author)
    first = parts[0].strip()
    # format first author
    if "," in first:
        last, firsts = first.split(",", 1)
        last = latex(last.strip())
        initials = []
        for token in firsts.split():
            token = token.strip().rstrip(".")
            if token and token not in ("Jr", "Jr.", "II", "III", "IV"):
                # take first letter of each word, skip particles lowercased
                initials.append(token[0].upper() + ".")
        if initials:
            return " ".join(initials) + " " + last
        return last
    else:
        # "First Last" order
        return latex(first)

def build(ref):
    author = get_field(ref, "author")
    title = latex(get_field(ref, "title"))
    journal = latex(get_field(ref, "journal"))
    year = get_field(ref, "year")
    volume = get_field(ref, "volume")
    pages = get_field(ref, "pages")
    number = get_field(ref, "number")

    who = fmt_author(author)
    n_authors = len(re.split(r"\s+and\s+", author)) if author else 1
    if n_authors > 1:
        who += " {\\em et al.}"

    parts = [who + ","]
    parts.append(title + ",")
    j = "{\\em " + journal + "}" if journal else ""
    if volume:
        j += " {\\bf " + volume + "}"
        if number:
            j += " (" + number + ")"
    if pages:
        j += ", " + pages
    if year:
        j += " (" + year + ")"
    parts.append(j + ".")
    return " ".join(parts)

out = []
for key, body in blocks:
    out.append("\\bibitem{" + key + "}")
    out.append(build(body))
    out.append("")

with open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(out))

print("wrote", len(blocks), "bibitems to", OUT)
