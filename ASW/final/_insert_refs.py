# -*- coding: utf-8 -*-
"""Replace the nachr-mutations reference block in main.tex with the fresh one."""
import io

refs = io.open("_nachr_refs.tex", encoding="utf-8").read().strip()
tex = io.open("main.tex", encoding="utf-8").read()

header = (
    "%--------------------------------------------------------------\n"
    "% Primary electrophysiology literature mined for the curated\n"
    "% nAChR variant dataset (nachr mutations.bib). Every source paper\n"
    "% from which LOF/GOF labels were extracted is listed below.\n"
    "%--------------------------------------------------------------\n"
)

# marker that begins the previously inserted block
start_marker = "% Primary electrophysiology literature mined for the curated"
end_marker = "\\end{thebibliography}"

i = tex.find(start_marker)
j = tex.find(end_marker)
if i == -1 or j == -1 or j < i:
    raise SystemExit("block markers not found (i=%s, j=%s)" % (i, j))

# keep everything before the block header, then fresh block, then end marker onward
new_block = header + "\n" + refs + "\n\n"
tex = tex[:i] + new_block + tex[j:]

io.open("main.tex", "w", encoding="utf-8").write(tex)
print("replaced block;", refs.count("\\bibitem{"), "bibitems")
