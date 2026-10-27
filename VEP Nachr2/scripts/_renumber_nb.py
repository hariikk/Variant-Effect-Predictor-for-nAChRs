"""One-off: renumber results.ipynb per the user's request and add explanatory rows/notes.

Edits:
  * Split "Idea A/B" into Run XXXV (Idea A) + Run XXXVI (Idea B + per-class companion).
  * α7 resting-structure probe -> Run XXXVII.
  * Remove Run XXXVI (structural-alignment gating) entirely.
  * functional -> XXXVIII, classwise -> XXXIX, single-frame -> XL.
  * Add a "New thing" row to every run's Item|Value conditions table.
  * Add a "how to read the baselines" note to the classwise table.
"""
import io
import nbformat

NB = "results.ipynb"
nb = nbformat.read(NB, as_version=4)
cells = nb.cells


def src(i):
    s = cells[i].get("source", "")
    if isinstance(s, list):
        s = "".join(s)
    return s


def set_src(i, text):
    cells[i]["source"] = text


def sub(i, old, new, count=1):
    s = src(i)
    assert old in s, f"cell {i}: anchor not found: {old!r}"
    s2 = s.replace(old, new, count)
    set_src(i, s2)
    return s2


# ── 1. Split Idea A/B (cells 149, 150, 151) into Run XXXV + Run XXXVI ────────
set_src(149, """## Run XXXV: Idea A — AlphaMissense as one more feature (82 → 83): null; pathogenicity ⊥ direction

*Idea A*: add AlphaMissense pathogenicity as one more feature on top of the 82-feat baseline.
Gene-aware `cv_mode="standard"`, 3-class, RF + CatBoost, **10 seeds × 15 trials**, full 760.

**Macro F1 (82 vs 82+AM):**

| Condition | Model | Macro F1 |
| --- | --- | --- |
| 82 | random_forest | 0.6451 ± 0.0096 |
| 82 | catboost | 0.6470 ± 0.0129 |
| 82+AM | random_forest | 0.6512 ± 0.0089 |
| 82+AM | catboost | 0.6514 ± 0.0121 |

**Significance (paired t-test, 10 seeds, one-sided):** 82+AM − 82: RF +0.006 (p = 0.085) · CB +0.004
(p = 0.149) — both ns.

*(raw: `results/idea_a/`)*""")

set_src(150, """### Run XXXV takeaway (Idea A) — AlphaMissense is null, re-confirming direction ⊥ pathogenicity

82 → 83 features: RF 0.6451→0.6512 (Δ +0.006, p = 0.085), CatBoost 0.6470→0.6514 (Δ +0.004, p = 0.149) —
a small positive drift that is NOT significant for either model. Re-confirms Run VIII: pathogenicity is
orthogonal to direction even as a single added feature. (The earlier 3-seed read showed RF −0.008; that sign
flip was seed variance.)""")

set_src(151, """## Run XXXVI: Idea B — disease→direction label mining: tautological, not additive (+ per-class F1 companion)

*Idea B*: expand labels by mining disease associations (disease → mechanism → direction). On the
clean-label variants, 66% have NO pathology annotation, and the only ≥80%-pure pathology buckets are the
myasthenic syndromes whose *name is the direction label* — CMS (86, 86% LOF), SCCMS (35, 94% GOF), FCCMS
(6, 100% LOF). The abundant common-disease tags (nicotine dependence 41% GOF, ADNFLE 52% GOF, schizophrenia
79% LOF) are direction-mixed. Mining disease would only re-derive labels we already have, not expand
GOF/NNE. **Verdict: tautological, not additive.**

**Companion — per-class F1 per group (82-feat baseline), decomposing *why* δ/γ/ε wins:**

| Group | Model | Macro | LOF | NNE | GOF |
| --- | --- | --- | --- | --- | --- |
| α (517) | random_forest | 0.6262 | 0.7406 | 0.5965 | 0.5416 |
| α (517) | catboost | 0.6158 | 0.7263 | 0.5891 | 0.5321 |
| β (128) | random_forest | 0.6310 | 0.6384 | 0.6620 | 0.5924 |
| β (128) | catboost | 0.6239 | 0.6347 | 0.6485 | 0.5886 |
| δ/γ/ε (115) | random_forest | 0.7001 | 0.8425 | 0.5700 | 0.6877 |
| δ/γ/ε (115) | catboost | 0.7046 | 0.8308 | 0.5996 | 0.6833 |

**Significance (paired t-test, 10 seeds, one-sided):** δ/γ/ε − α: RF +0.074 (p < 0.001) · CB +0.089
(p < 0.001); δ/γ/ε − β: RF +0.069 (p = 0.001) · CB +0.081 (p < 0.001).

**Conditions:** n = 760 (α 517 / β 128 / δγε 115); features = 82 baseline; 10 seeds × 15 trials; gene-aware
`cv_mode="standard"`; models RF + CatBoost; per-seed macro-F1 is the unit.

*(raw: `results/idea_a/`)*

### Run XXXVI takeaway — δ/γ/ε's edge is LOF+GOF-specific, not all-class

Per-class: δγε LOF 0.84/0.83 + GOF 0.69/0.68, but NNE only 0.57/0.60 — the WORST NNE of the three groups
(β 0.66/0.65, α 0.60/0.59). So δγε wins because its two directional classes are clean, not because
everything separates. Refines Runs XXXII–XXXIV. (Superseded in detail by Run XXXIX's full class-wise sweep.)""")

# ── 2. α7 -> Run XXXVII ─────────────────────────────────────────────────────
sub(152, "## α7 resting-structure probe: complete homomeric resting model vs 7EKT — null-to-negative",
    "## Run XXXVII: α7 resting-structure probe — complete homomeric resting model vs 7EKT (null-to-negative)")
sub(154, "### α7 resting-structure takeaways", "### Run XXXVII takeaways")

# ── 3. Renumber functional / classwise / single-frame ────────────────────────
sub(155, "## Run XXXV: functional (GO) gene embedding", "## Run XXXVIII: functional (GO) gene embedding")
sub(157, "### Run XXXV takeaways", "### Run XXXVIII takeaways")
sub(161, "## Run XXXVIII: class-wise F1", "## Run XXXIX: class-wise F1")
sub(163, "### Run XXXVIII takeaways", "### Run XXXIX takeaways")
sub(167, "## Run XXXIX: single-frame mapping", "## Run XL: single-frame mapping")

# ── 4. Fix a dangling cross-reference in Run XXXIII takeaway #4 ─────────────
sub(144, "(Idea B per-class)", "(the per-class F1 companion, Run XXXVI)")

# ── 5. Add "New thing" row to each conditions table ─────────────────────────
new_thing = {
    115: "Treat gene identity as a legitimate model input and evaluate under gene-aware `standard` CV (the prior ~0.50 ceiling was a gene-agnostic CV artifact, not a data limit).",
    119: "Add allosteric-interaction features (gate coupling × mutation Δchemistry) and re-test the GNM gating group under gene-aware CV.",
    127: "Cross-species augmentation — add mouse/rat rows to the human-only set and measure the per-species gain.",
    131: "Quantify how macro-F1 scales with training size and class balance (subsample N; rebalance 50/25/25 vs 33/33/33).",
    135: "Split the data by subunit type (α / β / δ/γ/ε) and train separate models per group.",
    137: "Drop the 16 gene one-hot columns to test whether δ/γ/ε's 0.70 is a gene-identity artifact or biological.",
    155: "Replace the 16 gene one-hots with 7 named functional attributes (α/β/δγε class + homomeric + high-Ca2+ + muscle-type + gene length).",
}
ANCHOR = "| Item | Value |\n|---|---|\n"
for idx, text in new_thing.items():
    s = src(idx)
    assert ANCHOR in s, f"cell {idx}: conditions anchor not found"
    s2 = s.replace(ANCHOR, ANCHOR + f"| New thing | {text} |\n", 1)
    set_src(idx, s2)

# ── 6. Add "how to read the baselines" note to the classwise table ───────────
sub(162, "*(raw: `results/classwise_f1/classwise_f1_summary.csv`)*",
    """*(raw: `results/classwise_f1/classwise_f1_summary.csv`)*

### How to read the baselines (rows 0–2 of each table)

- **majority_class** — "always predict the most common effect" (DummyClassifier `most_frequent`). It never
  predicts the other two classes, so their columns are 0.000 *by construction*; its macro F1 is
  (majority-F1 + 0 + 0)/3. For α and δ/γ/ε (LOF-majority) this is the "always-predict-LOF" floor; for β
  (NNE-majority) it is "always-predict-NNE".
- **uniform_random** — "predict each of the 3 classes with equal 1/3 chance" (the chance floor). Because it
  guesses NNE and GOF a third of the time, those columns are non-zero (~0.28–0.34), unlike majority.
- **stratified_random** — "predict each class with probability = its frequency" (chance, but class-prior-aware).
  Slightly better than uniform on the majority class, worse on the minority; net macro ≈ chance.

These are *analytic* floors (no CV, no learning) — they answer "is the model doing better than a dumb rule?".
The δ/γ/ε table is the explicit "is 0.70 just predicting LOF?" check: always-LOF reaches only 0.259, so the
model's ~0.70 is genuine 3-way discrimination.""")

# ── 7. Remove the structural-alignment gating run (cells 158, 159, 160) ──────
removed = [cells[158], cells[159], cells[160]]
assert "structural-alignment gating" in src(158), "cell 158 is not the structure-gating header"
assert "structure_gating_summary" in src(159), "cell 159 is not the structure-gating table"
assert "structure does NOT gate" in src(160), "cell 160 is not the structure-gating takeaways"
del cells[158:161]

nbformat.write(nb, NB)
print("DONE. Removed 3 cells (structure gating); total cells now", len(cells))
print("Removed headers:")
for c in removed:
    first = (c["source"] if isinstance(c["source"], str) else "".join(c["source"])).split("\n")[0]
    print("  -", first[:90])
