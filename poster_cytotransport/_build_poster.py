# Duplicate Poster_example.pptx and swap in VEP-nAChR content IN PLACE,
# preserving the example's positions, sizes, fonts, colors and alignment.
import shutil, sys
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR

SRC = "poster_cytotransport/Poster_example.pptx"
DST = sys.argv[1] if len(sys.argv) > 1 else "poster_cytotransport/VEP_poster.pptx"
FIG = "poster_cytotransport/"

shutil.copy(SRC, DST)
prs = Presentation(DST)
slide = prs.slides[0]

BLUE  = "009FD6"
DBLUE = "016ABE"
DARK  = "1F2937"
FONT  = "Helvetica"

A_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"
R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"

# ---------------- helpers ----------------
def find_shape(shapes, name):
    for sh in shapes:
        if sh.name == name:
            return sh
        if sh.shape_type == 6:  # group
            r = find_shape(sh.shapes, name)
            if r is not None:
                return r
    return None

def clear_tf(tf):
    # leave a single empty paragraph, no runs / breaks / fields
    while len(tf.paragraphs) > 1:
        p = tf.paragraphs[-1]._p
        p.getparent().remove(p)
    p0 = tf.paragraphs[0]
    for r in list(p0.runs):
        r._r.getparent().remove(r._r)
    for tag in ("br", "fld"):
        for el in p0._p.findall("{%s}%s" % (A_NS, tag)):
            p0._p.remove(el)

def set_text(shape, lines, size, bold=False, color=None, align=None, italic=False,
             line_spacing=None, space_after=None):
    tf = shape.text_frame
    tf.word_wrap = True
    clear_tf(tf)
    for i, ln in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        if align is not None:
            p.alignment = align
        if line_spacing is not None:
            p.line_spacing = line_spacing
        if space_after is not None:
            p.space_after = Pt(space_after)
        r = p.add_run(); r.text = ln
        r.font.name = FONT
        r.font.size = Pt(size)
        r.font.bold = bold
        r.font.italic = italic
        if color is not None:
            r.font.color.rgb = RGBColor.from_string(color)

def replace_picture(shape, path):
    blip = shape._element.blipFill.find("{%s}blip" % A_NS)
    rId = blip.get("{%s}embed" % R_NS)
    part = shape.part
    image_part = part.related_part(rId)
    with open(path, "rb") as f:
        image_part._blob = f.read()

# ---------------- header ----------------
set_text(find_shape(slide.shapes, "Textfeld 17"),
         ["A Variant Effect Predictor for Nicotinic Acetylcholine Receptors"],
         size=80, bold=True, color=BLUE)

# authors (bold lead + regular rest)
authors = find_shape(slide.shapes, "Textfeld 1062")
tf = authors.text_frame
clear_tf(tf)
p = tf.paragraphs[0]
r = p.add_run(); r.text = "Hari Krishna Mohan Raj"
r.font.name = FONT; r.font.size = Pt(32); r.font.bold = True
r = p.add_run(); r.text = ", Supervisors: Dr. Karl N. Kirschner, Dr. Alexander Hagg"
r.font.name = FONT; r.font.size = Pt(32); r.font.bold = False

set_text(find_shape(slide.shapes, "Textfeld 1063"),
         ["Hochschule Bonn-Rhein-Sieg (HBRS) · Sankt Augustin"],
         size=32, align=PP_ALIGN.RIGHT)

# ---------------- 1. Introduction ----------------
set_text(find_shape(slide.shapes, "Textfeld 22"), [
    "Nicotinic acetylcholine receptors (nAChRs) are pentameric ligand-gated ion channels. "
    "Missense variants cause disease by increasing channel activity (gain-of-function, GOF) or "
    "decreasing it (loss-of-function, LOF) — opposite effects mapping onto distinct disorders and therapies.",
    "Most variant-effect predictors (AlphaMissense, PolyPhen, CADD) score pathogenicity, not direction. "
    "A pathogenic variant can be GOF or LOF, so these tools cannot separate the two. We present the "
    "first nAChR-specific GOF-vs-LOF predictor.",
], size=32, align=PP_ALIGN.JUSTIFY, line_spacing=1.0)

# top-right "Challenges" note
_ch = find_shape(slide.shapes, "Flussdiagramm: Verzögerung 58")
_chtf = _ch.text_frame
_chtf.word_wrap = True
clear_tf(_chtf)
_chp = _chtf.paragraphs[0]
_chp.alignment = PP_ALIGN.CENTER
_chr = _chp.add_run(); _chr.text = "Challenges"
_chr.font.name = FONT; _chr.font.size = Pt(26); _chr.font.bold = True
_chr.font.color.rgb = RGBColor.from_string(DARK)
for _txt in [
    "• Small data (717 variants) — transfer over from-scratch training",
    "• Direction ≠ pathogenicity — GOF vs LOF, not damaging vs benign",
    "• Gene leakage — needs leave-one-subunit-out CV",
    "• Cross-species residue-numbering mismatch",
]:
    _chp = _chtf.add_paragraph()
    _chp.alignment = PP_ALIGN.LEFT
    _chp.line_spacing = 1.15
    _chp.space_after = Pt(6)
    _chr = _chp.add_run(); _chr.text = _txt
    _chr.font.name = FONT; _chr.font.size = Pt(22); _chr.font.bold = False
    _chr.font.color.rgb = RGBColor.from_string(DARK)

# ---------------- 2. Methods ----------------
set_text(find_shape(slide.shapes, "Textfeld 38"), [
    "717 curated missense variants across 16 nAChR genes and three species (human, rat, mouse), each "
    "labelled GOF, no-net-effect or LOF from electrophysiology literature.",
    "75 engineered features in seven groups: physicochemical (24), substitution (3), positional "
    "gene-identity (20), conservation (3), structural core (7), ESM-2 zero-shot (2) and ESM-2 "
    "embeddings (16).",
    "Ten classifiers (LR, SVM, random forest, LightGBM, XGBoost, CatBoost, k-NN, MLP, Gaussian NB), "
    "tuned with Optuna.",
    "Nested leave-one-subunit-out (gene-grouped) cross-validation prevents gene leakage.",
], size=32, line_spacing=1.25, space_after=8)

# ---------------- 3. Dataset / pipeline section ----------------
set_text(find_shape(slide.shapes, "Textfeld 6"),
         ["3. Pipeline & Data"], size=44, bold=True, color=BLUE)
# pipeline/data text — "data creation" highlighted (bold lead)
_ds = find_shape(slide.shapes, "Textfeld 50")
_dstf = _ds.text_frame
_dstf.word_wrap = True
clear_tf(_dstf)
_dsp = _dstf.paragraphs[0]
_dsp.line_spacing = 1.0
_dsr = _dsp.add_run(); _dsr.text = "Data creation: "
_dsr.font.name = FONT; _dsr.font.size = Pt(32); _dsr.font.bold = True
_dsr.font.color.rgb = RGBColor.from_string(DARK)
_dsr = _dsp.add_run(); _dsr.text = "717 missense variants manually curated from electrophysiology literature."
_dsr.font.name = FONT; _dsr.font.size = Pt(32); _dsr.font.bold = False
_dsr.font.color.rgb = RGBColor.from_string(DARK)
_dsp = _dstf.add_paragraph()
_dsp.line_spacing = 1.0
_dsr = _dsp.add_run(); _dsr.text = "16 nAChR genes · three species (507 human, 138 rat, 72 mouse) · 51% LOF / 23% NNE / 26% GOF."
_dsr.font.name = FONT; _dsr.font.size = Pt(32); _dsr.font.bold = False
_dsr.font.color.rgb = RGBColor.from_string(DARK)
set_text(find_shape(slide.shapes, "Textfeld 56"),
         ["Fig. 1: VEP-nAChR pipeline — curated data → features → classifiers → gene-grouped CV → direction."],
         size=28, color=BLUE, align=PP_ALIGN.JUSTIFY)
set_text(find_shape(slide.shapes, "Textfeld 1049"),
         ["Tab. 1: Dataset composition incl. target distribution"],
         size=28, color=BLUE, align=PP_ALIGN.JUSTIFY)

# ---------------- 4. Results ----------------
set_text(find_shape(slide.shapes, "Textfeld 14"),
         ["4. Results"], size=44, bold=True, color=BLUE)
set_text(find_shape(slide.shapes, "Textfeld 60"), [
    "Under gene-grouped CV, CatBoost reaches macro F1 0.51 (3-class) and Random Forest 0.65 (binary) — "
    "above the 0.33 baseline, but below clinical utility (Fig. 2).",
    "Conservation and ESM-2 scores lift 3-class F1 by +0.023 but leave binary unchanged (+0.004): they "
    "separate “effect” from “no effect”, not GOF from LOF.",
], size=32, line_spacing=1.0)
set_text(find_shape(slide.shapes, "Textfeld 5"),
         ["Fig. 2: Model performance (macro F1, mean of 3 seeds) under gene-grouped CV."],
         size=28, color=BLUE)

# ---------------- 5. Feature analysis ----------------
set_text(find_shape(slide.shapes, "Textfeld 48"),
         ["5. Feature Analysis"], size=44, bold=True, color=BLUE)
set_text(find_shape(slide.shapes, "Textfeld 49"), [
    "Gene identity dominates: dropping the positional features costs ΔF1 −0.078, followed by "
    "substitution (−0.033) and ESM-2 embeddings (−0.020); the remaining groups are near-neutral (Fig. 3).",
], size=32, line_spacing=1.0)
set_text(find_shape(slide.shapes, "Textfeld 12"),
         ["Fig. 3: Feature-group ablation (7 groups, mean of Random Forest & CatBoost) — Δ macro F1 when dropped."],
         size=28, color=BLUE, align=PP_ALIGN.JUSTIFY)

# ---------------- 6. Future Work ----------------
set_text(find_shape(slide.shapes, "Textfeld 41"),
         ["6. Future Work"], size=44, bold=True, color=BLUE)
set_text(find_shape(slide.shapes, "Textfeld 43"), [
    "• Scale and diversify the curated dataset with mechanistically labelled cross-species variants "
    "(principled residue-numbering alignment).",
    "• Engineer nAChR-specific mechanism features — gating energetics, allosteric coupling, subunit "
    "interfaces — instead of more generic pathogenicity scores.",
    "• Build a gene-conditioned predictor that uses subunit identity explicitly, to move past the "
    "~0.50 gene-agnostic ceiling.",
    "• Validate predictions against prospective electrophysiology before any clinical use.",
], size=30, line_spacing=1.3, space_after=10)

# ---------------- References ----------------
set_text(find_shape(slide.shapes, "Textfeld 1042"), [
    "(1) Cheng J, et al. Accurate proteome-wide missense variant effect prediction with AlphaMissense. Science 381:eadg7492 (2023).",
    "(2) Meier J, et al. Language models enable zero-shot prediction of the effects of mutations on protein function. NeurIPS 34 (2021).",
    "(3) Boßelmann CM, et al. Predicting functional effects of ion channel variants using phenotypic machine learning. PLoS Comput Biol 19:e1010959 (2023).",
    "(4) Zhong G, et al. PreMode predicts mode-of-action of missense variants by deep graph representation learning. bioRxiv (2024).",
    "(5) Ke G, et al. LightGBM: a highly efficient gradient boosting decision tree. NeurIPS 30 (2017).",
    "(6) Adzhubei IA, et al. A method and server for predicting damaging missense mutations. Nat Methods 7:248 (2010).",
    "(7) Kircher M, et al. A general framework for estimating the relative pathogenicity of human genetic variants. Nat Genet 46:310 (2014).",
    "(8) Ioannidis NM, et al. REVEL: an ensemble method for predicting the pathogenicity of rare missense variants. Am J Hum Genet 99:877 (2016).",
    "(9) Stein D, et al. Genome-wide prediction of pathogenic gain- and loss-of-function variants from ensemble learning of a diverse feature set. Genome Med 15:103 (2023).",
], size=13, align=PP_ALIGN.JUSTIFY, line_spacing=1.0)

# ---------------- figures ----------------
replace_picture(find_shape(slide.shapes, "Grafik 1071"), FIG + "fig1_pipeline.png")
replace_picture(find_shape(slide.shapes, "Grafik 1057"), FIG + "fig2_models.png")
replace_picture(find_shape(slide.shapes, "Grafik 1046"), FIG + "fig3_ablation.png")

# ---------------- table (replace) ----------------
old_table = find_shape(slide.shapes, "Tabelle 10")
x, y = old_table.left, old_table.top
w, h = old_table.width, old_table.height
old_table._element.getparent().remove(old_table._element)

rows_data = [
    ["Class", "Variants", "Share"],
    ["Loss-of-function (LOF)", "363", "51%"],
    ["No-net-effect (NNE)", "164", "23%"],
    ["Gain-of-function (GOF)", "190", "26%"],
    ["Total", "717", "100%"],
]
gt = slide.shapes.add_table(len(rows_data), 3, x, y, w, h).table
gt.first_row = True
for ri, row in enumerate(rows_data):
    for ci, val in enumerate(row):
        cell = gt.cell(ri, ci)
        cell.text = val
        cell.margin_left = Inches(0.08); cell.margin_right = Inches(0.08)
        for p in cell.text_frame.paragraphs:
            p.alignment = PP_ALIGN.CENTER
            for r in p.runs:
                r.font.name = FONT
                r.font.size = Pt(22 if ri == 0 else 20)
                r.font.bold = (ri == 0 or ri == len(rows_data)-1)
                if ri == 0:
                    r.font.color.rgb = RGBColor.from_string("FFFFFF")
                else:
                    r.font.color.rgb = RGBColor.from_string(DARK)

prs.save(DST)
print("saved", DST)
