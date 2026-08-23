# Part (e): Papers from Previous Proceedings to Download and Review

## Why These Papers?

Since our recommended venue is **PSB (Pacific Symposium on Biocomputing)**, and our topic is ML-based variant effect prediction for ion channels, the following papers from related venues are directly relevant. They represent the state of the art in the exact problem space we address.

---

## Paper 1 (Highly Recommended)

**"Predicting functional effects of ion channel variants using new phenotypic machine learning methods"**

- **Authors:** Bhatt et al.
- **Journal:** PLOS Computational Biology, 2023
- **PMC:** PMC10019634
- **URL:** https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1010959

### Why Review This Paper:
- **Directly comparable work**: Predicts functional effects (LOF/GOF/neutral) of ion channel variants using ML — the same problem we solve, but for voltage-gated potassium channels instead of nAChRs
- **Reports accuracy of 0.853 and AU-ROC of 0.912** using multi-task multi-kernel learning
- **Curated 959 experiments from 163 publications** — compare their curation approach to ours (351 mutations)
- **Uses electrophysiology-derived labels** — same as our ground truth source

### What to Pay Attention To:
- How they structure the methods section (feature engineering, model selection, evaluation)
- Their approach to multi-task learning (predicting multiple phenotype parameters simultaneously)
- How they handle class imbalance (LOF vs GOF vs neutral)
- Their figure showing the ML pipeline architecture
- How they compare against existing tools (PolyPhen-2, CADD, etc.)

---

## Paper 2 (Highly Recommended)

**"Predicting the functional effects of voltage-gated potassium channel missense variants with multi-task learning"**

- **Authors:** Heyne et al.
- **Preprint:** bioRxiv, 2021
- **URL:** https://www.biorxiv.org/content/10.1101/2021.12.02.470894

### Why Review This Paper:
- **Most similar methodology to ours**: Feature engineering from amino acid properties + structural features to classify LOF vs GOF
- Uses a curated dataset of experimentally characterized ion channel mutations
- Discusses the challenge of small datasets in this domain

### What to Pay Attention To:
- Their feature set (compare with our 50 features)
- How they validate on held-out data
- The types of diagrams they use (confusion matrices, ROC curves, feature importance plots)
- How they discuss limitations and future work

---

## Paper 3 (For PSB-specific formatting reference)

**Any recent PSB proceedings paper from https://psb.stanford.edu/psb-online/**

- **URL:** https://psb.stanford.edu/psb-online/
- Browse proceedings from PSB 2025 or PSB 2026

### Why Review This Paper:
- To understand the **exact formatting, structure, and level of detail** expected in a PSB proceedings paper
- PSB papers have a specific style (12-page limit, specific LaTeX template)
- Understanding how PSB papers balance biological context with computational methods

### What to Pay Attention To:
- Paper length and section breakdown (how much space for intro vs methods vs results)
- Level of biological background provided
- How figures and tables are formatted
- Reference style and density
- Whether they include code/data availability statements

---

## Review Checklist

When reviewing these papers, note the following for each:

| Aspect | Notes |
|--------|-------|
| **Structure** | What sections do they use? (Introduction, Related Work, Methods, Results, Discussion, Conclusion?) |
| **Level of detail** | How much biological background do they provide? |
| **Figures** | What types? (Pipeline diagrams, ROC curves, confusion matrices, feature importance, protein structure visualizations?) |
| **Tables** | What do they tabulate? (Model comparison, feature lists, dataset statistics?) |
| **Evaluation** | What metrics? (F1, accuracy, AUC-ROC, precision, recall?) How do they cross-validate? |
| **Comparison** | Do they compare against baseline tools? Which ones? |
| **Limitations** | How do they discuss limitations? |
| **Data/Code** | Do they share data and code? Where? |
