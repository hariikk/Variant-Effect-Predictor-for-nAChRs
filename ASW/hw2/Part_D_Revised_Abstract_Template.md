# Part (d): Revised Publishable Results via Abstract Template

## Revision Notes

After identifying the publishable results (Part b) and target venue (PSB 2027), the abstract is revised to:
1. Sharpen the novelty claim (first LOF/GOF predictor for nAChRs)
2. Add quantitative results more prominently
3. Frame the work for a biocomputing audience (emphasize clinical relevance)
4. Tighten language to fit typical PSB abstract length (~250 words)

---

## Revised Abstract Template

| Component | Revised Sentence |
|-----------|-----------------|
| **Introduction (area of study)** | Missense mutations in the 17 human genes encoding nicotinic acetylcholine receptor (nAChR) subunits cause a spectrum of neurological and neuromuscular disorders, yet the functional consequence of most variants remains unknown. |
| **The problem (that I tackle)** | Determining whether a given nAChR variant causes loss-of-function (LOF) or gain-of-function (GOF) is critical for diagnosis and treatment, but experimental characterization through electrophysiology is prohibitively slow for the growing number of discovered variants. |
| **What the literature says about this problem** | Existing computational predictors (PolyPhen-2, SIFT, CADD, REVEL) classify variants as pathogenic or benign but do not predict the direction of functional change, and no specialized predictor exists for the nAChR family. |
| **How I tackle this problem** | We develop VEP-nAChR, the first machine learning classifier specifically designed to predict LOF versus GOF for nAChR missense mutations, trained on a manually curated dataset of 351 experimentally characterized variants across 15 subunit genes. |
| **How I implement my solution** | Each mutation is represented by 50 engineered features capturing amino acid physicochemical changes, evolutionary substitution likelihood (BLOSUM62, Grantham distance), protein structural context extracted from cryo-EM structures (B-factor, secondary structure, packing density), and subunit identity; seven classifiers are benchmarked using stratified nested cross-validation with Bayesian hyperparameter optimization. |
| **The result** | Gradient boosting models (XGBoost, LightGBM) achieve F1 = 0.67 for GOF detection and 75% overall accuracy, with structural features providing consistent improvement, establishing a computational baseline for nAChR variant interpretation that can prioritize variants for experimental validation and support clinical decision-making. |

---

## Revised Abstract (Paragraph Form, ~230 words)

Missense mutations in the 17 human genes encoding nicotinic acetylcholine receptor (nAChR) subunits cause a spectrum of neurological and neuromuscular disorders, yet the functional consequence of most variants remains unknown. Determining whether a given nAChR variant causes loss-of-function (LOF) or gain-of-function (GOF) is critical for diagnosis and treatment, but experimental characterization through electrophysiology is prohibitively slow for the growing number of discovered variants. Existing computational predictors (PolyPhen-2, SIFT, CADD, REVEL) classify variants as pathogenic or benign but do not predict the direction of functional change, and no specialized predictor exists for the nAChR family. We develop VEP-nAChR, the first machine learning classifier specifically designed to predict LOF versus GOF for nAChR missense mutations, trained on a manually curated dataset of 351 experimentally characterized variants across 15 subunit genes. Each mutation is represented by 50 engineered features capturing amino acid physicochemical changes, evolutionary substitution likelihood (BLOSUM62, Grantham distance), protein structural context extracted from cryo-EM structures (B-factor, secondary structure, packing density), and subunit identity; seven classifiers are benchmarked using stratified nested cross-validation with Bayesian hyperparameter optimization. Gradient boosting models (XGBoost, LightGBM) achieve F1 = 0.67 for GOF detection and 75% overall accuracy, with structural features providing consistent improvement, establishing a computational baseline for nAChR variant interpretation that can prioritize variants for experimental validation and support clinical decision-making.

---

## Key Revisions from Version 1

| Aspect | Original (Part a) | Revised (Part d) |
|--------|-------------------|------------------|
| Opening framing | Described nAChRs generally | Leads with clinical impact of mutations |
| Novelty claim | Implicit | Explicit: "first ML classifier specifically designed for nAChR LOF/GOF" |
| Literature gap | Mentioned existing tools generically | Named specific tools (PolyPhen-2, SIFT, CADD, REVEL) and stated no nAChR-specific predictor exists |
| Clinical relevance | Mentioned at end | Threaded throughout and emphasized in conclusion |
| Word count | ~250 | ~230 (tighter) |
