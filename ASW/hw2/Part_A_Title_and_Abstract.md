# Part (a): Title and Abstract

## Title

**VEP-nAChR: A Machine Learning-Based Variant Effect Predictor for Classifying Loss-of-Function and Gain-of-Function Mutations in Human Nicotinic Acetylcholine Receptors**

---

## Abstract (Structured via Template)

| Component | Sentence |
|-----------|----------|
| **Introduction (area of study)** | Nicotinic acetylcholine receptors (nAChRs) are ligand-gated ion channels critical for neuromuscular signaling and synaptic transmission, and missense mutations in their 17 human subunit genes are implicated in disorders including congenital myasthenic syndromes, epilepsy, and nicotine dependence. |
| **The problem (that I tackle)** | However, experimentally characterizing the functional effect of each newly discovered nAChR variant through electrophysiology is slow and expensive, leaving the majority of variants as variants of uncertain significance. |
| **What the literature says about this problem** | Existing computational variant effect predictors such as PolyPhen-2, SIFT, and CADD provide pathogenicity scores but do not distinguish between loss-of-function (LOF) and gain-of-function (GOF) mechanisms, which is essential for understanding disease etiology and guiding treatment. |
| **How I tackle this problem** | We address this gap by developing VEP-nAChR, a supervised machine learning classifier trained on a curated dataset of 351 experimentally characterized nAChR missense mutations to predict whether a variant causes LOF or GOF. |
| **How I implement my solution** | Our approach engineers 50 numerical features per mutation encompassing amino acid physicochemical properties, evolutionary substitution scores (BLOSUM62, Grantham distance), protein structural context (B-factor, secondary structure, packing density), and subunit identity, and evaluates seven machine learning algorithms using nested cross-validation with Optuna hyperparameter optimization. |
| **The result** | XGBoost and LightGBM achieve the best performance with an F1 score of 0.67 for GOF detection at 75% accuracy, substantially above the random baseline of 0.38, demonstrating that engineered features from sequence and structure capture meaningful signal for distinguishing LOF from GOF mechanisms in nAChRs. |

---

## Abstract (Paragraph Form)

Nicotinic acetylcholine receptors (nAChRs) are ligand-gated ion channels critical for neuromuscular signaling and synaptic transmission, and missense mutations in their 17 human subunit genes are implicated in disorders including congenital myasthenic syndromes, epilepsy, and nicotine dependence. However, experimentally characterizing the functional effect of each newly discovered nAChR variant through electrophysiology is slow and expensive, leaving the majority of variants as variants of uncertain significance. Existing computational variant effect predictors such as PolyPhen-2, SIFT, and CADD provide pathogenicity scores but do not distinguish between loss-of-function (LOF) and gain-of-function (GOF) mechanisms, which is essential for understanding disease etiology and guiding treatment. We address this gap by developing VEP-nAChR, a supervised machine learning classifier trained on a curated dataset of 351 experimentally characterized nAChR missense mutations to predict whether a variant causes LOF or GOF. Our approach engineers 50 numerical features per mutation encompassing amino acid physicochemical properties, evolutionary substitution scores (BLOSUM62, Grantham distance), protein structural context (B-factor, secondary structure, packing density), and subunit identity, and evaluates seven machine learning algorithms using nested cross-validation with Optuna hyperparameter optimization. XGBoost and LightGBM achieve the best performance with an F1 score of 0.67 for GOF detection at 75% accuracy, substantially above the random baseline of 0.38, demonstrating that engineered features from sequence and structure capture meaningful signal for distinguishing LOF from GOF mechanisms in nAChRs.
