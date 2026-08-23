# Part (b): Identifying Publishable Results

## What Are the Publishable Results of This Research?

### 1. Primary Result: LOF vs GOF Classification Performance
- **XGBoost and LightGBM achieve F1 = 0.67** for GOF detection on 351 curated nAChR mutations using 5-fold cross-validation
- This is the **first ML-based variant effect predictor specifically for nAChRs** that distinguishes LOF from GOF (not just pathogenic vs benign)
- Performance is substantially above random baseline (F1 = 0.38)

### 2. Feature Engineering Contribution
- A **50-feature representation** combining:
  - 24 physicochemical features (8 AAIndex properties x 3: WT, MT, difference)
  - 3 substitution scores (BLOSUM62 raw, normalized, + novel Grantham distance)
  - 17 positional/subunit features (normalized position + 16-way one-hot encoding)
  - 6 structural features from PDB (B-factor, DSSP secondary structure, C-beta density)
- **Grantham distance is a novel addition** not present in the reference ENaC project

### 3. Structural Features Impact
- Adding structural features (B-factor, DSSP, C-beta density) provides a **modest but consistent improvement** (+1-2% F1) for gradient boosting models
- 76% of mutations (267/351) successfully mapped to experimental PDB structures across 4 cryo-EM structures

### 4. Curated nAChR Mutation Database
- **351 experimentally characterized missense mutations** across 15 of 17 human nAChR subunit genes
- Each labeled as LOF or GOF based on published electrophysiology data
- Manually curated from primary literature with PubMed references

### 5. Multi-Model Benchmarking
- Systematic comparison of 7 ML algorithms on the same dataset and features
- Gradient boosting methods (XGBoost, LightGBM) consistently outperform linear models, SVMs, and neural networks on this small-data regime

### 6. Cross-Protein Transferability
- The same feature engineering framework from ENaC (epithelial sodium channel) was successfully adapted to nAChRs (a different ion channel superfamily), suggesting the approach generalizes across ion channel families

---

## Mapping Results to the Abstract Template

| Abstract Component | How Results Support It |
|--------------------|----------------------|
| **Introduction** | nAChRs are clinically important; mutations cause CMS, epilepsy, nicotine dependence |
| **Problem** | Experimental characterization is slow; LOF vs GOF distinction is clinically critical but computationally unaddressed |
| **Literature** | Existing tools (PolyPhen-2, SIFT, CADD) predict pathogenicity but not mechanism (LOF vs GOF) |
| **Approach** | Supervised ML with engineered features from sequence + structure |
| **Implementation** | 50 features, 7 models, nested CV with Optuna, 351-mutation curated dataset |
| **Result** | F1 = 0.67, accuracy = 75%, above random baseline, structural features help modestly |
