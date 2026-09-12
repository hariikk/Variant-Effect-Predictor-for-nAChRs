# -*- coding: utf-8 -*-
"""Append 20 post-2020 papers to the Survey Tool.xlsx (11-column schema)."""
import openpyxl

papers = [
    # 1
    dict(
        title="Biological structure and function emerge from scaling unsupervised learning to 250 million protein sequences",
        doi="10.1073/pnas.2016239118",
        authors="Alexander Rives, Joshua Meier, Tom Sercu, Siddharth Goyal, Zeming Lin, Jason Liu, Demi Guo, Myle Ott, C. Lawrence Zitnick, Jerry Ma, Rob Fergus",
        year=2021,
        motivation="Determine whether unsupervised transformer language models trained on massive protein-sequence corpora can capture biological structure and function without explicit supervision.",
        methodology="Trained the ESM-1b transformer (650M parameters) on 250 million UniRef50 sequences with a masked-language-modelling objective, then probed internal attention maps for contact prediction and fine-tuned for downstream tasks.",
        sample="250 million protein sequences (UniRef50) spanning the tree of life; no per-family multiple sequence alignments.",
        tool="ESM-1b transformer (PyTorch / FAIRseq); attention-map contact prediction; fine-tuned secondary-structure and mutational-effect heads.",
        results="Attention heads learned residue-residue contacts comparable to co-evolutionary methods; fine-tuned representations reached state-of-the-art secondary-structure and variant-effect prediction without multiple sequence alignments.",
        limitations="Contact maps degrade on shallow-MSA / homology-depleted families; 650M parameters impose a GPU-memory cost that limits on-device deployment.",
        questions="Can scaled language models replace MSA-based co-evolution entirely for residue-contact and variant-effect prediction?",
    ),
    # 2
    dict(
        title="Language models enable zero-shot prediction of the effects of mutations on protein function",
        doi="10.1101/2021.07.09.450648",
        authors="Joshua Meier, Roshan Rao, Robert Verkuil, Jason Liu, Tom Sercu, Alexander Rives",
        year=2021,
        motivation="Predict mutational effects across protein families without requiring supervised labels or homologous-sequence alignments.",
        methodology="Scored each substitution as the log-likelihood ratio log P(mutant | context) - log P(wildtype | context) under an ensemble of ESM-1v masked language models, yielding a zero-shot effect score.",
        sample="ESM-1v trained on UniRef90 (98 million sequences); benchmarked on 41 deep-mutational-scan datasets (ProteinGym).",
        tool="ESM-1v masked language models (650M parameters); masked-marginal scoring.",
        results="Zero-shot predictions matched or exceeded supervised and MSA-based methods on DMS benchmarks (Spearman ~0.5 on stability) and matched AlphaFold-based methods on some clinical benchmarks.",
        limitations="Scores rank effect magnitude but not direction (GOF vs LOF); heavy-tailed log-likelihood ratios complicate downstream scaling.",
        questions="Can a zero-shot language model be augmented to resolve gain-of-function vs loss-of-function direction?",
    ),
    # 3
    dict(
        title="Disease variant prediction with deep protein language models",
        doi="10.1038/s41588-021-00952-2",
        authors="Jonathan Frazer, Pascal Notin, Mafalda Dias, Aidan Gomez, Joseph K. Min, Kelly Brock, Yarin Gal, Debora S. Marks",
        year=2021,
        motivation="Predict whether human missense variants are pathogenic using evolutionary sequence information only, avoiding supervised labels that carry circularity and bias.",
        methodology="Trained a Bayesian variational autoencoder (EVE) across 140k protein families from multiple sequence alignments, then scored variants by the log-likelihood ratio of mutant vs wild-type under the generative model.",
        sample="140,000 protein families; 250 million sequences; benchmarked on ~3,219 ClinVar-labelled genes and ~60 DMS datasets.",
        tool="EVE variational autoencoder; EVcouplings MSA generation.",
        results="Achieved ClinVar AUC ~0.91 and predicted DMS effects with correlations rivaling experimental reproducibility, outperforming unsupervised baselines.",
        limitations="Requires deep MSAs; MSA generation is computationally expensive; does not distinguish GOF from LOF.",
        questions="Can generative evolutionary models be extended to predict functional direction rather than pathogenicity alone?",
    ),
    # 4
    dict(
        title="Highly accurate protein structure prediction with AlphaFold",
        doi="10.1038/s41586-021-03819-2",
        authors="John Jumper, Richard Evans, Alexander Pritzel, Tim Green, Michael Figurnov, Olaf Ronneberger, Kathryn Tunyasuvunakool, Russ Bates, Augustin Zidek, Anna Potapenko, et al.",
        year=2021,
        motivation="Solve the protein structure prediction problem at experimental accuracy from amino-acid sequence alone.",
        methodology="Evoformer architecture combining MSA and pairwise representations with iterative recycling and a structure-module refinement, trained end-to-end and assessed in CASP14.",
        sample="~170,000 PDB structures; CASP14 blind assessment of ~90 target domains.",
        tool="AlphaFold2 deep learning system (Evoformer, structure module, Amber relaxation).",
        results="Achieved median GDT ~92.4 in CASP14, competitive with experimental structures, and enabled proteome-wide structure prediction.",
        limitations="Predicts a single static structure; limited modelling of conformational ensembles, ligands, and intrinsically disordered regions.",
        questions="How can predicted structures be leveraged for variant-effect and allosteric-mechanism prediction?",
    ),
    # 5
    dict(
        title="Structure and gating mechanism of the alpha7 nicotinic acetylcholine receptor",
        doi="10.1016/j.cell.2021.02.049",
        authors="Colleen M. Noviello, Anant Gharpure, Nuriya Mukhtasimova, Rico Cabuco, Leah Baxter, Dominika Borek, Steven M. Sine, Ryan E. Hibbs",
        year=2021,
        motivation="Resolve how the homomeric alpha7 nAChR couples agonist binding to channel gating, as a template for neuronal nAChR structure and pharmacology.",
        methodology="Cryo-EM structures of the human alpha7 nAChR in resting, activated (epibatidine-bound), and desensitized states, combined with functional mutation studies.",
        sample="Human alpha7 nAChR homopentamer; three conformational states at near-atomic resolution.",
        tool="Cryo-electron microscopy; single-particle reconstruction; electrophysiology.",
        results="Revealed a gating mechanism in which the extracellular domain rotates to open the pore, with the C-loop and M2-M3 linker transmitting the conformational wave; key gating residues identified.",
        limitations="alpha7-only; the homomeric receptor may not capture heteromeric (alpha4beta2) gating nuances; detergent-solubilized states may differ from native membrane.",
        questions="Can state-specific structural features (open vs closed) serve as predictive features for variant direction-of-effect?",
    ),
    # 6
    dict(
        title="Gene variant effects across sodium channelopathies predict function and guide precision therapy",
        doi="10.1093/brain/awac006",
        authors="Andreas Brunklaus, Tony Feng, Tobias Bruenger, Eduardo Perez-Palma, Henrike Heyne, Emma Matthews, Christopher Semsarian, Joseph D. Symonds, Sameer M. Zuberi, Dennis Lal, Stephanie Schorge",
        year=2022,
        motivation="Predict gain- vs loss-of-function for sodium-channel variants to guide precision therapy in epilepsy and cardiac disorders, where the two directions require opposite treatments.",
        methodology="Curated biophysically characterised NaV variants and trained the funNCion classifier on sequence, conservation, and channel-topology features to predict GOF vs LOF.",
        sample="369 biophysically characterised sodium-channel variants validated against whole-cell patch-clamp.",
        tool="funNCion classifier (Grantham, BLOSUM, channel-topology features); SCN-viewer online interpreter.",
        results="Predicted GOF with 78.5% agreement and LOF with 75.0% agreement vs patch-clamp; accuracy was highest in pore-loop / inactivation regions (78-96%) and lower in the S4 voltage sensor (<62%).",
        limitations="Restricted to voltage-gated sodium (and some calcium) channels; not validated on ligand-gated channels such as nAChRs; label quality limited by assay heterogeneity.",
        questions="Can direction-aware prediction be extended to ligand-gated ion channels using receptor-specific structural features?",
    ),
    # 7
    dict(
        title="Embeddings from protein language models predict conservation and variant effects",
        doi="10.1007/s00439-021-02411-y",
        authors="Celine Marquet, Michael Heinzinger, Tobias Olenyi, Christian Dallago, Kyra Erckert, Michael Bernhofer, Dmitrii Nechaev, Burkhard Rost",
        year=2022,
        motivation="Systematically assess whether protein language model embeddings capture evolutionary conservation and whether they improve variant effect prediction over classical methods.",
        methodology="Compared embeddings from ProtT5 and ESM-1b against sequence-based conservation measures and evaluated them on variant-effect benchmarks.",
        sample="Proteome-scale embeddings; variant-effect benchmarks including SIFT/PolyPhen annotations and experimental deep mutational scans.",
        tool="ProtT5 and ESM-1b embeddings; conservation predictors; downstream linear/logistic models.",
        results="Embeddings correlated strongly with conservation (Spearman up to ~0.7) and improved effect prediction when combined with classical features, but did not fully replace MSA-derived conservation.",
        limitations="Embedding quality depends on training-corpus representation; no directional (GOF/LOF) signal; limited interpretability.",
        questions="Can embeddings be distilled into interpretable features that retain directional variant information?",
    ),
    # 8
    dict(
        title="Tranception: protein fitness prediction with autoregressive transformers and inference-time retrieval",
        doi="10.48550/arXiv.2205.13760",
        authors="Pascal Notin, Mafalda Dias, Jonathan Frazer, Javier Marchena-Hurtado, Aidan N. Gomez, Debora S. Marks, Yarin Gal",
        year=2022,
        motivation="Improve protein fitness prediction over masked language models by modelling the full conditional distribution with autoregression and retrieval-augmented context.",
        methodology="Trained a large autoregressive transformer (Tranception) with autoregressive likelihood scoring and inference-time retrieval of homologous sequences from an MSA.",
        sample="~700 million protein sequences (UniRef); evaluated on ProteinGym deep-mutational-scan benchmarks.",
        tool="Tranception autoregressive transformer; MSA retrieval.",
        results="State-of-the-art zero-shot fitness prediction on many ProteinGym assays, outperforming ESM-1v on autoregression-friendly tasks.",
        limitations="Autoregressive scoring is slower than masked-marginal; retrieval adds latency and depends on MSA availability.",
        questions="Do retrieval-augmented models transfer better to low-homology families such as ion-channel subunits?",
    ),
    # 9
    dict(
        title="Genome-wide prediction of disease variant effects with a deep protein language model",
        doi="10.1038/s41588-023-01465-w",
        authors="Nadav Brandes, Grant Goldman, Charlotte H. Wang, Chun Jimmie Ye, Vasilis Ntranos",
        year=2023,
        motivation="Scale variant effect prediction to the whole proteome using a single protein language model, bypassing per-gene training and MSA generation.",
        methodology="Used ESM-1b embeddings with a logistic-regression head trained on ClinVar labels, evaluating genome-wide pathogenicity prediction.",
        sample="All ~19,000 human proteins; ClinVar training labels; benchmarked against CADD, REVEL, and AlphaMissense.",
        tool="ESM-1b embeddings + logistic regression (ESM-1v scoring variant).",
        results="Achieved competitive whole-proteome pathogenicity prediction with far lower compute than AlphaMissense, though below AlphaMissense accuracy.",
        limitations="Predicts pathogenicity, not direction; performance below MSA-based AlphaMissense; dependent on ClinVar label quality.",
        questions="Can a lightweight PLM head be trained for ion-channel-specific GOF/LOF direction?",
    ),
    # 10
    dict(
        title="Evolutionary-scale prediction of atomic-level protein structure with a language model",
        doi="10.1126/science.ade2574",
        authors="Zeming Lin, Halil Akin, Roshan Rao, Brian Hie, Zhongkai Zhu, Wenting Lu, Nikita Smetanin, Robert Verkuil, Ori Kabeli, Yaniv Shmueli, et al.",
        year=2023,
        motivation="Enable fast atomic-level structure prediction directly from a language model, without MSA input, for scale and speed.",
        methodology="Converted ESM-2 internal representations into a structure-prediction head (ESMFold) that predicts atom coordinates in a single forward pass.",
        sample="ESM-2 trained on hundreds of millions of sequences; ESMFold benchmarked against AlphaFold2 on CASP14 and proteomes.",
        tool="ESM-2 + ESMFold folding head.",
        results="Predicted structures at near-AlphaFold2 accuracy but orders of magnitude faster (seconds vs hours), enabling metagenomic-scale structure prediction.",
        limitations="Lower accuracy on difficult targets and low-MSA proteins; no explicit co-evolution coupling.",
        questions="Can ESMFold-predicted structures provide residue-level features for variant-effect prediction in under-studied proteins?",
    ),
    # 11
    dict(
        title="The landscape of tolerated genetic variation in humans and primates",
        doi="10.1126/science.abn8153",
        authors="Hong Gao, Tobias Hamp, Jeffrey Ede, Joshua G. Schraiber, Jeremy McRae, Moriel Singer-Berk, et al., Kyle Kai-How Farh",
        year=2023,
        motivation="Build a clinically accurate variant effect predictor by learning from common primate variation, which is shaped by natural selection against pathogenic alleles.",
        methodology="Trained a 3D convolutional neural network (PrimateAI-3D) on ~4.3 million common missense variants from 233 primate species, using protein 3D structure as input.",
        sample="233 primate species; 4.3 million common variants; validated in ~100,000 human patients.",
        tool="PrimateAI-3D 3D-CNN (sequence + AlphaFold structure).",
        results="Achieved clinically leading variant classification accuracy, surpassing REVEL and CADD and matching AlphaMissense on benchmarks.",
        limitations="Requires high-quality structures; limited to missense variants; does not predict functional direction.",
        questions="Can structure-based deep networks predict GOF vs LOF when trained on functionally annotated channel variants?",
    ),
    # 12
    dict(
        title="Genome-wide prediction of pathogenic gain- and loss-of-function variants from ensemble learning of a diverse feature set",
        doi="10.1186/s13073-023-01261-9",
        authors="David Stein, Meltem Ece Kars, Yiming Wu, Cigdem Sevim Bayrak, Peter D. Stenson, David N. Cooper, Avner Schlessinger, Yuval Itan",
        year=2023,
        motivation="Predict pathogenic gain-of-function (GOF), loss-of-function (LOF), and neutral variants genome-wide, since pathogenicity-only predictors collapse the two directions.",
        methodology="Trained an ensemble of gradient-boosted trees (LoGoFunc) on gene-, protein-, and variant-level features including conservation, biochemical properties, and domain annotations.",
        sample="Genome-wide ClinVar labels spanning GOF, LOF, and neutral classes across all human genes.",
        tool="LoGoFunc ensemble classifier; interactive server (itanlab.shinyapps.io/goflof).",
        results="Outperformed pathogenicity-only tools on GOF/LOF prediction (balanced accuracy ~0.72 for the three-class problem) and improved the resolution of clinically relevant functional direction.",
        limitations="Treats all genes uniformly; no receptor-specific structural context; label scarcity for GOF limits power.",
        questions="Does receptor-family-specific modelling (e.g., nAChR) beat a genome-wide GOF/LOF ensemble on ion-channel variants?",
    ),
    # 13
    dict(
        title="Rapid protein stability prediction using deep learning representations",
        doi="10.7554/eLife.82593",
        authors="Lasse M. Blaabjerg, Maher M. Kassem, Lydia L. Good, Nicolas Jonsson, Matteo Cagiada, Kristoffer E. Johansson, Wouter Boomsma, Amelie Stein, Kresten Lindorff-Larsen",
        year=2023,
        motivation="Provide fast, accurate prediction of protein stability change (delta-delta-G) on mutation using deep learning, for high-throughput variant screening.",
        methodology="Trained an ensemble of graph neural networks (RaSP) on experimental delta-delta-G data, using ESM-1v embeddings and structural features.",
        sample="~10,000 experimentally measured mutations across proteins for training and benchmarking (ProTherm / delta-delta-G datasets).",
        tool="RaSP ensemble (GNN + ESM-1v embeddings).",
        results="Predicted delta-delta-G with correlation ~0.8 at high speed, ranking among top stability predictors while orders of magnitude faster than physics-based methods.",
        limitations="Stability is one axis of effect, not direction; accuracy drops for stabilizing mutations; limited coverage of membrane proteins.",
        questions="Does delta-delta-G help separate GOF from LOF in membrane receptors where both classes can be destabilizing?",
    ),
    # 14
    dict(
        title="ProteinGym: Large-scale benchmarks for protein design and fitness prediction",
        doi="10.1101/2023.12.07.570727",
        authors="Pascal Notin, Aaron W. Kollasch, Daniel Ritter, Lood van Niekerk, Steffanie Paul, Han Spinner, Nathan Rollins, Ada Shaw, Rose Orenbuch, Ruben Weitzman, Jonathan Frazer, Mafalda Dias, Dinko Franceschi, Yarin Gal, Debora S. Marks",
        year=2023,
        motivation="Establish a standardized, large-scale benchmark for comparing protein fitness and variant effect predictors, replacing ad hoc datasets.",
        methodology="Curated ~90 deep-mutational-scan datasets spanning stability, binding, activity, and expression, with standardized scoring and evaluation protocols.",
        sample="~90 DMS datasets, ~2.4 million variants across diverse proteins.",
        tool="ProteinGym benchmark framework; standardized Spearman / MSE / AUC metrics.",
        results="Enabled systematic comparison of zero-shot (ESM-1v, EVE, Tranception) and supervised models and exposed failure modes in low-data regimes.",
        limitations="DMS assays biased toward soluble, high-throughput proteins; few ion channels or membrane receptors represented.",
        questions="How can benchmark coverage be extended to ion channels and membrane receptors for GOF/LOF tasks?",
    ),
    # 15
    dict(
        title="Accurate structure prediction of biomolecular interactions with AlphaFold3",
        doi="10.1038/s41586-024-07487-w",
        authors="Josh Abramson, Jonas Adler, Jack Dunger, Richard Evans, Tim Green, Alexander Pritzel, Olaf Ronneberger, Lindsay Willmore, Andrew J. Ballard, et al.",
        year=2024,
        motivation="Extend structure prediction to complexes of proteins, nucleic acids, small molecules, and ions with unified accuracy.",
        methodology="Diffusion-based architecture (AlphaFold3) that jointly models all biomolecular components without hand-coded constraints.",
        sample="Trained on PDB and related structure databases; benchmarked on protein-ligand, protein-DNA/RNA, and complex prediction.",
        tool="AlphaFold3 diffusion model (with AlphaFold Server).",
        results="Substantially improved protein-ligand and protein-nucleic-acid complex prediction over AlphaFold2 and enabled modelling of post-translational modification and ion coordination.",
        limitations="Server-only initial release (weights later released); less accurate for some antibody-antigen and disordered cases; not conformational ensembles.",
        questions="Can AlphaFold3-modelled ligand-bound states supply binding-site features for nAChR variant-effect prediction?",
    ),
    # 16
    dict(
        title="PreMode predicts mode-of-action of missense variants by deep graph representation learning of protein sequence and structural context",
        doi="10.1101/2024.02.20.581321",
        authors="Guojie Zhong, Yige Zhao, Demi Zhuang, Wendy K. Chung, Yufeng Shen",
        year=2024,
        motivation="Predict mode of action (GOF vs LOF) of missense variants, going beyond binary pathogenicity.",
        methodology="SE(3)-equivariant graph neural networks over sequence and structure, pretrained on pathogenicity then transfer-learned for gene-specific mode-of-action.",
        sample="41,081 DMS variants across 8 genes; 2,043 GOF and 7,889 LOF variants in ~1,300 genes.",
        tool="PreMode SE(3)-GNN (ESM-2 embeddings, AlphaFold2 structure, conservation).",
        results="State-of-the-art mode-of-action prediction; AUC ~0.75 for GOF/LOF in most proteins with few labels; improved DMS analysis.",
        limitations="Binary GOF/LOF only; low-pLDDT / disordered regions poorly handled; would benefit from protein-dynamics features.",
        questions="Can SE(3)-GNN representations transfer to pentameric ligand-gated channels with subunit-interface effects?",
    ),
    # 17
    dict(
        title="Structural bases for stoichiometry-selective calcium potentiation of a neuronal nicotinic receptor",
        doi="10.1111/bph.16321",
        authors="Simone Mazzaferro, Guipeun Kang, Kathiresan Natarajan, Ryan E. Hibbs, Steven M. Sine",
        year=2024,
        motivation="Determine the structural basis of calcium potentiation of alpha4beta2 nAChRs and how stoichiometry (alpha4)2(beta2)3 vs (alpha4)3(beta2)2 shapes it.",
        methodology="Cryo-EM structures of alpha4beta2 nAChRs in multiple stoichiometries with and without calcium, combined with electrophysiology.",
        sample="Human alpha4beta2 nAChR; two stoichiometries; multiple ligand/ion states.",
        tool="Cryo-EM; two-electrode voltage clamp; structural analysis.",
        results="Identified a calcium-binding site at the alpha4-beta2 interface that explains stoichiometry-dependent potentiation, with implications for receptor modulation.",
        limitations="Focused on a single receptor subtype; allosteric modulation may not generalize to other nAChR subtypes.",
        questions="Do subunit-interface and ligand-site features derived from such structures improve nAChR variant-effect prediction?",
    ),
    # 18
    dict(
        title="Democratizing protein language models with parameter-efficient fine-tuning",
        doi="10.1073/pnas.2405840121",
        authors="Samuel Sledzieski, Meghana Kshirsagar, Minkyung Baek, Rahul Dodhia, Juan Lavista Ferres, Bonnie Berger",
        year=2024,
        motivation="Make protein language model fine-tuning accessible without full model retraining, which is prohibitive for most labs.",
        methodology="Applied parameter-efficient fine-tuning (LoRA) to ESM-2, demonstrating strong downstream task performance with a tiny fraction of trainable parameters.",
        sample="ESM-2 (650M); downstream tasks including stability (delta-delta-G), secondary structure, and mutational-effect prediction.",
        tool="LoRA fine-tuning of ESM-2.",
        results="Matched or exceeded full fine-tuning on stability prediction while training <1% of parameters, reducing cost and hardware requirements.",
        limitations="Gains are task-dependent; retrieval-based and MSA methods still lead on some effect benchmarks.",
        questions="Can parameter-efficient fine-tuning adapt ESM-2 to a nAChR GOF/LOF task with a small curated dataset?",
    ),
    # 19
    dict(
        title="xTrimoPGLM: unified 100B-scale pretrained transformer for deciphering the language of protein",
        doi="10.1038/s41592-024-02305-z",
        authors="Bo Chen, Xingyi Cheng, Pan Li, Yangli-ao Geng, Jing Gong, Shen Li, Zhilei Bei, Xu Tan, Boyuan Li, Jiang Wang, Xin Zeng, et al., Le Song",
        year=2024,
        motivation="Unify autoregressive and bidirectional protein language modelling at 100B scale to advance protein understanding and generation.",
        methodology="Trained a 100-billion-parameter transformer (xTrimoPGLM) on ~1 trillion tokens using a unified GLM objective, then evaluated on folding, stability, and variant-effect tasks.",
        sample="~1 trillion protein tokens; benchmarks including stability (delta-delta-G), fitness, and structure.",
        tool="xTrimoPGLM (100B) unified transformer.",
        results="Achieved state-of-the-art on several protein benchmarks, including delta-delta-G prediction, and enabled large-scale protein generation.",
        limitations="Enormous compute requirements (100B parameters); limited accessibility; incremental gains over smaller models on some tasks.",
        questions="Do 100B-scale models yield directional (GOF/LOF) signal beyond what 150M models provide, or does it saturate?",
    ),
    # 20
    dict(
        title="Functional Effect Predictions For Ion Channel Missense Variants Using a Protein Language Model",
        doi="10.1101/2025.10.16.25337735",
        authors="Sean Gies, Artoghrul Alishbayli, Paul H.E. Tiesinga, Marijn B. Martens",
        year=2025,
        motivation="Predict GOF vs LOF for ion-channel missense variants, where the two directions demand opposite therapies and electrophysiology is too slow.",
        methodology="Trained a classifier (MissION) on ESM-2 embeddings plus sequence, structural, residue, Human Phenotype Ontology, and Gene Ontology features in separate channels, with a focal-loss objective.",
        sample="3,176 GOF/LOF-labelled ion-channel variants (largest such dataset); predictions provided for >600,000 channel variants.",
        tool="MissION (ESM-2 + multi-channel features).",
        results="ROC-AUC 0.925 for GOF/LOF, outperforming the prior leading model (0.897), and generalized to channel genes with little or no electrophysiology data.",
        limitations="Imperfect ground-truth labels; context-dependent effects; nAChR is not a focal gene family (aggregates diverse channels).",
        questions="Does a nAChR-specific model outperform channel-agnostic MissION on nAChR GOF/LOF, given subunit and pentameric context?",
    ),
]

path = "Survey Tool.xlsx"
wb = openpyxl.load_workbook(path)
ws = wb["Sheet1"]

headers = ["Title", "DOI", "Authors", "Year", "Motivation", "Methodology",
           "Sample Size/Context", "Intervention/Tool Used", "Results",
           "Limitations", "Questions"]

# existing data ends at row 16; rows 17-19 are empty. Start writing at 17.
start_row = 17
for i, p in enumerate(papers):
    r = start_row + i
    ws.cell(row=r, column=1, value=p["title"])
    ws.cell(row=r, column=2, value=p["doi"])
    ws.cell(row=r, column=3, value=p["authors"])
    ws.cell(row=r, column=4, value=p["year"])
    ws.cell(row=r, column=5, value=p["motivation"])
    ws.cell(row=r, column=6, value=p["methodology"])
    ws.cell(row=r, column=7, value=p["sample"])
    ws.cell(row=r, column=8, value=p["tool"])
    ws.cell(row=r, column=9, value=p["results"])
    ws.cell(row=r, column=10, value=p["limitations"])
    ws.cell(row=r, column=11, value=p["questions"])

wb.save(path)
print("Added", len(papers), "papers starting at row", start_row, "->", start_row + len(papers) - 1)
