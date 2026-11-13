# Graph Report - .  (2026-11-08)

## Corpus Check
- Corpus is ~8,729 words - fits in a single context window. You may not need a graph.

## Summary
- 39 nodes · 58 edges · 12 communities (10 shown, 2 thin omitted)
- Extraction: 60% EXTRACTED · 40% INFERRED · 0% AMBIGUOUS · INFERRED: 23 edges (avg confidence: 0.87)
- Token cost: 103,129 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_VEP-nAChR Work Packages|VEP-nAChR Work Packages]]
- [[_COMMUNITY_Literature Review Deliverables|Literature Review Deliverables]]
- [[_COMMUNITY_Proposal Template & LaTeX Build|Proposal Template & LaTeX Build]]
- [[_COMMUNITY_R&D Timeline DesignTesting|R&D Timeline Design/Testing]]
- [[_COMMUNITY_Report Writing Deliverables|Report Writing Deliverables]]
- [[_COMMUNITY_Travis CI Configuration|Travis CI Configuration]]
- [[_COMMUNITY_Release Packaging|Release Packaging]]
- [[_COMMUNITY_TeX Live Setup|TeX Live Setup]]
- [[_COMMUNITY_HBRS Logo|HBRS Logo]]

## God Nodes (most connected - your core abstractions)
1. `VEP-nAChR Project Gantt Chart (Oct 2026 - Mar 2027)` - 7 edges
2. `Travis CI Configuration (.travis.yml)` - 6 edges
3. `R&D Project Proposal Timeline (Month 0-8)` - 6 edges
4. `pdflatex PDF Build Stage` - 5 edges
5. `WP2 Dataset Assembly & Cross-Species Alignment` - 5 edges
6. `WP3 Subunit-Aware & Structural Extension` - 5 edges
7. `WP5 Project Report` - 5 edges
8. `Project Milestones M1-M6` - 5 edges
9. `Experimental Design or Model Definition Phase (Months 4-5)` - 5 edges
10. `Experiments / Model Testing Phase (Months 5-7)` - 5 edges

## Surprising Connections (you probably didn't know these)
- `Hochschule Bonn-Rhein-Sieg (University of Applied Sciences)` --conceptually_related_to--> `R&D Project Proposal Timeline (Month 0-8)`  [INFERRED]
  project-proposal-master/images/Logo-HBRS-cyan.png → project-proposal-master/images/rnd_deliverable_timeline.png
- `Hochschule Bonn-Rhein-Sieg Logo (cyan)` --conceptually_related_to--> `VEP-nAChR Project Gantt Chart (Oct 2026 - Mar 2027)`  [INFERRED]
  VEP-nAChR-proposal/images/Logo-HBRS-cyan.png → VEP-nAChR-proposal/images/gantt.png
- `Travis CI Build Status Badge` --references--> `Travis CI Configuration (.travis.yml)`  [INFERRED]
  project-proposal-master/README.md → project-proposal-master/.travis.yml
- `MAS Project Proposal Template` --shares_data_with--> `pdflatex PDF Build Stage`  [INFERRED]
  project-proposal-master/README.md → project-proposal-master/.travis.yml
- `Proposal File Naming Convention (DoeJ-RnDProposal.tex)` --shares_data_with--> `pdflatex PDF Build Stage`  [INFERRED]
  project-proposal-master/README.md → project-proposal-master/.travis.yml

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **PDF Build-and-Release CI Pipeline** — project_proposal_master_travis_texlive_install, project_proposal_master_travis_latex_build, project_proposal_master_travis_zip_package, project_proposal_master_travis_releases_deploy [EXTRACTED 1.00]
- **VEP-nAChR Work Package Plan (WP1-WP5)** — vep_nachr_proposal_images_gantt_wp1_literature_study, vep_nachr_proposal_images_gantt_wp2_dataset_assembly, vep_nachr_proposal_images_gantt_wp3_subunit_aware_structural_extension, vep_nachr_proposal_images_gantt_wp4_sota_comparison_ablation, vep_nachr_proposal_images_gantt_wp5_project_report [INFERRED 0.90]
- **R&D Project Proposal Phases (Literature to Reporting)** — r_and_d_project_proposal_master_images_rnd_deliverable_timeline_literature_search, r_and_d_project_proposal_master_images_rnd_deliverable_timeline_literature_analysis_problem_understanding, r_and_d_project_proposal_master_images_rnd_deliverable_timeline_experimental_design_model_definition, r_and_d_project_proposal_master_images_rnd_deliverable_timeline_experiments_model_testing, r_and_d_project_proposal_master_images_rnd_deliverable_timeline_report_writing [EXTRACTED 1.00]
- **R&D Project Proposal Milestone Deliverables (Month 0-8)** — r_and_d_project_proposal_master_images_rnd_deliverable_timeline_list_of_minimum_50_references, r_and_d_project_proposal_master_images_rnd_deliverable_timeline_experimental_setup_platform, r_and_d_project_proposal_master_images_rnd_deliverable_timeline_state_of_the_art_analysis, r_and_d_project_proposal_master_images_rnd_deliverable_timeline_outline_hypothesis, r_and_d_project_proposal_master_images_rnd_deliverable_timeline_initial_report_draft, r_and_d_project_proposal_master_images_rnd_deliverable_timeline_final_report_draft, r_and_d_project_proposal_master_images_rnd_deliverable_timeline_submission [EXTRACTED 1.00]

## Communities (12 total, 2 thin omitted)

### Community 0 - "VEP-nAChR Work Packages"
Cohesion: 0.53
Nodes (9): VEP-nAChR Project Gantt Chart (Oct 2026 - Mar 2027), Project Milestones M1-M6, WP1 Literature Study & Curation Protocol, WP2 Dataset Assembly & Cross-Species Alignment, WP3 Subunit-Aware & Structural Extension, WP4 SOTA Comparison & Ablation, WP5 Project Report, Hochschule Bonn-Rhein-Sieg Logo (cyan) (+1 more)

### Community 1 - "Literature Review Deliverables"
Cohesion: 0.40
Nodes (5): Deliverable: Getting to Know the Experimental Setup / Platform, Deliverable: List of Minimum 50 References, Literature Analysis + Problem Understanding Phase (Months 1-4), Literature Search Phase (Months 0-2), Deliverable: Detailed State of the Art Analysis

### Community 2 - "Proposal Template & LaTeX Build"
Cohesion: 0.83
Nodes (4): MAS Project Proposal Template, Project Proposal README, Proposal File Naming Convention (DoeJ-RnDProposal.tex), pdflatex PDF Build Stage

### Community 3 - "R&D Timeline Design/Testing"
Cohesion: 0.83
Nodes (4): Experimental Design or Model Definition Phase (Months 4-5), Experiments / Model Testing Phase (Months 5-7), Deliverable: Outline + Hypothesis, R&D Project Proposal Timeline (Month 0-8)

### Community 4 - "Report Writing Deliverables"
Cohesion: 0.50
Nodes (4): Deliverable: Final Report Draft, Deliverable: Initial Report Draft, Report Writing Phase (Months 7-8), Deliverable: Submission

### Community 5 - "Travis CI Configuration"
Cohesion: 1.00
Nodes (3): Travis CI Configuration (.travis.yml), Travis CI Build Status Badge, Travis CI

### Community 6 - "Release Packaging"
Cohesion: 0.67
Nodes (3): GitHub Releases, Tag-Triggered GitHub Releases Deployment, MAS-project-proposal.zip Packaging Step

## Knowledge Gaps
- **5 isolated node(s):** `GitHub Releases`, `Hochschule Bonn-Rhein-Sieg (University of Applied Sciences)`, `HBRS Cyan Logo Mark`, `Deliverable: Getting to Know the Experimental Setup / Platform`, `Deliverable: Submission`
  These have ≤1 connection - possible missing edges or undocumented components.
- **2 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `R&D Project Proposal Timeline (Month 0-8)` connect `R&D Timeline Design/Testing` to `HBRS Logo`, `Literature Review Deliverables`, `Report Writing Deliverables`?**
  _High betweenness centrality (0.064) - this node is a cross-community bridge._
- **Why does `Travis CI Configuration (.travis.yml)` connect `Travis CI Configuration` to `Proposal Template & LaTeX Build`, `Release Packaging`, `TeX Live Setup`?**
  _High betweenness centrality (0.036) - this node is a cross-community bridge._
- **Why does `Report Writing Phase (Months 7-8)` connect `Report Writing Deliverables` to `R&D Timeline Design/Testing`?**
  _High betweenness centrality (0.032) - this node is a cross-community bridge._
- **Are the 2 inferred relationships involving `Travis CI Configuration (.travis.yml)` (e.g. with `Travis CI` and `Travis CI Build Status Badge`) actually correct?**
  _`Travis CI Configuration (.travis.yml)` has 2 INFERRED edges - model-reasoned connections that need verification._
- **Are the 4 inferred relationships involving `pdflatex PDF Build Stage` (e.g. with `MAS Project Proposal Template` and `Proposal File Naming Convention (DoeJ-RnDProposal.tex)`) actually correct?**
  _`pdflatex PDF Build Stage` has 4 INFERRED edges - model-reasoned connections that need verification._
- **Are the 4 inferred relationships involving `WP2 Dataset Assembly & Cross-Species Alignment` (e.g. with `Project Milestones M1-M6` and `WP1 Literature Study & Curation Protocol`) actually correct?**
  _`WP2 Dataset Assembly & Cross-Species Alignment` has 4 INFERRED edges - model-reasoned connections that need verification._
- **What connects `GitHub Releases`, `Hochschule Bonn-Rhein-Sieg (University of Applied Sciences)`, `HBRS Cyan Logo Mark` to the rest of the system?**
  _5 weakly-connected nodes found - possible documentation gaps or missing edges._