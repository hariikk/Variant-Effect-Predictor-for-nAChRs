"""
Archived experimental feature extractors.

These modules were hypothesis-driven attempts to extract GOF/LOF direction
signal from single-residue structure/geometry. Each was tested under
leave-one-subunit-out CV and came back NULL (see README.md for the run-to-result
map). They are kept here for the thesis ablation section, NOT wired into the
default feature set (see vep_nachr2.features.orchestrator._get_default_extractors).

Do not import these from production code — they are dead by design.
"""
