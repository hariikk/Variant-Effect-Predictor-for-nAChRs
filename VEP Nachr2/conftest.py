"""pytest configuration: make vep_nachr2 importable and share fixtures."""

import importlib.util
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class _SilentStdout:
    """Minimal stdout stand-in used only while importing the ablation script.

    scripts/run_xxxiii_group_ablation.py calls ``sys.stdout.reconfigure(...)`` at
    import time, which fails under pytest's captured stdout. We swap stdout for
    this object during the import so the module loads cleanly (it prints nothing
    at module level — all printing happens inside the guarded ``main()``).
    """

    def reconfigure(self, *args, **kwargs):
        pass

    def write(self, s):
        pass

    def flush(self):
        pass

    def isatty(self):
        return False


@pytest.fixture(scope="session")
def names82() -> list[str]:
    """The 82-column feature layout, built from the real extractor names.

    75 base (physicochemical 24 + substitution 3 + positional 20 + conservation 3
    + structural_core 7 + esm2 2 + esm_embeddings 16) + gating 3 + allosteric 4.
    Extractor __init__ is cheap (torch/ESM load lazily inside extract()), so this
    does not pull in the language model.
    """
    from vep_nachr2.features.orchestrator import FeatureOrchestrator
    from vep_nachr2.features.gating import GatingExtractor
    from vep_nachr2.features.allosteric_interaction import AllostericInteractionExtractor

    base = FeatureOrchestrator(verbose=False).get_feature_names()
    gating = GatingExtractor().get_feature_names()
    allost = AllostericInteractionExtractor().get_feature_names()
    names = list(base) + list(gating) + list(allost)
    assert len(names) == 82, f"expected 82 features, got {len(names)}"
    return names


@pytest.fixture(scope="session")
def ablation_script():
    """Load scripts/run_xxxiii_group_ablation.py so tests read its real GROUPS
    and FEATURE_GROUPS (not a re-typed copy)."""
    path = PROJECT_ROOT / "scripts" / "run_xxxiii_group_ablation.py"
    spec = importlib.util.spec_from_file_location("run_xxxiii_group_ablation", path)
    mod = importlib.util.module_from_spec(spec)
    _old_stdout = sys.stdout
    sys.stdout = _SilentStdout()
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.stdout = _old_stdout
    return mod


@pytest.fixture(scope="session")
def loaded_df():
    """The cleaned 760-variant dataset (loaded once)."""
    from vep_nachr2.data.loader import load_mutation_data

    return load_mutation_data(data_file="final_mapped.xlsx")
