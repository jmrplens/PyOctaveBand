#  Copyright (c) 2026. Jose Manuel Requena Plens
"""The documentation figures, animations and their shared drawing conventions.

``scripts/generate_graphs.py`` is the command line over this package; the
figure builders themselves are grouped by the subject of the guide that
embeds them (:mod:`figures.signals`, :mod:`figures.building`, ...), over the
shared palette and saving of :mod:`figures.theme` and the Spanish edition of
:mod:`figures.i18n`.

This file exists to run one loop before anything else does. Pinning the
numerical thread pools has to happen BEFORE numpy/scipy/numba import their
backends, and every entry path into the figures -- the command line, ``import
generate_graphs``, a spawned pool worker importing :mod:`figures.registry`,
an editor importing a single domain module -- reaches its numerical imports
through a submodule of this package, so the package ``__init__`` is the one
place that is always first.
"""

import os
import sys

# Deterministic figure output: pin every numerical thread pool to a single
# thread BEFORE numpy/scipy/numba import their backends, so multi-threaded
# reductions cannot reorder floating-point sums and perturb the rendered bytes
# across machines (the CI "Documentation figures" job runs on a different core
# count than a dev box, which is what made the heavy compute figures flaky).
for _threads_var in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "NUMBA_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_threads_var, "1")

# Add src to path to use the local package
sys.path.insert(
    # The code that draws the clips is hashed into
    # scripts/animation_fingerprints.txt, so rewriting this asks for a
    # re-render that would draw the very same frames.
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")),  # noqa: PTH100, PTH118, PTH120
)


def _publish(**values: object) -> None:
    """Rebind *values* in every module of the package that holds a copy.

    The figure bodies read the switchable names (the palette entries the dark
    theme changes, the language and filename suffixes) as plain globals, the
    way they did when every generator lived in one module and
    ``set_lang``/``set_theme`` rebound that module's globals. Splitting the
    generators across modules means each holds its own binding, so the two
    switches push the new values into every module that already has the name
    -- which keeps the read side exactly as it was.
    """
    for module in list(sys.modules.values()):
        name = getattr(module, "__name__", "")
        if name != "generate_graphs" and not name.startswith(__name__ + "."):
            continue
        for key, value in values.items():
            if hasattr(module, key):
                setattr(module, key, value)
