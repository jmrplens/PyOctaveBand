#  Copyright (c) 2026. Jose Manuel Requena Plens
"""The committed example ``.report()`` fiches, one module per subject.

``scripts/generate_reports.py`` is the command line over this package; the
fiche builders themselves are grouped by the subject of the result they print
(:mod:`reports.building`, :mod:`reports.materials`, ...), and the registry
that fixes which file each one writes, and in what order, stays with the
command line.

This file exists to run one loop before anything else does. Pinning the
numerical thread pools has to happen BEFORE numpy/scipy import their
backends, and every entry path into the builders -- the command line, the
test suite loading that file by its path, an editor importing a single domain
module -- reaches its numerical imports through a submodule of this package,
so the package ``__init__`` is the one place that is always first.
"""
import os

# Deterministic fiche output: pin every numerical thread pool to a single
# thread BEFORE numpy/scipy import their backends, so multi-threaded reductions
# cannot reorder floating-point sums and perturb the rendered plot across
# machines. The figure generator does the same, and for the same reason: the
# CI runner has a different core count than a dev box.
for _threads_var in (
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_threads_var, "1")
