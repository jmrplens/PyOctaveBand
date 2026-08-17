#  Copyright (c) 2026. Jose Manuel Requena Plens
import os

# Select a non-interactive matplotlib backend for the headless test suite.
# The library no longer forces a backend (see issue #52), so the test harness
# must opt into Agg itself; otherwise matplotlib picks a GUI backend (e.g.
# TkAgg on Windows runners) and figure creation fails without a display/Tcl.
# Set at import time so it takes effect before any test module imports pyplot.
os.environ.setdefault("MPLBACKEND", "Agg")

def pytest_configure(config):
    """
    Configure environment variables for the test session.
    We disable Numba JIT to allow coverage tools to trace inside the kernels.
    """
    # Disable JIT by default for tests to ensure 100% coverage reporting.
    # setdefault: the CI tests-perf job sets NUMBA_DISABLE_JIT=0 explicitly
    # to exercise the jitted kernel; an externally-set value must win.
    os.environ.setdefault("NUMBA_DISABLE_JIT", "1")


# The modules whose tests dominate the wall clock (steady-state FDTD runs,
# the conformance registry, ECMA-418-2 chains, report generation). Collection
# order is dispatch order under pytest-xdist, so fronting them lets the long
# tests start while the workers still have plenty of short tests left to
# backfill with; leaving them in alphabetical collection order used to strand
# a 3-5 minute test on a lone worker at the end of the run.
_FRONTLOADED_MODULES = (
    "tests/simulation/test_fdtd_ntff.py",
    "tests/test_conformance_report.py",
    "tests/simulation/test_elastic_fdtd.py",
    "tests/aircraft/test_rotorcraft_norah2.py",
    "tests/test_generate_reports.py",
    "tests/simulation/test_elastic_fluid_solid.py",
    "tests/building/measurement/test_impact_insulation.py",
    "tests/building/measurement/test_insulation.py",
    "tests/psychoacoustics/quality/test_tonality_ecma.py",
    "tests/psychoacoustics/loudness/test_ecma.py",
    "tests/psychoacoustics/quality/test_fluctuation_strength_ecma.py",
    "tests/psychoacoustics/loudness/test_moore_glasberg_time.py",
    "tests/psychoacoustics/loudness/test_zwicker.py",
    "tests/underwater/propagation/test_numerical.py",
    "tests/filters/test_iec61260_report.py",
    "tests/room/test_impulse_response.py",
    "tests/test_golden_baseline.py",
)


def pytest_collection_modifyitems(config, items):
    """Move the known-heavy modules to the front of the dispatch order.

    The sort is stable, so relative order inside every module (and among all
    remaining modules) is untouched.
    """
    rank = {}
    for item in items:
        path = item.nodeid.split("::", 1)[0]
        if path not in rank:
            rank[path] = len(_FRONTLOADED_MODULES)
            for i, module in enumerate(_FRONTLOADED_MODULES):
                # Match on the tail so the rank survives being invoked from
                # a different rootdir (nodeids are rootdir-relative).
                if path.endswith(module) or module.endswith(path):
                    rank[path] = i
                    break
    items.sort(key=lambda item: rank[item.nodeid.split("::", 1)[0]])


def pytest_report_header(config):
    """Report which copy of every heavy oracle set this run will read.

    The suites in ``tests/oracle_data.DATASETS`` prefer a full local copy of
    reference material that cannot be committed and fall back to the small
    committed oracle under ``tests/data/``. Printing the resolution keeps a
    green run from being ambiguous about which of the two produced it.
    """
    import oracle_data

    lines = [oracle_data.resolve(d).describe() for d in oracle_data.DATASETS]
    return ["oracle data:", *[f"  {line}" for line in lines]]
