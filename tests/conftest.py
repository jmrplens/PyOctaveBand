#  Copyright (c) 2026. Jose Manuel Requena Plens
import os

import pytest

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


@pytest.fixture(autouse=True)
def handle_performance_tests(request):
    """
    Special fixture to re-enable JIT for performance tests.
    """
    if "test_performance.py" in request.node.fspath.strpath:
        # Re-enable JIT for performance measurements
        os.environ["NUMBA_DISABLE_JIT"] = "0"
        # Note: Numba might have already compiled/cached some things, 
        # but this ensures the performance test runs at native speed.
    yield
