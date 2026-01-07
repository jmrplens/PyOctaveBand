#  Copyright (c) 2026. Jose M. Requena-Plens
import os
import pytest

def pytest_configure(config):
    """
    Configure environment variables for the test session.
    We disable Numba JIT to allow coverage tools to trace inside the kernels.
    """
    # Disable JIT by default for tests to ensure 100% coverage reporting
    os.environ["NUMBA_DISABLE_JIT"] = "1"

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
