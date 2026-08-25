#  Copyright (c) 2026. Jose Manuel Requena Plens
"""The guard that keeps the jitted-kernel CI job honest.

``scripts/check_jit_kernel.py`` runs before the test suite in the job that
installs the ``perf`` extra, and fails when the impulse kernel produced no
machine code. Its verdict has to follow the environment in both directions:
green only where numba is installed *and* ``NUMBA_DISABLE_JIT`` is off, red
everywhere else, because a guard that always passes would hide exactly the
regression it is there to catch.
"""

from __future__ import annotations

import importlib.util
import os
import pathlib
import sys

import pytest

_SCRIPTS = str(pathlib.Path(__file__).resolve().parent.parent / "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import check_jit_kernel

#: True only in the environment the guard is meant to pass in: numba installed
#: (the ``perf`` extra) and the JIT left enabled.
_JITTED = (
    importlib.util.find_spec("numba") is not None
    and os.environ.get("NUMBA_DISABLE_JIT") == "0"
)


def test_guard_verdict_follows_the_environment() -> None:
    """The exit status must agree with whether the kernel really compiles."""
    assert (check_jit_kernel.main() == 0) is _JITTED


@pytest.mark.skipif(not _JITTED, reason="numba is absent or the JIT is disabled")
def test_a_pass_is_backed_by_a_printed_signature(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A pass has to name the compiled signatures, not just return zero."""
    assert check_jit_kernel.main() == 0
    out = capsys.readouterr().out
    assert "compiled signature" in out
    assert "array(float64" in out
