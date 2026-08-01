#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Assert that the numba-jitted impulse kernel really compiles, and warm its cache.

The CI job that installs the ``perf`` extra exists to exercise the compiled
kernel, so a silent fall back to the interpreted path (numba missing from the
environment, ``NUMBA_DISABLE_JIT`` left at its default, a decorator that stops
being applied) would leave the job green while testing nothing it is there for.
Run this before the suite: it fails loudly when no machine code was produced.

It has to run under the same environment as the suite it guards, which is why
the job declares ``NUMBA_DISABLE_JIT`` once for every step. The variable is
checked explicitly rather than inferred from whether compilation happened:
numba's own default is JIT on, but ``tests/conftest.py`` turns it off unless
the value is set, so a guard that only looked at the compiled signatures would
pass with the variable absent while the suite ran entirely interpreted.

Compiling here also fills numba's on-disk cache next to the module, so the
parallel test workers started afterwards load the kernel instead of each
compiling its own copy.
"""

from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np

from phonometry.metrology import parametric_filters as pf
from phonometry.metrology.parametric_filters import time_weighting


def main() -> int:
    if importlib.util.find_spec("numba") is None:
        print("FAIL: numba is not installed, the kernel is the interpreted fallback")
        return 1

    disable_jit = os.environ.get("NUMBA_DISABLE_JIT")
    if disable_jit != "0":
        print(
            f"FAIL: NUMBA_DISABLE_JIT is {disable_jit!r}, not '0'; "
            "tests/conftest.py leaves the JIT off at any other value and the "
            "suite would run the interpreted kernel"
        )
        return 1

    rng = np.random.default_rng(0)
    # Both shapes the library reaches the kernel with: a single channel (scalar
    # initial state) and a multi-channel block (array initial state). Each is a
    # separate numba signature, so both have to be compiled to be cached.
    time_weighting(rng.standard_normal(256), 48_000, mode="impulse")
    time_weighting(rng.standard_normal((2, 256)), 48_000, mode="impulse")

    signatures = getattr(pf._apply_impulse_kernel, "nopython_signatures", [])
    if not signatures:
        print(
            "FAIL: the kernel produced no compiled signature with numba "
            "installed and the JIT on; the decorator is no longer being "
            "applied and the job would exercise the interpreted path"
        )
        return 1

    print(f"OK: {len(signatures)} compiled signature(s) for the impulse kernel")
    for sig in signatures:
        print(f"  {sig}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
