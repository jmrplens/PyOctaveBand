#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Golden-baseline regression tests for the overhaul refactors.

Each case recomputes a performance-hotspot scenario (inputs defined once in
``scripts/bench.py``) and asserts numeric equivalence with the frozen arrays in
``tests/golden_data.py``. These tests are the drift guard for the 2026-07
modularization and vectorization work: a pure refactor must keep every value;
an intended numeric change must regenerate the goldens explicitly
(``python scripts/bench.py --golden``) and justify it in review.

Tolerances allow cross-platform last-ULP libm differences and the float
re-association inherent to vectorized reformulations, nothing more. The
long-filter-chain ECMA-418-2 analysers get a slightly looser relative
tolerance for the same reason the figure CI gate is tolerance-based.
"""

from __future__ import annotations

import pathlib
import sys

import matplotlib as mpl

mpl.use("Agg")
import numpy as np
import pytest

_SCRIPTS = str(pathlib.Path(__file__).resolve().parent.parent / "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import bench
from golden_data import GOLDEN

#: Per-case relative tolerance; default is 1e-9 (see module docstring).
_RTOL = {
    "ecma_loudness_0.5s": 1e-7,
    "ecma_roughness_0.5s": 1e-7,
    "ecma_tonality_0.5s": 1e-7,
}


def test_golden_covers_every_bench_case() -> None:
    # The golden file and the bench cases must never drift apart.
    assert set(GOLDEN) == set(bench.CASES)


@pytest.mark.parametrize("case", sorted(bench.CASES))
def test_golden_baseline(case: str) -> None:
    got = np.asarray(bench.CASES[case](), dtype=np.float64).ravel()
    ref = np.asarray(GOLDEN[case], dtype=np.float64)
    assert got.shape == ref.shape
    np.testing.assert_allclose(got, ref, rtol=_RTOL.get(case, 1e-9), atol=1e-9)
