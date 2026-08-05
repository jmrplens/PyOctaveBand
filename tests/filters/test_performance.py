#  Copyright (c) 2026. Jose Manuel Requena Plens
"""
Tests that the filter-bank design work is reused rather than repeated.
"""

import time
from typing import Any

import numpy as np
import pytest

import phonometry
from phonometry import OctaveFilterBank, octave_filter
from phonometry.filters import core


class _DesignCounter:
    """Wraps ``_design_sos_filter`` and counts how often it is invoked."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.calls = 0

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.calls += 1
        return self._inner(*args, **kwargs)


@pytest.fixture
def design_counter(monkeypatch: pytest.MonkeyPatch) -> _DesignCounter:
    """Count filter designs, and start from a cold design cache."""
    counter = _DesignCounter(core._design_sos_filter)
    monkeypatch.setattr(core, "_design_sos_filter", counter)
    core._cached_filter_bank.cache_clear()
    return counter


def test_filterbank_reuse_skips_redesign(design_counter: _DesignCounter) -> None:
    """
    Verify that reusing OctaveFilterBank designs its coefficients exactly once.

    **Purpose:**
    The class-based API exists so that the SOS design runs at construction and
    every subsequent ``filter()`` call skips it. That is a property of the code
    path, so it is asserted as one: the number of times the design routine runs.
    An earlier version of this test compared ``time.time()`` deltas instead and
    asserted the class path stayed within 1.5x of the functional one. Under a
    parallel suite on a shared runner that measures the scheduler, not the
    filter bank, and it failed on CI legs while passing locally.

    **Verification:**
    - Ten functional calls with the design cache cleared each time re-design the
      bank on every call.
    - One ``OctaveFilterBank`` plus ten ``filter()`` calls design exactly once.
    - Both paths return the same band levels, so the reuse is not a shortcut
      around the work.
    """
    fs = 48000
    rng = np.random.default_rng(42)
    x = rng.standard_normal(int(fs * 0.5))
    num_iterations = 10

    # 1. Functional API with a cold design cache on every call: octave_filter()
    #    caches bank designs, so clearing it isolates the redesign cost.
    for _ in range(num_iterations):
        core._cached_filter_bank.cache_clear()
        spl_func, freq_func = octave_filter(x, fs)
    assert design_counter.calls == num_iterations

    # 2. The class designs once at construction and never again.
    design_counter.calls = 0
    bank = OctaveFilterBank(fs)
    assert design_counter.calls == 1
    for _ in range(num_iterations):
        spl_class, freq_class = bank.filter(x)
    assert design_counter.calls == 1, "filter() must not re-design the bank"

    # 3. Reuse is not a shortcut: the two paths agree band for band.
    assert freq_class == freq_func
    np.testing.assert_allclose(spl_class, spl_func)


def test_octave_filter_cache_reuses_the_bank(design_counter: _DesignCounter) -> None:
    """
    Verify that repeated ``octave_filter`` calls share one design.

    **Purpose:**
    ``_cached_filter_bank`` is what makes the functional API affordable in a
    loop. Without it every call would pay a full design, which is the cost the
    class-based API was introduced to avoid.

    **Verification:**
    Five identical calls, cache left warm, must design the bank once; a call
    with a different sample rate is a different bank and designs again.
    """
    fs = 48000
    rng = np.random.default_rng(7)
    x = rng.standard_normal(fs // 4)

    for _ in range(5):
        octave_filter(x, fs)
    assert design_counter.calls == 1

    octave_filter(x[: fs // 8], 24000)
    assert design_counter.calls == 2


def test_filterbank_reuse_is_not_slower() -> None:
    """
    Report the wall-clock cost of both paths, without gating on it.

    **Purpose:**
    The timings are what a reader of this file wants to see, and they are worth
    printing on every run; they are not worth failing a build over, because a
    shared runner under a parallel suite can stretch either path arbitrarily.
    The gate is the design count in the tests above. This one only fails if the
    class path is more than an order of magnitude slower than the functional
    one, which no scheduling noise explains and which would mean the reuse has
    stopped being a saving at all.
    """
    fs = 48000
    rng = np.random.default_rng(42)
    x = rng.standard_normal(int(fs * 0.5))
    num_iterations = 10

    start_func = time.time()
    for _ in range(num_iterations):
        phonometry.filters.core._cached_filter_bank.cache_clear()
        octave_filter(x, fs)
    time_func = time.time() - start_func

    start_class_init = time.time()
    bank = OctaveFilterBank(fs)
    time_class_init = time.time() - start_class_init

    start_class_filter = time.time()
    for _ in range(num_iterations):
        bank.filter(x)
    time_class_filter = time.time() - start_class_filter

    print(f"\nFunctional API Time: {time_func:.4f}s")
    print(f"Class Init Time: {time_class_init:.4f}s")
    print(f"Class Filter Only Time: {time_class_filter:.4f}s")
    print(f"Class Total Time: {time_class_init + time_class_filter:.4f}s")

    assert time_class_filter < time_func * 10.0
