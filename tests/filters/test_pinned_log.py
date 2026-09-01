#  Copyright (c) 2026. Jose Manuel Requena Plens
"""The pinned logarithm returns :func:`math.log`'s own bits.

``_pinned_log`` exists so the weighting fit can take its logarithms at numpy
speed without moving a single shipped coefficient, and that claim is only as
good as the bit-for-bit agreement with the elementwise :func:`math.log` it
replaced. These tests pin a deterministic slice of the validation that
established it -- full exponent range, the near-one window and its edges,
subnormals, the specials -- and, separately, the inputs one actual design
evaluates, so the agreement is checked on the path that motivates the module
and not only on synthetic draws.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from phonometry.filters import _weighting_design
from phonometry.filters._pinned_log import _fused_multiply_add, pinned_log
from phonometry.filters.weighting import _analog_weighting_zpk, _fit_band


def _reference(values: np.ndarray) -> np.ndarray:
    return np.fromiter(map(math.log, values.tolist()), np.float64, values.size)


def _assert_same_bits(values: np.ndarray) -> None:
    mine = pinned_log(values)
    reference = _reference(values)
    mismatched = mine.view(np.uint64) != reference.view(np.uint64)
    assert not mismatched.any(), values[mismatched][:5]


def test_full_range_bits_match() -> None:
    """Random bit patterns over every finite positive binade."""
    rng = np.random.default_rng(0)
    bits = rng.integers(1, 0x7FF0000000000000, size=200_000, dtype=np.uint64)
    _assert_same_bits(bits.view(np.float64))


def test_near_one_window_bits_match() -> None:
    """The separately-polynomialised window, where the fused sites live."""
    rng = np.random.default_rng(1)
    window = 1.0 + rng.uniform(-(2.0**-4), float.fromhex("0x1.09p-4"), 200_000)
    _assert_same_bits(window)


def test_window_edges_and_anchors_match() -> None:
    """The branch boundaries themselves, one neighbour either side."""
    low = 1.0 - 2.0**-4
    high = 1.0 + float.fromhex("0x1.09p-4")
    eps = np.finfo(np.float64).eps
    edges = np.array(
        [
            low,
            np.nextafter(low, 0.0),
            np.nextafter(low, 2.0),
            high,
            np.nextafter(high, 0.0),
            np.nextafter(high, 2.0),
            1.0 - eps / 2.0,
            1.0 + eps,
            0.5,
            2.0,
            4.0 / 3.0,
            np.finfo(np.float64).tiny,
            np.finfo(np.float64).max,
        ]
    )
    _assert_same_bits(edges)


def test_subnormal_bits_match() -> None:
    """The normalisation branch, which no design input reaches."""
    rng = np.random.default_rng(2)
    bits = rng.integers(1, 1 << 52, size=50_000, dtype=np.uint64)
    _assert_same_bits(bits.view(np.float64))


def test_specials_follow_math_log() -> None:
    """``inf`` and ``nan`` propagate; zero and negatives raise."""
    assert pinned_log(np.array([np.inf]))[0] == np.inf
    assert math.isnan(pinned_log(np.array([np.nan]))[0])
    assert pinned_log(np.array([1.0]))[0] == 0.0
    for bad in (0.0, -1.0, -np.inf):
        with pytest.raises(ValueError, match="math domain error"):
            pinned_log(np.array([1.0, bad]))


def test_shape_and_empty_are_preserved() -> None:
    """The elementwise contract: any shape in, the same shape out."""
    square = np.array([[1.5, 2.5], [3.5, 4.5]])
    assert pinned_log(square).shape == (2, 2)
    assert pinned_log(np.empty(0)).shape == (0,)
    np.testing.assert_array_equal(
        pinned_log(square).reshape(-1), pinned_log(square.reshape(-1))
    )


def test_fused_multiply_add_matches_math_fma() -> None:
    """The emulation behind the near-one window's four fused sites."""
    rng = np.random.default_rng(3)
    n = 100_000
    r = rng.uniform(-(2.0**-4), 2.0**-4, n)
    cases = (
        (r, np.full(n, 0.375), np.full(n, -0.5)),
        (r * r, np.full(n, -0.25), rng.uniform(-0.4, -0.25, n)),
        (r**3, rng.uniform(-0.4, -0.25, n), rng.uniform(-1e-17, 1e-17, n)),
        (rng.uniform(-1, 1, n), rng.uniform(-1, 1, n), rng.uniform(-1, 1, n)),
    )
    for a, b, c in cases:
        mine = _fused_multiply_add(a, b, c)
        reference = np.fromiter(
            map(math.fma, a.tolist(), b.tolist(), c.tolist()), np.float64, n
        )
        np.testing.assert_array_equal(mine.view(np.uint64), reference.view(np.uint64))


def test_every_log_input_of_a_real_design_matches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The agreement holds on the inputs the fit actually produces.

    A spy captures every array the design hands ``_scalar_log`` and the test
    replays each element through :func:`math.log`. This is the property the
    module exists for: the fit's own universe of inputs, not a synthetic
    distribution, returns identical bits, so the design that ships is the
    design that shipped before the loop was retired.
    """
    captured: list[np.ndarray] = []
    original = _weighting_design._scalar_log

    def spy(values: np.ndarray) -> np.ndarray:
        captured.append(np.asarray(values, dtype=np.float64).reshape(-1).copy())
        return original(values)

    monkeypatch.setattr(_weighting_design, "_scalar_log", spy)
    zeros, poles, _ = _analog_weighting_zpk("A")
    band = _fit_band("A", 48000)
    _weighting_design.design_sos(zeros, poles, 48000.0, band, 4, 1000.0)
    assert captured, "the design evaluated no logarithm"
    everything = np.unique(np.concatenate(captured))
    _assert_same_bits(everything)
