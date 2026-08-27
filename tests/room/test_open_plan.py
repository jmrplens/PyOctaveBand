#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Tests for ISO 3382-3:2012 open-plan-office spatial metrics.

Validation strategy: exact analytic constructions (not self-consistency).

- D2,S (Clause 6.2, Eq.5) is the least-squares slope of the A-weighted
  speech level against lg(r/r0) scaled per distance doubling. For a field
  L(r) = L0 - k*log2(r) the fit is exact and D2,S = k, while the 4 m
  read-off Lp,A,S,4m = L0 - k*log2(4) = L0 - 2*k.
- rD / rP (Clause 6.3) are the crossings of the linear STI-vs-distance
  regression line with 0.50 and 0.20. For a field STI(r) = c + d*r the fit
  is exact and the crossings solve c + d*r = threshold.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from phonometry import room


def test_d2s_minus_6db_per_doubling_exact() -> None:
    """-6 dB per distance doubling -> D2,S = 6.0 exactly (Clause 6.2)."""
    r = np.array([2.0, 4.0, 8.0, 16.0])
    lp = 70.0 - 6.0 * np.log2(r)  # L0 = 70 dB, 6 dB / doubling
    sti = 0.6 - 0.02 * r
    res = room.open_plan_metrics(r, lp, sti)
    assert isinstance(res, room.OpenPlanResult)
    assert res.d2s == pytest.approx(6.0, abs=1e-9)
    # Lp,A,S,4m read off the regression line at r = 4 m.
    assert res.lp_as_4m == pytest.approx(58.0, abs=1e-9)


def test_lp_as_4m_recovered_exactly() -> None:
    """Lp,A,S,4m equals the regression value at 4 m (Clause 3.3, 6.2)."""
    r = np.array([2.0, 4.0, 8.0, 16.0])
    lp = 65.0 - 4.0 * np.log2(r)
    sti = np.array([0.55, 0.45, 0.35, 0.25])
    res = room.open_plan_metrics(r, lp, sti)
    assert res.d2s == pytest.approx(4.0, abs=1e-9)
    assert res.lp_as_4m == pytest.approx(65.0 - 8.0, abs=1e-9)  # 57 dB


def test_d2s_ignores_positions_outside_2_to_16_m() -> None:
    """Only 2-16 m positions feed D2,S / Lp,A,S,4m (Clause 5.2.2, 6.2)."""
    r = np.array([1.0, 2.0, 4.0, 8.0, 16.0, 20.0])
    lp = 70.0 - 6.0 * np.log2(r)
    lp = lp.copy()
    lp[0] = 1000.0  # r = 1 m, out of range: must be ignored
    lp[-1] = -1000.0  # r = 20 m, out of range: must be ignored
    sti = np.linspace(0.6, 0.1, r.size)
    res = room.open_plan_metrics(r, lp, sti)
    assert res.d2s == pytest.approx(6.0, abs=1e-9)
    assert res.lp_as_4m == pytest.approx(58.0, abs=1e-9)


def test_rd_within_range_rp_extrapolated() -> None:
    """rD / rP are exact regression-line crossings, rP extrapolated."""
    r = np.array([2.0, 5.0, 10.0, 15.0])
    sti = 0.6 - 0.02 * r  # STI = 0.6 - 0.02 r
    lp = 70.0 - 6.0 * np.log2(r)
    res = room.open_plan_metrics(r, lp, sti)
    # 0.50 = 0.6 - 0.02 r -> r = 5 m (within range)
    assert res.rd == pytest.approx(5.0, abs=1e-9)
    # 0.20 = 0.6 - 0.02 r -> r = 20 m (extrapolated beyond 15 m)
    assert res.rp == pytest.approx(20.0, abs=1e-9)


def test_rd_nonpositive_returns_nan_rp_valid() -> None:
    """STI already below 0.50 at the source -> rD NaN, rP still solved."""
    r = np.array([2.0, 4.0, 6.0, 8.0])
    sti = 0.4 - 0.02 * r  # crosses 0.50 only at negative distance
    lp = 60.0 - 3.0 * np.log2(r)
    res = room.open_plan_metrics(r, lp, sti)
    assert math.isnan(res.rd)
    # 0.20 = 0.4 - 0.02 r -> r = 10 m
    assert res.rp == pytest.approx(10.0, abs=1e-9)


def test_sti_non_decreasing_returns_nan() -> None:
    """STI not decreasing with distance -> rD, rP undeterminable (NaN)."""
    r = np.array([2.0, 4.0, 8.0, 16.0])
    sti = np.array([0.3, 0.4, 0.5, 0.6])  # increases with distance
    lp = 70.0 - 6.0 * np.log2(r)
    res = room.open_plan_metrics(r, lp, sti)
    assert math.isnan(res.rd)
    assert math.isnan(res.rp)


def test_too_few_positions_raises() -> None:
    """Fewer than 4 positions is a violation of Clause 5.2.2."""
    r = np.array([2.0, 4.0, 8.0])
    lp = 70.0 - 6.0 * np.log2(r)
    sti = np.array([0.6, 0.4, 0.2])
    with pytest.raises(
        ValueError, match=r"ISO 3382-3 requires at least \d+ measurement positions"
    ):
        room.open_plan_metrics(r, lp, sti)


def test_mismatched_lengths_raise() -> None:
    """Position / level / STI arrays must have equal shape."""
    r = np.array([2.0, 4.0, 8.0, 16.0])
    lp = 70.0 - 6.0 * np.log2(r)
    sti = np.array([0.6, 0.4, 0.2])  # one short
    with pytest.raises(
        ValueError,
        match=r"open_plan_metrics: .*'sti_values'.*must all have the same shape",
    ):
        room.open_plan_metrics(r, lp, sti)


def test_too_few_in_range_gives_nan_decay() -> None:
    """<2 positions in 2-16 m -> D2,S / Lp,A,S,4m NaN, rD/rP still work."""
    r = np.array([0.5, 1.0, 1.5, 18.0])  # all below 2 m or above 16 m: none in range
    lp = np.array([60.0, 58.0, 56.0, 40.0])
    sti = 0.6 - 0.02 * r
    res = room.open_plan_metrics(r, lp, sti)
    assert math.isnan(res.d2s)
    assert math.isnan(res.lp_as_4m)
    # STI regression uses all positions: 0.5 = 0.6 - 0.02 r -> r = 5
    assert res.rd == pytest.approx(5.0, abs=1e-9)


def test_exactly_one_in_range_gives_nan_decay() -> None:
    """Exactly one 2-16 m position -> D2,S / Lp,A,S,4m NaN (need >=2 to fit).

    A single in-range point cannot define a decay slope, so the module
    degrades gracefully to NaN (as documented) rather than crashing, while
    the distance-based STI regression still resolves rD.
    """
    r = np.array([0.5, 1.0, 4.0, 18.0])  # only 4 m lies in [2, 16] m
    lp = np.array([60.0, 58.0, 50.0, 40.0])
    sti = 0.6 - 0.02 * r
    res = room.open_plan_metrics(r, lp, sti)
    assert math.isnan(res.d2s)
    assert math.isnan(res.lp_as_4m)
    # STI regression uses all positions: 0.5 = 0.6 - 0.02 r -> r = 5 m.
    assert res.rd == pytest.approx(5.0, abs=1e-9)


def test_nan_in_positions_raises() -> None:
    """A NaN position must raise ValueError, not a raw LinAlgError."""
    r = np.array([2.0, 4.0, np.nan, 16.0])
    lp = np.array([70.0, 64.0, 58.0, 46.0])
    sti = np.array([0.6, 0.5, 0.4, 0.3])
    with pytest.raises(
        ValueError, match=r"positions_m, spl_a_speech and sti_values must be finite"
    ):
        room.open_plan_metrics(r, lp, sti)


def test_nan_in_spl_raises() -> None:
    """A NaN speech level must raise ValueError."""
    r = np.array([2.0, 4.0, 8.0, 16.0])
    lp = np.array([70.0, np.nan, 58.0, 46.0])
    sti = np.array([0.6, 0.5, 0.4, 0.3])
    with pytest.raises(
        ValueError, match=r"positions_m, spl_a_speech and sti_values must be finite"
    ):
        room.open_plan_metrics(r, lp, sti)


def test_inf_in_sti_raises() -> None:
    """An Inf STI value must raise ValueError."""
    r = np.array([2.0, 4.0, 8.0, 16.0])
    lp = np.array([70.0, 64.0, 58.0, 46.0])
    sti = np.array([0.6, np.inf, 0.4, 0.3])
    with pytest.raises(
        ValueError, match=r"positions_m, spl_a_speech and sti_values must be finite"
    ):
        room.open_plan_metrics(r, lp, sti)
