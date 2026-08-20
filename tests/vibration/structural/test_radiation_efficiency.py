#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Tests for the plate radiation efficiency of Hopkins Section 2.9.4.

Anchored on: the closed-form coincidence frequency (Hopkins Eq. 2.201 / Bies
Eq. 7.3), the exact above-coincidence limit ``sigma = (1 - fc/f)**-0.5``
(Eq. 2.229) tending to 1 well above ``fc``, and the digitized method-no.-1
curve of Fig. 2.65(a) for a 6 mm glass pane (Lx = 1.5 m, Ly = 1.25 m,
fc ~ 2079 Hz) read off the log axes to about +/- 20 %.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from phonometry import vibration

# 6 mm glass (Hopkins Fig. 4.8 / 2.65a declared parameters).
E, H, NU, RHO = 6.2e10, 0.006, 0.24, 2500.0
BP = vibration.plate_bending_stiffness(E, H, NU)
M2 = RHO * H
LX, LY = 1.5, 1.25


# ---------------------------------------------------------------------------
# Coincidence frequency (Hopkins Eq. 2.201).
# ---------------------------------------------------------------------------
def test_coincidence_frequency_closed_form() -> None:
    fc = vibration.coincidence_frequency(M2, BP)
    assert fc == pytest.approx(343.0**2 / (2.0 * math.pi) * math.sqrt(M2 / BP))


def test_coincidence_frequency_glass_near_2079() -> None:
    # Hopkins declares fc = 2079 Hz for this pane; the closed form gives ~2107.
    assert vibration.coincidence_frequency(M2, BP) == pytest.approx(2079.0, rel=0.02)


# ---------------------------------------------------------------------------
# Above coincidence (Hopkins Eq. 2.229): sigma = (1 - fc/f)**-0.5 -> 1.
# ---------------------------------------------------------------------------
def test_above_coincidence_closed_form() -> None:
    fc = 2000.0
    f = np.array([4000.0, 8000.0])
    res = vibration.radiation_efficiency(f, LX, LY, fc)
    expected = 1.0 / np.sqrt(1.0 - fc / f)
    np.testing.assert_allclose(res.radiation_efficiency, expected, rtol=1e-12)


def test_tends_to_unity_far_above_coincidence() -> None:
    res = vibration.radiation_efficiency([50000.0], LX, LY, 2000.0)
    assert res.radiation_efficiency[0] == pytest.approx(1.0, abs=0.03)


def test_at_coincidence_peak_finite() -> None:
    # The band containing fc uses Eq. 2.230; for this pane the peak is ~2.5.
    fc = vibration.coincidence_frequency(M2, BP)
    res = vibration.radiation_efficiency([2000.0], LX, LY, fc)
    assert res.radiation_efficiency[0] == pytest.approx(2.5, rel=0.15)


# ---------------------------------------------------------------------------
# Digitized method-1 curve (Hopkins Fig. 2.65a, 6 mm glass), +/- 25 %.
# ---------------------------------------------------------------------------
def test_below_coincidence_matches_fig_265a() -> None:
    fc = vibration.coincidence_frequency(M2, BP)
    freqs = np.array([125.0, 200.0, 500.0, 800.0, 1250.0])
    read_off = np.array([0.013, 0.02, 0.04, 0.06, 0.11])
    res = vibration.radiation_efficiency(freqs, LX, LY, fc)
    np.testing.assert_allclose(res.radiation_efficiency, read_off, rtol=0.3)


def test_radiation_index_is_ten_log_sigma() -> None:
    res = vibration.radiation_efficiency([500.0], LX, LY, 2000.0)
    assert res.radiation_index[0] == pytest.approx(
        10.0 * math.log10(res.radiation_efficiency[0])
    )


def test_clamped_boundary_radiates_more() -> None:
    fc = vibration.coincidence_frequency(M2, BP)
    ss = vibration.radiation_efficiency(
        [500.0], LX, LY, fc, boundary="simply_supported"
    )
    cl = vibration.radiation_efficiency([500.0], LX, LY, fc, boundary="clamped")
    assert cl.radiation_efficiency[0] > ss.radiation_efficiency[0]


# ---------------------------------------------------------------------------
# Validation.
# ---------------------------------------------------------------------------
def test_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="boundary"):
        vibration.radiation_efficiency([500.0], LX, LY, 2000.0, boundary="pinned")
    with pytest.raises(ValueError, match="critical_frequency"):
        vibration.radiation_efficiency([500.0], LX, LY, -1.0)
    with pytest.raises(ValueError, match="frequency"):
        vibration.radiation_efficiency([-1.0], LX, LY, 2000.0)
