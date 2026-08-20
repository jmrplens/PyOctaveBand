#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Tests for slit / aperture transmission (Hopkins Section 4.3.10, Gomperts).

Anchored on: the composite energy sum (Hopkins Eq. 4.92) whose area limit caps a
partition with a bare opening of relative area ``Sa/S`` at ``10 lg(S/Sa)``; the
slit transmission maxima at the resonances ``d + 2e = z lambda/2`` (Eq. 4.101);
and the large-aperture limit ``tau -> 1`` of the circular-hole model
(Eq. 4.102).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from phonometry import building


# ---------------------------------------------------------------------------
# Composite energy sum (Hopkins Eq. 4.92).
# ---------------------------------------------------------------------------
def test_composite_open_area_caps_reduction() -> None:
    # A 1 % bare opening caps the composite R at 10 lg(100) = 20 dB, whatever
    # the wall's own R.
    r = building.composite_transmission_loss([0.99, 0.01], [60.0, 0.0])
    assert float(r) == pytest.approx(10.0 * math.log10(1.0 / 0.01), abs=0.1)


def test_composite_cap_independent_of_wall() -> None:
    a = building.composite_transmission_loss([0.99, 0.01], [50.0, 0.0])
    b = building.composite_transmission_loss([0.99, 0.01], [90.0, 0.0])
    assert float(a) == pytest.approx(float(b), abs=0.1)


def test_composite_all_equal_returns_same() -> None:
    r = building.composite_transmission_loss([1.0, 2.0, 3.0], [40.0, 40.0, 40.0])
    assert float(r) == pytest.approx(40.0)


def test_composite_2d_per_band() -> None:
    r = building.composite_transmission_loss(
        [10.0, 0.1], np.array([[40.0, 50.0], [0.0, 0.0]])
    )
    assert r.shape == (2,)
    # A fixed open area gives the same cap in both bands.
    np.testing.assert_allclose(r, 10.0 * math.log10((10.1) / 0.1), atol=0.05)


def test_composite_matches_facade_energy_sum() -> None:
    # Same energetic combination as the EN 12354-3 facade model.
    areas = np.array([8.0, 2.0])
    r = np.array([55.0, 30.0])
    tau = 10.0 ** (-r / 10.0)
    expected = -10.0 * math.log10(np.sum(areas * tau) / np.sum(areas))
    assert float(building.composite_transmission_loss(areas, r)) == pytest.approx(
        expected
    )


def test_composite_rejects_bad_shape() -> None:
    with pytest.raises(ValueError, match="length must match"):
        building.composite_transmission_loss([1.0, 2.0], [40.0])


# ---------------------------------------------------------------------------
# Slit transmission (Hopkins Eq. 4.99 / 4.101).
# ---------------------------------------------------------------------------
def test_slit_transmission_peaks_at_first_resonance() -> None:
    w, d = 0.005, 0.1
    fr = building.slit_resonance_frequencies(d, w, orders=1)
    f = np.linspace(fr[0] - 300.0, fr[0] + 300.0, 601)
    res = building.slit_transmission_coefficient(f, w, d, field="normal")
    peak = f[int(np.argmax(res.transmission_coefficient))]
    assert peak == pytest.approx(fr[0], abs=15.0)


def test_slit_resonances_are_half_wavelength_spaced() -> None:
    # d + 2e = z lambda/2, so the orders are near-integer multiples.
    fr = building.slit_resonance_frequencies(0.1, 0.005, orders=3)
    assert fr[1] / fr[0] == pytest.approx(2.0, rel=0.05)
    assert fr[2] / fr[0] == pytest.approx(3.0, rel=0.05)


def test_slit_resonance_rejects_wide_shallow_slit() -> None:
    # A very wide, shallow slit drives the effective depth d + 2e non-positive;
    # the resonance is undefined and must raise rather than return NaN.
    with pytest.raises(ValueError, match="too wide"):
        building.slit_resonance_frequencies(0.001, 0.2, orders=1)


def test_slit_transmission_finite_through_cos_ke_zero() -> None:
    # The reformulated Eq. 4.99 (no division by cos(Ke)) stays finite where
    # cos(Ke) crosses zero.
    import warnings

    f = np.linspace(50.0, 10000.0, 8000)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        tau = building.slit_transmission_coefficient(
            f, 0.01, 0.02
        ).transmission_coefficient
    assert np.all(np.isfinite(tau))


def test_slit_reduction_index_from_coefficient() -> None:
    res = building.slit_transmission_coefficient([500.0], 0.002, 0.1)
    r = building.transmission_loss_from_coefficient(res.transmission_coefficient)
    np.testing.assert_allclose(r, res.transmission_loss)


def test_slit_edge_position_differs_from_mid() -> None:
    mid = building.slit_transmission_coefficient([400.0], 0.003, 0.05, position="mid")
    edge = building.slit_transmission_coefficient([400.0], 0.003, 0.05, position="edge")
    assert not np.isclose(
        mid.transmission_coefficient[0], edge.transmission_coefficient[0]
    )


# ---------------------------------------------------------------------------
# Circular aperture (Hopkins Eq. 4.102).
# ---------------------------------------------------------------------------
def test_circular_aperture_large_hole_transmits_fully() -> None:
    # ka >> 1: tau -> 1 (R -> 0 dB).
    res = building.circular_aperture_transmission_coefficient([50000.0], 0.02, 0.001)
    assert res.transmission_coefficient[0] == pytest.approx(1.0, abs=0.02)


def test_circular_aperture_thin_hole_reduction_positive_at_low_f() -> None:
    res = building.circular_aperture_transmission_coefficient([100.0], 0.005, 0.002)
    assert float(res.transmission_loss[0]) > 0.0


# ---------------------------------------------------------------------------
# Validation.
# ---------------------------------------------------------------------------
def test_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="field"):
        building.slit_transmission_coefficient([500.0], 0.002, 0.1, field="oblique")
    with pytest.raises(ValueError, match="width"):
        building.slit_transmission_coefficient([500.0], -0.002, 0.1)
    with pytest.raises(ValueError, match="tau"):
        building.transmission_loss_from_coefficient([-1.0])
