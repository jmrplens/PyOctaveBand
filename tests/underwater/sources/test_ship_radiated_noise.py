#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Tests for ship radiated noise and monopole source level (ISO 17208-1/-2).

The radiated noise level and the Lloyd's-mirror surface correction are exact
closed forms; the correction is checked directly against Formula 3 and its
high-frequency limit.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pytest

from phonometry import underwater
from phonometry.underwater.sources.ship_radiated_noise import _surface_correction


def _delta_l(u: float) -> float:
    return -10.0 * np.log10((2 * u**4 + 14 * u**2) / (14 + 2 * u**2 + u**4))


@pytest.mark.parametrize(
    ("u", "expected_delta_l"),
    [
        # Hand-computed independently of the implementation:
        #   u = 1: (2 + 14) / (14 + 2 + 1) = 16/17 -> -10 lg(16/17) = +0.2633 dB
        (1.0, 0.263289387),
        #   u = 2: (2·16 + 14·4) / (14 + 8 + 16) = 88/38 -> -10 lg(88/38) = -3.6470 dB
        (2.0, -3.646990755),
    ],
)
def test_surface_correction_independent_anchors(u: float, expected_delta_l: float) -> None:
    # With c = 2π and d_s = 1 m, u = k·d_s = (2πf/c)·1 = f, so f = u sets u exactly.
    delta_l = _surface_correction(np.array([u]), source_depth=1.0, sound_speed=2.0 * np.pi)
    assert float(delta_l[0]) == pytest.approx(expected_delta_l, abs=1e-7)


def test_radiated_noise_level_closed_form() -> None:
    # p = 2 µPa, r = 100 m -> 20 lg(2) + 20 lg(100) = 6.0206 + 40.
    lrn = underwater.radiated_noise_level(2e-6, 100.0)
    assert lrn == pytest.approx(20 * np.log10(2.0) + 40.0, rel=1e-9)


def test_radiated_noise_level_rejects_bad_input() -> None:
    with pytest.raises(ValueError):
        underwater.radiated_noise_level(0.0, 100.0)
    with pytest.raises(ValueError):
        underwater.radiated_noise_level(1e-6, -1.0)


def test_surface_correction_matches_formula() -> None:
    # d_s = 0.7*10 = 7 m, c = 1500. At f = 200 Hz, u = k*d_s.
    draught, c, f = 10.0, 1500.0, 200.0
    ds = 0.7 * draught
    u = 2 * np.pi * f / c * ds
    res = underwater.monopole_source_level(120.0, f, draught, c=c)
    assert res.source_depth == pytest.approx(ds)
    assert float(res.surface_correction[0]) == pytest.approx(_delta_l(u), rel=1e-9)
    assert float(res.source_level[0]) == pytest.approx(120.0 + _delta_l(u), rel=1e-9)


def test_surface_correction_high_frequency_limit() -> None:
    # As u -> infinity, ratio -> 2, so ΔL -> -10 lg(2) = -3.0103 dB.
    res = underwater.monopole_source_level(100.0, 1e6, 20.0, c=1500.0)
    assert float(res.surface_correction[0]) == pytest.approx(-3.010299957, abs=1e-3)


def test_monopole_source_level_array_broadcast() -> None:
    freqs = np.array([100.0, 500.0, 2000.0])
    rnl = np.array([150.0, 145.0, 140.0])
    res = underwater.monopole_source_level(rnl, freqs, 8.0)
    assert isinstance(res, underwater.ShipSourceLevelResult)
    assert res.source_level.shape == freqs.shape
    assert np.allclose(res.source_level, rnl + res.surface_correction)


def test_scalar_rnl_broadcasts_over_frequencies() -> None:
    freqs = np.array([100.0, 1000.0])
    res = underwater.monopole_source_level(130.0, freqs, 6.0)
    assert res.radiated_noise_level.shape == freqs.shape
    assert np.all(res.radiated_noise_level == 130.0)


def test_hydrophone_depths() -> None:
    depths = underwater.hydrophone_depths(100.0)
    expected = 100.0 * np.tan(np.radians([15.0, 30.0, 45.0]))
    assert np.allclose(depths, expected)
    assert float(depths[-1]) == pytest.approx(100.0)  # 45 deg -> depth = cpa


def test_hydrophone_depths_rejects_bad_angles() -> None:
    with pytest.raises(ValueError):
        underwater.hydrophone_depths(100.0, angles=(90.0,))


def test_hydrophone_depths_rejects_non_finite_angles() -> None:
    # NaN/inf angles must be rejected, not silently yield NaN depths.
    with pytest.raises(ValueError):
        underwater.hydrophone_depths(100.0, angles=(np.nan,))
    with pytest.raises(ValueError):
        underwater.hydrophone_depths(100.0, angles=(np.inf,))


def test_source_level_uncertainty_bands() -> None:
    assert underwater.source_level_uncertainty(63.0) == 5.0
    assert underwater.source_level_uncertainty(100.0) == 5.0
    assert underwater.source_level_uncertainty(1000.0) == 3.0
    assert underwater.source_level_uncertainty(16000.0) == 3.0
    assert underwater.source_level_uncertainty(20000.0) == 4.0


def test_result_plot_smoke() -> None:
    res = underwater.monopole_source_level(
        np.array([140.0, 135.0]), np.array([125.0, 500.0]), 9.0
    )
    ax = res.plot()
    assert ax is not None


def test_rejects_shape_mismatch() -> None:
    levels_2_bands = np.array([1.0, 2.0])
    frequencies_3_bands = np.array([100.0, 200.0, 300.0])
    with pytest.raises(ValueError):
        underwater.monopole_source_level(
            levels_2_bands, frequencies_3_bands, 8.0
        )
