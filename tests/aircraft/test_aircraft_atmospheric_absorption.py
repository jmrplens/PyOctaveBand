#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Tests for aircraft one-third-octave-band atmospheric absorption (SAE ARP 5534).

Oracles (independent of the implementation): the ISO 9613-1 pure-tone
coefficient (ARP 5534 §3.1 declares them identical), the SAE-Method piecewise
continuity at 150 dB, and the SAE-Method regression evaluated by hand from the
published constants.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pytest

from phonometry.aircraft.atmospheric_absorption import (
    AircraftBandAttenuation,
    sae_band_attenuation,
)
from phonometry.environmental.air_absorption import air_attenuation

# Published SAE-Method constants (ARP 5534 §3.2.2), for the independent oracle.
_A, _B, _C, _D, _E, _F, _G = 0.867942, 0.111761, 0.95824, 0.008191, 1.6, 9.2, 0.765


def _sae_oracle(delta_t: float) -> float:
    if delta_t < 150.0:
        return _A * delta_t * (1.0 + _B * (_C - _D * delta_t)) ** _E
    return _F + _G * delta_t


def test_coefficient_is_iso9613_at_exact_midband() -> None:
    f = np.array([50.0, 500.0, 1000.0, 8000.0])
    res = sae_band_attenuation(f, 500.0, temperature=15.0, relative_humidity=60.0)
    assert isinstance(res, AircraftBandAttenuation)
    expected = air_attenuation(f, 15.0, 60.0, 101.325, exact_midband=True)
    assert np.allclose(res.coefficient, expected)
    assert np.allclose(res.midband_attenuation, expected * 500.0)


def test_band_attenuation_matches_regression_oracle() -> None:
    f = np.array([100.0, 1000.0, 4000.0, 8000.0])
    res = sae_band_attenuation(f, 2000.0, temperature=25.0, relative_humidity=70.0)
    for i in range(f.size):
        assert res.band_attenuation[i] == pytest.approx(
            _sae_oracle(float(res.midband_attenuation[i])), abs=1e-9
        )


def test_piecewise_continuity_at_150_db() -> None:
    # The two SAE-Method branches meet at δ_t = 150 dB (rounding of the published
    # constants leaves a ~0.003 dB gap).
    lo = _A * 150.0 * (1.0 + _B * (_C - _D * 150.0)) ** _E
    hi = _F + _G * 150.0
    assert abs(lo - hi) < 0.01


def test_small_absorption_band_close_to_pure_tone() -> None:
    # For small δ_t the band attenuation is within a few percent of pure-tone.
    res = sae_band_attenuation([200.0], 100.0, temperature=25.0, relative_humidity=70.0)
    ratio = float(res.band_attenuation[0] / res.midband_attenuation[0])
    assert 1.0 <= ratio < 1.05


def test_zero_path_length_gives_zero() -> None:
    res = sae_band_attenuation([1000.0, 4000.0], 0.0)
    assert np.allclose(res.band_attenuation, 0.0)
    assert np.allclose(res.midband_attenuation, 0.0)


def test_attenuation_increases_with_frequency_and_distance() -> None:
    near = sae_band_attenuation([2000.0], 500.0, temperature=25.0, relative_humidity=70.0)
    far = sae_band_attenuation([2000.0], 5000.0, temperature=25.0, relative_humidity=70.0)
    assert far.band_attenuation[0] > near.band_attenuation[0]
    spec = sae_band_attenuation([500.0, 8000.0], 2000.0)
    assert spec.band_attenuation[1] > spec.band_attenuation[0]


def test_large_attenuation_no_nan_or_warning() -> None:
    # Very long path at high frequency drives δ_t far past the 150 dB split; the
    # result must stay finite (the discarded low branch must not emit NaN).
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        res = sae_band_attenuation([8000.0, 10000.0], 50_000.0,
                                   temperature=25.0, relative_humidity=70.0)
    assert np.all(np.isfinite(res.band_attenuation))
    # Above 150 dB the linear branch applies: δ_B = 9.2 + 0.765·δ_t.
    assert np.all(res.band_attenuation == pytest.approx(9.2 + 0.765 * res.midband_attenuation))


def test_invalid_inputs_rejected() -> None:
    with pytest.raises(ValueError, match="frequencies"):
        sae_band_attenuation([-1.0], 100.0)
    with pytest.raises(ValueError, match="path_length"):
        sae_band_attenuation([1000.0], -5.0)


def test_plot_smoke() -> None:
    res = sae_band_attenuation(np.array([50.0, 100.0, 500.0, 1000.0, 5000.0, 10000.0]), 3000.0)
    assert res.plot() is not None
