#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Tests for ocean ambient-noise spectrum levels (Wenz framework).

Oracles (independent of the implementation): the wind "rule of fives" anchor
(25 dB re 20 µPa = 51.02 dB re 1 µPa at 1 kHz for 5 knots, −5 dB/octave, +5 dB
per doubling of wind speed), graphical anchors read off the published Wenz
chart (Etter 2003, Figs. 6.2/6.5, axes in dB re 1 µPa), and the Mellen
thermal-noise formula ``<p²> = 4π·k·T·ρ·f²/c`` computed directly.
"""

from __future__ import annotations

import matplotlib as mpl

mpl.use("Agg")
import numpy as np
import pytest

from phonometry.underwater.sources.ambient_noise import (
    AmbientNoiseResult,
    ocean_ambient_noise,
    thermal_noise_spectrum,
    wind_noise_spectrum,
)

_BOLTZMANN = 1.380649e-23
_P_REF = 1e-6


def test_wind_rule_of_fives_anchor() -> None:
    # Wenz/Knudsen: 25 dB re 0.0002 dyn/cm2 (20 uPa) at 1 kHz for 5 knots,
    # i.e. 25 + 20*lg(20) = 51.0206 dB re 1 uPa (ISO 18405 reference).
    expected = 25.0 + 20.0 * np.log10(20.0)
    assert float(wind_noise_spectrum(1000.0, 5.0)[0]) == pytest.approx(
        expected, abs=1e-9
    )


def test_wind_matches_published_wenz_chart() -> None:
    # Graphical anchors from the published Wenz chart (Etter 2003, axes in
    # dB re 1 uPa): ~50 dB at 1 kHz for 4-6 kn (Fig. 6.2) and ~35 dB at
    # 10 kHz for the Knudsen sea-state-1/2 curve (Fig. 6.5, ~5 kn).
    assert float(wind_noise_spectrum(1000.0, 5.0)[0]) == pytest.approx(50.0, abs=3.0)
    assert float(wind_noise_spectrum(10000.0, 5.0)[0]) == pytest.approx(35.0, abs=3.0)


def test_wind_thermal_crossover_near_wenz_chart() -> None:
    # With the 1 uPa anchor the 5 kn wind and thermal curves cross where the
    # Wenz chart shows it, in the tens of kHz (about 63 kHz), not ~12 kHz as
    # the 20 uPa-anchored level would give.
    f = np.logspace(3.5, 5.5, 2001)
    res = ocean_ambient_noise(f, wind_speed_knots=5.0)
    cross = f[np.argmin(np.abs(res.wind - res.thermal))]
    assert 40e3 < cross < 90e3


def test_wind_minus_five_db_per_octave() -> None:
    nl = wind_noise_spectrum([1000.0, 2000.0], 5.0)
    assert float(nl[0] - nl[1]) == pytest.approx(
        5.0 * np.log10(2.0) * (10.0 / 3.0), abs=1e-9
    )
    assert float(nl[0] - nl[1]) == pytest.approx(5.017, abs=1e-3)


def test_wind_plus_five_db_per_doubling_of_speed() -> None:
    quiet = float(wind_noise_spectrum(1000.0, 5.0)[0])
    windy = float(wind_noise_spectrum(1000.0, 10.0)[0])
    assert windy - quiet == pytest.approx(5.017, abs=1e-3)


def test_thermal_matches_mellen_formula() -> None:
    f = np.array([1e4, 5e4, 1e5])
    t, rho, c = 16.85, 1025.0, 1500.0
    p2 = 4.0 * np.pi * _BOLTZMANN * (t + 273.15) * rho * f**2 / c
    expected = 10.0 * np.log10(p2 / _P_REF**2)
    got = thermal_noise_spectrum(f, temperature=t, density=rho, sound_speed=c)
    assert np.allclose(got, expected, atol=1e-9)


def test_thermal_rises_twenty_db_per_decade() -> None:
    nl = thermal_noise_spectrum([1e4, 1e5])
    assert float(nl[1] - nl[0]) == pytest.approx(20.0, abs=1e-9)


def test_composite_is_energy_sum_and_dominates_components() -> None:
    f = np.array([500.0, 1000.0, 5e4])
    res = ocean_ambient_noise(f, wind_speed_knots=10.0)
    assert isinstance(res, AmbientNoiseResult)
    expected = 10.0 * np.log10(10.0 ** (res.wind / 10.0) + 10.0 ** (res.thermal / 10.0))
    assert np.allclose(res.spectrum_level, expected, atol=1e-9)
    assert np.all(res.spectrum_level >= np.maximum(res.wind, res.thermal) - 1e-9)
    assert res.shipping is None


def test_shipping_component_added_in_energy() -> None:
    f = np.array([100.0, 1000.0])
    ship = np.array([80.0, 70.0])
    res = ocean_ambient_noise(f, wind_speed_knots=5.0, shipping=ship)
    expected = 10.0 * np.log10(
        10.0 ** (res.wind / 10.0) + 10.0 ** (res.thermal / 10.0) + 10.0 ** (ship / 10.0)
    )
    assert np.allclose(res.spectrum_level, expected, atol=1e-9)
    assert res.shipping is not None


def test_shipping_length_mismatch_rejected() -> None:
    with pytest.raises(ValueError, match="shipping.*same shape"):
        ocean_ambient_noise([100.0, 1000.0], wind_speed_knots=5.0, shipping=[80.0])


def test_shipping_non_finite_rejected() -> None:
    with pytest.raises(ValueError, match="shipping.*finite"):
        ocean_ambient_noise(
            [100.0, 1000.0], wind_speed_knots=5.0, shipping=[80.0, np.nan]
        )


def test_invalid_inputs_rejected() -> None:
    with pytest.raises(ValueError, match="frequency"):
        wind_noise_spectrum(-1.0, 5.0)
    with pytest.raises(ValueError, match="wind_speed"):
        wind_noise_spectrum(1000.0, -1.0)


def test_calm_sea_zero_wind_returns_minus_inf() -> None:
    # A calm sea (0 knots) has no wind-driven noise: -inf dB (0 energy).
    assert float(wind_noise_spectrum(1000.0, 0.0)[0]) == -np.inf
    # In the composite, the wind term then drops out (thermal only here).
    res = ocean_ambient_noise([5e4], wind_speed_knots=0.0)
    assert np.isneginf(res.wind[0])
    assert np.isfinite(res.spectrum_level[0])
    assert float(res.spectrum_level[0]) == pytest.approx(
        float(res.thermal[0]), abs=1e-9
    )


def test_ambient_plot_smoke() -> None:
    f = np.logspace(2, 5, 50)
    res = ocean_ambient_noise(f, wind_speed_knots=10.0, shipping=np.full(50, 60.0))
    assert res.plot() is not None


def test_thermal_noise_ainslie_closed_form_spot_value() -> None:
    # Ainslie, Principles of Sonar Performance Modelling (2010), Eq. (9.153)
    # p. 485: thermal noise NLf = -14.7 + 20 lg F(kHz) dB re uPa2/Hz. At
    # F = 100 kHz that is 25.3 dB re uPa2/Hz. Ainslie prints the constant to
    # one decimal, so 0.1 dB covers the rounding against the library's exact
    # Mellen physical form at its default conditions.
    got = thermal_noise_spectrum([100000.0])
    assert float(got[0]) == pytest.approx(-14.7 + 20.0 * 2.0, abs=0.1)
