#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Tests for the speed of sound in sea water (UNESCO / Del Grosso / Mackenzie / Medwin).

Oracles: the printed Wong & Zhu (1995) ITS-90 check tables (Tables III and IV,
the exact refit the module implements), the canonical Mackenzie check value
1550.744 m/s (published, absolute), mutual agreement of the four independent
equations within their common domain, the Leroy & Parthiot standard-ocean
pressure, and for Medwin the two partial derivatives Ainslie prints alongside
the formula (*Principles of Sonar Performance Modelling*, Springer 2010,
Equations 1.2 to 1.4, printed p. 20).
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pytest

from phonometry.underwater.sound_speed import (
    _KGCM2_PER_BAR,
    SoundSpeedProfile,
    _del_grosso,
    _unesco,
    depth_to_pressure,
    sea_water_sound_speed,
    sound_speed_profile,
)

# Wong & Zhu, J. Acoust. Soc. Am. 97 (3), 1995, Table III: speed of sound in
# sea water (m/s) from the t90-corrected UNESCO/Chen-Millero polynomial
# (factual reference values, transcribed from the printed page and
# cross-checked by recomputation). Grid: (pressure bar, t90 degC, S PSU, m/s).
_WONG_ZHU_TABLE_III = [
    (0, 0, 25, 1435.790),
    (100, 10, 25, 1494.127),
    (500, 20, 25, 1593.613),
    (1000, 40, 25, 1719.171),
    (0, 30, 30, 1540.416),
    (200, 0, 30, 1475.448),
    (600, 20, 30, 1615.686),
    (1000, 10, 30, 1653.261),
    (0, 0, 35, 1449.139),
    (300, 30, 35, 1595.909),
    (500, 20, 35, 1604.492),
    (900, 40, 35, 1712.175),
    (1000, 0, 35, 1623.150),
    (0, 40, 40, 1568.141),
    (400, 10, 40, 1562.547),
    (700, 30, 40, 1666.500),
    (1000, 20, 40, 1692.195),
]

# Wong & Zhu 1995, Table IV: same check grid for the t90-corrected Del Grosso
# polynomial (factual reference values). The table lists pressure in bars;
# Del Grosso's equation takes kg/cm2 (1 bar = _KGCM2_PER_BAR kg/cm2).
_WONG_ZHU_TABLE_IV = [
    (0, 0, 25, 1435.711),
    (100, 10, 25, 1494.457),
    (500, 20, 25, 1597.743),
    (1000, 40, 25, 1734.533),
    (0, 40, 30, 1558.221),
    (200, 0, 30, 1475.105),
    (500, 30, 30, 1622.209),
    (1000, 10, 30, 1653.848),
    (0, 0, 35, 1449.083),
    (300, 30, 35, 1593.159),
    (500, 20, 35, 1603.679),
    (900, 40, 35, 1704.948),
    (1000, 0, 35, 1622.269),
    (0, 40, 40, 1568.053),
    (400, 10, 40, 1562.595),
    (700, 30, 40, 1665.789),
    (1000, 20, 40, 1695.212),
]


@pytest.mark.parametrize(("p_bar", "t90", "s", "c_ref"), _WONG_ZHU_TABLE_III)
def test_unesco_wong_zhu_table_iii_printed_check_values(
    p_bar: float, t90: float, s: float, c_ref: float,
) -> None:
    # The module implements exactly this refit, so it must agree at the
    # table's printed resolution (0.001 m/s; measured max |dev| 0.0005 m/s).
    assert float(_unesco(t90, s, p_bar)) == pytest.approx(c_ref, abs=1e-3)


@pytest.mark.parametrize(("p_bar", "t90", "s", "c_ref"), _WONG_ZHU_TABLE_IV)
def test_del_grosso_wong_zhu_table_iv_printed_check_values(
    p_bar: float, t90: float, s: float, c_ref: float,
) -> None:
    # Same printed-decimal tolerance (measured max |dev| 0.0005 m/s).
    assert float(_del_grosso(t90, s, p_bar * _KGCM2_PER_BAR)) == pytest.approx(
        c_ref, abs=1e-3)


def test_mackenzie_canonical_check_value() -> None:
    # Mackenzie (1981) canonical check: c(25 C, 35 ppt, 1000 m) = 1550.744 m/s.
    c = sea_water_sound_speed(25.0, 35.0, 1000.0, model="mackenzie")
    assert c == pytest.approx(1550.744, abs=1e-3)


def test_depth_to_pressure_leroy_parthiot() -> None:
    # 1000 m at 45 deg -> ~10.106 MPa (standard ocean); ~1 MPa per 100 m.
    assert depth_to_pressure(1000.0, 45.0) == pytest.approx(10.1064, abs=1e-3)
    assert depth_to_pressure(0.0) == pytest.approx(0.0, abs=1e-9)


def test_three_models_agree_in_common_domain() -> None:
    # UNESCO, Del Grosso and Mackenzie must agree within ~1 m/s at a mid-ocean
    # point inside all three domains (10 C, 35 ppt, 1000 m).
    kw = {"latitude": 45.0}
    c_u = sea_water_sound_speed(10.0, 35.0, 1000.0, model="unesco", **kw)
    c_d = sea_water_sound_speed(10.0, 35.0, 1000.0, model="del_grosso", **kw)
    c_m = sea_water_sound_speed(10.0, 35.0, 1000.0, model="mackenzie", **kw)
    assert c_u == pytest.approx(1506.52, abs=0.05)
    assert c_d == pytest.approx(1506.31, abs=0.05)
    assert c_m == pytest.approx(1506.26, abs=0.05)
    assert max(abs(c_u - c_d), abs(c_u - c_m), abs(c_d - c_m)) < 1.0


def test_medwin_matches_hand_evaluation_of_equation_1_2() -> None:
    """Ainslie Eq. (1.2): 1449.2 + 4.6T + 0.016z − 0.055T² + [(1.34 − 0.010T)(S−35)
    + 2.9e-4·T³], recomputed term by term at 15 °C, 38 ppt, 200 m.
    """
    t, s, z = 15.0, 38.0, 200.0
    expected = (
        1449.2 + 4.6 * t + 0.016 * z - 0.055 * t**2
        + (1.34 - 0.010 * t) * (s - 35.0) + 2.9e-4 * t**3
    )
    assert sea_water_sound_speed(t, s, z, model="medwin") == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("temperature", [0.0, 10.0, 20.0, 30.0])
def test_medwin_temperature_derivative_matches_equation_1_3(temperature: float) -> None:
    """"∂c/∂T ≈ 4.6 − 0.110·T m/s per degree Celsius", neglecting the bracketed terms.

    Ainslie states 3.5 m/s per °C at T = 10 °C.
    """
    h = 1e-5
    kw = {"model": "medwin"}
    full = (
        sea_water_sound_speed(temperature + h, 35.0, 0.0, **kw)
        - sea_water_sound_speed(temperature - h, 35.0, 0.0, **kw)
    ) / (2.0 * h)
    # Remove the derivative of the bracketed cubic term the equation excludes.
    reduced = full - 3.0 * 2.9e-4 * temperature**2
    assert reduced == pytest.approx(4.6 - 0.110 * temperature, abs=1e-4)
    if temperature == 10.0:
        assert reduced == pytest.approx(3.5, abs=1e-4)


def test_medwin_depth_derivative_matches_equation_1_4() -> None:
    """"∂c/∂z ≈ 0.016 m/s per meter" -- exact for the Medwin form."""
    h = 1e-3
    kw = {"model": "medwin"}
    gradient = (
        sea_water_sound_speed(10.0, 35.0, 500.0 + h, **kw)
        - sea_water_sound_speed(10.0, 35.0, 500.0 - h, **kw)
    ) / (2.0 * h)
    assert gradient == pytest.approx(0.016, abs=1e-6)


def test_medwin_agrees_with_the_other_three_over_the_common_domain() -> None:
    """Medwin is a deliberately simplified fit; it stays within ~2.5 m/s of the rest."""
    worst = 0.0
    for t in (2.0, 10.0, 20.0, 30.0):
        for s in (30.0, 35.0, 40.0):
            for z in (0.0, 500.0, 1000.0, 2000.0):
                speeds = [
                    sea_water_sound_speed(t, s, z, model=m)
                    for m in ("unesco", "del_grosso", "mackenzie", "medwin")
                ]
                worst = max(worst, max(speeds) - min(speeds))
    assert worst < 2.5


def test_medwin_profile_gradient_is_constant_in_isothermal_water() -> None:
    depths = np.linspace(0.0, 1000.0, 21)
    profile = sound_speed_profile(depths, 10.0, 35.0, model="medwin")
    assert profile.model == "medwin"
    assert np.allclose(profile.gradient, 0.016, atol=1e-9)


def test_surface_speed_increases_with_temperature() -> None:
    cold = sea_water_sound_speed(5.0, 35.0, 0.0, model="unesco")
    warm = sea_water_sound_speed(20.0, 35.0, 0.0, model="unesco")
    assert warm > cold


def test_unknown_model_rejected() -> None:
    with pytest.raises(ValueError, match="model"):
        sea_water_sound_speed(10.0, 35.0, 100.0, model="wilson")


def test_negative_depth_rejected() -> None:
    with pytest.raises(ValueError, match="depth"):
        sea_water_sound_speed(10.0, 35.0, -5.0)


def test_profile_gradient_and_shape() -> None:
    depths = np.linspace(0.0, 2000.0, 21)
    prof = sound_speed_profile(depths, temperatures=10.0, salinities=35.0, model="unesco")
    assert isinstance(prof, SoundSpeedProfile)
    assert prof.sound_speed.shape == depths.shape
    assert prof.gradient.shape == depths.shape
    # Isothermal/isohaline column: speed rises with depth (pressure), gradient > 0.
    assert np.all(np.diff(prof.sound_speed) > 0.0)
    assert np.all(prof.gradient > 0.0)


def test_profile_requires_increasing_depths() -> None:
    with pytest.raises(ValueError, match="increasing"):
        sound_speed_profile([0.0, 100.0, 50.0], 10.0, 35.0)


def test_profile_plot_smoke() -> None:
    depths = np.linspace(0.0, 1000.0, 11)
    prof = sound_speed_profile(depths, 12.0, 35.0)
    assert prof.plot() is not None


def test_unesco_published_canonical_check_value() -> None:
    # Fofonoff & Millard 1983 (UNESCO Tech. Pap. Mar. Sci. 44) canonical check:
    # SVEL(S = 40, T68 = 40 C, P = 10000 dbar) = 1731.995 m/s. The module uses
    # the Wong-Zhu ITS-90 refit, so convert T90 = T68/1.00024; the tolerance
    # covers the published refit residual (~0.01 m/s).
    assert float(_unesco(40.0 / 1.00024, 40.0, 1000.0)) == pytest.approx(
        1731.995, abs=0.02)
