#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Tests for underwater transmission loss (spreading + volume absorption).

Oracles: the printed Francois & Garrison (1982, Part II) Table IV absorption
values, the closed-form spreading laws (hand recomputation), the Thorp,
Francois-Garrison and Ainslie-McColm absorption formulae (independent inline
recomputation), and the mutual agreement of Francois-Garrison and Ainslie-McColm
(the latter's paper states within 10 % of the former).
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pytest

from phonometry.underwater.propagation import (
    TransmissionLossResult,
    seawater_absorption,
    spreading_loss,
    transmission_loss,
)


def test_spherical_spreading_6db_per_doubling() -> None:
    loss = spreading_loss([1.0, 10.0, 100.0, 200.0], law="spherical")
    np.testing.assert_allclose(loss, [0.0, 20.0, 40.0, 40.0 + 20.0 * np.log10(2.0)])


def test_cylindrical_spreading_3db_per_doubling() -> None:
    loss = spreading_loss([100.0, 200.0], law="cylindrical")
    assert loss[1] - loss[0] == pytest.approx(10.0 * np.log10(2.0))


def test_practical_spreading_matches_pieces() -> None:
    r0 = 1000.0
    loss = spreading_loss([500.0, 1000.0, 4000.0], law="practical", transition_range=r0)
    assert loss[0] == pytest.approx(20.0 * np.log10(500.0))  # spherical below R0
    assert loss[1] == pytest.approx(20.0 * np.log10(1000.0))
    assert loss[2] == pytest.approx(20.0 * np.log10(r0) + 10.0 * np.log10(4.0))


def test_practical_requires_transition_range() -> None:
    with pytest.raises(ValueError, match="transition_range"):
        spreading_loss([100.0], law="practical")


def test_thorp_absorption_recompute() -> None:
    f_khz = 10.0
    expected = 1.0936 * (0.1 * f_khz**2 / (1 + f_khz**2) + 40 * f_khz**2 / (4100 + f_khz**2))
    got = seawater_absorption(10_000.0, model="thorp")
    assert got[0] == pytest.approx(expected, rel=1e-9)
    assert got[0] == pytest.approx(1.1498, abs=1e-3)


# Francois & Garrison, J. Acoust. Soc. Am. 72 (6), 1982, Part II, Table IV:
# total absorption alpha in dB/km at depth 0, pH 8 (factual reference values,
# transcribed from the printed page and cross-checked by recomputation).
# Points selected to span 0.4 kHz - 1 MHz, -1.8 to 30 degC and both tabulated
# salinities; each entry carries the unit of its last printed digit (ULP) and
# the module agrees within half of it, i.e. to the print's own rounding
# (measured max |dev| 0.45 ULP).
#
# The module implements the ORIGINAL paper's boric-acid factor A1 = (8.86/c)
# x 10^(0.78 pH - 5) (Eq. (10)/Fig. 7); the Medwin & Clay transcription
# prints 8.68 (digit transposition), which sits up to 4.6 ULP below the
# print in the boric-dominated 0.6-30 kHz cells (see docs/ERRATA.md). The
# boric-dominated rows below therefore pin the corrected constant directly.
_FG_TABLE_IV = [
    # (f kHz, T degC, S ppt, alpha dB/km, ULP)
    (0.4, 10.0, 30.0, 0.015, 0.001),
    (2.0, 10.0, 35.0, 0.123, 0.001),   # boric-dominated
    (10.0, 30.0, 35.0, 0.606, 0.001),  # boric-dominated
    (10.0, -1.8, 35.0, 1.36, 0.01),
    (14.0, 20.0, 35.0, 1.29, 0.01),
    (20.0, -1.8, 35.0, 4.40, 0.01),
    (20.0, 10.0, 35.0, 3.35, 0.01),
    (40.0, 4.0, 35.0, 11.5, 0.1),
    (50.0, 20.0, 35.0, 13.0, 0.1),
    (60.0, -1.8, 30.0, 13.6, 0.1),
    (80.0, 16.0, 35.0, 28.6, 0.1),
    (100.0, 0.0, 30.0, 21.0, 0.1),
    (100.0, 10.0, 35.0, 33.6, 0.1),
    (140.0, 25.0, 35.0, 58.4, 0.1),
    (200.0, 0.0, 35.0, 40.6, 0.1),
    (200.0, 30.0, 30.0, 79.3, 0.1),
    (300.0, 10.0, 35.0, 73.1, 0.1),
    (500.0, 10.0, 35.0, 125.0, 1.0),
    (700.0, 4.0, 35.0, 228.0, 1.0),
    (1000.0, 0.0, 30.0, 513.0, 1.0),
    (1000.0, 30.0, 35.0, 344.0, 1.0),
]


@pytest.mark.parametrize(("f_khz", "t", "s", "alpha_ref", "ulp"), _FG_TABLE_IV)
def test_francois_garrison_part2_table_iv_printed_values(
    f_khz: float, t: float, s: float, alpha_ref: float, ulp: float,
) -> None:
    got = seawater_absorption(f_khz * 1000.0, temperature=t, salinity=s,
                              depth=0.0, ph=8.0, model="francois-garrison")
    assert float(got[0]) == pytest.approx(alpha_ref, abs=0.5 * ulp)


def test_francois_garrison_recompute() -> None:
    # Independent inline recomputation at 10 kHz, 10 C, 35 ppt, 0 m, pH 8,
    # with the ORIGINAL paper's boric coefficient 8.86 (F&G 1982 Part II,
    # Eq. (10)/Fig. 7; the Medwin & Clay transcription prints 8.68, a digit
    # transposition -- see the note above _FG_TABLE_IV and docs/ERRATA.md).
    f = 10.0
    t, s, z, ph = 10.0, 35.0, 0.0, 8.0
    c = 1412 + 3.21 * t + 1.19 * s + 0.0167 * z
    a1 = (8.86 / c) * 10 ** (0.78 * ph - 5)
    f1 = 2.8 * (s / 35) ** 0.5 * 10 ** (4 - 1245 / (273 + t))
    a2 = 21.44 * (s / c) * (1 + 0.025 * t)
    f2 = 8.17 * 10 ** (8 - 1990 / (273 + t)) / (1 + 0.0018 * (s - 35))
    a3 = 4.937e-4 - 2.59e-5 * t + 9.11e-7 * t**2 - 1.50e-8 * t**3
    expected = a1 * f1 * f**2 / (f**2 + f1**2) + a2 * f2 * f**2 / (f**2 + f2**2) + a3 * f**2
    got = seawater_absorption(10_000.0, temperature=t, salinity=s, depth=z, ph=ph,
                              model="francois-garrison")
    assert got[0] == pytest.approx(expected, rel=1e-9)


def test_francois_garrison_and_ainslie_mccolm_agree() -> None:
    # Ainslie-McColm 1998: within 10 % of Francois-Garrison, 100 Hz - 1 MHz.
    freqs = np.array([1e3, 1e4, 1e5, 5e5])
    fg = seawater_absorption(freqs, temperature=10.0, salinity=35.0, depth=0.0, ph=8.0,
                             model="francois-garrison")
    am = seawater_absorption(freqs, temperature=10.0, salinity=35.0, depth=0.0, ph=8.0,
                             model="ainslie-mccolm")
    assert np.all(np.abs(am - fg) <= 0.10 * fg)


def test_ainslie_mccolm_depth_in_km() -> None:
    # Depth enters A-M in km; 1000 m must reduce the MgSO4 term vs the surface.
    shallow = seawater_absorption(50_000.0, depth=0.0, model="ainslie-mccolm")
    deep = seawater_absorption(50_000.0, depth=1000.0, model="ainslie-mccolm")
    assert deep[0] < shallow[0]


def test_unknown_absorption_model_rejected() -> None:
    with pytest.raises(ValueError, match="model"):
        seawater_absorption(1000.0, model="fisher-simmons")


def test_transmission_loss_is_spreading_plus_absorption() -> None:
    res = transmission_loss(
        [100.0, 1000.0, 10000.0], 10_000.0, law="spherical",
        temperature=10.0, salinity=35.0, depth=0.0, model="francois-garrison",
    )
    assert isinstance(res, TransmissionLossResult)
    alpha = seawater_absorption(10_000.0, temperature=10.0, salinity=35.0, depth=0.0,
                                model="francois-garrison")[0]
    expected = 20.0 * np.log10(res.range_m) + alpha * (res.range_m / 1000.0)
    np.testing.assert_allclose(res.tl, expected, rtol=1e-9)
    np.testing.assert_allclose(res.spreading + res.absorption, res.tl)
    assert res.absorption_coefficient == pytest.approx(alpha)


def test_transmission_loss_rejects_nonpositive_range() -> None:
    with pytest.raises(ValueError, match="range_m"):
        transmission_loss([0.0, 100.0], 1000.0)


def test_transmission_loss_plot_smoke() -> None:
    res = transmission_loss(np.linspace(10.0, 10000.0, 50), 12_000.0)
    assert res.plot() is not None


def test_ainslie_mccolm_table_i_oceans_within_ten_percent_of_fg() -> None:
    # Ainslie & McColm 1998 Table I reference oceans: the paper's Fig. 2 claim
    # (simplified formula within 10 % of Francois-Garrison) holds across
    # 100 Hz - 1 MHz for all four (measured headroom >= 1.65 %).
    oceans = [  # (pH, S, T, z_km)
        (7.7, 34.0, 4.0, 1.0),    # Pacific
        (8.2, 40.0, 22.0, 0.2),   # Red Sea
        (8.2, 30.0, -1.5, 0.0),   # Arctic
        (7.9, 8.0, 4.0, 0.0),     # Baltic
    ]
    f = np.logspace(2, 6, 200)
    for ph_v, s, t, z_km in oceans:
        fg = seawater_absorption(f, temperature=t, salinity=s,
                                 depth=z_km * 1000.0, ph=ph_v,
                                 model="francois-garrison")
        am = seawater_absorption(f, temperature=t, salinity=s,
                                 depth=z_km * 1000.0, ph=ph_v,
                                 model="ainslie-mccolm")
        assert float(np.max(np.abs(am - fg) / fg)) <= 0.10, (ph_v, s, t, z_km)


def test_ainslie_mccolm_book_spot_value_300hz() -> None:
    # Ainslie, Principles of Sonar Performance Modelling (Springer, 2010):
    # the passive worked example of Sec. 3.2.3.8 quotes the seawater
    # absorption at its 300 Hz tonal as 0.0086 dB/km (footnote, p. 78),
    # computed for the book's representative conditions T = 10 C, S = 35,
    # z = 0, pH = 8 (pp. 28-29). The book prints two significant figures
    # (exact Ainslie-McColm value 0.00858 dB/km), so 1e-4 dB/km covers the
    # rounding.
    alpha = seawater_absorption(
        300.0, temperature=10.0, salinity=35.0, depth=0.0, ph=8.0,
        model="ainslie-mccolm",
    )
    assert float(alpha[0]) == pytest.approx(0.0086, abs=1e-4)
