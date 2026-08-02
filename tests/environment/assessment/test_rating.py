#  Copyright (c) 2026. Jose Manuel Requena Plens
"""
Environmental noise descriptors (ISO 1996-1:2016).

Lden (3.6.4): 10*lg{(1/24)*[t_day*10^(0,1*Lday) + t_evening*10^(0,1*(Levening+5))
+ t_night*10^(0,1*(Lnight+10))]} with default periods 12/4/8 h.
Ldn (3.6.5): the day-night variant with defaults 15/9 h and +10 dB at night.
Composite whole-day rating levels (6.5, Formulae 5-6) generalize this to any
set of (level, hours, adjustment) periods.
"""

import numpy as np
import pytest

from phonometry import composite_rating_level, lden, ldn


def test_lden_constant_level_analytic() -> None:
    """Constant level L in all periods: Lden = L + 10*lg[(12 + 4*10^0.5 +
    8*10)/24] for the default 12/4/8 h split."""
    expected_offset = 10 * np.log10((12 + 4 * 10**0.5 + 8 * 10) / 24)
    assert lden(60.0, 60.0, 60.0) == pytest.approx(60.0 + expected_offset, abs=1e-9)


def test_lden_hand_computed() -> None:
    ld, le, ln_ = 60.0, 55.0, 50.0
    expected = 10 * np.log10(
        (12 * 10 ** (0.1 * ld) + 4 * 10 ** (0.1 * (le + 5)) + 8 * 10 ** (0.1 * (ln_ + 10)))
        / 24
    )
    assert lden(ld, le, ln_) == pytest.approx(expected, abs=1e-12)


def test_lden_custom_periods() -> None:
    """EU member states may shorten the evening (3.6.4 Note 1)."""
    ld, le, ln_ = 60.0, 55.0, 50.0
    expected = 10 * np.log10(
        (12 * 10 ** (0.1 * ld) + 2 * 10 ** (0.1 * (le + 5)) + 10 * 10 ** (0.1 * (ln_ + 10)))
        / 24
    )
    assert lden(ld, le, ln_, hours=(12, 2, 10)) == pytest.approx(expected, abs=1e-12)


def test_lden_periods_must_sum_24() -> None:
    with pytest.raises(ValueError, match="24"):
        lden(60, 55, 50, hours=(12, 4, 9))


def test_ldn_constant_level_analytic() -> None:
    """Defaults 15/9 h, night +10 dB (3.6.5)."""
    expected_offset = 10 * np.log10((15 + 9 * 10) / 24)
    assert ldn(60.0, 60.0) == pytest.approx(60.0 + expected_offset, abs=1e-9)


def test_composite_rating_level_formula5() -> None:
    """ISO 1996-1:2016 Formula (5): day-night rating with adjustments."""
    d, l_rd, k_d = 15.0, 60.0, 0.0
    n, l_rn, k_n = 9.0, 52.0, 10.0
    expected = 10 * np.log10(
        d / 24 * 10 ** (0.1 * (l_rd + k_d)) + n / 24 * 10 ** (0.1 * (l_rn + k_n))
    )
    got = composite_rating_level([(l_rd, d, k_d), (l_rn, n, k_n)])
    assert got == pytest.approx(expected, abs=1e-12)


def test_composite_generalizes_lden() -> None:
    ld, le, ln_ = 63.0, 58.0, 51.0
    composite = composite_rating_level([(ld, 12, 0.0), (le, 4, 5.0), (ln_, 8, 10.0)])
    assert composite == pytest.approx(lden(ld, le, ln_), abs=1e-12)


def test_composite_validation() -> None:
    with pytest.raises(ValueError, match="24"):
        composite_rating_level([(60.0, 10.0, 0.0)])
    with pytest.raises(ValueError, match="positive"):
        composite_rating_level([(60.0, -1.0, 0.0), (50.0, 25.0, 0.0)])


def test_composite_accepts_generator_and_rejects_empty() -> None:
    periods = [(60.0, 12, 0.0), (55.0, 4, 5.0), (50.0, 8, 10.0)]
    from_gen = composite_rating_level(p for p in periods)
    assert from_gen == pytest.approx(composite_rating_level(periods))
    with pytest.raises(ValueError, match="one period"):
        composite_rating_level([])


def test_composite_rejects_non_finite_hours() -> None:
    with pytest.raises(ValueError, match="finite"):
        composite_rating_level([(60.0, float("nan"), 0.0), (50.0, 12.0, 0.0)])


def test_lden_manual_es_published_example() -> None:
    # Aviles Lopez & Perera Martin, Manual de acustica ambiental y
    # arquitectonica (Paraninfo), Ejemplo 1.7 (p. 27): Ld = 65, Le = 62,
    # Ln = 54 dBA give Lden = 65.1 dB with the standard 12/4/8 h day.
    # The book prints one decimal (exact value 65.12), so 0.05 covers it.
    assert lden(65.0, 62.0, 54.0) == pytest.approx(65.1, abs=0.05)
