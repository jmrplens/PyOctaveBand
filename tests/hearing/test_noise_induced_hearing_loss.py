#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Tests for :mod:`phonometry.hearing.noise_induced_hearing_loss` (noise-induced hearing loss).

Validated against the worked examples of ISO 1999:2013 Annex D (Tables D.1 to
D.4), which tabulate the NIPTS in dB for the six audiometric frequencies, four
exposure levels (85, 90, 95, 100 dB), four durations (10, 20, 30, 40 years) and
the 90 % / 50 % / 10 % fractiles. In the standard's notation the percentage is
the fraction of the population with worse hearing, so its 10 % column is the
most-susceptible tenth (fractile 0.9 here) and its 90 % column the least
(fractile 0.1).
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from reference_data import (
    ISO1999_ANNEX_C_COMPRESSION_FENCE,
    ISO1999_ANNEX_C_H,
    ISO1999_ANNEX_C_H_MEAN,
    ISO1999_ANNEX_C_HTLAN,
    ISO1999_ANNEX_C_N,
    ISO1999_ANNEX_C_N_4K_COMPRESSED,
    ISO1999_ANNEX_C_N_MEAN,
    ISO1999_N10_3K_100_40,
    ISO1999_N10_4K_90_20,
    ISO1999_N50_4K_90_20,
)

from phonometry.hearing import noise_induced_hearing_loss as m

# Annex D, per (level, years): rows are the six frequencies 500..6000 Hz,
# columns the (90 %, 50 %, 10 %) fractiles -> here (0.10, 0.50, 0.90).
# Rounding note: the standard's own printed cells carry intermediate rounding;
# at half-integer boundaries a recomputed cell can differ by 1 dB from the
# print. Known case (Table D.4, not pinned here): 100 dB, 10 years, 3000 Hz,
# 10 % prints 41 where the unrounded value is 41.52 (rounds to 42). If a D.4
# row is ever added, allow that +/-1 dB ambiguity.
_ANNEX_D = {
    (85.0, 40.0): [(0, 0, 0), (0, 0, 0), (1, 2, 2), (3, 5, 7), (5, 7, 9), (2, 4, 6)],
    (90.0, 20.0): [
        (0, 0, 0),
        (0, 0, 0),
        (2, 4, 8),
        (7, 10, 16),
        (9, 13, 18),
        (4, 8, 14),
    ],
    (95.0, 10.0): [
        (0, 0, 1),
        (1, 2, 4),
        (0, 5, 13),
        (8, 16, 25),
        (13, 20, 27),
        (5, 14, 23),
    ],
    (100.0, 40.0): [
        (5, 7, 11),
        (8, 11, 19),
        (16, 24, 39),
        (29, 38, 60),
        (30, 41, 56),
        (19, 30, 48),
    ],
}


def test_annex_d_table_pins_shared_reference_data() -> None:
    """Pin the inline Annex D transcription to ``tests/reference_data/``.

    The CI conformance report asserts the shared constants; this guard keeps
    the (larger) inline table and the shared scalars from drifting apart.
    """
    assert _ANNEX_D[(90.0, 20.0)][4][1] == ISO1999_N50_4K_90_20  # 4 kHz median
    assert _ANNEX_D[(90.0, 20.0)][4][2] == ISO1999_N10_4K_90_20  # 4 kHz, worst 10 %
    assert _ANNEX_D[(100.0, 40.0)][3][2] == ISO1999_N10_3K_100_40  # 3 kHz, worst 10 %


@pytest.mark.parametrize(("key", "expected"), _ANNEX_D.items())
def test_nipts_matches_annex_d(key: tuple[float, float], expected: list) -> None:
    l_ex, years = key
    got = np.column_stack(
        [
            np.round(m.nipts(l_ex, years, frac).value).astype(int)
            for frac in (0.10, 0.50, 0.90)
        ]
    )
    np.testing.assert_array_equal(got, np.array(expected))


def test_median_formula_2_by_hand() -> None:
    # 4000 Hz, 20 years, 90 dB: N50 = (u + v*lg t)*(L - L0)^2
    # = (0.025 + 0.025*log10(20)) * (90 - 75)^2 = 12.94 dB.
    r = m.nipts(90.0, 20.0, 0.5)
    assert r.median[4] == pytest.approx(12.9435, abs=1e-3)
    assert r.frequencies[4] == 4000.0


def test_level_below_cutoff_gives_zero() -> None:
    # At 85 dB the 500 Hz (L0 = 93) and 1000 Hz (L0 = 89) bands are below the
    # cut-off, so the NIPTS is zero at every fractile.
    for frac in (0.1, 0.5, 0.9):
        r = m.nipts(85.0, 30.0, frac)
        assert r.value[0] == 0.0  # 500 Hz
        assert r.value[1] == 0.0  # 1000 Hz


def test_fractile_ordering_and_median() -> None:
    # The median is the 0.5 fractile; higher fractile => larger (worse) NIPTS.
    low = m.nipts(95.0, 30.0, 0.1).value
    mid = m.nipts(95.0, 30.0, 0.5).value
    high = m.nipts(95.0, 30.0, 0.9).value
    np.testing.assert_allclose(mid, m.nipts(95.0, 30.0, 0.5).median)
    assert np.all(low <= mid)
    assert np.all(mid <= high)


def test_negative_nipts_clamped_to_zero() -> None:
    # For short durations the lower-fractile NIPTS can go negative and must be
    # clamped to zero (clause 6.3.2).
    r = m.nipts(90.0, 1.0, 0.05)
    assert np.all(r.value >= 0.0)


def test_short_duration_extrapolation() -> None:
    # Formula (3): N50(t<10) = lg(t+1)/lg(11) * N50(t=10).
    n50_10 = m.nipts(95.0, 10.0, 0.5).median
    n50_5 = m.nipts(95.0, 5.0, 0.5).median
    factor = np.log10(5.0 + 1.0) / np.log10(11.0)
    np.testing.assert_allclose(n50_5, factor * n50_10, rtol=1e-9)


@pytest.mark.filterwarnings("ignore::phonometry.hearing.NoiseInducedHearingLossWarning")
def test_spreads_non_negative_subyear() -> None:
    # Below 1 year the linear spread term can go negative; du/dl are clamped so
    # the fractile ordering never inverts (spreads are standard deviations).
    # The sub-year duration is a deliberate extrapolation, so its
    # validated-domain warning is expected here.
    r_low = m.nipts(100.0, 0.02, 0.1)
    r_high = m.nipts(100.0, 0.02, 0.9)
    assert np.all(r_high.value >= r_low.value)
    assert np.all(r_low.spread_upper >= 0.0)
    assert np.all(r_low.spread_lower >= 0.0)


def test_htlan_formula_1() -> None:
    # H' = H + N - H*N/120 at each frequency (clause 6.1).
    r = m.htlan(60, "male", 90.0, 30.0, 0.5)
    expected = r.htla + r.nipts - r.htla * r.nipts / 120.0
    np.testing.assert_allclose(r.threshold, expected)
    # The combined threshold is at least the age component alone.
    assert np.all(r.threshold >= r.htla - 1e-9)


def test_htlan_matches_components() -> None:
    r = m.htlan(50, "female", 100.0, 20.0, 0.9)
    np.testing.assert_allclose(r.nipts, m.nipts(100.0, 20.0, 0.9).value)


def test_frequency_subset() -> None:
    r = m.nipts(95.0, 20.0, 0.5, frequencies=[4000.0, 6000.0])
    full = m.nipts(95.0, 20.0, 0.5)
    np.testing.assert_allclose(r.frequencies, [4000.0, 6000.0])
    np.testing.assert_allclose(r.value, full.value[[4, 5]])


def test_invalid_inputs_raise() -> None:
    with pytest.raises(ValueError, match="years must be positive"):
        m.nipts(90.0, 0.0)
    with pytest.raises(ValueError, match="fractile"):
        m.nipts(90.0, 20.0, 1.5)
    with pytest.raises(ValueError, match="not an ISO 1999"):
        m.nipts(90.0, 20.0, frequencies=[1234.0])
    with pytest.raises(ValueError, match="at least 18"):
        m.htlan(10, "male", 90.0, 20.0)


def test_annex_c_worked_example() -> None:
    """Reproduce the ISO 1999:2013 Annex C risk-assessment chain.

    A male population aged 50 exposed to L_EX,8h = 90 dB for 30 years, assessed
    on 1/2/4 kHz at Q = 10 % (the most-susceptible tenth, fractile 0.9). The
    annex's noise input is the Table D.2 row, which the library reproduces; its
    age input is the Table A.3 selection printed in the annex, fed here into
    Formula (1). The annex compresses only where "(H + N) > approximately
    40 dB", which of these bands is 4 kHz alone.
    """
    got = m.nipts(90.0, 30.0, 0.9, frequencies=[1000.0, 2000.0, 4000.0]).value
    np.testing.assert_array_equal(np.round(got), np.array(ISO1999_ANNEX_C_N))

    h = np.array(ISO1999_ANNEX_C_H)
    n = np.array(ISO1999_ANNEX_C_N)
    compressed = m.combine_age_and_noise(h, n) - h
    # C.5: 19 - 36 x 19 / 120 = 13,3 dB at 4 kHz.
    assert compressed[2] == pytest.approx(ISO1999_ANNEX_C_N_4K_COMPRESSED, abs=1e-9)

    noise = np.where(h + n > ISO1999_ANNEX_C_COMPRESSION_FENCE, compressed, n)
    # C.8 and C.3, both printed to one decimal by the annex.
    assert noise.mean() == pytest.approx(ISO1999_ANNEX_C_N_MEAN, abs=0.05)
    assert h.mean() == pytest.approx(ISO1999_ANNEX_C_H_MEAN, abs=0.05)
    # C.11: the combined threshold of the noise-exposed population.
    assert h.mean() + noise.mean() == pytest.approx(ISO1999_ANNEX_C_HTLAN, abs=0.05)


def test_combine_age_and_noise_matches_htlan() -> None:
    """Formula (1) exposed on its own agrees with the ``htlan`` composition."""
    r = m.htlan(55, "female", 92.0, 25.0, 0.75)
    np.testing.assert_allclose(m.combine_age_and_noise(r.htla, r.nipts), r.threshold)


def test_outside_validated_domain_warns() -> None:
    """Conditions beyond the standard's stated ranges warn (but still compute)."""
    with pytest.warns(m.NoiseInducedHearingLossWarning, match="1-40 year"):
        m.nipts(90.0, 60.0, 0.5)
    with pytest.warns(m.NoiseInducedHearingLossWarning, match="tails"):
        m.nipts(90.0, 20.0, 0.99)
    with pytest.warns(m.NoiseInducedHearingLossWarning, match="exceeds the 100 dB"):
        m.nipts(130.0, 20.0, 0.5)
    # The htlan noise component inherits the same guards.
    with pytest.warns(m.NoiseInducedHearingLossWarning, match="1-40 year"):
        m.htlan(60, "male", 90.0, 60.0, 0.5)


def test_inside_validated_domain_is_silent() -> None:
    """The domain edges themselves are inside the validated range."""
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error", m.NoiseInducedHearingLossWarning)
        m.nipts(100.0, 1.0, 0.05)
        m.nipts(100.0, 40.0, 0.95)


def test_nipts_rejects_a_fractile_spectrum_of_another_length() -> None:
    """``value`` is the one spectrum the median fiche never draws.

    At the default fractile the plot shows the median and its band alone, so a
    ``value`` computed on another frequency grid reaches a finished sheet: the
    table drops its tail without a word and the boxed 2/3/4 kHz mean, with the
    PASS/FAIL verdict read off it, comes from the wrong positions.
    """
    result = m.nipts(95.0, 20.0)
    on_another_grid = np.append(result.value, [12.0, 9.0])
    with pytest.raises(ValueError, match="'value'"):
        dataclasses.replace(result, value=on_another_grid)


def test_htlan_rejects_a_component_of_another_length() -> None:
    """The three components are combined, tabulated and drawn by position.

    None of them reaches a finished sheet at the wrong length, but without this
    check the complaint is an ``IndexError`` from inside the fiche table or a
    shape mismatch from inside matplotlib, neither of which names the field.
    """
    result = m.htlan(60, "male", 95.0, 20.0, 0.5)
    one_frequency_short = result.nipts[:-1]
    with pytest.raises(ValueError, match="'nipts'"):
        dataclasses.replace(result, nipts=one_frequency_short)


def test_plots_return_axes() -> None:
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    assert isinstance(m.nipts(95.0, 20.0, 0.9).plot(), plt.Axes)
    assert isinstance(m.htlan(60, "male", 95.0, 20.0, 0.9).plot(), plt.Axes)
    plt.close("all")
