#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Tests for marine-mammal hearing thresholds.

Oracles:

* Southall et al. (2019), *Aquatic Mammals* 45(2), Equation (1) with the fit
  parameters of Table 2 (printed p. 144) and the frequencies of best hearing of
  Table 4 (printed p. 148); the article's own prose quotes the resulting
  threshold at ``f0`` for three groups (54, 48 and 53 dB re 1 µPa, printed
  pp. 155-156), which is the published check used here.
* Ainslie, *Principles of Sonar Performance Modelling* (Springer 2010),
  Equation (11.159) and the two values printed with it (printed pp. 619-620 and
  Table 11.6): the audiogram minimum 39.0 dB re 1 µPa at 22.6 kHz and
  51.2 dB re 1 µPa at 50 kHz.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from phonometry.underwater.bioacoustics.audiograms import (
    AUDIOGRAM_GROUPS,
    BEST_HEARING_FREQUENCY_KHZ,
    AudiogramResult,
    audiogram_parameters,
    group_audiogram,
    orca_audiogram,
)

# ---------------------------------------------------------------------------
# Southall et al. (2019) Equation (1)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("group", "printed"),
    [
        # "the hearing threshold at f0 is 54 dB re 1 µPa" (printed p. 155)
        ("HF", 54.0),
        # "the hearing threshold at f0 is 48 dB re 1 µPa" (printed p. 155)
        ("VHF", 48.0),
        # "212 dB re 1 µPa (53 dB at f0 + 159 dB)" (printed p. 156)
        ("PCW", 53.0),
    ],
)
def test_threshold_at_f0_matches_the_published_prose_value(
    group: str, printed: float
) -> None:
    """Eq. (1) + Table 2 + Table 4 reproduce the three thresholds quoted in prose."""
    f0_khz = BEST_HEARING_FREQUENCY_KHZ[group][0]
    got = group_audiogram(f0_khz * 1000.0, group).threshold[0]
    # The article rounds these to whole decibels (PCW lands on 53.50, printed 53).
    assert got == pytest.approx(printed, abs=0.55)


def test_group_audiogram_matches_hand_evaluation() -> None:
    """HF at 55 kHz, term by term: 46.2 + 35.5·lg(1 + 25.9/55) + (55/47.8)^3.56."""
    expected = 46.2 + 35.5 * np.log10(1.0 + 25.9 / 55.0) + (55.0 / 47.8) ** 3.56
    got = group_audiogram(55e3, "HF").threshold[0]
    assert got == pytest.approx(expected, rel=1e-12)
    assert got == pytest.approx(53.797, abs=1e-3)


def test_normalized_audiogram_removes_the_absolute_sensitivity() -> None:
    """Table 3 refits thresholds normalised to each individual's best value.

    The refit is not constrained to pass exactly through 0 dB, but every group
    ends up within 10 dB of it, where the absolute fits of Table 2 span more
    than 70 dB between groups.
    """
    freqs = np.logspace(np.log10(20.0), np.log10(300e3), 20_000)
    normalized = [
        group_audiogram(freqs, group, normalized=True).best_threshold
        for group in AUDIOGRAM_GROUPS
    ]
    absolute = [
        group_audiogram(freqs, group).best_threshold for group in AUDIOGRAM_GROUPS
    ]
    assert max(abs(v) for v in normalized) < 10.0
    assert max(absolute) - min(absolute) > 70.0


@pytest.mark.parametrize("group", AUDIOGRAM_GROUPS)
@pytest.mark.parametrize("normalized", [False, True])
def test_best_frequency_agrees_with_table_4(group: str, normalized: bool) -> None:
    """The fit minimum lands on the ``f0`` column of Table 4 (printed p. 148).

    Table 4 is quoted to two significant figures and comes from the measured
    data rather than from the fitted curve, so the agreement is a 20 % check,
    not a digit-for-digit one.
    """
    freqs = np.logspace(np.log10(20.0), np.log10(300e3), 20_000)
    expected = BEST_HEARING_FREQUENCY_KHZ[group][1 if normalized else 0]
    got = group_audiogram(freqs, group, normalized=normalized).best_frequency / 1000.0
    assert got == pytest.approx(expected, rel=0.20)


def test_normalized_and_original_share_the_result_shape() -> None:
    res = group_audiogram([1e3, 10e3], "OCW", normalized=True)
    assert isinstance(res, AudiogramResult)
    assert res.group == "OCW"
    assert "Table 3" in res.source
    assert res.in_air is False
    assert audiogram_parameters("ocw", normalized=True).t0 == 2.36


def test_in_air_groups_are_flagged() -> None:
    assert group_audiogram(2.3e3, "PCA").in_air is True
    assert group_audiogram(10e3, "OCA").in_air is True


def test_lf_cetaceans_have_no_published_audiogram() -> None:
    """Southall et al. never print F1 for LF cetaceans, so the group is absent."""
    assert "LF" not in AUDIOGRAM_GROUPS
    with pytest.raises(ValueError, match="no fitted audiogram for LF"):
        group_audiogram(1e3, "LF")


# ---------------------------------------------------------------------------
# Ainslie (2010) Equation (11.159)
# ---------------------------------------------------------------------------


def test_orca_audiogram_minimum_matches_the_printed_value() -> None:
    """ "the minimum hearing threshold, of 39.0 dB re µPa², occurs at 22.6 kHz"."""
    assert orca_audiogram(22.6e3).threshold[0] == pytest.approx(39.0, abs=0.05)


def test_orca_audiogram_minimum_location() -> None:
    freqs = np.logspace(np.log10(500.0), np.log10(80e3), 20_001)
    res = orca_audiogram(freqs)
    assert res.best_frequency / 1000.0 == pytest.approx(22.6, abs=0.1)
    assert res.best_threshold == pytest.approx(39.0, abs=0.05)


def test_orca_audiogram_at_50_khz_needs_the_third_branch() -> None:
    """ "The threshold ... at the pulse center frequency (50 kHz) is 51.2 dB re µPa²".

    50 kHz sits above the 46.2 kHz break, so the third branch applies. Using
    the second one there returns 50.51 dB, 0.7 dB low: that is the regression
    this test guards.
    """
    got = orca_audiogram(50e3).threshold[0]
    assert got == pytest.approx(51.2, abs=0.05)
    second_branch = 242.9 * 50.0 ** (-0.7578) + 0.5643 * 50.0**1.076
    assert second_branch == pytest.approx(50.51, abs=0.01)
    assert abs(got - second_branch) > 0.5


def test_orca_audiogram_is_continuous_at_its_breaks() -> None:
    """The published coefficients join to within their own rounding."""
    for break_khz in (11.3, 46.2):
        below = orca_audiogram(break_khz * 1000.0 * (1.0 - 1e-9)).threshold[0]
        above = orca_audiogram(break_khz * 1000.0 * (1.0 + 1e-9)).threshold[0]
        assert abs(below - above) < 0.1


def test_orca_audiogram_supports_the_figure_of_merit_of_example_11_4_6() -> None:
    """FOM_HT = (SL + TS − HT)/2 = (198.2 − 29.0 − 51.2)/2 = 59.0 dB re m²."""
    ht = orca_audiogram(50e3).threshold[0]
    assert (198.2 - 29.0 - ht) / 2.0 == pytest.approx(59.0, abs=0.05)


@pytest.mark.parametrize("frequency", [400.0, 90e3])
def test_orca_audiogram_rejects_frequencies_outside_the_fit(frequency: float) -> None:
    with pytest.raises(ValueError, match="500-80000 Hz"):
        orca_audiogram(frequency)


# ---------------------------------------------------------------------------
# Validation and plotting
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("frequency", [[], [0.0], [-1.0], [np.nan]])
def test_invalid_frequencies_raise(frequency: list[float]) -> None:
    with pytest.raises(ValueError, match="frequency_hz"):
        group_audiogram(frequency, "HF")


def test_unknown_group_raises() -> None:
    with pytest.raises(ValueError, match="group"):
        group_audiogram(1e3, "dolphin")


def test_plot_returns_axes() -> None:
    freqs = np.logspace(3.0, 5.0, 60)
    assert group_audiogram(freqs, "VHF").plot() is not None
    orca = orca_audiogram(np.logspace(np.log10(500.0), np.log10(80e3), 60))
    assert orca.plot(language="es") is not None
    assert orca.plot().get_title() == "Orca audiogram"
    plt.close("all")
