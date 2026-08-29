#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Tests for wind-turbine acoustic noise (IEC 61400-11:2012+A1:2018).

The oracles are the closed-form geometry / critical bandwidth and a synthetic
clean tone on a flat noise floor whose tonal audibility is hand-derived
independently of the implementation.
"""

from __future__ import annotations

import matplotlib as mpl

mpl.use("Agg")
import numpy as np
import pytest

from phonometry.environment.sources.wind_turbine import (
    WindTurbineNoiseWarning,
    apparent_sound_power_level,
    critical_bandwidth,
    slant_distance,
    wind_turbine_tonality,
)


def test_critical_bandwidth_formula() -> None:
    # 25 + 75*(1 + 1.4*0.25)^0.69 = 117.256 Hz at 500 Hz.
    assert critical_bandwidth(500.0) == pytest.approx(117.256, abs=0.02)


def test_critical_bandwidth_low_frequency_band() -> None:
    # A candidate tone in 20-70 Hz uses the fixed 20-120 Hz (100 Hz) band.
    assert critical_bandwidth(50.0) == 100.0


def test_slant_distance() -> None:
    # R0 = 80 + 100/2 = 130; R1 = sqrt(80^2 + 130^2).
    assert slant_distance(80.0, 100.0) == pytest.approx(np.hypot(80.0, 130.0))


def test_apparent_sound_power_single_band() -> None:
    r1 = 150.0
    expected = 100.0 - 6.0 + 10.0 * np.log10(4.0 * np.pi * r1**2)
    assert apparent_sound_power_level([100.0], r1) == pytest.approx(expected, rel=1e-9)


def test_apparent_sound_power_energy_sum() -> None:
    r1 = 120.0
    bands = np.array([90.0, 92.0, 88.0])
    geom = 10.0 * np.log10(4.0 * np.pi * r1**2)
    expected = 10.0 * np.log10(np.sum(10.0 ** ((bands - 6.0 + geom) / 10.0)))
    assert apparent_sound_power_level(bands, r1) == pytest.approx(expected, rel=1e-9)


def _synthetic_tone() -> tuple[np.ndarray, np.ndarray]:
    df = 2.0
    freqs = np.arange(440.0, 560.0 + df, df)
    levels = np.full(freqs.size, 30.0)
    levels[int(np.argmin(np.abs(freqs - 500.0)))] = 60.0
    return levels, freqs


def test_tonal_audibility_synthetic_tone() -> None:
    # Hand-derived: CBW=117.256, ENBW=3.0, L_pn,avg=30 ->
    #   L_pn = 30 + 10 lg(117.256/3.0) = 45.92 dB
    #   L_pt = 60 (single line), ΔL_tn = 14.08
    #   L_a = -2 - lg(1+(500/502)^2.5) = -2.299
    #   ΔL_a = 14.08 - (-2.299) = 16.38 dB
    levels, freqs = _synthetic_tone()
    res = wind_turbine_tonality(levels, freqs)
    assert res.tone_frequency == pytest.approx(500.0)
    assert res.critical_bandwidth == pytest.approx(117.256, abs=0.02)
    assert res.tone_level == pytest.approx(60.0, abs=1e-6)
    assert res.masking_level == pytest.approx(45.92, abs=0.05)
    assert res.tonality == pytest.approx(14.08, abs=0.05)
    assert res.audibility_criterion == pytest.approx(-2.299, abs=0.01)
    assert res.tonal_audibility == pytest.approx(16.38, abs=0.06)
    assert res.is_audible is True


def test_low_frequency_fixed_critical_band() -> None:
    # A candidate tone at 40 Hz uses the FIXED 20-120 Hz band, not one centred
    # on 40 Hz. Loud lines below 20 Hz must be excluded (a centred band
    # [-10, 90] would wrongly fold them into the tone/masking classification).
    df = 2.0
    freqs = np.arange(0.0, 140.0 + df, df)
    levels = np.full(freqs.size, 30.0)
    levels[freqs < 20.0] = 58.0  # excluded by the fixed 20-120 band
    levels[int(np.argmin(np.abs(freqs - 40.0)))] = 60.0
    res = wind_turbine_tonality(levels, freqs, tone_frequency=40.0)
    # Only the 40 Hz tone survives -> single tone line, masking noise = 30 dB.
    assert res.tone_level == pytest.approx(60.0, abs=0.05)
    assert res.masking_level == pytest.approx(
        30.0 + 10.0 * np.log10(100.0 / 3.0), abs=0.05
    )


def test_non_contiguous_tone_lines_counted() -> None:
    # IEC 61400-11+A1:2018: tone lines need not be adjacent to the peak. Two
    # separated lines within 10 dB of the peak both count toward L_pt.
    df = 2.0
    freqs = np.arange(240.0, 360.0 + df, df)
    levels = np.full(freqs.size, 30.0)
    levels[int(np.argmin(np.abs(freqs - 300.0)))] = 60.0
    levels[int(np.argmin(np.abs(freqs - 260.0)))] = 56.0  # isolated, within 10 dB
    res = wind_turbine_tonality(levels, freqs)
    # Both separated tones (no Hanning: neither run has >= 2 adjacent lines).
    expected = 10.0 * np.log10(10.0**6.0 + 10.0**5.6)
    assert res.tone_level == pytest.approx(expected, abs=0.05)


def test_tonal_audibility_flat_spectrum_not_audible() -> None:
    freqs = np.arange(440.0, 560.0, 2.0)
    levels = np.full(freqs.size, 30.0)
    # The flat spectrum's argmax candidate sits at the first line, whose
    # critical band extends below the supplied range: the truncation warning
    # fires alongside the (correct) no-identified-tone outcome.
    with pytest.warns(
        WindTurbineNoiseWarning, match=r"does not cover the whole critical band"
    ):
        res = wind_turbine_tonality(levels, freqs)
    assert res.is_audible is False
    assert res.has_identified_tone is False


def test_tonal_audibility_rejects_short_input() -> None:
    with pytest.raises(ValueError, match=r"'levels' and 'frequencies' must carry"):
        wind_turbine_tonality([1.0, 2.0], [10.0, 12.0])


def test_tonal_audibility_rejects_two_dimensional_spectrum() -> None:
    # Two spectra stacked into one array: shapes agree, the rank does not.
    freqs = np.tile(np.arange(100.0, 108.0, 2.0), (2, 1))
    levels = np.full_like(freqs, 30.0)
    with pytest.raises(ValueError, match=r"'levels' and 'frequencies' must be 1-D"):
        wind_turbine_tonality(levels, freqs)


def test_tonal_audibility_rejects_mismatched_spectrum() -> None:
    # One level short of its own frequency axis: the message names both.
    with pytest.raises(
        ValueError,
        match=r"wind_turbine_tonality: 'levels' .*'frequencies' .*same shape",
    ):
        wind_turbine_tonality([30.0, 31.0, 32.0], [100.0, 102.0, 104.0, 106.0])


def test_tonal_audibility_rejects_non_monotonic_frequencies() -> None:
    # A non-increasing frequency axis is not a valid narrowband spectrum.
    with pytest.raises(ValueError, match=r"'frequencies' must be strictly increasing"):
        wind_turbine_tonality([30.0, 30.0, 30.0], [100.0, 98.0, 104.0])


def test_tonal_audibility_rejects_non_uniform_spacing() -> None:
    # Formulae 30-34 assume a uniform frequency resolution df. A near-miss axis
    # (2, 2, 2.4 Hz) that a loose 25 % tolerance would accept — but which biases
    # df and the ENBW/masking level — must be rejected by the tight tolerance.
    with pytest.raises(ValueError, match=r"'frequencies' must be uniformly spaced"):
        wind_turbine_tonality([30.0, 30.0, 30.0, 30.0], [100.0, 102.0, 104.0, 106.4])


def test_tonal_audibility_plot_smoke_and_export() -> None:
    import phonometry

    levels, freqs = _synthetic_tone()
    res = phonometry.environment.wind_turbine_tonality(levels, freqs)
    assert res.plot() is not None
    # Caller-supplied plot kwargs must override the defaults, not collide with
    # them (no duplicate-keyword TypeError).
    assert res.plot(lw=2.0, label="spectrum") is not None
    assert (
        phonometry.environment.apparent_sound_power_level is apparent_sound_power_level
    )


def test_two_tone_band_anchors_to_highest_tone_line() -> None:
    """Subclause 9.5.3/9.5.4: the 10 dB tone-line cut and the Formula 34
    frequency anchor to the highest classified tone line, not the probed
    candidate. Candidate 52 dB @ 480 Hz, maximum 60 dB @ 500 Hz, third line
    49.5 dB @ 520 Hz on a 30 dB floor (df = 2 Hz): the 49.5 dB line falls
    outside 10 dB of the 60 dB maximum (candidate-anchored it would count),
    L_pt = 10 lg(10^5.2 + 10^6) = 60.639 dB, L_pn = 30 + 10 lg(CBW(480)/3),
    L_a at 500 Hz, giving dL_a = 17.066 dB at 500 Hz (hand-derived).
    """
    df = 2.0
    freqs = np.arange(420.0, 540.0 + df, df)
    levels = np.full(freqs.size, 30.0)
    levels[int(np.argmin(np.abs(freqs - 480.0)))] = 52.0
    levels[int(np.argmin(np.abs(freqs - 500.0)))] = 60.0
    levels[int(np.argmin(np.abs(freqs - 520.0)))] = 49.5
    res = wind_turbine_tonality(levels, freqs, tone_frequency=480.0)
    assert res.has_identified_tone is True
    assert res.tone_frequency == pytest.approx(500.0)
    assert res.tone_level == pytest.approx(60.639, abs=0.005)
    assert res.tonal_audibility == pytest.approx(17.066, abs=0.005)
    assert res.is_audible is True


def test_no_identified_tone_is_flagged() -> None:
    """A broad 33 dB bump on a 30 dB floor produces no classified tone line
    (9.5.4): the result must say so instead of fabricating a tone, and such
    spectra are excluded from the 9.5.1 bin averaging.
    """
    df = 2.0
    freqs = np.arange(380.0, 620.0 + df, df)
    levels = np.full(freqs.size, 30.0)
    bump = np.abs(freqs - 500.0) <= 20.0
    levels[bump] = 33.0
    res = wind_turbine_tonality(levels, freqs, tone_frequency=500.0)
    assert res.has_identified_tone is False
    assert res.is_audible is False


def test_possible_tone_screen_rejects_non_local_maximum() -> None:
    """9.5.2: a candidate that is not a local maximum is not a possible tone,
    even when strong lines exist elsewhere in the band.
    """
    df = 2.0
    freqs = np.arange(420.0, 580.0 + df, df)
    levels = np.full(freqs.size, 30.0)
    i = int(np.argmin(np.abs(freqs - 500.0)))
    levels[i] = 60.0
    levels[i - 1] = 45.0  # candidate on the flank of the 500 Hz tone
    res = wind_turbine_tonality(levels, freqs, tone_frequency=float(freqs[i - 1]))
    assert res.has_identified_tone is False
    assert res.is_audible is False


def test_truncated_critical_band_warns() -> None:
    df = 2.0
    freqs = np.arange(460.0, 540.0 + df, df)  # narrower than CBW(500) = 117 Hz
    levels = np.full(freqs.size, 30.0)
    levels[int(np.argmin(np.abs(freqs - 500.0)))] = 60.0
    with pytest.warns(
        WindTurbineNoiseWarning, match=r"does not cover the whole critical band"
    ):
        wind_turbine_tonality(levels, freqs, tone_frequency=500.0)


def test_candidate_below_20hz_rejected() -> None:
    df = 1.0
    freqs = np.arange(1.0, 200.0 + df, df)
    levels = np.full(freqs.size, 30.0)
    levels[int(np.argmin(np.abs(freqs - 10.0)))] = 60.0
    with pytest.raises(ValueError, match=r"The candidate tone lies below 20 Hz"):
        wind_turbine_tonality(levels, freqs, tone_frequency=10.0)


@pytest.mark.parametrize(
    "field",
    [
        "tone_frequency",
        "critical_bandwidth",
        "tone_level",
        "masking_level",
        "tonality",
        "audibility_criterion",
        "tonal_audibility",
    ],
)
def test_tonality_result_refuses_a_non_finite_metric(field: str) -> None:
    """Every Formula 30-34 metric is pinned finite, by name, at construction.

    The spectrum is validated finite and each metric descends from it, so no
    producer emits a NaN here. One reaching the fiche prints ``nan dB`` in the
    boxed result beside a confident audibility decision, or dies in the
    metrics table's rounding with ``cannot convert float NaN to integer``.
    """
    import dataclasses

    res = wind_turbine_tonality(*_synthetic_tone())
    with pytest.raises(ValueError, match=rf"'{field}' must be finite"):
        dataclasses.replace(res, **{field: float("nan")})


def test_tonality_result_refuses_an_infinite_metric() -> None:
    """Infinity is refused by the same guard: it overflows the same rounding."""
    import dataclasses

    res = wind_turbine_tonality(*_synthetic_tone())
    with pytest.raises(ValueError, match=r"'tonal_audibility' must be finite"):
        dataclasses.replace(res, tonal_audibility=float("inf"))


def test_slant_distance_vertical_axis() -> None:
    """Formula 2: a vertical-axis turbine measures R0 = H + D."""
    h, d = 30.0, 20.0
    assert slant_distance(h, d, rotor_axis="vertical") == pytest.approx(
        np.hypot(h, h + d)
    )
    with pytest.raises(ValueError, match=r"'rotor_axis' must be"):
        slant_distance(h, d, rotor_axis="diagonal")
