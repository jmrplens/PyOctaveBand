#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Tests for pile-driving underwater sound metrics (ISO 18406).

The single-strike SEL matches the underwater SEL primitive; the cumulative SEL
is the exact energy sum, equal to SEL_ss + 10·lg(N) for identical strikes.
"""

from __future__ import annotations

import dataclasses

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from phonometry import underwater

FS = 48000


def _pulse(amplitude: float, seconds: float) -> np.ndarray:
    # A decaying sinusoidal burst, the shape of an impulsive strike.
    t = np.arange(round(seconds * FS)) / FS
    return amplitude * np.exp(-t / (0.3 * seconds)) * np.sin(2 * np.pi * 200.0 * t)


def test_single_strike_sel_matches_primitive() -> None:
    x = _pulse(50.0, 0.2)
    assert underwater.single_strike_sel(x, FS) == pytest.approx(
        underwater.sound_exposure_level(x, FS)
    )


def test_cumulative_sel_identical_strikes() -> None:
    # N identical strikes -> SEL_cum = SEL_ss + 10 lg(N).
    sel_ss = 180.0
    assert underwater.cumulative_sel_identical(sel_ss, 10) == pytest.approx(
        sel_ss + 10.0
    )
    assert underwater.cumulative_sel_identical(sel_ss, 100) == pytest.approx(
        sel_ss + 20.0
    )
    assert underwater.cumulative_sel_identical(sel_ss, 1) == pytest.approx(sel_ss)


def test_cumulative_sel_energy_sum_matches_identical() -> None:
    sel_ss = 175.0
    n = 8
    assert underwater.cumulative_sel([sel_ss] * n) == pytest.approx(
        underwater.cumulative_sel_identical(sel_ss, n)
    )


def test_cumulative_sel_differing_strikes() -> None:
    sels = [170.0, 176.0, 173.0]
    expected = 10.0 * np.log10(sum(10.0 ** (s / 10.0) for s in sels))
    assert underwater.cumulative_sel(sels) == pytest.approx(expected)


def test_cumulative_sel_rejects_empty() -> None:
    with pytest.raises(ValueError, match="'single_sels' must be a non-empty"):
        underwater.cumulative_sel([])


def test_cumulative_sel_identical_rejects_zero_strikes() -> None:
    with pytest.raises(ValueError, match="'n_strikes' must be at least"):
        underwater.cumulative_sel_identical(180.0, 0)


def test_cumulative_sel_identical_rejects_fractional_strikes() -> None:
    # A non-integer count must be rejected, not silently truncated to int().
    with pytest.raises(ValueError, match="'n_strikes' must be a whole number"):
        underwater.cumulative_sel_identical(180.0, 1.9)  # type: ignore[arg-type]


def test_pile_strike_metrics_bundle_and_plot() -> None:
    x = _pulse(100.0, 0.25)
    res = underwater.pile_strike_metrics(x, FS)
    assert isinstance(res, underwater.PileStrikeResult)
    assert res.single_strike_sel == pytest.approx(
        underwater.sound_exposure_level(x, FS)
    )
    assert res.peak_spl == pytest.approx(underwater.peak_sound_pressure_level(x))
    assert 0.0 < res.pulse_duration < 0.25
    axes = res.plot()
    assert len(axes) == 2


def test_pile_strike_metrics_rejects_short_signal() -> None:
    too_short = np.array([1.0])
    with pytest.raises(ValueError, match="'pressure' must contain at least"):
        underwater.pile_strike_metrics(too_short, FS)


def test_pile_strike_result_rejects_a_peak_level_its_trace_never_reached() -> None:
    """``peak_spl`` is the peak of the stored trace, and the figure marks it there.

    The waveform panel plots the sample at ``argmax(|pressure|)`` and labels
    that marker with ``peak_spl``, so a level raised by 12 dB used to print
    itself over the one sample that disproves it.
    """
    res = underwater.pile_strike_metrics(_pulse(100.0, 0.25), FS)
    with pytest.raises(ValueError, match="'peak_spl' must be the zero-to-peak"):
        dataclasses.replace(res, peak_spl=res.peak_spl + 12.0)


def test_pile_strike_result_rejects_a_trace_the_peak_no_longer_summarises() -> None:
    """Substituting the waveform without restating the peak is the same lie."""
    res = underwater.pile_strike_metrics(_pulse(100.0, 0.25), FS)
    halved = np.asarray(res.pressure) / 2.0
    with pytest.raises(ValueError, match="'peak_spl' must be the zero-to-peak"):
        dataclasses.replace(res, pressure=halved)


def test_pile_strike_result_accepts_a_variant_that_restates_every_metric() -> None:
    """A substituted waveform is legitimate once all four numbers follow it.

    The peak is recomputed here along a path of its own rather than through
    the library's, so the tolerance is exercised too: the two agree to the
    last bit but need not, and a caller who reached the same level another way
    must not be refused over it.
    """
    res = underwater.pile_strike_metrics(_pulse(100.0, 0.25), FS)
    halved = np.asarray(res.pressure) / 2.0
    variant = dataclasses.replace(
        res,
        pressure=halved,
        peak_spl=20.0 * np.log10(float(np.max(np.abs(halved))) / 1e-6),
        single_strike_sel=underwater.sound_exposure_level(halved, FS),
        spl=underwater.sound_pressure_level(halved),
    )
    assert variant.peak_spl == pytest.approx(res.peak_spl - 20.0 * np.log10(2.0))
    assert variant.single_strike_sel == pytest.approx(
        res.single_strike_sel - 20.0 * np.log10(2.0)
    )


def test_pile_strike_result_rejects_an_empty_trace() -> None:
    res = underwater.pile_strike_metrics(_pulse(100.0, 0.25), FS)
    with pytest.raises(ValueError, match="'pressure' must carry at least one sample"):
        dataclasses.replace(res, pressure=np.array([]))


def test_pile_strike_result_accepts_a_silent_trace_peaking_at_minus_infinity() -> None:
    """Zero pressure peaks at ``-inf`` dB, and ``-inf`` is what restates it.

    The producer refuses a silent record outright, but the guard must not
    answer an undetermined peak with a crash or a spurious contradiction: it
    is the same neutral value ``band_sel`` carries for an empty band.
    """
    silent = underwater.PileStrikeResult(
        single_strike_sel=-np.inf,
        peak_spl=-np.inf,
        spl=-np.inf,
        pulse_duration=0.0,
        pressure=np.zeros(16),
        fs=float(FS),
    )
    assert silent.peak_spl == -np.inf


# ---------------------------------------------------------------------------
# Band-resolved single-strike SEL and the marine-mammal assessment chain
# ---------------------------------------------------------------------------


def test_band_sel_energy_sum_reproduces_the_broadband_sel() -> None:
    """Parseval: the band energies sum to the total sound exposure of the record."""
    x = _pulse(50.0, 0.2)
    spec = underwater.strike_sel_spectrum(x, FS, fraction=3, limits=(10.0, 20_000.0))
    assert spec.total_sel == pytest.approx(spec.broadband_sel, abs=0.05)
    assert spec.total_sel == pytest.approx(
        underwater.single_strike_sel(x, FS), abs=0.05
    )


def test_band_sel_octave_and_third_octave_agree_on_the_total() -> None:
    x = _pulse(50.0, 0.2)
    octaves = underwater.strike_sel_spectrum(x, FS, fraction=1)
    thirds = underwater.strike_sel_spectrum(x, FS, fraction=3)
    assert octaves.total_sel == pytest.approx(thirds.total_sel, abs=0.05)
    assert octaves.frequencies.size < thirds.frequencies.size


def test_band_sel_peaks_at_the_tone_frequency() -> None:
    """The synthetic strike is a 200 Hz burst, so the 200 Hz band dominates."""
    spec = underwater.strike_sel_spectrum(_pulse(50.0, 0.2), FS, fraction=3)
    peak = spec.frequencies[int(np.nanargmax(spec.band_sel))]
    assert peak == pytest.approx(200.0, rel=0.3)


def test_weighted_exposure_of_a_pile_driving_campaign() -> None:
    """The pile-driving output feeds the regulatory weighting end to end.

    A 200 Hz strike sits inside the LF cetacean weighting passband and far
    outside the VHF one, so the same campaign weights very differently for the
    two groups; the accumulation over strikes is the ISO 18406 +10·lg(N).
    """
    spec = underwater.strike_sel_spectrum(_pulse(50.0, 0.2), FS, fraction=3)
    # No masking: the spectrum goes into the assessment exactly as returned.
    lf = underwater.weighted_exposure(
        spec.frequencies, spec.band_sel, "LF", n_events=1000, impulsive=True
    )
    vhf = underwater.weighted_exposure(
        spec.frequencies, spec.band_sel, "VHF", n_events=1000, impulsive=True
    )
    assert lf.weighted_sel > vhf.weighted_sel + 20.0
    assert lf.cumulative_sel == pytest.approx(lf.weighted_sel + 30.0, abs=1e-9)
    assert lf.criteria.injury_label == "AUD INJ"


@pytest.mark.parametrize(
    ("seconds", "fs"), [(1.0, 48000), (0.2, 48000), (0.1, 48000), (0.5, 10000)]
)
def test_the_advertised_chain_runs_unaided_on_short_records(
    seconds: float, fs: int
) -> None:
    """Bands narrower than the bin spacing fs/n are empty; the chain must still run.

    A one-third-octave band below about ``fs/n`` contains no discrete-spectrum
    bin, which is the normal case for a single strike: at 48 kHz a 0.1 s record
    empties three of the 34 bands. Those bands carry no energy, are reported as
    ``-inf`` and must pass through the weighting without being masked out.
    """
    t = np.arange(round(seconds * fs)) / fs
    strike = 50.0 * np.exp(-t / (0.3 * seconds)) * np.sin(2 * np.pi * 200.0 * t)
    spec = underwater.strike_sel_spectrum(strike, fs, fraction=3)
    assert not np.any(np.isnan(spec.band_sel))
    res = underwater.weighted_exposure(
        spec.frequencies, spec.band_sel, "LF", impulsive=True
    )
    # Empty bands are the neutral element: dropping them changes nothing.
    finite = np.isfinite(spec.band_sel)
    masked = underwater.weighted_exposure(
        spec.frequencies[finite], spec.band_sel[finite], "LF", impulsive=True
    )
    assert res.weighted_sel == pytest.approx(masked.weighted_sel, abs=1e-9)
    assert res.unweighted_sel == pytest.approx(spec.total_sel, abs=1e-9)


def test_empty_bands_are_reported_as_minus_infinity() -> None:
    """A 0.5 s record at 10 kHz empties the lowest bands (bin spacing 2 Hz)."""
    fs = 10_000
    t = np.arange(round(0.5 * fs)) / fs
    strike = 50.0 * np.exp(-t / 0.15) * np.sin(2 * np.pi * 200.0 * t)
    spec = underwater.strike_sel_spectrum(strike, fs, fraction=3)
    empty = ~np.isfinite(spec.band_sel)
    assert empty.any()
    assert np.all(spec.band_sel[empty] == -np.inf)


def test_strike_sel_spectrum_validates_its_arguments() -> None:
    x = _pulse(50.0, 0.2)
    silence = np.zeros(1024)
    with pytest.raises(ValueError, match="fraction"):
        underwater.strike_sel_spectrum(x, FS, fraction=6)
    with pytest.raises(ValueError, match="limits"):
        underwater.strike_sel_spectrum(x, FS, limits=(2000.0, 100.0))
    with pytest.raises(ValueError, match="no energy"):
        underwater.strike_sel_spectrum(silence, FS)


def test_strike_sel_spectrum_rejects_a_non_integer_fraction() -> None:
    # A non-integer fraction must be rejected, not silently truncated to int().
    x = _pulse(50.0, 0.2)
    with pytest.raises(ValueError, match="'fraction' must be 1"):
        underwater.strike_sel_spectrum(x, FS, fraction=3.9)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="'fraction' must be 1"):
        underwater.strike_sel_spectrum(x, FS, fraction=1.4)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="'fraction' must be 1"):
        underwater.strike_sel_spectrum(x, FS, fraction=None)  # type: ignore[arg-type]


def test_strike_sel_spectrum_rejects_a_malformed_limits_pair() -> None:
    # Anything but a two-element pair must be refused by name, not indexed
    # into an IndexError/TypeError or silently truncated to its first two.
    x = _pulse(50.0, 0.2)
    with pytest.raises(ValueError, match=r"'limits' must be a \(lower, upper\) pair"):
        underwater.strike_sel_spectrum(x, FS, limits=(10.0,))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=r"'limits' must be a \(lower, upper\) pair"):
        underwater.strike_sel_spectrum(x, FS, limits=None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=r"'limits' must be a \(lower, upper\) pair"):
        underwater.strike_sel_spectrum(
            x,
            FS,
            limits=(10.0, 20_000.0, 99.0),  # type: ignore[arg-type]
        )


def test_strike_sel_spectrum_plot_returns_axes() -> None:
    spec = underwater.strike_sel_spectrum(_pulse(50.0, 0.2), FS)
    assert spec.plot() is not None
    assert spec.plot(language="es") is not None
    plt.close("all")


def test_total_sel_rejects_a_column_it_no_longer_totals() -> None:
    """A mitigated band column keeps the unmitigated total, and the plot draws it.

    A bubble curtain taking 10 dB out of every band is substituted the
    sanctioned way, through :func:`dataclasses.replace`. The figure rules its
    dashed SEL_ss line at ``total_sel``, so the old total used to be drawn
    17 dB above the loudest band in the column beneath it.
    """
    spec = underwater.strike_sel_spectrum(_pulse(50.0, 0.2), FS)
    mitigated = np.asarray(spec.band_sel) - 10.0
    with pytest.raises(ValueError, match="'total_sel' must be the energy sum"):
        dataclasses.replace(spec, band_sel=mitigated)


def test_total_sel_is_the_energy_sum_and_not_the_arithmetic_one() -> None:
    """Adding the decibels themselves totals a strike at thousands of dB."""
    spec = underwater.strike_sel_spectrum(_pulse(50.0, 0.2), FS)
    bands = np.asarray(spec.band_sel)
    arithmetic = float(np.sum(bands[np.isfinite(bands)]))
    assert arithmetic > spec.total_sel + 1000.0
    with pytest.raises(ValueError, match="'total_sel' must be the energy sum"):
        dataclasses.replace(spec, total_sel=arithmetic)


def test_total_sel_accepts_a_mitigated_column_restated() -> None:
    """A uniform 10 dB attenuation moves the energy sum by exactly 10 dB."""
    spec = underwater.strike_sel_spectrum(_pulse(50.0, 0.2), FS)
    mitigated = np.asarray(spec.band_sel) - 10.0
    variant = dataclasses.replace(
        spec, band_sel=mitigated, total_sel=spec.total_sel - 10.0
    )
    assert variant.total_sel == pytest.approx(spec.total_sel - 10.0)


def test_total_sel_admits_empty_bands_but_not_a_nan_one() -> None:
    """``-inf`` is the level of zero exposure; nothing else non-finite is a level."""
    spec = underwater.strike_sel_spectrum(_pulse(50.0, 0.2), FS)
    assert np.any(np.isneginf(spec.band_sel))  # the guard passed them already
    spoiled = np.asarray(spec.band_sel).copy()
    spoiled[int(np.argmax(spoiled))] = np.nan
    with pytest.raises(ValueError, match="'band_sel' must be finite, or -inf"):
        dataclasses.replace(spec, band_sel=spoiled)


def test_total_sel_of_an_all_empty_column_is_minus_infinity() -> None:
    """Every band empty sums to ``-inf``, which is the truthful total over it."""
    spectrum = underwater.StrikeSelSpectrum(
        frequencies=np.array([100.0, 125.0, 160.0]),
        band_sel=np.full(3, -np.inf),
        total_sel=-np.inf,
        broadband_sel=140.0,
        fraction=3,
        fs=float(FS),
    )
    assert spectrum.total_sel == -np.inf


def test_a_strike_whose_trace_is_a_bare_number_is_refused() -> None:
    """One sample is a trace; one number is not, and the peak check cannot tell.

    The shared rank helper waives its pin when every field it was handed is a
    bare number, an exemption written for the entry points that answer in
    scalars. ``PileStrikeResult`` lists only ``pressure``, so a lone number
    satisfied the whole set and walked past it, carrying a size of one and a
    peak of its own that the comparison beside it was happy to confirm.
    """
    res = underwater.pile_strike_metrics(_pulse(100.0, 0.25), FS)
    lone = np.float64(np.max(np.abs(res.pressure)))
    with pytest.raises(
        ValueError, match=r"'pressure' must be a one-dimensional waveform"
    ):
        dataclasses.replace(res, pressure=lone)


def test_restating_the_peak_alone_does_not_save_the_other_three() -> None:
    """The dangerous half, because it looks like diligence.

    Halve a trace and the peak level follows it down by six decibels, so a
    caller who recomputes that one sees a result agreeing with itself where
    they looked. The exposure and the pressure level beside it are still the
    old trace's, six decibels high, and the marine-mammal dual-metric rule is
    decided on the peak and the exposure together.
    """
    res = underwater.pile_strike_metrics(_pulse(100.0, 0.25), FS)
    halved = np.asarray(res.pressure) / 2.0
    with pytest.raises(ValueError, match=r"'single_strike_sel' must be the exposure"):
        dataclasses.replace(
            res,
            pressure=halved,
            peak_spl=underwater.peak_sound_pressure_level(halved),
        )


def test_a_pulse_duration_left_behind_by_a_new_shape_is_refused() -> None:
    """The quiet one: a ratio of energies does not move when a trace is scaled.

    Only a change of shape parts it from its waveform, which is why scaling
    tests pass it over and it needs a guard of its own rather than being
    excused by one. Here the other three are restated and it alone is stale.
    """
    res = underwater.pile_strike_metrics(_pulse(100.0, 0.25), FS)
    reshaped = _pulse(100.0, 0.05)
    with pytest.raises(ValueError, match=r"'pulse_duration' must be the 5 % to 95 %"):
        dataclasses.replace(
            res,
            pressure=reshaped,
            peak_spl=underwater.peak_sound_pressure_level(reshaped),
            single_strike_sel=underwater.sound_exposure_level(reshaped, FS),
            spl=underwater.sound_pressure_level(reshaped),
        )
