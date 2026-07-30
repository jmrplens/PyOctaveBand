#  Copyright (c) 2026. Jose M. Requena-Plens
"""Tests for pile-driving underwater sound metrics (ISO 18406).

The single-strike SEL matches the underwater SEL primitive; the cumulative SEL
is the exact energy sum, equal to SEL_ss + 10·lg(N) for identical strikes.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from phonometry import (
    PileStrikeResult,
    cumulative_sel,
    cumulative_sel_identical,
    peak_sound_pressure_level,
    pile_strike_metrics,
    single_strike_sel,
    sound_exposure_level,
    strike_sel_spectrum,
    weighted_exposure,
)

FS = 48000


def _pulse(amplitude: float, seconds: float) -> np.ndarray:
    # A decaying sinusoidal burst, the shape of an impulsive strike.
    t = np.arange(round(seconds * FS)) / FS
    return amplitude * np.exp(-t / (0.3 * seconds)) * np.sin(2 * np.pi * 200.0 * t)


def test_single_strike_sel_matches_primitive() -> None:
    x = _pulse(50.0, 0.2)
    assert single_strike_sel(x, FS) == pytest.approx(sound_exposure_level(x, FS))


def test_cumulative_sel_identical_strikes() -> None:
    # N identical strikes -> SEL_cum = SEL_ss + 10 lg(N).
    sel_ss = 180.0
    assert cumulative_sel_identical(sel_ss, 10) == pytest.approx(sel_ss + 10.0)
    assert cumulative_sel_identical(sel_ss, 100) == pytest.approx(sel_ss + 20.0)
    assert cumulative_sel_identical(sel_ss, 1) == pytest.approx(sel_ss)


def test_cumulative_sel_energy_sum_matches_identical() -> None:
    sel_ss = 175.0
    n = 8
    assert cumulative_sel([sel_ss] * n) == pytest.approx(
        cumulative_sel_identical(sel_ss, n)
    )


def test_cumulative_sel_differing_strikes() -> None:
    sels = [170.0, 176.0, 173.0]
    expected = 10.0 * np.log10(sum(10.0 ** (s / 10.0) for s in sels))
    assert cumulative_sel(sels) == pytest.approx(expected)


def test_cumulative_sel_rejects_empty() -> None:
    with pytest.raises(ValueError):
        cumulative_sel([])


def test_cumulative_sel_identical_rejects_zero_strikes() -> None:
    with pytest.raises(ValueError):
        cumulative_sel_identical(180.0, 0)


def test_cumulative_sel_identical_rejects_fractional_strikes() -> None:
    # A non-integer count must be rejected, not silently truncated to int().
    with pytest.raises(ValueError):
        cumulative_sel_identical(180.0, 1.9)  # type: ignore[arg-type]


def test_pile_strike_metrics_bundle_and_plot() -> None:
    x = _pulse(100.0, 0.25)
    res = pile_strike_metrics(x, FS)
    assert isinstance(res, PileStrikeResult)
    assert res.single_strike_sel == pytest.approx(sound_exposure_level(x, FS))
    assert res.peak_spl == pytest.approx(peak_sound_pressure_level(x))
    assert 0.0 < res.pulse_duration < 0.25
    axes = res.plot()
    assert len(axes) == 2


def test_pile_strike_metrics_rejects_short_signal() -> None:
    with pytest.raises(ValueError):
        pile_strike_metrics(np.array([1.0]), FS)


# ---------------------------------------------------------------------------
# Band-resolved single-strike SEL and the marine-mammal assessment chain
# ---------------------------------------------------------------------------


def test_band_sel_energy_sum_reproduces_the_broadband_sel() -> None:
    """Parseval: the band energies sum to the total sound exposure of the record."""
    x = _pulse(50.0, 0.2)
    spec = strike_sel_spectrum(x, FS, fraction=3, limits=(10.0, 20_000.0))
    assert spec.total_sel == pytest.approx(spec.broadband_sel, abs=0.05)
    assert spec.total_sel == pytest.approx(single_strike_sel(x, FS), abs=0.05)


def test_band_sel_octave_and_third_octave_agree_on_the_total() -> None:
    x = _pulse(50.0, 0.2)
    octaves = strike_sel_spectrum(x, FS, fraction=1)
    thirds = strike_sel_spectrum(x, FS, fraction=3)
    assert octaves.total_sel == pytest.approx(thirds.total_sel, abs=0.05)
    assert octaves.frequencies.size < thirds.frequencies.size


def test_band_sel_peaks_at_the_tone_frequency() -> None:
    """The synthetic strike is a 200 Hz burst, so the 200 Hz band dominates."""
    spec = strike_sel_spectrum(_pulse(50.0, 0.2), FS, fraction=3)
    peak = spec.frequencies[int(np.nanargmax(spec.band_sel))]
    assert peak == pytest.approx(200.0, rel=0.3)


def test_weighted_exposure_of_a_pile_driving_campaign() -> None:
    """The pile-driving output feeds the regulatory weighting end to end.

    A 200 Hz strike sits inside the LF cetacean weighting passband and far
    outside the VHF one, so the same campaign weights very differently for the
    two groups; the accumulation over strikes is the ISO 18406 +10·lg(N).
    """
    spec = strike_sel_spectrum(_pulse(50.0, 0.2), FS, fraction=3)
    # No masking: the spectrum goes into the assessment exactly as returned.
    lf = weighted_exposure(spec.frequencies, spec.band_sel, "LF",
                           n_events=1000, impulsive=True)
    vhf = weighted_exposure(spec.frequencies, spec.band_sel, "VHF",
                            n_events=1000, impulsive=True)
    assert lf.weighted_sel > vhf.weighted_sel + 20.0
    assert lf.cumulative_sel == pytest.approx(lf.weighted_sel + 30.0, abs=1e-9)
    assert lf.criteria.injury_label == "AUD INJ"


@pytest.mark.parametrize(
    ("seconds", "fs"), [(1.0, 48000), (0.2, 48000), (0.1, 48000), (0.5, 10000)]
)
def test_the_advertised_chain_runs_unaided_on_short_records(seconds: float, fs: int) -> None:
    """Bands narrower than the bin spacing fs/n are empty; the chain must still run.

    A one-third-octave band below about ``fs/n`` contains no discrete-spectrum
    bin, which is the normal case for a single strike: at 48 kHz a 0.1 s record
    empties three of the 34 bands. Those bands carry no energy, are reported as
    ``-inf`` and must pass through the weighting without being masked out.
    """
    t = np.arange(round(seconds * fs)) / fs
    strike = 50.0 * np.exp(-t / (0.3 * seconds)) * np.sin(2 * np.pi * 200.0 * t)
    spec = strike_sel_spectrum(strike, fs, fraction=3)
    assert not np.any(np.isnan(spec.band_sel))
    res = weighted_exposure(spec.frequencies, spec.band_sel, "LF", impulsive=True)
    # Empty bands are the neutral element: dropping them changes nothing.
    finite = np.isfinite(spec.band_sel)
    masked = weighted_exposure(spec.frequencies[finite], spec.band_sel[finite],
                               "LF", impulsive=True)
    assert res.weighted_sel == pytest.approx(masked.weighted_sel, abs=1e-9)
    assert res.unweighted_sel == pytest.approx(spec.total_sel, abs=1e-9)


def test_empty_bands_are_reported_as_minus_infinity() -> None:
    """A 0.5 s record at 10 kHz empties the lowest bands (bin spacing 2 Hz)."""
    fs = 10_000
    t = np.arange(round(0.5 * fs)) / fs
    strike = 50.0 * np.exp(-t / 0.15) * np.sin(2 * np.pi * 200.0 * t)
    spec = strike_sel_spectrum(strike, fs, fraction=3)
    empty = ~np.isfinite(spec.band_sel)
    assert empty.any()
    assert np.all(spec.band_sel[empty] == -np.inf)


def test_strike_sel_spectrum_validates_its_arguments() -> None:
    x = _pulse(50.0, 0.2)
    silence = np.zeros(1024)
    with pytest.raises(ValueError, match="fraction"):
        strike_sel_spectrum(x, FS, fraction=6)
    with pytest.raises(ValueError, match="limits"):
        strike_sel_spectrum(x, FS, limits=(2000.0, 100.0))
    with pytest.raises(ValueError, match="no energy"):
        strike_sel_spectrum(silence, FS)


def test_strike_sel_spectrum_plot_returns_axes() -> None:
    spec = strike_sel_spectrum(_pulse(50.0, 0.2), FS)
    assert spec.plot() is not None
    assert spec.plot(language="es") is not None
    plt.close("all")
