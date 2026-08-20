#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Tests for the Hilbert envelope and instantaneous phase.

Oracles are the closed forms of Bendat & Piersol Chapter 13: the
Table 13.1 pair ``cos → sin`` recovered from the analytic signal, the
exact envelope of a known AM waveform (Eq. 13.27), a unit envelope and
constant instantaneous frequency for a pure tone (Eqs. 13.17-13.19), a
linear instantaneous-frequency ramp for a chirp, and exact consistency of
the plain-subsampling decimation with the ECMA-internal convention.
"""

from __future__ import annotations

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

import phonometry as ph

FS = 8192.0
N = 16384
#: Slice keeping clear of the discrete Hilbert transform's edge effects.
INTERIOR = slice(1024, N - 1024)


def _tone(f0: float, amp: float = 1.0) -> np.ndarray:
    t = np.arange(N) / FS
    return np.asarray(amp * np.cos(2.0 * np.pi * f0 * t), dtype=np.float64)


def _am(fc: float, fm: float, m: float) -> tuple[np.ndarray, np.ndarray]:
    """AM waveform and its exact envelope ``1 + m·cos(2πfm·t)``."""
    t = np.arange(N) / FS
    env = 1.0 + m * np.cos(2.0 * np.pi * fm * t)
    return env * np.cos(2.0 * np.pi * fc * t), env


# ---------------------------------------------------------------------------
# Closed forms (B&P Table 13.1, Eqs. 13.17-13.19, 13.27)
# ---------------------------------------------------------------------------


def test_pure_tone_has_unit_envelope_and_carrier_frequency() -> None:
    res = ph.signals.envelope(_tone(1000.0, amp=2.5), FS)
    np.testing.assert_allclose(res.envelope[INTERIOR], 2.5, atol=1e-9)
    np.testing.assert_allclose(res.instantaneous_frequency[INTERIOR], 1000.0, atol=1e-6)


def test_hilbert_transform_of_cos_is_sin() -> None:
    """Table 13.1: x̃(t) = A(t)·sin(θ(t)) recovers sin(2πf0t) from cos."""
    f0 = 500.0
    res = ph.signals.envelope(_tone(f0), FS)
    t = np.arange(N) / FS
    reconstructed = res.envelope * np.sin(res.phase)
    np.testing.assert_allclose(
        reconstructed[INTERIOR],
        np.sin(2.0 * np.pi * f0 * t)[INTERIOR],
        atol=1e-9,
    )


def test_am_envelope_is_recovered_exactly() -> None:
    """Eq. 13.27: the envelope of u(t)·cos(2πf0t) is u(t) exactly."""
    x, exact = _am(1000.0, 10.0, 0.5)
    res = ph.signals.envelope(x, FS)
    np.testing.assert_allclose(res.envelope[INTERIOR], exact[INTERIOR], atol=1e-9)


def test_instantaneous_phase_of_tone_advances_linearly() -> None:
    f0 = 250.0
    res = ph.signals.envelope(_tone(f0), FS)
    slope = float(np.polyfit(res.times[INTERIOR], res.phase[INTERIOR], 1)[0])
    assert slope / (2.0 * np.pi) == pytest.approx(f0, rel=1e-6)


def test_chirp_instantaneous_frequency_tracks_the_sweep() -> None:
    from scipy.signal import chirp

    t = np.arange(N) / FS
    f0, f1 = 200.0, 2000.0
    x = chirp(t, f0=f0, t1=t[-1], f1=f1, method="linear")
    res = ph.signals.envelope(x, FS)
    expected = f0 + (f1 - f0) * t / t[-1]
    np.testing.assert_allclose(
        res.instantaneous_frequency[INTERIOR], expected[INTERIOR], rtol=0.01
    )


# ---------------------------------------------------------------------------
# Decimation
# ---------------------------------------------------------------------------


def test_plain_subsampling_matches_the_ecma_internal_convention() -> None:
    """antialias=False is exactly |hilbert(x)|[::q] (ECMA Formulae 65/119)."""
    from scipy.signal import hilbert

    x, _ = _am(1000.0, 10.0, 0.5)
    res = ph.signals.envelope(x, FS, decimation_factor=32, antialias=False)
    np.testing.assert_array_equal(res.envelope, np.abs(hilbert(x))[::32])
    assert res.fs == pytest.approx(FS / 32.0)
    assert res.decimation_factor == 32
    assert not res.antialias


def test_antialiased_decimation_tracks_the_smooth_envelope() -> None:
    x, exact = _am(1000.0, 10.0, 0.5)
    res = ph.signals.envelope(x, FS, decimation_factor=16)
    exact_dec = exact[::16]
    inner = slice(64, res.envelope.size - 64)
    np.testing.assert_allclose(res.envelope[inner], exact_dec[inner], atol=1e-3)
    assert res.times[1] - res.times[0] == pytest.approx(16.0 / FS)


def test_decimated_outputs_share_one_time_axis() -> None:
    x, _ = _am(2000.0, 20.0, 0.3)
    res = ph.signals.envelope(x, FS, decimation_factor=8)
    assert res.envelope.size == res.phase.size
    assert res.envelope.size == res.instantaneous_frequency.size
    assert res.envelope.size == res.times.size
    # Phase/instantaneous frequency are subsampled from the full rate.
    full = ph.signals.envelope(x, FS)
    np.testing.assert_array_equal(res.phase, full.phase[::8])
    np.testing.assert_array_equal(
        res.instantaneous_frequency, full.instantaneous_frequency[::8]
    )


def test_no_decimation_keeps_the_full_rate() -> None:
    x, _ = _am(1000.0, 10.0, 0.5)
    res = ph.signals.envelope(x, FS)
    assert res.fs == FS
    assert res.envelope.size == x.size
    assert res.decimation_factor == 1
    np.testing.assert_array_equal(res.signal, x)
    assert res.signal_fs == FS


# ---------------------------------------------------------------------------
# Validation and API surface
# ---------------------------------------------------------------------------


def test_envelope_rejects_invalid_inputs() -> None:
    x, _ = _am(1000.0, 10.0, 0.5)
    two_dimensional = np.stack([x, x])
    non_finite = np.full(4096, np.nan)
    with pytest.raises(ValueError, match="one-dimensional"):
        ph.signals.envelope(two_dimensional, FS)
    with pytest.raises(ValueError, match="'fs'"):
        ph.signals.envelope(x, 0.0)
    with pytest.raises(ValueError, match="decimation_factor"):
        ph.signals.envelope(x, FS, decimation_factor=0)
    with pytest.raises(ValueError, match="decimation_factor"):
        ph.signals.envelope(x, FS, decimation_factor=x.size)
    with pytest.raises(ValueError, match="finite"):
        ph.signals.envelope(non_finite, FS)


def test_short_record_antialiased_decimation_is_supported() -> None:
    """scipy's FIR decimator runs through resample_poly, so even the
    32-sample minimum record decimates without a length cushion."""
    x = np.sin(0.3 * np.arange(32))
    res = ph.signals.envelope(x, FS, decimation_factor=8)
    assert res.envelope.size == 4
    assert res.fs == pytest.approx(FS / 8.0)


def test_result_is_frozen() -> None:
    res = ph.signals.envelope(_tone(1000.0), FS)
    with pytest.raises(AttributeError):
        res.envelope = res.envelope * 2.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# .plot() renderer
# ---------------------------------------------------------------------------


def test_envelope_plot_two_panels_and_external_ax() -> None:
    x, _ = _am(1000.0, 10.0, 0.5)
    res = ph.signals.envelope(x, FS)
    axes = res.plot()
    assert len(axes) == 2
    assert len(axes[0].lines) == 2  # signal + envelope
    _fig, ext = plt.subplots()
    assert res.plot(ax=ext) is ext
    plt.close("all")


def test_envelope_plot_with_decimated_result() -> None:
    x, _ = _am(1000.0, 10.0, 0.5)
    res = ph.signals.envelope(x, FS, decimation_factor=32)
    axes = res.plot()
    assert len(axes) == 2
    plt.close("all")


# ---------------------------------------------------------------------------
# Envelope spectrum (B&P Section 13.3): AM closed forms
# ---------------------------------------------------------------------------
#
# For x(t) = A0*(1 + m*cos(2*pi*fm*t))*cos(2*pi*fc*t) with 0 <= m < 1 the
# Hilbert envelope is exactly A0*(1 + m*cos(2*pi*fm*t)) (Eq. 13.27), so:
#
# * kind="magnitude": mean level A0, one line of amplitude A0*m at fm.
# * kind="squared" (the square-law detector of Figure 13.11):
#   A0^2*(1 + m*cos)^2 = A0^2*[1 + m^2/2 + 2m*cos(2*pi*fm*t)
#   + (m^2/2)*cos(2*pi*2*fm*t)]: mean level A0^2*(1 + m^2/2) and lines
#   2*A0^2*m at fm and A0^2*m^2/2 at 2*fm.
#
# fm is chosen on an exact FFT bin so the line amplitudes are read without
# scalloping; the tolerances absorb only the Hilbert edge effects.

ES_A0 = 2.0
ES_M = 0.35
ES_FM = 13.0  # 13 cycles in the 2 s record: an exact bin
ES_N = N


def _am_signal() -> np.ndarray:
    t = np.arange(ES_N) / FS
    return np.asarray(
        ES_A0
        * (1.0 + ES_M * np.cos(2.0 * np.pi * ES_FM * t))
        * np.cos(2.0 * np.pi * 1000.0 * t),
        dtype=np.float64,
    )


def _bin(freq: float, nfft: int = ES_N) -> int:
    return round(freq * nfft / FS)


def test_envelope_spectrum_magnitude_line_and_mean() -> None:
    res = ph.signals.envelope_spectrum(_am_signal(), FS)
    assert res.kind == "magnitude"
    assert res.mean_level == pytest.approx(ES_A0, abs=1e-3)
    assert res.amplitude[_bin(ES_FM)] == pytest.approx(ES_A0 * ES_M, rel=1e-3)
    assert res.frequencies[_bin(ES_FM)] == pytest.approx(ES_FM)


def test_envelope_spectrum_squared_lines_and_mean() -> None:
    res = ph.signals.envelope_spectrum(_am_signal(), FS, kind="squared")
    assert res.mean_level == pytest.approx(ES_A0**2 * (1.0 + ES_M**2 / 2.0), abs=1e-2)
    assert res.amplitude[_bin(ES_FM)] == pytest.approx(2.0 * ES_A0**2 * ES_M, rel=1e-3)
    assert res.amplitude[_bin(2.0 * ES_FM)] == pytest.approx(
        ES_A0**2 * ES_M**2 / 2.0, rel=1e-2
    )


def test_envelope_spectrum_dc_removal() -> None:
    res = ph.signals.envelope_spectrum(_am_signal(), FS)
    assert res.remove_dc
    assert res.amplitude[0] == pytest.approx(0.0, abs=1e-6)
    kept = ph.signals.envelope_spectrum(_am_signal(), FS, remove_dc=False)
    # Without the DC remover the zero-frequency bin carries the mean level
    # (not doubled), up to the taper's slight leakage.
    assert kept.amplitude[0] == pytest.approx(kept.mean_level, rel=5e-3)
    # ...and the modulation line is unaffected either way.
    assert kept.amplitude[_bin(ES_FM)] == pytest.approx(ES_A0 * ES_M, rel=1e-3)


def test_envelope_spectrum_pure_tone_has_no_lines() -> None:
    res = ph.signals.envelope_spectrum(_tone(1000.0, amp=1.5), FS)
    assert res.mean_level == pytest.approx(1.5, abs=1e-3)
    # No modulation: nothing anywhere above the Hilbert edge-effect floor.
    assert float(np.max(res.amplitude[1:])) < 1e-2


def test_envelope_spectrum_zero_padding() -> None:
    res = ph.signals.envelope_spectrum(_am_signal(), FS, nfft=2 * ES_N)
    assert res.nfft == 2 * ES_N
    assert res.frequencies.size == ES_N + 1
    assert res.amplitude[_bin(ES_FM, 2 * ES_N)] == pytest.approx(ES_A0 * ES_M, rel=1e-2)


def test_envelope_spectrum_off_bin_line_shows_hann_scalloping() -> None:
    """An off-bin modulation frequency reads low by the scalloping loss.

    Midway between bins the Hann taper's scalloping loss is about
    1.42 dB (a factor ~0.85): the documented limit of the on-bin 'exact
    amplitude' calibration.
    """
    df = FS / ES_N
    fm = (round(ES_FM / df) + 0.5) * df  # exactly midway between bins
    t = np.arange(ES_N) / FS
    x = (
        ES_A0
        * (1.0 + ES_M * np.cos(2.0 * np.pi * fm * t))
        * np.cos(2.0 * np.pi * 1000.0 * t)
    )
    res = ph.signals.envelope_spectrum(x, FS)
    peak = float(np.max(res.amplitude[1:]))
    # Hann scalloping at the bin midpoint: |W(1/2)| / |W(0)| ~ 0.8488.
    assert peak == pytest.approx(0.8488 * ES_A0 * ES_M, rel=0.02)
    assert peak < 0.9 * ES_A0 * ES_M


def test_envelope_spectrum_bandpass_prefilter_isolates_the_carrier() -> None:
    """The optional band isolates the modulated carrier before detection.

    With a strong out-of-band interfering tone the raw Hilbert envelope is
    dominated by the two-tone beat; band-passing around the carrier first
    (the classical bearing-envelope chain) recovers the clean modulation
    line at its exact amplitude.
    """
    t = np.arange(ES_N) / FS
    x = _am_signal() + 3.0 * np.cos(2.0 * np.pi * 3000.0 * t)

    raw = ph.signals.envelope_spectrum(x, FS)
    filtered = ph.signals.envelope_spectrum(x, FS, band=(700.0, 1300.0))

    assert filtered.band == (700.0, 1300.0)
    assert raw.band is None
    assert filtered.amplitude[_bin(ES_FM)] == pytest.approx(ES_A0 * ES_M, rel=1e-2)
    assert filtered.mean_level == pytest.approx(ES_A0, rel=1e-2)
    # Without the pre-filter the beat envelope swamps the modulation line.
    beat_error = abs(raw.amplitude[_bin(ES_FM)] - ES_A0 * ES_M)
    assert beat_error > 0.05 * ES_A0 * ES_M


def test_envelope_spectrum_band_validation() -> None:
    x = _am_signal()
    for band in ((0.0, 100.0), (200.0, 100.0), (100.0, FS / 2.0)):
        with pytest.raises(ValueError, match="band"):
            ph.signals.envelope_spectrum(x, FS, band=band)
    # Malformed shapes: not exactly two numeric edges.
    for bad in ((700.0,), (700.0, 900.0, 1300.0), (700.0, None)):
        with pytest.raises(ValueError, match="pair"):
            ph.signals.envelope_spectrum(x, FS, band=bad)  # type: ignore[arg-type]
    # A record shorter than the zero-phase padding fails informatively.
    with pytest.raises(ValueError, match="too short"):
        ph.signals.envelope_spectrum(x[:20], FS, band=(700.0, 1300.0))


def test_envelope_spectrum_validates_inputs() -> None:
    x = _am_signal()
    with pytest.raises(ValueError, match="kind"):
        ph.signals.envelope_spectrum(x, FS, kind="log")
    with pytest.raises(ValueError, match="nfft"):
        ph.signals.envelope_spectrum(x, FS, nfft=100)
    with pytest.raises(ValueError, match="fs"):
        ph.signals.envelope_spectrum(x, -1.0)


def test_envelope_spectrum_plot_two_panels_and_external_ax() -> None:
    res = ph.signals.envelope_spectrum(_am_signal(), FS)
    axes = res.plot()
    assert len(axes) == 2
    assert len(axes[0].lines) >= 1  # envelope (+ mean-level rule)
    _fig, ext = plt.subplots()
    assert res.plot(ax=ext) is ext
    plt.close("all")
