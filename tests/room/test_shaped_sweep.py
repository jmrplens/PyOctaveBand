#  Copyright (c) 2026. Jose Manuel Requena Plens
"""
Tests for the arbitrary-magnitude sweep synthesis (Mueller & Massarani
2001, Secs. 4.2-4.3).

Validation strategy (closed-form, not self-consistency):
- The measured (Welch) spectrum of the synthesized sweep follows the
  requested target magnitude within a stated dB tolerance across the band
  interior -- pinned for the pink target (the paper's canonical case, its
  Sec. 4) and for an arbitrary tilted target.
- The group delay is the paper's Eqs. (11)-(12): it grows in proportion to
  the cumulative target power and spans exactly ``seconds``.
- The crest factor stays near the swept-sine ideal (Sec. 4.3: "normally
  below 4 dB") regardless of the spectral shape.
- The synthesized sweep deconvolves through the existing
  ``impulse_response`` machinery: a known IIR system measured with the
  shaped sweep is recovered within 0.1 dB in band.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import signal

from phonometry import (
    ShapedSweepResult,
    impulse_response,
    shaped_sweep_signal,
)

FS = 48000


def _welch_deviation_db(res: ShapedSweepResult, target_db_fn) -> float:
    """Worst |Welch - target| deviation in dB over the band interior."""
    x = np.asarray(res)
    nperseg = 8192
    # 75 % overlap: at 50 % the squared Hann windows do not sum to a
    # constant and the sweep maps that power ripple onto the spectrum.
    freqs, psd = signal.welch(
        x, fs=res.fs, nperseg=nperseg, noverlap=3 * nperseg // 4
    )
    third = 2.0 ** (1.0 / 3.0)
    band = (freqs >= res.f_range[0] * third) & (
        freqs <= res.f_range[1] / third
    )
    measured_db = 10.0 * np.log10(psd[band])
    target_db = target_db_fn(freqs[band])
    diff = measured_db - target_db
    return float(np.max(np.abs(diff - np.median(diff))))


# --------------------------------------------------------------------------
# Spectrum follows the target (the paper's central claim)
# --------------------------------------------------------------------------
def test_pink_sweep_spectrum_follows_the_pink_target() -> None:
    res = shaped_sweep_signal(FS, 50.0, 5000.0, 2.0, target="pink")
    dev = _welch_deviation_db(res, lambda f: -10.0 * np.log10(f / 50.0))
    assert dev < 0.5


def test_white_sweep_spectrum_is_flat() -> None:
    # A white sweep reaches its low band edge right at the record start, so
    # a longer lead-in keeps the onset clear of the Welch estimator's edge
    # bias (the synthesis magnitude itself is exact either way, below).
    res = shaped_sweep_signal(
        FS, 100.0, 10000.0, 1.5, target="white", start_delay=0.15
    )
    dev = _welch_deviation_db(res, lambda f: np.zeros_like(f))
    assert dev < 0.5


def test_truncated_sweep_keeps_the_exact_synthesis_magnitude() -> None:
    # The stronger, estimator-free oracle: the FFT magnitude of the
    # *returned* (truncated, faded) samples still matches the target
    # within a few hundredths of a dB across the band interior.
    res = shaped_sweep_signal(FS, 100.0, 10000.0, 1.5, target="white")
    n = 2 * (res.frequencies.size - 1)
    mag_db = 20.0 * np.log10(np.abs(np.fft.rfft(np.asarray(res), n)))
    freqs = res.frequencies
    third = 2.0 ** (1.0 / 3.0)
    band = (freqs >= 100.0 * third) & (freqs <= 10000.0 / third)
    dev = mag_db[band] - np.median(mag_db[band])
    # 1/12-octave smoothing removes the bin-to-bin truncation ripple.
    fb = freqs[band]
    step = 2.0 ** (1.0 / 24.0)
    centres = 130.0 * 2.0 ** (np.arange(19) / 3.0)
    centres = centres[centres <= 10000.0 / third]
    smoothed = np.array([
        float(np.mean(dev[(fb >= fc / step) & (fb <= fc * step)]))
        for fc in centres
    ])
    assert float(np.max(np.abs(smoothed))) < 0.05


def test_arbitrary_target_is_followed() -> None:
    # A tilted target with a mid-band emphasis step (in dB).
    t_freq = np.array([50.0, 200.0, 500.0, 2000.0, 8000.0])
    t_db = np.array([10.0, 10.0, 0.0, 0.0, -6.0])
    res = shaped_sweep_signal(
        FS, 50.0, 8000.0, 2.0, target=(t_freq, t_db)
    )

    def target(f: np.ndarray) -> np.ndarray:
        return np.interp(np.log10(f), np.log10(t_freq), t_db)

    dev = _welch_deviation_db(res, target)
    assert dev < 0.5


# --------------------------------------------------------------------------
# Group delay (Eqs. (11)-(12)) and crest factor (Sec. 4.3)
# --------------------------------------------------------------------------
def test_group_delay_grows_with_cumulative_target_power() -> None:
    res = shaped_sweep_signal(FS, 50.0, 5000.0, 2.0, target="pink")
    tau = res.group_delay
    power = res.magnitude ** 2
    lead = tau[0]
    expected = lead + 2.0 * np.cumsum(power) / float(np.sum(power))
    np.testing.assert_allclose(tau, expected, atol=1e-12)
    # The group delay spans exactly 'seconds' end to end.
    assert tau[-1] - tau[0] == pytest.approx(2.0, abs=1e-12)
    assert np.all(np.diff(tau) >= 0.0)


def test_crest_factor_stays_near_the_swept_sine_ideal() -> None:
    for target in ("pink", "white"):
        res = shaped_sweep_signal(FS, 50.0, 5000.0, 2.0, target=target)
        assert 3.0 < res.crest_factor_db < 4.5


def test_crest_factor_is_finite_when_the_core_window_collapses() -> None:
    # A tiny 'seconds' next to a dominant 'start_delay' rounds the two edges
    # of the constant-envelope window onto the same sample, leaving an empty
    # core slice; the crest factor must still come out finite (the statistic
    # falls back to the whole retained sweep) rather than NaN/inf.
    res = shaped_sweep_signal(FS, 50.0, 5000.0, 1e-6, start_delay=1.0)
    assert np.isfinite(res.crest_factor_db)
    assert res.crest_factor_db > 0.0


def test_signal_length_amplitude_and_fades() -> None:
    res = shaped_sweep_signal(
        FS, 100.0, 10000.0, 1.0, amplitude=0.5, start_delay=0.1
    )
    assert res.size == round(1.2 * FS)  # seconds + 2*start_delay
    assert np.max(np.abs(np.asarray(res))) == pytest.approx(0.5, abs=1e-9)
    assert abs(res.signal[0]) < 1e-6
    assert abs(res.signal[-1]) < 1e-6


# --------------------------------------------------------------------------
# Integration with the deconvolution front end
# --------------------------------------------------------------------------
def test_shaped_sweep_deconvolves_a_known_system() -> None:
    res = shaped_sweep_signal(FS, 50.0, 5000.0, 2.0, target="pink")
    sweep = np.asarray(res)
    b, a = signal.butter(2, [200.0, 4000.0], btype="bandpass", fs=FS)
    excitation = np.concatenate([sweep, np.zeros(16384)])
    recorded = signal.lfilter(b, a, excitation)
    # return_full keeps the (tiny, band-limiting) anticausal tail of the
    # deconvolution, so the recovered spectrum is compared without a
    # truncation bias.
    ir = impulse_response(recorded, excitation, FS, return_full=True)
    freqs = np.fft.rfftfreq(ir.size, 1.0 / FS)
    h_est = np.fft.rfft(np.asarray(ir))
    _, h_true = signal.freqz(b, a, worN=freqs, fs=FS)
    band = (freqs >= 100.0) & (freqs <= 4500.0)
    err_db = 20.0 * np.log10(
        np.abs(h_est[band]) / np.abs(h_true[band])
    )
    assert float(np.max(np.abs(err_db))) < 0.01


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------
def test_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError, match="f1"):
        shaped_sweep_signal(FS, 0.0, 5000.0, 1.0)
    with pytest.raises(ValueError, match="f2"):
        shaped_sweep_signal(FS, 500.0, 100.0, 1.0)
    with pytest.raises(ValueError, match="Nyquist"):
        shaped_sweep_signal(FS, 50.0, 30000.0, 1.0)
    with pytest.raises(ValueError, match="seconds"):
        shaped_sweep_signal(FS, 50.0, 5000.0, 0.0)
    with pytest.raises(ValueError, match="amplitude"):
        shaped_sweep_signal(FS, 50.0, 5000.0, 1.0, amplitude=0.0)
    with pytest.raises(ValueError, match="fade"):
        shaped_sweep_signal(FS, 50.0, 5000.0, 1.0, fade=0.7)
    with pytest.raises(ValueError, match="start_delay"):
        shaped_sweep_signal(FS, 50.0, 5000.0, 1.0, start_delay=0.0)
    with pytest.raises(ValueError, match="unknown named target"):
        shaped_sweep_signal(FS, 50.0, 5000.0, 1.0, target="blue")
    mismatched = (np.array([100.0, 200.0]), np.array([0.0]))
    with pytest.raises(ValueError, match="equal-length"):
        shaped_sweep_signal(FS, 50.0, 5000.0, 1.0, target=mismatched)
    decreasing = (np.array([200.0, 100.0]), np.array([0.0, 0.0]))
    with pytest.raises(ValueError, match="increasing"):
        shaped_sweep_signal(FS, 50.0, 5000.0, 1.0, target=decreasing)
    non_finite = (np.array([100.0, 200.0]), np.array([0.0, np.inf]))
    with pytest.raises(ValueError, match="finite"):
        shaped_sweep_signal(FS, 50.0, 5000.0, 1.0, target=non_finite)


def test_result_behaves_like_array() -> None:
    res = shaped_sweep_signal(FS, 100.0, 5000.0, 0.5)
    assert len(res) == res.size == res.signal.size
    assert np.asarray(res).dtype == np.float64
    assert res[0] == res.signal[0]
