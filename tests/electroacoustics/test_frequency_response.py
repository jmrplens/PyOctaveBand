#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Tests for the frequency-response / coherence estimators (Bendat & Piersol).

The H1/H2 estimators recover a known LTI response, the coherence is unity for a
noiseless path and drops to ``SNR/(1+SNR)`` with additive output noise, and H2
is biased high (relative to H1) when the output is noisy.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pytest
import reference_data as ref
from scipy import signal as sp_signal

from phonometry import electroacoustics

FS = 48000
N = 400000


def _known_system() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """White-noise input through a first-order low-pass IIR; return x, y, b, a."""
    rng = np.random.default_rng(1)
    x = rng.standard_normal(N)
    b, a = sp_signal.butter(1, 2000.0 / (FS / 2.0), btype="low")
    y = sp_signal.lfilter(b, a, x)
    return x, y, b, a


def test_h1_recovers_known_magnitude_and_phase() -> None:
    x, y, b, a = _known_system()
    res = electroacoustics.transfer_function(x, y, FS, estimator="H1")
    _, h = sp_signal.freqz(b, a, worN=res.frequencies, fs=FS)
    idx = int(np.argmin(np.abs(res.frequencies - 1000.0)))
    assert abs(res.response[idx]) == pytest.approx(abs(h[idx]), rel=0.02)
    assert res.phase[idx] == pytest.approx(np.angle(h[idx]), abs=0.02)


def test_h2_recovers_known_magnitude() -> None:
    x, y, b, a = _known_system()
    res = electroacoustics.transfer_function(x, y, FS, estimator="H2")
    _, h = sp_signal.freqz(b, a, worN=res.frequencies, fs=FS)
    idx = int(np.argmin(np.abs(res.frequencies - 1000.0)))
    # Noiseless path: H1 and H2 agree with the true response.
    assert abs(res.response[idx]) == pytest.approx(abs(h[idx]), rel=0.02)


def test_coherence_unity_for_noiseless_path() -> None:
    x, y, _, _ = _known_system()
    f, g = electroacoustics.coherence(x, y, FS)
    band = (f > 100.0) & (f < 5000.0)
    assert np.mean(g[band]) == pytest.approx(1.0, abs=1e-3)
    assert np.all(g <= 1.0 + 1e-9)
    assert np.all(g >= 0.0)


def test_coherence_matches_snr_formula() -> None:
    # Identity system so the SNR is flat across frequency: y = x + noise with
    # x-power 1 and noise-power 1/SNR -> gamma^2 = SNR/(1+SNR).
    rng = np.random.default_rng(3)
    x = rng.standard_normal(N)
    noise = rng.standard_normal(N) * np.sqrt(1.0 / ref.COHERENCE_SNR)
    y = x + noise
    f, g = electroacoustics.coherence(x, y, FS)
    band = (f > 500.0) & (f < 20000.0)
    assert np.mean(g[band]) == pytest.approx(ref.COHERENCE_EXPECTED, abs=0.01)


def test_h2_biased_above_h1_with_output_noise() -> None:
    x, y, _, _ = _known_system()
    rng = np.random.default_rng(4)
    noise = rng.standard_normal(N) * np.sqrt(np.mean(y**2) / 5.0)
    yn = y + noise
    h1 = electroacoustics.transfer_function(x, yn, FS, estimator="H1")
    h2 = electroacoustics.transfer_function(x, yn, FS, estimator="H2")
    idx = int(np.argmin(np.abs(h1.frequencies - 1000.0)))
    # Output noise inflates Gyy, so |H2| > |H1| in the noisy band.
    assert abs(h2.response[idx]) > abs(h1.response[idx])


def test_result_fields_and_plot() -> None:
    x, y, _, _ = _known_system()
    res = electroacoustics.transfer_function(x, y, FS)
    assert isinstance(res, electroacoustics.FrequencyResponseResult)
    assert res.estimator == "H1"
    assert res.magnitude_db.shape == res.frequencies.shape
    axes = res.plot()
    assert len(axes) == 3


def test_rejects_mismatched_lengths() -> None:
    x = np.zeros(1000)
    shorter_y = np.zeros(500)
    with pytest.raises(ValueError):
        electroacoustics.transfer_function(x, shorter_y, FS)


def test_rejects_bad_estimator_and_overlap() -> None:
    x = np.zeros(1000)
    with pytest.raises(ValueError):
        electroacoustics.transfer_function(x, x, FS, estimator="H3")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        electroacoustics.transfer_function(x, x, FS, overlap=1.0)


def test_high_overlap_does_not_crash() -> None:
    # overlap close to 1.0 must clamp noverlap to nperseg - 1, not raise.
    x, y, _, _ = _known_system()
    res = electroacoustics.transfer_function(x, y, FS, overlap=0.99)
    assert np.all(np.isfinite(res.coherence))


def test_rejects_bad_nperseg() -> None:
    x = np.zeros(1000)
    with pytest.raises(ValueError):
        electroacoustics.transfer_function(x, x, FS, nperseg=5000)


def test_rejects_too_short_signal() -> None:
    too_short = np.zeros(10)
    with pytest.raises(ValueError):
        electroacoustics.coherence(too_short, too_short, FS)


def test_h1_input_noise_bias_matches_theory() -> None:
    # Bendat & Piersol: noise on the MEASURED INPUT biases H1 low by
    # SNR/(1+SNR). With input-noise power 0.25 on a unit-variance input
    # (SNR = 4) through an identity path, |H1| -> 4/5 = 0.800; the Welch
    # estimate at nperseg=4096 sits within 0.03 of it. Complements the
    # output-noise coherence check above.
    rng = np.random.default_rng(3)
    n = 200000
    x = rng.standard_normal(n)
    x_measured = x + rng.standard_normal(n) * 0.5  # SNR = 1 / 0.25 = 4
    res = electroacoustics.transfer_function(
        x_measured, x, FS, estimator="H1", nperseg=4096
    )
    mean_mag = float(np.mean(np.abs(res.response)))
    assert mean_mag == pytest.approx(0.800, abs=0.03)
