#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Tests for correlation estimates and time-delay estimation.

Every oracle is a closed form or a published number: the autocorrelation
of a sine (``(A²/2)·cos(2πf0τ)``) with the exact ``1-|τ|/T`` taper of the
biased estimator, the ``sin(2πBτ)/(2πBτ)`` autocorrelation of
bandwidth-limited white noise (B&P Eq. 8.120), exact integer and synthetic
fractional delays for the direct, GCC (all five Knapp & Carter Table I
weightings) and phase-slope estimators, the ``ε = [1+ρ⁻²]^½/√(2BT)``
random error (Eqs. 8.109/8.112) against seeded Monte Carlo and the
Example 8.5 numbers, the ideal-delta GCC-PHAT of a noiseless delay
(Knapp & Carter Eq. 23), and sub-sample IR delay/alignment against exact
frequency-domain fractional shifts, pinning the achievable accuracy.
"""

from __future__ import annotations

import dataclasses

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

import phonometry as ph

FS = 8192.0
N = 32768


def _white(seed: int, rms: float = 1.0, n: int = N) -> np.ndarray:
    return ph.signals.noise_signal(FS, n / FS, color="white", rms=rms, seed=seed)


def _bandlimited(seed: int, cutoff: float, n: int = N) -> np.ndarray:
    """White noise brick-wall low-passed to ``cutoff`` Hz."""
    x = _white(seed, n=n)
    spectrum = np.fft.rfft(x)
    spectrum[np.fft.rfftfreq(n, 1.0 / FS) > cutoff] = 0.0
    return np.asarray(np.fft.irfft(spectrum, n), dtype=np.float64)


def _fractional_delay(x: np.ndarray, shift: float) -> np.ndarray:
    """Exact circular fractional delay via a frequency-domain phase ramp."""
    ramp = np.exp(-2j * np.pi * np.fft.rfftfreq(x.size) * shift)
    return np.asarray(np.fft.irfft(np.fft.rfft(x) * ramp, x.size))


def _bandlimited_pulse(n: int, position: float, cutoff_norm: float) -> np.ndarray:
    """Band-limited impulse centred at a (fractional) sample position."""
    f = np.fft.rfftfreq(n)
    spectrum = np.exp(-((f / cutoff_norm) ** 2)) * np.exp(-2j * np.pi * f * position)
    return np.asarray(np.fft.irfft(spectrum, n), dtype=np.float64)


# ---------------------------------------------------------------------------
# Correlation estimates: closed forms
# ---------------------------------------------------------------------------


def test_sine_autocorrelation_closed_form_unbiased() -> None:
    """Unbiased R̂xx(τ) of A·sin recovers (A²/2)·cos(2πf0τ)."""
    amp, f0 = 2.0, 128.0
    t = np.arange(N) / FS
    x = amp * np.sin(2.0 * np.pi * f0 * t)
    res = ph.signals.correlation(x, fs=FS, normalization="unbiased", max_lag=0.05)
    expected = (amp**2 / 2.0) * np.cos(2.0 * np.pi * f0 * res.lags)
    np.testing.assert_allclose(res.values, expected, atol=5e-3 * amp**2)


def test_sine_autocorrelation_biased_tapers_by_one_minus_lag_over_t() -> None:
    """Biased estimate = unbiased × (1-|r|/N) exactly (Eq. 11.95)."""
    x = 3.0 * np.sin(2.0 * np.pi * 100.0 * np.arange(4096) / FS)
    biased = ph.signals.correlation(x, fs=FS, normalization="biased")
    unbiased = ph.signals.correlation(x, fs=FS, normalization="unbiased")
    taper = 1.0 - np.abs(biased.lags) * FS / x.size
    np.testing.assert_allclose(
        biased.values, unbiased.values * taper, rtol=1e-9, atol=1e-12
    )


def test_bandlimited_white_noise_autocorrelation_is_sinc() -> None:
    """ρxx(τ) of bandwidth-B white noise = sin(2πBτ)/(2πBτ) (Eq. 8.120)."""
    bandwidth = FS / 5.0
    x = _bandlimited(21, bandwidth)
    res = ph.signals.correlation(x, fs=FS, normalization="coefficient", max_lag=0.01)
    arg = 2.0 * np.pi * bandwidth * res.lags
    expected = np.ones_like(arg)
    nz = arg != 0.0
    expected[nz] = np.sin(arg[nz]) / arg[nz]
    # 2BT = 13107 -> per-lag random scatter ~1/sqrt(2BT) = 0.009.
    np.testing.assert_allclose(res.coefficient, expected, atol=0.03)


def test_coefficient_normalization_is_unity_at_zero_lag_and_bounded() -> None:
    x = _white(1)
    res = ph.signals.correlation(x, fs=FS, normalization="coefficient")
    zero = int(np.argmin(np.abs(res.lags)))
    assert res.values[zero] == pytest.approx(1.0, abs=1e-12)
    assert float(np.max(np.abs(res.values))) <= 1.0 + 1e-12
    assert res.kind == "autocorrelation"


def test_cross_correlation_peaks_at_the_imposed_delay() -> None:
    """B&P Eq. 5.21: R̂xy peaks at τ0 with ρ ≈ α·σx/σy (Eq. 5.22)."""
    delay = 40
    x = _white(2)
    noise = ph.signals.noise_signal(FS, N / FS, color="white", rms=0.75, seed=3)
    y = 0.5 * np.roll(x, delay) + noise
    res = ph.signals.correlation(x, y, FS, normalization="coefficient")
    peak = int(np.argmax(res.values))
    assert res.lags[peak] == pytest.approx(delay / FS)
    # α·σx/σy = 0.5/sqrt(0.25 + 0.5625)
    assert res.values[peak] == pytest.approx(0.5 / np.sqrt(0.25 + 0.5625), abs=0.01)
    assert res.kind == "cross-correlation"


def test_correlation_max_lag_restricts_the_axis() -> None:
    x = _white(4, n=4096)
    res = ph.signals.correlation(x, fs=FS, max_lag=0.02)
    m = round(0.02 * FS)
    assert res.lags.size == 2 * m + 1
    assert float(np.max(np.abs(res.lags))) == pytest.approx(m / FS)


# ---------------------------------------------------------------------------
# Correlation estimates: random error (Eqs. 8.109/8.111/8.112)
# ---------------------------------------------------------------------------


def test_autocorrelation_zero_lag_error_is_one_over_sqrt_bt() -> None:
    """ε[R̂xx(0)] = 1/√(BT) (Eq. 8.111) by seeded Monte Carlo."""
    values = []
    n = 8192
    for seed in range(200):
        # Raw Gaussian noise: noise_signal rescales to an exact RMS, which
        # would remove precisely the zero-lag fluctuation measured here.
        x = np.random.default_rng(100 + seed).standard_normal(n)
        res = ph.signals.correlation(x, fs=FS, normalization="biased", max_lag=0.002)
        zero = int(np.argmin(np.abs(res.lags)))
        values.append(res.values[zero])
    arr = np.asarray(values)
    empirical = float(np.std(arr) / np.mean(arr))
    # White noise: B = fs/2, so BT = n/2.
    assert empirical == pytest.approx(1.0 / np.sqrt(n / 2.0), rel=0.15)


def test_random_error_method_matches_the_closed_form() -> None:
    x = _bandlimited(5, 2000.0)
    res = ph.signals.correlation(x, fs=FS, normalization="coefficient", max_lag=0.01)
    eps = res.random_error(2000.0)
    expected = np.sqrt(1.0 + res.coefficient**-2.0) / np.sqrt(
        2.0 * 2000.0 * res.duration
    )
    ok = res.coefficient != 0.0
    np.testing.assert_allclose(eps[ok], expected[ok], rtol=1e-12)
    zero = int(np.argmin(np.abs(res.lags)))
    assert eps[zero] == pytest.approx(1.0 / np.sqrt(2000.0 * res.duration), rel=1e-6)


def test_correlation_random_error_reproduces_example_8_5() -> None:
    """B&P Example 8.5: B = 100 Hz, T = 5 s, M/S = N/S = 10 -> ε ≈ 0.35."""
    rho = 1.0 / np.sqrt(11.0 * 11.0)  # S/sqrt((S+M)(S+N)) with S=1, M=N=10
    eps = ph.signals.correlation_random_error(rho, 100.0, 5.0)
    assert eps == pytest.approx(0.35, abs=0.001)  # book rounds sqrt(122/1000)


def test_correlation_random_error_validation() -> None:
    assert ph.signals.correlation_random_error(0.0, 100.0, 5.0) == np.inf
    with pytest.raises(ValueError, match="coefficient"):
        ph.signals.correlation_random_error(1.5, 100.0, 5.0)
    with pytest.raises(ValueError, match="signal_bandwidth"):
        ph.signals.correlation_random_error(0.5, 0.0, 5.0)
    with pytest.raises(ValueError, match="duration"):
        ph.signals.correlation_random_error(0.5, 100.0, -1.0)


# ---------------------------------------------------------------------------
# Time-delay estimation: integer delays (all methods and weightings)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["direct", "gcc", "phase"])
def test_integer_delay_recovered_by_every_method(method: str) -> None:
    delay = 16
    x = _white(6)
    y = np.roll(x, delay)
    res = ph.signals.time_delay(x, y, FS, method=method, nperseg=2048)  # type: ignore[arg-type]
    assert res.delay_samples == pytest.approx(delay, abs=1e-3)
    assert res.delay == pytest.approx(delay / FS, rel=1e-4)
    assert res.peak_correlation == pytest.approx(1.0, abs=1e-3)


@pytest.mark.parametrize("weighting", ["none", "roth", "scot", "phat", "ml"])
def test_integer_delay_recovered_by_every_gcc_weighting(weighting: str) -> None:
    delay = 25
    x = _white(7)
    y = np.roll(x, delay)
    res = ph.signals.time_delay(
        x,
        y,
        FS,
        method="gcc",
        weighting=weighting,
        nperseg=2048,  # type: ignore[arg-type]
    )
    assert res.delay_samples == pytest.approx(delay, abs=1e-3)
    assert res.weighting == weighting
    assert res.method == "gcc"


def test_negative_delay_and_sign_convention() -> None:
    """y leading x (y(t) = x(t+|τ0|)) gives a negative delay."""
    x = _white(8)
    y = np.roll(x, -30)
    res = ph.signals.time_delay(x, y, FS, method="direct")
    assert res.delay_samples == pytest.approx(-30.0, abs=1e-3)


def test_gcc_phat_of_noiseless_delay_is_a_near_delta() -> None:
    """Knapp & Carter Eq. 23: ideal PHAT correlation is δ(τ-D)."""
    x = _white(9)
    y = np.roll(x, 12)
    res = ph.signals.time_delay(x, y, FS, method="gcc", weighting="phat", nperseg=2048)
    curve = np.abs(res.correlation)
    peak = int(np.argmax(curve))
    assert curve[peak] == pytest.approx(1.0)  # normalized to unit peak
    rest = np.delete(curve, peak)
    assert float(np.max(rest)) < 0.05


def test_gcc_phat_sharpens_the_peak_of_a_colored_signal() -> None:
    """The plain correlator smears the delta by the signal autocorrelation
    (Knapp & Carter Eq. 9); PHAT prewhitens and restores a sharp peak.
    The coloring keeps some power everywhere (a Butterworth roll-off, not
    a brick wall) - K&C's condition for the PHAT phase to stay defined.
    """
    from scipy import signal as sp_signal

    delay = 20
    b, a = sp_signal.butter(2, 800.0 / (FS / 2.0))
    x = np.asarray(sp_signal.lfilter(b, a, _white(10)), dtype=np.float64)
    y = np.roll(x, delay)

    def half_width(res: ph.signals.TimeDelayResult) -> int:
        curve = np.abs(res.correlation)
        return int(np.sum(curve > 0.5 * np.max(curve)))

    direct = ph.signals.time_delay(x, y, FS, method="direct")
    phat = ph.signals.time_delay(x, y, FS, method="gcc", weighting="phat", nperseg=2048)
    assert half_width(phat) < half_width(direct)
    assert phat.delay_samples == pytest.approx(delay, abs=1e-3)


def test_ml_weighting_suppresses_empty_bands_where_phat_jitters() -> None:
    """K&C: PHAT weights 1/|Gxy| regardless of coherence, so bands without
    signal contribute unit-magnitude random phase; the ML (HT) weighting
    scales them by γ²/(1-γ²) and stays accurate on a band-limited signal
    observed through noisy sensors.
    """
    s = _bandlimited(11, 0.4 * FS)
    noise_a = ph.signals.noise_signal(FS, N / FS, color="white", rms=0.1, seed=311)
    noise_b = ph.signals.noise_signal(FS, N / FS, color="white", rms=0.1, seed=312)
    x = s + noise_a
    y = _fractional_delay(s, 12.25) + noise_b
    ml = ph.signals.time_delay(
        x, y, FS, method="gcc", weighting="ml", nperseg=2048, upsample=16
    )
    phat = ph.signals.time_delay(
        x, y, FS, method="gcc", weighting="phat", nperseg=2048, upsample=16
    )
    assert ml.delay_samples == pytest.approx(12.25, abs=5e-3)
    assert abs(ml.delay_samples - 12.25) < abs(phat.delay_samples - 12.25)


# ---------------------------------------------------------------------------
# Time-delay estimation: sub-sample accuracy (documented tolerances)
# ---------------------------------------------------------------------------


def test_fractional_delay_direct_parabolic_accuracy() -> None:
    """Parabolic interpolation on a 0.4·fs band-limited pair: |err| < 0.1
    sample (the parabola only approximates the sinc-shaped peak).
    """
    x = _bandlimited(12, 0.4 * FS)
    y = _fractional_delay(x, 12.25)
    res = ph.signals.time_delay(x, y, FS, method="direct")
    assert res.delay_samples == pytest.approx(12.25, abs=0.1)


def test_fractional_delay_direct_upsampled_accuracy() -> None:
    """Band-limited local upsampling ×16 fixes the parabola bias:
    |err| < 2e-3 samples on the same pair.
    """
    x = _bandlimited(12, 0.4 * FS)
    y = _fractional_delay(x, 12.25)
    res = ph.signals.time_delay(x, y, FS, method="direct", upsample=16)
    assert res.delay_samples == pytest.approx(12.25, abs=2e-3)


def test_phase_slope_handles_polarity_inversion() -> None:
    """A polarity-inverted path (α < 0) has θ(0) = π; referencing the
    phase to DC keeps the Eq. 5.101b slope unbiased.
    """
    x = _white(22)
    y = -np.roll(x, 5)
    res = ph.signals.time_delay(x, y, FS, method="phase", nperseg=2048)
    assert res.delay_samples == pytest.approx(5.0, abs=1e-3)


def test_ml_weighting_requires_averaged_coherence() -> None:
    """K&C Eq. 45b exists only for |γ|² < 1: a single-segment coherence
    is identically one, so the estimator refuses to run.
    """
    x = _white(23, n=2048)
    y = np.roll(x, 5)
    with pytest.raises(ValueError, match="averaged coherence"):
        ph.signals.time_delay(x, y, FS, method="gcc", weighting="ml", nperseg=2048)


def test_time_delay_rejects_constant_records() -> None:
    flat = np.zeros(4096)
    x = _white(24, n=4096)
    for method in ("direct", "gcc", "phase"):
        with pytest.raises(ValueError, match="constant"):
            ph.signals.time_delay(flat, x, FS, method=method)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="constant"):
            ph.signals.time_delay(x, flat, FS, method=method)  # type: ignore[arg-type]


def test_fractional_delay_phase_slope_accuracy() -> None:
    """The Eq. 5.101b phase-slope estimate is exact to <1e-3 samples for a
    clean fractional delay (the phase of a pure delay is exactly linear).
    """
    x = _bandlimited(13, 0.4 * FS)
    y = _fractional_delay(x, 12.25)
    res = ph.signals.time_delay(x, y, FS, method="phase", nperseg=2048)
    assert res.delay_samples == pytest.approx(12.25, abs=1e-3)


def test_interpolation_none_stays_on_the_sample_grid() -> None:
    x = _bandlimited(14, 0.4 * FS)
    y = _fractional_delay(x, 12.25)
    res = ph.signals.time_delay(x, y, FS, method="direct", interpolation="none")
    assert res.delay_samples == pytest.approx(round(res.delay_samples))
    assert abs(res.delay_samples - 12.25) <= 0.5


# ---------------------------------------------------------------------------
# Time-delay estimation: peak-location uncertainty (Eqs. 8.129-8.130)
# ---------------------------------------------------------------------------


def _two_detector_pair(
    seed: int, delay: int, nsr: float, bandwidth: float, n: int = N
) -> tuple[np.ndarray, np.ndarray]:
    """x = s + m, y = s(t-τ0) + n with N/S = M/S = nsr (Section 8.4.2).

    The common signal is bandwidth-limited white noise of the given
    bandwidth, the model behind Eqs. 8.120-8.129.
    """
    s = _bandlimited(2000 + seed, bandwidth, n=n)
    rms = float(np.sqrt(nsr) * np.std(s))
    m = ph.signals.noise_signal(FS, n / FS, color="white", rms=rms, seed=4000 + seed)
    nn = ph.signals.noise_signal(FS, n / FS, color="white", rms=rms, seed=6000 + seed)
    return s + m, np.roll(s, delay) + nn


def test_delay_std_fields_follow_the_closed_form() -> None:
    bandwidth = 1000.0
    x, y = _two_detector_pair(1, 50, 3.0, bandwidth)
    res = ph.signals.time_delay(x, y, FS, method="direct", signal_bandwidth=bandwidth)
    assert res.delay_std is not None
    assert res.delay_interval is not None
    eps = ph.signals.correlation_random_error(
        abs(res.peak_correlation), bandwidth, N / FS
    )
    expected = (0.75**0.25) * np.sqrt(eps) / (np.pi * bandwidth)
    assert res.delay_std == pytest.approx(expected, rel=1e-9)
    assert res.delay_interval[0] == pytest.approx(res.delay - 2.0 * res.delay_std)
    assert res.delay_interval[1] == pytest.approx(res.delay + 2.0 * res.delay_std)
    # ρpeak -> S/sqrt((S+M)(S+N)) = 1/4 (Eq. 8.115).
    assert res.peak_correlation == pytest.approx(0.25, abs=0.02)


def test_delay_scatter_is_consistent_with_the_eq_8_129_prediction() -> None:
    """Monte Carlo scatter of τ̂0 against the B&P prediction. Eq. 8.129
    models the peak of the continuous correlation function, so it is an
    order-of-magnitude tool; the observed scatter sits below the
    prediction (the ±2σ interval of Eq. 8.130 is conservative) and within
    a factor of a few of it.
    """
    bandwidth = 1000.0
    delays = []
    predicted = 0.0
    for seed in range(60):
        x, y = _two_detector_pair(seed, 50, 10.0, bandwidth, n=8192)
        res = ph.signals.time_delay(
            x, y, FS, method="direct", signal_bandwidth=bandwidth
        )
        delays.append(res.delay)
        if seed == 0:
            assert res.delay_std is not None
            predicted = res.delay_std
    scatter = float(np.std(np.asarray(delays)))
    assert 0.15 * predicted < scatter < 1.2 * predicted


def test_no_bandwidth_means_no_error_fields() -> None:
    x = _white(15)
    res = ph.signals.time_delay(x, np.roll(x, 5), FS, method="direct")
    assert res.delay_std is None
    assert res.delay_interval is None
    assert res.signal_bandwidth is None


# ---------------------------------------------------------------------------
# Impulse-response delay and alignment (B5)
# ---------------------------------------------------------------------------


def test_single_ir_subsample_delay_accuracy() -> None:
    """Documented accuracy on a smooth band-limited pulse: a few
    hundredths of a sample with the parabola alone, ~1e-3 samples at the
    default upsample=8, below 1e-5 at x32.
    """
    ir = _bandlimited_pulse(4096, 700.35, 0.15)
    got = ph.signals.impulse_response_delay(ir, FS) * FS
    assert got == pytest.approx(700.35, abs=1e-3)
    coarse = ph.signals.impulse_response_delay(ir, FS, upsample=1) * FS
    assert coarse == pytest.approx(700.35, abs=5e-2)
    fine = ph.signals.impulse_response_delay(ir, FS, upsample=32) * FS
    assert fine == pytest.approx(700.35, abs=1e-5)


def test_single_ir_delay_handles_negative_polarity_peaks() -> None:
    ir = -_bandlimited_pulse(4096, 700.35, 0.15)
    got = ph.signals.impulse_response_delay(ir, FS) * FS
    assert got == pytest.approx(700.35, abs=1e-3)


def test_ir_pair_delay_and_alignment() -> None:
    reference = _bandlimited_pulse(4096, 600.0, 0.15)
    ir = _bandlimited_pulse(4096, 700.35, 0.15)
    delay = ph.signals.impulse_response_delay(ir, FS, reference=reference) * FS
    assert delay == pytest.approx(100.35, abs=1e-3)
    res = ph.signals.align_impulse_responses(ir, reference, FS)
    assert res.delay_samples == pytest.approx(100.35, abs=1e-3)
    assert res.delay == pytest.approx(res.delay_samples / FS)
    residual = float(np.max(np.abs(res.aligned - reference)))
    assert residual < 1e-4 * float(np.max(np.abs(reference)))


def test_single_ir_delay_near_the_record_start() -> None:
    """A peak close to the boundary keeps a full-size (clamped) upsampling
    window instead of a shrunken one.
    """
    ir = _bandlimited_pulse(4096, 5.25, 0.15)
    got = ph.signals.impulse_response_delay(ir, FS) * FS
    assert got == pytest.approx(5.25, abs=5e-3)


def test_peak_coefficient_guard_for_out_of_record_delays() -> None:
    """The phase-slope delay is not bounded by a search window; a delay
    beyond the record length must yield a zero coefficient, not a crash.
    """
    from phonometry.signals.correlation import _delay_error

    x = _white(25, n=4096)
    rho, std, interval = _delay_error(x, x, 5000.0, None, FS)
    assert rho == 0.0
    assert std is None
    assert interval is None


def test_alignment_of_identical_irs_is_a_no_op() -> None:
    reference = _bandlimited_pulse(4096, 600.0, 0.15)
    res = ph.signals.align_impulse_responses(reference, reference, FS)
    assert res.delay_samples == pytest.approx(0.0, abs=1e-9)
    np.testing.assert_allclose(res.aligned, reference, atol=1e-9)


# ---------------------------------------------------------------------------
# Validation and API surface
# ---------------------------------------------------------------------------


def test_correlation_rejects_invalid_inputs() -> None:
    x = _white(16, n=4096)
    with pytest.raises(ValueError, match="same length"):
        ph.signals.correlation(x, x[:-1], FS)
    with pytest.raises(ValueError, match="normalization"):
        ph.signals.correlation(x, fs=FS, normalization="raw")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="max_lag"):
        ph.signals.correlation(x, fs=FS, max_lag=10.0)
    with pytest.raises(ValueError, match="max_lag"):
        ph.signals.correlation(x, fs=FS, max_lag=-1.0)
    with pytest.raises(ValueError, match="'fs'"):
        ph.signals.correlation(x, fs=0.0)
    res = ph.signals.correlation(x, fs=FS)
    with pytest.raises(ValueError, match="signal_bandwidth"):
        res.random_error(0.0)


def test_correlation_rejects_a_coefficient_of_another_length() -> None:
    """``coefficient`` is the silent one of the three columns.

    ``random_error()`` reads it position by position and never looks at
    ``lags`` again, so a coefficient one bin short still returns a
    well-formed error curve, shifted by one lag: at the peak of this
    estimate it quotes 1.574, the uncertainty belonging to the next lag,
    where the true figure is 0.064. ``plot()`` draws its two lines as
    usual. A ``lags`` or ``values`` disagreement is loud instead, refused
    from inside matplotlib for want of a matching first dimension.
    """
    x = _white(26, n=4096)
    res = ph.signals.correlation(x, np.roll(x, 64), FS, max_lag=0.01)
    one_lag_short = res.coefficient[1:]
    with pytest.raises(ValueError, match="'coefficient'.*per lag"):
        dataclasses.replace(res, coefficient=one_lag_short)


def test_time_delay_rejects_invalid_inputs() -> None:
    x = _white(17, n=4096)
    y = np.roll(x, 5)
    with pytest.raises(ValueError, match="same length"):
        ph.signals.time_delay(x, y[:-1], FS)
    with pytest.raises(ValueError, match="method"):
        ph.signals.time_delay(x, y, FS, method="fft")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="weighting"):
        ph.signals.time_delay(x, y, FS, weighting="eckart")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="interpolation"):
        ph.signals.time_delay(x, y, FS, interpolation="cubic")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="upsample"):
        ph.signals.time_delay(x, y, FS, upsample=0)
    with pytest.raises(ValueError, match="max_delay"):
        ph.signals.time_delay(x, y, FS, method="gcc", nperseg=256, max_delay=1.0)
    with pytest.raises(ValueError, match="max_delay"):
        ph.signals.time_delay(x, y, FS, method="direct", max_delay=x.size / FS)
    with pytest.raises(ValueError, match="signal_bandwidth"):
        ph.signals.time_delay(x, y, FS, signal_bandwidth=-100.0)


def test_ir_utilities_reject_invalid_inputs() -> None:
    ir = _bandlimited_pulse(1024, 300.0, 0.2)
    with pytest.raises(ValueError, match="same length"):
        ph.signals.align_impulse_responses(ir, ir[:-1], FS)
    with pytest.raises(ValueError, match="upsample"):
        ph.signals.impulse_response_delay(ir, FS, upsample=-2)
    with pytest.raises(ValueError, match="'fs'"):
        ph.signals.impulse_response_delay(ir, 0.0)


def test_alignment_rejects_a_stacked_aligned_record() -> None:
    """A second axis on ``aligned`` is the silent way to break the pair.

    It passes every count, and ``plot()`` draws one trace per column over
    the single time axis built from the reference: three lines where there
    should be two, the aligned label and its one stored delay repeated in
    the legend for a column it was never measured on.
    """
    reference = _bandlimited_pulse(1024, 300.0, 0.2)
    ir = _bandlimited_pulse(1024, 310.0, 0.2)
    res = ph.signals.align_impulse_responses(ir, reference, FS)
    stacked = np.column_stack([res.aligned, res.reference])
    with pytest.raises(ValueError, match="'aligned' must have one axis"):
        dataclasses.replace(res, aligned=stacked)


def test_alignment_rejects_records_of_different_lengths() -> None:
    """The loud half of the same pairing, named here rather than left to
    matplotlib's 'x and y must have same first dimension' from inside the
    plotter: records that cannot be read sample against sample are refused
    at construction, where the field is still called by its name.
    """
    reference = _bandlimited_pulse(1024, 300.0, 0.2)
    ir = _bandlimited_pulse(1024, 310.0, 0.2)
    res = ph.signals.align_impulse_responses(ir, reference, FS)
    one_sample_short = res.aligned[:-1]
    with pytest.raises(ValueError, match="'aligned'.*per sample"):
        dataclasses.replace(res, aligned=one_sample_short)


def test_results_are_frozen() -> None:
    x = _white(18, n=4096)
    res = ph.signals.correlation(x, fs=FS)
    with pytest.raises(AttributeError):
        res.values = res.values * 2.0  # type: ignore[misc]
    tde = ph.signals.time_delay(x, np.roll(x, 5), FS)
    with pytest.raises(AttributeError):
        tde.delay = 0.0  # type: ignore[misc]


def test_max_delay_windows_the_search() -> None:
    delay = 10
    x = _white(19)
    y = np.roll(x, delay)
    res = ph.signals.time_delay(x, y, FS, method="direct", max_delay=0.01)
    m = round(0.01 * FS)  # the window rounds to whole samples
    assert float(np.max(np.abs(res.lags))) <= m / FS + 1e-12
    assert res.delay_samples == pytest.approx(delay, abs=1e-3)


# ---------------------------------------------------------------------------
# .plot() renderers
# ---------------------------------------------------------------------------


def test_correlation_plot_line_and_external_ax() -> None:
    x = _white(20, n=4096)
    res = ph.signals.correlation(x, fs=FS, max_lag=0.01)
    ax = res.plot()
    assert ax.lines
    _fig, ext = plt.subplots()
    assert res.plot(ax=ext) is ext
    plt.close("all")


def test_time_delay_plot_marks_the_delay() -> None:
    x = _white(21)
    y = np.roll(x, 12)
    res = ph.signals.time_delay(x, y, FS, nperseg=2048, signal_bandwidth=FS / 2.0)
    ax = res.plot()
    labels = [str(line.get_label()) for line in ax.lines]
    assert any("tau" in lab for lab in labels)
    _fig, ext = plt.subplots()
    assert res.plot(ax=ext) is ext
    plt.close("all")


def test_alignment_plot_two_lines() -> None:
    reference = _bandlimited_pulse(4096, 600.0, 0.15)
    ir = _bandlimited_pulse(4096, 700.35, 0.15)
    res = ph.signals.align_impulse_responses(ir, reference, FS)
    ax = res.plot()
    assert len(ax.lines) == 2
    _fig, ext = plt.subplots()
    assert res.plot(ax=ext) is ext
    plt.close("all")
