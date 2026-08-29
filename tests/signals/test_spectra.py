#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Tests for the calibrated spectral-density estimators (Bendat & Piersol).

Every oracle is a closed form from Bendat & Piersol, *Random Data* (4th ed.):
the 1/√nd random error of the autospectrum (Eq. 8.158) and its chi-square
confidence interval (Eq. 8.163) verified by seeded Monte Carlo, the
cross-spectrum magnitude and phase errors (Eqs. 9.33/9.52), the coherent
output spectrum of a known-SNR path where ``γ² = SNR/(1+SNR)`` and its
random error (Eq. 9.73), Parseval consistency of the calibrated scaling,
and the closed-form behaviour of the 1/n-octave smoother (flat invariance,
single-line width and level).

Every estimator here is a segment-averaged one. The Thomson multitaper
estimator answers the same question by tapering the whole record with K Slepian
sequences instead, so its oracles are Percival & Walden's rather than
Bendat & Piersol's; it lives in ``test_multitaper.py``.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

import phonometry as ph

if TYPE_CHECKING:
    from collections.abc import Callable

FS = 8192.0
N = 32768


def _white(seed: int, rms: float = 1.0, n: int = N) -> np.ndarray:
    return ph.signals.noise_signal(FS, n / FS, color="white", rms=rms, seed=seed)


def _known_snr_pair(seed: int, n: int = N) -> tuple[np.ndarray, np.ndarray, float]:
    """White noise through a gain of 0.8 plus white noise of RMS 0.5.

    Broadband SNR = 0.64/0.25 at every frequency, so the true coherence is
    the closed form ``γ² = SNR/(1+SNR)`` (module docstring of
    ``frequency_response``).
    """
    x = _white(1000 + seed, n=n)
    noise = ph.signals.noise_signal(
        FS, n / FS, color="white", rms=0.5, seed=5000 + seed
    )
    return x, 0.8 * x + noise, 0.64 / 0.25


# ---------------------------------------------------------------------------
# Autospectral density: calibration
# ---------------------------------------------------------------------------


def test_white_noise_density_level_is_flat_at_sigma2_over_bandwidth() -> None:
    x = _white(1, rms=2.0)
    res = ph.signals.power_spectral_density(x, FS, nperseg=1024)
    band = (res.frequencies > 200.0) & (res.frequencies < 3800.0)
    # Unit-variance-scaled white noise: Gxx = sigma^2 / (fs/2) everywhere.
    expected = 4.0 / (FS / 2.0)
    assert float(np.mean(res.psd[band])) == pytest.approx(expected, rel=0.03)


def test_boxcar_density_integrates_to_signal_power_parseval() -> None:
    x = _white(2)
    res = ph.signals.power_spectral_density(
        x, FS, nperseg=1024, overlap=0.0, window="boxcar"
    )
    df = float(res.frequencies[1] - res.frequencies[0])
    power = float(np.mean(x[: 1024 * (x.size // 1024)] ** 2))
    assert float(np.sum(res.psd) * df) == pytest.approx(power, rel=1e-10)


def test_spectrum_scaling_recovers_sine_power() -> None:
    t = np.arange(N) / FS
    amp = 3.0
    x = amp * np.sin(2.0 * np.pi * 400.0 * t)  # 400 Hz = bin 50 of 1024
    res = ph.signals.power_spectral_density(x, FS, nperseg=1024, scaling="spectrum")
    peak = int(np.argmax(res.psd))
    assert res.frequencies[peak] == pytest.approx(400.0)
    assert float(res.psd[peak]) == pytest.approx(amp**2 / 2.0, rel=1e-6)


def test_density_and_spectrum_scalings_differ_by_enbw() -> None:
    x = _white(3)
    dens = ph.signals.power_spectral_density(x, FS, nperseg=1024, scaling="density")
    spec = ph.signals.power_spectral_density(x, FS, nperseg=1024, scaling="spectrum")
    ratio = spec.psd[10:-10] / dens.psd[10:-10]
    assert np.allclose(ratio, dens.resolution_bandwidth, rtol=1e-9)


# ---------------------------------------------------------------------------
# Autospectral density: averaging statistics and random error (Eq. 8.158)
# ---------------------------------------------------------------------------


def test_no_overlap_averages_equal_segment_count() -> None:
    res = ph.signals.power_spectral_density(_white(4), FS, nperseg=1024, overlap=0.0)
    assert res.n_segments == N // 1024
    assert res.n_averages == pytest.approx(res.n_segments)
    assert res.degrees_of_freedom == pytest.approx(2.0 * res.n_segments)
    assert res.random_error == pytest.approx(1.0 / np.sqrt(res.n_segments))


def test_overlap_increases_segments_but_effective_averages_grow_less() -> None:
    plain = ph.signals.power_spectral_density(_white(5), FS, nperseg=1024, overlap=0.0)
    lapped = ph.signals.power_spectral_density(_white(5), FS, nperseg=1024, overlap=0.5)
    assert lapped.n_segments == 2 * plain.n_segments - 1
    # Correlated segments: nd grows, but stays below the raw count.
    assert plain.n_averages < lapped.n_averages < lapped.n_segments


def test_hann_resolution_bandwidth_is_1_5_bins() -> None:
    res = ph.signals.power_spectral_density(_white(6), FS, nperseg=1024)
    # ENBW of the periodic Hann taper = 1.5 * fs / nperseg.
    assert res.resolution_bandwidth == pytest.approx(1.5 * FS / 1024.0, rel=1e-6)


@pytest.mark.parametrize("overlap", [0.0, 0.5])
def test_random_error_matches_monte_carlo(overlap: float) -> None:
    """ε[Ĝxx] = 1/√nd (Eq. 8.158), with the Welch effective nd on overlap."""
    estimates = []
    nd = 0.0
    for seed in range(150):
        res = ph.signals.power_spectral_density(
            _white(100 + seed), FS, nperseg=1024, overlap=overlap
        )
        estimates.append(res.psd[50:200])
        nd = res.n_averages
    stack = np.asarray(estimates)
    empirical = float(np.mean(np.std(stack, axis=0) / np.mean(stack, axis=0)))
    assert empirical == pytest.approx(1.0 / np.sqrt(nd), rel=0.06)


def test_chi_square_interval_covers_true_psd_at_stated_confidence() -> None:
    """Monte Carlo coverage of the Eq. 8.163 interval over seeded records."""
    true_psd = 1.0 / (FS / 2.0)
    hits, total = 0, 0
    for seed in range(200):
        res = ph.signals.power_spectral_density(_white(300 + seed), FS, nperseg=1024)
        for b in (60, 120, 240):
            hits += int(res.ci_lower[b] <= true_psd <= res.ci_upper[b])
            total += 1
    assert hits / total == pytest.approx(0.95, abs=0.025)


def test_dc_and_nyquist_bins_get_wider_single_component_intervals() -> None:
    """DC/Nyquist carry one real Fourier component: n = nd, not 2·nd."""
    res = ph.signals.power_spectral_density(_white(8), FS, nperseg=1024)
    ratio = res.ci_upper / res.ci_lower
    interior = float(ratio[1])
    assert np.allclose(ratio[1:-1], interior, rtol=1e-9)
    assert float(ratio[0]) > 1.05 * interior
    assert float(ratio[-1]) > 1.05 * interior  # even nperseg -> Nyquist bin
    # Odd segment length: the last bin is not Nyquist and stays interior.
    odd = ph.signals.power_spectral_density(_white(8), FS, nperseg=1023)
    odd_ratio = odd.ci_upper / odd.ci_lower
    assert float(odd_ratio[-1]) == pytest.approx(float(odd_ratio[1]), rel=1e-9)
    assert float(odd_ratio[0]) > 1.05 * float(odd_ratio[1])


def test_confidence_interval_widens_at_higher_confidence() -> None:
    x = _white(7)
    r90 = ph.signals.power_spectral_density(x, FS, nperseg=1024, confidence=0.90)
    r99 = ph.signals.power_spectral_density(x, FS, nperseg=1024, confidence=0.99)
    assert np.all(r99.ci_lower <= r90.ci_lower)
    assert np.all(r90.ci_upper <= r99.ci_upper)
    assert np.all(r90.ci_lower < r90.psd)
    assert np.all(r90.psd < r90.ci_upper)


def test_resolution_bias_error_closed_form() -> None:
    # Eq. 8.141: εb = -(Be/Br)²/3; quarter-bandwidth analysis -> -1/48.
    assert ph.signals.resolution_bias_error(1.0, 4.0) == pytest.approx(-1.0 / 48.0)
    with pytest.raises(
        ValueError, match=r"'resolution_bandwidth' must be a positive, finite number"
    ):
        ph.signals.resolution_bias_error(0.0, 4.0)
    with pytest.raises(
        ValueError, match=r"'half_power_bandwidth' must be a positive, finite number"
    ):
        ph.signals.resolution_bias_error(1.0, -1.0)


# ---------------------------------------------------------------------------
# Cross-spectral density (Eqs. 9.33 / 9.52)
# ---------------------------------------------------------------------------


def test_cross_spectrum_of_known_gain_path() -> None:
    x, y, snr = _known_snr_pair(1)
    res = ph.signals.cross_spectral_density(x, y, FS, nperseg=1024)
    band = slice(50, 400)
    # Gxy = H·Gxx with H = 0.8 (uncorrelated noise drops out on average).
    expected = 0.8 / (FS / 2.0)
    assert float(np.mean(res.magnitude[band])) == pytest.approx(expected, rel=0.05)
    assert float(np.mean(np.abs(res.phase[band]))) < 0.05
    gamma2 = snr / (1.0 + snr)
    assert float(np.median(res.coherence[band])) == pytest.approx(gamma2, abs=0.03)


def test_cross_spectrum_error_formulas_match_monte_carlo() -> None:
    """Eqs. 9.33 and 9.52 against seeded Monte Carlo on a known-SNR path."""
    mags, phases = [], []
    nd = 0.0
    reported_mag_err = reported_phase_std = 0.0
    b = 100
    for seed in range(150):
        x, y, snr = _known_snr_pair(seed)
        res = ph.signals.cross_spectral_density(x, y, FS, nperseg=1024, overlap=0.0)
        mags.append(res.magnitude[b])
        phases.append(res.phase[b])
        nd = res.n_averages
        if seed == 0:
            reported_mag_err = float(np.median(res.magnitude_random_error))
            reported_phase_std = float(np.median(res.phase_std))
    gamma2 = snr / (1.0 + snr)
    gamma = np.sqrt(gamma2)
    expected_mag = 1.0 / (gamma * np.sqrt(nd))
    expected_phase = np.sqrt(1.0 - gamma2) / (gamma * np.sqrt(2.0 * nd))
    mags_arr = np.asarray(mags)
    assert float(np.std(mags_arr) / np.mean(mags_arr)) == pytest.approx(
        expected_mag, rel=0.15
    )
    assert float(np.std(np.asarray(phases))) == pytest.approx(expected_phase, rel=0.15)
    # The result reports the same formulas with the estimated coherence.
    assert reported_mag_err == pytest.approx(expected_mag, rel=0.10)
    assert reported_phase_std == pytest.approx(expected_phase, rel=0.10)


def test_cross_spectrum_phase_slope_measures_pure_delay() -> None:
    delay = 16
    x = _white(8)
    y = np.roll(x, delay)
    res = ph.signals.cross_spectral_density(x, y, FS, nperseg=2048)
    band = (res.frequencies > 100.0) & (res.frequencies < 2000.0)
    slope = np.polyfit(res.frequencies[band], res.phase[band], 1)[0]
    assert slope / (-2.0 * np.pi) * FS == pytest.approx(delay, rel=1e-3)


def test_cross_spectrum_matches_scipy_csd() -> None:
    x, y, _ = _known_snr_pair(2)
    from scipy import signal as sp_signal

    res = ph.signals.cross_spectral_density(x, y, FS, nperseg=1024)
    f, gxy = sp_signal.csd(
        x, y, fs=FS, window="hann", nperseg=1024, noverlap=512, detrend=False
    )
    np.testing.assert_array_equal(res.frequencies, f)
    np.testing.assert_array_equal(res.csd, gxy)


# ---------------------------------------------------------------------------
# Coherent output spectrum (Eqs. 9.55-9.56, 9.73)
# ---------------------------------------------------------------------------


def test_coherent_output_recovers_known_snr_path() -> None:
    x, y, snr = _known_snr_pair(3)
    res = ph.signals.coherent_output_spectrum(x, y, FS, nperseg=1024)
    band = slice(50, 400)
    gamma2 = snr / (1.0 + snr)
    assert float(np.median(res.coherence[band])) == pytest.approx(gamma2, abs=0.03)
    assert float(np.median(res.snr[band])) == pytest.approx(snr, rel=0.10)
    assert float(np.median(res.snr_db[band])) == pytest.approx(
        10.0 * np.log10(snr), abs=0.5
    )
    # Gvv recovers the linearly-explained output power 0.64/(fs/2).
    assert float(np.median(res.coherent_psd[band])) == pytest.approx(
        0.64 / (FS / 2.0), rel=0.10
    )
    # And Gnn the additive-noise power 0.25/(fs/2).
    assert float(np.median(res.noise_psd[band])) == pytest.approx(
        0.25 / (FS / 2.0), rel=0.10
    )


def test_coherent_plus_noise_equals_output_spectrum() -> None:
    x, y, _ = _known_snr_pair(4)
    res = ph.signals.coherent_output_spectrum(x, y, FS, nperseg=1024)
    np.testing.assert_allclose(
        res.coherent_psd + res.noise_psd, res.output_psd, rtol=1e-12
    )


def test_sine_in_noise_coherent_output_at_tone() -> None:
    """γ² at a tone bin follows SNR/(1+SNR) with the in-bin noise power."""
    t = np.arange(N) / FS
    x = np.sqrt(2.0) * np.sin(2.0 * np.pi * 400.0 * t)
    noise = ph.signals.noise_signal(FS, N / FS, color="white", rms=0.5, seed=9)
    y = x + noise
    res = ph.signals.coherent_output_spectrum(x, y, FS, nperseg=1024)
    i = int(np.argmin(np.abs(res.frequencies - 400.0)))
    # Tone power 1.0 against the noise power inside the analysis bandwidth.
    snr = 1.0 / (0.25 / (FS / 2.0) * res.resolution_bandwidth)
    assert float(res.coherence[i]) == pytest.approx(snr / (1.0 + snr), abs=2e-3)
    assert float(res.coherent_psd[i] / res.output_psd[i]) == pytest.approx(
        snr / (1.0 + snr), abs=2e-3
    )


def test_coherent_output_random_error_matches_monte_carlo() -> None:
    """ε[Ĝvv] = (2-γ²)^½/(|γ|·√nd) (Eq. 9.73) by seeded Monte Carlo."""
    values = []
    nd = 0.0
    reported = 0.0
    b = 100
    for seed in range(150):
        x, y, snr = _known_snr_pair(seed)
        res = ph.signals.coherent_output_spectrum(x, y, FS, nperseg=1024, overlap=0.0)
        values.append(res.coherent_psd[b])
        nd = res.n_averages
        if seed == 0:
            reported = float(np.median(res.random_error))
    gamma2 = snr / (1.0 + snr)
    expected = np.sqrt(2.0 - gamma2) / (np.sqrt(gamma2) * np.sqrt(nd))
    arr = np.asarray(values)
    assert float(np.std(arr) / np.mean(arr)) == pytest.approx(expected, rel=0.15)
    assert reported == pytest.approx(expected, rel=0.10)


def test_noiseless_path_has_unit_coherence_and_tiny_bias() -> None:
    x = _white(10)
    res = ph.signals.coherent_output_spectrum(x, 2.0 * x, FS, nperseg=1024)
    band = slice(50, 400)
    assert float(np.min(res.coherence[band])) > 0.999
    # Eq. 9.75: b[γ̂²] ≈ (1-γ²)²/nd -> essentially zero here.
    assert float(np.max(res.coherence_bias[band])) < 1e-4
    saturated = res.coherence == 1.0
    assert np.any(saturated), "expected bins with exactly unit coherence"
    assert np.all(np.isinf(res.snr[saturated]))


def test_snr_error_is_sqrt2_over_gamma_sqrt_nd() -> None:
    x, y, _ = _known_snr_pair(5)
    res = ph.signals.coherent_output_spectrum(x, y, FS, nperseg=1024)
    gamma = np.sqrt(res.coherence)
    ok = gamma > 0.0
    np.testing.assert_allclose(
        res.snr_random_error[ok],
        np.sqrt(2.0) / (gamma[ok] * np.sqrt(res.n_averages)),
        rtol=1e-12,
    )


# ---------------------------------------------------------------------------
# Fractional-octave smoothing
# ---------------------------------------------------------------------------


def test_smoothing_leaves_flat_spectrum_exactly_unchanged() -> None:
    f = np.linspace(0.0, 4000.0, 512)
    flat = np.full(512, 3.7)
    out = ph.signals.fractional_octave_smoothing(f, flat, 3.0)
    np.testing.assert_allclose(out, flat, rtol=0.0, atol=1e-12)


def test_smoothing_flat_invariance_in_db_and_amplitude_domains() -> None:
    f = np.linspace(1.0, 4000.0, 512)
    for domain, level in (("db", -12.0), ("amplitude", 0.5)):
        out = ph.signals.fractional_octave_smoothing(
            f,
            np.full(512, level),
            6.0,
            domain=domain,  # type: ignore[arg-type]
        )
        np.testing.assert_allclose(out, np.full(512, level), atol=1e-12)


def test_smoothed_line_has_third_octave_width_and_conserved_level() -> None:
    """A single spectral line spreads into the 1/n-octave kernel exactly."""
    df = 1.0
    f = np.arange(1.0, 4001.0, df)
    power = np.zeros_like(f)
    f0 = 1000.0
    i0 = int(np.argmin(np.abs(f - f0)))
    power[i0] = 5.0
    out = ph.signals.fractional_octave_smoothing(f, power, 3.0)
    width = f0 * (2.0 ** (1.0 / 6.0) - 2.0 ** (-1.0 / 6.0))
    # Level at the line: the bin's power spread over the kernel width.
    assert float(out[i0]) == pytest.approx(5.0 * df / width, rel=1e-9)
    # Support: the nonzero region spans one kernel width (± a couple bins).
    nonzero = f[out > 0.0]
    measured = float(nonzero[-1] - nonzero[0])
    assert measured == pytest.approx(width, abs=3.0 * df)


def test_smoothing_narrows_variance_of_noisy_spectrum() -> None:
    res = ph.signals.power_spectral_density(_white(11), FS, nperseg=4096)
    band = (res.frequencies > 200.0) & (res.frequencies < 3800.0)
    smooth = ph.signals.fractional_octave_smoothing(res.frequencies, res.psd, 3.0)
    assert float(np.std(smooth[band])) < 0.35 * float(np.std(res.psd[band]))
    # Same mean level: smoothing is power-preserving on average.
    assert float(np.mean(smooth[band])) == pytest.approx(
        float(np.mean(res.psd[band])), rel=0.02
    )


def test_smoothing_zero_frequency_point_is_copied() -> None:
    f = np.concatenate(([0.0], np.linspace(10.0, 1000.0, 100)))
    v = np.concatenate(([9.0], np.ones(100)))
    out = ph.signals.fractional_octave_smoothing(f, v, 3.0)
    assert out[0] == 9.0


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        (
            {"frequencies": [[1.0, 2.0]], "values": [1.0, 2.0]},
            r"'frequencies' and 'values' must be one-dimensional",
        ),
        (
            {"frequencies": [1.0, 2.0], "values": [1.0]},
            r"'frequencies'.*'values'.*one value per frequency",
        ),
        (
            {"frequencies": [1.0], "values": [1.0]},
            r"At least two frequency points are required",
        ),
        (
            {"frequencies": [2.0, 1.0], "values": [1.0, 1.0]},
            r"'frequencies' must be strictly increasing",
        ),
        (
            {"frequencies": [1.0, np.nan], "values": [1.0, 1.0]},
            r"'frequencies' and 'values' must be finite",
        ),
        (
            {"frequencies": [1.0, 2.0], "values": [1.0, -1.0]},
            r"'values' must be non-negative in the power domain",
        ),
        (
            {
                "frequencies": [1.0, 2.0],
                "values": [1.0, -1.0],
                "domain": "amplitude",
            },
            r"'values' must be non-negative in the amplitude domain",
        ),
        (
            {"frequencies": [1.0, 2.0], "values": [1.0, 1.0], "fraction": 0.0},
            r"'fraction' must be a positive, finite number",
        ),
        (
            {"frequencies": [1.0, 2.0], "values": [1.0, 1.0], "domain": "bad"},
            r"'domain' must be",
        ),
    ],
)
def test_smoothing_rejects_invalid_inputs(kwargs: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        ph.signals.fractional_octave_smoothing(**kwargs)


# ---------------------------------------------------------------------------
# Validation and API surface
# ---------------------------------------------------------------------------


def test_psd_rejects_invalid_inputs() -> None:
    x = _white(12)
    two_dimensional = np.stack([x, x])
    non_finite = np.full(4096, np.nan)
    with pytest.raises(ValueError, match=r"'x' must be one-dimensional"):
        ph.signals.power_spectral_density(two_dimensional, FS)
    with pytest.raises(ValueError, match=r"Signal too short for a spectral estimate"):
        ph.signals.power_spectral_density(x[:16], FS)
    with pytest.raises(ValueError, match=r"'x' must be finite"):
        ph.signals.power_spectral_density(non_finite, FS)
    with pytest.raises(ValueError, match=r"'fs' must be a positive, finite number"):
        ph.signals.power_spectral_density(x, 0.0)
    with pytest.raises(
        ValueError, match=r"'nperseg' must be between .* and the signal length"
    ):
        ph.signals.power_spectral_density(x, FS, nperseg=16)
    with pytest.raises(
        ValueError, match=r"'nperseg' must be between .* and the signal length"
    ):
        ph.signals.power_spectral_density(x, FS, nperseg=x.size + 1)
    with pytest.raises(ValueError, match=r"'overlap' must be in"):
        ph.signals.power_spectral_density(x, FS, overlap=1.0)
    with pytest.raises(ValueError, match=r"'confidence' must be in"):
        ph.signals.power_spectral_density(x, FS, confidence=1.0)
    with pytest.raises(ValueError, match=r"'scaling' must be"):
        ph.signals.power_spectral_density(x, FS, scaling="power")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "func",
    [ph.signals.cross_spectral_density, ph.signals.coherent_output_spectrum],
)
def test_two_channel_functions_reject_mismatched_lengths(
    func: Callable[[np.ndarray, np.ndarray, float], object],
) -> None:
    x = _white(13)
    with pytest.raises(
        ValueError, match=rf"{func.__name__}: 'x'.*'y'.*one value per sample"
    ):
        func(x, x[:-1], FS)


def test_density_band_must_sit_on_its_own_frequency_axis() -> None:
    """The chi-square band is refused when it does not match the density.

    The plot cuts the positive-frequency mask from ``frequencies`` and
    applies it to ``psd``, ``ci_lower`` and ``ci_upper``, so all four reach
    the reader only as numpy's complaint about a boolean index and two axis
    sizes, naming neither the field nor the result, and leaving an open
    figure behind. An extra axis passes every count, so the ranks are pinned
    too.
    """
    good = ph.signals.power_spectral_density(_white(40, n=4096), FS, nperseg=256)
    per_frequency = "one value per frequency"
    cases = (
        ("frequencies", good.frequencies[:-1], per_frequency),
        ("psd", good.psd[:-1], per_frequency),
        ("psd", np.append(good.psd, good.psd[-1]), per_frequency),
        ("ci_lower", good.ci_lower[:-1], per_frequency),
        ("ci_upper", np.append(good.ci_upper, good.ci_upper[-1]), per_frequency),
        ("psd", np.column_stack([good.psd] * 2), "must have one axis"),
    )
    for field, value, fragment in cases:
        with pytest.raises(ValueError, match=rf"'{field}'.*{fragment}"):
            dataclasses.replace(good, **{field: value})


def test_cross_spectrum_curves_must_share_one_frequency_axis() -> None:
    """A cross-spectrum whose seven curves disagree cannot be built.

    Five of them are drawn against the positive-frequency mask and surface
    as numpy's boolean-index complaint, which names two sizes and neither
    the field nor the result. The other two are silent: nothing plots
    ``csd`` or ``magnitude_random_error``, so the three-panel figure comes
    out complete and ordinary while the complex spectrum the caller reads no
    longer lines up with the magnitude and phase taken from it.
    """
    x, y, _ = _known_snr_pair(8, n=4096)
    good = ph.signals.cross_spectral_density(x, y, FS, nperseg=256)
    per_frequency = "one value per frequency"
    for field in (
        "frequencies",
        "csd",
        "magnitude",
        "phase",
        "coherence",
        "magnitude_random_error",
        "phase_std",
    ):
        curve = getattr(good, field)
        for value in (curve[:-1], np.append(curve, curve[-1])):
            with pytest.raises(ValueError, match=rf"'{field}'.*{per_frequency}"):
                dataclasses.replace(good, **{field: value})
    two_axes = np.column_stack([good.coherence] * 2)
    with pytest.raises(ValueError, match=r"'coherence' must have one axis"):
        dataclasses.replace(good, coherence=two_axes)


def test_coherent_output_split_must_share_one_frequency_axis() -> None:
    """The ten curves of the split are refused unless they agree.

    Only ``output_psd``, ``coherent_psd``, ``noise_psd`` and ``snr_db``
    reach the figure, so those raise from inside numpy's boolean index. The
    remaining five - the coherence and the Eq. 9.73 and 9.75 errors - are
    drawn by nothing at all, and a wrong length there returns the complete
    two-panel figure without a word while the quantities quoted beside it
    sit a bin off the axis they are quoted on.
    """
    x, y, _ = _known_snr_pair(9, n=4096)
    good = ph.signals.coherent_output_spectrum(x, y, FS, nperseg=256)
    per_frequency = "one value per frequency"
    for field in (
        "frequencies",
        "output_psd",
        "coherent_psd",
        "noise_psd",
        "coherence",
        "snr",
        "snr_db",
        "random_error",
        "snr_random_error",
        "coherence_bias",
    ):
        curve = getattr(good, field)
        for value in (curve[:-1], np.append(curve, curve[-1])):
            with pytest.raises(ValueError, match=rf"'{field}'.*{per_frequency}"):
                dataclasses.replace(good, **{field: value})
    two_axes = np.column_stack([good.snr_db] * 2)
    with pytest.raises(ValueError, match=r"'snr_db' must have one axis"):
        dataclasses.replace(good, snr_db=two_axes)


def test_default_nperseg_targets_4hz_resolution() -> None:
    res = ph.signals.power_spectral_density(_white(14), FS)
    df = float(res.frequencies[1] - res.frequencies[0])
    assert 2.0 <= df <= 8.0


def test_results_are_frozen() -> None:
    res = ph.signals.power_spectral_density(_white(15), FS, nperseg=1024)
    with pytest.raises(dataclasses.FrozenInstanceError):
        res.psd = res.psd * 2.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# .plot() renderers
# ---------------------------------------------------------------------------


def test_psd_plot_line_and_confidence_band() -> None:
    res = ph.signals.power_spectral_density(_white(16), FS, nperseg=1024)
    ax = res.plot()
    assert ax.lines, "expected the density line"
    labels = [str(c.get_label()) for c in ax.collections]
    assert any("confidence" in lab for lab in labels)
    _fig, ext = plt.subplots()
    assert res.plot(ax=ext) is ext
    plt.close("all")


def test_csd_plot_three_panels_and_external_ax() -> None:
    x, y, _ = _known_snr_pair(6)
    res = ph.signals.cross_spectral_density(x, y, FS, nperseg=1024)
    axes = res.plot()
    assert len(axes) == 3
    assert axes[2].get_ylim()[1] > 1.0  # coherence panel
    _fig, ext = plt.subplots()
    assert res.plot(ax=ext) is ext
    plt.close("all")


def test_coherent_output_plot_two_panels_and_external_ax() -> None:
    x, y, _ = _known_snr_pair(7)
    res = ph.signals.coherent_output_spectrum(x, y, FS, nperseg=1024)
    axes = res.plot()
    assert len(axes) == 2
    assert len(axes[0].lines) == 3  # Gyy, Gvv, Gnn
    _fig, ext = plt.subplots()
    assert res.plot(ax=ext) is ext
    plt.close("all")
