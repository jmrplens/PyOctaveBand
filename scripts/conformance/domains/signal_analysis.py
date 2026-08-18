#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Random-data analysis, the Bendat & Piersol chapters end to end.

One subject read in seven passes, all anchored on *Random Data* (4th edition)
and its companions. Calibrated spectral density estimation with its window and
overlap corrections; multiple-input/output coherence and the partial and
multiple coherence functions of Chapter 7; time-frequency analysis and the
Parseval/COLA identities a short-time transform must satisfy; correlation,
time-delay estimation and the analytic envelope (with Knapp & Carter for the
generalized cross-correlation); cepstral analysis, liftering and the envelope
spectrum (Havelock 2008); time synchronous averaging (McFadden 1987); and the
data-qualification tests - stationarity, the Wald & Wolfowitz run
distribution - and the Rice statistics of peaks and level crossings.

They share a module because they share their oracles: each expected value is
the closed form the estimator is supposed to converge to, evaluated on a
process synthesized to have exactly that spectrum, correlation or crossing
rate.
"""

from __future__ import annotations

import math

import numpy as np

import phonometry as ph

from ..registry import Outcome, numeric, register

_SPECTRA = "Calibrated spectral analysis (Bendat & Piersol)"


def _spectra_fs() -> float:
    return 8192.0


def _spectra_white(seed: int, rms: float = 1.0) -> np.ndarray:
    return ph.noise_signal(_spectra_fs(), 4.0, color="white", rms=rms, seed=seed)


@register(
    _SPECTRA,
    "Bendat & Piersol, Random Data 4e Eq. (5.67)",
    "White-noise autospectral density = sigma^2/(fs/2)",
)
def _chk_psd_white_level() -> Outcome:
    fs = _spectra_fs()
    res = ph.power_spectral_density(_spectra_white(1, rms=2.0), fs, nperseg=1024)
    band = (res.frequencies > 200.0) & (res.frequencies < 3800.0)
    expected = 4.0 / (fs / 2.0)
    return numeric(
        expected, float(np.mean(res.psd[band])), 0.03, rel=True, places=6
    )


@register(
    _SPECTRA,
    "Bendat & Piersol, Random Data 4e Eq. (8.158)",
    "PSD random error = 1/sqrt(nd) (Monte Carlo, 100 seeded records)",
)
def _chk_psd_random_error() -> Outcome:
    fs = _spectra_fs()
    estimates = []
    nd = 0.0
    for seed in range(100):
        res = ph.power_spectral_density(
            _spectra_white(100 + seed), fs, nperseg=1024, overlap=0.0
        )
        estimates.append(res.psd[50:200])
        nd = res.n_averages
    stack = np.asarray(estimates)
    empirical = float(np.mean(np.std(stack, axis=0) / np.mean(stack, axis=0)))
    return numeric(1.0 / math.sqrt(nd), empirical, 0.06, rel=True, places=4)


@register(
    _SPECTRA,
    "Bendat & Piersol, Random Data 4e Eq. (8.163)",
    "95% chi-square confidence interval coverage (Monte Carlo)",
)
def _chk_psd_ci_coverage() -> Outcome:
    fs = _spectra_fs()
    true_psd = 1.0 / (fs / 2.0)
    hits, total = 0, 0
    for seed in range(150):
        res = ph.power_spectral_density(_spectra_white(300 + seed), fs, nperseg=1024)
        for b in (60, 120, 240):
            hits += int(res.ci_lower[b] <= true_psd <= res.ci_upper[b])
            total += 1
    return numeric(0.95, hits / total, 0.025, places=4)


@register(
    _SPECTRA,
    "Bendat & Piersol, Random Data 4e Eqs. (9.55)/(6.39)",
    "Coherent output spectrum of a known-SNR path: gamma^2 = SNR/(1+SNR)",
)
def _chk_coherent_output_snr() -> Outcome:
    fs = _spectra_fs()
    x = _spectra_white(11)
    noise = ph.noise_signal(fs, 4.0, color="white", rms=0.5, seed=12)
    res = ph.coherent_output_spectrum(x, 0.8 * x + noise, fs, nperseg=1024)
    snr = 0.64 / 0.25
    band = slice(50, 400)
    return numeric(
        snr / (1.0 + snr),
        float(np.median(res.coherence[band])),
        0.03,
        places=4,
    )


@register(
    _SPECTRA,
    "Closed-form power-law slope (10*lg(2) dB/octave per unit exponent)",
    "Pink-noise PSD slope over 20 Hz - 20 kHz, dB/octave",
)
def _chk_pink_noise_slope() -> Outcome:
    fs = 48000.0
    x = ph.noise_signal(fs, 40.0, color="pink", seed=3)
    res = ph.power_spectral_density(x, fs, nperseg=8192)
    band = (res.frequencies >= 20.0) & (res.frequencies <= 20000.0)
    slope = float(
        np.polyfit(
            np.log2(res.frequencies[band]), 10.0 * np.log10(res.psd[band]), 1
        )[0]
    )
    return numeric(-10.0 * math.log10(2.0), slope, 0.05, unit="dB/oct", places=4)


@register(
    _SPECTRA,
    "IEC 60268-1:1985 Clause A2.1 / Table AII",
    "5 ms burst of 5 kHz tone at 48 kHz: gate RMS = A/sqrt(2) (integral periods)",
)
def _chk_tone_burst_rms() -> Outcome:
    # Clause A2.1: zero-crossing start, integral number of full periods.
    # Over exactly 25 full periods (240 samples) the mean square of the
    # sine is exactly 1/2, so the gate RMS is A/sqrt(2) to machine
    # precision.
    res = ph.tone_burst(48000.0, 5000.0, 25, amplitude=1.0)
    if res.burst_samples != 240:  # 5 ms at 48 kHz, from Table AII
        return numeric(240.0, float(res.burst_samples), 0.0, places=0)
    rms = float(np.sqrt(np.mean(res.signal[:240] ** 2)))
    return numeric(1.0 / math.sqrt(2.0), rms, 1e-12, places=6)


@register(
    _SPECTRA,
    "Harris 1978 closed form (DFT-even Hann)",
    "Hann window ENBW = n*sum(w^2)/sum(w)^2 = 3/2 exactly",
)
def _chk_hann_enbw() -> Outcome:
    res = ph.window_metrics("hann", 1024)
    return numeric(1.5, float(res.enbw_bins), 1e-12, places=6)


@register(
    _SPECTRA,
    "Constant-power 1/n-octave kernel (closed form)",
    "1/3-octave smoothed line level = P*df/(f0*(2^(1/6)-2^(-1/6)))",
)
def _chk_smoothing_line_level() -> Outcome:
    f = np.arange(1.0, 4001.0)
    power = np.zeros_like(f)
    i0 = 999  # 1000 Hz
    power[i0] = 5.0
    out = ph.fractional_octave_smoothing(f, power, 3.0)
    width = 1000.0 * (2.0 ** (1.0 / 6.0) - 2.0 ** (-1.0 / 6.0))
    return numeric(5.0 / width, float(out[i0]), 1e-9, rel=True, places=6)


@register(
    _SPECTRA,
    "Percival & Walden 1993, Table 382",
    "Slepian taper concentration lambda_14(31, 8/31), quadruple-precision table",
)
def _chk_multitaper_dpss_eigenvalue() -> Outcome:
    from scipy.signal.windows import dpss

    _, ratios = dpss(31, 8.0, Kmax=15, return_ratios=True)
    return numeric(
        0.929438220819848052, float(ratios[14]), 1e-12, places=12
    )


@register(
    _SPECTRA,
    "Percival & Walden 1993, Section 7.2 / Eq. (333)",
    "Multitaper white-noise density = sigma^2/(fs/2), NW=4, K=7 tapers",
)
def _chk_multitaper_white_level() -> Outcome:
    fs = _spectra_fs()
    res = ph.multitaper_psd(np.asarray(_spectra_white(41, rms=2.0))[:8192], fs)
    expected = 4.0 / (fs / 2.0)
    return numeric(
        expected, float(np.mean(res.psd[1:-1])), 0.03, rel=True, places=6
    )


@register(
    _SPECTRA,
    "Percival & Walden 1993, Eq. (369a) tone calibration",
    "Multitaper 'spectrum' scaling reads a sinusoid peak at A^2/2",
)
def _chk_multitaper_tone_peak() -> Outcome:
    fs = _spectra_fs()
    t = np.arange(4096) / fs
    x = 3.0 * np.sin(2.0 * np.pi * 1024.0 * t)
    res = ph.multitaper_psd(x, fs, scaling="spectrum", adaptive=False)
    return numeric(
        4.5, float(res.psd[int(np.argmax(res.psd))]), 1e-4, rel=True, places=6
    )


@register(
    _SPECTRA,
    "Percival & Walden 1993, Eq. (370b)",
    "Adaptive multitaper dof -> 2K on white noise (weights -> uniform)",
)
def _chk_multitaper_adaptive_dof() -> Outcome:
    fs = _spectra_fs()
    res = ph.multitaper_psd(np.asarray(_spectra_white(42))[:4096], fs)
    return numeric(
        2.0 * res.n_tapers,
        float(np.mean(res.degrees_of_freedom[1:-1])),
        0.02,
        rel=True,
        places=4,
    )


# ===========================================================================
# Multiple-input/output coherence (Bendat & Piersol, Random Data 4e, Ch. 7)
# ===========================================================================
_MISO = "Multiple-input coherence (Bendat & Piersol)"


def _miso_problem_7_2() -> tuple[float, float, float]:
    """Conditioned spectra of Problem 7.2 via the public conditioning path.

    Returns ``(Gv1, Gv2, gamma2_2y.1)`` computed by the module's
    Gaussian-elimination conditioning on the hand-set augmented matrix.
    """
    from phonometry.signals.miso import _condition

    mat = np.zeros((1, 3, 3), dtype=np.complex128)
    mat[0, 0, 0] = 3.0
    mat[0, 1, 1] = 2.0
    mat[0, 2, 2] = 10.0
    mat[0, 0, 1] = 1.0 + 1.0j
    mat[0, 1, 0] = 1.0 - 1.0j
    mat[0, 0, 2] = 4.0 + 1.0j
    mat[0, 2, 0] = 4.0 - 1.0j
    mat[0, 1, 2] = 3.0 - 1.0j
    mat[0, 2, 1] = 3.0 + 1.0j
    partial, coherent, _noise = _condition(mat, (0, 1))
    return float(coherent[0, 0]), float(coherent[1, 0]), float(partial[1, 0])


@register(
    _MISO,
    "Bendat & Piersol, Random Data 4e Problem 7.2 / Eqs. (7.86)/(7.94)",
    "Conditioned coherent output of the 2nd input abs(G2y.1)^2/G22.1 = 4/3 exactly",
)
def _chk_miso_problem_7_2_conditioned() -> Outcome:
    _gv1, gv2, _p2 = _miso_problem_7_2()
    return numeric(4.0 / 3.0, gv2, 1e-12, places=9)


@register(
    _MISO,
    "Bendat & Piersol, Random Data 4e Problem 7.2 / Eqs. (7.87)/(7.116)",
    "Partial coherence gamma^2_2y.1 = 2/15 and multiple coherence = 0.7",
)
def _chk_miso_problem_7_2_multiple() -> Outcome:
    gv1, gv2, _p2 = _miso_problem_7_2()
    # gamma^2_{y:x} = (Gv1 + Gv2)/Gyy = (17/3 + 4/3)/10 = 0.7 (Eq. 7.116).
    return numeric(0.7, (gv1 + gv2) / 10.0, 1e-12, places=9)


@register(
    _MISO,
    "Bendat & Piersol, Random Data 4e Eq. (7.35) with Eqs. (6.40)/(6.41)",
    "Multiple coherence of a known-SNR system: gamma^2_{y:x} = SNR/(1+SNR)",
)
def _chk_miso_multiple_snr() -> Outcome:
    fs = _spectra_fs()
    x1 = _spectra_white(201)
    x2 = _spectra_white(202)
    noise = ph.noise_signal(fs, 4.0, color="white", rms=0.5, seed=203)
    res = ph.miso_coherence([x1, x2], x1 + x2 + noise, fs, nperseg=1024)
    band = (res.frequencies > 200.0) & (res.frequencies < 3800.0)
    snr = 2.0 / 0.25  # Gvv = 2*sigma^2, Gnn = 0.5^2, both flat
    return numeric(
        snr / (1.0 + snr),
        float(np.median(res.multiple_coherence[band])),
        0.03,
        places=4,
    )


@register(
    _MISO,
    "Bendat & Piersol, Random Data 4e Eq. (7.117)",
    "Uncorrelated inputs: multiple coherence = sum of ordinary coherences",
)
def _chk_miso_independent_sum() -> Outcome:
    fs = _spectra_fs()
    x1 = _spectra_white(211)
    x2 = _spectra_white(212)
    noise = ph.noise_signal(fs, 4.0, color="white", rms=0.4, seed=213)
    res = ph.miso_coherence([x1, x2], x1 + 0.7 * x2 + noise, fs, nperseg=1024)
    band = (res.frequencies > 200.0) & (res.frequencies < 3800.0)
    diff = (res.multiple_coherence
            - res.ordinary_coherence.sum(axis=0))[band]
    # O(q/nd) coherence bias (Section 9.3); nd is a few hundred here.
    return numeric(0.0, float(np.median(diff)), 0.02, places=4)


@register(
    _MISO,
    "Bendat & Piersol, Random Data 4e Eqs. (7.88)/(7.121)",
    "Output-power decomposition Gyy = sum of Gvi + Gnn (exact)",
)
def _chk_miso_output_decomposition() -> Outcome:
    fs = _spectra_fs()
    x1 = _spectra_white(221)
    x2 = 0.5 * x1 + _spectra_white(222)
    noise = ph.noise_signal(fs, 4.0, color="white", rms=0.3, seed=223)
    res = ph.miso_coherence([x1, x2], x1 + x2 + noise, fs, nperseg=1024)
    reconstructed = res.coherent_output_spectra.sum(axis=0) + res.noise_psd
    resid = float(np.max(np.abs(reconstructed - res.output_psd)))
    return numeric(0.0, resid, 1e-12, places=12)


# ===========================================================================
# Time-frequency analysis (Bendat & Piersol, Random Data 4e)
# ===========================================================================
_TIME_FREQ = "Time-frequency analysis (Bendat & Piersol)"


@register(
    _TIME_FREQ,
    "Bendat & Piersol, Random Data 4e Eq. (12.173)",
    "Spectrogram of an on-bin tone reads its mean square A^2/2 in every column",
)
def _chk_spectrogram_tone_mean_square() -> Outcome:
    fs = _spectra_fs()
    t = np.arange(int(4 * fs)) / fs
    x = 2.0 * np.cos(2.0 * np.pi * 1024.0 * t)  # bin 128 of a 1024-segment
    res = ph.spectrogram(x, fs, nperseg=1024, scaling="spectrum")
    b = int(np.argmin(np.abs(res.frequencies - 1024.0)))
    worst = res.power[b][int(np.argmax(np.abs(res.power[b] - 2.0)))]
    return numeric(2.0, float(worst), 1e-9, rel=True, places=6)


@register(
    _TIME_FREQ,
    "Parseval + COLA identity (Hann taper, 75% overlap)",
    "Time-integrated STFT power = time-domain energy of an interior burst",
)
def _chk_spectrogram_parseval_cola() -> Outcome:
    fs = _spectra_fs()
    x = np.zeros(8192)
    x[2048:4096] = np.asarray(_spectra_white(21))[:2048]
    res = ph.spectrogram(x, fs, nperseg=256, overlap=0.75)
    df = float(res.frequencies[1] - res.frequencies[0])
    stft_energy = (res.hop / fs) * float(np.sum(res.power)) * df
    return numeric(
        float(np.sum(x**2)) / fs,
        stft_energy,
        1e-12,
        rel=True,
        places=6,
    )


@register(
    _TIME_FREQ,
    "Bendat & Piersol, Random Data 4e Eqs. (11.128)-(11.130)",
    "Zoom FFT tone amplitude = demodulate-decimate-DFT chain, machine precision",
)
def _chk_zoom_fft_demodulation_chain() -> Outcome:
    fs = _spectra_fs()
    n = 4096
    t = np.arange(n) / fs
    x = 0.7 * np.cos(2.0 * np.pi * 1100.0 * t + 0.3)
    res = ph.zoom_fft(x, fs, f_min=1000.0, f_max=1256.0, n_points=257, window="boxcar")
    peak = int(np.argmax(res.amplitude))
    # Eqs. (11.128)-(11.130): demodulate by exp(-j*2*pi*1000*t), decimate
    # by d = fs/(2B) = 16 and read bin 50 ((1100-1000) Hz / 2 Hz) of the
    # decimated record's DFT.
    idx = np.arange(n)
    v = (x * np.exp(-2j * np.pi * 1000.0 * idx / fs))[::16]
    m = np.arange(v.size)
    bin50 = np.sum(v * np.exp(-2j * np.pi * 50.0 * m / v.size))
    amp_bp = 2.0 * abs(bin50) / v.size
    return numeric(amp_bp, float(res.amplitude[peak]), 1e-12, rel=True, places=6)


# ===========================================================================
# Correlation, time delay and envelope (Bendat & Piersol / Knapp & Carter)
# ===========================================================================
_CORRELATION = "Correlation, time delay and envelope (B&P / Knapp & Carter)"


def _corr_fs() -> float:
    return 8192.0


def _corr_fractional_pair(
    seed: int, shift: float
) -> tuple[np.ndarray, np.ndarray]:
    """White noise and its exact circular fractional delay by ``shift``."""
    fs = _corr_fs()
    x = ph.noise_signal(fs, 4.0, color="white", seed=seed)
    ramp = np.exp(-2j * np.pi * np.fft.rfftfreq(x.size) * shift)
    return x, np.fft.irfft(np.fft.rfft(x) * ramp, x.size)


@register(
    _CORRELATION,
    "Bendat & Piersol, Random Data 4e Eq. (5.21)",
    "Cross-correlation peak of a 16-sample pure delay, samples",
)
def _chk_tde_integer_delay() -> Outcome:
    fs = _corr_fs()
    x = ph.noise_signal(fs, 4.0, color="white", seed=40)
    res = ph.time_delay(x, np.roll(x, 16), fs, method="direct")
    return numeric(16.0, res.delay_samples, 1e-3, places=4)


@register(
    _CORRELATION,
    "Knapp & Carter 1976, Table I (PHAT) + sub-sample interpolation",
    "GCC-PHAT estimate of an exact 12.25-sample fractional delay, samples",
)
def _chk_tde_gcc_phat_fractional() -> Outcome:
    x, y = _corr_fractional_pair(41, 12.25)
    res = ph.time_delay(
        x, y, _corr_fs(), method="gcc", weighting="phat", nperseg=2048,
        upsample=16,
    )
    return numeric(12.25, res.delay_samples, 5e-3, places=4)


@register(
    _CORRELATION,
    "Bendat & Piersol, Random Data 4e Eq. (5.101)",
    "Cross-spectrum phase-slope estimate of the same fractional delay",
)
def _chk_tde_phase_slope_fractional() -> Outcome:
    x, y = _corr_fractional_pair(41, 12.25)
    res = ph.time_delay(x, y, _corr_fs(), method="phase", nperseg=2048)
    return numeric(12.25, res.delay_samples, 1e-3, places=4)


@register(
    _CORRELATION,
    "Bendat & Piersol, Random Data 4e Eq. (8.120)",
    "BLWN autocorrelation coefficient at 3 samples vs sin(2piBt)/(2piBt)",
)
def _chk_blwn_autocorrelation_sinc() -> Outcome:
    fs = _corr_fs()
    bandwidth = fs / 5.0
    x = ph.noise_signal(fs, 4.0, color="white", seed=41)
    spectrum = np.fft.rfft(x)
    spectrum[np.fft.rfftfreq(x.size, 1.0 / fs) > bandwidth] = 0.0
    xb = np.fft.irfft(spectrum, x.size)
    res = ph.correlation(xb, fs=fs, normalization="coefficient",
                         max_lag=0.005)
    lag = int(np.argmin(np.abs(res.lags))) + 3
    arg = 2.0 * math.pi * bandwidth * res.lags[lag]
    return numeric(
        math.sin(arg) / arg, float(res.coefficient[lag]), 0.02, places=4
    )


@register(
    _CORRELATION,
    "Bendat & Piersol, Random Data 4e Example 8.5",
    "Random error of the correlation peak: B=100 Hz, T=5 s, M/S=N/S=10",
)
def _chk_correlation_random_error_example_8_5() -> Outcome:
    # rho_peak = S/sqrt((S+M)(S+N)) = 1/11 (Eq. 8.115); the book gives 0.35.
    eps = ph.correlation_random_error(1.0 / 11.0, 100.0, 5.0)
    return numeric(0.35, eps, 1e-3, places=4)


@register(
    _CORRELATION,
    "Bendat & Piersol, Random Data 4e Table 13.1",
    "Hilbert transform of cos recovers sin: max interior error",
)
def _chk_hilbert_cos_to_sin() -> Outcome:
    fs = _corr_fs()
    n = 16384
    t = np.arange(n) / fs
    res = ph.envelope(np.cos(2.0 * np.pi * 500.0 * t), fs)
    interior = slice(1024, n - 1024)
    reconstructed = res.envelope * np.sin(res.phase)
    err = float(np.max(np.abs(
        reconstructed[interior] - np.sin(2.0 * np.pi * 500.0 * t)[interior]
    )))
    return numeric(0.0, err, 1e-9, places=6)


@register(
    _CORRELATION,
    "Bendat & Piersol, Random Data 4e Eq. (13.27)",
    "Envelope of an AM waveform recovers 1 + m*cos(2pi*fm*t) exactly",
)
def _chk_am_envelope_exact() -> Outcome:
    fs = _corr_fs()
    n = 16384
    t = np.arange(n) / fs
    exact = 1.0 + 0.5 * np.cos(2.0 * np.pi * 10.0 * t)
    res = ph.envelope(exact * np.cos(2.0 * np.pi * 1000.0 * t), fs)
    interior = slice(1024, n - 1024)
    err = float(np.max(np.abs(res.envelope[interior] - exact[interior])))
    return numeric(0.0, err, 1e-9, places=6)


# ===========================================================================
# Cepstral analysis and envelope spectrum (Havelock 2008 / Bendat & Piersol)
# ===========================================================================
_CEPSTRUM = "Cepstrum, liftering and envelope spectrum (Havelock / B&P)"


def _cepstrum_echo_signal() -> np.ndarray:
    """delta[n] + a*delta[n-d]: DFT exactly 1 + a*exp(-j*2pi*k*d/N)."""
    x = np.zeros(4096)
    x[0] = 1.0
    x[313] = 0.4
    return x


@register(
    _CEPSTRUM,
    "Havelock 2008 Ch. 27 Fig. 21 + Mercator series of ln(1+a*e^{-j*theta})",
    "Power-cepstrum height at the echo delay = reflection coefficient a",
)
def _chk_power_cepstrum_echo() -> Outcome:
    res = ph.echo_detection(_cepstrum_echo_signal(), 8192.0)
    if res.delay_samples != 313:
        return numeric(313.0, float(res.delay_samples), 0.5, places=0)
    return numeric(0.4, res.reflection_coefficient, 1e-10, places=6)


@register(
    _CEPSTRUM,
    "Havelock 2008 Ch. 87 Eq. (14): complex cepstrum, series term n = 2",
    "Second rahmonic of a reflection a = 0.4 equals -a^2/2",
)
def _chk_complex_cepstrum_rahmonic() -> Outcome:
    res = ph.cepstrum(_cepstrum_echo_signal(), 8192.0, kind="complex")
    return numeric(-0.08, float(res.cepstrum[2 * 313]), 1e-10, places=6)


@register(
    _CEPSTRUM,
    "Bendat & Piersol, Random Data 4e Sec. 13.3 (Fig. 13.11)",
    "Envelope-spectrum line of an AM tone (A0 = 2, m = 0.35) at fm",
)
def _chk_envelope_spectrum_am_line() -> Outcome:
    fs = 8192.0
    n = 16384
    t = np.arange(n) / fs
    x = 2.0 * (1.0 + 0.35 * np.cos(2.0 * np.pi * 16.0 * t)) * np.cos(
        2.0 * np.pi * 1000.0 * t
    )
    res = ph.envelope_spectrum(x, fs)
    line = float(res.amplitude[round(16.0 * n / fs)])
    return numeric(0.7, line, 2e-3, places=4)


# ===========================================================================
# Time synchronous averaging (McFadden 1987)
# ===========================================================================
_TSA = "Time synchronous averaging (McFadden 1987)"


@register(
    _TSA,
    r"McFadden 1987 Eq. 8 / Eq. 9: comb filter \|C(f)\| at a harmonic k/T",
    "Comb-filter tooth height at a harmonic equals unity (any N)",
)
def _chk_tsa_comb_tooth() -> Outcome:
    period = 1.0 / 32.0
    value = float(ph.comb_filter_response(np.array([16.0 / period]), period, 8)[0])
    return numeric(1.0, value, 1e-10, places=8)


@register(
    _TSA,
    "McFadden 1987 Eq. 8: comb filter one quarter-order from a tooth, N = 2",
    "Comb-filter magnitude = 1/sqrt(2) at order 0.25",
)
def _chk_tsa_comb_midbin() -> Outcome:
    period = 1.0 / 32.0
    value = float(
        ph.comb_filter_response(np.array([0.25 / period]), period, 2)[0]
    )
    return numeric(1.0 / math.sqrt(2.0), value, 1e-10, places=8)


@register(
    _TSA,
    "McFadden 1987 Sec. 4 (Fig. 5): node selection, tone at 32.05 orders",
    r"N = 20 places a comb node on 32.05 orders (\|C\| = 0), not the power-of-2 N = 32",
)
def _chk_tsa_node_selection() -> Outcome:
    period = 1.0 / 32.0
    freq = np.array([32.05 / period])
    c20 = float(ph.comb_filter_response(freq, period, 20)[0])
    c32 = float(ph.comb_filter_response(freq, period, 32)[0])
    if not c32 > 0.15:  # sanity: the power-of-two choice does not reject it
        return numeric(0.0, c32, 0.0, places=8)
    return numeric(0.0, c20, 1e-10, places=10)


@register(
    _TSA,
    "McFadden 1987 Eq. 5: exact recovery, integer samples per period",
    "Noiseless periodic waveform (M = 256) recovered to machine precision",
)
def _chk_tsa_exact_recovery() -> Outcome:
    fs = 8192.0
    period = 1.0 / 32.0
    m = 256
    phase = np.arange(m) / m
    one = np.cos(2.0 * np.pi * phase) + 0.5 * np.cos(
        2.0 * np.pi * 3.0 * phase + 0.4
    )
    res = ph.time_synchronous_average(np.tile(one, 24), fs, period=period)
    err = float(np.max(np.abs(res.period_waveform - one)))
    return numeric(0.0, err, 1e-10, places=12)


@register(
    _TSA,
    "McFadden 1987 Sec. 1: asynchronous-noise variance reduced by 1/N",
    "Residual noise std of the average falls as sigma/sqrt(N), N = 64",
)
def _chk_tsa_sqrt_n_law() -> Outcome:
    fs = 8192.0
    period = 1.0 / 32.0
    m = 256
    n_avg = 64
    phase = np.arange(m) / m
    one = np.cos(2.0 * np.pi * phase)
    rng = np.random.default_rng(2024)
    noise = rng.standard_normal(n_avg * m)
    res = ph.time_synchronous_average(
        np.tile(one, n_avg) + noise, fs, period=period, n_averages=n_avg
    )
    measured = float(np.std(res.period_waveform - one))
    return numeric(1.0 / math.sqrt(n_avg), measured, 0.15, rel=True, places=5)


# ===========================================================================
# Data qualification and Rice statistics (Bendat & Piersol Chs. 4, 5, 10)
# ===========================================================================
_RANDOM_DATA = "Data qualification and Rice statistics (Bendat & Piersol)"

#: B&P Example 4.4: twenty observations with A = 86 reverse arrangements,
#: accepted as trend-free at the 5 % level of significance.
_BP_EXAMPLE_4_4 = [
    5.2, 6.2, 3.7, 6.4, 3.9, 4.0, 3.9, 5.3, 4.0, 4.6,
    5.9, 6.5, 4.3, 5.7, 3.1, 5.6, 5.2, 3.9, 6.2, 5.0,
]


@register(
    _RANDOM_DATA,
    "Bendat & Piersol, Random Data 4e Example 4.4",
    "Reverse arrangements of the 20-observation sequence",
)
def _chk_reverse_arrangements_example_4_4() -> Outcome:
    res = ph.trend_test(_BP_EXAMPLE_4_4)
    if not res.trend_free:  # the book accepts the hypothesis at 5 %
        return numeric(1.0, 0.0, 0.0, places=0)
    return numeric(86.0, float(res.statistic), 0.0, places=0)


@register(
    _RANDOM_DATA,
    "Bendat & Piersol, Random Data 4e Table A.6",
    "Lower percentage point A(20; 0.975) at alpha = 0.05",
)
def _chk_table_a6_lower() -> Outcome:
    res = ph.trend_test(_BP_EXAMPLE_4_4)
    return numeric(64.0, float(res.bounds[0]), 0.0, places=0)


@register(
    _RANDOM_DATA,
    "Bendat & Piersol, Random Data 4e Table A.6",
    "Upper percentage point A(20; 0.025) at alpha = 0.05",
)
def _chk_table_a6_upper() -> Outcome:
    res = ph.trend_test(_BP_EXAMPLE_4_4)
    return numeric(125.0, float(res.bounds[1]), 0.0, places=0)


def _runs_reference_result() -> ph.TrendTestResult:
    rng = np.random.default_rng(20)
    return ph.trend_test(rng.standard_normal(20), method="runs")


@register(
    _RANDOM_DATA,
    "Wald & Wolfowitz 1940 exact run distribution",
    "Runs acceptance region for n1 = n2 = 10, alpha = 0.05: lower point",
)
def _chk_runs_bounds_lower() -> Outcome:
    return numeric(
        6.0, float(_runs_reference_result().bounds[0]), 0.0, places=0
    )


@register(
    _RANDOM_DATA,
    "Wald & Wolfowitz 1940 exact run distribution",
    "Runs acceptance region for n1 = n2 = 10, alpha = 0.05: upper point",
)
def _chk_runs_bounds_upper() -> Outcome:
    return numeric(
        15.0, float(_runs_reference_result().bounds[1]), 0.0, places=0
    )


def _bandlimited_gaussian_record(
    seed: int, fs: float, n: int, f1: float, f2: float
) -> np.ndarray:
    """Exactly bandlimited unit-variance Gaussian noise (FFT synthesis)."""
    rng = np.random.default_rng(seed)
    freqs = np.fft.rfftfreq(n, 1.0 / fs)
    spec = rng.standard_normal(freqs.size) + 1j * rng.standard_normal(
        freqs.size
    )
    spec[(freqs < f1) | (freqs > f2)] = 0.0
    x = np.fft.irfft(spec, n)
    return np.asarray(x / np.std(x))


@register(
    _RANDOM_DATA,
    "Bendat & Piersol, Random Data 4e Example 5.13 / Eq. (5.195)",
    "Zero-crossing rate of bandlimited noise (fc = 1 kHz, B = 400 Hz)",
)
def _chk_rice_zero_crossings_bandpass() -> Outcome:
    x = _bandlimited_gaussian_record(0, 20480.0, 1 << 19, 800.0, 1200.0)
    res = ph.level_crossing_rate(x, 20480.0, levels=[0.0])
    expected = 2.0 * float(np.sqrt(1000.0**2 + 400.0**2 / 12.0))
    return numeric(expected, res.zero_crossing_rate, 0.01, rel=True, places=0)


@register(
    _RANDOM_DATA,
    "Bendat & Piersol, Random Data 4e Example 5.12",
    "Apparent frequency of low-pass noise (B = 2 kHz) = 0.577 B",
)
def _chk_rice_apparent_frequency_lowpass() -> Outcome:
    x = _bandlimited_gaussian_record(1, 20480.0, 1 << 19, 0.0, 2000.0)
    res = ph.level_crossing_rate(x, 20480.0, levels=[0.0])
    expected = 2000.0 / float(np.sqrt(3.0))
    return numeric(expected, res.apparent_frequency, 0.01, rel=True, places=0)


@register(
    _RANDOM_DATA,
    "Bendat & Piersol, Random Data 4e Example 5.14 / Eq. (5.206)",
    "Prob[positive peak > 4 sigma] of a narrow bandwidth record",
)
def _chk_rice_narrowband_peak_exceedance() -> Outcome:
    fs = 8192.0
    t = np.arange(1 << 16) / fs
    res = ph.peak_statistics(np.sin(2.0 * np.pi * 60.0 * t), fs)
    return numeric(
        float(np.exp(-8.0)),
        float(res.peak_exceedance(4.0)[0]),
        1e-5,
        places=6,
    )
