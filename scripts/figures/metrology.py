#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Figures for the metrology guides: the calibration, the record, the budget.

The checks made around a measurement rather than on its result: the
short-term stability of the calibration tone against its class limit, the
nonparametric trend, stationarity and runs tests that qualify a record before
it is analysed, the Rice level-crossing and peak laws its statistics are read
against, and the GUM budget that puts an uncertainty on the number finally
reported. Everything here is embedded by a page under ``signals/metrology/``.
"""

from collections.abc import Callable
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from phonometry._plot.common import format_frequency_axis, theme_fill

from .i18n import _fmt_minus
from .theme import (
    COLOR_FG,
    COLOR_GRID,
    COLOR_MUTED,
    COLOR_PRIMARY,
    COLOR_SECONDARY,
    COLOR_TERTIARY,
    save_figure,
)


def generate_calibration_stability(output_dir: str) -> None:
    """Stable vs unstable calibration tone against the IEC 60942 limit."""
    print("Generating calibration_stability.png...")
    from phonometry import time_weighting

    fs = 48000
    seconds = 6.0
    tt = np.arange(int(fs * seconds)) / fs
    stable = 0.5 * np.sin(2 * np.pi * 1000 * tt)
    # 3 % amplitude modulation at 2 Hz: ~0.14 dB deviation, clearly over
    unstable = 0.5 * (1 + 0.03 * np.sin(2 * np.pi * 2.0 * tt)) * np.sin(2 * np.pi * 1000 * tt)

    _, ax = plt.subplots(figsize=(10, 6))
    skip = fs  # discard the F-integrator attack (~8*tau = 1 s)
    for x, color, label in [
        (stable, COLOR_PRIMARY, "Stable tone (good coupling)"),
        (unstable, COLOR_SECONDARY, "3% AM tone (loose coupling)"),
    ]:
        env = time_weighting(x, fs, mode="fast")[skip:]
        level = 10 * np.log10(np.maximum(env, np.finfo(float).eps))
        rel = level - np.mean(level)
        ax.plot(tt[skip:], rel, color=color, linewidth=1.4, label=label)

    ax.axhline(0.07, color=COLOR_FG, linestyle="--", linewidth=1.2, alpha=0.7)
    ax.axhline(-0.07, color=COLOR_FG, linestyle="--", linewidth=1.2, alpha=0.7,
               label="IEC 60942:2017 class 1 limit (deviation from mean)")
    ax.fill_between([1, seconds], -0.07, 0.07, color=COLOR_PRIMARY, alpha=0.06)
    ax.set_title("Calibration Tone Stability Check (IEC 60942:2017, 5.3.3)",
                 pad=12)
    ax.set_xlim(1, seconds)
    ax.set_ylim(-0.2, 0.2)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("F-weighted level re mean [dB]")
    ax.legend(loc="upper right", fontsize=9)
    save_figure(output_dir, "calibration_stability.png")
    plt.close()


def generate_calibration_narrowband_bias(output_dir: str) -> None:
    """Sensitivity error against coupler SNR: broadband RMS vs Goertzel."""
    print("Generating calibration_narrowband_bias...")
    import warnings

    from phonometry import sensitivity

    fs = 48000
    tt = np.arange(int(fs * 6.0)) / fs
    # 1 Pa RMS tone landing on exactly 1.0 digital unit RMS, so the true
    # sensitivity is the 94 dB pressure itself and the error reads directly.
    tone = np.sqrt(2.0) * np.sin(2 * np.pi * 1000.0 * tt)
    s_true = 2e-5 * 10 ** (94.0 / 20.0)
    rng = np.random.default_rng(7)
    raw = rng.standard_normal(tone.size)
    raw /= np.sqrt(np.mean(raw ** 2))

    snr_db = np.arange(0.0, 40.5, 1.0)
    err_rms, err_nb = [], []
    for snr in snr_db:
        x = tone + raw * 10 ** (-snr / 20.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            err_rms.append(20 * np.log10(sensitivity(x, fs=fs) / s_true))
            err_nb.append(
                20 * np.log10(sensitivity(x, fs=fs, narrowband=True) / s_true)
            )
    closed = -10 * np.log10(1.0 + 10 ** (-snr_db / 10.0))
    # Where the broadband estimator alone eats the whole class 1 acceptance
    # limit for the generated level (IEC 60942:2017 Table 2, 160 Hz-1.25 kHz).
    limit = 0.25
    crossing = 10 * np.log10(1.0 / (10 ** (limit / 10.0) - 1.0))

    _fig, ax = plt.subplots(figsize=(10, 6))
    ax.axhspan(-limit, limit, color=theme_fill(COLOR_PRIMARY, ax), zorder=0,
               label="IEC 60942 Table 2 class 1 acceptance limit (±0.25 dB)")
    ax.plot(snr_db, err_rms, color=COLOR_SECONDARY, linewidth=2.0,
            label="narrowband=False (full-band RMS)")
    ax.plot(snr_db, closed, color=COLOR_FG, linestyle=":", linewidth=1.4,
            label=r"closed form $-10\log_{10}(1 + 1/\mathrm{SNR})$")
    ax.plot(snr_db, err_nb, color=COLOR_PRIMARY, linewidth=2.0,
            label="narrowband=True (Goertzel)")
    ax.axvline(crossing, color=COLOR_SECONDARY, linestyle="--", linewidth=1.2,
               alpha=0.8)
    ax.annotate(f"below {crossing:.1f} dB SNR the estimator\nalone spends the "
                "whole class 1 limit",
                xy=(crossing, -limit), xytext=(crossing + 1.5, -0.95),
                fontsize=9, color=COLOR_SECONDARY,
                arrowprops={"arrowstyle": "->", "color": COLOR_SECONDARY,
                            "lw": 1.1})
    ax.set_xlim(0, 40)
    ax.set_ylim(-2.0, 0.5)
    ax.set_xlabel("Coupler signal-to-noise ratio [dB]")
    ax.set_ylabel(r"Sensitivity error $20\log_{10}(S/S_\mathrm{true})$ [dB]")
    ax.set_title("Broadband noise in the calibrator take biases every later "
                 "level", pad=12)
    ax.grid(color=COLOR_GRID, linestyle="-", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    save_figure(output_dir, "calibration_narrowband_bias.png")
    plt.close()


def generate_dbfs_versus_spl(output_dir: str) -> None:
    """The two reference frames of a band spectrum, and the peak overshoot."""
    print("Generating dbfs_versus_spl...")
    import warnings

    from phonometry import LevelCalibration, octave_filter, sensitivity, tone_burst

    fs = 48000
    tt = np.arange(fs) / fs
    # A calibrator take sitting 12 dB below full scale, as a gain set for a
    # louder measurement would leave it.
    cal = 0.25 * np.sqrt(2.0) * np.sin(2 * np.pi * 1000.0 * tt)
    rng = np.random.default_rng(3)
    # A machine-like record: pink background, a fan hump around 250 Hz and a
    # roll-off above 4 kHz, so the two frames have a shape to superimpose.
    noise = rng.standard_normal(fs)
    freqs = np.fft.rfftfreq(fs, 1.0 / fs)
    shape = np.zeros_like(freqs)
    shape[1:] = ((freqs[1:] / 1000.0) ** -0.5
                 * (1.0 + 3.0 * np.exp(-((np.log2(freqs[1:] / 250.0)) ** 2) / 0.5))
                 / np.sqrt(1.0 + (freqs[1:] / 4000.0) ** 4))
    record = np.fft.irfft(np.fft.rfft(noise) * shape, n=fs)
    record *= 0.02 / np.sqrt(np.mean(record ** 2))

    limits = [22.0, 11300.0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        factor = sensitivity(cal, fs=fs)
        spl, bands = octave_filter(record, fs, fraction=3, limits=limits,
                                   calibration=LevelCalibration(factor=factor))
        dbfs, _ = octave_filter(record, fs, fraction=3, limits=limits,
                                calibration=LevelCalibration(dbfs=True))
    offset = 20 * np.log10(factor / 2e-5)

    _fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12.5, 5.4))

    # --- Left: one spectrum, two origins. ---
    ax_l.semilogx(bands, spl, color=COLOR_PRIMARY, marker="o", markersize=3.5,
                  linewidth=1.8, label="dB SPL (factor from the calibrator)")
    ax_l.semilogx(bands, dbfs + offset, color=COLOR_SECONDARY, linestyle="--",
                  linewidth=1.8, label=f"dBFS + {offset:.2f} dB")
    ax_l.set_xlabel("Frequency [Hz]")
    ax_l.set_ylabel("Third-octave band level [dB]")
    ax_l.set_title(f"Same shape, different origin: {offset:.2f} dB apart",
                   pad=10)
    ax_l.grid(which="both", color=COLOR_GRID, linestyle="-", alpha=0.5)
    ax_l.set_axisbelow(True)
    format_frequency_axis(ax_l, 22.0, 11300.0)
    # Lower left, not upper: the spectrum peaks near 250 Hz and an upper-left
    # box clips the peak, which is the one feature the panel exists to show.
    ax_l.legend(loc="lower left", fontsize=9)

    # --- Right: the band-filter onset overshoot the peak mode reads. ---
    drive = 0.5
    burst = tone_burst(fs, 1000.0, 40, amplitude=drive, post_silence=0.03)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _levels, _bands, filtered = octave_filter(burst.signal, fs, fraction=1,
                                                  sigbands=True)
    idx = int(np.argmin(np.abs(np.asarray(_bands, dtype=float) - 1000.0)))
    band = np.asarray(filtered[idx], dtype=float)
    overshoot = float(np.max(np.abs(band)))
    axis_ms = np.arange(band.size) / fs * 1000.0
    ax_r.plot(axis_ms, band, color=COLOR_MUTED, linewidth=0.8,
              label="1 kHz band signal")
    ax_r.axhline(drive, color=COLOR_FG, linestyle=":", linewidth=1.2,
                 label=f"drive amplitude {drive}")
    ax_r.axhline(overshoot, color=COLOR_SECONDARY, linewidth=1.4,
                 label=f"band peak, +{20 * np.log10(overshoot / drive):.2f} dB "
                       "over the steady tone")
    ax_r.set_xlim(0, 12.0)
    ax_r.set_xlabel("Time [ms] (onset of a 40-cycle burst)")
    ax_r.set_ylabel("Band signal [digital units]")
    ax_r.set_title("mode='peak' reads the filter's onset transient",
                   pad=10)
    ax_r.grid(color=COLOR_GRID, linestyle="-", alpha=0.5)
    ax_r.set_axisbelow(True)
    ax_r.legend(loc="lower right", fontsize=9)

    plt.tight_layout()
    save_figure(output_dir, "dbfs_versus_spl.png")
    plt.close()


def generate_trend_test(output_dir: str) -> None:
    """Reverse arrangement trend test: B&P Example 4.4 vs a rising drift."""
    print("Generating trend_test...")
    from phonometry import trend_test

    # B&P Example 4.4: twenty observations, A = 86, accepted (no trend).
    example = np.array([
        5.2, 6.2, 3.7, 6.4, 3.9, 4.0, 3.9, 5.3, 4.0, 4.6,
        5.9, 6.5, 4.3, 5.7, 3.1, 5.6, 5.2, 3.9, 6.2, 5.0,
    ])
    # The same fluctuations with a slow upward drift: fewer reverse
    # arrangements (A = 38, below the Table A.6 lower bound of 64), rejected.
    drifting = example + np.linspace(0.0, 4.0, example.size)
    res_flat = trend_test(example)
    res_drift = trend_test(drifting)

    _fig, ax = plt.subplots(figsize=(10, 6))
    index = np.arange(1, example.size + 1)
    ax.plot(index, res_flat.values, "o-", color=COLOR_PRIMARY,
            linewidth=1.2, markersize=5,
            label="B&P Example 4.4: $A$ = 86, accepted (no trend)")
    ax.plot(index, res_drift.values, "s-", color=COLOR_SECONDARY,
            linewidth=1.2, markersize=5,
            label="Added rising drift: $A$ = 38, rejected (trend)")
    ax.set_xticks(index[1::2])
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Sequence value")
    ax.set_title("Nonparametric Trend Test by Reverse Arrangements (B&P 4.5.2)",
                 pad=12)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=9)
    ax.text(0.02, 0.80,
            "20 observations; the count $A$ of pairs $i < j$ with "
            "$x_i > x_j$\n"
            "must fall in (64, 125] at the 5 % level (Table A.6). A rising\n"
            "trend depresses $A$ below the acceptance region",
            transform=ax.transAxes, va="top", ha="left", fontsize=8.5,
            color=COLOR_FG)
    plt.tight_layout()
    save_figure(output_dir, "trend_test.svg")
    plt.close()


def generate_stationarity_test(output_dir: str) -> None:
    """Reverse arrangement stationarity test: steady noise vs a gain ramp."""
    print("Generating stationarity_test...")
    from phonometry import stationarity_test

    fs = 8192.0
    n = 1 << 16
    steady = np.random.default_rng(42).standard_normal(n)
    # The B&P Example 10.3 scenario: the same noise with a slow +20 % gain
    # increase over the record.
    ramp = np.random.default_rng(42).standard_normal(n) * np.linspace(
        1.0, 1.2, n
    )
    res_steady = stationarity_test(steady, fs)
    res_ramp = stationarity_test(ramp, fs)

    _fig, ax = plt.subplots(figsize=(10, 6))
    index = np.arange(1, res_steady.n_segments + 1)
    ax.plot(index, res_steady.segment_values, "o-", color=COLOR_PRIMARY,
            linewidth=1.2, markersize=5,
            label="Steady noise: $A$ = 91, accepted (stationary)")
    ax.plot(index, res_ramp.segment_values, "s-", color=COLOR_SECONDARY,
            linewidth=1.2, markersize=5,
            label="+20 % gain ramp: $A$ = 7, rejected (nonstationary)")
    ax.set_xticks(index[1::2])
    ax.set_xlabel("Segment index")
    ax.set_ylabel("Segment mean square")
    ax.set_title("Stationarity Test by Reverse Arrangements (B&P 10.3.1.1)",
                 pad=12)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=9)
    ax.text(0.02, 0.80,
            "20 segment mean squares; the count $A$ of pairs $i < j$ with\n"
            "$x_i > x_j$ must fall in (64, 125] at the 5 % level (Table A.6)",
            transform=ax.transAxes, va="top", ha="left", fontsize=8.5,
            color=COLOR_FG)
    plt.tight_layout()
    save_figure(output_dir, "stationarity_test.svg")
    plt.close()


def _bandlimited_gaussian_figure_record(
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


def generate_rice_level_crossings(output_dir: str) -> None:
    """Measured level-crossing rates against the Rice curve."""
    print("Generating rice_level_crossings...")
    from phonometry import level_crossing_rate

    fs = 20480.0
    x = _bandlimited_gaussian_figure_record(0, fs, 1 << 19, 800.0, 1200.0)
    res = level_crossing_rate(x, fs, levels=np.linspace(-3.5, 3.5, 29))

    _fig, ax = plt.subplots(figsize=(10, 6))
    order = np.argsort(res.levels)
    ax.plot(res.levels[order], res.rice_rates[order], color=COLOR_SECONDARY,
            linewidth=1.6,
            label=r"Rice: $N_0\,\exp(-a^2/2\sigma_x^2)$ (Eq. 5.196)")
    ax.plot(res.levels, res.rates, "o", color=COLOR_PRIMARY, markersize=6,
            label="Measured crossing rate")
    ax.set_yscale("log")
    ax.set_xlabel("Level $a$ [signal units]")
    ax.set_ylabel("Crossings per second [1/s]")
    ax.set_title("Level-Crossing Rates of Bandlimited Gaussian Noise (Rice)",
                 pad=12)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="lower center", fontsize=9)
    ax.text(0.975, 0.965,
            "800-1200 Hz Gaussian band: 2014 zero crossings/s, an\n"
            r"apparent frequency $N_0/2 \approx$ 1007 Hz"
            " (B&P Example 5.13)",
            transform=ax.transAxes, va="top", ha="right", fontsize=8.5,
            color=COLOR_FG)
    plt.tight_layout()
    save_figure(output_dir, "rice_level_crossings.svg")
    plt.close()


def generate_rice_peak_distribution(output_dir: str) -> None:
    """Peak-height exceedance between the Gaussian and Rayleigh limits."""
    print("Generating rice_peak_distribution...")
    from phonometry import peak_statistics
    from phonometry.metrology.data_qualification import _rice_peak_exceedance

    fs = 20480.0
    x = _bandlimited_gaussian_figure_record(3, fs, 1 << 19, 0.0, 2000.0)
    res = peak_statistics(x, fs)

    _fig, ax = plt.subplots(figsize=(10, 6))
    peaks = res.peak_values
    exceedance = 1.0 - np.arange(1, peaks.size + 1) / peaks.size
    z = np.linspace(-2.5, 4.5, 400)
    ax.plot(z, _rice_peak_exceedance(z, 1.0), color=COLOR_FG, linewidth=1.0,
            linestyle="--", alpha=0.6,
            label="Rayleigh limit ($r = 1$, narrowband)")
    ax.plot(z, _rice_peak_exceedance(z, 0.0), color=COLOR_FG, linewidth=1.0,
            linestyle=":", alpha=0.6,
            label="Gaussian limit ($r = 0$, wideband)")
    ax.plot(z, res.peak_exceedance(z), color=COLOR_SECONDARY, linewidth=1.7,
            label="Rice mixture at $r = 0.746$ (Eq. 5.223)")
    ax.plot(peaks, exceedance, drawstyle="steps-post", color=COLOR_PRIMARY,
            linewidth=1.2, label="Empirical peak exceedance (0-2 kHz noise)")
    ax.set_yscale("log")
    ax.set_xlim(-2.5, 4.5)
    ax.set_ylim(1e-5, 1.5)
    ax.set_xlabel(r"Standardized peak height $z = a/\sigma_x$")
    ax.set_ylabel(r"$\mathrm{Prob}[\mathrm{peak} > z]$")
    ax.set_title("Peak-Height Distribution and the Irregularity Factor (Rice)",
                 pad=12)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", fontsize=9)
    ax.text(0.975, 0.965,
            r"low-pass noise: $r = N_0/2M = \sqrt{5}/3$;"
            " negative maxima exist,\nso the peak law sits between"
            " Gaussian and Rayleigh (B&P 5.5.4)",
            transform=ax.transAxes, va="top", ha="right", fontsize=8.5,
            color=COLOR_FG)
    plt.tight_layout()
    save_figure(output_dir, "rice_peak_distribution.svg")
    plt.close()


def generate_uncertainty(output_dir: str) -> None:
    """GUM uncertainty budget and Monte Carlo distribution (Guide 98-3 + S1)."""
    print("Generating uncertainty_budget.png...")
    import phonometry as ph

    # A-weighted level: reading plus calibration, instrument and positional
    # corrections (all zero-mean); the model is their sum.
    quantities = [
        ph.Quantity(74.0, 0.0, name="Reading"),
        ph.rectangular(0.0, 0.20, name="Calibration"),
        ph.rectangular(0.0, 0.30, name="Instrument"),
        ph.Quantity(0.0, 0.35, dof=9, name="Position (Type A)"),
    ]
    model = lambda a, b, c, d: a + b + c + d

    result = ph.combine_uncertainty(model, quantities)
    mc = ph.monte_carlo(model, quantities, trials=1_000_000, coverage=0.95, seed=1)
    k, big = result.expanded(0.95)

    _fig, (ax_b, ax_m) = plt.subplots(1, 2, figsize=(12.5, 5.4))

    # --- Left: uncertainty budget (contributions). ---
    contrib = result.contributions
    names = list(result.names)
    pos = np.arange(len(names))
    ax_b.barh(pos, contrib, color=COLOR_PRIMARY, zorder=2)
    ax_b.axvline(result.combined_uncertainty, color=COLOR_SECONDARY, ls="--",
                 label=f"$u_\\mathrm{{c}}$ = {result.combined_uncertainty:.3f} dB")
    ax_b.set_yticks(pos)
    ax_b.set_yticklabels(names)
    ax_b.invert_yaxis()
    ax_b.set_xlabel("Contribution to combined uncertainty [dB]")
    ax_b.set_title("GUM uncertainty budget", pad=10)
    ax_b.grid(which="major", axis="x", color=COLOR_GRID, linestyle="-", alpha=0.5)
    ax_b.set_axisbelow(True)
    ax_b.legend(loc="lower right")

    # --- Right: Monte Carlo output vs the GUM Gaussian. ---
    rng = np.random.default_rng(1)
    samples = (74.0 + rng.uniform(-0.20, 0.20, 200000)
               + rng.uniform(-0.30, 0.30, 200000)
               + rng.normal(0.0, 0.35, 200000))
    ax_m.hist(samples, bins=120, density=True, color=COLOR_PRIMARY, alpha=0.35,
              label="Monte Carlo (Suppl 1)")
    grid = np.linspace(samples.min(), samples.max(), 400)
    gauss = (np.exp(-0.5 * ((grid - result.value) / result.combined_uncertainty) ** 2)
             / (result.combined_uncertainty * np.sqrt(2 * np.pi)))
    ax_m.plot(grid, gauss, color=COLOR_SECONDARY, lw=2, label="GUM Gaussian")
    ax_m.axvspan(mc.interval[0], mc.interval[1], zorder=0,
                 color=theme_fill(COLOR_PRIMARY, ax_m),
                 label="95 % coverage interval")
    ax_m.set_xlabel("A-weighted level [dB]")
    ax_m.set_ylabel("Probability density")
    ax_m.set_title(f"$Y$ = {result.value:.2f} dB,  $U$ = {big:.2f} dB "
                   f"($k$ = {k:.2f})",
                   pad=10)
    ax_m.grid(color=COLOR_GRID, linestyle="-", alpha=0.4)
    ax_m.set_axisbelow(True)
    ax_m.legend(loc="upper right")

    plt.tight_layout()
    save_figure(output_dir, "uncertainty_budget.png")
    plt.close()


def generate_stationarity_glide_blind_spot(output_dir: str) -> None:
    """The nonstationary record the full-band mean square cannot see."""
    print("Generating stationarity_glide_blind_spot...")
    from scipy import signal as scipy_signal

    from phonometry import stationarity_test

    fs = 8192.0
    n = 1 << 16
    tt = np.arange(n) / fs
    # Constant-amplitude linear glide, 200 Hz to 2 kHz over the record.
    f0, f1 = 200.0, 2000.0
    glide = np.sin(2 * np.pi * (f0 * tt + (f1 - f0) / (2 * tt[-1]) * tt ** 2))
    lo, hi = 200.0, 400.0
    sos = scipy_signal.butter(6, [lo / (fs / 2), hi / (fs / 2)], btype="band",
                              output="sos")
    banded = scipy_signal.sosfiltfilt(sos, glide)

    full = stationarity_test(glide, fs)
    band = stationarity_test(banded, fs)

    _fig, axes = plt.subplots(3, 1, figsize=(10, 11))

    # (a) The instantaneous frequency of the glide, and the band the test
    # of panel (c) looks through. A line rather than a spectrogram: the
    # figures of this project stay vector.
    axes[0].plot(tt, f0 + (f1 - f0) * tt / tt[-1], color=COLOR_PRIMARY,
                 linewidth=2.0, label="instantaneous frequency")
    axes[0].axhspan(lo, hi, color=theme_fill(COLOR_SECONDARY, axes[0]),
                    zorder=0, label=f"the {lo:.0f}-{hi:.0f} Hz band of panel (c)")
    axes[0].set_xlim(0, tt[-1])
    axes[0].set_ylim(0, 2200)
    axes[0].set_xlabel("Time [s]")
    axes[0].set_ylabel("Frequency [Hz]")
    axes[0].grid(color=COLOR_GRID, linestyle="-", alpha=0.5)
    axes[0].set_axisbelow(True)
    axes[0].legend(loc="upper left", fontsize=9)
    axes[0].set_title(f"(a) The record: a {f0:.0f} Hz to {f1:.0f} Hz glide at "
                      "constant amplitude", pad=8)

    title_full = (f"(b) Full band: $A$ = {full.count}, inside "
                  f"{res_bounds(full)} — accepted, and blind")
    title_band = (f"(c) Band-limited {lo:.0f}-{hi:.0f} Hz: $A$ = {band.count}, "
                  f"outside {res_bounds(band)} — rejected")
    for ax, res, color, title in (
        (axes[1], full, COLOR_PRIMARY, title_full),
        (axes[2], band, COLOR_SECONDARY, title_band),
    ):
        idx = np.arange(1, res.n_segments + 1)
        ax.plot(idx, res.segment_values, "o-", color=color, linewidth=1.6,
                markersize=5)
        ax.set_xlabel("Segment index")
        ax.set_ylabel("Segment mean square")
        # Both panels share one scale, so the flat sequence reads as flat
        # rather than as the fifth-decimal noise a zoomed axis would show.
        ax.set_ylim(-0.03, 0.58)
        low, high = float(np.min(res.segment_values)), float(np.max(res.segment_values))
        ax.text(0.98, 0.08, f"segment values span {low:.4f} to {high:.4f}",
                transform=ax.transAxes, ha="right", fontsize=9,
                color=COLOR_MUTED)
        ax.set_title(title, pad=8)
        ax.grid(color=COLOR_GRID, linestyle="-", alpha=0.5)
        ax.set_axisbelow(True)

    plt.tight_layout()
    save_figure(output_dir, "stationarity_glide_blind_spot.png")
    plt.close()


def res_bounds(res: object) -> str:
    """The half-open acceptance region of a qualification result, as text."""
    lo, hi = res.bounds  # type: ignore[attr-defined]
    return f"({lo}, {hi}]"


def generate_rice_nongaussian_screen(output_dir: str) -> None:
    """Crossing rates off the Rice curve, and what an out-of-band floor does."""
    print("Generating rice_nongaussian_screen...")
    from scipy import signal as scipy_signal

    from phonometry import level_crossing_rate, peak_statistics

    fs = 20480.0
    n = 1 << 19
    gauss = _bandlimited_gaussian_figure_record(0, fs, n, 800.0, 1200.0)
    clipped = np.clip(gauss, -2.5, 2.5)
    rng = np.random.default_rng(11)
    impulsive = gauss.copy()
    impulsive[rng.choice(n, 400, replace=False)] += 6.0 * rng.choice(
        [-1.0, 1.0], 400)

    _fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12.5, 5.6))

    levels = np.linspace(-4.0, 4.0, 33)
    for record, color, marker, size, label in (
        (gauss, COLOR_PRIMARY, "o", 6, "Gaussian reference"),
        (clipped, COLOR_SECONDARY, "s", 4, r"hard-clipped at $2.5\,\sigma$"),
        (impulsive, COLOR_TERTIARY, "^", 4,
         r"Gaussian + sparse $6\,\sigma$ spikes"),
    ):
        res = level_crossing_rate(record, fs, levels=levels * np.std(record))
        ax_l.plot(res.levels / np.std(record), res.rice_rates, color=color,
                  linestyle="--", linewidth=1.2, alpha=0.8)
        ax_l.plot(res.levels / np.std(record), res.rates, marker, color=color,
                  markersize=size, linestyle="none", label=label)
    # The guides stop above the caption row: full height they ran straight
    # through both sentences, and the dots merged with the glyphs (they share
    # the red of the clipped record). Above the row they still cross every
    # decade the rates occupy, which is all they are there to mark.
    for edge in (-2.5, 2.5):
        ax_l.axvline(edge, ymin=0.18, color=COLOR_SECONDARY, linestyle=":",
                     linewidth=1.1, alpha=0.8)
    # Two rows, not one: side by side at the same height the Spanish pair
    # overlapped mid-panel (the committed asset shows them run together).
    ax_l.text(-3.6, 0.62, "spikes lift both tails above the curve", fontsize=9,
              color=COLOR_TERTIARY, ha="left")
    ax_l.text(4.0, 0.2, r"clipping: no crossings past $2.5\,\sigma$",
              fontsize=9, color=COLOR_SECONDARY, ha="right")
    ax_l.set_yscale("log")
    # A decade of headroom above the highest rate: the three-entry legend is
    # wider in Spanish than in English, and at the old top it sat on the Rice
    # curve and hid the marker groups underneath it.
    ax_l.set_ylim(1e-1, 1e5)
    ax_l.set_xlabel(r"Crossing level $a/\sigma$")
    ax_l.set_ylabel("Crossings per second [1/s]")
    ax_l.set_title("Measured rates against each record's own Rice curve",
                   pad=10)
    ax_l.grid(color=COLOR_GRID, linestyle="-", alpha=0.5)
    ax_l.set_axisbelow(True)
    # Opaque: the guide at $-2.5\,\sigma$ passes behind the legend on its way
    # to the top, and through a translucent frame it printed a dotted trace
    # across the entries.
    ax_l.legend(loc="upper left", fontsize=9, framealpha=1.0)

    # Right: what an out-of-band noise floor does to the irregularity factor.
    fs2 = 51200.0
    n2 = 1 << 20
    rng2 = np.random.default_rng(3)
    freqs = np.fft.rfftfreq(n2, 1.0 / fs2)
    spec = rng2.standard_normal(freqs.size) + 1j * rng2.standard_normal(freqs.size)
    floor = np.where(freqs <= 2000.0, 1.0, 10 ** (-50.0 / 20.0))
    wide = np.fft.irfft(spec * floor, n2)
    cutoffs = np.array([2500.0, 4000.0, 6000.0, 8000.0, 10000.0, 13000.0,
                        16000.0, 19000.0, 22000.0])
    factors = []
    for cut in cutoffs:
        sos = scipy_signal.butter(8, cut / (fs2 / 2), btype="low", output="sos")
        factors.append(peak_statistics(scipy_signal.sosfiltfilt(sos, wide),
                                       fs2).irregularity_factor)
    ideal = float(np.sqrt(5) / 3)
    ax_r.plot(cutoffs / 1000.0, factors, "o-", color=COLOR_PRIMARY, linewidth=2)
    ax_r.axhline(ideal, color=COLOR_FG, linestyle="--", linewidth=1.2,
                 label=f"ideal low-pass, $\\sqrt{{5}}/3$ = {ideal:.3f}")
    ax_r.axvline(2.0, color=COLOR_SECONDARY, linestyle=":", linewidth=1.4,
                 label="the physical band ends at 2 kHz")
    ax_r.set_xlabel("Analysis upper cut-off [kHz]")
    ax_r.set_ylabel("Irregularity factor $r$")
    ax_r.set_title("A floor 50 dB down, and $m_4$ weighting by $f^4$",
                   pad=10)
    ax_r.grid(color=COLOR_GRID, linestyle="-", alpha=0.5)
    ax_r.set_axisbelow(True)
    ax_r.legend(loc="lower left", fontsize=9)

    plt.tight_layout()
    save_figure(output_dir, "rice_nongaussian_screen.png")
    plt.close()


def generate_uncertainty_gum_vs_mc(output_dir: str) -> None:
    """The three regimes in which the GUM interval and Monte Carlo diverge."""
    print("Generating uncertainty_gum_vs_mc...")
    import phonometry as ph

    # The three panels differ in arity and in whether they cap the y axis,
    # so say so: inferring this from the first entry rejects the others.
    panels: list[tuple[str, Callable[..., Any], list[Any], str, float | None]] = []

    # A. One dominant rectangular input: the output is nearly rectangular and
    # the Gaussian interval overcovers it.
    q_a = [ph.rectangular(0.0, 1.0, name="Dominant rectangular"),
           ph.Quantity(0.0, 0.1, name="Small Gaussian")]
    panels.append(("A dominant rectangular input", lambda a, b: a + b, q_a,
                   "Correction [dB]", None))

    # B. Non-linear model: the energy sum of two levels with wide inputs.
    q_b = [ph.Quantity(70.0, 3.0, name="L1"), ph.Quantity(70.0, 3.0, name="L2")]
    panels.append(("A non-linear model (energy sum)",
                   lambda l1, l2: 10 * np.log10(10 ** (l1 / 10) + 10 ** (l2 / 10)),
                   q_b, "Combined level [dB]", None))

    # C. An output against a physical bound.
    q_c = [ph.Quantity(0.95, 0.06, name="alpha")]
    # The clamp piles a finite probability mass exactly on the bound, so the
    # histogram spike there is cut off to keep the rest of the shape readable.
    panels.append(("An output against a physical bound",
                   lambda a: np.minimum(a, 1.0), q_c, "Absorption coefficient",
                   9.0))

    _fig, axes = plt.subplots(3, 1, figsize=(10, 11))
    for ax, (title, model, quantities, xlabel, ymax) in zip(axes, panels):
        gum = ph.combine_uncertainty(model, quantities)
        mc = ph.monte_carlo(model, quantities, trials=400_000, coverage=0.95,
                            seed=1, keep_samples=True)
        _k, big = gum.expanded(0.95)
        # Asked for with `keep_samples=True` above. The type is optional
        # because callers that only want the interval do not pay for them.
        samples = np.asarray(mc.samples if mc.samples is not None else [])
        ax.hist(samples, bins=160, density=True, color=COLOR_PRIMARY,
                alpha=0.35, label="Monte Carlo (Supplement 1)")
        span = np.linspace(float(np.min(samples)), float(np.max(samples)), 500)
        u_c = gum.combined_uncertainty
        ax.plot(span, np.exp(-0.5 * ((span - gum.value) / u_c) ** 2)
                / (u_c * np.sqrt(2 * np.pi)), color=COLOR_SECONDARY, lw=2,
                label="GUM Gaussian")
        ax.axvspan(mc.interval[0], mc.interval[1], zorder=0,
                   color=theme_fill(COLOR_PRIMARY, ax),
                   label=f"MC 95 % interval [{_fmt_minus(mc.interval[0], '.2f')}, "
                         f"{_fmt_minus(mc.interval[1], '.2f')}]")
        for edge in (gum.value - big, gum.value + big):
            ax.axvline(edge, color=COLOR_SECONDARY, ls="--", lw=1.4)
        ax.axvline(gum.value - big, color=COLOR_SECONDARY, ls="--", lw=1.4,
                   label=rf"GUM $Y \pm U$  [{_fmt_minus(gum.value - big, '.2f')}, "
                         f"{_fmt_minus(gum.value + big, '.2f')}]")
        if ymax is not None:
            ax.set_ylim(0, ymax)
            ax.annotate("all the mass beyond the bound\npiles up on it",
                        xy=(1.0, ymax * 0.92), xytext=(0.87, ymax * 0.62),
                        fontsize=9, color=COLOR_FG,
                        arrowprops={"arrowstyle": "->", "color": COLOR_FG,
                                    "lw": 1.1})
        ax.set_title(title, pad=8)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Probability density")
        ax.grid(color=COLOR_GRID, linestyle="-", alpha=0.4)
        ax.set_axisbelow(True)
        ax.legend(loc="upper left", fontsize=8.5)

    plt.tight_layout()
    save_figure(output_dir, "uncertainty_gum_vs_mc.png")
    plt.close()


def generate_uncertainty_correlation(output_dir: str) -> None:
    """What ignoring correlation costs a two-term budget."""
    print("Generating uncertainty_correlation...")
    import phonometry as ph

    u_i = 0.3
    pair = [ph.Quantity(0.0, u_i, name="Term 1"),
            ph.Quantity(0.0, u_i, name="Term 2")]
    rho = np.linspace(-1.0, 1.0, 81)
    same, opposite = [], []
    for r in rho:
        corr = [[1.0, float(r)], [float(r), 1.0]]
        same.append(ph.combine_uncertainty(lambda a, b: a + b, pair,
                                           correlation=corr).combined_uncertainty)
        opposite.append(ph.combine_uncertainty(lambda a, b: a - b, pair,
                                               correlation=corr).combined_uncertainty)
    quad = float(np.hypot(u_i, u_i))

    _fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12.5, 5.4))
    ax_l.plot(rho, same, color=COLOR_PRIMARY, lw=2.0,
              label="sensitivities of the same sign")
    ax_l.plot(rho, opposite, color=COLOR_SECONDARY, lw=2.0, ls="--",
              label="sensitivities of opposite sign")
    ax_l.axhline(quad, color=COLOR_FG, ls=":", lw=1.3,
                 label=rf"quadrature value {quad:.3f} dB (assumes $\rho = 0$)")
    ax_l.set_xlabel(r"Correlation coefficient $\rho$ between the two terms")
    ax_l.set_ylabel("Combined standard uncertainty [dB]")
    ax_l.set_title("Two terms of 0.3 dB each", pad=10)
    ax_l.grid(color=COLOR_GRID, linestyle="-", alpha=0.5)
    ax_l.set_axisbelow(True)
    ax_l.legend(loc="upper left", fontsize=9)

    linear = 2 * u_i
    bars = ax_r.bar(["assumed\nindependent", "traceable to the\nsame calibrator"],
                    [quad, linear], color=[COLOR_PRIMARY, COLOR_SECONDARY],
                    width=0.55, zorder=2)
    for rect, val in zip(bars, (quad, linear)):
        ax_r.text(rect.get_x() + rect.get_width() / 2, val + 0.012,
                  f"{val:.3f} dB", ha="center", fontweight="bold")
    ax_r.annotate(f"+{100 * (linear / quad - 1):.0f} % understated",
                  xy=(1, linear), xytext=(0.28, linear * 0.72), fontsize=10,
                  color=COLOR_SECONDARY,
                  arrowprops={"arrowstyle": "->", "color": COLOR_SECONDARY,
                              "lw": 1.2})
    ax_r.set_ylim(0, linear * 1.22)
    ax_r.set_ylabel("Combined standard uncertainty [dB]")
    ax_r.set_title("The same budget, read two ways", pad=10)
    ax_r.grid(axis="y", color=COLOR_GRID, linestyle="-", alpha=0.5)
    ax_r.set_axisbelow(True)

    plt.tight_layout()
    save_figure(output_dir, "uncertainty_correlation.png")
    plt.close()


def generate_runs_test(output_dir: str) -> None:
    """Runs test about the median: accepted noise vs rejected alternation."""
    print("Generating runs_test...")
    from phonometry import trend_test

    rng = np.random.default_rng(3)
    sequences = [rng.standard_normal(40), np.tile([1.0, -1.0], 10)]

    _fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4))
    for ax, seq in zip(axes, sequences):
        res = trend_test(seq, method="runs")
        idx = np.arange(1, seq.size + 1)
        median = float(np.median(seq))
        above = seq > median
        ax.plot(idx, seq, color="#7f7f7f", linewidth=0.7, alpha=0.7,
                zorder=1)
        ax.scatter(idx[above], seq[above], color=COLOR_PRIMARY, s=28,
                   zorder=3, label="Above the median")
        ax.scatter(idx[~above], seq[~above], color=COLOR_SECONDARY, s=28,
                   zorder=3, label="Below the median")
        ax.axhline(median, color=COLOR_FG, linestyle="--", linewidth=1.1,
                   alpha=0.7, label="Sequence median")
        verdict = "trend-free" if res.trend_free else "rejected"
        ax.set_title(f"$r$ = {res.statistic} runs, accept "
                     f"({res.bounds[0]}, {res.bounds[1]}]: {verdict}",
                     pad=10)
        ax.set_xlabel("Sample index")
        ax.set_ylabel("Sequence value")
        ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5)
        ax.set_axisbelow(True)
        ax.legend(loc="lower left", fontsize=8.5)
    plt.suptitle("Runs Test About the Median (Wald & Wolfowitz)",
                 )
    plt.tight_layout()
    save_figure(output_dir, "runs_test.svg")
    plt.close()
