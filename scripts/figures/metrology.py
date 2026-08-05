#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Figures for the metrology guides: the calibration, the record, the budget.

The checks made around a measurement rather than on its result: the
short-term stability of the calibration tone against its class limit, the
nonparametric trend, stationarity and runs tests that qualify a record before
it is analysed, the Rice level-crossing and peak laws its statistics are read
against, and the GUM budget that puts an uncertainty on the number finally
reported. Everything here is embedded by a page under ``signals/metrology/``.
"""

import matplotlib.pyplot as plt
import numpy as np

from phonometry._plot.common import theme_fill

from .theme import (
    COLOR_FG,
    COLOR_GRID,
    COLOR_PRIMARY,
    COLOR_SECONDARY,
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
                 fontweight="bold", pad=12)
    ax.set_xlim(1, seconds)
    ax.set_ylim(-0.2, 0.2)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("F-weighted level re mean [dB]")
    ax.legend(loc="upper right", fontsize=9)
    save_figure(output_dir, "calibration_stability.png")
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
            label="B&P Example 4.4: A = 86, accepted (no trend)")
    ax.plot(index, res_drift.values, "s-", color=COLOR_SECONDARY,
            linewidth=1.2, markersize=5,
            label="Added rising drift: A = 38, rejected (trend)")
    ax.set_xticks(index[1::2])
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Sequence value")
    ax.set_title("Nonparametric Trend Test by Reverse Arrangements (B&P 4.5.2)",
                 fontweight="bold", pad=12)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=9)
    ax.text(0.02, 0.80,
            "20 observations; the count A of pairs i < j with x[i] > x[j]\n"
            "must fall in (64, 125] at the 5 % level (Table A.6). A rising\n"
            "trend depresses A below the acceptance region",
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
            label="Steady noise: A = 91, accepted (stationary)")
    ax.plot(index, res_ramp.segment_values, "s-", color=COLOR_SECONDARY,
            linewidth=1.2, markersize=5,
            label="+20 % gain ramp: A = 7, rejected (nonstationary)")
    ax.set_xticks(index[1::2])
    ax.set_xlabel("Segment index")
    ax.set_ylabel("Segment mean square")
    ax.set_title("Stationarity Test by Reverse Arrangements (B&P 10.3.1.1)",
                 fontweight="bold", pad=12)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=9)
    ax.text(0.02, 0.80,
            "20 segment mean squares; the count A of pairs i < j with\n"
            "x[i] > x[j] must fall in (64, 125] at the 5 % level (Table A.6)",
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
    ax.set_xlabel("Level a [signal units]")
    ax.set_ylabel("Crossings per second [1/s]")
    ax.set_title("Level-Crossing Rates of Bandlimited Gaussian Noise (Rice)",
                 fontweight="bold", pad=12)
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
            linestyle="--", alpha=0.6, label="Rayleigh limit (r = 1, narrowband)")
    ax.plot(z, _rice_peak_exceedance(z, 0.0), color=COLOR_FG, linewidth=1.0,
            linestyle=":", alpha=0.6, label="Gaussian limit (r = 0, wideband)")
    ax.plot(z, res.peak_exceedance(z), color=COLOR_SECONDARY, linewidth=1.7,
            label="Rice mixture at r = 0.746 (Eq. 5.223)")
    ax.plot(peaks, exceedance, drawstyle="steps-post", color=COLOR_PRIMARY,
            linewidth=1.2, label="Empirical peak exceedance (0-2 kHz noise)")
    ax.set_yscale("log")
    ax.set_xlim(-2.5, 4.5)
    ax.set_ylim(1e-5, 1.5)
    ax.set_xlabel(r"Standardized peak height $z = a/\sigma_x$")
    ax.set_ylabel("Prob[peak > z]")
    ax.set_title("Peak-Height Distribution and the Irregularity Factor (Rice)",
                 fontweight="bold", pad=12)
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
                 label=f"$u_c$ = {result.combined_uncertainty:.3f} dB")
    ax_b.set_yticks(pos)
    ax_b.set_yticklabels(names)
    ax_b.invert_yaxis()
    ax_b.set_xlabel("Contribution to combined uncertainty [dB]")
    ax_b.set_title("GUM uncertainty budget", fontweight="bold", pad=10)
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
    ax_m.set_title(f"Y = {result.value:.2f} dB,  U = {big:.2f} dB (k = {k:.2f})",
                   fontweight="bold", pad=10)
    ax_m.grid(color=COLOR_GRID, linestyle="-", alpha=0.4)
    ax_m.set_axisbelow(True)
    ax_m.legend(loc="upper right")

    plt.tight_layout()
    save_figure(output_dir, "uncertainty_budget.png")
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
        ax.set_title(f"r = {res.statistic} runs, accept "
                     f"({res.bounds[0]}, {res.bounds[1]}]: {verdict}",
                     fontweight="bold", pad=10)
        ax.set_xlabel("Sample index")
        ax.set_ylabel("Sequence value")
        ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5)
        ax.set_axisbelow(True)
        ax.legend(loc="lower left", fontsize=8.5)
    plt.suptitle("Runs Test About the Median (Wald & Wolfowitz)",
                 fontweight="bold")
    plt.tight_layout()
    save_figure(output_dir, "runs_test.svg")
    plt.close()
