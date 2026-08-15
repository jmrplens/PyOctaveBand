#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Figures for the correlation and cepstrum guides: what a record repeats.

The analyses that answer the lag question instead of the frequency one: the
correlators and their three normalizations, GCC-PHAT and the sub-sample
alignment it supports, the cepstra and lifters that separate a source wavelet
from its echo, and the envelope and synchronous averages that pull a periodic
waveform out of noise. Everything here is embedded by a page under
``signals/spectra/``.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal as scipy_signal

from phonometry._plot.common import format_frequency_axis, theme_fill

from .theme import (
    COLOR_FG,
    COLOR_GRID,
    COLOR_PRIMARY,
    COLOR_SECONDARY,
    COLOR_TERTIARY,
    LABEL_FREQ_HZ,
    save_figure,
)


def generate_gcc_phat_delay(output_dir: str) -> None:
    """GCC-PHAT vs direct correlation for TDE on a colored signal pair."""
    print("Generating gcc_phat_delay...")
    from scipy import signal as sp_signal

    from phonometry import noise_signal, time_delay

    fs = 8192.0
    delay = 20  # samples -> 2.44 ms
    # Colored common signal: a Butterworth roll-off keeps some power in
    # every band (the Knapp & Carter condition for a usable PHAT phase).
    b, a = sp_signal.butter(2, 800.0 / (fs / 2.0))
    s = sp_signal.lfilter(b, a, noise_signal(fs, 4.0, color="white", seed=10))
    noise_x = noise_signal(fs, 4.0, color="white", rms=0.02, seed=11)
    noise_y = noise_signal(fs, 4.0, color="white", rms=0.02, seed=12)
    x = s + noise_x
    y = np.roll(s, delay) + noise_y

    direct = time_delay(x, y, fs, method="direct", max_delay=0.01)
    phat = time_delay(x, y, fs, method="gcc", weighting="phat",
                      nperseg=2048, max_delay=0.01)

    _fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(1e3 * direct.lags,
            direct.correlation / np.max(np.abs(direct.correlation)),
            color=COLOR_PRIMARY, linewidth=1.6,
            label="Direct cross-correlation")
    ax.plot(1e3 * phat.lags, phat.correlation, color=COLOR_SECONDARY,
            linewidth=1.6, label="GCC-PHAT")
    ax.axvline(1e3 * delay / fs, color=COLOR_FG, linestyle="--",
               linewidth=1.4, alpha=0.7, label="True delay (20 samples)")
    ax.set_xlabel("Lag [ms]")
    ax.set_ylabel("Normalized correlation")
    ax.set_title("Time-Delay Estimation: GCC-PHAT vs Direct Correlation "
                 "(Knapp & Carter)", pad=12)
    ax.set_xlim(-10.0, 10.0)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=9)
    ax.text(0.985, 0.965,
            "colored signal: the plain correlator smears the peak,\n"
            "PHAT prewhitens the cross-spectrum and restores it",
            transform=ax.transAxes, va="top", ha="right", fontsize=8.5,
            color=COLOR_FG)
    plt.tight_layout()
    save_figure(output_dir, "gcc_phat_delay.svg")
    plt.close()


def generate_cepstrum_echo(output_dir: str) -> None:
    """Echo detection on the power cepstrum of an IR with one reflection."""
    print("Generating cepstrum_echo...")
    from scipy import signal as sp_signal

    from phonometry import echo_detection, noise_signal

    fs = 48000.0
    delay_s = 0.008  # one floor reflection 8 ms after the direct sound
    a = 0.5
    # Broadband direct sound: a band-passed click, plus the scaled echo and
    # a -80 dB noise floor.
    n = 12000
    impulse = np.zeros(n)
    impulse[0] = 1.0
    b, bb = sp_signal.butter(2, [0.004, 0.9], btype="bandpass")
    direct = sp_signal.lfilter(b, bb, impulse)
    ir = direct + a * np.roll(direct, round(delay_s * fs))
    ir += noise_signal(fs, n / fs, color="white", rms=1e-4, seed=13)

    res = echo_detection(ir, fs, min_quefrency=0.002)

    _fig, ax = plt.subplots(figsize=(10, 6))
    half = res.nfft // 2 + 1
    ax.plot(1e3 * res.quefrencies[:half], res.cepstrum[:half],
            color=COLOR_PRIMARY, linewidth=1.1, label="Power cepstrum")
    ax.axvspan(1e3 * res.search_range[0], 1e3 * res.search_range[1],
               color=theme_fill(COLOR_PRIMARY, ax), zorder=0,
               label="Searched band")
    ax.axvline(1e3 * delay_s, color=COLOR_FG, linestyle="--", linewidth=1.3,
               alpha=0.7, label="True echo delay (8 ms)")
    ax.plot([1e3 * res.delay], [res.reflection_coefficient], "v",
            color=COLOR_SECONDARY, markersize=10,
            label="Detected peak (height = reflection $a$)")
    ax.set_xlim(0.0, 30.0)
    ax.set_ylim(-0.3, 0.65)
    ax.set_xlabel("Quefrency [ms]")
    ax.set_ylabel("Cepstrum")
    ax.set_title("Echo Detection on the Power Cepstrum (Quefrency Analysis)",
                 pad=12)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=9)
    ax.text(0.985, 0.60,
            "spectral ripple of period 1/(8 ms) collapses to one\n"
            "spike at 8 ms whose height reads the reflection",
            transform=ax.transAxes, va="top", ha="right", fontsize=8.5,
            color=COLOR_FG)
    plt.tight_layout()
    save_figure(output_dir, "cepstrum_echo.svg")
    plt.close()


def generate_envelope_spectrum(output_dir: str) -> None:
    """Envelope spectrum of an AM tone in noise: the line at fm."""
    print("Generating envelope_spectrum...")
    from phonometry import envelope_spectrum, noise_signal

    fs = 8192.0
    seconds = 4.0
    t = np.arange(int(seconds * fs)) / fs
    a0, m, fm = 1.0, 0.4, 25.0
    x = a0 * (1.0 + m * np.cos(2.0 * np.pi * fm * t)) * np.cos(
        2.0 * np.pi * 1000.0 * t
    )
    x += noise_signal(fs, seconds, color="white", rms=0.03, seed=8)

    res = envelope_spectrum(x, fs)

    _fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(res.frequencies, res.amplitude, color=COLOR_PRIMARY,
            linewidth=1.4, label="Envelope spectrum")
    ax.axvline(fm, color=COLOR_FG, linestyle="--", linewidth=1.3, alpha=0.7,
               label="Modulation frequency (25 Hz)")
    ax.axhline(a0 * m, color=COLOR_SECONDARY, linestyle=":", linewidth=1.4,
               label=r"Exact line amplitude $A_0 m$ = 0.4")
    ax.set_xlim(0.0, 100.0)
    ax.set_ylim(0.0, 0.5)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Modulation amplitude")
    ax.set_title("Envelope Spectrum of an AM Tone (Bendat & Piersol 13.3)",
                 pad=12)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=9)
    ax.text(0.985, 0.70,
            "the carrier is at 1 kHz; its amplitude modulation\n"
            "appears as one line at exactly $f_m$",
            transform=ax.transAxes, va="top", ha="right", fontsize=8.5,
            color=COLOR_FG)
    plt.tight_layout()
    save_figure(output_dir, "envelope_spectrum.svg")
    plt.close()


def generate_synchronous_average(output_dir: str) -> None:
    """TSA: a periodic waveform pulled from noise, and the comb filter."""
    print("Generating synchronous_average...")
    from phonometry import (
        comb_filter_response,
        noise_signal,
        time_synchronous_average,
    )

    fs = 8192.0
    period = 1.0 / 32.0  # one revolution: 256 samples at this rate
    m = 256
    n_avg = 40
    phase = np.arange((n_avg + 1) * m) / m
    periodic = (
        np.cos(2.0 * np.pi * phase)
        + 0.5 * np.cos(2.0 * np.pi * 3.0 * phase + 0.4)
        - 0.3 * np.cos(2.0 * np.pi * 6.0 * phase)
    )
    signal = periodic + noise_signal(fs, phase.size / fs, rms=0.9, seed=11)
    res = time_synchronous_average(signal, fs, period, n_averages=n_avg)
    true_one = periodic[:m]

    _fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 4.6))

    # Panel (a): one noisy period against the recovered average.
    t_ms = 1e3 * res.times
    ax0.plot(t_ms, signal[:m], color=COLOR_GRID, linewidth=1.0,
             label="One noisy period")
    ax0.plot(t_ms, res.period_waveform, color=COLOR_PRIMARY, linewidth=1.8,
             label=f"Average of $N$ = {n_avg} periods")
    ax0.plot(t_ms, true_one, color=COLOR_SECONDARY, linestyle="--",
             linewidth=1.2, label="True periodic waveform")
    ax0.set_xlim(0.0, 1e3 * period)
    ax0.set_xlabel("Time [ms]")
    ax0.set_ylabel("Amplitude")
    ax0.set_title("Periodic Waveform Extracted from Noise",
                  pad=10)
    ax0.grid(color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax0.set_axisbelow(True)
    ax0.legend(loc="upper right", fontsize=8)
    ax0.text(0.02, 0.03,
             "averaging $N$ periods lowers the asynchronous\n"
             "noise by $\\sqrt{N}$ in amplitude",
             transform=ax0.transAxes, va="bottom", ha="left", fontsize=8.5,
             color=COLOR_FG)

    # Panel (b): comb filter, node selection at 32.05 orders.
    orders = np.linspace(31.0, 33.0, 4000)
    freqs = orders / period
    c20 = comb_filter_response(freqs, period, 20)
    c32 = comb_filter_response(freqs, period, 32)
    ax1.plot(orders, c32, color=COLOR_TERTIARY, linewidth=1.2,
             label="$N$ = 32 (power of two)")
    ax1.plot(orders, c20, color=COLOR_PRIMARY, linewidth=1.4,
             label="$N$ = 20 (node on 32.05)")
    ax1.axvline(32.05, color=COLOR_SECONDARY, linestyle=":", linewidth=1.3,
                label="Interfering tone (32.05)")
    ax1.set_xlim(31.0, 33.0)
    ax1.set_ylim(0.0, 1.05)
    ax1.set_xlabel("Frequency [orders]")
    ax1.set_ylabel("Comb filter magnitude")
    ax1.set_title("Rejecting a Tone by Choosing $N$ (McFadden 1987)",
                  pad=10)
    ax1.grid(color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax1.set_axisbelow(True)
    ax1.legend(loc="upper right", fontsize=8)
    ax1.text(0.02, 0.55,
             "$N$ = 20 puts a node on 32.05 orders and removes\n"
             "it; the power-of-two $N$ = 32 lets it through",
             transform=ax1.transAxes, va="top", ha="left", fontsize=8.5,
             color=COLOR_FG)

    plt.tight_layout()
    save_figure(output_dir, "synchronous_average.svg")
    plt.close()


def generate_cepstrum_variants(output_dir: str) -> None:
    """Power, real and complex cepstra of one echo-carrying record."""
    print("Generating cepstrum_variants...")
    from phonometry import cepstrum

    fs = 48000.0
    # A band-limited source wavelet (its cepstrum concentrates below 1 ms)
    # plus one reflection at 8 ms with reflection coefficient a = 0.5.
    b, a_coef = scipy_signal.butter(2, 0.3)
    s = np.zeros(4096)
    s[37: 37 + 256] = scipy_signal.lfilter(
        b, a_coef, np.r_[1.0, np.zeros(255)]
    )
    x = s + 0.5 * np.roll(s, 384)                # echo: 8 ms, a = 0.5

    _fig, ax = plt.subplots(figsize=(10, 6))
    axins = ax.inset_axes((0.60, 0.28, 0.26, 0.42))
    variants = (
        ("power", COLOR_PRIMARY, "-", "Power cepstrum"),
        ("real", COLOR_TERTIARY, "--", "Real cepstrum (exactly half the power)"),
        ("complex", COLOR_SECONDARY, ":", "Complex cepstrum"),
    )
    for kind, colour, style, label in variants:
        res = cepstrum(x, fs, kind=kind)
        q_ms = 1e3 * res.quefrencies
        mask = (q_ms > 0.5) & (q_ms <= 20.0)
        ax.plot(q_ms[mask], res.cepstrum[mask], color=colour, linestyle=style,
                linewidth=1.1, label=label)
        zoom = (q_ms > 7.8) & (q_ms <= 8.2)
        axins.plot(q_ms[zoom], res.cepstrum[zoom], color=colour,
                   linestyle=style, linewidth=1.3, marker="o", markersize=3)
    axins.set_xlim(7.85, 8.15)
    axins.set_ylim(-0.05, 0.55)
    axins.tick_params(labelsize=7)
    axins.grid(color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax.indicate_inset_zoom(axins, edgecolor=COLOR_FG, alpha=0.5)
    ax.annotate("first rahmonic at 8 ms:\n"
                "height $\\approx a$ on the power cepstrum",
                xy=(8.0, 0.5), xytext=(2.5, 0.42), fontsize=9,
                color=COLOR_FG,
                arrowprops={"arrowstyle": "->", "lw": 0.9, "color": COLOR_FG})
    ax.annotate("second rahmonic: $-a^2/2$",
                xy=(16.0, -0.125), xytext=(12.0, -0.22), fontsize=9,
                color=COLOR_FG,
                arrowprops={"arrowstyle": "->", "lw": 0.9, "color": COLOR_FG})
    ax.set_xlabel("Quefrency [ms]")
    ax.set_ylabel("Cepstrum")
    ax.set_xlim(0.5, 20.0)
    ax.set_ylim(-0.3, 0.6)
    ax.set_title("The Three Cepstrum Variants of One Echo-Carrying Record",
                 pad=12)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    save_figure(output_dir, "cepstrum_variants.svg")
    plt.close()


def generate_lifter_split(output_dir: str) -> None:
    """Lowpass/highpass liftering of a log spectrum with an 8 ms echo."""
    print("Generating lifter_split...")
    from phonometry import lifter

    fs = 48000.0
    # The same wavelet-plus-echo record as the cepstrum-variants figure:
    # a smooth source envelope carrying a pure 8 ms echo ripple (a = 0.5),
    # so the highpass ripple swings exactly between the closed forms.
    b, a_coef = scipy_signal.butter(2, 0.3)
    s = np.zeros(4096)
    s[37: 37 + 256] = scipy_signal.lfilter(
        b, a_coef, np.r_[1.0, np.zeros(255)]
    )
    x = s + 0.5 * np.roll(s, 384)                # the same 8 ms echo

    low = lifter(x, fs, cutoff=0.004, mode="lowpass")
    high = lifter(x, fs, cutoff=0.004, mode="highpass")
    band = (low.frequencies >= 500.0) & (low.frequencies <= 2000.0)

    _fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].semilogx(low.frequencies[band], low.spectrum_db[band],
                     color="#7f7f7f", linewidth=0.7, alpha=0.8,
                     label="Log spectrum of the record")
    axes[0].semilogx(low.frequencies[band], low.liftered_db[band],
                     color=COLOR_PRIMARY, linewidth=2.0,
                     label="Lowpass lifter: spectral envelope")
    axes[0].set_ylabel("Magnitude [dB]")
    axes[0].legend(loc="upper right", fontsize=9)

    axes[1].semilogx(high.frequencies[band], high.liftered_db[band],
                     color=COLOR_SECONDARY, linewidth=1.1,
                     label="Highpass lifter: the echo ripple alone")
    for bound in (20.0 * np.log10(1.5), 20.0 * np.log10(0.5)):
        axes[1].axhline(bound, color=COLOR_TERTIARY, linestyle="--",
                        linewidth=1.2,
                        label=(r"closed-form ripple bounds $20\log_{10}(1\pm a)$"
                               if bound > 0 else None))
    axes[1].set_ylabel("Magnitude [dB]")
    axes[1].set_xlabel(LABEL_FREQ_HZ)
    axes[1].legend(loc="upper right", fontsize=9)

    for ax in axes:
        ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5, which="both")
        ax.set_axisbelow(True)
        format_frequency_axis(ax, 500.0, 2000.0)
    axes[0].set_title("Liftering at 4 ms: Envelope Versus Echo Ripple",
                      pad=12)
    plt.tight_layout()
    save_figure(output_dir, "lifter_split.svg")
    plt.close()


def generate_correlation_normalizations(output_dir: str) -> None:
    """The three correlation normalizations of a two-sensor delay model."""
    print("Generating correlation_normalizations...")
    from phonometry import correlation, noise_signal

    fs = 8192.0
    delay = 102                                   # 12.45 ms
    x = noise_signal(fs, 2.0, seed=4)
    interference = noise_signal(fs, 2.0, rms=0.5, seed=5)
    y = 0.8 * np.concatenate([np.zeros(delay), x[:-delay]]) + interference

    coeff = correlation(x, y, fs, normalization="coefficient", max_lag=0.05)
    biased = correlation(x, y, fs, normalization="biased")
    unbiased = correlation(x, y, fs, normalization="unbiased")

    _fig, (ax_c, ax_n) = plt.subplots(2, 1, figsize=(10, 7))
    ax_c.plot(1e3 * coeff.lags, coeff.values, color=COLOR_PRIMARY,
              linewidth=1.1,
              label=r"Coefficient $\rho_{xy}(\tau)$ (bounded by $\pm 1$)")
    ax_c.axvline(1e3 * delay / fs, color=COLOR_SECONDARY, linestyle="--",
                 linewidth=1.2, label="true delay +12.5 ms")
    ax_c.set_xlabel("Lag [ms]")
    ax_c.set_ylabel("Correlation")
    ax_c.set_ylim(-1.05, 1.05)
    ax_c.legend(loc="upper left", fontsize=9)

    ax_n.plot(unbiased.lags, unbiased.values, color=COLOR_SECONDARY,
              linewidth=0.5, alpha=0.9,
              label=r"Unbiased $1/(N-|r|)$ (variance grows at the ends)")
    ax_n.plot(biased.lags, biased.values, color=COLOR_PRIMARY,
              linewidth=0.5,
              label=r"Biased $1/N$ (tapers toward the ends)")
    ax_n.set_xlabel("Lag [s]")
    ax_n.set_ylabel("Correlation")
    ax_n.legend(loc="upper left", fontsize=9)

    for ax in (ax_c, ax_n):
        ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5)
        ax.set_axisbelow(True)
    ax_c.set_title("Correlation Normalizations of a Two-Sensor Delay Model",
                   pad=12)
    plt.tight_layout()
    save_figure(output_dir, "correlation_normalizations.svg")
    plt.close()


def generate_ir_alignment(output_dir: str) -> None:
    """Sub-sample alignment of an impulse response onto a reference."""
    print("Generating ir_alignment...")
    from phonometry import align_impulse_responses, fractional_delay

    fs = 48000.0
    t = np.arange(int(0.03 * fs)) / fs
    rng = np.random.default_rng(6)
    # Band-limited reference pulse: a 2 kHz Gaussian tone burst at 5 ms.
    ir_a = scipy_signal.gausspulse(t - 0.005, fc=2000.0, bw=0.5)
    ir_b = fractional_delay(ir_a, 7.37)[: ir_a.size]
    ir_b += 0.005 * rng.standard_normal(ir_a.size)

    res = align_impulse_responses(ir_b, ir_a, fs)
    t_ms = 1e3 * t

    _fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(t_ms, res.reference, color=COLOR_PRIMARY, linewidth=1.6,
            label="Reference IR")
    ax.plot(t_ms, ir_b, color="#7f7f7f", linewidth=1.0, linestyle="--",
            alpha=0.8, label="Measured IR (delayed 7.37 samples)")
    ax.plot(t_ms, res.aligned[: t.size], color=COLOR_SECONDARY,
            linewidth=1.0, linestyle=":", label="Aligned IR (delay removed)")
    ax.text(0.985, 0.05,
            f"estimated delay removed: {res.delay_samples:.2f} samples",
            transform=ax.transAxes, va="bottom", ha="right", fontsize=9,
            color=COLOR_FG)
    ax.set_xlim(2.0, 9.0)
    ax.set_xlabel("Time [ms]")
    ax.set_ylabel("Amplitude")
    ax.set_title("Sub-Sample Impulse-Response Alignment",
                 pad=12)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    save_figure(output_dir, "ir_alignment.svg")
    plt.close()


def generate_hilbert_envelope(output_dir: str) -> None:
    """Hilbert envelope and instantaneous frequency of a decaying mode."""
    print("Generating hilbert_envelope...")
    from phonometry import envelope

    fs = 8192.0
    t = np.arange(int(0.4 * fs)) / fs
    rng = np.random.default_rng(7)
    decay = np.exp(-t / 0.1)                     # a struck 250 Hz mode
    x = decay * np.sin(2.0 * np.pi * 250.0 * t)
    x += 0.001 * rng.standard_normal(t.size)

    # The envelope of a narrowband signal is low-frequency: the anti-aliased
    # x8 decimation keeps the outputs compact without losing the decay.
    res = envelope(x, fs, decimation_factor=8)

    _fig, (ax_e, ax_f) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    ax_e.plot(t, res.signal, color=COLOR_PRIMARY, linewidth=0.6,
              label="Signal")
    ax_e.plot(res.times, res.envelope, color=COLOR_SECONDARY, linewidth=1.8,
              label="Envelope $A(t)$")
    ax_e.plot(res.times, -res.envelope, color=COLOR_SECONDARY, linewidth=1.8)
    ax_e.set_ylabel("Amplitude")
    ax_e.legend(loc="upper right", fontsize=9)

    ax_f.plot(res.times, res.instantaneous_frequency, color=COLOR_PRIMARY,
              linewidth=0.9, label="Instantaneous frequency $f(t)$")
    ax_f.axhline(250.0, color=COLOR_TERTIARY, linestyle="--", linewidth=1.4,
                 label="carrier 250 Hz")
    ax_f.set_ylim(230.0, 270.0)
    ax_f.set_ylabel("Instantaneous frequency [Hz]")
    ax_f.set_xlabel("Time [s]")
    ax_f.legend(loc="upper right", fontsize=9)

    for ax in (ax_e, ax_f):
        ax.set_xlim(0.0, 0.3)
        ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5)
        ax.set_axisbelow(True)
    ax_e.set_title("Hilbert Envelope and Instantaneous Frequency",
                   pad=12)
    plt.tight_layout()
    save_figure(output_dir, "hilbert_envelope.svg")
    plt.close()


def generate_tsa_noise_reduction(output_dir: str) -> None:
    """TSA: measured error of the average against the ideal 1/sqrt(N)."""
    print("Generating tsa_noise_reduction...")
    from phonometry import time_synchronous_average

    fs = 8192.0
    samples = 256
    period = samples / fs
    m = np.arange(samples) / fs
    true = (np.cos(2.0 * np.pi * m / period)
            + 0.5 * np.cos(2.0 * np.pi * 3.0 * m / period + 0.7)
            + 0.25 * np.cos(2.0 * np.pi * 5.0 * m / period + 1.1))
    rng = np.random.default_rng(5)
    n_max = 128
    x = np.tile(true, n_max) + rng.standard_normal(n_max * samples)

    counts = [1, 2, 4, 8, 16, 32, 64, 128]
    errors = []
    for n in counts:
        res = time_synchronous_average(x[: n * samples], fs, period,
                                       n_averages=n)
        errors.append(float(np.sqrt(np.mean(
            (res.period_waveform - true) ** 2))))

    _fig, ax = plt.subplots(figsize=(10, 6))
    ax.loglog(counts, errors, "o-", color=COLOR_PRIMARY, linewidth=1.6,
              markersize=6, label="Measured RMS error of the average")
    ax.loglog(counts, 1.0 / np.sqrt(np.asarray(counts, dtype=np.float64)),
              color=COLOR_SECONDARY, linestyle="--", linewidth=1.4,
              label=r"Ideal $\sigma/\sqrt{N}$")
    ax.set_xticks(counts)
    ax.set_xticklabels([str(n) for n in counts])
    ax.set_xlabel("Number of averages $N$")
    ax.set_ylabel("RMS error of the averaged waveform")
    ax.set_title(r"TSA Noise Reduction: the $\sqrt{N}$ Law (McFadden 1987)",
                 pad=12)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5, which="both")
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    save_figure(output_dir, "tsa_noise_reduction.svg")
    plt.close()
