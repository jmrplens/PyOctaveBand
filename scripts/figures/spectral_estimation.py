#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Figures for the spectral estimation guides: how much power, and how sure.

The estimators that answer the frequency question about a record and state
their own uncertainty: the Welch and Thomson multitaper densities with their
chi-square confidence intervals, the window trade-off underneath them, the
zoom and the calibrated time-frequency views of one record, and the
cross-spectral estimates that carry the same question to a second and a third
input. Everything here is embedded by a page under ``signals/spectra/``.
"""

from typing import Literal

import matplotlib.pyplot as plt
import numpy as np

from phonometry._plot.common import (
    format_frequency_axis,
    theme_fill_alpha,
    theme_line,
)

from .theme import (
    COLOR_FG,
    COLOR_GRID,
    COLOR_PRIMARY,
    COLOR_QUATERNARY,
    COLOR_SECONDARY,
    COLOR_TERTIARY,
    LABEL_FREQ_HZ,
    save_figure,
)

#: The colour names ``noise_signal`` accepts, kept in step with its signature.
NoiseColor = Literal["white", "pink", "red", "blue", "violet"]


def generate_psd_confidence_smoothing(output_dir: str) -> None:
    """Calibrated PSD of pink noise: chi-square CI plus 1/3-oct smoothing."""
    print("Generating psd_confidence_smoothing...")
    from phonometry import (
        fractional_octave_smoothing,
        noise_signal,
        power_spectral_density,
    )

    fs = 48000.0
    x = noise_signal(fs, 20.0, color="pink", seed=11)
    res = power_spectral_density(x, fs, nperseg=4096)
    band = (res.frequencies >= 20.0) & (res.frequencies <= 20000.0)
    freqs = res.frequencies[band]
    smooth = fractional_octave_smoothing(res.frequencies, res.psd, 3.0)[band]
    # The exact -3.01 dB/oct power law through the level at 1 kHz.
    i0 = int(np.argmin(np.abs(freqs - 1000.0)))
    ref_db = 10.0 * np.log10(smooth[i0]) - 10.0 * np.log10(freqs / freqs[i0])

    _fig, ax = plt.subplots(figsize=(10, 6.2))
    ax.fill_between(
        freqs,
        10.0 * np.log10(res.ci_lower[band]),
        10.0 * np.log10(res.ci_upper[band]),
        color=COLOR_PRIMARY, alpha=0.28, lw=0.0,
        label="95 % chi-square confidence interval")
    ax.semilogx(freqs, 10.0 * np.log10(res.psd[band]), color=COLOR_PRIMARY,
                linewidth=1.0, alpha=0.85, label="Welch PSD estimate")
    ax.semilogx(freqs, 10.0 * np.log10(smooth), color=COLOR_SECONDARY,
                linewidth=2.2, label="1/3-octave smoothed")
    ax.semilogx(freqs, ref_db, color=COLOR_FG, linestyle="--", linewidth=1.4,
                alpha=0.7, label="Exact -3.01 dB/octave power law")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("PSD [dB re 1/Hz]")
    ax.set_title("Calibrated Spectral Density of Pink Noise (Bendat & Piersol)",
                 fontweight="bold", pad=12)
    ax.set_xlim(20.0, 20000.0)
    format_frequency_axis(ax, 20.0, 20000.0)
    ax.grid(which="both", color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", fontsize=9)
    ax.text(0.985, 0.965,
            f"$n_d$ = {res.n_averages:.0f} averages, "
            f"$\\varepsilon_r$ = {100.0 * res.random_error:.1f} %",
            transform=ax.transAxes, va="top", ha="right", fontsize=8.5,
            color=COLOR_FG)
    plt.tight_layout()
    save_figure(output_dir, "psd_confidence_smoothing.svg")


def generate_multitaper_psd_confidence(output_dir: str) -> None:
    """Thomson multitaper PSD of a short record vs a single-taper estimate."""
    print("Generating multitaper_psd_confidence...")
    from phonometry import multitaper_psd, noise_signal

    fs = 48000.0
    n = 8192  # a genuinely short record: 171 ms
    x = noise_signal(fs, n / fs, color="pink", seed=11)
    single = multitaper_psd(x, fs, n_tapers=1, adaptive=False)
    res = multitaper_psd(x, fs)
    band = (res.frequencies >= 20.0) & (res.frequencies <= 20000.0)
    freqs = res.frequencies[band]
    # The exact -3.01 dB/oct power law through the mean level around 1 kHz.
    anchor = (freqs >= 800.0) & (freqs <= 1250.0)
    level_1k = float(np.mean(10.0 * np.log10(res.psd[band][anchor])))
    ref_db = level_1k - 10.0 * np.log10(freqs / 1000.0)

    _fig, (ax, ax_hd) = plt.subplots(1, 2, figsize=(13.0, 6.0))
    # The estimator the reader would otherwise reach for on this record: Welch
    # with a 2048-sample segment, which fits about seven segments into 171 ms.
    from phonometry import power_spectral_density

    welch = power_spectral_density(x, fs, nperseg=2048)
    wband = (welch.frequencies >= 20.0) & (welch.frequencies <= 20000.0)
    ax.fill_between(
        welch.frequencies[wband],
        10.0 * np.log10(welch.ci_lower[wband]),
        10.0 * np.log10(welch.ci_upper[wband]),
        color=COLOR_SECONDARY, alpha=0.20, lw=0.0,
        label=f"Welch 95 % interval ($n_d$ = {welch.n_averages:.1f})")
    ax.semilogx(welch.frequencies[wband],
                10.0 * np.log10(welch.psd[wband]), color=COLOR_SECONDARY,
                linewidth=0.9, alpha=0.9, label="Welch, nperseg = 2048")
    # The single-taper estimate is the noisy thing the multitaper estimate is
    # being compared against, so it is drawn back: as a grey chosen against
    # the page, not as a grey diluted into it.
    ax.semilogx(freqs, 10.0 * np.log10(single.psd[band]),
                color=theme_line("gray", ax, quiet=0.45), linewidth=0.7,
                label="Single Slepian taper ($K$ = 1, $\\nu$ = 2)")
    ax.fill_between(
        freqs,
        10.0 * np.log10(res.ci_lower[band]),
        10.0 * np.log10(res.ci_upper[band]),
        color=COLOR_PRIMARY, alpha=0.28, lw=0.0,
        label="95 % chi-square confidence interval")
    ax.semilogx(freqs, 10.0 * np.log10(res.psd[band]), color=COLOR_PRIMARY,
                linewidth=1.2, label="Multitaper estimate ($K$ = 7, adaptive)")
    ax.semilogx(freqs, ref_db, color=COLOR_FG, linestyle="--", linewidth=1.4,
                alpha=0.7, label="Exact -3.01 dB/octave power law")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("PSD [dB re 1/Hz]")
    ax.set_title(
        "Thomson Multitaper Density of a Short Record (Percival & Walden)",
        fontweight="bold", pad=12)
    ax.set_xlim(20.0, 20000.0)
    format_frequency_axis(ax, 20.0, 20000.0)
    ax.grid(which="both", color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", fontsize=9)
    nu = float(np.mean(res.degrees_of_freedom[1:-1]))
    ax.text(0.985, 0.965,
            f"171 ms record, $NW$ = 4, "
            f"$\\bar\\nu$ = {nu:.1f} equivalent dof",
            transform=ax.transAxes, va="top", ha="right", fontsize=8.5,
            color=COLOR_FG)

    # --- Right: a high-dynamic-range case, where the adaptive weights earn
    # their keep and the equivalent degrees of freedom pay for it. ---
    n_hd = 1 << 15
    tt = np.arange(n_hd) / fs
    floor = noise_signal(fs, n_hd / fs, color="pink", seed=3)
    tone_amp = float(np.sqrt(np.mean(floor ** 2))) * 10 ** (60.0 / 20.0)
    hd = floor + tone_amp * np.sin(2 * np.pi * 1000.0 * tt)
    w_hd = power_spectral_density(hd, fs, nperseg=4096)
    m_hd = multitaper_psd(hd, fs)
    hb = (m_hd.frequencies >= 200.0) & (m_hd.frequencies <= 5000.0)
    wb = (w_hd.frequencies >= 200.0) & (w_hd.frequencies <= 5000.0)
    ax_hd.semilogx(w_hd.frequencies[wb], 10.0 * np.log10(w_hd.psd[wb]),
                   color=COLOR_SECONDARY, linewidth=1.0,
                   label="Welch, Hann taper")
    ax_hd.semilogx(m_hd.frequencies[hb], 10.0 * np.log10(m_hd.psd[hb]),
                   color=COLOR_PRIMARY, linewidth=1.2,
                   label="Multitaper, adaptive")
    ax_hd.set_xlabel("Frequency [Hz]")
    ax_hd.set_ylabel("PSD [dB re 1/Hz]")
    ax_hd.set_title("A 60 dB tone over a pink floor, and what it costs",
                    fontweight="bold", pad=12)
    ax_hd.set_xlim(200.0, 5000.0)
    format_frequency_axis(ax_hd, 200.0, 5000.0)
    ax_hd.grid(which="both", color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax_hd.set_axisbelow(True)
    ax_dof = ax_hd.twinx()
    ax_dof.semilogx(m_hd.frequencies[hb], m_hd.degrees_of_freedom[hb],
                    color=COLOR_TERTIARY, linewidth=1.1, alpha=0.85,
                    label="equivalent dof (right axis)")
    ax_dof.set_ylabel("Equivalent degrees of freedom")
    ax_dof.set_ylim(0.0, 2.4 * float(np.max(m_hd.degrees_of_freedom[hb])))
    # The twin axis carries its own x formatter, which would print 10^n.
    format_frequency_axis(ax_dof, 200.0, 5000.0)
    handles, labels = ax_hd.get_legend_handles_labels()
    h2, l2 = ax_dof.get_legend_handles_labels()
    ax_hd.legend(handles + h2, labels + l2, loc="upper left", fontsize=8.5)

    plt.tight_layout()
    save_figure(output_dir, "multitaper_psd_confidence.svg")


def generate_psd_segment_tradeoff(output_dir: str) -> None:
    """Segment length against a resonance: bias one way, variance the other."""
    print("Generating psd_segment_tradeoff...")
    from scipy import signal as scipy_signal

    from phonometry import (
        noise_signal,
        power_spectral_density,
        resolution_bias_error,
    )

    fs = 48000.0
    f_r, b_r = 1000.0, 25.0                 # a resonance of half-power width 25 Hz
    x = noise_signal(fs, 20.0, color="white", seed=5)
    q = f_r / b_r
    b_coef, a_coef = scipy_signal.iirpeak(f_r, q, fs=fs)
    resonant = scipy_signal.lfilter(b_coef, a_coef, x)

    lengths = (512, 1024, 4096, 16384)
    _fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12.5, 5.6))

    # Left: the peak itself, at four segment lengths.
    colors = (COLOR_FG, COLOR_TERTIARY, COLOR_PRIMARY, COLOR_SECONDARY)
    peak_ref = None
    for nperseg, color in zip(lengths, colors):
        res = power_spectral_density(resonant, fs, nperseg=nperseg)
        band = (res.frequencies >= 880.0) & (res.frequencies <= 1120.0)
        level = 10.0 * np.log10(res.psd[band])
        if peak_ref is None or nperseg == lengths[-1]:
            peak_ref = float(np.max(level))
        ax_l.plot(res.frequencies[band], level, color=color, linewidth=1.5,
                  label=f"nperseg = {nperseg}, "
                        f"$B_e$ = {res.resolution_bandwidth:.1f} Hz")
    ax_l.set_xlim(880.0, 1120.0)
    ax_l.set_xlabel(LABEL_FREQ_HZ)
    ax_l.set_ylabel("PSD [dB re 1/Hz]")
    ax_l.set_title(f"A {b_r:.0f} Hz-wide resonance at {f_r:.0f} Hz",
                   fontweight="bold", pad=10)
    ax_l.grid(color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax_l.set_axisbelow(True)
    ax_l.legend(loc="upper right", fontsize=8.5)

    # Right: the two errors against segment length, and where they cross.
    grid = 2 ** np.arange(9, 17)
    bias_db, random_db, measured = [], [], []
    for segment in grid:
        res = power_spectral_density(resonant, fs, nperseg=int(segment))
        eps_b = resolution_bias_error(res.resolution_bandwidth, b_r)
        bias_db.append(-10.0 * np.log10(max(1.0 + eps_b, 1e-6)))
        random_db.append(10.0 * np.log10(1.0 + res.random_error))
        band = (res.frequencies >= 850.0) & (res.frequencies <= 1150.0)
        measured.append(float(np.max(10.0 * np.log10(res.psd[band]))))
    measured_db = np.asarray(measured)
    ax_r.semilogx(grid, bias_db, "o-", color=COLOR_SECONDARY, base=2,
                  label="resolution bias, $-10\\lg(1+\\varepsilon_b)$ [dB]")
    ax_r.semilogx(grid, random_db, "s-", color=COLOR_PRIMARY, base=2,
                  label="random error, $10\\lg(1+1/\\sqrt{n_d})$ [dB]")
    ax_r.semilogx(grid, measured_db.max() - measured_db, "^--", color=COLOR_TERTIARY,
                  base=2, label="measured peak deficit [dB]")
    # bias - random falls monotonically with nperseg, so interpolate on the
    # reversed (increasing) difference to find where the two are equal.
    diff = np.asarray(bias_db) - np.asarray(random_db)
    cross = float(np.interp(0.0, diff[::-1], np.log2(grid)[::-1]))
    ax_r.axvline(2 ** cross, color=COLOR_FG, linestyle=":", linewidth=1.3)
    ax_r.text(2 ** cross * 1.25, 5.6,
              f"the two errors are equal\nnear nperseg = {2 ** cross:.0f}",
              fontsize=9, color=COLOR_FG)
    ax_r.set_ylim(0.0, 8.0)
    ax_r.set_xlabel("Segment length [samples]")
    ax_r.set_ylabel("Error [dB]")
    ax_r.set_title("Bias falls, variance rises", fontweight="bold", pad=10)
    ax_r.grid(which="both", color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax_r.set_axisbelow(True)
    ax_r.legend(loc="upper center", fontsize=8.5)

    plt.tight_layout()
    save_figure(output_dir, "psd_segment_tradeoff.svg")
    plt.close()


def generate_noise_colors(output_dir: str) -> None:
    """The five exact power-law generators, and their measured slopes."""
    print("Generating noise_colors...")
    from phonometry import noise_signal, power_spectral_density

    fs = 48000.0
    # The names are the generator's own literals, not free-form strings, so
    # spell the type out and a colour it does not know fails here.
    colors: tuple[tuple[NoiseColor, float, str], ...] = (
        ("violet", 2.0, COLOR_QUATERNARY), ("blue", 1.0, COLOR_TERTIARY),
        ("white", 0.0, COLOR_FG), ("pink", -1.0, COLOR_PRIMARY),
        ("red", -2.0, COLOR_SECONDARY))

    _fig, ax = plt.subplots(figsize=(10, 6.4))
    for name, alpha, color in colors:
        x = noise_signal(fs, 20.0, color=name, seed=7)
        res = power_spectral_density(x, fs, nperseg=8192)
        band = (res.frequencies >= 20.0) & (res.frequencies <= 20000.0)
        freqs = res.frequencies[band]
        level = 10.0 * np.log10(res.psd[band])
        # Normalise every colour to 0 dB at 1 kHz so the five slopes fan out
        # from one point instead of overlapping at random offsets.
        anchor = (freqs >= 900.0) & (freqs <= 1100.0)
        level = level - float(np.mean(level[anchor]))
        slope = float(np.polyfit(np.log2(freqs), level, 1)[0])
        # The measured trace sits behind its own exact law, so it has to
        # recede; by colour rather than by opacity, which on the dark page
        # takes the red and the blue down to the ground they sit on.
        ax.semilogx(freqs, level, color=theme_line(color, ax, quiet=0.55),
                    linewidth=1.0)
        ax.semilogx(freqs, 3.0103 * alpha * np.log2(freqs / 1000.0),
                    color=color, linestyle="--", linewidth=1.8,
                    label=f"{name}: measured {slope:+.4f}, "
                          f"exact {3.0103 * alpha:+.4f} dB/octave")
    ax.set_xlim(20.0, 20000.0)
    ax.set_ylim(-42.0, 48.0)
    format_frequency_axis(ax, 20.0, 20000.0)
    ax.set_xlabel(LABEL_FREQ_HZ)
    ax.set_ylabel("PSD re its own 1 kHz level [dB]")
    ax.set_title("The five colours of noise_signal, over three decades",
                 fontweight="bold", pad=12)
    ax.grid(which="both", color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=8.5)
    plt.tight_layout()
    save_figure(output_dir, "noise_colors.svg")
    plt.close()


def generate_calibrated_spectrogram(output_dir: str) -> None:
    """Calibrated STFT spectrogram of a nonstationary signal, in dB SPL."""
    print("Generating calibrated_spectrogram...")
    from phonometry import noise_signal, spectrogram

    fs = 16000.0
    duration = 4.0
    t = np.arange(int(fs * duration)) / fs
    p_ref = 2e-5

    # A siren sweeping 600-1200 Hz at 70 dB SPL, an impact at t = 2.5 s
    # and a pink-noise floor at 45 dB SPL, all in pascals.
    siren_rms = p_ref * 10.0 ** (70.0 / 20.0)
    phase = 2.0 * np.pi * 900.0 * t - 600.0 * np.cos(np.pi * t)
    x = siren_rms * np.sqrt(2.0) * np.cos(phase)
    rng = np.random.default_rng(9)
    n_imp = int(0.06 * fs)
    impact = rng.standard_normal(n_imp) * np.exp(
        -np.arange(n_imp) / (0.012 * fs)
    )
    x[int(2.5 * fs):int(2.5 * fs) + n_imp] += 0.4 * impact
    x += noise_signal(fs, duration, color="pink",
                      rms=p_ref * 10.0 ** (45.0 / 20.0), seed=10)

    res = spectrogram(x, fs, nperseg=1024, overlap=0.75, scaling="spectrum")
    level = 10.0 * np.log10(res.power / p_ref**2)
    vmax = float(np.ceil(level.max()))

    fig, ax = plt.subplots(figsize=(10, 6.2))
    half_hop = 0.5 * res.hop / fs
    df = float(res.frequencies[1] - res.frequencies[0])
    img = ax.imshow(
        level, cmap="magma", vmin=vmax - 55.0, vmax=vmax, aspect="auto",
        origin="lower", interpolation="nearest",
        extent=(float(res.times[0]) - half_hop,
                float(res.times[-1]) + half_hop,
                0.0, float(res.frequencies[-1]) + 0.5 * df),
    )
    fig.colorbar(img, ax=ax, label="Sound pressure level [dB SPL]")
    ax.set_ylim(0.0, 3000.0)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(LABEL_FREQ_HZ)
    ax.set_title("Calibrated Spectrogram in dB SPL (Bendat & Piersol)",
                 fontweight="bold", pad=12)
    ax.text(0.02, 0.965,
            "a siren, an impact and a pink-noise floor:\n"
            "every cell reads an absolute level",
            transform=ax.transAxes, va="top", ha="left", fontsize=8.5,
            color="white")
    plt.tight_layout()
    save_figure(output_dir, "calibrated_spectrogram.png")
    plt.close()


def generate_zoom_fft_resolution(output_dir: str) -> None:
    """Zoom FFT resolving two tones closer than a coarse FFT bin."""
    print("Generating zoom_fft_resolution...")
    from scipy import signal as sp_signal

    from phonometry import zoom_fft

    fs = 8192.0
    t = np.arange(8192) / fs  # 1 s record: 1 Hz true resolution
    x = 0.8 * np.cos(2.0 * np.pi * 997.0 * t) + 0.5 * np.cos(
        2.0 * np.pi * 1000.0 * t
    )

    # Coarse view: a 1024-point FFT of the same record (8 Hz bins).
    w = sp_signal.get_window("hann", 1024)
    coarse = 2.0 * np.abs(np.fft.rfft(x[:1024] * w)) / np.sum(w)
    coarse_f = np.fft.rfftfreq(1024, 1.0 / fs)
    band = (coarse_f >= 950.0) & (coarse_f <= 1050.0)

    # 0.25 Hz grid: four points per record-length resolution, so the two
    # mainlobes are drawn as smooth curves with their exact peaks.
    res = zoom_fft(x, fs, 980.0, 1016.0, n_points=145)

    _fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(coarse_f[band], 20.0 * np.log10(np.maximum(coarse[band], 1e-12)),
            color=COLOR_SECONDARY, marker="o",
            ms=4.0, lw=1.2, ls="--", label="1024-point FFT (8 Hz bins)")
    ax.plot(res.frequencies,
            20.0 * np.log10(np.maximum(res.amplitude, 1e-12)),
            color=COLOR_PRIMARY, lw=1.6,
            label="Zoom FFT of the same record")
    for f0 in (997.0, 1000.0):
        ax.axvline(f0, color=COLOR_FG, ls=":", lw=1.0, alpha=0.6)
    ax.set_xlim(950.0, 1050.0)
    ax.set_ylim(-70.0, 5.0)
    ax.set_xlabel(LABEL_FREQ_HZ)
    ax.set_ylabel("Amplitude [dB]")
    ax.set_title("Zoom FFT Resolves Tones One Coarse Bin Apart "
                 "(Bendat & Piersol)", fontweight="bold", pad=12)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=9)
    ax.text(0.015, 0.965,
            "997 and 1000 Hz, 3 Hz apart:\n"
            "one lump on the 8 Hz grid,\n"
            "two exact lines on the zoom grid",
            transform=ax.transAxes, va="top", ha="left", fontsize=8.5,
            color=COLOR_FG)
    plt.tight_layout()
    save_figure(output_dir, "zoom_fft_resolution.svg")


def generate_window_functions_tradeoff(output_dir: str) -> None:
    """Window spectra with their Harris figures of merit in the legend."""
    print("Generating window_functions_tradeoff...")
    from phonometry import window_metrics

    n, oversample = 1024, 256
    cases = [
        ("boxcar", COLOR_FG, "-"),
        ("hann", COLOR_PRIMARY, "-"),
        ("hamming", COLOR_SECONDARY, "-"),
        ("blackman", COLOR_TERTIARY, "-"),
    ]
    _fig, ax = plt.subplots(figsize=(10, 6.2))
    for name, color, style in cases:
        res = window_metrics(name, n)
        spectrum = np.abs(np.fft.rfft(res.taps, n=n * oversample))
        level = 20.0 * np.log10(spectrum / spectrum[0])
        bins = np.arange(level.size) / oversample
        shown = bins <= 16.0
        ax.plot(bins[shown], level[shown], color=color, linestyle=style,
                linewidth=1.4, alpha=0.9,
                label=(f"{name}: ENBW {res.enbw_bins:.2f} bins, "
                       f"sidelobe {res.highest_sidelobe_db:.1f} dB"))
    ax.set_xlim(0.0, 16.0)
    ax.set_ylim(-100.0, 5.0)
    ax.set_xlabel("Frequency offset [DFT bins]")
    ax.set_ylabel("Level re main lobe [dB]")
    ax.set_title("Window Functions: The Spectral Trade-off (Harris 1978)",
                 fontweight="bold", pad=12)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    save_figure(output_dir, "window_functions_tradeoff.svg")
    plt.close()


def generate_miso_coherence(output_dir: str) -> None:
    """MISO coherence: which correlated source dominates each band."""
    print("Generating miso_coherence...")
    from scipy import signal as sp_signal

    from phonometry import miso_coherence, noise_signal

    fs = 8192.0
    seconds = 32.0
    # x1 drives a low-frequency path; x2 = 0.7*x1 + independent noise drives a
    # high-frequency path. x2 is correlated with x1, so its ORDINARY coherence
    # with the output is inflated in the low band (borrowed through x1), while
    # its PARTIAL coherence is clean once x1 is conditioned out.
    x1 = noise_signal(fs, seconds, color="white", seed=1)
    x2 = 0.7 * x1 + noise_signal(fs, seconds, color="white", seed=2)
    low = sp_signal.butter(4, 400.0, fs=fs, output="sos")
    high = sp_signal.butter(4, 1500.0, btype="high", fs=fs, output="sos")
    noise = noise_signal(fs, seconds, color="white", rms=0.05, seed=3)
    y = sp_signal.sosfilt(low, x1) + sp_signal.sosfilt(high, x2) + noise
    res = miso_coherence([x1, x2], y, fs, nperseg=2048)

    f = res.frequencies
    band = (f >= 20.0) & (f <= 4000.0)
    fb = f[band]

    _fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(10, 7.4), sharex=True)

    def db(v: np.ndarray) -> np.ndarray:
        with np.errstate(divide="ignore"):
            return 10.0 * np.log10(v)

    ax_top.semilogx(fb, db(res.output_psd[band]), color="gray", linewidth=1.4,
                    label="Measured output")
    floor = db(res.output_psd[band]).min() - 5.0
    for i, color in ((0, COLOR_PRIMARY), (1, COLOR_TERTIARY)):
        level = db(res.coherent_output_spectra[i][band])
        # Both contributions rise from the same floor, so they stay
        # translucent and only their opacity follows the page.
        ax_top.fill_between(fb, floor, level, color=color, lw=0.0,
                            alpha=theme_fill_alpha(color, ax_top))
        ax_top.semilogx(fb, level, color=color, linewidth=1.4,
                        label=f"Input {i + 1} contribution")
    ax_top.semilogx(fb, db(res.noise_psd[band]), color=COLOR_SECONDARY,
                    linestyle="--", linewidth=1.1, label="Residual noise")
    ax_top.set_ylim(floor, db(res.output_psd[band]).max() + 3.0)
    ax_top.set_ylabel("Coherent output [dB re 1/Hz]")
    ax_top.set_title(
        "Multiple-Input Coherence: Which Source Dominates Each Band "
        "(Bendat & Piersol Ch. 7)", fontweight="bold", pad=12)
    ax_top.grid(which="both", color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax_top.set_axisbelow(True)
    ax_top.legend(loc="lower center", fontsize=8.5, ncol=2)

    ax_bot.semilogx(fb, res.ordinary_coherence[1][band], color=COLOR_TERTIARY,
                    linestyle=":", linewidth=1.6,
                    label=r"Input 2 ordinary $\gamma^2_{2y}$ (inflated by x1)")
    ax_bot.semilogx(fb, res.partial_coherence[1][band], color=COLOR_TERTIARY,
                    linewidth=1.8,
                    label=r"Input 2 partial $\gamma^2_{2y\cdot 1}$ (x1 removed)")
    ax_bot.semilogx(fb, res.multiple_coherence[band], color=COLOR_FG,
                    linewidth=1.4, alpha=0.8,
                    label=r"Multiple $\gamma^2_{y:x}$")
    ax_bot.set_ylim(0.0, 1.05)
    ax_bot.set_xlim(20.0, 4000.0)
    ax_bot.set_xlabel("Frequency [Hz]")
    ax_bot.set_ylabel("Coherence")
    ax_bot.grid(which="both", color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax_bot.set_axisbelow(True)
    ax_bot.legend(loc="center right", fontsize=8.5)
    ax_bot.text(0.015, 0.05,
                "conditioning removes the shared x1 component:\n"
                "the low-band ordinary coherence of x2 collapses",
                transform=ax_bot.transAxes, va="bottom", ha="left",
                fontsize=8.5, color=COLOR_FG)
    for ax in (ax_top, ax_bot):
        format_frequency_axis(ax, 20.0, 4000.0)
    plt.tight_layout()
    save_figure(output_dir, "miso_coherence.svg")
    plt.close()


def generate_cross_spectral_density_delay(output_dir: str) -> None:
    """Cross-spectral density of a delay path: magnitude and linear phase."""
    print("Generating cross_spectral_density...")
    from phonometry import cross_spectral_density, noise_signal

    fs = 8000.0
    tau = 0.002                                   # 2 ms = 16 samples
    delay = int(tau * fs)
    x = noise_signal(fs, 8.0, seed=8)
    noise = noise_signal(fs, 8.0, rms=0.3, seed=9)
    y = 0.9 * np.concatenate([np.zeros(delay), x[:-delay]]) + noise

    res = cross_spectral_density(x, y, fs)
    tiny = np.finfo(np.float64).tiny
    band = (res.frequencies >= 20.0) & (res.frequencies <= 3500.0)
    freqs = res.frequencies[band]

    _fig, (ax_m, ax_p) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    ax_m.semilogx(freqs, 10.0 * np.log10(np.maximum(res.magnitude[band], tiny)),
                  color=COLOR_PRIMARY, linewidth=1.1,
                  label=r"$|\hat{G}_{xy}|$ (Welch estimate)")
    ax_m.set_ylabel("Magnitude [dB]")
    ax_m.legend(loc="lower left", fontsize=9)

    ax_p.semilogx(freqs, res.phase[band], color=COLOR_PRIMARY, linewidth=1.1,
                  label="Unwrapped phase")
    ax_p.fill_between(freqs, res.phase[band] - res.phase_std[band],
                      res.phase[band] + res.phase_std[band],
                      color=COLOR_PRIMARY, alpha=0.25,
                      label=r"$\pm 1$ s.d. band (Eq. 9.52)")
    ax_p.semilogx(freqs, -2.0 * np.pi * freqs * tau, color=COLOR_SECONDARY,
                  linestyle="--", linewidth=1.4,
                  label=r"slope $-2\pi f\tau$ ($\tau$ = 2 ms)")
    ax_p.set_ylabel("Phase [rad]")
    ax_p.set_xlabel(LABEL_FREQ_HZ)
    ax_p.legend(loc="lower left", fontsize=9)

    for ax in (ax_m, ax_p):
        ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5, which="both")
        ax.set_axisbelow(True)
        format_frequency_axis(ax, 20.0, 3500.0)
    ax_m.set_title("Cross-Spectral Density of a 2 ms Delay Path",
                   fontweight="bold", pad=12)
    plt.tight_layout()
    save_figure(output_dir, "cross_spectral_density.svg")
    plt.close()


def generate_coherent_output_snr(output_dir: str) -> None:
    """Coherent output spectrum split and the spectral SNR."""
    print("Generating coherent_output_snr...")
    from phonometry import coherent_output_spectrum, noise_signal

    fs = 48000.0
    x = noise_signal(fs, 8.0, color="white", seed=1)
    noise = noise_signal(fs, 8.0, color="white", rms=0.5, seed=2)
    y = 0.8 * x + noise                      # SNR = 0.64/0.25 per band

    res = coherent_output_spectrum(x, y, fs, nperseg=2048)
    tiny = np.finfo(np.float64).tiny
    band = np.flatnonzero(
        (res.frequencies >= 20.0) & (res.frequencies <= 20000.0)
    )
    # Thin the flat spectra to ~1000 log-spaced bins: on a log axis the
    # linear Welch grid bunches thousands of vertices at the right edge
    # without adding visual information (and bloats the SVG).
    band = band[np.unique(np.geomspace(1, band.size, 1000).astype(int) - 1)]
    freqs = res.frequencies[band]

    _fig, (ax_g, ax_s) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for values, colour, style, label in (
        (res.output_psd, COLOR_PRIMARY, "-",
         r"$\hat{G}_{yy}$ (measured output)"),
        (res.coherent_psd, COLOR_TERTIARY, "--",
         r"$\hat{G}_{vv} = \gamma^2\hat{G}_{yy}$ (coherent part)"),
        (res.noise_psd, COLOR_SECONDARY, ":",
         r"$\hat{G}_{nn}$ (uncorrelated noise)"),
    ):
        ax_g.semilogx(freqs,
                      10.0 * np.log10(np.maximum(values[band], tiny)),
                      color=colour, linestyle=style, linewidth=1.1,
                      label=label)
    ax_g.set_ylabel("Spectral density [dB re 1/Hz]")
    ax_g.legend(loc="lower left", fontsize=9)

    ax_s.semilogx(freqs, res.snr_db[band], color=COLOR_PRIMARY,
                  linewidth=1.1, label="Spectral SNR [dB]")
    ax_s.axhline(10.0 * np.log10(0.64 / 0.25), color=COLOR_SECONDARY,
                 linestyle="--", linewidth=1.4,
                 label=r"closed form $10\log_{10}(2.56)$ = 4.1 dB")
    ax_s.set_ylabel("Spectral SNR [dB]")
    ax_s.set_xlabel(LABEL_FREQ_HZ)
    ax_s.legend(loc="lower left", fontsize=9)

    for ax in (ax_g, ax_s):
        ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5, which="both")
        ax.set_axisbelow(True)
        format_frequency_axis(ax, 20.0, 20000.0)
    ax_g.set_title(
        "Coherent Output Spectrum and Spectral SNR (Bendat & Piersol 9.2.2)",
        fontweight="bold", pad=12)
    plt.tight_layout()
    save_figure(output_dir, "coherent_output_snr.svg")
    plt.close()
