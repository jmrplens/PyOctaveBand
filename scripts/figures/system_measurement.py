#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Figures for the test signal and system measurement guides: the stimulus.

What a measurement puts into a system and what it takes back out: the gated
tone bursts, the shaped sweep and the Golay pair used as excitation, the
polyphase resampling that carries such a record between sampling rates, and
the regularized inversion of the response once it has been recovered.
Everything here is embedded by a page under ``signals/spectra/``.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal as scipy_signal

from phonometry._plot.common import format_frequency_axis, theme_fill

from .theme import (
    COLOR_FG,
    COLOR_GRID,
    COLOR_PANEL,
    COLOR_PRIMARY,
    COLOR_SECONDARY,
    COLOR_TERTIARY,
    LABEL_FREQ_HZ,
    save_figure,
)


def generate_tone_burst_train(output_dir: str) -> None:
    """IEC 60268-1 tone bursts: one gated burst and the repetitive train."""
    print("Generating tone_burst_train...")
    from phonometry import tone_burst

    fs = 48000.0
    single = tone_burst(fs, 5000.0, 25, pre_silence=0.001, post_silence=0.001)
    train = tone_burst(fs, 5000.0, 25, repetitions=4, repetition_rate=10.0)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6.4))
    t_ms = 1e3 * np.arange(single.signal.size) / single.fs
    axes[0].plot(t_ms, single.signal, color=COLOR_PRIMARY, linewidth=0.9)
    axes[0].plot(t_ms, single.envelope, color=COLOR_SECONDARY, linewidth=1.6,
                 linestyle="--", label="Gating envelope")
    axes[0].plot(t_ms, -single.envelope, color=COLOR_SECONDARY, linewidth=1.6,
                 linestyle="--")
    axes[0].set_title("Single 5 ms burst of 5 kHz tone (25 full periods)",
                      )
    axes[0].set_xlabel("Time [ms]")
    axes[0].legend(loc="upper right", fontsize=9)

    t_s = np.arange(train.signal.size) / train.fs
    axes[1].plot(t_s, train.signal, color=COLOR_PRIMARY, linewidth=0.5)
    axes[1].plot(t_s, train.envelope, color=COLOR_SECONDARY, linewidth=1.6,
                 linestyle="--", label="Gating envelope")
    axes[1].plot(t_s, -train.envelope, color=COLOR_SECONDARY, linewidth=1.6,
                 linestyle="--")
    axes[1].set_title("Repetitive train: 10 bursts per second (duty cycle 5 %)",
                      )
    axes[1].set_xlabel("Time [s]")
    axes[1].legend(loc="upper right", fontsize=9)

    for ax in axes:
        ax.set_ylabel("Amplitude")
        ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5)
        ax.set_axisbelow(True)
    fig.suptitle("Tone-Burst Test Signal (IEC 60268-1)")
    plt.tight_layout()
    save_figure(output_dir, "tone_burst_train.svg")
    plt.close()


def generate_regularized_inversion(output_dir: str) -> None:
    """Kirkeby inversion of a loudspeaker-like band-pass response."""
    print("Generating regularized_inversion...")
    from scipy import signal as sp_signal

    from phonometry import regularized_inverse_filter

    fs = 48000.0
    b, bb = sp_signal.butter(2, [100.0, 8000.0], btype="bandpass", fs=fs)
    imp = np.zeros(2048)
    imp[0] = 1.0
    h = sp_signal.lfilter(b, bb, imp)

    res = regularized_inverse_filter(h, fs, f_range=(200.0, 4000.0))

    freqs = res.frequencies
    pos = freqs > 0.0
    tiny = np.finfo(np.float64).tiny
    h_mag = np.abs(res.response_spectrum)
    peak = float(np.max(h_mag))
    inv_mag = np.abs(res.spectrum)
    eq_mag = h_mag * inv_mag

    _fig, ax = plt.subplots(figsize=(10, 6))
    ax.semilogx(freqs[pos],
                20.0 * np.log10(np.maximum(h_mag[pos], tiny) / peak),
                color=COLOR_PRIMARY, linewidth=1.4,
                label="Measured response $|H|$")
    ax.semilogx(freqs[pos],
                20.0 * np.log10(np.maximum(inv_mag[pos] * peak, tiny)),
                color=COLOR_SECONDARY, linewidth=1.4,
                label=r"Inverse filter $|H_{\mathrm{inv}}|$")
    ax.semilogx(freqs[pos],
                20.0 * np.log10(np.maximum(eq_mag[pos], tiny)),
                color=COLOR_TERTIARY, linewidth=1.8,
                label=r"Equalized $|H \cdot H_{\mathrm{inv}}|$")
    ax.axvspan(200.0, 4000.0, color=theme_fill(COLOR_PRIMARY, ax), zorder=0,
               label="Equalized band (200 Hz - 4 kHz)")
    ax.set_xlim(20.0, fs / 2.0)
    ax.set_ylim(-50.0, 15.0)
    format_frequency_axis(ax, 20.0, fs / 2.0)
    ax.set_xlabel(LABEL_FREQ_HZ)
    ax.set_ylabel("Magnitude [dB]")
    ax.set_title("Regularized Spectral Inversion (Kirkeby Frequency-"
                 "Dependent Regularization)", pad=12)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5, which="both")
    ax.set_axisbelow(True)
    ax.legend(loc="lower center", fontsize=9)
    ax.text(0.985, 0.97,
            "unity in-band; outside, the frequency-dependent\n"
            "regularization caps the gain instead of boosting noise",
            transform=ax.transAxes, va="top", ha="right", fontsize=8.5,
            color=COLOR_FG)
    plt.tight_layout()
    save_figure(output_dir, "regularized_inversion.svg")
    plt.close()


def generate_shaped_sweep(output_dir: str) -> None:
    """A pink shaped sweep: waveform and spectrum against the target."""
    print("Generating shaped_sweep...")
    from scipy import signal as sp_signal

    from phonometry import shaped_sweep_signal

    fs = 48000
    res = shaped_sweep_signal(fs, 50.0, 5000.0, 2.0, target="pink")
    x = np.asarray(res)
    t = np.arange(x.size) / fs

    nperseg = 8192
    freqs_w, psd = sp_signal.welch(x, fs=fs, nperseg=nperseg,
                                   noverlap=3 * nperseg // 4)
    tiny = np.finfo(np.float64).tiny
    band_w = (freqs_w >= 50.0) & (freqs_w <= 5000.0)
    welch_db = 10.0 * np.log10(np.maximum(psd, tiny))
    welch_db -= float(np.max(welch_db[band_w]))
    grid = res.frequencies
    band_g = (grid >= 50.0) & (grid <= 5000.0)
    target_db = 20.0 * np.log10(np.maximum(res.magnitude, tiny))
    target_db -= float(np.max(target_db[band_g]))

    _fig, axes = plt.subplots(2, 1, figsize=(10, 7))
    axes[0].plot(t, x, color=COLOR_PRIMARY, linewidth=0.5)
    axes[0].set_xlabel("Time [s]")
    axes[0].set_ylabel("Amplitude")
    axes[0].set_xlim(0.0, float(t[-1]))
    axes[0].set_title("Shaped Sweep with an Arbitrary Target Spectrum "
                      "(Group-Delay Synthesis)", pad=12)
    axes[0].grid(color=COLOR_GRID, linestyle="--", alpha=0.5)
    axes[0].set_axisbelow(True)
    axes[0].text(0.985, 0.95,
                 "nearly constant envelope: the energy shaping lives\n"
                 "in the dwell time, not in the amplitude",
                 transform=axes[0].transAxes, va="top", ha="right",
                 fontsize=8.5, color=COLOR_FG)

    posw = freqs_w > 0.0
    axes[1].semilogx(freqs_w[posw], welch_db[posw], color=COLOR_PRIMARY,
                     linewidth=1.3, label="Welch spectrum of the sweep")
    posg = grid > 0.0
    axes[1].semilogx(grid[posg], target_db[posg], color=COLOR_SECONDARY,
                     linewidth=1.5, linestyle="--",
                     label="Pink target (−3 dB per octave)")
    axes[1].axvspan(50.0, 5000.0, color=theme_fill(COLOR_PRIMARY, axes[1]),
                    zorder=0, label="Sweep band (50 Hz - 5 kHz)")
    axes[1].set_xlim(20.0, 20000.0)
    axes[1].set_ylim(-60.0, 8.0)
    format_frequency_axis(axes[1], 20.0, 20000.0)
    axes[1].set_xlabel(LABEL_FREQ_HZ)
    axes[1].set_ylabel("Level re in-band max [dB]")
    axes[1].grid(color=COLOR_GRID, linestyle="--", alpha=0.5, which="both")
    axes[1].set_axisbelow(True)
    axes[1].legend(loc="lower left", fontsize=9)
    plt.tight_layout()
    save_figure(output_dir, "shaped_sweep.svg")
    plt.close()


def generate_resampling_antialias(output_dir: str) -> None:
    """Polyphase resampling: the delivered anti-alias filter vs its spec."""
    print("Generating resampling_antialias...")
    from phonometry import noise_signal, resample_signal

    x = noise_signal(44100, 5.0, color="pink", seed=1)
    res = resample_signal(x, 44100, fs_new=48000)      # 120 dB alias rejection

    fs_up = res.original_fs * res.up
    freqs, h = scipy_signal.freqz(res.filter_taps, worN=1 << 18, fs=fs_up)
    tiny = np.finfo(np.float64).tiny
    mag_db = 20.0 * np.log10(np.maximum(np.abs(h), tiny))
    f_lo, f_hi = res.stopband_edge_hz / 8.0, 4.0 * res.stopband_edge_hz
    view = (freqs > 0.0) & (freqs <= f_hi)

    _fig, ax = plt.subplots(figsize=(10, 6))
    ax.semilogx(freqs[view], mag_db[view], color=COLOR_PRIMARY, linewidth=1.2,
                label="Anti-alias filter $|H(f)|$")
    ax.axvline(res.passband_edge_hz, color=COLOR_TERTIARY, linestyle="--",
               linewidth=1.4, label="Passband edge")
    ax.axvline(res.stopband_edge_hz, color=COLOR_SECONDARY, linestyle="--",
               linewidth=1.4, label="Stopband edge (alias fold)")
    ax.axhline(-res.stopband_attenuation_db, color=COLOR_FG, linestyle=":",
               linewidth=1.2, alpha=0.7, label="Design attenuation −120 dB")
    ax.axvspan(res.stopband_edge_hz, f_hi, color=theme_fill(COLOR_SECONDARY, ax),
               zorder=0,
               label="Rejected band (would fold back as aliases)")
    ax.set_xlim(f_lo, f_hi)
    ax.set_ylim(-170.0, 10.0)
    format_frequency_axis(ax, f_lo, f_hi)
    ax.set_xlabel(LABEL_FREQ_HZ)
    ax.set_ylabel("Magnitude [dB]")
    ax.set_title("Polyphase Resampling 44.1 kHz → 48 kHz: "
                 "the Delivered Anti-Alias Filter", pad=12)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5, which="both")
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", fontsize=9)
    plt.tight_layout()
    save_figure(output_dir, "resampling_antialias.svg")
    plt.close()


def generate_golay_ir(output_dir: str) -> None:
    """Golay-pair impulse response: exact complementary recovery."""
    print("Generating golay_ir...")
    from phonometry import golay_impulse_response, golay_pair

    fs = 48000
    pair = golay_pair(14)                        # two 16384-sample codes
    b, a = scipy_signal.butter(2, [200.0, 2000.0], btype="bandpass", fs=fs)
    length = pair[0].size
    rec_a = scipy_signal.lfilter(b, a, np.tile(pair[0], 3))[2 * length:]
    rec_b = scipy_signal.lfilter(b, a, np.tile(pair[1], 3))[2 * length:]

    ir = np.asarray(golay_impulse_response(rec_a, rec_b, pair, fs=fs))
    impulse = np.zeros(length)
    impulse[0] = 1.0
    true_ir = scipy_signal.lfilter(b, a, impulse)
    err = float(np.max(np.abs(ir - true_ir)))
    mantissa, exponent = f"{err:.1e}".split("e")

    t_ms = 1e3 * np.arange(length) / fs
    view = t_ms <= 6.0
    _fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(t_ms[view], ir[view], color=COLOR_PRIMARY, linewidth=1.6,
            label="Recovered IR (golay_impulse_response)")
    ax.plot(t_ms[view], true_ir[view], color=COLOR_SECONDARY, linewidth=1.2,
            linestyle="--", label="True system response")
    ax.text(0.985, 0.05,
            rf"$\max\,|\mathrm{{recovered}} - \mathrm{{true}}|"
            rf" = {mantissa}\times10^{{{int(exponent)}}}$"
            "\nnoise-free closed-form identity",
            transform=ax.transAxes, va="bottom", ha="right", fontsize=9,
            color=COLOR_FG,
            bbox={"boxstyle": "round,pad=0.4", "facecolor": COLOR_PANEL,
                  "edgecolor": COLOR_GRID})
    ax.set_xlabel("Time [ms]")
    ax.set_ylabel("Amplitude")
    ax.set_title("Golay-Pair Impulse Response: Exact Complementary Recovery",
                 pad=12)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    save_figure(output_dir, "golay_ir.svg")
    plt.close()
