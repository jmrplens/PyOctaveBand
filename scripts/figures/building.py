#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Figures for the building-acoustics guides: insulation, flanking, ratings.

Airborne and impact sound insulation of a building element, measured and
predicted: laboratory and field quantities, the single-number ratings, the
flanking paths of EN 12354 and the junction and covering terms that feed them.
Everything here is embedded by a page under ``buildings/``.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal as scipy_signal

from phonometry._plot.common import format_frequency_axis, theme_fill

from .i18n import _LANG
from .theme import (
    _THIRD_OCTAVE_16,
    COLOR_FG,
    COLOR_GRID,
    COLOR_MUTED,
    COLOR_PANEL,
    COLOR_PRIMARY,
    COLOR_SECONDARY,
    COLOR_TERTIARY,
    LABEL_FREQ_HZ,
    _band_index_axis,
    save_figure,
)


def generate_schroeder_decay(output_dir: str) -> None:
    """Schroeder backward integration with T20/T30/EDT regressions (ISO 3382)."""
    print("Generating schroeder_decay.png...")
    from phonometry import decay_curve, room_parameters
    from phonometry.room.acoustics import (
        _EDT_RANGE,
        _T20_RANGE,
        _T30_RANGE,
        _onset_index,
    )

    fs = 48000
    reverb_t = 1.2  # target reverberation time (s)
    duration = 2.5
    rng = np.random.default_rng(2026)
    t = np.arange(int(duration * fs)) / fs
    # Exponential decay (T = 1.2 s) excited by white noise + a realistic
    # background-noise floor (~-45 dB re the peak envelope).
    ir = rng.standard_normal(t.size) * np.exp(-6.9077 * t / reverb_t)
    ir = ir + rng.standard_normal(t.size) * 10.0 ** (-45.0 / 20.0)

    # Library outputs: the annotated numbers are exactly these.
    time, level = decay_curve(ir, fs)
    res = room_parameters(ir, fs, limits=None)
    edt, t20, t30 = float(res.edt[0]), float(res.t20[0]), float(res.t30[0])

    # Raw squared-IR level trace (onset-trimmed, normalized to its peak):
    # the noisy line the backward integration smooths into the decay curve.
    p2 = ir.astype(np.float64) ** 2
    p2 = p2[_onset_index(p2):]
    t_raw = np.arange(p2.size) / fs
    raw_db = 10.0 * np.log10(np.maximum(p2, p2.max() * 1e-12) / p2.max())

    def fit_line(decay_range: tuple[float, float]) -> tuple[float, float]:
        """Least-squares (slope, intercept) over an evaluation range,
        replicating room_acoustics._fit_decay_time so the drawn line has
        slope -60/T with the annotated T."""
        mask = (level <= -decay_range[0]) & (level >= -decay_range[1])
        slope, intercept = np.polyfit(time[mask], level[mask], 1)
        return float(slope), float(intercept)

    _, ax = plt.subplots(figsize=(10, 6.5))
    ax.plot(t_raw, raw_db, color="gray", alpha=0.28, linewidth=0.6, zorder=0,
            label="Raw squared IR level")
    ax.plot(time, level, color=COLOR_PRIMARY, linewidth=2.4, zorder=5,
            label="Schroeder decay curve")

    lines = [
        (_EDT_RANGE, "#9467bd", "-", "EDT fit (0 to −10 dB)", (0.0, -13.0)),
        (_T20_RANGE, COLOR_SECONDARY, "--", "T20 fit (−5 to −25 dB)", (0.0, -60.0)),
        (_T30_RANGE, COLOR_TERTIARY, "-.", "T30 fit (−5 to −35 dB)", (0.0, -60.0)),
    ]
    for decay_range, color, style, label, (lo, hi) in lines:
        slope, intercept = fit_line(decay_range)
        t_lo, t_hi = (lo - intercept) / slope, (hi - intercept) / slope
        ax.plot([t_lo, t_hi], [lo, hi], color=color, linestyle=style,
                linewidth=1.7, zorder=4, label=label)

    # Evaluation levels -5 / -25 / -35 dB and the decay-curve crossings.
    for target in (-5.0, -25.0, -35.0):
        ax.axhline(target, color=COLOR_FG, linestyle=":", alpha=0.35, linewidth=1)
        t_cross = float(np.interp(target, level[::-1], time[::-1]))
        ax.plot(t_cross, target, "o", color=COLOR_FG, markersize=5, zorder=6)
        # Place level labels clear of the upper-right legend.
        ax.text(1.40, target + 0.8, f"{target:.0f} dB", ha="left",
                va="bottom", fontsize=8, color=COLOR_FG, alpha=0.85)

    ax.text(0.12, -7.0, "EDT slope", fontsize=8, color="#9467bd", rotation=0)
    facecolor = plt.rcParams["axes.facecolor"]
    ax.text(0.04, 0.06,
            f"EDT = {edt:.2f} s\nT20 = {t20:.2f} s\nT30 = {t30:.2f} s",
            transform=ax.transAxes, va="bottom", ha="left", fontsize=11,
            bbox={"boxstyle": "round", "facecolor": facecolor,
                  "edgecolor": COLOR_FG, "alpha": 0.85})

    ax.set_title("Schroeder Integration and Reverberation Time (ISO 3382)",
                 fontweight="bold", pad=12)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Level re steady state [dB]")
    ax.set_xlim(0, duration)
    ax.set_ylim(-65, 3)
    ax.grid(which="major", color=COLOR_GRID, linestyle="-", alpha=0.5)
    ax.legend(loc="upper right", fontsize=9)
    save_figure(output_dir, "schroeder_decay.png")
    plt.close()


def generate_excitation_signals(output_dir: str) -> None:
    """ISO 18233 excitations: ESS waveform + spectrogram and MLS + spectrum."""
    print("Generating excitation_signals.png...")
    from phonometry import mls_signal, sweep_signal

    fs = 48000
    f1, f2, secs = 50.0, 20000.0, 1.0
    sweep = sweep_signal(fs, f1, f2, secs)
    t = np.arange(sweep.size) / fs
    mls = mls_signal(12)  # length 2**12 - 1 = 4095

    fig, axes = plt.subplots(2, 2, figsize=(12, 7.2))
    (ax_sw, ax_sp), (ax_ml, ax_ms) = axes

    # Exponential sine sweep: time-domain waveform.
    ax_sw.plot(t, sweep, color=COLOR_PRIMARY, linewidth=0.5)
    ax_sw.set_title("Exponential sine sweep — waveform", fontweight="bold")
    ax_sw.set_xlabel("Time [s]")
    ax_sw.set_ylabel("Amplitude")
    ax_sw.set_xlim(0.0, secs)
    ax_sw.set_ylim(-1.2, 1.2)
    ax_sw.grid(color=COLOR_GRID, linestyle="--", alpha=0.5)

    # Sweep spectrogram: the exponential frequency rise.
    ax_sp.specgram(sweep, NFFT=1024, Fs=fs, noverlap=512, cmap="magma")
    ax_sp.set_title("Sweep spectrogram (exponential rise)", fontweight="bold")
    ax_sp.set_xlabel("Time [s]")
    ax_sp.set_ylabel("Frequency [Hz]")
    ax_sp.set_ylim(0.0, fs / 2)

    # MLS: first samples of the bipolar sequence.
    show = 100
    ax_ml.step(np.arange(show), mls[:show], where="mid", color=COLOR_PRIMARY,
               linewidth=1.2)
    ax_ml.set_title(f"MLS — first {show} of {mls.size} samples", fontweight="bold")
    ax_ml.set_xlabel("Sample")
    ax_ml.set_ylabel("Amplitude")
    ax_ml.set_ylim(-1.4, 1.4)
    ax_ml.grid(color=COLOR_GRID, linestyle="--", alpha=0.5)

    # MLS magnitude spectrum: essentially flat (white excitation).
    spec = np.abs(np.fft.rfft(mls))
    freqs = np.fft.rfftfreq(mls.size, d=1.0 / fs)
    ax_ms.semilogx(freqs[1:], 20.0 * np.log10(spec[1:] / np.median(spec[1:])),
                   color=COLOR_SECONDARY, linewidth=0.7)
    ax_ms.set_title("MLS magnitude spectrum (flat)", fontweight="bold")
    ax_ms.set_xlabel("Frequency [Hz]")
    ax_ms.set_ylabel("Magnitude [dB]")
    ax_ms.set_xlim(20.0, fs / 2)
    format_frequency_axis(ax_ms, 20.0, fs / 2)
    ax_ms.set_ylim(-12.0, 12.0)
    ax_ms.grid(which="both", color=COLOR_GRID, linestyle="--", alpha=0.5)

    fig.suptitle("ISO 18233 excitation signals", fontweight="bold")
    plt.tight_layout()
    save_figure(output_dir, "excitation_signals.png")
    plt.close()


def generate_impulse_response(output_dir: str) -> None:
    """ISO 18233 recovered IR: waveform + log-magnitude / Schroeder decay."""
    print("Generating impulse_response.png...")
    from scipy.signal import fftconvolve

    from phonometry import impulse_response, sweep_signal

    fs = 48000
    sweep = sweep_signal(fs, 20.0, 20000.0, 1.5)

    # A synthetic room: direct sound, two early reflections and an
    # exponentially decaying diffuse tail (T ~ 0.6 s) plus a low noise floor.
    rng = np.random.default_rng(2026)
    n = int(0.7 * fs)
    system = np.zeros(n)
    system[80] = 1.0                       # direct sound
    system[1400] = 0.5                     # early reflection
    system[3100] = 0.32                    # second reflection
    tail_t = np.arange(n) / fs
    system += rng.standard_normal(n) * np.exp(-6.9077 * tail_t / 0.6) * 0.08
    system += rng.standard_normal(n) * 10.0 ** (-60.0 / 20.0)

    recorded = fftconvolve(sweep, system)
    ir = impulse_response(recorded, sweep, fs, length=n)

    h = np.asarray(ir, dtype=np.float64)
    time = np.arange(h.size) / fs
    peak = float(np.max(np.abs(h)))
    tiny = np.finfo(np.float64).tiny
    env_db = 20.0 * np.log10(np.maximum(np.abs(h), tiny) / peak)
    energy = np.cumsum(h[::-1] ** 2)[::-1]
    edc_db = 10.0 * np.log10(np.maximum(energy, tiny) / energy[0])

    _fig, (ax_w, ax_d) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    ax_w.plot(time, h / peak, color=COLOR_PRIMARY, linewidth=0.7)
    ax_w.set_title("Recovered room impulse response (ISO 18233)",
                   fontweight="bold", pad=10)
    ax_w.set_ylabel("Amplitude (norm.)")
    ax_w.set_ylim(-1.1, 1.1)
    ax_w.grid(color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax_w.annotate("direct sound", xy=(80 / fs, 1.0), xytext=(0.06, 0.86),
                  textcoords="axes fraction", fontsize=9, color=COLOR_FG,
                  arrowprops={"arrowstyle": "->", "color": COLOR_FG, "alpha": 0.7})
    ax_w.annotate("reflections", xy=(1400 / fs, 0.5), xytext=(0.20, 0.62),
                  textcoords="axes fraction", fontsize=9, color=COLOR_FG,
                  arrowprops={"arrowstyle": "->", "color": COLOR_FG, "alpha": 0.7})

    ax_d.plot(time, env_db, color="#9ecae1", linewidth=0.7,
              label="Log-magnitude envelope")
    ax_d.plot(time, edc_db, color=COLOR_SECONDARY, linewidth=1.9,
              label="Schroeder decay (EDC)")
    ax_d.set_xlabel("Time [s]")
    ax_d.set_ylabel("Level re peak [dB]")
    ax_d.set_xlim(0.0, n / fs)
    ax_d.set_ylim(-80.0, 5.0)
    ax_d.grid(color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax_d.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    save_figure(output_dir, "impulse_response.png")
    plt.close()


def generate_insulation_rating(output_dir: str) -> None:
    """ISO 717-1 weighted rating: measured R', shifted reference, deviations."""
    print("Generating insulation_rating.png...")
    from phonometry.building.measurement.insulation import (
        _INDEX_500_THIRD,
        _REF_THIRD_OCTAVE,
        weighted_rating,
    )

    # ISO 717-1 Annex C worked example (Table C.1), 100 Hz .. 3150 Hz.
    freqs = np.array([100, 125, 160, 200, 250, 315, 400, 500, 630, 800,
                      1000, 1250, 1600, 2000, 2500, 3150], dtype=float)
    measured = np.array([20.4, 16.3, 17.7, 22.6, 22.4, 22.7, 24.8, 26.6,
                         28.0, 30.5, 31.8, 32.5, 33.4, 33.0, 31.0, 25.5])

    result = weighted_rating(measured)
    reference = np.asarray(_REF_THIRD_OCTAVE, dtype=float)
    shift = result.rating - _REF_THIRD_OCTAVE[_INDEX_500_THIRD]
    shifted = reference + shift  # shifted reference read at 500 Hz == Rw

    _, ax = plt.subplots(figsize=(10, 6.5))
    ax.fill_between(freqs, measured, shifted, where=(measured < shifted).tolist(),
                    interpolate=True, color=COLOR_SECONDARY, alpha=0.25,
                    zorder=1, label="Unfavourable deviations")
    ax.semilogx(freqs, shifted, marker="s", color=COLOR_FG, linewidth=1.6,
                linestyle="--", markersize=4, zorder=3,
                label="Shifted reference curve (ISO 717-1)")
    ax.semilogx(freqs, measured, marker="o", color=COLOR_PRIMARY, linewidth=1.8,
                markersize=5, markerfacecolor="white", markeredgewidth=1.4,
                zorder=4, label="Measured R' (third octave)")

    # Rw is the shifted reference read at 500 Hz.
    ax.axvline(500, color=COLOR_FG, linestyle=":", alpha=0.4)
    ax.plot(500, result.rating, "D", color=COLOR_SECONDARY, markersize=9, zorder=6)
    ax.annotate(f"Rw = {result.rating} dB", xy=(500, result.rating),
                xytext=(560, result.rating - 9), fontsize=12, fontweight="bold",
                arrowprops={"arrowstyle": "->", "lw": 1.0})

    for dy, text in (
        (0.97, f"Reference curve shifted by {shift} dB"),
        (0.90, (f"Sum of unfavourable deviations = {result.unfavourable_sum:.1f}"
               f" dB  (limit 32.0 dB)")),
        (0.83, (f"Rw (C ; Ctr) = {result.rating} "
               f"({result.c:+d} ; {result.ctr:+d}) dB")),
    ):
        ax.text(0.03, dy, text, transform=ax.transAxes, va="top", ha="left",
                fontsize=9.5, color=COLOR_FG)

    ax.set_title("ISO 717-1 Weighted Sound Reduction Index (Annex C example)",
                 fontweight="bold", pad=12)
    ax.set_xlabel(LABEL_FREQ_HZ)
    ax.set_ylabel("Apparent sound reduction index R' [dB]")
    ax.set_xscale("log")
    ax.set_xlim(90, 3600)
    ax.set_ylim(8, 44)
    from matplotlib.ticker import NullFormatter
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xticks(freqs)
    ax.set_xticklabels(
        ["100", "125", "160", "200", "250", "315", "400", "500", "630", "800",
         "1k", "1.25k", "1.6k", "2k", "2.5k", "3.15k"], fontsize=8)
    ax.grid(which="major", color=COLOR_GRID, linestyle="-", alpha=0.5)
    ax.legend(loc="lower right", fontsize=9)
    save_figure(output_dir, "insulation_rating.png")
    plt.close()


def generate_impact_rating(output_dir: str) -> None:
    """ISO 717-2 weighted impact rating: measured Ln, shifted reference, CI."""
    print("Generating impact_rating.png...")
    from phonometry.building.measurement.insulation import (
        _INDEX_500_THIRD,
        _REF_IMPACT_THIRD_OCTAVE,
        weighted_impact_rating,
    )

    # ISO 717-2 Annex C worked example (Table C.1): laboratory bare massive
    # floor, one-third octave 100 Hz .. 3150 Hz.
    freqs = np.array([100, 125, 160, 200, 250, 315, 400, 500, 630, 800,
                      1000, 1250, 1600, 2000, 2500, 3150], dtype=float)
    measured = np.array([62.1, 63.2, 63.5, 66.2, 68.5, 70.0, 71.7, 73.1,
                         73.8, 73.5, 73.8, 73.3, 73.1, 73.0, 72.4, 71.2])

    result = weighted_impact_rating(measured)
    reference = np.asarray(_REF_IMPACT_THIRD_OCTAVE, dtype=float)
    shift = result.rating - _REF_IMPACT_THIRD_OCTAVE[_INDEX_500_THIRD]
    shifted = reference + shift  # shifted reference read at 500 Hz == Ln,w

    _, ax = plt.subplots(figsize=(10, 6.5))
    # For impact sound an unfavourable deviation occurs where the MEASURED
    # curve lies ABOVE the reference (opposite sign to ISO 717-1 airborne).
    ax.fill_between(freqs, shifted, measured, where=(measured > shifted).tolist(),
                    interpolate=True, color=COLOR_SECONDARY, alpha=0.25,
                    zorder=1, label="Unfavourable deviations (measured above reference)")
    ax.semilogx(freqs, shifted, marker="s", color=COLOR_FG, linewidth=1.6,
                linestyle="--", markersize=4, zorder=3,
                label="Shifted reference curve (ISO 717-2)")
    ax.semilogx(freqs, measured, marker="o", color=COLOR_PRIMARY, linewidth=1.8,
                markersize=5, markerfacecolor="white", markeredgewidth=1.4,
                zorder=4, label="Measured Ln (third octave)")

    # Ln,w is the shifted reference read at 500 Hz.
    ax.axvline(500, color=COLOR_FG, linestyle=":", alpha=0.4)
    ax.plot(500, result.rating, "D", color=COLOR_SECONDARY, markersize=9, zorder=6)
    # The annotation sits in the clear gap between the rising measured curve
    # and the flat low-frequency reference plateau; the string is identical
    # in both languages, so the placement holds for every variant.
    ax.annotate(f"Ln,w = {result.rating} dB", xy=(500, result.rating),
                xytext=(135, result.rating - 4.2), fontsize=12,
                fontweight="bold",
                arrowprops={"arrowstyle": "->", "lw": 1.0})

    for dy, text in (
        (0.97, f"Reference curve shifted by {shift} dB"),
        (0.90, (f"Sum of unfavourable deviations = {result.unfavourable_sum:.1f}"
               f" dB  (limit 32.0 dB)")),
        (0.83, f"Ln,w = {result.rating} dB ; CI = {result.ci:+d} dB"),
    ):
        ax.text(0.03, dy, text, transform=ax.transAxes, va="top", ha="left",
                fontsize=9.5, color=COLOR_FG)

    ax.set_title("ISO 717-2 Weighted Normalized Impact Sound Level "
                 "(Annex C example)", fontweight="bold", pad=12)
    ax.set_xlabel(LABEL_FREQ_HZ)
    ax.set_ylabel("Normalized impact sound pressure level Ln [dB]")
    ax.set_xscale("log")
    ax.set_xlim(90, 3600)
    ax.set_ylim(55, 86)
    from matplotlib.ticker import NullFormatter
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xticks(freqs)
    ax.set_xticklabels(
        ["100", "125", "160", "200", "250", "315", "400", "500", "630", "800",
         "1k", "1.25k", "1.6k", "2k", "2.5k", "3.15k"], fontsize=8)
    ax.grid(which="major", color=COLOR_GRID, linestyle="-", alpha=0.5)
    ax.legend(loc="lower left", fontsize=9)
    save_figure(output_dir, "impact_rating.png")
    plt.close()


def generate_open_plan_decay(output_dir: str) -> None:
    """ISO 3382-3 spatial decay: speech SPL and STI vs source distance."""
    print("Generating open_plan_decay.png...")
    from phonometry import open_plan_metrics

    # The worked example from the room-acoustics guide (matches its numbers).
    r = np.array([2.0, 4.0, 6.0, 8.0, 12.0, 16.0])   # distances (m)
    lp = 65.0 - 7.0 * np.log2(r)                      # A-weighted speech level (dB)
    sti = 0.70 - 0.03 * r                             # STI per position
    m = open_plan_metrics(r, lp, sti)

    # Reconstruct the two regressions the metrics come from (2-16 m window).
    b_log = -m.d2s / np.log10(2.0)                    # slope vs log10(r/r0)
    a_lp = m.lp_as_4m - b_log * np.log10(4.0)
    d_sti, c_sti = np.polyfit(r, sti, 1)              # STI vs distance

    _fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.set_xscale("log")
    rr = np.logspace(np.log10(2.0), np.log10(16.0), 100)
    line_spl, = ax.plot(rr, a_lp + b_log * np.log10(rr), color=COLOR_PRIMARY,
                        linestyle="--", linewidth=1.8,
                        label=f"Spatial decay D2,S = {m.d2s:.1f} dB")
    pts_spl, = ax.plot(r, lp, "o", color=COLOR_PRIMARY, markersize=7,
                       markerfacecolor="white", markeredgewidth=1.6,
                       label="Measured Lp,A,S")
    ax.axvline(4.0, color=COLOR_FG, linestyle=":", alpha=0.35, linewidth=1)
    mark_4m, = ax.plot(4.0, m.lp_as_4m, "D", color=COLOR_SECONDARY, markersize=9,
                       zorder=6, label=f"Lp,A,S,4m = {m.lp_as_4m:.0f} dB")
    ax.annotate(f"Lp,A,S,4m = {m.lp_as_4m:.0f} dB", xy=(4.0, m.lp_as_4m),
                xytext=(4.7, m.lp_as_4m + 4.5), fontsize=10,
                arrowprops={"arrowstyle": "->", "lw": 1.0})

    ax.set_title("Open-Plan Spatial Decay of Speech (ISO 3382-3)",
                 fontweight="bold", pad=12)
    ax.set_xlabel("Distance from the talker r [m]")
    ax.set_ylabel("A-weighted SPL [dB]", color=COLOR_PRIMARY)
    ax.set_xlim(1.7, 18.0)
    ax.set_ylim(30, 62)
    from matplotlib.ticker import NullFormatter, ScalarFormatter
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xticks([2, 3, 4, 6, 8, 12, 16])
    ax.set_xticklabels(["2", "3", "4", "6", "8", "12", "16"])
    ax.grid(which="major", color=COLOR_GRID, linestyle="-", alpha=0.4)

    # Right axis: STI vs distance with the distraction / privacy crossings.
    ax2 = ax.twinx()
    rr2 = np.linspace(1.7, 18.0, 100)
    line_sti, = ax2.plot(rr2, c_sti + d_sti * rr2, color=COLOR_TERTIARY,
                         linewidth=1.7, label="STI vs distance")
    ax2.plot(r, sti, "s", color=COLOR_TERTIARY, markersize=5,
             markerfacecolor="white", markeredgewidth=1.3)
    for dist, level, name in [(m.rd, 0.50, "rD"), (m.rp, 0.20, "rP")]:
        ax2.axhline(level, color=COLOR_FG, linestyle=":", alpha=0.25, linewidth=1)
        ax2.plot(dist, level, "v", color=COLOR_SECONDARY, markersize=9, zorder=6)
        ax2.annotate(f"{name} = {dist:.1f} m", xy=(dist, level),
                     xytext=(dist * 0.62, level + 0.03), fontsize=9,
                     color=COLOR_SECONDARY,
                     arrowprops={"arrowstyle": "->", "lw": 0.9,
                                 "color": COLOR_SECONDARY})
    ax2.set_ylabel("STI", color=COLOR_TERTIARY)
    ax2.set_ylim(0.1, 0.75)

    handles = [pts_spl, line_spl, mark_4m, line_sti]
    ax.legend(handles, [str(h.get_label()) for h in handles],
              loc="upper right", fontsize=9)
    save_figure(output_dir, "open_plan_decay.png")
    plt.close()


def generate_rectangular_room_modes(output_dir: str) -> None:
    """Mode ladder and modal density of Long's 7 x 5 x 3 m example room.

    One concept: where the eigenfrequencies of a rectangular room sit, which
    family each belongs to, and where the Schroeder frequency ends the modal
    regime.
    """
    print("Generating rectangular_room_modes...")
    from phonometry import room_modes

    # Long, Architectural Acoustics 2nd ed., Table 8.1 (printed p. 325).
    result = room_modes(
        (7.0, 5.0, 3.0), max_frequency=200.0, speed_of_sound=344.0,
        reverberation_time=0.8,
    )
    result.plot(language=_LANG)
    plt.gcf().set_size_inches(10, 6.5)
    plt.tight_layout()
    save_figure(output_dir, "rectangular_room_modes.svg")
    plt.close()


def generate_restaurant_crowd_noise(output_dir: str) -> None:
    """Restaurant self-noise against occupancy for three absorption areas.

    One concept: the crowd generates its own background, and only absorption
    keeps it below the level at which cross-table conversation fails.
    """
    print("Generating restaurant_crowd_noise...")
    from phonometry import crowd_noise

    # Long, Chapter 17 (printed pp. 665-666): a hard room of about 20 metric
    # sabins, the same room with a partly absorptive ceiling, and with the
    # full alpha 0.9 ceiling of his 13.7 x 13.7 m example (+170 sabins).
    result = crowd_noise([20.0, 95.0, 190.0], distance=1.2)
    ax = result.plot(language=_LANG)
    ax.set_xlim(1.0, 20.0)
    plt.gcf().set_size_inches(10, 6)
    plt.tight_layout()
    save_figure(output_dir, "restaurant_crowd_noise.svg")
    plt.close()


def generate_image_source_reflectogram(output_dir: str) -> None:
    """Image-source reflectogram: the synthetic RIR as reflections by order."""
    print("Generating image_source_reflectogram...")
    from phonometry import image_source_rir

    dims = (7.0, 5.0, 3.0)
    res = image_source_rir(dims, (2.0, 1.6, 1.5), (5.2, 3.4, 1.7),
                           0.12, fs=48000, max_order=10)
    times = np.asarray(res.times) * 1e3  # ms
    amp = np.asarray(res.amplitudes)
    orders = np.asarray(res.orders)
    direct = float(np.max(np.abs(amp)))
    level = 20.0 * np.log10(np.maximum(np.abs(amp), 1e-30) / direct)
    dist = np.asarray(res.distances)
    d0 = float(dist[int(np.argmin(dist))])
    envelope = 20.0 * np.log10(np.maximum(d0 / dist, 1e-30))

    window = times <= 120.0
    fig, ax = plt.subplots(figsize=(10, 6))
    env_sort = np.argsort(times[window])
    ax.plot(times[window][env_sort], envelope[window][env_sort],
            color=COLOR_GRID, linestyle="--", linewidth=1.2,
            label=r"$1/r$ spreading envelope", zorder=1)
    is_direct = orders == 0
    refl = window & ~is_direct
    sc = ax.scatter(times[refl], level[refl], c=orders[refl], cmap="viridis",
                    s=18, alpha=0.85, zorder=3,
                    label="Reflections (image sources)")
    d = window & is_direct
    ax.vlines(times[d], -60.0, level[d], color=COLOR_PRIMARY, linewidth=2.0,
              zorder=4)
    ax.plot(times[d], level[d], "o", color=COLOR_PRIMARY, ms=9, zorder=5,
            label="Direct sound (order 0)")

    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("Reflection order")
    ax.set_xlabel("Arrival time [ms]")
    ax.set_ylabel("Reflection level re direct [dB]")
    ax.set_title("Image-Source Room Impulse Response: a 7x5x3 m room "
                 "(order <= 10)", fontweight="bold", pad=12)
    ax.set_xlim(0.0, 120.0)
    ax.set_ylim(-60.0, 5.0)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=9)
    ax.text(0.015, 0.03,
            "each reflection is a mirror image of the source;\n"
            "amplitude = product of wall reflection factors / (4 pi r)",
            transform=ax.transAxes, va="bottom", ha="left", fontsize=8.5,
            color=COLOR_FG)
    plt.tight_layout()
    save_figure(output_dir, "image_source_reflectogram.svg")
    plt.close()


def generate_prediction_flanking_demo(output_dir: str) -> None:
    """EN 12354-1 simplified flanking prediction (Annex H.3 worked example)."""
    print("Generating prediction_flanking_demo.png...")
    from phonometry.building.prediction.simplified_model import (
        FlankingPath,
        flanking_element,
        predicted_airborne_insulation,
    )

    # Annex H.3 inputs: separating wall Rs,w = 57 dB, Ss = 11.5 m², four
    # flanking elements. Columns: (label, Rw, KFf, KFd=KDf, coupling length lf).
    elements = [
        ("floor", 49, 12.4, 8.9, 4.50),
        ("ceiling", 46, 14.4, 9.2, 4.50),
        ("facade", 42, 12.6, 6.7, 2.55),
        ("wall", 33, 33.5, 15.7, 2.55),
    ]
    paths: list[FlankingPath] = []
    for name, rw, k_ff, k_side, lf in elements:
        ff, df, fd = flanking_element(
            label=name, r_flanking=float(rw), r_separating=57.0,
            k_ff=k_ff, k_fd=k_side, k_df=k_side,
            separating_area=11.5, coupling_length=lf,
        )
        paths.extend((ff, df, fd))
    result = predicted_airborne_insulation(r_direct=57.0, flanking_paths=paths)

    # Sort every path (direct + 12 flanking) by its share of the transmitted
    # energy, largest first.
    contribs = sorted(result.paths, key=lambda c: c.fraction, reverse=True)
    labels = [c.label for c in contribs]
    fracs = [c.fraction * 100.0 for c in contribs]
    df_orange = "#ff7f0e"
    kind_color = {
        "Dd": COLOR_TERTIARY, "Ff": COLOR_PRIMARY,
        "Fd": COLOR_SECONDARY, "Df": df_orange,
    }
    colors = [kind_color[c.kind] for c in contribs]

    direct_share = next(c.fraction for c in result.paths if c.kind == "Dd") * 100.0
    flank_share = 100.0 - direct_share

    _fig, ax = plt.subplots(figsize=(11, 6.4))
    bars = ax.bar(range(len(fracs)), fracs, color=colors, edgecolor=COLOR_FG,
                  linewidth=0.7, zorder=3)
    bars[0].set_linewidth(2.2)  # highlight the dominant path
    ax.annotate("dominant path", xy=(0, fracs[0]), xytext=(1.5, fracs[0] + 3.5),
                fontsize=10, fontweight="bold", color=COLOR_FG,
                arrowprops={"arrowstyle": "->", "lw": 1.1})

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=9)
    ax.set_ylabel("Share of transmitted energy [%]")
    ax.set_xlabel("Transmission path")
    ax.set_ylim(0, max(fracs) + 9.0)
    ax.set_title("EN 12354-1 Flanking Transmission (Annex H.3 example)",
                 fontweight="bold", pad=12)
    ax.grid(axis="y", color=COLOR_GRID, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)

    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor=COLOR_TERTIARY, edgecolor=COLOR_FG, label="Dd — direct"),
        Patch(facecolor=COLOR_PRIMARY, edgecolor=COLOR_FG,
              label="Ff — flanking–flanking"),
        Patch(facecolor=COLOR_SECONDARY, edgecolor=COLOR_FG,
              label="Fd — flanking–separating"),
        Patch(facecolor=df_orange, edgecolor=COLOR_FG,
              label="Df — separating–flanking"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=9)

    rw_dd = result.r_direct_w
    rpw = result.r_prime_w
    lines = [
        f"Rw (Dd) = {rw_dd:.1f} dB",
        f"R'w = {rpw:.1f} dB",
        f"R'w − Rw = {rpw - rw_dd:.1f} dB",
        f"Dd {direct_share:.1f} %   ΣFf,Fd,Df {flank_share:.1f} %",
    ]
    ax.text(0.985, 0.62, "\n".join(lines), transform=ax.transAxes,
            va="top", ha="right", fontsize=11, color=COLOR_FG,
            bbox={"boxstyle": "round,pad=0.5", "facecolor": COLOR_PANEL,
                  "edgecolor": COLOR_GRID})
    plt.tight_layout()
    save_figure(output_dir, "prediction_flanking_demo.png")
    plt.close()


def generate_facade_prediction(output_dir: str) -> None:
    """EN 12354-3 façade airborne insulation prediction (Annex F worked example)."""
    print("Generating facade_prediction.png...")
    from phonometry import FacadeElement, facade_sound_reduction

    bands = [125.0, 250.0, 500.0, 1000.0, 2000.0]
    # Annex F elements: double wall, two windows (area, R) + a small air inlet (Dn,e).
    elements = [
        FacadeElement(name="wall", area=6.0, r=[41, 46, 52, 58, 64]),
        FacadeElement(name="window", area=4.5, r=[23, 22, 30, 36, 37]),
        FacadeElement(name="skylight", area=0.5, r=[24, 27, 30, 33, 30]),
        FacadeElement(name="air inlet", dn_e=[28, 23, 25, 38, 44]),
    ]
    result = facade_sound_reduction(
        elements, area=11.3, volume=50.0, frequencies=bands, bands="octave"
    )

    x = np.arange(len(bands))
    _fig, ax = plt.subplots(figsize=(10, 6.2))
    # Per-element partial indices Rp: thin, faded; they set the transmission floor.
    el_colors = [COLOR_PRIMARY, COLOR_SECONDARY, "#9467bd", "#ff7f0e"]
    for (name, rp), colour in zip(result.element_r.items(), el_colors):
        ax.plot(x, rp, "--", color=colour, linewidth=1.1, alpha=0.65,
                marker=".", markersize=6, label=f"Rp — {name}")
    # Façade apparent reduction R' and standardized level difference D2m,nT.
    ax.plot(x, result.r_prime, "-", color=COLOR_FG, linewidth=2.6, marker="o",
            markersize=6, zorder=5, label="R′ (façade)")
    ax.plot(x, result.d_2m_nt, "-", color=COLOR_TERTIARY, linewidth=2.2, marker="s",
            markersize=6, zorder=5, label="D2m,nT")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(b)}" for b in bands])
    ax.set_xlabel(LABEL_FREQ_HZ)
    ax.set_ylabel("Reduction index / level difference [dB]")
    ax.set_title("EN 12354-3 Façade Sound Insulation (Annex F example)",
                 fontweight="bold", pad=12)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=9, ncol=2)

    info = [
        f"R′tr,s,w = {result.r_tr_s_w} dB   (Ctr = {result.c_tr})",
        f"D2m,nT,w = {result.d_2m_nt_w} dB",
        "air inlet limits the low bands",
    ]
    ax.text(0.985, 0.03, "\n".join(info), transform=ax.transAxes,
            va="bottom", ha="right", fontsize=11, color=COLOR_FG,
            bbox={"boxstyle": "round,pad=0.5", "facecolor": COLOR_PANEL,
                  "edgecolor": COLOR_GRID})
    plt.tight_layout()
    save_figure(output_dir, "facade_prediction.png")
    plt.close()


def generate_intensity_insulation(output_dir: str) -> None:
    """ISO 15186-1 intensity SRI and the Kc-modified index RI,M = RI + Kc."""
    print("Generating intensity_insulation...")
    from phonometry import adaptation_term_kc, intensity_sound_reduction

    # 16 one-third-octave bands (100-3150 Hz); reuse the ISO 717-1 Annex C
    # airborne shape as the intensity SRI target (RI,w = 30 dB, a light wall).
    freqs = np.array([100, 125, 160, 200, 250, 315, 400, 500, 630, 800,
                      1000, 1250, 1600, 2000, 2500, 3150], dtype=float)
    ri = np.array([20.4, 16.3, 17.7, 22.6, 22.4, 22.7, 24.8, 26.6,
                   28.0, 30.5, 31.8, 32.5, 33.4, 33.0, 31.0, 25.5])
    lp1, sm, s = 85.0, 12.0, 10.0
    # Levels that make Formula (7) land on RI, then modify with Kc (Annex B).
    l_in = lp1 - 6.0 - 10.0 * np.log10(sm / s) - ri
    kc = adaptation_term_kc(freqs)
    result = intensity_sound_reduction(
        np.full(16, lp1), l_in, measurement_area=sm, area=s, kc=kc
    )
    assert result.r_i_modified is not None
    assert result.rating is not None and result.rating_modified is not None

    x = np.arange(len(freqs))
    _fig, ax = plt.subplots(figsize=(10, 6.2))
    # Shade the Kc adaptation lift between RI and RI,M (largest at low bands).
    ax.fill_between(x, result.r_i, result.r_i_modified, color=COLOR_TERTIARY,
                    alpha=0.18, zorder=0, label="Kc adaptation")
    ax.plot(x, result.r_i, "-", color=COLOR_PRIMARY, linewidth=2.6, marker="o",
            markersize=6, zorder=5, label="RI (intensity)")
    ax.plot(x, result.r_i_modified, "--", color=COLOR_TERTIARY, linewidth=2.2,
            marker="s", markersize=6, zorder=5, label="RI,M = RI + Kc")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(f)}" for f in freqs], rotation=45, ha="right",
                       fontsize=8)
    ax.set_xlabel(LABEL_FREQ_HZ)
    ax.set_ylabel("Sound reduction index [dB]")
    ax.set_title("ISO 15186-1 Intensity Sound Reduction Index (RI and RI,M)",
                 fontweight="bold", pad=12)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=9)

    # Data-only info box (language-neutral); the Kc lift is explained by the
    # shaded "Kc adaptation" legend entry, which the ES translator handles.
    info = [
        f"RI,w = {result.rating.rating} dB",
        f"RI,M,w = {result.rating_modified.rating} dB",
    ]
    ax.text(0.985, 0.03, "\n".join(info), transform=ax.transAxes,
            va="bottom", ha="right", fontsize=11, color=COLOR_FG,
            bbox={"boxstyle": "round,pad=0.5", "facecolor": COLOR_PANEL,
                  "edgecolor": COLOR_GRID})
    plt.tight_layout()
    save_figure(output_dir, "intensity_insulation.png")
    plt.close()


def generate_survey_insulation(output_dir: str) -> None:
    """ISO 10052 survey method: the reverberation-index correction D -> DnT."""
    print("Generating survey_insulation...")
    from phonometry import reverberation_index, survey_airborne_insulation

    bands = [125.0, 250.0, 500.0, 1000.0, 2000.0]
    # A masonry partition: raw level difference D and the measured receiving-
    # room reverberation time T per octave band.
    l1 = np.array([88.0, 90.0, 92.0, 92.0, 90.0])
    l2 = np.array([55.0, 51.0, 47.0, 41.0, 35.0])
    t = np.array([0.7, 0.6, 0.5, 0.45, 0.4])
    k = reverberation_index(t)
    res = survey_airborne_insulation(l1, l2, k, volume=50.0)
    assert res.rating is not None

    x = np.arange(len(bands))
    _fig, ax = plt.subplots(figsize=(10, 6.2))
    # Shade the reverberation-index correction k between D and DnT.
    ax.fill_between(x, res.d, res.d_nt, color=COLOR_TERTIARY, alpha=0.18,
                    zorder=0, label="k = 10 log10(T/T0)")
    ax.plot(x, res.d, "--", color=COLOR_PRIMARY, linewidth=1.8, marker="o",
            markersize=6, zorder=5, label="D (level difference)")
    ax.plot(x, res.d_nt, "-", color=COLOR_FG, linewidth=2.6, marker="s",
            markersize=6, zorder=5, label="DnT (standardized)")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(b)}" for b in bands])
    ax.set_xlabel(LABEL_FREQ_HZ)
    ax.set_ylabel("Level difference [dB]")
    ax.set_title("ISO 10052 Survey Method: Reverberation-Index Correction",
                 fontweight="bold", pad=12)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=9)

    info = [
        f"DnT,w = {res.rating.rating} dB  (C = {res.rating.c})",
        "octave bands, T0 = 0.5 s",
    ]
    ax.text(0.985, 0.03, "\n".join(info), transform=ax.transAxes,
            va="bottom", ha="right", fontsize=11, color=COLOR_FG,
            bbox={"boxstyle": "round,pad=0.5", "facecolor": COLOR_PANEL,
                  "edgecolor": COLOR_GRID})
    plt.tight_layout()
    save_figure(output_dir, "survey_insulation.png")
    plt.close()


def generate_floor_covering_improvement(output_dir: str) -> None:
    """ISO 16251-1 floor-covering impact-sound improvement spectrum ΔL with ΔLw."""
    print("Generating floor_covering_improvement...")
    from phonometry import impact_improvement

    freqs = [100.0, 125.0, 160.0, 200.0, 250.0, 315.0, 400.0, 500.0,
             630.0, 800.0, 1000.0, 1250.0, 1600.0, 2000.0, 2500.0, 3150.0]
    # A real textile carpet measured on the CSTB mock-up: the improvement
    # spectrum digitized from Figure 4 of Foret, Chene & Guigou-Carter, Forum
    # Acusticum 2011 (ISO 16251-1 series). The published weighted improvement
    # is delta-Lw = 29 dB.
    bare = np.full(16, 78.0)
    covering = bare - np.array([5, 8, 10, 14, 18, 23, 30, 31, 39, 49,
                                53, 57, 60, 67, 68, 71], dtype=float)
    res = impact_improvement(bare, covering, freqs)
    assert res.delta_lw is not None

    x = np.arange(len(freqs))
    _fig, ax = plt.subplots(figsize=(10, 6.2))
    ax.fill_between(x, 0.0, res.improvement, color=COLOR_TERTIARY, alpha=0.18,
                    zorder=0)
    ax.plot(x, res.improvement, "-", color=COLOR_PRIMARY, linewidth=2.4,
            marker="o", markersize=6, zorder=5, label="delta-L (improvement)")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(b)}" for b in freqs], rotation=45, fontsize=8)
    ax.set_xlabel(LABEL_FREQ_HZ)
    ax.set_ylabel("Improvement of impact sound insulation [dB]")
    ax.set_ylim(bottom=0.0)
    ax.set_title("ISO 16251-1 Floor-Covering Impact Sound Improvement",
                 fontweight="bold", pad=12)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=9)

    info = [
        f"delta-Lw = {res.delta_lw} dB  (ISO 717-2)",
        "one-third octave, mock-up (a0 = 1e-6 m/s^2)",
    ]
    ax.text(0.985, 0.03, "\n".join(info), transform=ax.transAxes,
            va="bottom", ha="right", fontsize=11, color=COLOR_FG,
            bbox={"boxstyle": "round,pad=0.5", "facecolor": COLOR_PANEL,
                  "edgecolor": COLOR_GRID})
    plt.tight_layout()
    save_figure(output_dir, "floor_covering_improvement.png")
    plt.close()



def generate_heavy_impact_sources(output_dir: str) -> None:
    """Rubber-ball and bang-machine LFE spectra with their printed tolerances."""
    print("Generating heavy_impact_sources...")
    from phonometry import (
        a_weighted_maximum_impact_level,
        heavy_impact_source_limits,
        heavy_impact_source_specification,
    )

    _fig, (ax_src, ax_rate) = plt.subplots(1, 2, figsize=(13.0, 5.6))

    x = np.arange(5)
    for source, colour in (
        ("rubber_ball", COLOR_PRIMARY), ("bang_machine", COLOR_SECONDARY)
    ):
        spec = heavy_impact_source_specification(source)
        _f, lower, upper = heavy_impact_source_limits(source)
        label = source.replace("_", " ")
        ax_src.fill_between(x, lower, upper, color=colour, alpha=0.30, zorder=1,
                            label=f"{label} tolerance")
        ax_src.plot(x, spec.force_exposure_level, "-o", color=colour, linewidth=2.4,
                    markersize=7, zorder=4, label=f"{label} nominal")
    ax_src.set_xticks(x)
    ax_src.set_xticklabels(["31.5", "63", "125", "250", "500"])
    ax_src.set_xlabel(LABEL_FREQ_HZ)
    ax_src.set_ylabel("Impact force exposure level LFE [dB re 1 N]")
    ax_src.set_title("Standard heavy impact sources\n(ISO 16283-2 Table A.1, "
                     "JIS A 1418-2 Tables A.1/A.2)", fontweight="bold", pad=10)
    ax_src.grid(color=COLOR_GRID, linestyle="--", alpha=0.5, zorder=0)
    ax_src.set_axisbelow(True)
    ax_src.legend(loc="upper right", fontsize=9)

    # ISO 717-2:2020 Table D.4: a field measurement in octave bands.
    levels = [65.3, 64.5, 58.0, 55.8]
    res = a_weighted_maximum_impact_level(levels)
    xr = np.arange(4)
    ax_rate.bar(xr - 0.19, res.levels, width=0.36, color=COLOR_PRIMARY,
                label="Li,Fmax (measured)", zorder=3)
    ax_rate.bar(xr + 0.19, res.corrected, width=0.36, color=COLOR_TERTIARY,
                label="Li,Fmax + A (Table D.3)", zorder=3)
    ax_rate.axhline(res.rating, color=COLOR_SECONDARY, linewidth=2.0, zorder=4,
                    label=f"LiA,Fmax = {res.rating} dB")
    ax_rate.set_xticks(xr)
    ax_rate.set_xticklabels(["63", "125", "250", "500"])
    ax_rate.set_xlabel(LABEL_FREQ_HZ)
    ax_rate.set_ylabel("Maximum impact sound pressure level [dB]")
    ax_rate.set_title("A-weighted heavy-impact rating\n(ISO 717-2 Annex D, "
                      "Table D.4 worked example)", fontweight="bold", pad=10)
    ax_rate.grid(axis="y", color=COLOR_GRID, linestyle="--", alpha=0.5, zorder=0)
    ax_rate.set_axisbelow(True)
    ax_rate.legend(loc="upper right", fontsize=9)
    ax_rate.text(
        0.02, 0.03,
        f"unrounded sum = {res.unrounded:.6f} dB",
        transform=ax_rate.transAxes, va="bottom", ha="left", fontsize=10,
        color=COLOR_FG,
        bbox={"boxstyle": "round,pad=0.4", "facecolor": COLOR_PANEL,
              "edgecolor": COLOR_GRID},
    )

    plt.tight_layout()
    save_figure(output_dir, "heavy_impact_sources.png")
    plt.close()


def generate_ceiling_plenum_flanking(output_dir: str) -> None:
    """Ceiling/plenum flanking path Rcl and the CAC of an accredited report."""
    print("Generating ceiling_plenum_flanking...")
    from phonometry import (
        ceiling_attenuation_class,
        plenum_flanking_reduction_index,
    )

    _fig, (ax_path, ax_cac) = plt.subplots(1, 2, figsize=(13.0, 5.6))

    # Vigran Figs. 9.11-9.13 geometry: LS = LR = 4,75 m, plenum h = 0,43 m,
    # 9,5 mm plasterboard ceiling, reflecting plenum sidewalls.
    freqs = np.array([63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0])
    ceiling = np.array([17.0, 21.0, 25.0, 29.0, 32.0, 30.0, 38.0])
    x = np.arange(freqs.size)
    for depth, colour, style in (
        (0.43, COLOR_PRIMARY, "-"), (0.86, COLOR_TERTIARY, "--")
    ):
        res = plenum_flanking_reduction_index(
            ceiling, ceiling, ceiling_length=4.75, plenum_height=depth,
            frequency=freqs,
        )
        ax_path.plot(x, res.reduction_index, style, color=colour, linewidth=2.4,
                     marker="o", markersize=6, zorder=4,
                     label=f"Rcl, plenum h = {depth:g} m")
    ax_path.plot(x, 2.0 * ceiling, ":", color=COLOR_SECONDARY, linewidth=2.0,
                 marker="s", markersize=5, zorder=3, label="RS + RR (two ceilings)")
    ax_path.set_xticks(x)
    ax_path.set_xticklabels(["63", "125", "250", "500", "1k", "2k", "4k"])
    ax_path.set_xlabel(LABEL_FREQ_HZ)
    ax_path.set_ylabel("Sound reduction index [dB]")
    ax_path.set_title("Suspended-ceiling plenum path\n(one-dimensional model, "
                      "LR = 4.75 m, reflecting sidewalls)",
                      fontweight="bold", pad=10)
    ax_path.grid(color=COLOR_GRID, linestyle="--", alpha=0.5, zorder=0)
    ax_path.set_axisbelow(True)
    ax_path.legend(loc="upper left", fontsize=9)

    # ALA 16-091-4 (2016), tested to ASTM E1414/E1414M-11a: CAC 34.
    dnc = np.array([14.4, 18.6, 21.7, 24.1, 23.4, 30.3, 33.7, 35.2,
                    41.6, 44.2, 42.1, 36.8, 35.7, 36.0, 36.9, 37.9])
    cac = ceiling_attenuation_class(dnc)
    xc = np.arange(dnc.size)
    ax_cac.fill_between(xc, cac.measured, cac.shifted_reference,
                        where=(cac.measured < cac.shifted_reference).tolist(),
                        color=COLOR_SECONDARY, alpha=0.25, interpolate=True,
                        zorder=1, label="deficiencies")
    ax_cac.plot(xc, cac.measured, "-o", color=COLOR_PRIMARY, linewidth=2.4,
                markersize=6, zorder=4, label="Dn,c (measured)")
    ax_cac.plot(xc, cac.shifted_reference, "--s", color=COLOR_TERTIARY,
                linewidth=2.0, markersize=5, zorder=3,
                label="ASTM E413 contour, fitted")
    ax_cac.set_xticks(xc[::2])
    ax_cac.set_xticklabels(["125", "200", "315", "500", "800", "1.25k",
                            "2k", "3.15k"], rotation=45, fontsize=8)
    ax_cac.set_xlabel(LABEL_FREQ_HZ)
    ax_cac.set_ylabel("Normalized ceiling attenuation Dn,c [dB]")
    ax_cac.set_title(f"Ceiling attenuation class\n(ASTM E1414/E413, "
                     f"CAC = {cac.rating} dB)", fontweight="bold", pad=10)
    ax_cac.grid(color=COLOR_GRID, linestyle="--", alpha=0.5, zorder=0)
    ax_cac.set_axisbelow(True)
    ax_cac.legend(loc="upper left", fontsize=9)

    plt.tight_layout()
    save_figure(output_dir, "ceiling_plenum_flanking.png")
    plt.close()


def generate_masonry_wall_ties(output_dir: str) -> None:
    """Wall-tie coupling loss factor and the resonance shift it causes."""
    print("Generating masonry_wall_ties...")
    from phonometry import (
        double_wall_transmission_loss,
        wall_tie_coupling_loss_factor,
        wall_tie_stiffness,
        wall_tie_stiffness_per_area,
    )

    _fig, (ax_clf, ax_tl) = plt.subplots(1, 2, figsize=(13.0, 5.6))

    # Two 100 mm masonry leaves, 150 and 170 kg/m2 (Hopkins Fig. 5.30
    # flanking-laboratory cavity wall), 2,5 ties per m2.
    freq = np.logspace(np.log10(50.0), np.log10(5000.0), 200)
    stiffness1 = 150.0 * 2000.0**2 * 0.1**2 / 12.0
    stiffness2 = 170.0 * 2000.0**2 * 0.1**2 / 12.0
    ties = ("butterfly", "double_triangle", "vertical_twist")
    colours = (COLOR_PRIMARY, COLOR_TERTIARY, COLOR_SECONDARY)
    for tie, colour in zip(ties, colours, strict=True):
        clf = wall_tie_coupling_loss_factor(
            freq, 150.0, 170.0, stiffness1, stiffness2,
            ties_per_area=2.5, tie=tie,
        )
        label = f"{tie.replace('_', ' ')} ({wall_tie_stiffness(tie)[1] / 1e6:g} MN/m)"
        ax_clf.loglog(freq, clf.coupling_loss_factor, color=colour, linewidth=2.4,
                      zorder=4, label=label)
    rigid = wall_tie_coupling_loss_factor(
        freq, 150.0, 170.0, stiffness1, stiffness2, ties_per_area=2.5
    )
    ax_clf.loglog(freq, rigid.rigid_coupling_loss_factor, "--", color=COLOR_MUTED,
                  linewidth=2.0, zorder=3, label="rigid connection (Yc = 0)")
    ax_clf.set_xticks([50, 125, 250, 500, 1000, 2000, 4000])
    ax_clf.set_xticklabels(["50", "125", "250", "500", "1k", "2k", "4k"])
    ax_clf.set_xlim(50.0, 5000.0)
    # Plain ASCII decade labels: the mathtext 10^-n exponent would put a
    # U+2212 minus in the SVG, which not every reader's sans-serif font has.
    ax_clf.set_yticks([1e-8, 1e-6, 1e-4, 1e-2, 1.0])
    ax_clf.set_yticklabels(["1e-8", "1e-6", "1e-4", "1e-2", "1"])
    ax_clf.set_ylim(1e-8, 2.0)
    ax_clf.set_xlabel(LABEL_FREQ_HZ)
    ax_clf.set_ylabel("Coupling loss factor eta_ij")
    ax_clf.set_title("Wall-tie structure-borne coupling\n(point-connection model, "
                     "2.5 ties/m2)", fontweight="bold", pad=10)
    ax_clf.grid(which="both", color=COLOR_GRID, linestyle="--", alpha=0.5, zorder=0)
    ax_clf.set_axisbelow(True)
    ax_clf.legend(loc="lower left", fontsize=9)

    # Hopkins Fig. 4.35: two 140 kg/m2 leaves, empty 75 mm cavity, 2,5 ties/m2
    # of s_75mm = 2e6 N/m, which lifts fmsm from 26 Hz to 50 Hz.
    bands = np.array([20.0, 25.0, 31.5, 40.0, 50.0, 63.0, 80.0, 100.0,
                      125.0, 160.0, 200.0, 250.0, 315.0, 400.0, 500.0])
    per_area = wall_tie_stiffness_per_area(2.5, 2.0e6)
    plain = double_wall_transmission_loss(bands, 140.0, 140.0, 0.075)
    tied = double_wall_transmission_loss(
        bands, 140.0, 140.0, 0.075, tie_stiffness_per_area=per_area
    )
    xb = np.arange(bands.size)
    ax_tl.plot(xb, plain.transmission_loss, "-o", color=COLOR_PRIMARY,
               linewidth=2.4, markersize=6, zorder=4, label="cavity wall, no ties")
    ax_tl.plot(xb, tied.transmission_loss, "--s", color=COLOR_SECONDARY,
               linewidth=2.4, markersize=6, zorder=4, label="2.5 ties/m2, k = 2 MN/m")
    positions = []
    for curve, colour in ((plain, COLOR_PRIMARY), (tied, COLOR_SECONDARY)):
        f0 = curve.resonance_frequency
        assert f0 is not None
        pos = float(np.interp(np.log10(f0), np.log10(bands), xb))
        positions.append(pos)
        ax_tl.axvline(pos, color=colour, linestyle=":", linewidth=1.8, zorder=2)
        ax_tl.annotate(
            f"fmsm = {f0:.0f} Hz", xy=(pos, 0.46),
            xycoords=("data", "axes fraction"), ha="right" if positions[:1] else "left",
            va="bottom", fontsize=9, color=colour, rotation=90,
        )
    ax_tl.axvspan(positions[0], positions[1], color=COLOR_SECONDARY, alpha=0.30,
                  zorder=1, label="combined-mass range added by the ties")
    ax_tl.set_xticks(xb[::2])
    ax_tl.set_xticklabels(["20", "31.5", "50", "80", "125", "200", "315", "500"])
    ax_tl.set_xlabel(LABEL_FREQ_HZ)
    ax_tl.set_ylabel("Sound reduction index R [dB]")
    ax_tl.set_title("Ties stiffen the cavity\n(140 kg/m2 leaves, 75 mm cavity, "
                    "2.5 ties/m2)", fontweight="bold", pad=10)
    ax_tl.grid(color=COLOR_GRID, linestyle="--", alpha=0.5, zorder=0)
    ax_tl.set_axisbelow(True)
    ax_tl.legend(loc="upper left", fontsize=9)

    plt.tight_layout()
    save_figure(output_dir, "masonry_wall_ties.png")
    plt.close()


def generate_floating_floor_prediction(output_dir: str) -> None:
    """Floating-floor improvement DeltaL(f) under the three prediction laws."""
    print("Generating floating_floor_prediction...")
    from phonometry import (
        floating_floor_improvement_spectrum,
        floating_floor_resonance_frequency,
        weighted_floating_floor_improvement,
    )

    # The worked floating floor of ISO 12354-2:2017 Annex G: a 35 mm screed
    # (m' = 73,5 kg/m2) on a resilient layer of s' = 8 MN/m3.
    mass_per_area, stiffness = 73.5, 8.0e6
    f0 = floating_floor_resonance_frequency(stiffness, mass_per_area)
    delta_lw = weighted_floating_floor_improvement(mass_per_area, stiffness)

    bands = np.array([50.0, 63.0, 80.0, 100.0, 125.0, 160.0, 200.0, 250.0,
                      315.0, 400.0, 500.0, 630.0, 800.0, 1000.0, 1250.0,
                      1600.0, 2000.0, 2500.0, 3150.0, 4000.0, 5000.0])
    freqs = np.logspace(np.log10(40.0), np.log10(5000.0), 400)
    screed = floating_floor_improvement_spectrum(freqs, resonance_frequency=f0)
    asphalt = floating_floor_improvement_spectrum(
        freqs, resonance_frequency=f0, model="cremer")
    lightweight = floating_floor_improvement_spectrum(
        freqs, resonance_frequency=f0, model="cremer_hammer",
        limiting_frequency=521.0)
    printed = floating_floor_improvement_spectrum(bands, resonance_frequency=f0)

    _fig, ax = plt.subplots(figsize=(10, 6.2))
    ax.fill_betweenx([-5.0, 100.0], 40.0, f0, color=theme_fill(COLOR_FG, ax),
                     zorder=0)
    ax.plot(freqs, lightweight.improvement, "-.", color=COLOR_TERTIARY,
            linewidth=2.0, label="40 log10(f/f0) + hammer term (chipboard)")
    ax.plot(freqs, asphalt.improvement, "--", color=COLOR_SECONDARY,
            linewidth=2.0, label="40 log10(f/f0) (asphalt, dry)")
    ax.plot(freqs, screed.improvement, "-", color=COLOR_PRIMARY, linewidth=2.4,
            label="30 log10(f/f0) (sand-cement screed)")
    ax.plot(bands, printed.improvement, "o", color=COLOR_PRIMARY,
            markersize=5.5, zorder=6, label="ISO 12354-2 Annex G bands")
    ax.axvline(f0, color=COLOR_FG, linestyle=":", linewidth=1.3)
    ax.annotate("f0 = 52.8 Hz", xy=(f0, 46.0), xytext=(f0 * 1.15, 46.0),
                fontsize=10, color=COLOR_FG, va="center")

    ax.set_xscale("log")
    format_frequency_axis(ax, 40.0, 5000.0)
    ax.set_xlabel(LABEL_FREQ_HZ)
    ax.set_ylabel("Improvement of impact sound insulation [dB]")
    ax.set_ylim(-5.0, 100.0)
    ax.set_title("Floating-Floor Impact Improvement Above the Mass-Spring "
                 "Resonance", fontweight="bold", pad=12)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=9)

    info = [
        "35 mm screed m' = 73.5 kg/m2 on s' = 8 MN/m3",
        f"delta-Lw = {delta_lw:.1f} dB  (ISO 12354-2 Formula C.4)",
    ]
    ax.text(0.985, 0.03, "\n".join(info), transform=ax.transAxes,
            va="bottom", ha="right", fontsize=11, color=COLOR_FG,
            bbox={"boxstyle": "round,pad=0.5", "facecolor": COLOR_PANEL,
                  "edgecolor": COLOR_GRID})
    plt.tight_layout()
    save_figure(output_dir, "floating_floor_prediction.png")
    plt.close()


def generate_soft_covering_prediction(output_dir: str) -> None:
    """Soft-covering DeltaL(f) from the tapping-machine contact stiffness."""
    print("Generating soft_covering_prediction...")
    from phonometry import (
        covering_contact_stiffness,
        covering_improvement,
        infinite_plate_impedance,
        plate_bending_stiffness,
        plate_contact_stiffness,
    )

    # A 140 mm cast in-situ concrete slab (Hopkins Table A2: 2 200 kg/m3,
    # cL = 3 800 m/s, nu = 0,2) carrying the two soft coverings of Hopkins
    # Fig. 4.64: E/d = 1,5e11 N/m3 (a few mm of solid PVC) and 2,8e8 N/m3
    # (a vinyl or carpet with a resilient backing).
    density, longitudinal, poisson, thickness = 2200.0, 3800.0, 0.2, 0.14
    modulus = density * longitudinal**2 * (1.0 - poisson**2)
    plate_stiffness = plate_contact_stiffness(modulus, poisson_ratio=poisson)
    impedance = infinite_plate_impedance(
        plate_bending_stiffness(modulus, thickness, poisson),
        density * thickness)

    # covering_improvement returns the band value directly: the tapping
    # machine's force spectrum is a line spectrum at multiples of the 10 Hz
    # impact rate, and the band improvement is the ratio of the mean-square
    # forces summed over the lines that fall in each band (Hopkins Eq. 3.91).
    bands = np.array([50.0, 63.0, 80.0, 100.0, 125.0, 160.0, 200.0, 250.0,
                      315.0, 400.0, 500.0, 630.0, 800.0, 1000.0, 1250.0,
                      1600.0, 2000.0, 2500.0, 3150.0, 4000.0, 5000.0])
    layer = 0.005
    series = (
        ("No. 1: E/d = 1.5e11 N/m3", 1.5e11, COLOR_SECONDARY, "--"),
        ("No. 2: E/d = 2.8e8 N/m3", 2.8e8, COLOR_PRIMARY, "-"),
    )

    _fig, ax = plt.subplots(figsize=(10, 6.2))
    labels = []
    for label, stiffness_per_volume, colour, style in series:
        result = covering_improvement(
            bands,
            covering_contact_stiffness(stiffness_per_volume * layer, layer),
            plate_stiffness, impedance)
        ax.plot(bands, result.improvement, style, color=colour, linewidth=2.2,
                marker="o", markersize=5, label=label)
        ax.plot(bands, result.two_line, ":", color=colour, linewidth=1.5)
        ax.axvline(result.cut_off_frequency, color=colour, linestyle=":",
                   linewidth=1.0, alpha=0.7)
        labels.append(f"{label.split(':')[0]}: fco = "
                      f"{result.cut_off_frequency:.0f} Hz")

    ax.plot([], [], ":", color=COLOR_MUTED, linewidth=1.5,
            label="two-line estimate (0 dB, 12 dB/oct)")
    ax.set_xscale("log")
    format_frequency_axis(ax, 50.0, 5000.0)
    ax.set_xlabel(LABEL_FREQ_HZ)
    ax.set_ylabel("Improvement of impact sound insulation [dB]")
    ax.set_ylim(-5.0, 80.0)
    ax.set_title("Soft Floor Covering Improvement From the Hammer Contact "
                 "Stiffness", fontweight="bold", pad=12)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=9)

    info = ["140 mm concrete slab, hammer 0.5 kg, r = 15 mm", *labels]
    ax.text(0.02, 0.70, "\n".join(info), transform=ax.transAxes,
            va="top", ha="left", fontsize=11, color=COLOR_FG,
            bbox={"boxstyle": "round,pad=0.5", "facecolor": COLOR_PANEL,
                  "edgecolor": COLOR_GRID})
    plt.tight_layout()
    save_figure(output_dir, "soft_covering_prediction.png")
    plt.close()


def generate_flanking_transmission(output_dir: str) -> None:
    """ISO 10848 vibration reduction index Kij per band with the mean K̄ij."""
    print("Generating flanking_transmission...")
    from phonometry import vibration_reduction_index

    freqs = [100.0, 125.0, 160.0, 200.0, 250.0, 315.0, 400.0, 500.0, 630.0,
             800.0, 1000.0, 1250.0, 1600.0, 2000.0, 2500.0, 3150.0, 4000.0, 5000.0]
    # A rigid T-junction of two heavy walls: measured direction-averaged velocity
    # level difference rising gently with frequency (typical laboratory data).
    dv = np.array([4.5, 4.8, 5.2, 5.6, 6.0, 6.5, 7.0, 7.6, 8.1, 8.7, 9.2, 9.8,
                   10.3, 10.9, 11.4, 11.9, 12.3, 12.7])
    res = vibration_reduction_index(
        dv, junction_length=4.0, area_i=12.0, area_j=10.0,
        frequency=freqs,
        structural_reverberation_time_i=0.35,
        structural_reverberation_time_j=0.40,
    )
    assert res.single_number is not None

    x = np.arange(len(freqs))
    _fig, ax = plt.subplots(figsize=(10, 6.2))
    ax.plot(x, res.k_ij, "-", color=COLOR_PRIMARY, linewidth=2.4, marker="o",
            markersize=6, zorder=5, label="Kij (ISO 10848)")
    ax.axhline(res.single_number, color=COLOR_SECONDARY, linestyle="--",
               linewidth=1.6, zorder=4, label="mean Kij (200-1250 Hz)")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(b)}" for b in freqs], rotation=45, fontsize=8)
    ax.set_xlabel(LABEL_FREQ_HZ)
    ax.set_ylabel("Vibration reduction index Kij [dB]")
    ax.set_title("ISO 10848 Junction Vibration Reduction Index",
                 fontweight="bold", pad=12)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=9)

    info = [
        "rigid T-junction, two heavy walls",
        "lij = 4 m, Si = 12 m^2, Sj = 10 m^2",
        "Formula (13), one-third octave",
        f"mean Kij = {res.single_number:.1f} dB",
    ]
    ax.text(0.985, 0.03, "\n".join(info), transform=ax.transAxes,
            va="bottom", ha="right", fontsize=11, color=COLOR_FG,
            bbox={"boxstyle": "round,pad=0.5", "facecolor": COLOR_PANEL,
                  "edgecolor": COLOR_GRID})
    plt.tight_layout()
    save_figure(output_dir, "flanking_transmission.png")
    plt.close()


def generate_reverberation_models(output_dir: str) -> None:
    """Sabine/Eyring/Millington/Fitzroy/Arau reverberation time over octaves."""
    print("Generating reverberation_models...")
    from phonometry import air_attenuation_m, reverberation_time_models

    bands = [125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0]
    # A 10 x 7 x 3.5 m room (V = 245 m3, S = 259 m2) with a strongly anisotropic
    # absorption distribution: a very absorptive floor/ceiling pair (carpet plus
    # an acoustic ceiling) against hard end walls and lightly treated side walls.
    # This is where the axial models (Fitzroy, Arau-Puchades) part company with
    # the isotropic Sabine and Eyring estimates.
    alpha_x = [0.06, 0.07, 0.08, 0.09, 0.10, 0.10]   # hard end walls
    alpha_y = [0.12, 0.14, 0.16, 0.18, 0.20, 0.20]   # lightly treated side walls
    alpha_z = [0.30, 0.50, 0.65, 0.78, 0.82, 0.80]   # carpet + acoustic ceiling
    m = air_attenuation_m(bands, 20.0, 50.0)
    res = reverberation_time_models(
        (10.0, 7.0, 3.5), (alpha_x, alpha_y, alpha_z),
        air_attenuation=m, frequencies=bands,
    )

    x = np.arange(len(bands))
    _fig, ax = plt.subplots(figsize=(10, 6.2))
    styles = [
        ("Sabine", res.sabine, COLOR_SECONDARY, "s", 1.8),
        ("Eyring", res.eyring, COLOR_TERTIARY, "^", 1.8),
        ("Millington-Sette", res.millington_sette, "#9467bd", "v", 1.8),
        ("Fitzroy", res.fitzroy, "#ff7f0e", "D", 1.8),
        ("Arau-Puchades", res.arau_puchades, COLOR_PRIMARY, "o", 2.6),
    ]
    for label, curve, color, marker, lw in styles:
        ax.plot(x, curve, color=color, linewidth=lw, marker=marker,
                markersize=6, label=label, zorder=5)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(b)}" for b in bands])
    ax.set_xlabel(LABEL_FREQ_HZ)
    ax.set_ylabel(r"Reverberation time $T$ [s]")
    ax.set_title("Reverberation-time prediction models", fontweight="bold", pad=12)
    ax.set_ylim(bottom=0.0)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=9)

    info = [
        "room 10 x 7 x 3.5 m",
        "V = 245 m^3, S = 259 m^2",
        "anisotropic: absorptive floor/ceiling",
        "c0 = 343 m/s, air at 20 C / 50 % RH",
    ]
    ax.text(0.015, 0.03, "\n".join(info), transform=ax.transAxes,
            va="bottom", ha="left", fontsize=11, color=COLOR_FG,
            bbox={"boxstyle": "round,pad=0.5", "facecolor": COLOR_PANEL,
                  "edgecolor": COLOR_GRID})
    plt.tight_layout()
    save_figure(output_dir, "reverberation_models.svg")
    plt.close()


def generate_structure_borne_power(output_dir: str) -> None:
    """EN 15657 structure-borne sound power injected into the reception plate."""
    print("Generating structure_borne_power...")
    from phonometry import reception_plate_power

    bands = np.array([50.0, 100.0, 200.0, 400.0, 800.0, 1600.0, 3150.0])
    # A pump-like source on a low-mobility (heavy) and a high-mobility (light)
    # reception plate; the two determinations should agree within the method.
    lv_low = np.array([88.0, 90.0, 87.0, 84.0, 80.0, 76.0, 71.0])
    lv_high = lv_low + 6.0                      # lighter plate vibrates more
    res_low = reception_plate_power(lv_low, bands, mass_per_area=600.0, area=2.0,
                                    reverberation_time=0.8)
    res_high = reception_plate_power(lv_high, bands, mass_per_area=150.0, area=2.0,
                                     reverberation_time=0.5)

    x = np.arange(bands.size)
    _fig, ax = plt.subplots(figsize=(10, 6.2))
    ax.bar(x - 0.2, res_low.power_level, width=0.4, color=COLOR_PRIMARY,
           edgecolor=COLOR_FG, linewidth=0.6, label="low-mobility plate")
    ax.bar(x + 0.2, res_high.power_level, width=0.4, color=COLOR_SECONDARY,
           edgecolor=COLOR_FG, linewidth=0.6, label="high-mobility plate")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{b:g}" for b in bands])
    ax.set_xlabel(LABEL_FREQ_HZ)
    ax.set_ylabel(r"Structure-borne power level $L_{Ws}$ [dB re 1 pW]")
    ax.set_title("EN 15657 Reception-Plate Structure-Borne Sound Power",
                 fontweight="bold", pad=12)
    ax.grid(which="major", axis="y", color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", fontsize=9)

    info = [
        "LWs = 10 log10(2 pi f eta m S) + Lv - 60 dB",
        "eta = 2.2/(f Ts),  v0 = 1 nm/s",
        "reception-plate method (clause 7)",
    ]
    ax.text(0.015, 0.02, "\n".join(info), transform=ax.transAxes,
            va="bottom", ha="left", fontsize=9, color=COLOR_FG, family="monospace",
            bbox={"boxstyle": "round,pad=0.5", "facecolor": COLOR_PANEL,
                  "edgecolor": COLOR_GRID})
    plt.tight_layout()
    save_figure(output_dir, "structure_borne_power.svg")
    plt.close()


def generate_installed_structure_borne(output_dir: str) -> None:
    """EN 12354-5 installed structure-borne sound: characteristic power to SPL."""
    print("Generating installed_structure_borne...")
    from phonometry import (
        coupling_term,
        installed_source_prediction,
        installed_structure_borne_power_level,
    )

    bands = np.array([63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0])
    lws_c = np.array([78.0, 82.0, 84.0, 81.0, 77.0, 72.0, 66.0])   # EN 15657 source
    # Frequency-dependent source / receiver point mobilities (illustrative).
    ys = (2.0e-4 + 1.0e-4j) * (bands / 250.0)
    yi = (3.0e-5 + 1.0e-5j) * np.ones_like(bands)
    dc = np.array([float(coupling_term(a, b)) for a, b in zip(ys, yi)])
    lws_inst = installed_structure_borne_power_level(lws_c, dc)
    paths = [
        {"adjustment_term": 6.0,
         "flanking_reduction_index": np.array([44., 47., 50., 53., 56., 59., 62.]),
         "element_area": 12.0},
        {"adjustment_term": 7.0,
         "flanking_reduction_index": np.array([46., 49., 52., 55., 58., 61., 64.]),
         "element_area": 9.0},
    ]
    res = installed_source_prediction(lws_c, dc, paths, frequencies=bands)

    x = np.arange(bands.size)
    _fig, ax = plt.subplots(figsize=(10, 6.2))
    ax.plot(x, lws_c, color=COLOR_SECONDARY, marker="o", lw=2.0,
            label=r"characteristic $L_{Ws,c}$ (EN 15657)")
    ax.plot(x, lws_inst, color=COLOR_TERTIARY, marker="s", lw=2.0,
            label=r"installed $L_{Ws,inst}$ = $L_{Ws,c}-D_C$")
    for k, p in enumerate(res.path_levels):
        ax.plot(x, p, color=COLOR_GRID, lw=1.0, ls=":", marker=".",
                label="paths $L_{n,s,ij}$" if k == 0 else None)
    ax.plot(x, res.total_level, color=COLOR_PRIMARY, marker="D", lw=2.4,
            label=r"total $L_{n,s}$")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{b:g}" for b in bands])
    ax.set_xlabel(LABEL_FREQ_HZ)
    ax.set_ylabel("Level [dB]")
    ax.set_title("EN 12354-5 Installed Structure-Borne Sound",
                 fontweight="bold", pad=12)
    ax.grid(which="major", axis="y", color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", fontsize=9)

    info = [
        "DC = 10 log10(|Ys+Yi|^2 / (|Ys| Re Yi))",
        "Ln,s,ij = LWs,inst - Dsa - Rij - 10 log10(Si/S0) - 10 log10(A0/4)",
        "Ln,s = 10 log10(sum 10^(Ln,s,ij/10)),  S0 = A0 = 10 m2",
    ]
    ax.text(0.015, 0.02, "\n".join(info), transform=ax.transAxes,
            va="bottom", ha="left", fontsize=8.5, color=COLOR_FG, family="monospace",
            bbox={"boxstyle": "round,pad=0.5", "facecolor": COLOR_PANEL,
                  "edgecolor": COLOR_GRID})
    plt.tight_layout()
    save_figure(output_dir, "installed_structure_borne.svg")
    plt.close()


def generate_insulation_uncertainty_demo(output_dir: str) -> None:
    """ISO 12999-1 per-band + single-number measurement uncertainty (situation B)."""
    print("Generating insulation_uncertainty_demo.png...")
    from phonometry.building.measurement.insulation import weighted_rating
    from phonometry.building.measurement.uncertainty import (
        band_uncertainty,
        insulation_coverage_factor,
        insulation_expanded_uncertainty,
        single_number_uncertainty,
    )

    # Reuse the ISO 717-1 Annex C measured R' curve (100 Hz .. 3150 Hz); its
    # weighted rating is R'w = 30 dB.
    freqs = np.array([100, 125, 160, 200, 250, 315, 400, 500, 630, 800,
                      1000, 1250, 1600, 2000, 2500, 3150], dtype=float)
    measured = np.array([20.4, 16.3, 17.7, 22.6, 22.4, 22.7, 24.8, 26.6,
                         28.0, 30.5, 31.8, 32.5, 33.4, 33.0, 31.0, 25.5])
    rating = weighted_rating(measured).rating

    # Per-band standard uncertainty u (ISO 12999-1 Table 2, situation B); match
    # each measured band to its tabulated value, then expand at k = 1.96 (95 %).
    band = band_uncertainty("airborne", "B")
    band_f, band_u = band.to_arrays()
    idx = [int(np.argmin(np.abs(band_f - f))) for f in freqs]
    u_band = band_u[idx]
    k = insulation_coverage_factor(0.95)
    exp_band = np.array(
        [insulation_expanded_uncertainty(float(v), 0.95) for v in u_band]
    )

    # Single-number expanded uncertainty for the rating.
    u_single = single_number_uncertainty("r_w", "B")
    exp_single = insulation_expanded_uncertainty(u_single, 0.95)

    _fig, ax = plt.subplots(figsize=(10, 6.3))
    # Nested bands: the inner one is placed twice as far from the page as the
    # outer, so the two read as distinct steps of the same hue.
    ax.fill_between(freqs, measured - exp_band, measured + exp_band,
                    color=theme_fill(COLOR_PRIMARY, ax), zorder=0,
                    label="Expanded uncertainty ±U (95 %)")
    ax.fill_between(freqs, measured - u_band, measured + u_band,
                    color=theme_fill(COLOR_PRIMARY, ax, delta_e=26.0), zorder=0.1,
                    label="Standard uncertainty ±u")
    ax.semilogx(freqs, measured, marker="o", color=COLOR_PRIMARY, linewidth=1.9,
                markersize=5, markerfacecolor="white", markeredgewidth=1.4,
                zorder=4, label="Measured R'")

    # Single-number R'w with its expanded uncertainty, read at 500 Hz.
    ax.errorbar(500, rating, yerr=exp_single, fmt="D", color=COLOR_SECONDARY,
                markersize=9, capsize=6, elinewidth=1.8, zorder=6,
                label="R'w ± U (single number)")
    ax.axvline(500, color=COLOR_FG, linestyle=":", alpha=0.35, zorder=0)

    # Word-free box (the situation and band meanings are in the title/legend, so
    # translation reduces to the automatic decimal-comma substitution).
    box = [
        f"R'w = {rating} ± {exp_single:.1f} dB",
        f"U = k·u ,  k = {k:g} (95 %)",
    ]
    ax.text(0.03, 0.97, "\n".join(box), transform=ax.transAxes, va="top",
            ha="left", fontsize=10, color=COLOR_FG,
            bbox={"boxstyle": "round,pad=0.5", "facecolor": COLOR_PANEL,
                  "edgecolor": COLOR_GRID})

    ax.set_title("ISO 12999-1 Measurement Uncertainty (situation B, airborne)",
                 fontweight="bold", pad=12)
    ax.set_xlabel(LABEL_FREQ_HZ)
    ax.set_ylabel("Apparent sound reduction index R' [dB]")
    ax.set_xscale("log")
    ax.set_xlim(90, 3600)
    ax.set_ylim(8, 42)
    from matplotlib.ticker import NullFormatter
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xticks(freqs)
    ax.set_xticklabels(
        ["100", "125", "160", "200", "250", "315", "400", "500", "630", "800",
         "1k", "1.25k", "1.6k", "2k", "2.5k", "3.15k"], fontsize=8)
    ax.grid(which="major", color=COLOR_GRID, linestyle="-", alpha=0.5)
    ax.legend(loc="lower right", fontsize=9)
    save_figure(output_dir, "insulation_uncertainty_demo.png")
    plt.close()


def generate_image_source_plan(output_dir: str) -> None:
    """Plan view of the image-source lattice of the reflectogram example.

    The 7 x 5 x 3 m room with its source and receiver, and every image up
    to third order coloured by reflection order over the mirror-room grid.
    One concept: where the reflections of the reflectogram actually come
    from.
    """
    print("Generating image_source_plan...")
    from phonometry import image_source_rir

    res = image_source_rir(
        (7.0, 5.0, 3.0), (2.0, 1.6, 1.5), (5.2, 3.4, 1.7), 0.3,
        fs=16000, max_order=3,
    )
    _fig, ax = plt.subplots(figsize=(10, 6.6))
    res.plot_geometry(ax=ax, language=_LANG)
    plt.tight_layout()
    save_figure(output_dir, "image_source_plan.svg")
    plt.close()


def generate_aperture_slit_geometry(output_dir: str) -> None:
    """To-scale section of a 2 mm slit through a 100 mm wall.

    The slit of the aperture transmission-loss example: a deep, narrow gap
    whose depth resonances puncture the wall's insulation. One concept: the
    tiny geometry behind a large leak.
    """
    print("Generating aperture_slit_geometry...")
    from phonometry import plot_aperture_geometry

    _fig, ax = plt.subplots(figsize=(9.0, 6.2))
    plot_aperture_geometry(0.1, ax=ax, width=0.002, language=_LANG)
    plt.tight_layout()
    save_figure(output_dir, "aperture_slit_geometry.svg")
    plt.close()


def generate_facade_elevation_geometry(output_dir: str) -> None:
    """Composite facade elevation with the element areas to scale.

    A masonry wall of 6 m2 with a 1,5 m2 window and its 0,3 m2 roller
    shutter box, the classic composite of the facade prediction. One
    concept: the areas the composite sound reduction index weighs.
    """
    print("Generating facade_elevation_geometry...")
    from phonometry import FacadeElement, plot_facade_elements

    elements = [
        FacadeElement("Masonry wall", area=6.0, r=[50.0] * 5),
        FacadeElement("Window", area=1.5, r=[30.0] * 5),
        FacadeElement("Roller shutter box", area=0.3, r=[22.0] * 5),
    ]
    _fig, ax = plt.subplots(figsize=(9.0, 6.0))
    plot_facade_elements(elements, ax=ax, language=_LANG)
    plt.tight_layout()
    save_figure(output_dir, "facade_elevation_geometry.svg")
    plt.close()


def generate_double_wall_geometry(output_dir: str) -> None:
    """Mass-spring-mass double wall to scale.

    Two 12,5 mm plasterboard leaves (8,8 kg/m2 each) on a 100 mm stud
    cavity, the classic lightweight double wall, with its mass-air-mass
    resonance annotated. One concept: the geometry behind the double-wall
    resonance dip.
    """
    print("Generating double_wall_geometry...")
    from phonometry import mass_spring_mass_resonance, plot_double_wall_geometry

    f0 = mass_spring_mass_resonance(8.8, 8.8, 0.1)
    _fig, ax = plt.subplots(figsize=(9.0, 6.0))
    plot_double_wall_geometry(
        8.8, 8.8, 0.1, ax=ax, resonance_frequency=f0, language=_LANG,
    )
    plt.tight_layout()
    save_figure(output_dir, "double_wall_geometry.svg")
    plt.close()


def generate_open_plan_line_geometry(output_dir: str) -> None:
    """The open-plan measurement line to scale.

    Six microphones from 2 m to 16 m across the workstations with the
    distraction and privacy distances marked. One concept: the single
    line every ISO 3382-3 quantity comes from.
    """
    print("Generating open_plan_line_geometry...")
    from phonometry import plot_open_plan_geometry

    _fig, ax = plt.subplots(figsize=(10, 4.6))
    plot_open_plan_geometry(
        [2.0, 4.0, 6.0, 8.0, 12.0, 16.0], ax=ax, rd=6.5, rp=13.0,
        language=_LANG,
    )
    plt.tight_layout()
    save_figure(output_dir, "open_plan_line_geometry.svg")
    plt.close()


def generate_enclosed_space_absorption(output_dir: str) -> None:
    """EN 12354-6: absorption area and reverberation time of a room."""
    print("Generating enclosed_space_absorption.png...")
    from phonometry import enclosed_space_reverberation
    from phonometry.room.enclosed_space_absorption import OCTAVE_BANDS

    # A 5 x 4 x 3 m office (60 m3): hard plaster walls and floor; the ceiling is
    # either bare plaster or lined with an absorbing acoustic tile.
    volume = 60.0
    plaster = [0.02, 0.03, 0.03, 0.04, 0.05, 0.05, 0.05]
    tile = [0.15, 0.35, 0.65, 0.85, 0.90, 0.90, 0.85]
    walls_floor = [(54.0, plaster), (20.0, plaster)]  # walls + floor
    bare = enclosed_space_reverberation(
        [*walls_floor, (20.0, plaster)], volume, air_condition="20C_50-70")
    treated = enclosed_space_reverberation(
        [*walls_floor, (20.0, tile)], volume, air_condition="20C_50-70")

    _fig, (ax_a, ax_t) = plt.subplots(1, 2, figsize=(12.5, 5.4))
    freq = OCTAVE_BANDS
    labels = [f"{f:g}" if f < 1000 else f"{f / 1000:g}k" for f in freq]

    for res, colour, name in ((bare, COLOR_SECONDARY, "bare ceiling"),
                              (treated, COLOR_PRIMARY, "acoustic ceiling")):
        ax_a.semilogx(freq, res.absorption_area, color=colour, marker="o", label=name)
        ax_t.semilogx(freq, res.reverberation_time, color=colour, marker="o",
                      label=name)
    for ax, ylab, title in (
        (ax_a, "Equivalent absorption area $A$ [m$^2$]", "Absorption area (Formula 1)"),
        (ax_t, "Reverberation time $T$ [s]", "Reverberation time (Formula 5)"),
    ):
        ax.set_xticks(freq)
        ax.set_xticklabels(labels)
        ax.set_xlabel("Octave-band centre frequency [Hz]")
        ax.set_ylabel(ylab)
        ax.set_title(title, fontweight="bold", pad=10)
        ax.set_ylim(bottom=0.0)
        ax.grid(which="both", color=COLOR_GRID, linestyle="-", alpha=0.4)
        ax.set_axisbelow(True)
        ax.legend(loc="upper right")

    plt.tight_layout()
    save_figure(output_dir, "enclosed_space_absorption.png")
    plt.close()


def generate_room_noise_criteria(output_dir: str) -> None:
    """ANSI S12.2-2019: NC tangency rating and RC Mark II classification."""
    print("Generating room_noise_criteria.png...")
    from phonometry import noise_criterion, room_criterion
    from phonometry.room.noise_criteria import NC_CURVES, NC_INDICES, OCTAVE_BANDS

    # A ventilation-dominated room spectrum: the low-frequency bands rise well
    # above the sloped RC reference (a rumble tag under RC Mark II) while the
    # mid bands set the NC tangency.
    spectrum = np.array([62.0, 62.0, 59.0, 57.0, 52.0, 42.0, 35.0, 29.0, 24.0, 19.0])
    nc = noise_criterion(spectrum)
    rc = room_criterion(spectrum)

    _fig, (ax_nc, ax_rc) = plt.subplots(1, 2, figsize=(12.5, 5.6))

    # --- Left: NC curves + tangency rating. ---
    for row, idx in zip(NC_CURVES, NC_INDICES):
        ax_nc.plot(OCTAVE_BANDS, row, color=COLOR_GRID, lw=0.8, zorder=1)
        ax_nc.annotate(f"{idx:.0f}", (OCTAVE_BANDS[-1], row[-1]),
                       fontsize=7, color="#999999", va="center")
    ax_nc.plot(OCTAVE_BANDS, spectrum, "o-", color=COLOR_PRIMARY, zorder=3,
               label="Measured")
    gov = spectrum[OCTAVE_BANDS == nc.governing_frequency][0]
    ax_nc.plot([nc.governing_frequency], [gov], "D", color=COLOR_SECONDARY,
               ms=9, zorder=4, label=f"Tangent @ {nc.governing_frequency:g} Hz")
    ax_nc.set_xscale("log")
    ax_nc.set_xticks(list(OCTAVE_BANDS))
    ax_nc.set_xticklabels([f"{f:g}" for f in OCTAVE_BANDS], rotation=45, ha="right")
    ax_nc.set_xlabel("Octave-band center frequency [Hz]")
    ax_nc.set_ylabel("Octave-band sound pressure level [dB]")
    ax_nc.set_title(f"Noise Criteria — tangency method   NC-{nc.rating:g}",
                    fontweight="bold", pad=10)
    ax_nc.grid(which="both", axis="y", color=COLOR_GRID, linestyle="-", alpha=0.4)
    ax_nc.set_axisbelow(True)
    ax_nc.legend(loc="upper right")

    # --- Right: RC Mark II reference + rumble/hiss tolerances. ---
    ref = rc.reference_curve
    low = OCTAVE_BANDS <= 500.0
    high = OCTAVE_BANDS >= 1000.0
    ax_rc.plot(OCTAVE_BANDS, ref, "s--", color="#7f7f7f",
               label=f"Reference RC-{rc.rating}")
    ax_rc.fill_between(OCTAVE_BANDS[low], ref[low], ref[low] + 5.0, zorder=0,
                       color=theme_fill("#ff7f0e", ax_rc),
                       label="Rumble tol. (+5 dB)")
    ax_rc.fill_between(OCTAVE_BANDS[high], ref[high], ref[high] + 3.0, zorder=0,
                       color=theme_fill(COLOR_PRIMARY, ax_rc),
                       label="Hiss tol. (+3 dB)")
    ax_rc.plot(OCTAVE_BANDS, spectrum, "o-", color=COLOR_PRIMARY, zorder=3,
               label="Measured")
    ax_rc.set_xscale("log")
    ax_rc.set_xticks(list(OCTAVE_BANDS))
    ax_rc.set_xticklabels([f"{f:g}" for f in OCTAVE_BANDS], rotation=45, ha="right")
    ax_rc.set_xlabel("Octave-band center frequency [Hz]")
    ax_rc.set_ylabel("Octave-band sound pressure level [dB]")
    ax_rc.set_title(f"Room Criteria Mark II   {rc.label}",
                    fontweight="bold", pad=10)
    ax_rc.grid(which="both", axis="y", color=COLOR_GRID, linestyle="-", alpha=0.4)
    ax_rc.set_axisbelow(True)
    ax_rc.legend(loc="upper right")

    plt.tight_layout()
    save_figure(output_dir, "room_noise_criteria.png")
    plt.close()


def generate_panel_insulation_concept(output_dir: str) -> None:
    """Theoretical panel sound insulation: the four PR-I predictions."""
    print("Generating panel_insulation_concept.png...")
    from phonometry import (
        coincidence_frequency,
        composite_transmission_loss,
        double_wall_transmission_loss,
        mass_law_transmission_loss,
        mass_spring_mass_resonance,
        plate_bending_stiffness,
        radiation_efficiency,
        single_panel_transmission_loss,
    )

    bands = np.array(
        [50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000,
         1250, 1600, 2000, 2500, 3150, 4000, 5000], dtype=float
    )
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # (a) Single panel: field-incidence mass law and the coincidence dip.
    bp = plate_bending_stiffness(6.2e10, 0.006, 0.24)
    fc = coincidence_frequency(15.0, bp)
    ml = mass_law_transmission_loss(bands, 15.0, incidence="field")
    sharp = single_panel_transmission_loss(
        bands, 15.0, critical_frequency=fc, loss_factor=0.024
    )
    ax = axes[0, 0]
    ax.semilogx(bands, ml, color=COLOR_TERTIARY, ls="--", lw=1.6,
                label="field-incidence mass law")
    ax.semilogx(bands, sharp.transmission_loss, color=COLOR_PRIMARY, lw=2.0,
                marker="o", markersize=3, label="single panel R (Sharp)")
    ax.axvline(fc, color=COLOR_SECONDARY, ls=":", lw=1.2, label="$f_c$")
    ax.set_title("Single panel: mass law and coincidence",
                 fontweight="bold", pad=10)
    ax.set_ylabel("Sound reduction index R [dB]")
    ax.set_xlabel("Frequency [Hz]")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    format_frequency_axis(ax, float(bands.min()), float(bands.max()))

    # (b) Double wall: mass-spring-mass resonance and cavity gain.
    dw = double_wall_transmission_loss(bands, 12.0, 12.0, 0.075)
    single = mass_law_transmission_loss(bands, 24.0, incidence="field")
    f0 = mass_spring_mass_resonance(12.0, 12.0, 0.075)
    ax = axes[0, 1]
    ax.semilogx(bands, single, color=COLOR_TERTIARY, ls="--", lw=1.6,
                label="single leaf (total mass)")
    ax.semilogx(bands, dw.transmission_loss, color=COLOR_PRIMARY, lw=2.0,
                marker="o", markersize=3, label="double wall R")
    ax.axvline(f0, color=COLOR_SECONDARY, ls=":", lw=1.2, label="$f_0$")
    ax.set_title("Double wall: mass-spring-mass resonance",
                 fontweight="bold", pad=10)
    ax.set_ylabel("Sound reduction index R [dB]")
    ax.set_xlabel("Frequency [Hz]")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    format_frequency_axis(ax, float(bands.min()), float(bands.max()))

    # (c) Radiation efficiency of a bending plate.
    sigma = radiation_efficiency(bands, 1.5, 1.25, fc)
    ax = axes[1, 0]
    ax.loglog(bands, sigma.radiation_efficiency, color=COLOR_PRIMARY, lw=2.0,
              marker="o", markersize=3, label=r"$\sigma(f)$")
    ax.axhline(1.0, color=COLOR_FG, ls=":", lw=0.9, alpha=0.5, label="$\\sigma = 1$")
    ax.axvline(fc, color=COLOR_SECONDARY, ls=":", lw=1.2, label="$f_c$")
    ax.set_title("Radiation efficiency of a bending plate",
                 fontweight="bold", pad=10)
    ax.set_ylabel(r"Radiation efficiency $\sigma$")
    ax.set_xlabel("Frequency [Hz]")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    format_frequency_axis(ax, float(bands.min()), float(bands.max()))

    # (d) Composite wall with a small aperture (open-area cap).
    wall = sharp.transmission_loss
    open_area = 0.01
    n = bands.size
    composite = np.array([
        float(composite_transmission_loss(
            [1.0 - open_area, open_area], [wall[i], 0.0]))
        for i in range(n)
    ])
    ax = axes[1, 1]
    ax.semilogx(bands, wall, color=COLOR_PRIMARY, lw=2.0, marker="o",
                markersize=3, label="solid wall alone")
    ax.semilogx(bands, composite, color=COLOR_SECONDARY, lw=2.0, marker="s",
                markersize=3, label="wall + 1 % open slit")
    ax.axhline(10.0 * np.log10(1.0 / open_area), color=COLOR_FG, ls=":", lw=0.9,
               alpha=0.5, label="open-area limit")
    ax.set_title("Composite wall with a small aperture",
                 fontweight="bold", pad=10)
    ax.set_ylabel("Sound reduction index R [dB]")
    ax.set_xlabel("Frequency [Hz]")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    format_frequency_axis(ax, float(bands.min()), float(bands.max()))

    fig.suptitle(
        "Theoretical panel sound insulation (Bies / Hopkins / Cremer)",
        fontweight="bold", fontsize=14,
    )
    plt.tight_layout()
    save_figure(output_dir, "panel_insulation_concept.png")
    plt.close()


def generate_steady_state_field(output_dir: str) -> None:
    """Bies steady-state room field: direct, reverberant, total and rc."""
    print("Generating steady_state_field...")
    from phonometry import room

    # A 90 dB re 1 pW source in a 12 x 8 x 4 m workshop (S = 352 m^2) with a
    # mean Sabine absorption of 0.15: the total level follows the 1/r^2 direct
    # field close in and flattens onto the reverberant plateau beyond rc.
    field = room.steady_state_field(
        sound_power_level=90.0, surface_area=352.0, mean_absorption=0.15,
    )
    field.plot(language=_LANG)
    plt.gcf().set_size_inches(10, 6)
    plt.tight_layout()
    save_figure(output_dir, "steady_state_field.svg")
    plt.close()


def generate_room_parameters_bands(output_dir: str) -> None:
    """ISO 3382: per-band EDT/T20/T30 and C50/C80 of a synthetic room IR."""
    print("Generating room_parameters_bands...")
    from phonometry import room

    # A synthetic room impulse response with a frequency-dependent decay:
    # octave-band noise carriers whose T60 falls from 1.4 s at 125 Hz to
    # 0.7 s at 4 kHz, the classic behaviour of a furnished, treated room.
    fs = 48000
    rng = np.random.default_rng(3382)
    t = np.arange(int(1.6 * fs)) / fs
    t60 = {125.0: 1.4, 250.0: 1.25, 500.0: 1.1,
           1000.0: 1.0, 2000.0: 0.85, 4000.0: 0.7}
    ir = np.zeros_like(t)
    for fc, t60_band in t60.items():
        sos = scipy_signal.butter(
            4, [fc / np.sqrt(2.0), fc * np.sqrt(2.0)], btype="bandpass",
            fs=fs, output="sos",
        )
        carrier = scipy_signal.sosfilt(sos, rng.standard_normal(t.size))
        ir += carrier * np.exp(-3.0 * np.log(10.0) / t60_band * t)
    res = room.room_parameters(ir, fs)
    res.plot(language=_LANG)
    plt.gcf().set_size_inches(12.5, 5.4)
    plt.tight_layout()
    save_figure(output_dir, "room_parameters_bands.svg")
    plt.close()


def generate_extended_insulation_rating(output_dir: str) -> None:
    """ISO 717-1 Annex B enlarged-range rating of the Annex C Table C.2 wall."""
    print("Generating extended_insulation_rating...")
    from phonometry import weighted_rating_extended

    # ISO 717-1 Annex C: the 16-band R spectrum (Table C.1) enlarged to
    # 50 Hz - 5 kHz with the Table C.2 values.
    r_core = [20.4, 16.3, 17.7, 22.6, 22.4, 22.7, 24.8, 26.6,
              28.0, 30.5, 31.8, 32.5, 33.4, 33.0, 31.0, 25.5]
    freqs = [50, 63, 80, *(_THIRD_OCTAVE_16), 4000, 5000]
    ext = weighted_rating_extended([18.7, 19.2, 20.0, *r_core, 26.8, 29.2],
                                   freqs)
    assert ext.measured is not None and ext.core.shifted_reference is not None
    assert ext.core.measured is not None

    _fig, ax = plt.subplots(figsize=(10, 6.2))
    x = _band_index_axis(ax, freqs)
    core = slice(3, 19)                     # the 16 core bands 100-3150 Hz
    enlarged = theme_fill(COLOR_FG, ax)
    ax.axvspan(-0.5, 2.5, color=enlarged, zorder=0,
               label="enlarged range (Annex B)")
    ax.axvspan(18.5, 20.5, color=enlarged, zorder=0)
    ax.plot(x, ext.measured, "-o", color=COLOR_PRIMARY, linewidth=2.2,
            markersize=5, zorder=5, label="measured R")
    ax.plot(x[core], ext.core.shifted_reference, "--s", color=COLOR_FG,
            linewidth=1.8, markersize=5, zorder=5,
            label="shifted reference (100-3150 Hz)")
    unfav = ext.core.measured < ext.core.shifted_reference
    ax.fill_between(x[core], ext.core.measured, ext.core.shifted_reference,
                    where=unfav.tolist(), color=COLOR_SECONDARY, alpha=0.25,
                    interpolate=True, zorder=1,
                    label="unfavourable deviations")

    ax.set_ylabel("Sound reduction index R [dB]")
    ax.set_title("ISO 717-1 Enlarged-Range Rating (Annex B)",
                 fontweight="bold", pad=12)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=9)

    info = [
        f"Rw(C;Ctr) = {ext.rating:g}({ext.c:g};{ext.ctr:g})",
        f"C50-5000 = {ext.c_50_5000:g},  Ctr,50-5000 = {ext.ctr_50_5000:g}",
        "rating on the core bands, terms on the full range",
    ]
    ax.text(0.985, 0.03, "\n".join(info), transform=ax.transAxes,
            va="bottom", ha="right", fontsize=10, color=COLOR_FG,
            bbox={"boxstyle": "round,pad=0.5", "facecolor": COLOR_PANEL,
                  "edgecolor": COLOR_GRID})
    plt.tight_layout()
    save_figure(output_dir, "extended_insulation_rating.svg")
    plt.close()


def generate_field_airborne_insulation(output_dir: str) -> None:
    """ISO 16283-1 field airborne chain: D and DnT with the T correction."""
    print("Generating field_airborne_insulation...")
    from phonometry import airborne_insulation, weighted_rating

    # A separating wall between dwellings: source/receiving levels and the
    # measured receiving-room T per one-third-octave band.
    l1 = np.array([92.3, 93.1, 94.0, 94.4, 94.8, 95.0, 95.2, 95.4,
                   95.3, 95.1, 94.8, 94.4, 93.9, 93.3, 92.5, 91.6])
    d = np.array([38.2, 40.1, 42.6, 45.2, 47.8, 50.1, 52.3, 54.0,
                  55.6, 57.1, 58.2, 59.0, 59.6, 60.1, 60.3, 59.8])
    t2 = np.array([0.62, 0.58, 0.55, 0.53, 0.52, 0.50, 0.49, 0.48,
                   0.47, 0.46, 0.45, 0.45, 0.44, 0.43, 0.43, 0.42])
    field = airborne_insulation(l1, l1 - d, t2, area=12.5, volume=30.4)
    assert field.r_prime is not None
    w = weighted_rating(field.dnt)
    w_rp = weighted_rating(field.r_prime)

    _fig, ax = plt.subplots(figsize=(10, 6.2))
    x = _band_index_axis(ax, _THIRD_OCTAVE_16)
    ax.fill_between(x, field.d, field.dnt, color=COLOR_TERTIARY, alpha=0.18,
                    zorder=0, label="10 log10(T/T0)")
    ax.plot(x, field.d, "--o", color=COLOR_PRIMARY, linewidth=1.8,
            markersize=5, zorder=5, label="D (level difference)")
    ax.plot(x, field.dnt, "-s", color=COLOR_FG, linewidth=2.4, markersize=5,
            zorder=5, label="DnT (standardized)")

    ax.set_ylabel("Level difference [dB]")
    ax.set_title("ISO 16283-1 Field Airborne Insulation",
                 fontweight="bold", pad=12)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=9)

    info = [
        f"DnT,w(C;Ctr) = {w.rating}({w.c};{w.ctr}) dB",
        f"R'w = {w_rp.rating} dB   (S = 12.5 m², V = 30.4 m³)",
    ]
    ax.text(0.985, 0.03, "\n".join(info), transform=ax.transAxes,
            va="bottom", ha="right", fontsize=10, color=COLOR_FG,
            bbox={"boxstyle": "round,pad=0.5", "facecolor": COLOR_PANEL,
                  "edgecolor": COLOR_GRID})
    plt.tight_layout()
    save_figure(output_dir, "field_airborne_insulation.svg")
    plt.close()


def generate_facade_field_insulation(output_dir: str) -> None:
    """ISO 16283-3 façade quantities D2m, D2m,nT, D2m,n and R'45 per band."""
    print("Generating facade_field_insulation...")
    from phonometry import facade_insulation, weighted_rating

    # A dwelling façade under the 45-degree loudspeaker method: the outdoor
    # level 2 m in front, the receiving-room level and T per band.
    l1_2m = np.array([76.0, 77.0, 78.0, 78.5, 79.0, 79.0, 79.0, 79.0,
                      78.5, 78.0, 77.5, 77.0, 76.5, 76.0, 75.0, 74.0])
    d2m = np.array([24.0, 25.5, 27.0, 28.5, 30.0, 31.5, 33.0, 34.5,
                    36.0, 37.0, 38.0, 38.5, 39.0, 39.0, 38.5, 38.0])
    t2 = np.array([0.65, 0.62, 0.58, 0.55, 0.52, 0.50, 0.49, 0.48,
                   0.47, 0.46, 0.45, 0.44, 0.43, 0.43, 0.42, 0.42])
    fac = facade_insulation(l1_2m, l1_2m - d2m, t2, volume=32.0, area=10.8,
                            surface_level=l1_2m + 3.0, method="loudspeaker",
                            frequencies=np.asarray(_THIRD_OCTAVE_16, float))
    assert fac.d_2m_n is not None and fac.r_prime is not None
    w = weighted_rating(fac.d_2m_nt)

    _fig, ax = plt.subplots(figsize=(10, 6.2))
    x = _band_index_axis(ax, _THIRD_OCTAVE_16)
    ax.plot(x, fac.d_2m_nt, "-s", color=COLOR_FG, linewidth=2.4, markersize=5,
            zorder=6, label="D2m,nT (standardized)")
    ax.plot(x, fac.d_2m, "--o", color=COLOR_PRIMARY, linewidth=1.8,
            markersize=5, zorder=5, label="D2m = L1,2m - L2")
    ax.plot(x, fac.d_2m_n, ":", color=COLOR_SECONDARY, linewidth=1.8,
            marker=".", zorder=5, label="D2m,n (normalized)")
    ax.plot(x, fac.r_prime, "-.", color=COLOR_TERTIARY, linewidth=1.8,
            marker="^", markersize=5, zorder=5, label="R'45° (element)")

    ax.set_ylabel("Level difference / reduction index [dB]")
    ax.set_title("ISO 16283-3 Field Facade Insulation",
                 fontweight="bold", pad=12)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=9)

    info = [
        f"Dls,2m,nT,w(C;Ctr) = {w.rating}({w.c};{w.ctr}) dB",
        "45° loudspeaker method (-1.5 dB on R')",
    ]
    ax.text(0.985, 0.03, "\n".join(info), transform=ax.transAxes,
            va="bottom", ha="right", fontsize=10, color=COLOR_FG,
            bbox={"boxstyle": "round,pad=0.5", "facecolor": COLOR_PANEL,
                  "edgecolor": COLOR_GRID})
    plt.tight_layout()
    save_figure(output_dir, "facade_field_insulation.svg")
    plt.close()


def generate_survey_impact_insulation(output_dir: str) -> None:
    """ISO 10052 survey impact method: Li -> L'nT with the k correction."""
    print("Generating survey_impact_insulation...")
    from phonometry import reverberation_index, survey_impact_insulation

    bands = [125.0, 250.0, 500.0, 1000.0, 2000.0]
    # Tapping machine on the floor above: octave-band receiving-room levels
    # and the measured receiving-room reverberation time.
    li = np.array([66.0, 64.0, 62.0, 60.0, 55.0])
    k = reverberation_index([0.70, 0.60, 0.50, 0.45, 0.40])
    res = survey_impact_insulation(li, k, volume=50.0)
    assert res.rating is not None

    _fig, ax = plt.subplots(figsize=(10, 6.2))
    x = _band_index_axis(ax, bands, fontsize=10)
    ax.fill_between(x, res.l_i, res.l_nt, color=COLOR_TERTIARY, alpha=0.18,
                    zorder=0, label="-k = -10 log10(T/T0)")
    ax.plot(x, res.l_i, "--o", color=COLOR_PRIMARY, linewidth=1.8,
            markersize=6, zorder=5, label="Li (impact level)")
    ax.plot(x, res.l_nt, "-s", color=COLOR_FG, linewidth=2.4, markersize=6,
            zorder=5, label="L'nT (standardized)")

    ax.set_ylabel("Impact sound pressure level [dB]")
    ax.set_title("ISO 10052 Survey Method: Impact Sound",
                 fontweight="bold", pad=12)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", fontsize=9)

    info = [
        f"L'nT,w(CI) = {res.rating.rating}({res.rating.ci}) dB",
        "note the minus sign: a live room lowers L'nT",
    ]
    ax.text(0.985, 0.97, "\n".join(info), transform=ax.transAxes,
            va="top", ha="right", fontsize=10, color=COLOR_FG,
            bbox={"boxstyle": "round,pad=0.5", "facecolor": COLOR_PANEL,
                  "edgecolor": COLOR_GRID})
    plt.tight_layout()
    save_figure(output_dir, "survey_impact_insulation.svg")
    plt.close()


def generate_lab_insulation_result(output_dir: str) -> None:
    """ISO 10140 laboratory quantities: the airborne R and the impact Ln."""
    print("Generating lab_insulation_result...")
    from phonometry import lab_airborne_insulation, lab_impact_insulation

    # ISO 717-1 Annex C wall measured in an ISO 10140 suite (S = 10 m2,
    # V = 50 m3, T = 0.8 s) and the ISO 717-2 Annex C floor tapping levels.
    r = np.array([20.4, 16.3, 17.7, 22.6, 22.4, 22.7, 24.8, 26.6,
                  28.0, 30.5, 31.8, 32.5, 33.4, 33.0, 31.0, 25.5])
    l1 = np.full(16, 90.0)
    t2 = np.full(16, 0.8)
    lab = lab_airborne_insulation(l1, l1 - r, t2, area=10.0, volume=50.0)
    li = np.array([62.1, 63.2, 63.5, 66.2, 68.5, 70.0, 71.7, 73.1,
                   73.8, 73.5, 73.8, 73.3, 73.1, 73.0, 72.4, 71.2])
    imp = lab_impact_insulation(li, t2, volume=50.0)
    assert lab.rating is not None and imp.rating is not None

    _fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.6))
    for ax in (ax1, ax2):
        _band_index_axis(ax, _THIRD_OCTAVE_16, fontsize=7)
        ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5, zorder=0)
        ax.set_axisbelow(True)
    x = np.arange(16)

    ax1.plot(x, lab.r, "-o", color=COLOR_PRIMARY, linewidth=2.2,
             markersize=5, zorder=5, label="measured R")
    assert lab.rating.shifted_reference is not None
    ax1.plot(x, lab.rating.shifted_reference, "--s", color=COLOR_FG,
             linewidth=1.6, markersize=4, zorder=5, label="shifted reference")
    ax1.set_ylabel("Sound reduction index R [dB]")
    ax1.set_title(f"Airborne: Rw(C;Ctr) = {lab.rating.rating}"
                  f"({lab.rating.c};{lab.rating.ctr}) dB", fontsize=11)
    ax1.legend(loc="upper left", fontsize=9)

    ax2.plot(x, imp.l_n, "-o", color=COLOR_SECONDARY, linewidth=2.2,
             markersize=5, zorder=5, label="normalized Ln")
    assert imp.rating.shifted_reference is not None
    ax2.plot(x, imp.rating.shifted_reference, "--s", color=COLOR_FG,
             linewidth=1.6, markersize=4, zorder=5, label="shifted reference")
    ax2.set_ylabel("Impact sound pressure level Ln [dB]")
    ax2.set_title(f"Impact: Ln,w(CI) = {imp.rating.rating}"
                  f"({imp.rating.ci}) dB", fontsize=11)
    ax2.legend(loc="upper right", fontsize=9)

    plt.suptitle("ISO 10140 Laboratory Insulation (flanking suppressed)",
                 fontweight="bold")
    plt.tight_layout()
    save_figure(output_dir, "lab_insulation_result.svg")
    plt.close()


def generate_intensity_element_insulation(output_dir: str) -> None:
    """ISO 15186-1 element-normalized level difference of a small element."""
    print("Generating intensity_element_insulation...")
    from phonometry import intensity_element_normalized_difference

    # A trickle ventilator in a masonry wall: source-room SPL 85 dB and the
    # normal intensity level over the Sm = 12 m2 measurement surface.
    l_in = np.array([57.9, 62.0, 60.6, 55.7, 55.9, 55.6, 53.5, 51.7,
                     50.3, 47.8, 46.5, 45.8, 44.9, 45.3, 47.3, 52.8])
    res = intensity_element_normalized_difference(
        np.full(16, 85.0), l_in, measurement_area=12.0, n=1
    )
    assert res.rating is not None
    assert res.rating.shifted_reference is not None
    assert res.rating.measured is not None

    _fig, ax = plt.subplots(figsize=(10, 6.2))
    x = _band_index_axis(ax, _THIRD_OCTAVE_16)
    ax.plot(x, res.d_i_n_e, "-o", color=COLOR_PRIMARY, linewidth=2.2,
            markersize=5, zorder=5, label="DI,n,e (element)")
    ax.plot(x, res.rating.shifted_reference, "--s", color=COLOR_FG,
            linewidth=1.8, markersize=5, zorder=5, label="shifted reference")
    unfav = res.rating.measured < res.rating.shifted_reference
    ax.fill_between(x, res.rating.measured, res.rating.shifted_reference,
                    where=unfav.tolist(), color=COLOR_SECONDARY, alpha=0.25,
                    interpolate=True, zorder=1,
                    label="unfavourable deviations")

    ax.set_ylabel("Element normalized level difference [dB]")
    ax.set_title("ISO 15186-1 Small-Element Insulation by Intensity",
                 fontweight="bold", pad=12)
    ax.margins(y=0.1)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=9)

    info = [
        (f"DI,n,e,w(C;Ctr) = {res.rating.rating}"
         f"({res.rating.c};{res.rating.ctr}) dB"),
        "DI,n,e = Lp1 - 6 - [LIn + 10 log10(Sm/A0)] + 10 log10 N",
    ]
    ax.text(0.985, 0.03, "\n".join(info), transform=ax.transAxes,
            va="bottom", ha="right", fontsize=10, color=COLOR_FG,
            bbox={"boxstyle": "round,pad=0.5", "facecolor": COLOR_PANEL,
                  "edgecolor": COLOR_GRID})
    plt.tight_layout()
    save_figure(output_dir, "intensity_element_insulation.svg")
    plt.close()


def generate_flanking_level_difference(output_dir: str) -> None:
    """ISO 10848 overall flanking descriptor Dn,f with its Dn,f,w rating."""
    print("Generating flanking_level_difference...")
    from phonometry import normalized_flanking_level_difference

    # A lightweight junction measured in the laboratory: source-room level,
    # receiving-room level over the flanking path and the absorption area.
    l1 = np.full(16, 80.0)
    dnf_target = np.array([48, 49, 50, 51, 52, 54, 55, 57,
                           58, 59, 60, 61, 62, 63, 64, 65], dtype=float)
    res = normalized_flanking_level_difference(
        l1, l1 - dnf_target, absorption_area=np.full(16, 10.0)
    )
    assert res.rating is not None
    assert res.rating.shifted_reference is not None
    assert res.rating.measured is not None

    _fig, ax = plt.subplots(figsize=(10, 6.2))
    x = _band_index_axis(ax, _THIRD_OCTAVE_16)
    ax.plot(x, res.d_n_f, "-o", color=COLOR_PRIMARY, linewidth=2.2,
            markersize=5, zorder=5, label="Dn,f (flanking)")
    ax.plot(x, res.rating.shifted_reference, "--s", color=COLOR_FG,
            linewidth=1.8, markersize=5, zorder=5, label="shifted reference")
    unfav = res.rating.measured < res.rating.shifted_reference
    ax.fill_between(x, res.rating.measured, res.rating.shifted_reference,
                    where=unfav.tolist(), color=COLOR_SECONDARY, alpha=0.25,
                    interpolate=True, zorder=1,
                    label="unfavourable deviations")

    ax.set_ylabel("Normalized flanking level difference [dB]")
    ax.set_title("ISO 10848 Airborne Flanking Transmission",
                 fontweight="bold", pad=12)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=9)

    info = [
        (f"Dn,f,w(C;Ctr) = {res.rating.rating}"
         f"({res.rating.c};{res.rating.ctr}) dB"),
        "Dn,f = L1 - L2 - 10 log10(A/A0)",
    ]
    ax.text(0.985, 0.03, "\n".join(info), transform=ax.transAxes,
            va="bottom", ha="right", fontsize=10, color=COLOR_FG,
            bbox={"boxstyle": "round,pad=0.5", "facecolor": COLOR_PANEL,
                  "edgecolor": COLOR_GRID})
    plt.tight_layout()
    save_figure(output_dir, "flanking_level_difference.svg")
    plt.close()


def generate_impact_prediction_terms(output_dir: str) -> None:
    """EN 12354-2 Annex E.3 impact prediction as its Formula (21) terms."""
    print("Generating impact_prediction_terms...")
    from phonometry import (
        equivalent_impact_level,
        impact_flanking_correction,
        predicted_impact_insulation,
    )

    # EN 12354-2 Annex E.3: a 0.14 m concrete floor (m' = 322 kg/m2) with a
    # floating floor (delta-Lw = 33 dB), mean flanking mass 145 kg/m2.
    ln_eq = float(equivalent_impact_level(322.0))
    k = float(impact_flanking_correction(322.0, 145.0))
    imp = predicted_impact_insulation(ln_w_eq=ln_eq, delta_l_w=33.0,
                                      k_correction=k)

    labels = ["$L_{n,w,eq}$", r"$-\Delta L_w$", "$+K$", "$L'_{n,w}$"]
    values = [imp.ln_w_eq, -imp.delta_l_w, imp.k_correction, imp.l_prime_n_w]
    # COLOR_MUTED rather than the gridline grey for the starting term: a bar
    # is a read value, so it has to hold up against the page on both themes.
    colors = [COLOR_MUTED, COLOR_TERTIARY, COLOR_SECONDARY, COLOR_PRIMARY]

    _fig, ax = plt.subplots(figsize=(10, 6.2))
    bars = ax.bar(np.arange(4), values, color=colors, zorder=5)
    ax.axhline(0.0, color=COLOR_FG, linewidth=0.8)
    ax.set_xticks(np.arange(4))
    ax.set_xticklabels(labels, fontsize=12)
    for bar, value in zip(bars, values):
        ax.annotate(f"{value:+.1f}",
                    xy=(bar.get_x() + bar.get_width() / 2.0, value),
                    xytext=(0, 5 if value >= 0 else -14),
                    textcoords="offset points", ha="center", fontsize=10,
                    color=COLOR_FG)
    ax.set_ylim(min(values) - 9.0, max(values) + 9.0)

    ax.set_ylabel("Level / correction [dB]")
    ax.set_title("EN 12354-2 Impact Sound Prediction (Annex E.3)",
                 fontweight="bold", pad=12)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5, axis="y", zorder=0)
    ax.set_axisbelow(True)

    info = [
        "L'n,w = Ln,w,eq - ΔLw + K",
        f"Ln,w,eq = 164 - 35 log10(m'/m'0) = {ln_eq:.1f} dB",
        f"L'n,w = {imp.l_prime_n_w:.1f} dB → 45 dB",
    ]
    ax.text(0.985, 0.97, "\n".join(info), transform=ax.transAxes,
            va="top", ha="right", fontsize=10, color=COLOR_FG,
            bbox={"boxstyle": "round,pad=0.5", "facecolor": COLOR_PANEL,
                  "edgecolor": COLOR_GRID})
    plt.tight_layout()
    save_figure(output_dir, "impact_prediction_terms.svg")
    plt.close()


def generate_detailed_prediction_paths(output_dir: str) -> None:
    """ISO 12354-1 Annex L: which transmission path dominates each band."""
    print("Generating detailed_prediction_paths...")
    import sys
    from pathlib import Path

    tests = str(Path(__file__).resolve().parent.parent / "tests")
    if tests not in sys.path:
        sys.path.insert(0, tests)
    import iso12354_building as bld

    from phonometry import (
        detailed_airborne_prediction,
        direct_reduction_index,
        floating_floor_improvement,
        in_situ_element,
    )

    # The heavy homogeneous building of ISO 12354-1:2017 Annex L: a 220 mm
    # concrete separating floor with a floating floor, two 365 mm AAC external
    # walls and two 200 mm calcium-silicate internal walls. The building and
    # its twelve flanking paths come from the shared test fixture, which the
    # conformance report reads too, so the figure cannot drift from the rows.
    bands = np.array([50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630,
                      800, 1000, 1250, 1600, 2000, 2500, 3150], dtype=float)
    situ = {k: in_situ_element(e, bands) for k, e in bld.elements().items()}
    delta = floating_floor_improvement(
        bands, resonance_frequency=bld.floating_floor_resonance()
    )
    res = detailed_airborne_prediction(
        bands,
        direct_index=direct_reduction_index(
            situ["floor"].sound_reduction_index, delta_r_source=delta),
        flanking_paths=bld.airborne_paths(situ, delta),
    )
    assert res.rating is not None

    _fig, ax = plt.subplots(figsize=(10, 6.2))
    x = _band_index_axis(ax, bands)
    order = list(np.argsort(-res.fractions.max(axis=1)))
    palette = [COLOR_PRIMARY, COLOR_SECONDARY, COLOR_TERTIARY, "#9467bd",
               "#aec7e8", "#ff9896"]
    bottom = np.zeros(x.size)
    for colour, k in zip(palette, order[:6]):
        share = 100.0 * res.fractions[k]
        ax.bar(x, share, bottom=bottom, width=0.85, color=colour,
               edgecolor="none", zorder=2, label=res.paths[k].label)
        bottom = bottom + share
    rest = order[6:]
    if rest:
        share = 100.0 * res.fractions[rest].sum(axis=0)
        # The pooled segment is de-emphasised *data*, so it takes the mid grey
        # that reads on both pages, not the gridline grey that hides in one.
        ax.bar(x, share, bottom=bottom, width=0.85, color=COLOR_MUTED,
               edgecolor="none", zorder=2, label="other paths")
    ax.set_ylim(0.0, 100.0)
    ax.set_ylabel("Share of transmitted energy [%]")
    ax.set_title("ISO 12354-1 Detailed Model: Dominant Path per Band (Annex L)",
                 fontweight="bold", pad=12)
    ax.set_axisbelow(True)

    twin = ax.twinx()
    twin.plot(x, res.r_prime, "-o", color=COLOR_FG, linewidth=2.0,
              markersize=4, zorder=5, label="R' (apparent)")
    twin.set_ylabel("Apparent sound reduction index R' [dB]")
    handles, labels = ax.get_legend_handles_labels()
    extra, extra_labels = twin.get_legend_handles_labels()
    ax.legend(handles + extra, labels + extra_labels, loc="upper left",
              fontsize=9, ncol=3)

    info = [
        "R' = -10 log10(Σ 10^(-Rij/10))",
        f"R'w (C; Ctr) = {res.rating.rating} ({res.rating.c}; {res.rating.ctr}) dB",
    ]
    ax.text(0.985, 0.03, "\n".join(info), transform=ax.transAxes,
            va="bottom", ha="right", fontsize=10, color=COLOR_FG,
            bbox={"boxstyle": "round,pad=0.5", "facecolor": COLOR_PANEL,
                  "edgecolor": COLOR_GRID})
    plt.tight_layout()
    save_figure(output_dir, "detailed_prediction_paths.svg")
    plt.close()


def generate_radiated_power_outdoor(output_dir: str) -> None:
    """EN 12354-4 Annex G radiated sound power of a wall with a door."""
    print("Generating radiated_power_outdoor...")
    from phonometry import FacadeElement, radiated_sound_power

    # EN 12354-4 Annex G, side 1: a 10 x 20 m concrete wall segment with a
    # 6 x 4 m industrial door, inside level Lp,in and Cd = -5 dB.
    bands = [63, 125, 250, 500, 1000, 2000, 4000, 8000]
    seg = radiated_sound_power(
        [FacadeElement("wall", area=176.0, r=[32, 36, 36, 33, 39, 49, 57, 63]),
         FacadeElement("door", area=24.0, r=[21, 23, 28, 30, 30, 30, 30, 30])],
        lp_in=[70, 74, 76, 72, 70, 67, 62, 57], area=200.0, c_d=-5.0,
        r_prime_cap=40.0, octave_bands=bands)
    assert seg.l_w_dba is not None

    x = np.arange(len(bands))
    _fig, ax = plt.subplots(figsize=(10, 6.2))
    ax.bar(x, seg.l_w, color=COLOR_PRIMARY, alpha=0.85, zorder=5,
           label="radiated $L_W$ per octave")
    ax.axhline(seg.l_w_dba, color=COLOR_SECONDARY, linestyle="--",
               linewidth=1.6, zorder=6,
               label=f"$L_{{WA}}$ = {seg.l_w_dba:.1f} dB(A)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{b:g}" for b in bands])
    ax.set_xlabel(LABEL_FREQ_HZ)
    ax.set_ylabel("Radiated sound power level [dB re 1 pW]")
    ax.set_ylim(0.0, float(np.max(seg.l_w)) * 1.35)
    ax.set_title("EN 12354-4 Radiated Sound Power (Annex G)",
                 fontweight="bold", pad=12)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5, axis="y", zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=9)

    info = [
        "LW = Lp,in + Cd - R' + 10 log10(S/S0)",
        "wall 176 m² + industrial door 24 m², Cd = -5 dB",
    ]
    ax.text(0.015, 0.97, "\n".join(info), transform=ax.transAxes,
            va="top", ha="left", fontsize=10, color=COLOR_FG, zorder=10,
            bbox={"boxstyle": "round,pad=0.5", "facecolor": COLOR_PANEL,
                  "edgecolor": COLOR_GRID})
    plt.tight_layout()
    save_figure(output_dir, "radiated_power_outdoor.svg")
    plt.close()


def generate_single_panel_rating(output_dir: str) -> None:
    """Sharp single-panel R(f) prediction rated per ISO 717-1 (6 mm glass)."""
    print("Generating single_panel_rating...")
    from phonometry import (
        coincidence_frequency,
        plate_bending_stiffness,
        single_panel_transmission_loss,
    )

    # 6 mm float glass: E = 62 GPa, rho = 2500 kg/m3, nu = 0.24, eta = 0.024.
    bands = np.asarray(_THIRD_OCTAVE_16, dtype=float)
    mass = 2500.0 * 0.006
    bp = plate_bending_stiffness(6.2e10, 0.006, 0.24)
    fc = float(coincidence_frequency(mass, bp))
    res = single_panel_transmission_loss(bands, mass, critical_frequency=fc,
                                         loss_factor=0.024)
    w = res.rating()
    assert w.shifted_reference is not None and w.measured is not None

    _fig, ax = plt.subplots(figsize=(10, 6.2))
    x = _band_index_axis(ax, _THIRD_OCTAVE_16)
    ax.plot(x, res.transmission_loss, "-o", color=COLOR_PRIMARY,
            linewidth=2.2, markersize=5, zorder=5, label="predicted R (Sharp)")
    ax.plot(x, w.shifted_reference, "--s", color=COLOR_FG, linewidth=1.8,
            markersize=5, zorder=5, label="shifted reference")
    unfav = w.measured < w.shifted_reference
    ax.fill_between(x, w.measured, w.shifted_reference, where=unfav.tolist(),
                    color=COLOR_SECONDARY, alpha=0.25, interpolate=True,
                    zorder=1, label="unfavourable deviations")
    idx_fc = float(np.interp(np.log10(fc), np.log10(bands), x))
    ax.axvline(idx_fc, color=COLOR_TERTIARY, linestyle=":", linewidth=1.6,
               zorder=4, label=f"coincidence fc = {fc:.0f} Hz")

    ax.set_ylabel("Sound reduction index R [dB]")
    ax.set_title("Predicted Single-Panel Insulation Rated per ISO 717-1",
                 fontweight="bold", pad=12)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=9)

    info = [
        f"Rw(C;Ctr) = {w.rating}({w.c};{w.ctr}) dB",
        "6 mm float glass, m'' = 15 kg/m², η = 0.024",
    ]
    ax.text(0.985, 0.03, "\n".join(info), transform=ax.transAxes,
            va="bottom", ha="right", fontsize=10, color=COLOR_FG,
            bbox={"boxstyle": "round,pad=0.5", "facecolor": COLOR_PANEL,
                  "edgecolor": COLOR_GRID})
    plt.tight_layout()
    save_figure(output_dir, "single_panel_rating.svg")
    plt.close()


def generate_plateau_transmission_loss(output_dir: str) -> None:
    """Plateau-method TL estimate against the full physical model."""
    print("Generating plateau_transmission_loss...")
    from phonometry import (
        plateau_transmission_loss,
        single_panel_transmission_loss,
    )

    # 6 mm float glass, 2.47 kg/m2 per mm (Norton Table 3.1); its critical
    # frequency from the same book's problem 3.13 is 2033 Hz.
    thickness_mm, f_c, eta = 6.0, 2033.0, 0.02
    mass = 2.47 * thickness_mm
    nominal = [*_THIRD_OCTAVE_16, 4000, 5000, 6300, 8000, 10000]
    bands = np.asarray(nominal, dtype=float)
    quick = plateau_transmission_loss(bands, material="glass",
                                      thickness_mm=thickness_mm,
                                      field_correction=5.5)
    physical = single_panel_transmission_loss(bands, mass,
                                              critical_frequency=f_c,
                                              loss_factor=eta)

    _fig, ax = plt.subplots(figsize=(10, 6.2))
    x = _band_index_axis(ax, nominal)
    assert quick.plateau_start is not None and quick.plateau_end is not None
    idx_a = float(np.interp(np.log10(quick.plateau_start), np.log10(bands), x))
    idx_b = float(np.interp(np.log10(quick.plateau_end), np.log10(bands), x))
    ax.axvspan(idx_a, idx_b, color=theme_fill(COLOR_SECONDARY, ax), lw=0, zorder=0,
               label="coincidence plateau (A to B)")
    ax.plot(x, physical.transmission_loss, "-o", color=COLOR_PRIMARY,
            linewidth=2.2, markersize=5, zorder=5,
            label="physical model (mass law + coincidence + damping)")
    ax.plot(x, quick.transmission_loss, "--s", color=COLOR_SECONDARY,
            linewidth=2.0, markersize=5, zorder=5,
            label="plateau estimate (Norton Table 3.1)")
    idx_fc = float(np.interp(np.log10(f_c), np.log10(bands), x))
    ax.axvline(idx_fc, color=COLOR_TERTIARY, linestyle=":", linewidth=1.6,
               zorder=4, label="critical frequency fc")

    ax.set_ylabel("Transmission loss TL [dB]")
    ax.set_title("Plateau Estimate Against the Physical Panel Model",
                 fontweight="bold", pad=12)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=9)

    panel = "#f0f2f5" if COLOR_FG == "black" else "#1c2128"
    info = [
        f"6 mm float glass, m'' = {mass:.1f} kg/m², η = {eta:g}",
        (f"plateau height 27 dB, B/A = 10 → A = {quick.plateau_start:.0f} Hz, "
         f"B = {quick.plateau_end:.0f} Hz"),
        "identical below A; the plateau replaces the whole coincidence region",
    ]
    ax.text(0.985, 0.03, "\n".join(info), transform=ax.transAxes,
            va="bottom", ha="right", fontsize=9, color=COLOR_FG,
            bbox={"boxstyle": "round,pad=0.5", "facecolor": panel,
                  "edgecolor": COLOR_GRID})
    plt.tight_layout()
    save_figure(output_dir, "plateau_transmission_loss.svg")
    plt.close()


def generate_orthotropic_transmission_loss(output_dir: str) -> None:
    """Corrugating a sheet flattens its transmission loss (Vigran 6.5.3).

    One concept: the same steel sheet, flat and corrugated. The flat plate has
    a single coincidence frequency far above the audio range and follows the
    mass law; corrugating it drags the lower coincidence frequency down by more
    than a decade and opens a range over which R collapses, despite the 9 %
    extra mass the developed length adds.
    """
    print("Generating orthotropic_transmission_loss...")
    from phonometry import (
        coincidence_frequency,
        corrugated_plate_mass_factor,
        corrugated_plate_stiffness,
        orthotropic_critical_frequencies,
        orthotropic_transmission_loss,
        plate_bending_stiffness,
        single_panel_transmission_loss,
    )

    # Vigran's worked example (printed p. 96): 1 mm steel, corrugation of total
    # height 20 mm (H = 10 mm) at a 100 mm pitch, E = 2,1e11 Pa, nu = 0,3.
    modulus, thickness, poisson, eta = 2.1e11, 1.0e-3, 0.3, 0.011
    amplitude, pitch, mass_flat = 0.010, 0.100, 7.8
    flat_b = plate_bending_stiffness(modulus, thickness, poisson)
    flat_fc = coincidence_frequency(mass_flat, flat_b)
    b_x, b_z, _b_xz = corrugated_plate_stiffness(
        thickness, amplitude, pitch, youngs_modulus=modulus,
        poisson_ratio=poisson,
    )
    mass_corr = mass_flat * corrugated_plate_mass_factor(amplitude, pitch)
    fc1, fc2 = orthotropic_critical_frequencies(mass_corr, b_x, b_z)

    nominal = [*_THIRD_OCTAVE_16, 4000, 5000, 6300, 8000, 10000, 12500, 16000]
    bands = np.asarray(nominal, dtype=float)
    flat = single_panel_transmission_loss(
        bands, mass_flat, critical_frequency=flat_fc, loss_factor=eta,
    )
    corrugated = orthotropic_transmission_loss(
        bands, mass_corr, critical_frequency_lower=fc1,
        critical_frequency_upper=fc2, loss_factor=eta,
    )
    heckl = orthotropic_transmission_loss(
        bands, mass_corr, critical_frequency_lower=fc1,
        critical_frequency_upper=fc2, method="heckl",
    )

    _fig, ax = plt.subplots(figsize=(10, 6.2))
    x = _band_index_axis(ax, nominal)
    idx_1 = float(np.interp(np.log10(fc1), np.log10(bands), x))
    idx_2 = float(np.interp(np.log10(fc2), np.log10(bands), x))
    ax.axvspan(idx_1, idx_2, color=theme_fill(COLOR_TERTIARY, ax), lw=0, zorder=0,
               label=r"coincidence range $f_{c1}$ to $f_{c2}$")
    ax.plot(x, flat.transmission_loss, "-o", color=COLOR_PRIMARY,
            linewidth=2.2, markersize=5, zorder=5,
            label=r"flat 1 mm sheet (isotropic, single $f_c$)")
    ax.plot(x, corrugated.transmission_loss, "-s", color=COLOR_SECONDARY,
            linewidth=2.2, markersize=5, zorder=5,
            label="corrugated sheet (orthotropic, diffuse-field integral)")
    ax.plot(x, heckl.transmission_loss, "--", color=COLOR_TERTIARY,
            linewidth=1.8, zorder=4, label="Heckl's approximation")

    ax.set_ylabel("Transmission loss TL [dB]")
    ax.set_title("Corrugating a Sheet Flattens Its Sound Reduction Index",
                 fontweight="bold", pad=12)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=9)

    panel = "#f0f2f5" if COLOR_FG == "black" else "#1c2128"
    worst = int(np.argmax(flat.transmission_loss - corrugated.transmission_loss))
    penalty = float(
        flat.transmission_loss[worst] - corrugated.transmission_loss[worst]
    )
    # Plain symbol names, not mathtext: the Spanish variant rewrites decimal
    # points to commas everywhere except in mathtext strings.
    info = [
        (f"1 mm steel sheet, m'' = {mass_flat:.1f} kg/m², flat "
         f"fc = {flat_fc / 1000.0:.1f} kHz"),
        (f"corrugated H = 10 mm, L = 100 mm, m'' = {mass_corr:.1f} kg/m², "
         f"fc1 = {fc1:.0f} Hz, fc2 = {fc2 / 1000.0:.1f} kHz"),
        (f"worst penalty {penalty:.0f} dB at {nominal[worst]:g} Hz, "
         "for a stiffer and only 9 % heavier panel"),
    ]
    ax.text(0.985, 0.03, "\n".join(info), transform=ax.transAxes,
            va="bottom", ha="right", fontsize=9, color=COLOR_FG,
            bbox={"boxstyle": "round,pad=0.5", "facecolor": panel,
                  "edgecolor": COLOR_GRID})
    plt.tight_layout()
    save_figure(output_dir, "orthotropic_transmission_loss.svg")
    plt.close()


def generate_dbhr_global_index(output_dir: str) -> None:
    """CTE DB-HR global index R'A of a measured wall (Manual Ejemplo 7.2).

    The published eighteen-band apparent sound reduction index R' of a field
    test, weighted with the normalised pink-noise spectrum of DB-HR Annex A
    Table A.5. The energy sum of the per-band transmitted level L_Ar,i - R'_i
    sets the global index, R'A = 51,4 dBA.
    """
    print("Generating dbhr_global_index.png...")
    from phonometry import ra

    r_prime = [
        36.2, 41.5, 36.9, 40.4, 44.7, 42.4, 45.7, 46.1, 47.1,
        52.3, 54.3, 57.5, 57.8, 57.3, 59.0, 62.8, 64.7, 65.3,
    ]

    _, ax = plt.subplots(figsize=(10, 6))
    # The result's own .plot() draws the band insulation as bars and the
    # weighted per-band transmitted level as a line on a 1k/2k-labelled axis.
    ra(r_prime).plot(ax=ax, language=_LANG)
    save_figure(output_dir, "dbhr_global_index.png")
    plt.close()
