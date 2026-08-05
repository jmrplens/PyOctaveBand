#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Figures for the room-acoustics guides: the sound field inside one room.

The impulse response of a room and the ISO 3382 parameters read from it, the
ISO 18233 signals that recover it, the modes and the image sources behind it,
the absorption area and reverberation time predicted from the surfaces, the
decay of speech across an open plan, and the criteria that rate the background
noise the room is left with. Everything here is embedded by a page under
``buildings/rooms/``.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal as scipy_signal

from phonometry._plot.common import format_frequency_axis, theme_fill

from .i18n import _LANG
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
