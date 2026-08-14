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

from .i18n import _LANG, _fmt_minus
from .theme import (
    COLOR_FG,
    COLOR_GRID,
    COLOR_MUTED,
    COLOR_PANEL,
    COLOR_PRIMARY,
    COLOR_SECONDARY,
    COLOR_TERTIARY,
    LABEL_FREQ_HZ,
    save_figure,
    series_colors,
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
        ax.text(1.40, target + 0.8, f"{_fmt_minus(target, '.0f')} dB", ha="left",
                va="bottom", fontsize=8, color=COLOR_FG, alpha=0.85)

    ax.text(0.12, -7.0, "EDT slope", fontsize=8, color="#9467bd", rotation=0)
    facecolor = plt.rcParams["axes.facecolor"]
    ax.text(0.04, 0.06,
            f"EDT = {edt:.2f} s\nT20 = {t20:.2f} s\nT30 = {t30:.2f} s",
            transform=ax.transAxes, va="bottom", ha="left", fontsize=11,
            bbox={"boxstyle": "round", "facecolor": facecolor,
                  "edgecolor": COLOR_FG, "alpha": 0.85})

    ax.set_title("Schroeder Integration and Reverberation Time (ISO 3382)",
                 pad=12)
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
    ax_sw.set_title("Exponential sine sweep — waveform")
    ax_sw.set_xlabel("Time [s]")
    ax_sw.set_ylabel("Amplitude")
    ax_sw.set_xlim(0.0, secs)
    ax_sw.set_ylim(-1.2, 1.2)
    ax_sw.grid(color=COLOR_GRID, linestyle="--", alpha=0.5)

    # Sweep spectrogram: the exponential frequency rise.
    ax_sp.specgram(sweep, NFFT=1024, Fs=fs, noverlap=512, cmap="magma")
    ax_sp.set_title("Sweep spectrogram (exponential rise)")
    ax_sp.set_xlabel("Time [s]")
    ax_sp.set_ylabel("Frequency [Hz]")
    ax_sp.set_ylim(0.0, fs / 2)

    # MLS: first samples of the bipolar sequence.
    show = 100
    ax_ml.step(np.arange(show), mls[:show], where="mid", color=COLOR_PRIMARY,
               linewidth=1.2)
    ax_ml.set_title(f"MLS — first {show} of {mls.size} samples")
    ax_ml.set_xlabel("Sample")
    ax_ml.set_ylabel("Amplitude")
    ax_ml.set_ylim(-1.4, 1.4)
    ax_ml.grid(color=COLOR_GRID, linestyle="--", alpha=0.5)

    # MLS magnitude spectrum: essentially flat (white excitation).
    spec = np.abs(np.fft.rfft(mls))
    freqs = np.fft.rfftfreq(mls.size, d=1.0 / fs)
    ax_ms.semilogx(freqs[1:], 20.0 * np.log10(spec[1:] / np.median(spec[1:])),
                   color=COLOR_SECONDARY, linewidth=0.7)
    ax_ms.set_title("MLS magnitude spectrum (flat)")
    ax_ms.set_xlabel("Frequency [Hz]")
    ax_ms.set_ylabel("Magnitude [dB]")
    ax_ms.set_xlim(20.0, fs / 2)
    format_frequency_axis(ax_ms, 20.0, fs / 2)
    ax_ms.set_ylim(-12.0, 12.0)
    ax_ms.grid(which="both", color=COLOR_GRID, linestyle="--", alpha=0.5)

    fig.suptitle("ISO 18233 excitation signals")
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
                   pad=10)
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


def _synthetic_room_system(fs: int, seconds: float = 1.0,
                           reverberation: float = 0.7) -> np.ndarray:
    """Direct sound, two early reflections and an exponential diffuse tail."""
    rng = np.random.default_rng(2026)
    n = int(seconds * fs)
    t = np.arange(n) / fs
    system = rng.standard_normal(n) * np.exp(-6.9077 * t / reverberation) * 0.02
    system[80] += 1.0                      # direct sound
    system[1500] += 0.45                   # early reflection
    system[3200] += 0.28                   # second reflection
    return system


def generate_deconvolution_snr_gain(output_dir: str) -> None:
    """ISO 18233 6.2.3: what deconvolution buys over an impulsive source."""
    print("Generating deconvolution_snr_gain...")
    from scipy.signal import fftconvolve

    from phonometry import impulse_response, sweep_signal

    fs = 48000
    seconds = 2.0
    n = int(seconds * fs)
    system = _synthetic_room_system(fs, seconds)
    # One background-noise level in the recording, the same for every
    # excitation: it belongs to the room, not to the signal being played.
    noise_rms = 10.0 ** (-46.0 / 20.0)
    # The effective SNR is read as peak-to-floor, with the floor measured in a
    # window that the room's own decay (T ~ 0.7 s) has long left behind.
    win = slice(int(1.0 * fs), int(1.6 * fs))

    def envelope(h: np.ndarray) -> tuple[np.ndarray, float]:
        mag = np.abs(np.asarray(h, dtype=np.float64))
        floor = float(np.sqrt(np.mean(mag[win] ** 2)))
        peak = float(mag.max())
        # 2 ms running RMS, so the trace reads as a level rather than as a
        # sample-by-sample scatter.
        span = int(0.002 * fs)
        smooth = np.sqrt(np.convolve(mag ** 2, np.ones(span) / span, mode="same"))
        tiny = np.finfo(np.float64).tiny
        return 20.0 * np.log10(np.maximum(smooth, tiny) / peak), 20.0 * np.log10(peak / floor)

    rng = np.random.default_rng(4242)
    cases = []
    # (a) an impulsive source of the same peak amplitude, recorded raw.
    recorded = system + rng.standard_normal(n) * noise_rms
    cases.append(("Pistol shot (no deconvolution)", COLOR_TERTIARY, recorded))
    # (b), (c) the same room through a 1 s and a 4 s sweep, deconvolved.
    for secs, color in ((1.0, COLOR_SECONDARY), (4.0, COLOR_PRIMARY)):
        sweep = sweep_signal(fs, 20.0, 20000.0, secs)
        played = fftconvolve(sweep, system)
        played = played + rng.standard_normal(played.size) * noise_rms
        cases.append((f"{secs:g} s sweep, deconvolved",
                      color, np.asarray(impulse_response(played, sweep, fs, length=n))))

    time = np.arange(n) / fs
    _fig, ax = plt.subplots(figsize=(10.5, 6.2))
    snrs = []
    for label, color, h in cases:
        trace, snr = envelope(h)
        snrs.append(snr)
        ax.plot(time, trace, color=color, linewidth=1.1,
                label=f"{label} — {snr:.0f} dB")
    # A hairline bracket rather than a filled band: a light wash over 30 % of
    # the canvas cannot clear the contrast gate against either background.
    ax.axvline(time[win.start], color=COLOR_FG, linestyle=":", linewidth=1.2)
    ax.axvline(time[win.stop - 1], color=COLOR_FG, linestyle=":", linewidth=1.2)
    ax.annotate("noise floor read here", xy=(1.3, -12.0), fontsize=9,
                color=COLOR_FG, ha="center")
    ax.set_title("Effective signal-to-noise ratio of the recovered impulse "
                 "response", pad=10)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Level re peak [dB]")
    ax.set_xlim(0.0, seconds)
    ax.set_ylim(-105.0, 6.0)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", fontsize=9)
    ax.text(0.03, 0.06,
            f"sweep over pistol: +{snrs[1] - snrs[0]:.0f} dB\n"
            f"two doublings of sweep length: +{snrs[2] - snrs[1]:.0f} dB",
            transform=ax.transAxes, fontsize=10, va="bottom", ha="left",
            bbox={"boxstyle": "round", "facecolor": plt.rcParams["axes.facecolor"],
                  "edgecolor": COLOR_FG, "alpha": 0.85})
    plt.tight_layout()
    save_figure(output_dir, "deconvolution_snr_gain.svg")
    plt.close()


def generate_sweep_distortion_separation(output_dir: str) -> None:
    """ISO 18233 B.5: harmonic packets at negative arrival times."""
    print("Generating sweep_distortion_separation...")
    from scipy.signal import fftconvolve

    from phonometry import impulse_response, sweep_signal

    fs = 48000
    f1, f2, seconds = 20.0, 20000.0, 3.0
    sweep = sweep_signal(fs, f1, f2, seconds)
    system = np.zeros(int(0.4 * fs))
    system[80], system[1400], system[3100] = 1.0, 0.5, 0.32

    played = fftconvolve(sweep, system)
    # A mild memoryless nonlinearity: the loudspeaker driven near its limit.
    played = (played + 0.08 * played ** 2 + 0.04 * played ** 3
              + 0.02 * played ** 4)
    full = np.asarray(impulse_response(played, sweep, fs, return_full=True),
                      dtype=np.float64)

    # Wrap the sequence so t = 0 sits in the middle: what the default return
    # keeps is the causal half, and the harmonic packets are the negative half.
    size = full.size
    half = size // 2
    wrapped = np.roll(np.abs(full), half)
    time = (np.arange(size) - half) / fs
    tiny = np.finfo(np.float64).tiny
    level = 20.0 * np.log10(np.maximum(wrapped, tiny) / wrapped.max())

    _fig, ax = plt.subplots(figsize=(10.5, 5.8))
    ax.axvspan(0.0, time[-1], color=theme_fill(COLOR_PRIMARY, ax), zorder=0)
    ax.plot(time, level, color=COLOR_PRIMARY, linewidth=0.7)
    # The predicted advance of the N-th harmonic, ISO 18233 B.5.
    span = np.log(f2 / f1)
    for order, color in ((2, COLOR_SECONDARY), (3, COLOR_TERTIARY), (4, "#9467bd")):
        advance = seconds * np.log(order) / span
        ax.axvline(-advance, color=color, linestyle="--", linewidth=1.3)
        ax.annotate(f"$H_{order:d}$\n−{advance:.2f} s", xy=(-advance, 2.0),
                    xytext=(-advance, 6.0), fontsize=9, color=color,
                    ha="center", va="bottom")
    ax.annotate("causal part: what impulse_response() returns", xy=(0.35, -6.0),
                fontsize=10, color=COLOR_FG, ha="left")
    ax.set_title("Harmonic distortion lands before $t$ = 0 (ISO 18233 B.5)",
                 pad=18)
    ax.set_xlabel("Arrival time relative to the linear impulse response [s]")
    ax.set_ylabel("Level re peak [dB]")
    ax.set_xlim(-0.78, 0.42)
    ax.set_ylim(-95.0, 18.0)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5)
    plt.tight_layout()
    save_figure(output_dir, "sweep_distortion_separation.svg")
    plt.close()


def generate_source_distance_bias(output_dir: str) -> None:
    """ISO 3382-2 Eq. (1): what a microphone inside d_min gets wrong."""
    print("Generating source_distance_bias...")
    from phonometry import room

    # The same 10 x 6 x 3.5 m room the two setup plates draw, V = 210 m3,
    # with the absorption tuned so the fitted T30 lands near the 0.6 s those
    # plates assume -- which is what makes d_min exactly 2.0 m there too.
    dims = (10.0, 6.0, 3.5)
    volume = float(np.prod(dims))
    alpha = 0.32
    source = (2.0, 3.0, 1.5)               # acoustic centre at 1.5 m
    distances = np.array([0.6, 0.8, 1.0, 1.3, 1.6, 2.0, 2.5, 3.0, 3.5, 4.0,
                          5.0, 6.0, 7.0])
    edt, t30, c80 = [], [], []
    for dist in distances:
        band = []
        # Four lateral offsets per distance: one specular receiver scatters,
        # and the spread is not what this figure is about.
        for offset in (-0.7, -0.25, 0.25, 0.7):
            res = room.image_source_rir(
                dims, source, (2.0 + dist, 3.0 + offset, 1.2), alpha,
                fs=48000, max_order=40,     # c*0.58*T/L_min, not the default
            )
            par = room.room_parameters(res.ir, res.fs, limits=(125.0, 4000.0))
            centres = np.asarray(par.frequency, dtype=np.float64)
            mid = (centres > 400.0) & (centres < 1200.0)
            band.append((float(np.mean(par.edt[mid])),
                         float(np.mean(par.t30[mid])),
                         float(np.mean(par.c80[mid]))))
        edt.append(np.mean([b[0] for b in band]))
        t30.append(np.mean([b[1] for b in band]))
        c80.append(np.mean([b[2] for b in band]))

    d_min = 2.0 * np.sqrt(volume / (343.0 * 0.6))
    surface = 2.0 * (10.0 * 6.0 + 10.0 * 3.5 + 6.0 * 3.5)
    r_c = float(room.critical_distance(surface * alpha / (1.0 - alpha)))

    _fig, (ax_t, ax_c) = plt.subplots(2, 1, figsize=(10, 7.4), sharex=True)
    ax_t.plot(distances, t30, "s-", color=COLOR_PRIMARY, label="T30 (500–1000 Hz)")
    ax_t.plot(distances, edt, "o-", color=COLOR_SECONDARY, label="EDT (500–1000 Hz)")
    ax_t.set_ylabel("Decay time [s]")
    ax_t.set_ylim(0.0, 0.80)
    ax_c.plot(distances, c80, "D-", color=COLOR_TERTIARY, label="C80 (500–1000 Hz)")
    ax_c.set_ylabel("Clarity C80 [dB]")
    ax_c.set_xlabel("Source–receiver distance [m]")
    ax_c.set_ylim(8.0, 20.0)

    for axis in (ax_t, ax_c):
        axis.axvspan(0.0, d_min, color=theme_fill(COLOR_SECONDARY, axis), zorder=0)
        axis.axvline(r_c, color=COLOR_FG, linestyle=":", linewidth=1.3, zorder=1)
        axis.grid(color=COLOR_GRID, linestyle="--", alpha=0.5)
        axis.set_xlim(0.0, 7.4)
        axis.legend(loc="lower right", fontsize=9)
    ax_t.annotate(rf"excluded: $r < d_{{\mathrm{{min}}}}$ = {d_min:.1f} m",
                  xy=(d_min / 2, 0.73),
                  fontsize=10, color=COLOR_SECONDARY, ha="center")
    ax_t.annotate(f"critical distance {r_c:.1f} m", xy=(r_c, 0.08),
                  xytext=(r_c + 0.4, 0.08), fontsize=9, color=COLOR_FG,
                  va="center", ha="left")
    ax_t.set_title(r"A microphone inside $d_{\mathrm{min}}$ returns wrong "
                   "numbers, not noisy ones", pad=10)
    plt.tight_layout()
    save_figure(output_dir, "source_distance_bias.svg")
    plt.close()


def generate_modal_count_per_band(output_dir: str) -> None:
    """How many modes an octave band holds, and why the low ones scatter."""
    print("Generating modal_count_per_band...")
    from phonometry import room

    # The same 7 x 5 x 3 m room as the image-source guide, T ~ 0.9 s.
    dims = (7.0, 5.0, 3.0)
    volume = float(np.prod(dims))
    reverberation = 0.9
    bands = np.array([63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0])
    edges = np.sqrt(2.0)
    counts = np.array([
        float(room.room_mode_count(f * edges, dims)
              - room.room_mode_count(f / edges, dims))
        for f in bands
    ])
    f_schroeder = float(room.schroeder_frequency(reverberation, volume))

    _fig, ax = plt.subplots(figsize=(10, 6.0))
    below = bands < f_schroeder
    ax.bar(bands[below], counts[below], width=bands[below] * 0.62,
           color=COLOR_SECONDARY, label="Below the Schroeder frequency")
    ax.bar(bands[~below], counts[~below], width=bands[~below] * 0.62,
           color=COLOR_PRIMARY, label="Above it")
    for centre, count in zip(bands, counts):
        ax.annotate(f"{count:,.0f}", xy=(centre, count * 1.25), fontsize=9,
                    color=COLOR_FG, ha="center")
    ax.axvline(f_schroeder, color=COLOR_FG, linestyle="--", linewidth=1.4)
    ax.annotate(f"Schroeder frequency {f_schroeder:.0f} Hz",
                xy=(f_schroeder * 1.06, counts.max() * 0.28), fontsize=10,
                color=COLOR_FG, ha="left", rotation=90, va="center")
    ax.set_xscale("log")
    ax.set_yscale("log")
    format_frequency_axis(ax, 45.0, 5600.0)
    ax.set_xlabel(LABEL_FREQ_HZ)
    ax.set_ylabel("Modes inside the octave band")
    ax.set_ylim(5.0, counts.max() * 6.0)
    ax.set_title("What the analysis band averages over (7 × 5 × 3 m room, "
                 "$V$ = 105 m³)", pad=10)
    ax.grid(which="both", axis="y", color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    save_figure(output_dir, "modal_count_per_band.svg")
    plt.close()


def generate_excitation_robustness(output_dir: str) -> None:
    """ISO 18233 B.7 / A.4: sweep and MLS under a slow temperature drift."""
    print("Generating excitation_robustness...")
    from scipy.signal import fftconvolve

    from phonometry import (
        impulse_response,
        mls_impulse_response,
        mls_signal,
        sweep_signal,
    )

    fs = 48000
    seconds = 1.4
    n = int(0.5 * fs)
    system = _synthetic_room_system(fs, 0.5, reverberation=0.4)
    noise_rms = 10.0 ** (-80.0 / 20.0)

    def drift(signal: np.ndarray, parts_per_million: float) -> np.ndarray:
        """Resample by a slowly growing factor: the room warming during a take.

        A stretch of ``parts_per_million`` by the end of the record is a speed
        of sound falling by the same fraction; since c ~ sqrt(T), 500 ppm is a
        rise of about 2 x 5e-4 x 293 K = 0.3 K across the take.
        """
        idx = np.arange(signal.size, dtype=np.float64)
        stretched = idx * (1.0 + parts_per_million * 1e-6 * idx / signal.size)
        resampled = np.interp(idx, stretched, signal, left=0.0, right=0.0)
        return np.asarray(resampled, dtype=np.float64)

    def envelope(h: np.ndarray) -> np.ndarray:
        mag = np.abs(np.asarray(h, dtype=np.float64))[:n]
        span = int(0.002 * fs)
        smooth = np.sqrt(np.convolve(mag ** 2, np.ones(span) / span, mode="same"))
        tiny = np.finfo(np.float64).tiny
        level = 20.0 * np.log10(np.maximum(smooth, tiny) / smooth.max())
        return np.asarray(level, dtype=np.float64)

    sweep = sweep_signal(fs, 20.0, 20000.0, seconds)
    mls = mls_signal(16)                        # (2^16 - 1)/fs = 1.37 s period
    period = mls.size
    # The drifted MLS trips the library's own circular-aliasing warning. That
    # warning is the result this figure draws, not a defect in the run.
    import warnings

    from phonometry import ImpulseResponseWarning
    traces = {}
    for name, ppm in (("stationary", 0.0), ("+0.3 K during the take", 500.0)):
        rng = np.random.default_rng(97)
        played = fftconvolve(sweep, system)
        played = drift(played, ppm) + rng.standard_normal(played.size) * noise_rms
        traces[("sweep", name)] = envelope(
            np.asarray(impulse_response(played, sweep, fs, length=n)))
        # Four periods played, the first discarded as warm-up: with the first
        # period kept the recovered floor sits 23 dB higher even with no drift.
        rec = fftconvolve(np.tile(mls, 4), system)[period: 4 * period]
        rec = drift(rec, ppm) + rng.standard_normal(rec.size) * noise_rms
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ImpulseResponseWarning)
            recovered = np.asarray(mls_impulse_response(rec, mls, length=n))
        traces[("MLS", name)] = envelope(recovered)

    time = np.arange(n) / fs
    _fig, ax = plt.subplots(figsize=(10.5, 6.2))
    for (kind, name), trace in traces.items():
        color = COLOR_PRIMARY if kind == "sweep" else COLOR_SECONDARY
        style = "-" if name == "stationary" else "--"
        ax.plot(time, trace, color=color, linestyle=style,
                linewidth=1.0 if style == "-" else 1.5,
                label=f"{kind}, {name}")
    ax.set_title("Time variance costs the MLS its dynamic range, not the sweep",
                 pad=10)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Level re peak [dB]")
    ax.set_xlim(0.0, n / fs)
    ax.set_ylim(-98.0, 5.0)
    ax.annotate("the MLS floor rises to the room's own early decay:\n"
                "the tail is gone", xy=(0.30, -15.0), xytext=(0.20, -40.0),
                fontsize=10, color=COLOR_SECONDARY, ha="left",
                arrowprops={"arrowstyle": "->", "color": COLOR_SECONDARY})
    ax.annotate("sweep: the two traces lie on top of each other",
                xy=(0.18, -34.0), xytext=(0.02, -76.0), fontsize=10,
                color=COLOR_PRIMARY, ha="left",
                arrowprops={"arrowstyle": "->", "color": COLOR_PRIMARY})
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    save_figure(output_dir, "excitation_robustness.svg")
    plt.close()


def generate_open_plan_quality(output_dir: str) -> None:
    """Two ISO 3382-3 offices read against the Annex A ends of the scale."""
    print("Generating open_plan_quality...")
    from phonometry import open_plan_metrics

    positions = np.array([2.0, 3.0, 4.0, 6.0, 8.0, 11.0, 16.0])
    cases = (
        # (label, D2,S, Lp,A,S,4m, STI intercept, STI slope, colour)
        ("Treated", 8.0, 47.0, 0.62, 0.0267, COLOR_PRIMARY),
        ("Untreated", 4.0, 52.0, 0.75, 0.0227, COLOR_SECONDARY),
    )

    _fig, (ax_l, ax_s) = plt.subplots(2, 1, figsize=(10, 7.6), sharex=True)
    # Annex A: the two ends of the informative scale, shaded on each axis.
    ax_l.axhspan(50.0, 62.0, color=theme_fill(COLOR_SECONDARY, ax_l), zorder=0)
    ax_l.axhspan(30.0, 48.0, color=theme_fill(COLOR_PRIMARY, ax_l), zorder=0)
    ax_l.annotate(r"$L_{p,A,S,4\,\mathrm{m}} > 50$ dB: poor", xy=(2.1, 59.0),
                  fontsize=9, color=COLOR_FG)
    ax_l.annotate(r"$L_{p,A,S,4\,\mathrm{m}} \leq 48$ dB: good target",
                  xy=(2.1, 32.0), fontsize=9, color=COLOR_FG)
    ax_s.axvspan(1.8, 5.0, color=theme_fill(COLOR_PRIMARY, ax_s), zorder=0)
    ax_s.axvspan(10.0, 20.0, color=theme_fill(COLOR_SECONDARY, ax_s), zorder=0)
    ax_s.annotate(r"$r_D \leq 5$ m: good", xy=(2.1, 0.10), fontsize=9,
                  color=COLOR_FG)
    ax_s.annotate(r"$r_D > 10$ m: poor", xy=(10.4, 0.10), fontsize=9,
                  color=COLOR_FG)

    for label, d2s, lp_4m, sti_0, sti_slope, color in cases:
        slope = -d2s / np.log10(2.0)
        levels = (lp_4m - slope * np.log10(4.0)) + slope * np.log10(positions)
        sti = sti_0 - sti_slope * positions
        res = open_plan_metrics(positions, levels, sti)
        span = np.logspace(np.log10(1.8), np.log10(20.0), 200)
        ax_l.plot(span, (res.lp_as_4m - slope * np.log10(4.0))
                  + slope * np.log10(span), "--", color=color, linewidth=1.6)
        ax_l.plot(positions, levels, "o", color=color, markersize=7,
                  markerfacecolor="white", markeredgewidth=1.6,
                  label=rf"{label}: $D_{{2,S}}$ = {res.d2s:.0f} dB, "
                        rf"$L_{{p,A,S,4\,\mathrm{{m}}}}$ = {res.lp_as_4m:.0f} dB")
        ax_l.plot(4.0, res.lp_as_4m, "D", color=color, markersize=9, zorder=6)
        ax_s.plot(span, sti_0 - sti_slope * span, "--", color=color,
                  linewidth=1.6)
        ax_s.plot(positions, sti, "o", color=color, markersize=7,
                  markerfacecolor="white", markeredgewidth=1.6,
                  label=f"{label}: $r_D$ = {res.rd:.1f} m, "
                        f"$r_P$ = {res.rp:.0f} m")
        ax_s.plot(res.rd, 0.50, "D", color=color, markersize=9, zorder=6)

    ax_l.axvline(4.0, color=COLOR_FG, linestyle=":", alpha=0.35, linewidth=1)
    ax_l.set_ylabel("A-weighted speech level [dB]")
    ax_l.set_ylim(30.0, 62.0)
    ax_l.set_title("The same two quantities at the two ends of Annex A",
                   pad=10)
    ax_s.axhline(0.50, color=COLOR_FG, linestyle="-", linewidth=1.1)
    ax_s.annotate("STI = 0.50", xy=(16.5, 0.52), fontsize=9, color=COLOR_FG)
    ax_s.axhline(0.20, color=COLOR_FG, linestyle="--", linewidth=1.0)
    ax_s.annotate("STI = 0.20", xy=(16.5, 0.22), fontsize=9, color=COLOR_FG)
    ax_s.set_ylabel("Speech transmission index")
    ax_s.set_ylim(0.0, 0.85)
    ax_s.set_xlabel("Distance from the talker $r$ [m]")

    from matplotlib.ticker import NullFormatter, ScalarFormatter
    for axis in (ax_l, ax_s):
        axis.set_xscale("log")
        axis.set_xlim(1.8, 20.0)
        axis.xaxis.set_major_formatter(ScalarFormatter())
        axis.xaxis.set_minor_formatter(NullFormatter())
        axis.set_xticks([2, 3, 4, 6, 8, 11, 16])
        axis.set_xticklabels(["2", "3", "4", "6", "8", "11", "16"])
        axis.grid(which="major", color=COLOR_GRID, linestyle="-", alpha=0.4)
        axis.set_axisbelow(True)
        axis.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    save_figure(output_dir, "open_plan_quality.svg")
    plt.close()


def generate_absorption_per_table(output_dir: str) -> None:
    """Long Eqs. (17.53)-(17.54): the absorption window a layout leaves open."""
    print("Generating absorption_per_table...")
    from phonometry import room

    span = np.linspace(0.6, 3.2, 200)
    communication = np.asarray(room.absorption_per_table(span, -6.0))
    privacy = np.asarray(room.absorption_per_table(span, -9.0))
    # A realistic restaurant layout: 1.2 m across the table, 2.0 m between.
    r_s, r_t = 1.2, 2.0
    lower = float(room.absorption_per_table(r_s, -6.0))
    upper = float(room.absorption_per_table(r_t, -9.0))
    closure = float(np.sqrt(6.31 / 3.16))

    _fig, (ax_a, ax_w) = plt.subplots(1, 2, figsize=(12.5, 5.8))

    ax_a.plot(span, communication, color=COLOR_PRIMARY, linewidth=2.0,
              label=r"Communication: $A_{\mathrm{tab}} > 6.31\,r_s^2$  "
                    r"($L_{\mathrm{SN}} > -6$ dB)")
    ax_a.plot(span, privacy, color=COLOR_SECONDARY, linewidth=2.0,
              label=r"Privacy: $A_{\mathrm{tab}} < 3.16\,r_t^2$  "
                    r"($L_{\mathrm{SN}} < -9$ dB)")
    ax_a.axhspan(lower, upper, color=theme_fill(COLOR_TERTIARY, ax_a), zorder=0)
    ax_a.annotate(rf"feasible $A_{{\mathrm{{tab}}}}$: {lower:.1f} to "
                  f"{upper:.1f} m²",
                  xy=(2.55, (lower + upper) / 2 + 1.4), fontsize=9,
                  color=COLOR_FG, ha="center")
    ax_a.plot([r_s], [lower], "o", color=COLOR_PRIMARY, markersize=9)
    ax_a.plot([r_t], [upper], "o", color=COLOR_SECONDARY, markersize=9)
    ax_a.annotate(rf"$r_s$ = {r_s:g} m → $A_{{\mathrm{{tab}}}}$ > {lower:.1f} m²",
                  xy=(r_s, lower), xytext=(0.66, 22.0), fontsize=9,
                  color=COLOR_PRIMARY,
                  arrowprops={"arrowstyle": "->", "color": COLOR_PRIMARY})
    ax_a.annotate(rf"$r_t$ = {r_t:g} m → $A_{{\mathrm{{tab}}}}$ < {upper:.1f} m²",
                  xy=(r_t, upper), xytext=(2.05, 6.0), fontsize=9,
                  color=COLOR_SECONDARY,
                  arrowprops={"arrowstyle": "->", "color": COLOR_SECONDARY})
    ax_a.plot([1.0], [float(room.absorption_per_table(1.0, -6.0))], "s",
              color=COLOR_PRIMARY, markersize=7, markerfacecolor="none")
    ax_a.plot([2.5], [float(room.absorption_per_table(2.5, -9.0))], "s",
              color=COLOR_SECONDARY, markersize=7, markerfacecolor="none")
    ax_a.set_xlabel("Separation [m]")
    ax_a.set_ylabel(r"Absorption per occupied table $A_{\mathrm{tab}}$ [m²]")
    ax_a.set_ylim(0.0, 60.0)
    ax_a.set_xlim(0.6, 3.2)
    ax_a.set_title("The design window, for one layout",
                   pad=10)
    ax_a.grid(color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax_a.set_axisbelow(True)
    ax_a.legend(loc="upper left", fontsize=9)

    # Right: the window width against how far apart the tables stand.
    ratios = np.linspace(1.0, 2.2, 200)
    width = (np.asarray(room.absorption_per_table(ratios * r_s, -9.0))
             - float(room.absorption_per_table(r_s, -6.0)))
    ax_w.axhline(0.0, color=COLOR_FG, linewidth=1.2)
    ax_w.plot(ratios, width, color=COLOR_TERTIARY, linewidth=2.2)
    ax_w.fill_between(ratios, 0.0, width, where=width > 0.0, zorder=0,
                      color=theme_fill(COLOR_TERTIARY, ax_w))
    ax_w.axvline(closure, color=COLOR_SECONDARY, linestyle="--", linewidth=1.6)
    ax_w.annotate(f"window closes at $r_t/r_s$ = {closure:.2f}",
                  xy=(closure, -4.0), xytext=(closure + 0.05, -4.0),
                  fontsize=10, color=COLOR_SECONDARY)
    ax_w.plot([r_t / r_s], [upper - lower], "o", color=COLOR_TERTIARY,
              markersize=9)
    ax_w.annotate(f"this layout: {r_t / r_s:.2f}, {upper - lower:.1f} m² wide",
                  xy=(r_t / r_s, upper - lower), xytext=(1.02, 9.6),
                  fontsize=9, color=COLOR_TERTIARY,
                  arrowprops={"arrowstyle": "->", "color": COLOR_TERTIARY})
    ax_w.set_xlabel("Table spacing over cross-table separation, $r_t/r_s$")
    ax_w.set_ylabel(r"Width of the feasible $A_{\mathrm{tab}}$ window [m²]")
    ax_w.set_xlim(1.0, 2.2)
    ax_w.set_title(f"Packed tables close it ($r_s$ = {r_s:g} m)",
                   pad=10)
    ax_w.grid(color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax_w.set_axisbelow(True)

    plt.tight_layout()
    save_figure(output_dir, "absorption_per_table.svg")
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
                        label=f"Spatial decay $D_{{2,S}}$ = {m.d2s:.1f} dB")
    pts_spl, = ax.plot(r, lp, "o", color=COLOR_PRIMARY, markersize=7,
                       markerfacecolor="white", markeredgewidth=1.6,
                       label=r"Measured $L_{p,A,S}$")
    ax.axvline(4.0, color=COLOR_FG, linestyle=":", alpha=0.35, linewidth=1)
    mark_4m, = ax.plot(4.0, m.lp_as_4m, "D", color=COLOR_SECONDARY, markersize=9,
                       zorder=6,
                       label=rf"$L_{{p,A,S,4\,\mathrm{{m}}}}$ = {m.lp_as_4m:.0f} dB")
    ax.annotate(rf"$L_{{p,A,S,4\,\mathrm{{m}}}}$ = {m.lp_as_4m:.0f} dB",
                xy=(4.0, m.lp_as_4m),
                xytext=(4.7, m.lp_as_4m + 4.5), fontsize=10,
                arrowprops={"arrowstyle": "->", "lw": 1.0})

    ax.set_title("Open-Plan Spatial Decay of Speech (ISO 3382-3)",
                 pad=12)
    ax.set_xlabel("Distance from the talker $r$ [m]")
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
    for dist, level, name in [(m.rd, 0.50, "$r_D$"), (m.rp, 0.20, "$r_P$")]:
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
    ax.set_title("Image-Source Room Impulse Response: a 7 × 5 × 3 m room "
                 "(order ≤ 10)", pad=12)
    ax.set_xlim(0.0, 120.0)
    ax.set_ylim(-60.0, 5.0)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=9)
    ax.text(0.015, 0.03,
            "each reflection is a mirror image of the source;\n"
            r"amplitude = product of wall reflection factors / ($4\pi r$)",
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
    ax.set_title("Reverberation-time prediction models", pad=12)
    ax.set_ylim(bottom=0.0)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=9)

    info = [
        "room 10 × 7 × 3.5 m",
        "$V$ = 245 m³, $S$ = 259 m²",
        "anisotropic: absorptive floor/ceiling",
        "$c_0$ = 343 m/s, air at 20 °C / 50 % RH",
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
        (ax_a, "Equivalent absorption area $A$ [m²]", "Absorption area (Formula 1)"),
        (ax_t, "Reverberation time $T$ [s]", "Reverberation time (Formula 5)"),
    ):
        ax.set_xticks(freq)
        ax.set_xticklabels(labels)
        ax.set_xlabel("Octave-band centre frequency [Hz]")
        ax.set_ylabel(ylab)
        ax.set_title(title, pad=10)
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
    from phonometry import noise_criterion, rc_curve, room_criterion
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
                    pad=10)
    ax_nc.grid(which="both", axis="y", color=COLOR_GRID, linestyle="-", alpha=0.4)
    ax_nc.set_axisbelow(True)
    ax_nc.legend(loc="upper right")

    # --- Right: RC Mark II family, reference + rumble/hiss tolerances. ---
    ref = rc.reference_curve
    low = OCTAVE_BANDS <= 500.0
    high = OCTAVE_BANDS >= 1000.0
    # The Table D.1 family, mirroring the NC family on the left: constant
    # -5 dB/octave keyed to the 1 kHz value, with the 55 dB low-frequency
    # floor flattening the 16/31.5 Hz end of RC-25 to RC-30.
    for index in range(25, 51, 5):
        family = rc_curve(float(index))
        ax_rc.plot(OCTAVE_BANDS, family, color=COLOR_GRID, lw=0.8, zorder=1)
        ax_rc.annotate(f"{index:d}", (OCTAVE_BANDS[-1], family[-1]),
                       fontsize=7, color="#999999", va="center")
    ax_rc.annotate("55 dB floor\n(16 Hz = 31.5 Hz)", xy=(22.0, 55.0),
                   xytext=(70.0, 66.0), fontsize=8, color=COLOR_FG, ha="left",
                   arrowprops={"arrowstyle": "->", "color": COLOR_FG, "lw": 0.9})
    ax_rc.plot(OCTAVE_BANDS, ref, "s--", color="#7f7f7f", zorder=2,
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
                    pad=10)
    ax_rc.grid(which="both", axis="y", color=COLOR_GRID, linestyle="-", alpha=0.4)
    ax_rc.set_axisbelow(True)
    ax_rc.legend(loc="upper right")

    plt.tight_layout()
    save_figure(output_dir, "room_noise_criteria.png")
    plt.close()


def generate_nc_blind_spot(output_dir: str) -> None:
    """Two rooms with the same NC rating and opposite spectral character."""
    print("Generating nc_blind_spot...")
    from phonometry import noise_criterion, room_criterion
    from phonometry.room.noise_criteria import NC_CURVES, NC_INDICES, OCTAVE_BANDS

    # Two ANSI/ASA S12.2 spectra built to touch the NC-40 curve at one band
    # each: a duct-rumble room tangent at 125 Hz, and a diffuser-hiss room
    # tangent at 4 kHz. Both rate NC-40; nothing in the NC number separates
    # them, and the RC Mark II tag does it in one letter.
    rumbly = np.array([66.0, 64.0, 62.0, 56.0, 48.0, 40.0, 34.0, 28.0, 24.0, 19.0])
    hissy = np.array([52.0, 52.0, 50.0, 46.0, 43.0, 39.0, 35.0, 34.0, 38.0, 34.0])
    cases = (
        ("Duct rumble", rumbly, COLOR_PRIMARY, "o-"),
        ("Diffuser hiss", hissy, COLOR_SECONDARY, "s-"),
    )
    nc_left = noise_criterion(rumbly)

    _fig, (ax_nc, ax_rc) = plt.subplots(1, 2, figsize=(12.5, 5.6))

    # --- Left: one NC label for two spectra. ---
    for row, idx in zip(NC_CURVES, NC_INDICES):
        ax_nc.plot(OCTAVE_BANDS, row, color=COLOR_GRID, lw=0.8, zorder=1)
        ax_nc.annotate(f"{idx:.0f}", (OCTAVE_BANDS[-1], row[-1]),
                       fontsize=7, color="#999999", va="center")
    for label, spectrum, color, style in cases:
        nc = noise_criterion(spectrum)
        ax_nc.plot(OCTAVE_BANDS, spectrum, style, color=color, zorder=3,
                   label=f"{label} — tangent at {nc.governing_frequency:g} Hz")
        gov = spectrum[OCTAVE_BANDS == nc.governing_frequency][0]
        ax_nc.plot([nc.governing_frequency], [gov], "D", color=color, ms=10,
                   mec=COLOR_FG, mew=0.8, zorder=4)
    ax_nc.set_title(f"One rating: NC-{nc_left.rating:g} for both rooms",
                    pad=10)

    # --- Right: the deviation from each room's own RC reference curve. ---
    # The tag rule read directly: +5 dB at and below 500 Hz, +3 dB at and
    # above 1 kHz (clause D.3), drawn as one step threshold.
    threshold = np.where(OCTAVE_BANDS <= 500.0, 5.0, 3.0)
    ax_rc.step(OCTAVE_BANDS, threshold, where="mid", color=COLOR_FG, lw=1.4,
               ls="--", zorder=2, label="Tag threshold (D.3): +5 / +3 dB")
    ax_rc.fill_between(OCTAVE_BANDS, threshold, 17.0, step="mid", zorder=0,
                       color=theme_fill(COLOR_TERTIARY, ax_rc))
    ax_rc.axhline(0.0, color=COLOR_GRID, lw=1.0, zorder=1)
    for label, spectrum, color, style in cases:
        rc = room_criterion(spectrum)
        ax_rc.plot(OCTAVE_BANDS, spectrum - rc.reference_curve, style,
                   color=color, zorder=3, label=f"{label} — {rc.label}")
    ax_rc.set_ylim(-17.0, 17.0)
    ax_rc.set_title("Two ratings: the RC Mark II tag reads the character",
                    pad=10)

    for axis in (ax_nc, ax_rc):
        axis.set_xscale("log")
        axis.set_xticks(list(OCTAVE_BANDS))
        axis.set_xticklabels([f"{f:g}" for f in OCTAVE_BANDS], rotation=45,
                             ha="right")
        axis.set_xlabel("Octave-band center frequency [Hz]")
        axis.grid(which="both", axis="y", color=COLOR_GRID, linestyle="-",
                  alpha=0.4)
        axis.set_axisbelow(True)
        axis.legend(loc="upper right", fontsize=9)
    ax_nc.set_ylabel("Octave-band sound pressure level [dB]")
    ax_rc.set_ylabel("Level minus the room's own RC curve [dB]")
    ax_rc.legend(loc="lower left", fontsize=9)

    plt.tight_layout()
    save_figure(output_dir, "nc_blind_spot.svg")
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


def generate_reverberation_model_absorption(output_dir: str) -> None:
    """Sabine, Eyring and Millington-Sette against the mean absorption.

    The single axis along which the first three models differ. One concept:
    why Sabine survives at low absorption and breaks at high.
    """
    print("Generating reverberation_model_absorption...")
    from phonometry import (
        eyring_reverberation_time,
        millington_sette_reverberation_time,
        sabine_reverberation_time,
    )

    # The 8 x 5 x 3 m shoebox of the guide's section 1 (V = 120 m3, S = 158 m2).
    volume, area = 120.0, 158.0
    alpha = np.linspace(0.02, 0.99, 240)
    sabine = np.array([float(sabine_reverberation_time(volume, [(area, a)]))
                       for a in alpha])
    eyring = np.array([float(eyring_reverberation_time(volume, [(area, a)]))
                       for a in alpha])
    millington = np.array(
        [float(millington_sette_reverberation_time(volume, [(area, a)]))
         for a in alpha])

    _fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(10, 7.4), sharex=True,
        gridspec_kw={"height_ratios": [2.0, 1.0]})

    # Sabine's stated domain is a mean absorption up to about 0.2.
    for ax in (top, bottom):
        ax.axvspan(0.2, 1.0, color=theme_fill(COLOR_SECONDARY, ax), zorder=0)
        ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5, zorder=0)
        ax.set_axisbelow(True)

    top.semilogy(alpha, sabine, color=COLOR_SECONDARY, linewidth=2.2,
                 label="Sabine", zorder=5)
    top.semilogy(alpha, eyring, color=COLOR_TERTIARY, linewidth=2.2,
                 linestyle="--", label="Eyring", zorder=5)
    top.semilogy(alpha, millington, color="#9467bd", linewidth=1.4,
                 linestyle=":", label="Millington-Sette", zorder=6)
    top.set_ylabel(r"Reverberation time $T$ [s]")
    top.set_ylim(0.02, 4.0)
    top.set_title("Model behaviour against the mean absorption",
                  pad=12)
    top.legend(loc="upper right", fontsize=9)
    top.annotate(
        "Sabine stays finite:\n0.12 s at " + r"$\alpha = 1$",
        xy=(0.99, float(sabine[-1])), xytext=(0.66, 0.55),
        textcoords="axes fraction", fontsize=9, color=COLOR_FG,
        arrowprops={"arrowstyle": "->", "color": COLOR_FG, "lw": 1.0})
    top.annotate(
        "Eyring falls to zero", xy=(0.99, float(eyring[-1])),
        xytext=(0.60, 0.10), textcoords="axes fraction", fontsize=9,
        color=COLOR_FG,
        arrowprops={"arrowstyle": "->", "color": COLOR_FG, "lw": 1.0})

    departure = 100.0 * (eyring / sabine - 1.0)
    bottom.plot(alpha, departure, color=COLOR_TERTIARY, linewidth=2.2, zorder=5)
    for mark in (0.2, 0.5, 0.9):
        value = float(np.interp(mark, alpha, departure))
        bottom.plot([mark], [value], "o", color=COLOR_FG, ms=5, zorder=6)
        bottom.annotate(f"{_fmt_minus(value, '.0f')} %", xy=(mark, value),
                        xytext=(mark + 0.02, value + 9.0), fontsize=9,
                        color=COLOR_FG)
    bottom.set_xlabel(r"Mean absorption coefficient $\bar\alpha$")
    bottom.set_ylabel("Departure from\nSabine [%]")
    bottom.set_xlim(0.0, 1.0)
    bottom.set_ylim(-85.0, 12.0)

    info = [
        "room 8 × 5 × 3 m",
        "$V$ = 120 m³, $S$ = 158 m²",
        "uniform absorption, no air term",
        "shaded: outside Sabine's domain",
    ]
    bottom.text(0.015, 0.06, "\n".join(info), transform=bottom.transAxes,
                va="bottom", ha="left", fontsize=9, color=COLOR_FG,
                bbox={"boxstyle": "round,pad=0.4", "facecolor": COLOR_PANEL,
                      "edgecolor": COLOR_GRID})
    plt.tight_layout()
    save_figure(output_dir, "reverberation_model_absorption.svg")
    plt.close()


def generate_enclosed_space_air_term(output_dir: str) -> None:
    """EN 12354-6 air term: six climate profiles and two room volumes.

    One concept: the air absorption is a volume effect, so the frequency
    threshold and the volume threshold of clause 4.3 are one rule.
    """
    print("Generating enclosed_space_air_term...")
    from phonometry import enclosed_space_reverberation
    from phonometry.room.enclosed_space_absorption import (
        AIR_ATTENUATION,
        OCTAVE_BANDS,
    )

    freq = np.asarray(OCTAVE_BANDS)
    soft = [0.15] * freq.size          # a plausible ordinary room, per band
    hall_area, hall_volume = 1000.0, 2000.0

    _fig, (left, right) = plt.subplots(1, 2, figsize=(12.6, 5.4))

    still = enclosed_space_reverberation([(hall_area, soft)], hall_volume)
    colours = series_colors(len(AIR_ATTENUATION))
    for colour, name in zip(colours, AIR_ATTENUATION):
        humid = enclosed_space_reverberation(
            [(hall_area, soft)], hall_volume, air_condition=name)
        left.loglog(freq, humid.absorption_area - still.absorption_area,
                    color=colour, marker="o", markersize=4, linewidth=1.8,
                    label=name.replace("C_", " °C, ") + " % RH")
    left.set_ylabel(r"Air term $A_{\mathrm{air}} = 4mV(1-\psi)$ [m²]")
    left.set_title(r"Six climate profiles, $V$ = 2000 m³",
                   pad=10)
    left.legend(loc="upper left", fontsize=8)

    styles = ((60.0, 94.0, COLOR_PRIMARY, "60 m³ office"),
              (2000.0, 1000.0, COLOR_SECONDARY, "2000 m³ hall"))
    for volume, area, colour, label in styles:
        for condition, dash in ((None, "-"), ("20C_50-70", "--")):
            res = enclosed_space_reverberation([(area, soft)], volume,
                                               air_condition=condition)
            suffix = " (no air)" if condition is None else " (20 °C, 50-70 %)"
            right.semilogx(freq, res.reverberation_time, dash, color=colour,
                           marker="o", markersize=4, linewidth=1.8,
                           label=label + suffix)
    right.set_ylabel(r"Reverberation time $T$ [s]")
    right.set_ylim(bottom=0.0)
    right.set_title("The same absorption in two volumes",
                    pad=10)
    right.legend(loc="lower left", fontsize=8)
    right.annotate("−42 % at 8 kHz", xy=(8000.0, 1.24), xytext=(0.42, 0.30),
                   textcoords="axes fraction", fontsize=9, color=COLOR_FG,
                   arrowprops={"arrowstyle": "->", "color": COLOR_FG, "lw": 1.0})
    right.annotate("−1.7 % at 1 kHz", xy=(1000.0, 0.67), xytext=(0.05, 0.62),
                   textcoords="axes fraction", fontsize=9, color=COLOR_FG,
                   arrowprops={"arrowstyle": "->", "color": COLOR_FG, "lw": 1.0})

    for ax in (left, right):
        format_frequency_axis(ax, float(freq[0]), float(freq[-1]), minor=None)
        ax.set_xlabel(LABEL_FREQ_HZ)
        ax.grid(which="both", color=COLOR_GRID, linestyle="--", alpha=0.45)
        ax.set_axisbelow(True)
    plt.tight_layout()
    save_figure(output_dir, "enclosed_space_air_term.svg")
    plt.close()


def generate_enclosed_space_objects(output_dir: str) -> None:
    """EN 12354-6 Annex E: the object term and the volume-displacement psi.

    One concept: the objects' own absorption dominates, and the object
    fraction contributes a separate, constant few per cent.
    """
    print("Generating enclosed_space_objects...")
    from phonometry import (
        enclosed_space_reverberation,
        hard_object_absorption,
        object_fraction,
    )
    from phonometry.room.enclosed_space_absorption import OCTAVE_BANDS

    freq = np.asarray(OCTAVE_BANDS)
    n = freq.size
    # EN 12354-6 Annex E: a 4.54 x 2.73 x 2.40 m room, V = 29.75 m3, with the
    # annex's 1 kHz coefficients held across the band set.
    surfaces = [(12.39, [0.05] * n), (12.39, [0.02] * n),
                (10.90, [0.04] * n), (10.90, [0.04] * n),
                (6.55, [0.04] * n), (6.55, [0.04] * n)]
    volumes = [0.15, 0.60, 0.05, 0.05, 0.65, 0.65]     # Annex E case 2
    objects = hard_object_absorption(volumes)
    psi = float(object_fraction(volumes, 29.75))

    bare = enclosed_space_reverberation(surfaces, 29.75,
                                        air_condition="20C_50-70")
    absorbing = enclosed_space_reverberation(
        surfaces, 29.75, objects=objects, air_condition="20C_50-70")
    furnished = enclosed_space_reverberation(
        surfaces, 29.75, objects=objects, object_fraction=psi,
        air_condition="20C_50-70")

    dry = enclosed_space_reverberation(surfaces, 29.75)
    air = bare.absorption_area - dry.absorption_area
    surface_area = dry.absorption_area
    objects_area = absorbing.absorption_area - bare.absorption_area

    _fig, (left, right) = plt.subplots(1, 2, figsize=(12.6, 5.4))
    x = np.arange(n)
    width = 0.38
    left.bar(x - width / 2, surface_area, width, color=COLOR_PRIMARY,
             label="surfaces", zorder=3)
    left.bar(x - width / 2, air, width, bottom=surface_area,
             color=COLOR_MUTED, label="air", zorder=3)
    left.bar(x + width / 2, surface_area, width, color=COLOR_PRIMARY, zorder=3)
    left.bar(x + width / 2, air, width, bottom=surface_area,
             color=COLOR_MUTED, zorder=3)
    left.bar(x + width / 2, objects_area, width, bottom=surface_area + air,
             color=COLOR_TERTIARY, label="objects (Formula 4)", zorder=3)
    for centre, text in ((x - width / 2, "bare"), (x + width / 2, "furnished")):
        left.text(float(centre[0]), 0.12, text, rotation=90, fontsize=8,
                  ha="center", va="bottom", color=COLOR_FG, zorder=5)
    left.set_xticks(x)
    left.set_xticklabels([f"{f:g}" if f < 1000 else f"{f / 1000:g}k"
                          for f in freq])
    left.set_xlabel("Octave-band centre frequency [Hz]")
    left.set_ylabel("Equivalent absorption area $A$ [m²]")
    left.set_title("Where the absorption comes from", pad=10)
    left.legend(loc="upper left", fontsize=9)

    right.semilogx(freq, bare.reverberation_time, color=COLOR_SECONDARY,
                   marker="o", markersize=4, linewidth=2.0, label="bare")
    right.semilogx(freq, absorbing.reverberation_time, color=COLOR_TERTIARY,
                   marker="s", markersize=4, linewidth=2.0,
                   label=r"furnished, $\psi$ = 0 (absorption only)")
    right.semilogx(freq, furnished.reverberation_time, color=COLOR_PRIMARY,
                   marker="^", markersize=4, linewidth=2.0,
                   label=rf"furnished, $\psi$ = {psi:.3f}")
    right.set_xlabel(LABEL_FREQ_HZ)
    right.set_ylabel(r"Reverberation time $T$ [s]")
    right.set_ylim(bottom=0.0)
    right.set_title("The volume the objects displace", pad=10)
    right.legend(loc="lower left", fontsize=9)
    format_frequency_axis(right, float(freq[0]), float(freq[-1]), minor=None)
    right.annotate(
        rf"the gap is $\psi$ alone: {100.0 * psi:.1f} %",
        xy=(1000.0, float(furnished.reverberation_time[3])),
        xytext=(0.30, 0.72), textcoords="axes fraction", fontsize=9,
        color=COLOR_FG,
        arrowprops={"arrowstyle": "->", "color": COLOR_FG, "lw": 1.0})

    for ax in (left, right):
        ax.grid(which="both", color=COLOR_GRID, linestyle="--", alpha=0.45,
                zorder=0)
        ax.set_axisbelow(True)
    plt.tight_layout()
    save_figure(output_dir, "enclosed_space_objects.svg")
    plt.close()


def _specular_decay_time(res: object, window: float = 0.004) -> float:
    """Initial slope of the reverberant energy density of an image-source run.

    The estimator the conformance suite uses for the Eyring check: histogram
    the exact reflection table into ``window``-second bins and fit the -1 to
    -20 dB part of the resulting energy decay. Unlike a T30 read from the
    sampled RIR it does not depend on ``max_order``, because it only uses the
    first tenth of a second of arrivals.
    """
    times = np.asarray(res.times)                       # type: ignore[attr-defined]
    amp = np.atleast_1d(np.asarray(res.amplitudes))     # type: ignore[attr-defined]
    if amp.ndim == 2:
        amp = amp[0]
    edges = np.arange(0.0, float(times.max()), window)
    energy, _ = np.histogram(times, bins=edges, weights=amp**2)
    centres = 0.5 * (edges[:-1] + edges[1:])
    good = energy > 0.0
    level = 10.0 * np.log10(np.where(good, energy, 1.0) / energy[good][0])
    band = good & (level <= -1.0) & (level >= -20.0)
    return float(-60.0 / np.polyfit(centres[band], level[band], 1)[0])


def generate_image_source_order_convergence(output_dir: str) -> None:
    """T30 of the synthetic RIR against the reflection-order cut-off.

    One concept: ``max_order`` is a time horizon, and the completeness rule
    is a floor rather than a convergence criterion.
    """
    print("Generating image_source_order_convergence...")
    from phonometry import (
        audible_image_count,
        eyring_reverberation_time,
        image_source_rir,
        room_parameters,
    )

    # The guide's own room: 7 x 5 x 3 m, V = 105 m3, S = 142 m2, alpha = 0.12.
    orders = np.arange(8, 62, 2)
    t30_acc = []
    images = []
    for order in orders:
        res = image_source_rir((7.0, 5.0, 3.0), (2.0, 1.6, 1.5),
                               (5.2, 3.4, 1.7), 0.12, fs=48000,
                               max_order=int(order))
        params = room_parameters(res.ir, res.fs, limits=None)
        t30_acc.append(float(params.t30[0]))
        images.append(float(audible_image_count(int(order))))
    t30 = np.asarray(t30_acc)
    eyring = float(eyring_reverberation_time(105.0, [(142.0, 0.12)]))

    _fig, ax = plt.subplots(figsize=(10, 6.2))
    ax.axhspan(0.9 * eyring, 1.1 * eyring, color=theme_fill(COLOR_TERTIARY, ax),
               zorder=0)
    ax.axhline(eyring, color=COLOR_TERTIARY, linestyle="--", linewidth=1.8,
               label=f"Eyring, {eyring:.2f} s", zorder=4)
    ax.plot(orders, t30, color=COLOR_PRIMARY, marker="o", markersize=5,
            linewidth=2.0, label="T30 from the synthetic RIR", zorder=5)
    crossing = float(np.interp(eyring, t30, orders))
    ax.plot([crossing], [eyring], "o", color=COLOR_SECONDARY, ms=8, zorder=6)
    ax.annotate(f"crosses Eyring at order {crossing:.0f}\nand keeps rising",
                xy=(crossing, eyring), xytext=(0.08, 0.74),
                textcoords="axes fraction", fontsize=9, color=COLOR_FG,
                arrowprops={"arrowstyle": "->", "color": COLOR_FG, "lw": 1.0})
    ax.set_xlabel("Reflection-order cut-off  max_order")
    ax.set_ylabel(r"Fitted $T_{30}$ [s]")
    ax.set_title("The image lattice is a time horizon",
                 pad=12)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=9)

    count = ax.twinx()
    count.plot(orders, images, color=COLOR_MUTED, linestyle=":", linewidth=1.6,
               zorder=3)
    count.set_yscale("log")
    count.set_ylabel("Audible images", color=COLOR_MUTED)
    count.tick_params(axis="y", colors=COLOR_MUTED)

    info = [
        r"room 7 × 5 × 3 m, $\alpha$ = 0.12",
        "$V$ = 105 m³, $S$ = 142 m², $f_s$ = 48 kHz",
        "shaded: ±10 % around Eyring",
    ]
    ax.text(0.985, 0.04, "\n".join(info), transform=ax.transAxes,
            va="bottom", ha="right", fontsize=9, color=COLOR_FG,
            bbox={"boxstyle": "round,pad=0.4", "facecolor": COLOR_PANEL,
                  "edgecolor": COLOR_GRID})
    plt.tight_layout()
    save_figure(output_dir, "image_source_order_convergence.svg")
    plt.close()


def generate_image_source_anisotropy(output_dir: str) -> None:
    """Specular decay against room elongation, at constant volume.

    One concept: where the image-source model leaves the diffuse-field
    estimate it is validated against.
    """
    print("Generating image_source_anisotropy...")
    from phonometry import eyring_reverberation_time, image_source_rir

    volume, alpha = 105.0, 0.12
    side = volume ** (1.0 / 3.0)
    # Four source/receiver pairs per room, as fractions of each dimension, so
    # one unlucky pair cannot set the curve.
    pairs = (((0.28, 0.32, 0.50), (0.74, 0.68, 0.57)),
             ((0.20, 0.55, 0.42), (0.80, 0.30, 0.62)),
             ((0.35, 0.20, 0.55), (0.62, 0.80, 0.40)),
             ((0.15, 0.70, 0.35), (0.85, 0.45, 0.65)))
    ratios = np.array([1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0])
    specular_acc, eyring_acc = [], []
    for ratio in ratios:
        lx = side * ratio ** (2.0 / 3.0)
        ly = side * ratio ** (-1.0 / 3.0)
        area = 2.0 * (2.0 * lx * ly + ly * ly)
        eyring_acc.append(float(eyring_reverberation_time(volume, [(area, alpha)])))
        runs = []
        for source, receiver in pairs:
            res = image_source_rir(
                (lx, ly, ly),
                (lx * source[0], ly * source[1], ly * source[2]),
                (lx * receiver[0], ly * receiver[1], ly * receiver[2]),
                alpha, fs=48000, max_order=60)
            runs.append(_specular_decay_time(res))
        specular_acc.append(float(np.mean(runs)))
    specular = np.asarray(specular_acc)
    eyring = np.asarray(eyring_acc)

    _fig, ax = plt.subplots(figsize=(10, 6.2))
    ax.fill_between(ratios, 0.9 * eyring, 1.1 * eyring,
                    color=theme_fill(COLOR_TERTIARY, ax), zorder=0,
                    label=r"$\pm$ 10 % around Eyring")
    ax.plot(ratios, eyring, color=COLOR_TERTIARY, linestyle="--", linewidth=2.0,
            marker="s", markersize=5, label="Eyring (diffuse field)", zorder=5)
    ax.plot(ratios, specular, color=COLOR_PRIMARY, linewidth=2.2, marker="o",
            markersize=6, label="specular (image source)", zorder=6)
    for ratio, value, reference in zip(ratios, specular, eyring):
        if ratio in (1.0, 3.0, 6.0):
            offset = -0.16 if ratio >= 5.0 else 0.11
            ax.annotate(f"×{value / reference:.2f}", xy=(ratio, value),
                        xytext=(ratio - 0.32, value + offset), fontsize=9,
                        color=COLOR_FG)
    ax.set_xlabel(r"Room elongation  $L_x : L_y = L_z$")
    ax.set_ylabel(r"Reverberation time [s]")
    ax.set_title("Where the specular decay leaves the diffuse-field estimate",
                 pad=12)
    ax.set_ylim(bottom=0.0)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=9)

    info = [
        r"$V$ = 105 m³ and mean $\alpha$ = 0.12 held fixed",
        "cube (1:1) through a 6:1 corridor",
        "mean of 4 source-receiver pairs, max_order = 60",
    ]
    ax.text(0.985, 0.04, "\n".join(info), transform=ax.transAxes,
            va="bottom", ha="right", fontsize=9, color=COLOR_FG,
            bbox={"boxstyle": "round,pad=0.4", "facecolor": COLOR_PANEL,
                  "edgecolor": COLOR_GRID})
    plt.tight_layout()
    save_figure(output_dir, "image_source_anisotropy.svg")
    plt.close()


def generate_image_source_bands(output_dir: str) -> None:
    """Per-band image-source decay, with and without the air term.

    One concept: what the banded branch of ``image_source_rir`` produces,
    and how small the air loss is in a small room.
    """
    print("Generating image_source_bands...")
    from phonometry import air_attenuation_m, decay_curve, image_source_rir

    freqs = [250.0, 500.0, 1000.0, 2000.0, 4000.0]
    alpha = np.array([[0.10, 0.15, 0.25, 0.40, 0.50]] * 6)
    m = air_attenuation_m(freqs, 20.0, 50.0)

    _fig, ax = plt.subplots(figsize=(10, 6.2))
    colours = series_colors(len(freqs))
    for attenuation, dash, tag in ((0.0, "-", ""), (m, "--", " (with air)")):
        banded = image_source_rir((7.0, 5.0, 3.0), (2.0, 1.6, 1.5),
                                  (5.2, 3.4, 1.7), alpha, fs=48000,
                                  max_order=60, frequencies=freqs,
                                  air_attenuation=attenuation)
        for row, freq, colour in zip(banded.ir, freqs, colours):
            time, level = decay_curve(row, banded.fs)
            label = f"{freq:g} Hz{tag}" if not tag else None
            ax.plot(time, level, dash, color=colour, linewidth=1.8,
                    label=label, zorder=5)
    ax.set_xlim(0.0, 1.6)
    ax.set_ylim(-60.0, 3.0)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Level re steady state [dB]")
    ax.set_title("Per-band decay: solid without air, dashed with air",
                 pad=12)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=9)

    info = [
        "room 7 × 5 × 3 m, max_order = 60",
        r"wall $\alpha$ 0.10 → 0.50 with frequency",
        "air at 20 °C / 50 % RH: −0.4 % of T30 at 250 Hz,",
        "−4.4 % at 4 kHz",
    ]
    ax.text(0.015, 0.04, "\n".join(info), transform=ax.transAxes,
            va="bottom", ha="left", fontsize=9, color=COLOR_FG,
            bbox={"boxstyle": "round,pad=0.4", "facecolor": COLOR_PANEL,
                  "edgecolor": COLOR_GRID})
    plt.tight_layout()
    save_figure(output_dir, "image_source_bands.svg")
    plt.close()


def generate_room_proportion_modes(output_dir: str) -> None:
    """Mode ladders of three rooms of the same volume: degeneracy, counted.

    One concept: proportion decides how many *distinct* frequencies a room
    has, which is why a cube is the worst listening room.
    """
    print("Generating room_proportion_modes...")
    from phonometry import room_modes

    volume = 105.0
    half = (volume / 2.0) ** (1.0 / 3.0)
    bolt = (volume / (1.0 * 1.4 * 1.9)) ** (1.0 / 3.0)
    shapes = (
        ("cube", (volume ** (1.0 / 3.0),) * 3),
        ("2 : 1 : 1", (2.0 * half, half, half)),
        ("Bolt 1 : 1.4 : 1.9", (bolt, 1.4 * bolt, 1.9 * bolt)),
    )
    family = {"axial": COLOR_SECONDARY, "tangential": COLOR_TERTIARY,
              "oblique": COLOR_PRIMARY}

    _fig, axes = plt.subplots(4, 1, figsize=(10.5, 7.6), sharex=True,
                             gridspec_kw={"height_ratios": [1, 1, 1, 1.5]})
    for ax, (name, dims) in zip(axes[:3], shapes):
        modes = room_modes(dims, max_frequency=200.0)
        freqs = np.asarray(modes.frequencies)
        kinds = np.asarray(modes.kinds)
        for kind, colour in family.items():
            sel = kinds == kind
            ax.vlines(freqs[sel], 0.0, 1.0, color=colour, linewidth=1.4)
        distinct = int(np.unique(np.round(freqs, 1)).size)
        gap = float(np.max(np.diff(np.unique(np.round(freqs, 1)))))
        ax.set_ylim(0.0, 1.0)
        ax.set_yticks([])
        ax.set_ylabel(f"{name}\n{dims[0]:.2f} × {dims[1]:.2f} × {dims[2]:.2f} m",
                      fontsize=9, rotation=0, ha="right", va="center")
        ax.text(0.99, 0.82, f"{freqs.size} modes, {distinct} distinct "
                            f"frequencies; largest gap {gap:.1f} Hz",
                transform=ax.transAxes, ha="right", va="top", fontsize=9,
                color=COLOR_FG,
                bbox={"boxstyle": "round,pad=0.3", "facecolor": COLOR_PANEL,
                      "edgecolor": COLOR_GRID})

    spacing = axes[3]
    for (name, dims), style in zip(shapes, ("-", "--", ":")):
        modes = room_modes(dims, max_frequency=200.0)
        unique = np.unique(np.round(np.asarray(modes.frequencies), 1))
        spacing.step(unique[1:], np.diff(unique), style, where="post",
                     linewidth=1.8, label=name)
    spacing.set_xlabel(LABEL_FREQ_HZ)
    spacing.set_ylabel("Spacing to the next\ndistinct mode [Hz]")
    spacing.set_xlim(0.0, 200.0)
    spacing.grid(color=COLOR_GRID, linestyle="--", alpha=0.5, zorder=0)
    spacing.set_axisbelow(True)
    spacing.legend(loc="upper right", fontsize=9)

    from matplotlib.lines import Line2D

    handles = [Line2D([], [], color=colour, linewidth=2.0, label=kind)
               for kind, colour in family.items()]
    axes[0].legend(handles=handles, loc="upper left", fontsize=8, ncol=3)
    axes[0].set_title("Three rooms of 105 m³, modes up to 200 Hz",
                      pad=10)
    plt.tight_layout()
    save_figure(output_dir, "room_proportion_modes.svg")
    plt.close()


def generate_steady_state_directivity(output_dir: str) -> None:
    """The steady-state field against Q and against absorption.

    One concept: Q moves the crossover, absorption moves the plateau, and
    neither does the other's job.
    """
    print("Generating steady_state_directivity...")
    from phonometry import steady_state_field

    grid = np.logspace(-1.0, 1.3, 240)
    _fig, (left, right) = plt.subplots(1, 2, figsize=(12.6, 5.6), sharey=True)

    colours = series_colors(4)
    for colour, q in zip(colours, (1.0, 2.0, 4.0, 8.0)):
        field = steady_state_field(sound_power_level=90.0, surface_area=352.0,
                                   mean_absorption=0.15, distances=grid,
                                   directivity=q)
        left.semilogx(field.distances, field.total, color=colour, linewidth=2.0,
                      label=f"$Q$ = {q:g}  ($r_c$ = {field.critical_distance:.2f} m)",
                      zorder=5)
        left.axvline(field.critical_distance, color=colour, linestyle=":",
                     linewidth=1.2, zorder=3)
    left.set_title(r"$Q$ moves $r_c$, not the plateau",
                   pad=10)
    left.legend(loc="upper right", fontsize=9)

    for colour, absorption in zip(colours[::2].tolist() + [colours[3]],
                                  (0.05, 0.15, 0.35)):
        field = steady_state_field(sound_power_level=90.0, surface_area=352.0,
                                   mean_absorption=absorption, distances=grid,
                                   directivity=2.0)
        right.semilogx(
            field.distances, field.total, color=colour, linewidth=2.0,
            label=rf"$\bar\alpha$ = {absorption:g}  ($R$ = "
                  f"{field.room_constant:.0f} m²)", zorder=5)
    right.set_title(r"Absorption moves the plateau, not the direct field",
                    pad=10)
    right.legend(loc="upper right", fontsize=9)
    right.annotate(r"10.1 dB = $10\,\mathrm{lg}(R_2/R_1)$", xy=(8.0, 73.2),
                   xytext=(0.06, 0.22),
                   textcoords="axes fraction", fontsize=9, color=COLOR_FG,
                   arrowprops={"arrowstyle": "->", "color": COLOR_FG, "lw": 1.0})

    for ax in (left, right):
        ax.set_xlabel("Distance from source [m]")
        ax.grid(which="both", color=COLOR_GRID, linestyle="--", alpha=0.45,
                zorder=0)
        ax.set_axisbelow(True)
    left.set_ylabel("Sound pressure level [dB]")

    info = "12 × 8 × 4 m workshop, $S$ = 352 m², $L_W$ = 90 dB re 1 pW"
    left.text(0.015, 0.04, info, transform=left.transAxes, va="bottom",
              ha="left", fontsize=9, color=COLOR_FG,
              bbox={"boxstyle": "round,pad=0.4", "facecolor": COLOR_PANEL,
                    "edgecolor": COLOR_GRID})
    plt.tight_layout()
    save_figure(output_dir, "steady_state_directivity.svg")
    plt.close()


def generate_decay_signatures(output_dir: str) -> None:
    """Three decay shapes and the three diagnoses read off them.

    One concept: what a curvature above 10 % and a collapsed EDT look like.
    """
    print("Generating decay_signatures...")
    from phonometry import decay_curve, room_parameters

    fs = 48000
    rng = np.random.default_rng(3382)
    span = np.arange(int(4.0 * fs)) / fs

    def envelope(t60: float) -> np.ndarray:
        return np.asarray(np.exp(-6.0 * np.log(10.0) / t60 * span))

    noise = rng.standard_normal(span.size)
    tail = noise * np.sqrt(envelope(1.0))
    early = tail.copy()
    amplitude = np.sqrt(6.0 * float(np.sum(tail**2)) / 8.0)
    for k, index in enumerate(np.linspace(0, int(0.025 * fs), 8).astype(int)):
        early[index] += amplitude * 0.85**k * (1.0 if k % 2 == 0 else -1.0)

    cases = (
        ("single slope", tail, "T20 = T30, curvature ~ 0"),
        ("coupled volume", noise * np.sqrt(0.98 * envelope(0.6)
                                           + 0.02 * envelope(2.5)),
         "T30 > T20: report both"),
        ("strong early energy", early, "EDT ≪ T30: a dry seat"),
    )

    _fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.8), sharey=True)
    for ax, (name, signal, verdict) in zip(axes, cases):
        res = room_parameters(signal, fs, limits=None)
        time, level = decay_curve(signal, fs)
        ax.plot(time, level, color=COLOR_PRIMARY, linewidth=1.8, zorder=5)
        for edge, colour in ((-5.0, COLOR_TERTIARY), (-25.0, COLOR_TERTIARY),
                             (-35.0, COLOR_SECONDARY)):
            ax.axhline(edge, color=colour, linestyle=":", linewidth=1.0,
                       zorder=3)
        ax.set_xlim(0.0, 3.0)
        ax.set_ylim(-60.0, 3.0)
        ax.set_xlabel("Time [s]")
        ax.set_title(name, pad=10)
        ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5, zorder=0)
        ax.set_axisbelow(True)
        summary = (f"EDT {float(res.edt[0]):.2f} s\n"
                   f"T20 {float(res.t20[0]):.2f} s\n"
                   f"T30 {float(res.t30[0]):.2f} s\n"
                   f"$C$ = {_fmt_minus(float(res.curvature[0]), '.0f')} %"
                   f"\n{verdict}")
        ax.text(0.97, 0.95, summary, transform=ax.transAxes, ha="right",
                va="top", fontsize=9, color=COLOR_FG,
                bbox={"boxstyle": "round,pad=0.4", "facecolor": COLOR_PANEL,
                      "edgecolor": COLOR_GRID})
    axes[0].set_ylabel("Level re steady state [dB]")
    plt.tight_layout()
    save_figure(output_dir, "decay_signatures.svg")
    plt.close()


def generate_decay_range_bias(output_dir: str) -> None:
    """The truncation-and-compensation bias against the usable decay range.

    One concept: why ISO 3382's 35/45 dB minima are tightened to 46/54 dB.
    """
    print("Generating decay_range_bias...")
    from phonometry import room_parameters

    fs = 48000
    true_t = 1.0
    rng = np.random.default_rng(3382)
    span = np.arange(int(3.0 * fs)) / fs
    clean = rng.standard_normal(span.size) * np.exp(
        -3.0 * np.log(10.0) / true_t * span)
    noise = rng.standard_normal(span.size)
    peak = float(np.max(np.abs(clean)))

    ranges_acc, bias20_acc, bias30_acc = [], [], []
    for inr in range(25, 81, 3):
        res = room_parameters(clean + noise * peak * 10.0 ** (-inr / 20.0),
                              fs, limits=None)
        ranges_acc.append(float(res.dynamic_range[0]))
        bias20_acc.append(100.0 * (float(res.t20[0]) / true_t - 1.0))
        bias30_acc.append(100.0 * (float(res.t30[0]) / true_t - 1.0))
    ranges = np.asarray(ranges_acc)
    bias20 = np.asarray(bias20_acc)
    bias30 = np.asarray(bias30_acc)

    _fig, ax = plt.subplots(figsize=(10, 6.2))
    ax.axhspan(-5.0, 5.0, color=theme_fill(COLOR_TERTIARY, ax), zorder=0)
    ax.axvspan(float(ranges.min()) - 1.0, 46.0,
               color=theme_fill(COLOR_SECONDARY, ax), zorder=0)
    ax.plot(ranges, bias20, color=COLOR_PRIMARY, marker="o", markersize=5,
            linewidth=2.0, label="T20", zorder=5)
    ax.plot(ranges, bias30, color=COLOR_SECONDARY, marker="s", markersize=5,
            linewidth=2.0, label="T30", zorder=5)
    for limit, label, colour in ((35.0, "ISO min. T20", COLOR_MUTED),
                                 (45.0, "ISO min. T30", COLOR_MUTED),
                                 (46.0, "flag T20", COLOR_PRIMARY),
                                 (54.0, "flag T30", COLOR_SECONDARY)):
        ax.axvline(limit, color=colour, linestyle="--", linewidth=1.3, zorder=4)
        ax.text(limit, 9.4, label, rotation=90, fontsize=8, ha="right",
                va="top", color=COLOR_FG)
    ax.set_xlim(30.0, 80.0)
    ax.set_ylim(-1.0, 10.0)
    ax.set_xlabel("Usable decay range  dynamic_range (INR) [dB]")
    ax.set_ylabel(r"Error of the fitted decay time [%]")
    ax.set_title("The bias an undersized decay range leaves behind",
                 pad=12)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="center right", fontsize=9)

    info = [
        "synthetic single-slope decay, $T$ = 1.0 s",
        "white noise floor swept, $f_s$ = 48 kHz",
        "green band: the 5 % JND",
        "red band: flagged invalid for T20",
        "below ~34 dB the fit returns NaN",
    ]
    ax.text(0.40, 0.96, "\n".join(info), transform=ax.transAxes,
            va="top", ha="left", fontsize=9, color=COLOR_FG,
            bbox={"boxstyle": "round,pad=0.4", "facecolor": COLOR_PANEL,
                  "edgecolor": COLOR_GRID})
    plt.tight_layout()
    save_figure(output_dir, "decay_range_bias.svg")
    plt.close()
