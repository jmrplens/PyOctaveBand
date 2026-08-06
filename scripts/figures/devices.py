#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Figures for the devices guides: transducers, sound power and intensity.

The measuring and reproducing hardware: loudspeaker and microphone datasheet
curves, distortion and swept-sine measurements, the sound-power methods
(pressure, reverberation room, intensity scan) and the field indicators and
instrument classes that qualify them. Everything here is embedded by a page
under ``devices/``.
"""

from typing import Any

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
    apply_axis_styling,
    save_figure,
)


def generate_intensity_demo(output_dir: str) -> None:
    """p-p intensity: plane progressive wave vs reactive standing wave."""
    print("Generating intensity_demo.png...")
    from phonometry import sound_intensity

    fs = 48000
    dr, c = 0.012, 343.0
    duration = 4.0
    n = int(fs * duration)

    # Broadband noise, band-limited and scaled to ~70 dB SPL, in pascals.
    rng = np.random.default_rng(2026)
    noise = rng.standard_normal(n)
    sos = scipy_signal.butter(4, [80.0, 6000.0], btype="bandpass", fs=fs, output="sos")
    noise = scipy_signal.sosfilt(sos, noise)
    noise *= 0.063 / np.std(noise)

    spectrum = np.fft.rfft(noise)
    freqs = np.fft.rfftfreq(n, 1 / fs)
    k = 2 * np.pi * freqs / c

    # Plane progressive wave: microphone 2 sees the wave dr/c later.
    p1_plane = noise
    p2_plane = np.fft.irfft(spectrum * np.exp(-2j * np.pi * freqs * dr / c), n)
    plane = sound_intensity(p1_plane, p2_plane, fs, dr, fraction=3, limits=[100.0, 5000.0])

    # Standing wave: equal counter-propagating waves, probe centred at x0.
    x0 = 0.30
    def standing_pressure(pos: float) -> np.ndarray:
        return np.fft.irfft(spectrum * 2.0 * np.cos(k * pos), n)
    standing = sound_intensity(
        standing_pressure(x0 - dr / 2), standing_pressure(x0 + dr / 2),
        fs, dr, fraction=3, limits=[100.0, 5000.0],
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, res, title in [
        (ax1, plane, "Plane wave: Lp ≈ LI"),
        (ax2, standing, "Standing wave: reactive field"),
    ]:
        ax.semilogx(res.frequency, res.pressure_level, marker="o", markersize=5,
                    color=COLOR_PRIMARY, linewidth=1.5, markerfacecolor="white",
                    markeredgewidth=1.3, label="Pressure level Lp")
        ax.semilogx(res.frequency, res.intensity_level, marker="s", markersize=5,
                    color=COLOR_SECONDARY, linewidth=1.5, linestyle="--",
                    markerfacecolor="white", markeredgewidth=1.3,
                    label="Intensity level LI")
        apply_axis_styling(ax, title, xlim=(90, 5600), ylim=(0, 85))
        # The standard octave ticks extend past the band range: re-clamp.
        ax.set_xlim(90, 5600)
        dpi_db = round(float(res.total_pressure_intensity_index), 1) + 0.0
        ax.text(0.05, 0.33, f"Pressure-intensity index\nδpI = {dpi_db:.1f} dB",
                transform=ax.transAxes, fontsize=10, va="bottom", color=COLOR_FG)
        ax.legend(loc="upper right", fontsize=9)
    ax2.set_ylabel("")

    fig.suptitle("Sound Intensity with a p-p Probe (IEC 61043)", fontweight="bold")
    plt.tight_layout()
    save_figure(output_dir, "intensity_demo.png")
    plt.close()


def generate_sound_reinforcement_geometry(output_dir: str) -> None:
    """The four points of a reinforcement feedback loop, schematically.

    One concept: the distances that set the two direct-field levels
    ``L(H-M)`` and ``L(H-L)`` of Long's stability criterion. The layout is
    schematic and each path carries its own length, because a talker 0.3 m
    from the microphone and a listener 12 m from the loudspeaker cannot share
    one usable drawing scale.
    """
    print("Generating sound_reinforcement_geometry...")
    from phonometry import plot_sound_reinforcement_geometry

    _fig, ax = plt.subplots(figsize=(10, 4.6))
    plot_sound_reinforcement_geometry(0.3, 4.0, 12.0, ax=ax, language=_LANG)
    plt.tight_layout()
    save_figure(output_dir, "sound_reinforcement_geometry.svg")
    plt.close()


_FS_ELECTRO = 48000  # audio sample rate for the electroacoustic demos


def generate_distortion(output_dir: str) -> None:
    """Annotated harmonic spectrum with the THD of a synthetic amplifier output."""
    print("Generating distortion...")
    from phonometry import harmonic_analysis

    fs = _FS_ELECTRO
    n = fs  # 1 s -> 1 Hz bins; every harmonic lands on a bin
    t = np.arange(n) / fs
    f0 = 1000.0
    # A 1 kHz fundamental with a decaying harmonic series over a broadband
    # noise floor, the kind of output an amplifier under a single-tone test
    # produces. The noise makes THD+N exceed the harmonic-only THD (it also
    # counts the noise), while SINAD reports the noise-and-distortion headroom.
    amps = {1: 1.0, 2: 0.02, 3: 0.012, 4: 0.006, 5: 0.003}
    sig = sum(a * np.sin(2 * np.pi * k * f0 * t) for k, a in amps.items())
    rng = np.random.default_rng(2026)
    sig = sig + rng.standard_normal(n) * 1.2e-2

    res = harmonic_analysis(sig, fs, f0, n_harmonics=len(amps))

    # Magnitude spectrum (coherent-gain normalised) in dB re the fundamental.
    window = np.hanning(n)
    spectrum = np.abs(np.fft.rfft(sig * window)) * 2.0 / np.sum(window)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    ref = np.max(spectrum)
    spec_db = 20.0 * np.log10(np.maximum(spectrum, 1e-12) / ref)

    _fig, ax = plt.subplots(figsize=(10, 6.0))
    ax.plot(freqs, spec_db, color=COLOR_PRIMARY, linewidth=1.0, alpha=0.8,
            label="Magnitude spectrum")
    hz = np.asarray(res.harmonic_frequencies)
    ha = np.asarray(res.harmonic_amplitudes)
    hdb = 20.0 * np.log10(np.maximum(ha, 1e-12) / ha[0])
    ax.plot(hz, hdb, "o", color=COLOR_SECONDARY, markersize=7, zorder=6,
            label="Harmonics n·f₁")
    for k, (fk, lk) in enumerate(zip(hz, hdb), start=1):
        ax.annotate(f"n={k}", xy=(fk, lk), xytext=(0, 7),
                    textcoords="offset points", ha="center", fontsize=8,
                    color=COLOR_FG)

    ax.set_xlim(0.0, 9000.0)
    ax.set_ylim(-100.0, 8.0)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Level re fundamental [dB]")
    ax.set_title("Harmonic Distortion of a Single-Tone Test (IEC 60268-3)",
                 fontweight="bold", pad=12)
    ax.grid(which="major", color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=9)

    info = [
        f"THD (F) = {res.thd_f * 100:.2f}%",
        f"THD (R) = {res.thd_r * 100:.2f}%",
        f"THD+N   = {res.thd_plus_noise * 100:.2f}%",
        f"SINAD   = {res.sinad_db:.1f} dB",
    ]
    ax.text(0.015, 0.03, "\n".join(info), transform=ax.transAxes,
            va="bottom", ha="left", fontsize=9, color=COLOR_FG,
            family="monospace",
            bbox={"boxstyle": "round,pad=0.5", "facecolor": COLOR_PANEL,
                  "edgecolor": COLOR_GRID})
    plt.tight_layout()
    save_figure(output_dir, "distortion.svg")
    plt.close()


def generate_frequency_response(output_dir: str) -> None:
    """Bode magnitude and coherence of an estimated frequency response (H1)."""
    print("Generating frequency_response...")
    from scipy import signal as sp_signal

    from phonometry import transfer_function

    fs = _FS_ELECTRO
    n = 400000
    rng = np.random.default_rng(7)
    x = rng.standard_normal(n)
    # A resonant second-order band-pass "device under test".
    b, a = sp_signal.butter(2, [400.0, 4000.0], btype="band", fs=fs)
    y = sp_signal.lfilter(b, a, x)
    # Additive output noise pulls the coherence down where the signal is weak.
    y = y + rng.standard_normal(n) * np.sqrt(np.mean(y**2)) * 0.05

    res = transfer_function(x, y, fs, estimator="H1")
    _, h_true = sp_signal.freqz(b, a, worN=res.frequencies, fs=fs)
    pos = res.frequencies > 0.0
    freqs = res.frequencies[pos]
    true_db = 20.0 * np.log10(np.maximum(np.abs(h_true[pos]), 1e-12))

    _fig, (ax_mag, ax_coh) = plt.subplots(
        2, 1, figsize=(10, 7.2), sharex=True,
        gridspec_kw={"height_ratios": [2.0, 1.0]})
    ax_mag.semilogx(freqs, true_db, color=COLOR_FG, linestyle="--",
                    linewidth=1.6, alpha=0.7, label="True |H|")
    ax_mag.semilogx(freqs, res.magnitude_db[pos], color=COLOR_PRIMARY,
                    linewidth=1.8, label="Estimated |H| (H1)")
    ax_mag.set_ylabel("Magnitude [dB]")
    ax_mag.set_ylim(-80.0, 5.0)
    ax_mag.set_title("Frequency Response and Coherence (Bendat & Piersol)",
                     fontweight="bold", pad=12)
    ax_mag.grid(which="both", color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax_mag.set_axisbelow(True)
    ax_mag.legend(loc="lower center", fontsize=9)

    ax_coh.semilogx(freqs, res.coherence[pos], color=COLOR_TERTIARY,
                    linewidth=1.8)
    ax_coh.set_ylabel(r"Coherence $\gamma^2$")
    ax_coh.set_xlabel("Frequency [Hz]")
    ax_coh.set_ylim(0.0, 1.05)
    ax_coh.set_xlim(20.0, fs / 2.0)
    ax_coh.grid(which="both", color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax_coh.set_axisbelow(True)
    for _axf in (ax_mag, ax_coh):
        format_frequency_axis(_axf, 20.0, fs / 2.0)
    plt.tight_layout()
    save_figure(output_dir, "frequency_response.svg")
    plt.close()


def generate_swept_sine_thd(output_dir: str) -> None:
    """THD(f) by order from one synchronized sweep (Farina / Novak)."""
    print("Generating swept_sine_thd...")
    from scipy import signal as sp_signal

    from phonometry import swept_sine_distortion, synchronized_sweep_signal

    fs = 48000
    f1, f2, seconds = 20.0, 6000.0, 4.0
    a2, a3 = 0.12, 0.08
    # Hammerstein chain: a memoryless cubic polynomial (exact Chebyshev
    # harmonic levels) followed by a 3 kHz low-pass, so each order rolls off
    # where its own product n*f crosses the filter corner.
    x = synchronized_sweep_signal(fs, f1, f2, seconds)
    b, a = sp_signal.butter(2, 3000.0, fs=fs)
    y = sp_signal.lfilter(b, a, x + a2 * x**2 + a3 * x**3)
    res = swept_sine_distortion(y, fs, f1, f2, seconds, n_harmonics=3)

    sel = (res.thd_frequencies >= 30.0) & (res.thd_frequencies <= 2800.0)
    freqs = res.thd_frequencies[sel]
    h1_ref = 1.0 + 3.0 * a3 / 4.0  # Chebyshev fundamental gain

    _fig, ax = plt.subplots(figsize=(10, 6))
    ax.loglog(freqs, 100.0 * res.thd[sel], color=COLOR_PRIMARY,
              linewidth=2.0, label="Total THD(f)")
    ax.loglog(freqs, 100.0 * res.distortion_ratios[0][sel],
              color=COLOR_SECONDARY, linewidth=1.5, linestyle="--",
              label="2nd harmonic d₂(f)")
    ax.loglog(freqs, 100.0 * res.distortion_ratios[1][sel],
              color=COLOR_TERTIARY, linewidth=1.5, linestyle="--",
              label="3rd harmonic d₃(f)")
    ax.axhline(100.0 * (a2 / 2.0) / h1_ref, color=COLOR_SECONDARY,
               linestyle=":", linewidth=1.2, alpha=0.8,
               label="Chebyshev asymptote (a₂/2)/H₁")
    ax.axhline(100.0 * (a3 / 4.0) / h1_ref, color=COLOR_TERTIARY,
               linestyle=":", linewidth=1.2, alpha=0.8,
               label="Chebyshev asymptote (a₃/4)/H₁")
    ax.set_xlabel("Excitation frequency [Hz]")
    ax.set_ylabel("Distortion re fundamental [%]")
    ax.set_title("Swept-Sine Harmonic Distortion by Order (Farina / Novak)",
                 fontweight="bold", pad=12)
    ax.set_xlim(30.0, 2800.0)
    ax.set_ylim(0.05, 20.0)
    format_frequency_axis(ax, 30.0, 2800.0)
    ax.grid(which="both", color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", fontsize=9)
    ax.text(0.985, 0.965,
            "one sweep separates every distortion order;\n"
            "each rolls off where its product n·f crosses the 3 kHz corner",
            transform=ax.transAxes, va="top", ha="right", fontsize=8.5,
            color=COLOR_FG)
    plt.tight_layout()
    save_figure(output_dir, "swept_sine_thd.svg")
    plt.close()


def generate_piston_directivity(output_dir: str) -> None:
    """Far-field beam pattern of a baffled circular piston at three ka values."""
    print("Generating piston_directivity...")
    from phonometry import piston_directivity_pattern

    piston_directivity_pattern([3.0, 8.0, 16.0]).plot(language=_LANG)
    save_figure(output_dir, "piston_directivity.svg")
    plt.close()


def _loudspeaker_datasheet_example() -> Any:
    """The IEC 60268-5 loudspeaker result shared by the section-7 .plot() figures."""
    from phonometry import (
        LoudspeakerDirectivity,
        loudspeaker_characteristics,
        radiating_piston,
    )

    freqs = np.geomspace(30, 24000, 320)
    spl = 87.0 + 1.2 * np.sin(2 * np.log2(freqs / 900.0))
    spl -= 10 * np.log10(1 + (50.0 / freqs) ** 6)       # low-frequency roll-off
    spl -= 10 * np.log10(1 + (freqs / 16000.0) ** 7)    # high-frequency roll-off
    fz = np.geomspace(20, 20000, 260)
    thd_f = np.geomspace(50, 5000, 140)
    return loudspeaker_characteristics(
        freqs, spl, rated_impedance=8.0, sensitivity_band=(200.0, 4000.0),
        impedance=(fz, 6.6 + 24 * np.exp(-(np.log2(fz / 52.0) ** 2) / 0.12)),
        distortion=(thd_f, 0.3 + 2.6 * np.exp(-(np.log2(thd_f / 70.0) ** 2) / 0.45)),
        directivity=LoudspeakerDirectivity(
            piston=radiating_piston(0.075, np.array([1000.0, 2000.0, 4000.0]),
                                    angles=np.radians(np.linspace(0, 90, 46))),
            frequency=2000.0,
        ),
    )


def _microphone_datasheet_example() -> Any:
    """The IEC 60268-4 microphone result shared by the section-8 .plot() figures."""
    from phonometry import (
        MicrophoneDirectivity,
        MicrophoneElectrical,
        MicrophoneNoise,
        MicrophoneOverload,
        microphone_characteristics,
    )

    freqs = np.geomspace(20, 20000, 400)
    response = -10 * np.log10(1 + (30.0 / freqs) ** 4)      # low-frequency roll-off
    response -= 10 * np.log10(1 + (freqs / 19000.0) ** 8)   # high-frequency roll-off
    response += 2.0 * np.exp(-(np.log2(freqs / 9000.0) ** 2) / 0.3)  # presence region
    angles = np.linspace(0, 179, 359)
    cardioid = 20 * np.log10((1 + np.cos(np.radians(angles))) / 2)
    noise_f = np.geomspace(20, 20000, 31)
    spl_axis = np.linspace(100, 140, 81)
    return microphone_characteristics(
        freqs, response, 12.5, tolerance_db=3.0,          # 12.5 mV/Pa at 1 kHz
        directivity=MicrophoneDirectivity(polar=(angles, cardioid), frequency=1000.0),
        noise=MicrophoneNoise(
            voltage=1.25e-6,
            spectrum=(noise_f, 6.0 + 12.0 * np.log10(1000.0 / noise_f)),
        ),
        overload=MicrophoneOverload(
            distortion=(spl_axis, 0.5 * 10 ** ((spl_axis - 130.0) * 0.08)),
            thd_percent=0.5,
        ),
        electrical=MicrophoneElectrical(
            rated_impedance=150.0, minimum_load_impedance=1000.0,
            powering="Phantom P48 (IEC 61938)", supply_current_ma=3.1,
        ),
    )


def generate_loudspeaker_response(output_dir: str) -> None:
    """On-axis SPL response with tolerance band and effective range (IEC 60268-5)."""
    print("Generating loudspeaker_response...")
    _loudspeaker_datasheet_example().plot(quantity="response", language=_LANG)
    save_figure(output_dir, "loudspeaker_response.svg")
    plt.close()


def generate_loudspeaker_impedance(output_dir: str) -> None:
    """Impedance modulus with the rated and 80 %-of-rated lines (IEC 60268-5)."""
    print("Generating loudspeaker_impedance...")
    _loudspeaker_datasheet_example().plot(quantity="impedance", language=_LANG)
    save_figure(output_dir, "loudspeaker_impedance.svg")
    plt.close()


def generate_loudspeaker_thd(output_dir: str) -> None:
    """Total harmonic distortion against frequency (IEC 60268-5)."""
    print("Generating loudspeaker_thd...")
    _loudspeaker_datasheet_example().plot(quantity="thd", language=_LANG)
    save_figure(output_dir, "loudspeaker_thd.svg")
    plt.close()


def generate_loudspeaker_directivity(output_dir: str) -> None:
    """Polar directivity on the IEC 60263 25 dB reference circle (IEC 60268-5)."""
    print("Generating loudspeaker_directivity...")
    _loudspeaker_datasheet_example().plot(quantity="directivity", language=_LANG)
    save_figure(output_dir, "loudspeaker_directivity.svg")
    plt.close()


def generate_microphone_response(output_dir: str) -> None:
    """Free-field response with tolerance band and reference markers (IEC 60268-4)."""
    print("Generating microphone_response...")
    _microphone_datasheet_example().plot(quantity="response", language=_LANG)
    save_figure(output_dir, "microphone_response.svg")
    plt.close()


def generate_microphone_directivity(output_dir: str) -> None:
    """Cardioid directional pattern on the 25 dB reference circle (IEC 60268-4)."""
    print("Generating microphone_directivity...")
    _microphone_datasheet_example().plot(quantity="directivity", language=_LANG)
    save_figure(output_dir, "microphone_directivity.svg")
    plt.close()


def generate_microphone_noise(output_dir: str) -> None:
    """Inherent-noise equivalent band-level spectrum (IEC 60268-4)."""
    print("Generating microphone_noise...")
    _microphone_datasheet_example().plot(quantity="noise", language=_LANG)
    save_figure(output_dir, "microphone_noise.svg")
    plt.close()


def generate_microphone_distortion(output_dir: str) -> None:
    """Total harmonic distortion against sound pressure level (IEC 60268-4)."""
    print("Generating microphone_distortion...")
    _microphone_datasheet_example().plot(quantity="distortion", language=_LANG)
    save_figure(output_dir, "microphone_distortion.svg")
    plt.close()


def generate_program_loudness(output_dir: str) -> None:
    """EBU R 128 metering of a synthetic programme: M, S, I and LRA."""
    print("Generating program_loudness...")
    from scipy import signal as sp_signal

    from phonometry import program_loudness

    fs = 48000
    rng = np.random.default_rng(1770)
    # A one-minute synthetic programme: quiet ambience, two dialogue
    # passages around the -23 LUFS target, a louder music sting and a
    # fade-out. Pink-ish noise stands in for programme material.
    sections = [("ambience", -38.0, 8.0), ("dialogue", -23.0, 16.0),
                ("music", -17.0, 12.0), ("dialogue", -25.0, 16.0),
                ("fade-out", -45.0, 8.0)]
    sos = sp_signal.butter(2, 2000.0, fs=fs, output="sos")
    chunks = []
    for _, level, duration in sections:
        n = int(duration * fs)
        noise = sp_signal.sosfilt(sos, rng.standard_normal(n))
        noise /= np.sqrt(np.mean(noise**2))
        # Slow, aperiodic amplitude modulation so the momentary trace
        # breathes like real programme material.
        t = np.arange(n) / fs
        wobble = 1.0 + 0.22 * np.sin(2.0 * np.pi * 0.9 * t) \
            + 0.14 * np.sin(2.0 * np.pi * 2.83 * t + 1.0)
        chunks.append(10.0 ** (level / 20.0) * noise * wobble)
    x = np.concatenate(chunks)
    # Loudness-normalise the programme to the R 128 target of -23.0 LUFS,
    # exactly what a broadcast workflow does before delivery.
    raw = program_loudness(np.vstack([x, x]), fs)
    x *= 10.0 ** ((-23.0 - raw.integrated) / 20.0)
    res = program_loudness(np.vstack([x, x]), fs)

    _fig, ax = plt.subplots(figsize=(11.5, 5.8))
    ax.axhspan(res.lra_low, res.lra_high, color=theme_fill(COLOR_TERTIARY, ax),
               zorder=0,
               label=f"LRA = {res.loudness_range:.1f} LU (P10-P95)")
    ax.plot(res.momentary_time, res.momentary, color="#9e9e9e",
            linewidth=0.8, label="Momentary M (400 ms)")
    ax.plot(res.short_term_time, res.short_term, color=COLOR_PRIMARY,
            linewidth=2.2, label="Short-term S (3 s)")
    ax.axhline(res.integrated, color=COLOR_SECONDARY, linestyle="--",
               linewidth=1.8, label=f"Integrated I = {res.integrated:.1f} LUFS")

    # Label the programme sections along the top.
    t0 = 0.0
    for name, _, duration in sections:
        ax.text(t0 + duration / 2.0, -11.0, name, ha="center", va="top",
                fontsize=8.5, color=COLOR_FG, alpha=0.75, style="italic")
        t0 += duration
        if t0 < 60.0:
            ax.axvline(t0, color=COLOR_GRID, linewidth=0.8)

    info = [
        f"I     = {res.integrated:6.1f} LUFS",
        f"LRA   = {res.loudness_range:6.1f} LU",
        f"max M = {res.max_momentary:6.1f} LUFS",
        f"max S = {res.max_short_term:6.1f} LUFS",
        f"TPmax = {res.true_peak:6.1f} dBTP",
    ]
    ax.text(0.985, 0.03, "\n".join(info), transform=ax.transAxes,
            va="bottom", ha="right", fontsize=8.5, color=COLOR_FG,
            family="monospace",
            bbox={"boxstyle": "round,pad=0.5", "facecolor": COLOR_PANEL,
                  "edgecolor": COLOR_GRID})

    ax.set_xlim(0.0, 60.0)
    ax.set_ylim(-58.0, -10.0)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Loudness [LUFS]")
    ax.set_title("Programme Loudness Metering (EBU R 128)",
                 fontweight="bold", pad=12)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", fontsize=9)
    plt.tight_layout()
    save_figure(output_dir, "program_loudness.svg")
    plt.close()


def generate_k_weighting_response(output_dir: str) -> None:
    """K-weighting magnitude frequency response (ITU-R BS.1770-5 Annex 1)."""
    print("Generating k_weighting_response...")
    from phonometry import k_weighting_response

    _fig, ax = plt.subplots(figsize=(10, 6))
    k_weighting_response().plot(ax=ax, language=_LANG)
    plt.tight_layout()
    save_figure(output_dir, "k_weighting_response.svg")
    plt.close()


def generate_vibration_sound_power(output_dir: str) -> None:
    """ISO/TS 7849 sound power from surface vibration: upper limit vs engineering."""
    print("Generating vibration_sound_power...")
    from phonometry import radiated_sound_power_level

    bands = np.array([125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0])
    # A plausible surface velocity level spectrum and a measured radiation factor.
    lv = np.array([78.0, 82.0, 85.0, 83.0, 79.0, 74.0])
    eps = np.array([0.20, 0.45, 0.75, 0.95, 1.00, 1.00])
    area = 1.6

    lw_max = radiated_sound_power_level(lv, area)                    # Part 1, eps=1
    lw_eng = radiated_sound_power_level(lv, area, radiation_factor=eps)  # Part 2

    x = np.arange(bands.size)
    _fig, ax = plt.subplots(figsize=(10, 6.2))
    ax.bar(x - 0.2, lw_max, width=0.4, color=COLOR_SECONDARY, edgecolor=COLOR_FG,
           linewidth=0.6, label="Part 1 upper limit ($\\varepsilon$ = 1)")
    ax.bar(x + 0.2, lw_eng, width=0.4, color=COLOR_PRIMARY, edgecolor=COLOR_FG,
           linewidth=0.6, label="Part 2 engineering ($\\varepsilon$ measured)")

    total_max = 10.0 * np.log10(np.sum(10.0 ** (0.1 * lw_max)))
    total_eng = 10.0 * np.log10(np.sum(10.0 ** (0.1 * lw_eng)))
    ax.axhline(total_max, color=COLOR_SECONDARY, ls="--", lw=1.2,
               label=f"total (limit) {total_max:.1f} dB")
    ax.axhline(total_eng, color=COLOR_PRIMARY, ls="--", lw=1.2,
               label=f"total (eng.) {total_eng:.1f} dB")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{b:g}" for b in bands])
    ax.set_xlabel(LABEL_FREQ_HZ)
    ax.set_ylabel(r"Sound power level $L_W$ [dB re 1 pW]")
    ax.set_title("ISO/TS 7849 Sound Power from Surface Vibration",
                 fontweight="bold", pad=12)
    ax.grid(which="major", axis="y", color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", fontsize=9)

    info = [
        "LW = Lv + 10 log10(S/S0) + 10 log10(e) + 10 log10(411/400)",
        f"S = {area:g} m2,  S0 = 1 m2",
        "Part 1: e = 1 -> upper limit LW,max",
    ]
    ax.text(0.015, 0.02, "\n".join(info), transform=ax.transAxes,
            va="bottom", ha="left", fontsize=9, color=COLOR_FG, family="monospace",
            bbox={"boxstyle": "round,pad=0.5", "facecolor": COLOR_PANEL,
                  "edgecolor": COLOR_GRID})
    plt.tight_layout()
    save_figure(output_dir, "vibration_sound_power.svg")
    plt.close()


def generate_expansion_chamber_geometry(output_dir: str) -> None:
    """To-scale cross-section of the 4:1 expansion chamber of the TL figure.

    The 0,3 m chamber with 0,04 m2 over a 0,01 m2 pipe, drawn with the
    equivalent circular diameters. One concept: the geometry behind the
    expansion-chamber transmission-loss curve.
    """
    print("Generating expansion_chamber_geometry...")
    from phonometry import plot_silencer_geometry

    _fig, ax = plt.subplots(figsize=(10, 6.2))
    plot_silencer_geometry(
        "expansion chamber", ax=ax, length=0.3, chamber_area=0.04,
        pipe_area=0.01, language=_LANG,
    )
    plt.tight_layout()
    save_figure(output_dir, "expansion_chamber_geometry.svg")
    plt.close()


def generate_helmholtz_branch_geometry(output_dir: str) -> None:
    """To-scale cross-section of the side-branch Helmholtz resonator.

    The resonator of the side-branch TL figure (1 cm2 neck of 2 cm on a
    1 L cavity over a 0,01 m2 duct), cavity drawn as its equal-volume cube.
    One concept: the branch geometry that shorts the duct at its tuning.
    """
    print("Generating helmholtz_branch_geometry...")
    from phonometry import plot_silencer_geometry

    _fig, ax = plt.subplots(figsize=(10, 6.2))
    plot_silencer_geometry(
        "Helmholtz resonator", ax=ax, duct_area=0.01, neck_area=1e-4,
        neck_length=0.02, cavity_volume=1e-3, language=_LANG,
    )
    plt.tight_layout()
    save_figure(output_dir, "helmholtz_branch_geometry.svg")
    plt.close()


def generate_quarter_wave_geometry(output_dir: str) -> None:
    """To-scale cross-section of the quarter-wave side branch.

    The 0,3 m closed tube (20 cm2) of the side-branch TL figure on the same
    0,01 m2 duct. One concept: a quarter-wave stub is just a closed tube of
    the right length.
    """
    print("Generating quarter_wave_geometry...")
    from phonometry import plot_silencer_geometry

    _fig, ax = plt.subplots(figsize=(10, 6.2))
    plot_silencer_geometry(
        "quarter-wave resonator", ax=ax, duct_area=0.01, length=0.3,
        branch_area=2e-3, language=_LANG,
    )
    plt.tight_layout()
    save_figure(output_dir, "quarter_wave_geometry.svg")
    plt.close()


def generate_microphone_positions_hemisphere(output_dir: str) -> None:
    """The ISO 3744 engineering hemisphere microphone array in 3-D.

    The tabulated 10-microphone array on a 2 m hemisphere over one
    reflecting plane, numbered as in the standard. One concept: where the
    sound-power microphones actually sit.
    """
    print("Generating microphone_positions_hemisphere...")
    from phonometry import measurement_positions, plot_microphone_positions

    positions = measurement_positions("hemisphere", radius=2.0)
    plt.figure(figsize=(9.0, 7.0))
    ax = plt.subplot(projection="3d")
    plot_microphone_positions(positions, ax=ax, radius=2.0, language=_LANG)
    plt.tight_layout()
    save_figure(output_dir, "microphone_positions_hemisphere.svg")
    plt.close()


def generate_piston_baffle_geometry(output_dir: str) -> None:
    """The baffled piston to scale with its high-frequency lobe.

    A 10 cm piston in an infinite baffle with the normalised far-field
    directivity overlaid at ka about 7 (where the main lobe has narrowed).
    One concept: the piston the radiation-impedance and directivity curves
    describe.
    """
    print("Generating piston_baffle_geometry...")
    from phonometry import radiating_piston

    result = radiating_piston(
        0.1, np.array([500.0, 2000.0, 4000.0]),
        angles=np.linspace(-np.pi / 2.0, np.pi / 2.0, 181),
    )
    _fig, ax = plt.subplots(figsize=(9.0, 6.2))
    result.plot_geometry(ax=ax, language=_LANG)
    plt.tight_layout()
    save_figure(output_dir, "piston_baffle_geometry.svg")
    plt.close()


def generate_plenum_geometry(output_dir: str) -> None:
    """Plenum chamber section honouring the acoustic geometry exactly.

    A 1,2 m line of sight at 20 degrees off the inlet axis, a 0,09 m2
    outlet and 6 m2 of lined walls: the two geometric parameters of the
    attenuation formula are drawn exactly and the areas annotated. One
    concept: what the plenum formula actually measures.
    """
    print("Generating plenum_geometry...")
    from phonometry import plot_plenum_geometry

    _fig, ax = plt.subplots(figsize=(10, 6.2))
    plot_plenum_geometry(0.09, 1.2, 6.0, ax=ax, angle=0.35, language=_LANG)
    plt.tight_layout()
    save_figure(output_dir, "plenum_geometry.svg")
    plt.close()


def generate_radiation_plate_geometry(output_dir: str) -> None:
    """The baffled plate of the radiation-efficiency model, to scale.

    The 1,5 x 1,25 m simply supported plate in its rigid baffle, the
    geometry the sigma(f) curves describe. One concept: the radiator
    behind the radiation-efficiency plateau and coincidence peak.
    """
    print("Generating radiation_plate_geometry...")
    from phonometry import plot_plate_geometry

    _fig, ax = plt.subplots(figsize=(9.0, 6.2))
    plot_plate_geometry(1.5, 1.25, ax=ax, language=_LANG)
    plt.tight_layout()
    save_figure(output_dir, "radiation_plate_geometry.svg")
    plt.close()


def generate_pp_probe_geometry(output_dir: str) -> None:
    """The face-to-face p-p intensity probe to scale.

    Two phase-matched half-inch microphones on the classic 12 mm solid
    spacer, the intensity axis through both. One concept: the finite
    difference the p-p method is built on.
    """
    print("Generating pp_probe_geometry...")
    from phonometry import plot_pp_probe_geometry

    _fig, ax = plt.subplots(figsize=(9.0, 5.4))
    plot_pp_probe_geometry(ax=ax, language=_LANG)
    plt.tight_layout()
    save_figure(output_dir, "pp_probe_geometry.svg")
    plt.close()


def generate_sound_power_pressure_result(output_dir: str) -> None:
    """ISO 3744: enveloping-surface LW spectrum from hemisphere pressure levels."""
    print("Generating sound_power_pressure_result.png...")
    from phonometry import RoomEnvironment, sound_power_pressure

    # The sound-power guide's section-1 example: octave-band SPL at the 10
    # hemisphere positions of ISO 3744 (Annex B) around a machine on one
    # reflecting plane, with a flat 55 dB background, corrected for background
    # (K1) and for the test room (K2 from T = 0.6 s, V = 300 m^3). The library
    # forms LW = Lp_bar - K1 - K2 + 10 log10(S/S0) per band and the A-weighted
    # total LWA.
    freqs = np.array([63, 125, 250, 500, 1000, 2000, 4000, 8000], dtype=float)
    base = np.array([70.0, 74.0, 78.0, 80.0, 79.0, 76.0, 72.0, 66.0])
    rng = np.random.default_rng(0)
    levels = base + rng.normal(0.0, 0.5, size=(10, 8))
    background = np.full((10, 8), 55.0)
    result = sound_power_pressure(
        levels, "hemisphere", radius=1.5, reflecting_planes=1,
        background_levels=background, frequencies=freqs,
        room=RoomEnvironment(reverberation_time=0.6, volume=300.0),
    )

    lw = result.sound_power_level
    lwa = result.sound_power_level_a
    positions = np.arange(freqs.size, dtype=float)
    _fig, ax = plt.subplots(figsize=(10, 6.3))
    ax.bar(positions, lw, width=0.7, color=COLOR_PRIMARY, edgecolor=COLOR_FG,
           linewidth=0.7, zorder=3)
    ax.set_xticks(positions)
    ax.set_xticklabels([f"{f:g}" for f in freqs], rotation=45, ha="right")
    ax.set_title(f"Enveloping-surface sound power (ISO 3744)  LWA = {lwa:.1f} dB(A)",
                 fontweight="bold", pad=12)
    ax.set_xlabel(LABEL_FREQ_HZ)
    ax.set_ylabel("Sound power level LW [dB]")
    ax.set_ylim(0.0, float(np.nanmax(lw)) + 8.0)
    ax.grid(axis="y", color=COLOR_GRID, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    plt.tight_layout()
    save_figure(output_dir, "sound_power_pressure_result.png")
    plt.close()


def generate_sound_power_reverberation_result(output_dir: str) -> None:
    """ISO 3741: reverberation-room LW spectrum (direct method)."""
    print("Generating sound_power_reverberation_result.png...")
    from phonometry.emission import sound_power_reverberation

    # The sound-power guide's section-2 example: one-third-octave mean room
    # SPL from 100 Hz to 10 kHz in a qualified 200 m^3 reverberation room with
    # T60 = 2 s, carried to LW through the Sabine absorption area, the
    # Waterhouse correction and the meteorological corrections C1/C2
    # (ISO 3741 Eq. 20).
    freqs = np.array([100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000,
                      1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000,
                      10000], dtype=float)
    lp = np.linspace(80.0, 70.0, freqs.size)
    t60 = np.full(freqs.size, 2.0)
    result = sound_power_reverberation(
        lp, t60, volume=200.0, surface_area=220.0, frequencies=freqs,
        temperature=20.0, static_pressure=101.0,
    )

    lw = result.sound_power_level
    lwa = result.sound_power_level_a
    positions = np.arange(freqs.size, dtype=float)
    _fig, ax = plt.subplots(figsize=(10, 6.3))
    ax.bar(positions, lw, width=0.7, color=COLOR_PRIMARY, edgecolor=COLOR_FG,
           linewidth=0.7, zorder=3)
    ax.set_xticks(positions)
    ax.set_xticklabels([f"{f:g}" for f in freqs], rotation=45, ha="right")
    ax.set_title(
        f"Reverberation-room sound power (ISO 3741)  LWA = {lwa:.1f} dB(A)",
        fontweight="bold", pad=12)
    ax.set_xlabel(LABEL_FREQ_HZ)
    ax.set_ylabel("Sound power level LW [dB]")
    ax.set_ylim(0.0, float(np.nanmax(lw)) + 8.0)
    ax.grid(axis="y", color=COLOR_GRID, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    plt.tight_layout()
    save_figure(output_dir, "sound_power_reverberation_result.png")
    plt.close()


def generate_sound_power_intensity_result(output_dir: str) -> None:
    """ISO 9614-2: intensity-scanning LW spectrum from segment sweeps."""
    print("Generating sound_power_intensity_result.png...")
    from phonometry.emission import sound_power_intensity

    # The sound-power guide's section-3 example: two repeated intensity sweeps
    # over 6 surface segments and 6 octave bands, with the segment surface SPL
    # and the probe's pressure-residual intensity index. The partial powers
    # In_i * Si sum to the band LW; every band passes the field-indicator
    # criteria at engineering grade here (no SoundPowerWarning fires).
    freqs = np.array([125, 250, 500, 1000, 2000, 4000], dtype=float)
    areas = np.full(6, 0.5)
    rng = np.random.default_rng(0)
    scan1 = np.abs(rng.normal(1e-4, 2e-5, size=(6, 6)))
    scan2 = scan1 * (1.0 + rng.normal(0.0, 0.02, size=(6, 6)))
    pressure = np.full((6, 6), 80.0)
    result = sound_power_intensity(
        scan1, areas, normal_intensity_2=scan2, pressure_levels=pressure,
        pressure_residual_index=12.0, frequencies=freqs,
        band_type="octave", grade="engineering",
    )

    lw = result.sound_power_level
    lwa = result.sound_power_level_a
    positions = np.arange(freqs.size, dtype=float)
    _fig, ax = plt.subplots(figsize=(10, 6.3))
    # Plot only the determinable (finite-LW) bands; an undeterminable band
    # (net inflow -> NaN) is left as a gap rather than faked to 0 dB. All six
    # bands are finite with this synthetic data, so this is future-proofing.
    finite = np.isfinite(lw)
    ax.bar(positions[finite], lw[finite], width=0.7, color=COLOR_PRIMARY,
           edgecolor=COLOR_FG, linewidth=0.7, zorder=3)
    ax.set_xticks(positions)
    ax.set_xticklabels([f"{f:g}" for f in freqs], rotation=45, ha="right")
    ax.set_title(
        f"Intensity-scanning sound power (ISO 9614-2)  LWA = {lwa:.1f} dB(A)",
        fontweight="bold", pad=12)
    ax.set_xlabel(LABEL_FREQ_HZ)
    ax.set_ylabel("Sound power level LW [dB]")
    ax.set_ylim(0.0, float(np.nanmax(lw)) + 8.0)
    ax.grid(axis="y", color=COLOR_GRID, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    plt.tight_layout()
    save_figure(output_dir, "sound_power_intensity_result.png")
    plt.close()


def generate_precision_anechoic_power(output_dir: str) -> None:
    """ISO 3745: precision LW spectrum from a hemisphere pressure measurement."""
    print("Generating precision_anechoic_power.png...")
    from phonometry import sound_power_anechoic

    # A mid-frequency-peaked machine measured over the 40-position hemisphere
    # array (ISO 3745 Annex E) in a hemi-anechoic room. levels_positions is the
    # (40, NB) surface pressure spectrum: a base machine spectrum peaked near
    # 1 kHz plus a small per-position spatial variation. The library forms the
    # surface-averaged LW = Lp_bar + 10 log10(S/S0) + C1+C2+C3 and the A-weighted
    # total LWA.
    freqs = np.array([125, 250, 500, 1000, 2000, 4000, 8000], dtype=float)
    base = 70.0 + 8.0 * np.exp(-(np.log2(freqs / 1000.0) ** 2) / 2.0)
    rng = np.random.default_rng(7)
    levels = base[None, :] + rng.normal(0.0, 1.0, (40, freqs.size))
    result = sound_power_anechoic(levels, "hemisphere", radius=1.0,
                                  frequencies=freqs)

    lw = result.sound_power_level
    lwa = result.sound_power_level_a
    positions = np.arange(freqs.size, dtype=float)
    _fig, ax = plt.subplots(figsize=(10, 6.3))
    ax.bar(positions, lw, width=0.7, color=COLOR_PRIMARY, edgecolor=COLOR_FG,
           linewidth=0.7, zorder=3)
    ax.set_xticks(positions)
    ax.set_xticklabels([f"{f:g}" for f in freqs], rotation=45, ha="right")
    ax.set_title(f"Precision sound power (ISO 3745)  LWA = {lwa:.1f} dB(A)",
                 fontweight="bold", pad=12)
    ax.set_xlabel(LABEL_FREQ_HZ)
    ax.set_ylabel("Sound power level LW [dB]")
    ax.set_ylim(0.0, float(np.nanmax(lw)) + 8.0)
    ax.grid(axis="y", color=COLOR_GRID, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    plt.tight_layout()
    save_figure(output_dir, "precision_anechoic_power.png")
    plt.close()


def generate_intensity_scan_power(output_dir: str) -> None:
    """ISO 9614-3: precision LW spectrum by intensity scanning (with a NaN band)."""
    print("Generating intensity_scan_power.png...")
    import warnings

    from phonometry import sound_power_intensity_precision

    # Four partial surfaces scanned over five one-third-octave bands. Each cell
    # of partial_intensity is the signed normal intensity In_i (W/m^2) already
    # reduced to the two-scan result; areas are the partial-surface areas Si.
    # The 250 Hz band has net-negative power (more energy flowing in than out),
    # so ISO 9614-3 flags it not-applicable (clause 9.2) and it is hatched.
    freqs = np.array([250, 500, 1000, 2000, 4000], dtype=float)
    areas = np.array([0.5, 1.0, 0.75, 0.5])
    base_intensity = np.array([2.0e-6, 8.0e-6, 2.0e-5, 1.0e-5, 3.0e-6])
    per_segment = np.array([1.0, 1.1, 0.9, 1.05])
    partial_intensity = base_intensity[None, :] * per_segment[:, None]
    # A locally reactive 250 Hz band: the segment intensities cancel to a
    # net-negative total.
    partial_intensity[:, 0] = np.array([2.0e-6, -3.0e-6, -4.0e-6, -1.0e-6])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = sound_power_intensity_precision(partial_intensity, areas,
                                                 frequencies=freqs)

    lw = result.sound_power_level
    neg = result.not_applicable_band
    lwa = result.sound_power_level_a
    positions = np.arange(freqs.size, dtype=float)
    _fig, ax = plt.subplots(figsize=(10, 6.3))
    # Determinate bands: a solid LW bar. Non-applicable bands carry no LW (NaN),
    # so instead of a zero-height bar they are flagged by a full-height greyed,
    # hatched span - clearly a marker, not a plotted value (ISO 9614-3, 9.2).
    ax.bar(positions[~neg], np.nan_to_num(lw)[~neg], width=0.7, color=COLOR_PRIMARY,
           edgecolor=COLOR_FG, linewidth=0.7, zorder=3)
    for pos, is_neg in zip(positions, neg):
        if is_neg:
            ax.axvspan(pos - 0.35, pos + 0.35, facecolor="#888888", alpha=0.28,
                       hatch="//", edgecolor="#888888", linewidth=0.8, zorder=2)
    ax.set_xticks(positions)
    ax.set_xticklabels([f"{f:g}" for f in freqs], rotation=45, ha="right")
    ax.set_title(f"Precision intensity scanning (ISO 9614-3)  LWA = {lwa:.1f} dB(A)",
                 fontweight="bold", pad=12)
    ax.set_xlabel(LABEL_FREQ_HZ)
    ax.set_ylabel("Sound power level LW [dB]")
    ax.set_ylim(0.0, float(np.nanmax(lw)) + 8.0)
    from matplotlib.patches import Patch
    handle = Patch(facecolor="#888888", alpha=0.28, hatch="//",
                   edgecolor="#888888", label="Non-applicable band")
    ax.legend(handles=[handle], loc="upper right", fontsize=9)
    ax.grid(axis="y", color=COLOR_GRID, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    plt.tight_layout()
    save_figure(output_dir, "intensity_scan_power.png")
    plt.close()


def generate_silencer_expansion_chamber(output_dir: str) -> None:
    """Expansion-chamber transmission loss for four area ratios (Bies 8.111)."""
    print("Generating silencer_expansion_chamber.svg...")
    from phonometry import expansion_chamber

    freqs = np.linspace(20.0, 2000.0, 2000)
    pipe_area, length = 0.01, 0.3
    ratios = (2.0, 4.0, 8.0, 16.0)
    colors = (COLOR_PRIMARY, COLOR_SECONDARY, COLOR_TERTIARY, "#9467bd")

    _fig, ax = plt.subplots(figsize=(9.0, 5.2))
    for m, color in zip(ratios, colors):
        res = expansion_chamber(freqs, length, m * pipe_area, pipe_area)
        peak = 10.0 * np.log10(1.0 + 0.25 * (m - 1.0 / m) ** 2)
        ax.plot(freqs, res.transmission_loss, color=color, lw=1.8,
                label=f"m = {int(m)}  →  {peak:.1f} dB")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Transmission loss [dB]")
    ax.set_title("Expansion-chamber transmission loss (Bies Eq. 8.111)",
                 fontweight="bold", pad=10)
    ax.set_xlim(20.0, 2000.0)
    ax.set_ylim(0.0, 20.0)
    format_frequency_axis(ax, 20.0, 2000.0)
    ax.grid(True, which="both", alpha=0.4)
    ax.legend(loc="upper right", fontsize="small", title="Area ratio m = Sexp/Sduct")
    plt.tight_layout()
    save_figure(output_dir, "silencer_expansion_chamber.svg")
    plt.close()
def generate_modulation_distortion(output_dir: str) -> None:
    """IEC 60268-3 14.12.7 modulation sidebands via ModulationDistortionResult.plot()."""
    print("Generating modulation_distortion.svg...")
    from phonometry import modulation_distortion

    # The standard two-tone test: a large low tone f1 = 60 Hz and a small high
    # tone f2 = 7 kHz (4:1), captured at the output of a weakly non-linear
    # amplifier (a memoryless polynomial stands in for the device). One second
    # at 48 kHz puts every tone on an FFT bin (coherent sampling).
    fs = 48000
    t = np.arange(fs) / fs
    x = 1.0 * np.sin(2.0 * np.pi * 60.0 * t) + 0.25 * np.sin(2.0 * np.pi * 7000.0 * t)
    y = x + 0.04 * x**2 + 0.012 * x**3

    res = modulation_distortion(y, fs, 60.0, 7000.0)
    _fig, ax = plt.subplots(figsize=(10, 6))
    # The result's own .plot() draws the carrier (0 dB reference) and the four
    # modulation sidebands at f2 +/- f1 and f2 +/- 2 f1, annotated with the
    # per-order d2/d3 and the SMPTE combined RMS.
    res.plot(ax=ax)
    ax.set_xlim(6600.0, 7400.0)
    plt.tight_layout()
    save_figure(output_dir, "modulation_distortion.svg")
    plt.close()


def generate_piston_radiation_impedance(output_dir: str) -> None:
    """Baffled-piston R1/X1 against ka via RadiatingPistonResult.plot()."""
    print("Generating piston_radiation_impedance.svg...")
    from phonometry import radiating_piston

    # A 75 mm-radius piston (a typical mid-woofer cone) over the audio band:
    # ka runs from well below 0.1 (mass-controlled) past 10 (resistive).
    res = radiating_piston(radius=0.075, frequencies=np.geomspace(20.0, 20000.0, 400))
    _fig, ax = plt.subplots(figsize=(10, 6))
    # The result's own .plot() draws the normalized radiation resistance R1 and
    # reactance X1 against ka (the classic Beranek & Mellow figure).
    res.plot(ax=ax)
    plt.tight_layout()
    save_figure(output_dir, "piston_radiation_impedance.svg")
    plt.close()


def generate_field_indicators(output_dir: str) -> None:
    """ISO 9614-1 F2/F3/F4 per band vs dynamic capability via FieldIndicators.plot()."""
    print("Generating field_indicators.svg...")
    from phonometry import emission

    # A 10-position discrete-point scan (ISO 9614-1) of a machine in an
    # ordinary room. The surface pressure is nearly uniform; the normal
    # intensity per band is set so the pressure-intensity gap widens towards
    # low frequency (the field turns reactive), with two inward-flowing
    # positions in the 125 Hz band (F3 rises above F2 there).
    freqs = np.array([125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0])
    delta_pi = np.array([10.5, 8.5, 6.0, 4.5, 3.5, 3.0])   # target Lp - L|In| per band
    rng = np.random.default_rng(9614)
    lp = 78.0 + rng.normal(0.0, 0.4, (10, freqs.size))
    i_mean = 10.0 ** ((78.0 - delta_pi) / 10.0) * 1.0e-12
    i_n = i_mean[None, :] * (1.0 + rng.normal(0.0, 0.18, (10, freqs.size)))
    # Two positions see net inward flow in the 125 Hz band; rescale the rest so
    # the algebraic band mean keeps its target (the band stays determinable).
    i_n[:2, 0] = -0.35 * i_mean[0]
    i_n[2:, 0] *= (10.0 * i_mean[0] - i_n[:2, 0].sum()) / i_n[2:, 0].sum()

    fi = emission.field_indicators(lp, i_n, freqs)
    ld = emission.dynamic_capability_index(18.0)   # delta_pI0 = 18 dB, K = 10 dB
    _fig, ax = plt.subplots(figsize=(10, 6))
    # The result's own .plot() draws F2 and F3 per band, the dynamic
    # capability Ld (criterion 1: adequate where Ld > F2) and F4 on a twin axis.
    fi.plot(ax=ax, dynamic_capability=ld)
    plt.tight_layout()
    save_figure(output_dir, "field_indicators.svg")
    plt.close()


def generate_intensity_class(output_dir: str) -> None:
    """Measured delta_pI0 of a p-p chain over the IEC 61043 Table 2 masks."""
    print("Generating intensity_class.svg...")
    from phonometry import metrology

    # A complete instrument verified with the common 12 mm spacer, so the
    # printed 25 mm minima come down by 10 log10(12/25) = -3,2 dB (Table 2,
    # Note 1). The measured index is modelled from the physics behind the
    # table: a residual channel phase mismatch phi_s turns into
    # delta_pI0 = 10 log10(kd/phi_s), so a mismatch that is constant in degrees
    # already buys 10 dB per decade of index - which is exactly the slope of
    # the Table 2 requirement below 250 Hz. Above 1 kHz the mismatch of a real
    # chain grows with frequency instead of staying put, so the index levels
    # off where the requirement does.
    spacing = 0.012
    freqs, _, _ = metrology.residual_index_limits("instrument", spacing=spacing)
    phase_mismatch = 0.05 * np.maximum(1.0, freqs / 1000.0)   # degrees
    measured = metrology.residual_index_from_phase_mismatch(
        phase_mismatch, freqs, spacing
    )
    # A vent resonance of the capsules costs 4 dB around 100 Hz, the one band
    # that drops out of class 1 and drags the whole verdict down to class 2.
    measured = measured - 4.0 * np.exp(-(((np.log(freqs / 100.0)) / 0.25) ** 2))

    result = metrology.intensity_class_compliance(measured, freqs, spacing=spacing)
    _fig, ax = plt.subplots(figsize=(10, 6))
    # The result's own .plot() shades the pass region of the achieved class and
    # rings the bands that cost the chain the next class up.
    result.plot(ax=ax, language=_LANG)
    plt.tight_layout()
    save_figure(output_dir, "intensity_class.svg")
    plt.close()


def generate_silencer_side_branch(output_dir: str) -> None:
    """Helmholtz and quarter-wave side branches shorting the duct at tuning."""
    print("Generating silencer_side_branch.svg...")
    from phonometry import helmholtz_resonator, quarter_wave_resonator

    freqs = np.linspace(20.0, 600.0, 4000)
    hr = helmholtz_resonator(freqs, duct_area=0.01, neck_area=1e-4,
                             neck_length=0.02, cavity_volume=1e-3)
    qw = quarter_wave_resonator(freqs, duct_area=0.01, length=0.3,
                                branch_area=2e-3)

    _fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(freqs, hr.transmission_loss, color=COLOR_PRIMARY, lw=1.8,
            label="Helmholtz resonator")
    ax.plot(freqs, qw.transmission_loss, color=COLOR_SECONDARY, lw=1.8,
            ls="--", label="Quarter-wave tube")
    for resonances in (hr.resonances, qw.resonances):
        if resonances is not None:
            ax.axvline(float(resonances[0]), color=COLOR_TERTIARY, ls=":", lw=1.0)
    ax.set_xlim(20.0, 600.0)
    ax.set_ylim(0.0, 50.0)
    format_frequency_axis(ax, 20.0, 600.0)
    ax.set_xlabel(LABEL_FREQ_HZ)
    ax.set_ylabel("Transmission loss [dB]")
    ax.set_title("Side-branch resonators: transmission loss (Bies Eqs. 8.44, 8.46)",
                 fontweight="bold", pad=10)
    ax.grid(True, which="both", alpha=0.4)
    ax.legend(loc="upper right", fontsize="small")
    plt.tight_layout()
    save_figure(output_dir, "silencer_side_branch.svg")
    plt.close()


def generate_hvac_end_reflection(output_dir: str) -> None:
    """ASHRAE duct end reflection loss for three diameters (HvacSpectrumResult)."""
    print("Generating hvac_end_reflection.svg...")
    from phonometry.noise_control import hvac

    bands = [63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0]
    _fig, ax = plt.subplots(figsize=(10, 6))
    for diameter, color in ((0.15, COLOR_PRIMARY), (0.30, COLOR_SECONDARY),
                            (0.60, COLOR_TERTIARY)):
        er = hvac.end_reflection_loss(bands, diameter=diameter,
                                      termination="flush")
        ax.semilogx(np.asarray(er.frequencies), np.asarray(er.values), "o-",
                    color=color, lw=1.8, ms=4,
                    label=f"D = {int(diameter * 1000)} mm")
    ax.set_xlim(50.0, 2500.0)
    ax.set_ylim(bottom=0.0)
    format_frequency_axis(ax, 50.0, 2500.0)
    ax.set_xlabel(LABEL_FREQ_HZ)
    ax.set_ylabel("End reflection loss [dB]")
    ax.set_title("Duct end reflection loss (ASHRAE Table 8.14)",
                 fontweight="bold", pad=10)
    ax.grid(True, which="both", alpha=0.4)
    ax.legend(loc="upper right", fontsize="small", title="Duct diameter")
    plt.tight_layout()
    save_figure(output_dir, "hvac_end_reflection.svg")
    plt.close()


def _long_duct_paths() -> tuple[Any, Any]:
    """The supply and return paths of Long's worked HVAC sheet (Table 14.9).

    Every row is the one printed in the sheet, including the manufacturer data
    for the silencers and the terminal devices, so the figure shows what the
    published calculation delivers into the room rather than a re-derivation.
    """
    from phonometry import DuctElement, duct_path

    bands = [63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0]
    fan = [90.0, 86.0, 82.0, 79.0, 77.0, 75.0, 71.0, 61.0]
    source = "Fan, centrifugal FC, 5000 cfm, 2 in w.g."
    supply = duct_path(
        bands, fan,
        [
            DuctElement("Elbow, 36 x 24 in, unlined",
                        [0, 1, 2, 3, 3, 3, 3, 3],
                        [41, 39, 36, 29, 20, 6, 0, 0], code="2"),
            DuctElement("Silencer, 3 ft, standard pressure drop",
                        [7, 12, 16, 28, 35, 35, 28, 17],
                        [49, 43, 44, 42, 42, 45, 35, 24], code="3"),
            DuctElement("Duct, 36 x 24 in, 5 ft, 1 in lining",
                        [2, 2, 3, 7, 15, 12, 11, 9], code="4"),
            DuctElement("Split, 25 per cent", 6.0, code="5"),
            DuctElement("Duct, 18 x 12 in, 6 ft, 1 in lining",
                        [3, 3, 5, 11, 25, 22, 16, 13], code="6"),
            DuctElement("Flexible duct, 12 in, 6 ft",
                        [14, 14, 16, 15, 17, 22, 16, 13], code="7"),
            DuctElement("Rectangular diffuser, 312 cfm", None,
                        [33, 32, 29, 23, 15, 4, 0, 0], code="8"),
        ],
        room_effect=[6, 6, 5, 5, 6, 7, 6, 6],
        source_label=source, target=30.0, label="Supply",
    )
    ret = duct_path(
        bands, fan,
        [
            DuctElement("Elbow, 36 x 24 in, unlined",
                        [0, 1, 2, 3, 3, 3, 3, 3],
                        [43, 42, 39, 33, 24, 12, 0, 0], code="2"),
            DuctElement("Silencer, 5 ft, low-frequency type",
                        [16, 21, 35, 41, 41, 28, 21, 15],
                        [51, 49, 53, 56, 56, 59, 60, 53], code="3"),
            DuctElement("Elbow, 36 x 24 in, lined, 1 in",
                        [1, 2, 3, 4, 5, 6, 8, 10],
                        [39, 38, 34, 28, 18, 4, 0, 0], code="4"),
            DuctElement("Plenum, 800 sq ft, 50 per cent lined",
                        [12, 13, 19, 20, 20, 20, 21, 21], code="5"),
            DuctElement("Rectangular grille, 24 x 24 in, 563 cfm", None,
                        [30, 29, 26, 20, 12, 1, 0, 0], code="6"),
        ],
        room_effect=[9, 8, 6, 8, 8, 8, 9, 10],
        source_label=source, target=30.0, label="Return",
    )
    return supply, ret


def generate_duct_path_cascade(output_dir: str) -> None:
    """Long Table 14.9 supply + return duct paths against NC 30."""
    print("Generating duct_path_cascade.svg...")
    from phonometry import combine_duct_paths

    supply, ret = _long_duct_paths()
    total = combine_duct_paths([supply, ret], label="Supply + return")

    _fig, ax = plt.subplots(figsize=(10, 6))
    # The result's own .plot() draws each contributing path, the received
    # spectrum and the design criterion curve.
    total.plot(ax=ax, language=_LANG)
    ax.set_ylim(-6.0, 62.0)
    ax.set_title("Duct-borne noise into the room: supply, return and NC 30",
                 fontweight="bold", pad=10)
    plt.tight_layout()
    save_figure(output_dir, "duct_path_cascade.svg")
    plt.close()


def _norton_plant_room_chain() -> Any:
    """Norton problem 4.18: a blower in a plant room, into the operator room.

    Every number is the one printed in the problem statement, so the figure
    shows the published calculation rather than a re-derivation: the blower
    sound power level, the ceiling, floor and wall absorption of the
    8 x 10 x 3 m plant room, the transmission loss of the separating wall and
    the carpeted floor of the 5 x 5 x 3 m operator room.
    """
    from phonometry import (
        DesignCriterion,
        SourceRoom,
        equivalent_absorption_area,
        mean_absorption,
        room_constant,
        room_to_room_transmission,
    )

    bands = [125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0]
    ceiling = [0.07, 0.20, 0.40, 0.52, 0.60, 0.67]
    walls = [0.03, 0.03, 0.03, 0.04, 0.05, 0.07]
    plant = [
        (80.0, [0.01, 0.01, 0.015, 0.02, 0.02, 0.02]),
        (80.0, ceiling),
        (108.0, walls),
    ]
    operator = [
        (25.0, [0.08, 0.24, 0.57, 0.69, 0.71, 0.73]),
        (25.0, ceiling),
        (60.0, walls),
    ]
    return room_to_room_transmission(
        bands,
        [39.0, 42.0, 50.0, 58.0, 63.0, 67.0],
        5.0 * 3.0,
        equivalent_absorption_area(operator),
        source=SourceRoom(
            power_level=[105.0, 103.0, 98.0, 108.0, 107.0, 109.0],
            room_constant=room_constant(268.0, mean_absorption(plant)),
            # The blower sits on the floor along the middle of a wall (Q = 4)
            # and the problem asks for a conservative estimate, i.e. the
            # constant-volume sound power model of Norton Table 4.5.
            directivity=4.0,
            model="constant_volume",
        ),
        criterion=DesignCriterion(target=45.0),
        label="Plant room to operator room",
    )


def generate_room_to_room_chain(output_dir: str) -> None:
    """Norton problem 4.18: plant room to operator room against NC 45."""
    print("Generating room_to_room_chain.svg...")
    result = _norton_plant_room_chain()

    _fig, ax = plt.subplots(figsize=(10, 6))
    # The result's own .plot() draws both reverberant spectra, the criterion
    # curve and the band-by-band noise reduction on the twin axis.
    result.plot(ax=ax, language=_LANG)
    ax.set_ylim(20.0, 115.0)
    ax.set_title(
        "Plant room to operator room: what the wall delivers, and NC 45",
        fontweight="bold", pad=10,
    )
    plt.tight_layout()
    save_figure(output_dir, "room_to_room_chain.svg")
    plt.close()


def generate_duct_mode_cut_on(output_dir: str) -> None:
    """Higher-order-mode cut-on ladder of a 254 mm steam line at 200 m/s."""
    print("Generating duct_mode_cut_on.svg...")
    from phonometry import circular_duct_cut_on

    # Norton & Karczub problem 7.1: a 254 mm circular duct carrying steam
    # (c = 405 m/s) at a mean flow velocity of 200 m/s, i.e. M = 0.494, where
    # the sqrt(1 - M^2) shift separates the two ladders visibly.
    modes = circular_duct_cut_on(0.254, flow_velocity=200.0,
                                 speed_of_sound=405.0, count=6)

    _fig, ax = plt.subplots(figsize=(10, 6))
    # The result's own .plot() draws the still-air ladder, the with-flow
    # ladder and the plane-wave band below the first cut-on.
    modes.plot(ax=ax, language=_LANG)
    ax.set_title("Duct higher-order-mode cut-on: 254 mm steam line at 200 m/s",
                 fontweight="bold", pad=10)
    plt.tight_layout()
    save_figure(output_dir, "duct_mode_cut_on.svg")
    plt.close()


def generate_enclosure_insertion_loss(output_dir: str) -> None:
    """Machine-enclosure IL = R - C per band via EnclosureResult.plot()."""
    print("Generating enclosure_insertion_loss.svg...")
    from phonometry import enclosure_insertion_loss

    # A measured panel transmission loss combined with a modest interior
    # lining (mean absorption 0.3): the reverberant build-up inside the small
    # hard cavity spends part of the panel R.
    bands = np.array([125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0])
    panel_r = np.array([18.0, 24.0, 30.0, 36.0, 42.0, 46.0])
    enc = enclosure_insertion_loss(
        panel_r, external_area=6.0, internal_area=5.0,
        internal_absorption=0.3, frequencies=bands,
    )
    _fig, ax = plt.subplots(figsize=(10, 6))
    # The result's own .plot() draws the panel R, the interior correction C
    # and the net insertion loss IL = R - C.
    enc.plot(ax=ax)
    plt.tight_layout()
    save_figure(output_dir, "enclosure_insertion_loss.svg")
    plt.close()


def generate_phase_decomposition(output_dir: str) -> None:
    """Minimum-phase / all-pass split of a delayed band-pass response."""
    print("Generating phase_decomposition.svg...")
    from scipy import signal as sp_signal

    from phonometry import phase_decomposition

    # A strictly minimum-phase equalizer response (an RBJ +6 dB peaking biquad
    # at 1 kHz) measured through a 2.5 ms latency (a digital processing
    # delay): the minimum-phase part carries what an equalizer can invert, the
    # all-pass excess carries the pure delay, read directly off the flat
    # excess group delay.
    fs = 48000.0
    delay = int(0.0025 * fs)                      # 2.5 ms
    gain_a = 10.0 ** (6.0 / 40.0)                 # +6 dB peaking gain
    w0 = 2.0 * np.pi * 1000.0 / fs
    alpha = np.sin(w0) / (2.0 * 1.0)              # Q = 1
    b = np.array([1.0 + alpha * gain_a, -2.0 * np.cos(w0), 1.0 - alpha * gain_a])
    a = np.array([1.0 + alpha / gain_a, -2.0 * np.cos(w0), 1.0 - alpha / gain_a])
    imp = np.zeros(16384)
    imp[delay] = 1.0
    ir = sp_signal.lfilter(b / a[0], a / a[0], imp)

    res = phase_decomposition(np.fft.rfft(ir), fs)
    # The result's own .plot() draws three stacked panels: |H|, the measured /
    # minimum / excess phases and the total and excess group delays.
    res.plot()
    plt.gcf().set_size_inches(9.0, 8.0)
    plt.tight_layout()
    save_figure(output_dir, "phase_decomposition.svg")
    plt.close()


def _r128_noise_programme(sections: list[tuple[float, float]], fs: int) -> np.ndarray:
    """Deterministic band-limited noise programme from (level dBFS, seconds)."""
    from scipy import signal as sp_signal

    rng = np.random.default_rng(3341)
    sos = sp_signal.butter(2, 2000.0, fs=fs, output="sos")
    chunks = []
    for level, seconds in sections:
        noise = sp_signal.sosfilt(sos, rng.standard_normal(int(seconds * fs)))
        noise /= np.sqrt(np.mean(noise**2))
        chunks.append(10.0 ** (level / 20.0) * noise)
    return np.concatenate(chunks)


def generate_loudness_gating(output_dir: str) -> None:
    """BS.1770 gating: a long quiet tail does not drag the integrated loudness."""
    print("Generating loudness_gating.svg...")
    from phonometry import program_loudness

    # 20 s of programme material on the -23 LUFS target followed by 40 s of
    # quiet room ambience ~29 LU lower: the -70 LUFS absolute gate keeps
    # everything, but the relative gate (10 LU below the survivors) drops the
    # tail, so the integrated loudness stays on the foreground while the
    # ungated mean sinks. The programme is loudness-normalised to -23.0 LUFS
    # first, exactly as a delivery workflow would.
    fs = 48000
    from phonometry import integrated_loudness

    x = _r128_noise_programme([(-23.0, 20.0), (-52.0, 40.0)], fs)
    x *= 10.0 ** ((-23.0 - integrated_loudness(np.vstack([x, x]), fs)) / 20.0)
    res = program_loudness(np.vstack([x, x]), fs)

    _fig, ax = plt.subplots(figsize=(10.5, 5.8))
    # The result's own .plot() draws the momentary and short-term traces, the
    # integrated line and the LRA band.
    res.plot(ax=ax)
    finite = res.momentary[np.isfinite(res.momentary)]
    ungated = 10.0 * np.log10(np.mean(10.0 ** (finite / 10.0)))
    ax.axhline(ungated, color=COLOR_TERTIARY, ls="-.", lw=1.6,
               label=f"Ungated mean {ungated:.1f} LUFS")
    ax.legend(loc="center right", fontsize=9)
    plt.tight_layout()
    save_figure(output_dir, "loudness_gating.svg")
    plt.close()


def generate_loudness_range(output_dir: str) -> None:
    """EBU Tech 3342 loudness range on its two-step reference case (LRA = 10 LU)."""
    print("Generating loudness_range.svg...")
    from phonometry import program_loudness

    # EBU Tech 3342 test case 1: 20 s at -20 dBFS then 20 s at -30 dBFS. The
    # short-term distribution has two plateaus 10 LU apart, and the P10-P95
    # spread reads exactly LRA = 10.0 LU (the shaded band of the plot).
    fs = 48000
    t = np.arange(20 * fs) / fs
    tone = np.sin(2.0 * np.pi * 1000.0 * t)
    x = np.concatenate([10.0 ** (-20.0 / 20.0) * tone,
                        10.0 ** (-30.0 / 20.0) * tone])
    res = program_loudness(np.vstack([x, x]), fs)

    _fig, ax = plt.subplots(figsize=(10.5, 5.8))
    # The result's own .plot() shades the loudness range between its 10th and
    # 95th percentile edges under the momentary / short-term / integrated traces.
    res.plot(ax=ax)
    plt.tight_layout()
    save_figure(output_dir, "loudness_range.svg")
