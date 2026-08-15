#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Figures for the devices guides: transducers, sound power and intensity.

The measuring and reproducing hardware: loudspeaker and microphone datasheet
curves, distortion and swept-sine measurements, the sound-power methods
(pressure, reverberation room, intensity scan) and the field indicators and
instrument classes that qualify them. Everything here is embedded by a page
under ``devices/``.
"""

import re
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from scipy import signal as scipy_signal

from phonometry._plot.common import format_frequency_axis, theme_fill, theme_line

from .i18n import _LANG, _fmt_minus, lookup
from .theme import (
    COLOR_FG,
    COLOR_GRID,
    COLOR_MUTED,
    COLOR_PANEL,
    COLOR_PRIMARY,
    COLOR_SECONDARY,
    COLOR_TERTIARY,
    LABEL_FREQ_HZ,
    apply_axis_styling,
    measure_weighting_response,
    save_figure,
)

# A negative reading a library plot assembled with ``format()``: the ASCII
# hyphen before a digit, in a position no word or bracket claims. Restated to
# the typographic U+2212 the axis formatters ship (the criterion of the
# corpus-wide sweep over rendered SVG text).
_ASCII_MINUS_RE = re.compile(r"(?<![\w)\]])-(?=\d)")


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
        (ax1, plane, "Plane wave: $L_p \\approx L_I$"),
        (ax2, standing, "Standing wave: reactive field"),
    ]:
        ax.semilogx(res.frequency, res.pressure_level, marker="o", markersize=5,
                    color=COLOR_PRIMARY, linewidth=1.5, markerfacecolor="white",
                    markeredgewidth=1.3, label="Pressure level $L_p$")
        ax.semilogx(res.frequency, res.intensity_level, marker="s", markersize=5,
                    color=COLOR_SECONDARY, linewidth=1.5, linestyle="--",
                    markerfacecolor="white", markeredgewidth=1.3,
                    label="Intensity level $L_I$")
        apply_axis_styling(ax, title, xlim=(90, 5600), ylim=(0, 85))
        # The standard octave ticks extend past the band range: re-clamp.
        ax.set_xlim(90, 5600)
        dpi_db = round(float(res.total_pressure_intensity_index), 1) + 0.0
        ax.text(0.05, 0.33,
                f"Pressure-intensity index\n"
                f"$\\delta_{{pI}}$ = {_fmt_minus(dpi_db, '.1f')} dB",
                transform=ax.transAxes, fontsize=10, va="bottom", color=COLOR_FG)
        ax.legend(loc="upper right", fontsize=9)
    ax2.set_ylabel("")

    fig.suptitle("Sound Intensity with a p-p Probe (IEC 61043)")
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
            label="Harmonics $n f_1$")
    for k, (fk, lk) in enumerate(zip(hz, hdb), start=1):
        ax.annotate(f"$n$ = {k}", xy=(fk, lk), xytext=(0, 7),
                    textcoords="offset points", ha="center", fontsize=8,
                    color=COLOR_FG)

    ax.set_xlim(0.0, 9000.0)
    ax.set_ylim(-100.0, 8.0)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Level re fundamental [dB]")
    ax.set_title("Harmonic Distortion of a Single-Tone Test (IEC 60268-3)",
                 pad=12)
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
    """H1 against H2 with the noise on each channel in turn, and coherence.

    Bendat & Piersol's rule is that the unbiased estimator is the one whose
    noisy channel sits in the denominator, and that the coherence floor of an
    output-noise measurement is SNR/(1 + SNR). Both are statements a single
    H1 curve cannot make: the same record has to be analysed twice, and the
    noise has to be moved from one channel to the other.
    """
    print("Generating frequency_response...")
    from scipy import signal as sp_signal

    from phonometry import transfer_function

    fs = _FS_ELECTRO
    n = 400000
    rng = np.random.default_rng(7)
    clean_x = rng.standard_normal(n)
    # A resonant second-order band-pass "device under test".
    b, a = sp_signal.butter(2, [400.0, 4000.0], btype="band", fs=fs)
    clean_y = sp_signal.lfilter(b, a, clean_x)
    noise_frac = 0.05

    # Case 1: the noise is on the output, which is the usual acoustic case.
    y_out = clean_y + rng.standard_normal(n) * np.sqrt(
        np.mean(clean_y**2)) * noise_frac
    # Case 2: the same path with the noise on the reference channel instead.
    # A larger fraction, because an input-noise bias is uniform across the
    # band and a 5 % one would be invisible at this scale.
    noise_frac_in = 0.5
    x_in = clean_x + rng.standard_normal(n) * np.sqrt(
        np.mean(clean_x**2)) * noise_frac_in

    cases = (
        ("Noise on the output — $H_1$ is unbiased", clean_x, y_out),
        ("Noise on the input — $H_2$ is unbiased", x_in, clean_y),
    )

    _fig, axes = plt.subplots(
        2, 2, figsize=(11.4, 7.6), sharex=True,
        gridspec_kw={"height_ratios": [2.0, 1.0]})
    for col, (title, xx, yy) in enumerate(cases):
        h1 = transfer_function(xx, yy, fs, estimator="H1")
        h2 = transfer_function(xx, yy, fs, estimator="H2")
        _, h_true = sp_signal.freqz(b, a, worN=h1.frequencies, fs=fs)
        pos = h1.frequencies > 0.0
        freqs = h1.frequencies[pos]
        true_db = 20.0 * np.log10(np.maximum(np.abs(h_true[pos]), 1e-12))

        ax_mag, ax_coh = axes[0][col], axes[1][col]
        ax_mag.semilogx(freqs, true_db, color=COLOR_FG, linestyle="--",
                        linewidth=1.6, alpha=0.7, label="True $|H|$")
        ax_mag.semilogx(freqs, h1.magnitude_db[pos], color=COLOR_PRIMARY,
                        linewidth=1.8, label="$H_1 = G_{xy}/G_{xx}$")
        ax_mag.semilogx(freqs, h2.magnitude_db[pos], color=COLOR_SECONDARY,
                        linewidth=1.5, linestyle="-.",
                        label="$H_2 = G_{yy}/G_{yx}$")
        ax_mag.set_ylim(-80.0, 12.0)
        ax_mag.set_title(title, fontsize=11, pad=10)
        ax_mag.grid(which="both", color=COLOR_GRID, linestyle="--", alpha=0.5)
        ax_mag.set_axisbelow(True)
        ax_mag.legend(loc="lower center", fontsize=8.5)

        ax_coh.semilogx(freqs, h1.coherence[pos], color=COLOR_TERTIARY,
                        linewidth=1.8, label="measured")
        # The closed-form floor the section states, from the local SNR.
        if col == 0:
            # Output noise: the SNR follows |H|, so the coherence collapses
            # wherever the filter has thrown the output into the noise.
            snr = (np.abs(h_true[pos]) ** 2 * np.mean(clean_x**2)
                   / (np.mean(clean_y**2) * noise_frac**2))
        else:
            # Input noise: the ratio is the same in every band, so the
            # coherence is depressed uniformly and never at the band edges.
            snr = np.full_like(freqs, 1.0 / noise_frac_in**2)
        snr = np.maximum(snr, 1e-9)
        ax_coh.semilogx(freqs, snr / (1.0 + snr), color=COLOR_FG,
                        linestyle=":", linewidth=1.4,
                        label="$\\mathrm{SNR}/(1 + \\mathrm{SNR})$")
        ax_coh.axhline(0.9, color=COLOR_SECONDARY, linestyle="--",
                       linewidth=1.2, alpha=0.8)
        ax_coh.set_ylabel("Coherence" if col == 0 else "")
        ax_coh.set_xlabel(LABEL_FREQ_HZ)
        ax_coh.set_ylim(0.0, 1.05)
        ax_coh.set_xlim(20.0, fs / 2.0)
        ax_coh.grid(which="both", color=COLOR_GRID, linestyle="--", alpha=0.5)
        ax_coh.set_axisbelow(True)
        ax_coh.legend(loc="lower center", fontsize=8)
        for _axf in (ax_mag, ax_coh):
            format_frequency_axis(_axf, 20.0, fs / 2.0)
    axes[0][0].set_ylabel("Magnitude [dB]")
    _fig.suptitle("Choosing Between $H_1$ and $H_2$ (Bendat & Piersol)",
                  fontsize=13)
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
              linewidth=2.0, label="Total $\\mathrm{THD}(f)$")
    ax.loglog(freqs, 100.0 * res.distortion_ratios[0][sel],
              color=COLOR_SECONDARY, linewidth=1.5, linestyle="--",
              label="2nd harmonic $d_2(f)$")
    ax.loglog(freqs, 100.0 * res.distortion_ratios[1][sel],
              color=COLOR_TERTIARY, linewidth=1.5, linestyle="--",
              label="3rd harmonic $d_3(f)$")
    ax.axhline(100.0 * (a2 / 2.0) / h1_ref, color=COLOR_SECONDARY,
               linestyle=":", linewidth=1.2, alpha=0.8,
               label="Chebyshev asymptote $(a_2/2)/H_1$")
    ax.axhline(100.0 * (a3 / 4.0) / h1_ref, color=COLOR_TERTIARY,
               linestyle=":", linewidth=1.2, alpha=0.8,
               label="Chebyshev asymptote $(a_3/4)/H_1$")
    ax.set_xlabel("Excitation frequency [Hz]")
    ax.set_ylabel("Distortion re fundamental [%]")
    ax.set_title("Swept-Sine Harmonic Distortion by Order (Farina / Novak)",
                 pad=12)
    ax.set_xlim(30.0, 2800.0)
    ax.set_ylim(0.05, 20.0)
    format_frequency_axis(ax, 30.0, 2800.0)
    ax.grid(which="both", color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", fontsize=9)
    ax.text(0.985, 0.965,
            "one sweep separates every distortion order;\n"
            "each rolls off where its product $n f$ crosses the 3 kHz corner",
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
               linewidth=1.8,
               label=f"Integrated I = {_fmt_minus(res.integrated, '.1f')} LUFS")

    # Label the programme sections along the top.
    t0 = 0.0
    for name, _, duration in sections:
        ax.text(t0 + duration / 2.0, -11.0, name, ha="center", va="top",
                fontsize=8.5, color=COLOR_FG, alpha=0.75, style="italic")
        t0 += duration
        if t0 < 60.0:
            ax.axvline(t0, color=COLOR_GRID, linewidth=0.8)

    info = [
        f"I     = {_fmt_minus(res.integrated, '6.1f')} LUFS",
        f"LRA   = {res.loudness_range:6.1f} LU",
        f"max M = {_fmt_minus(res.max_momentary, '6.1f')} LUFS",
        f"max S = {_fmt_minus(res.max_short_term, '6.1f')} LUFS",
        f"TPmax = {_fmt_minus(res.true_peak, '6.1f')} dBTP",
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
                 pad=12)
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
                 pad=12)
    ax.grid(which="major", axis="y", color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", fontsize=9)

    info = [
        ("$L_W = L_v + 10\\,\\log_{10}(S/S_0) + 10\\,\\log_{10}(\\varepsilon)"
        " + 10\\,\\log_{10}(411/400)$"),
        f"$S$ = {area:g} m²,  $S_0$ = 1 m²",
        "Part 1: $\\varepsilon = 1$ → upper limit $L_{W,\\mathrm{max}}$",
    ]
    ax.text(0.015, 0.02, "\n".join(info), transform=ax.transAxes,
            va="bottom", ha="left", fontsize=9, color=COLOR_FG,
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


def _es(text: str) -> str:
    """Localise a caller-supplied label the save-time pass cannot reach.

    A branch label is passed *into* the library, which renders it verbatim
    inside a two-line callout; the Spanish pass looks up whole rendered
    strings and would never match the compound. Translating the label here
    keeps it in the same table as every other Spanish string of the corpus.
    """
    return lookup(text) if _LANG == "es" else text


def generate_silencer_chain_geometry(output_dir: str) -> None:
    """To-scale drawing of a silencer assembled element by element.

    A run of nominal 200 mm circular duct opening into a 400 mm shell of
    600 mm, with a quarter-wave stub on the inlet run and a Helmholtz
    resonator at the outlet junction. One concept: what a hand-built chain
    can honestly draw, which is its ducts to scale and its branches marked.
    """
    print("Generating silencer_chain_geometry...")
    from phonometry import SilencerChain
    from phonometry.noise_control.silencers import (
        helmholtz_impedance,
        quarter_wave_impedance,
    )

    freqs = np.linspace(20.0, 500.0, 481)
    s_duct = np.pi * 0.100**2                # nominal 200 mm duct
    s_shell = np.pi * 0.200**2               # nominal 400 mm shell
    s_stub = np.pi * 0.050**2                # nominal 100 mm branch
    chain = (
        SilencerChain(freqs)
        .duct(0.10, s_duct)
        .shunt(
            quarter_wave_impedance(freqs, 343.0 / (4.0 * 125.0), s_stub),
            label=_es("Quarter-wave stub"),
        )
        .duct(0.20, s_duct)
        .duct(0.60, s_shell)
        .shunt(
            helmholtz_impedance(freqs, np.pi * 0.025**2, 0.05, 2e-3),
            label=_es("Helmholtz resonator"),
        )
        .duct(0.30, s_duct)
    )
    _fig, ax = plt.subplots(figsize=(10, 6.2))
    chain.plot_geometry(ax=ax, language=_LANG)
    plt.tight_layout()
    save_figure(output_dir, "silencer_chain_geometry.svg")
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
    ax.set_title("Enveloping-surface sound power (ISO 3744)  "
                 f"$L_{{W\\!A}}$ = {lwa:.1f} dB(A)",
                 pad=12)
    ax.set_xlabel(LABEL_FREQ_HZ)
    ax.set_ylabel("Sound power level $L_W$ [dB]")
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
        "Reverberation-room sound power (ISO 3741)  "
        f"$L_{{W\\!A}}$ = {lwa:.1f} dB(A)",
        pad=12)
    ax.set_xlabel(LABEL_FREQ_HZ)
    ax.set_ylabel("Sound power level $L_W$ [dB]")
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
        "Intensity-scanning sound power (ISO 9614-2)  "
        f"$L_{{W\\!A}}$ = {lwa:.1f} dB(A)",
        pad=12)
    ax.set_xlabel(LABEL_FREQ_HZ)
    ax.set_ylabel("Sound power level $L_W$ [dB]")
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
    ax.set_title("Precision sound power (ISO 3745)  "
                 f"$L_{{W\\!A}}$ = {lwa:.1f} dB(A)",
                 pad=12)
    ax.set_xlabel(LABEL_FREQ_HZ)
    ax.set_ylabel("Sound power level $L_W$ [dB]")
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
    ax.set_title("Precision intensity scanning (ISO 9614-3)  "
                 f"$L_{{W\\!A}}$ = {lwa:.1f} dB(A)",
                 pad=12)
    ax.set_xlabel(LABEL_FREQ_HZ)
    ax.set_ylabel("Sound power level $L_W$ [dB]")
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
                label=f"$m$ = {int(m)}  →  {peak:.1f} dB")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Transmission loss [dB]")
    ax.set_title("Expansion-chamber transmission loss (Bies Eq. 8.111)",
                 pad=10)
    ax.set_xlim(20.0, 2000.0)
    ax.set_ylim(0.0, 20.0)
    format_frequency_axis(ax, 20.0, 2000.0)
    ax.grid(True, which="both", alpha=0.4)
    ax.legend(loc="upper right", fontsize="small",
              title="Area ratio $m = S_{\\mathrm{exp}}/S_{\\mathrm{duct}}$")
    plt.tight_layout()
    save_figure(output_dir, "silencer_expansion_chamber.svg")
    plt.close()


def generate_silencer_insertion_loss(output_dir: str) -> None:
    """The same chamber: one transmission loss, two insertion losses."""
    print("Generating silencer_insertion_loss.svg...")
    import warnings as _warnings

    from phonometry import expansion_chamber, radiating_piston

    pipe_area, rho, c = 0.01, 1.206, 343.0
    radius = float(np.sqrt(pipe_area / np.pi))
    z0 = rho * c / pipe_area                       # 41.4 kPa.s/m3
    freqs = np.linspace(20.0, 880.0, 1600)         # up to the cut-on, 890.8 Hz
    piston = radiating_piston(radius, freqs, speed_of_sound=c, density=rho)
    # Acoustic (not mechanical) radiation impedance of the open end:
    # Z_r = (rho c / S) (R_1 + j X_1), with the normalised R_1 / X_1 the
    # result carries as .resistance / .reactance.
    z_r = z0 * (np.asarray(piston.resistance) + 1j * np.asarray(piston.reactance))

    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore")
        plain = expansion_chamber(freqs, 0.3, 0.04, pipe_area)
        matched = expansion_chamber(freqs, 0.3, 0.04, pipe_area,
                                    source_impedance=z0, radiation_impedance=z_r)
        stiff = expansion_chamber(freqs, 0.3, 0.04, pipe_area,
                                  source_impedance=20.0 * z0,
                                  radiation_impedance=z_r)

    _fig, ax = plt.subplots(figsize=(10, 6.2))
    ax.axhline(0.0, color=COLOR_GRID, lw=1.0)
    ax.plot(freqs, plain.transmission_loss, color=COLOR_FG, lw=2.6,
            label="Transmission loss (anechoic ports)")
    ax.plot(freqs, np.asarray(stiff.insertion_loss), color=COLOR_SECONDARY,
            lw=1.7, ls="--",
            label="Insertion loss, stiff source ($Z_\\mathrm{s} = 20\\,\\rho c/S$)")
    ax.plot(freqs, np.asarray(matched.insertion_loss), color=COLOR_PRIMARY,
            lw=1.7, label="Insertion loss, matched source ($Z_\\mathrm{s} = \\rho c/S$)")
    ax.set_xlim(20.0, 880.0)
    ax.set_ylim(-30.0, 24.0)
    ax.set_xlabel(LABEL_FREQ_HZ)
    ax.set_ylabel("Attenuation [dB]")
    ax.set_title("One chamber, one TL, and as many insertion losses as "
                 "installations", pad=10)
    ax.annotate("near 180 Hz the chamber and the open end it\ndischarges "
                "through resonate together: the\ninstallation is LOUDER with "
                "the silencer in it",
                xy=(182.0, -22.6), xytext=(250.0, -25.0), fontsize="small",
                color=COLOR_SECONDARY,
                arrowprops={"arrowstyle": "->", "color": COLOR_SECONDARY,
                            "lw": 1.0})
    ax.annotate("near 645 Hz, just past the half-wave trough\nof the TL, both "
                "insertion losses dip again",
                xy=(645.0, -8.5), xytext=(560.0, -17.5),
                fontsize="small", color=COLOR_FG,
                arrowprops={"arrowstyle": "->", "color": COLOR_FG, "lw": 1.0})
    ax.grid(True, which="both", alpha=0.4)
    ax.legend(loc="upper left", fontsize="small")
    plt.tight_layout()
    save_figure(output_dir, "silencer_insertion_loss.svg")
    plt.close()


def generate_silencer_selection(output_dir: str) -> None:
    """Reactive against dissipative, on one axis, with the validity ceiling."""
    print("Generating silencer_selection.svg...")
    import warnings as _warnings

    from phonometry import expansion_chamber, helmholtz_resonator
    from phonometry.noise_control import hvac

    inch, foot = 0.0254, 0.3048
    bands = np.array([63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0])
    freqs = np.linspace(30.0, 8000.0, 4000)

    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore")
        chamber = expansion_chamber(freqs, 0.3, 0.04, 0.01)
        branch = helmholtz_resonator(freqs, duct_area=0.01, neck_area=1e-4,
                                     neck_length=0.0296, cavity_volume=1e-3)
    limit = float(chamber.plane_wave_limit or 8000.0)
    inside = freqs <= limit

    _fig, ax = plt.subplots(figsize=(10, 6.4))
    ax.semilogx(freqs[inside], np.asarray(chamber.transmission_loss)[inside],
                color=COLOR_PRIMARY, lw=1.9,
                label="Reactive: 0.3 m expansion chamber, $m = 4$")
    ax.semilogx(freqs[inside], np.asarray(branch.transmission_loss)[inside],
                color=COLOR_PRIMARY, lw=1.7, ls="--",
                label="Reactive: Helmholtz branch tuned to 100 Hz")
    for thickness, style in ((0.025, ":"), (0.050, "-.")):
        lined = hvac.lined_rectangular_duct_attenuation(
            bands, 36 * inch, 24 * inch, 1.5, thickness, include_unlined=True)
        ax.semilogx(bands, np.asarray(lined.values), style, color=COLOR_TERTIARY,
                    lw=1.8, marker="o", ms=4,
                    label=f"Dissipative: 1.5 m of {thickness * 1000:.0f} mm "
                          f"lined 36 × 24 in duct")
    splitter = hvac.splitter_silencer_insertion_loss(
        bands, 24 * inch, 5 * foot, [4 * inch] * 5, 8 * inch)
    ax.semilogx(bands, np.asarray(splitter.values), "-", color=COLOR_SECONDARY,
                lw=2.0, marker="s", ms=5,
                label="Dissipative: five-airway splitter unit, 5 ft")
    ax.axvline(limit, color=COLOR_PRIMARY, lw=1.6, ls=":")
    ax.text(limit * 1.10, 45.0, f"first duct cut-on, {limit:.0f} Hz: the blue "
                                "four-pole\ncurves stop applying beyond it; the "
                                "dissipative\nregressions carry on",
            fontsize="small", color=COLOR_FG, va="top")
    ax.set_xlim(40.0, 8000.0)
    ax.set_ylim(0.0, 48.0)
    format_frequency_axis(ax, 40.0, 8000.0)
    ax.set_xlabel(LABEL_FREQ_HZ)
    ax.set_ylabel("Attenuation [dB]")
    ax.set_title("Choosing the family: where each one is worth having",
                 pad=10)
    ax.grid(True, which="both", alpha=0.4)
    ax.legend(loc="upper left", fontsize="small")
    plt.tight_layout()
    save_figure(output_dir, "silencer_selection.svg")
    plt.close()


def generate_silencer_extended_tube(output_dir: str) -> None:
    """The internal extensions that fill the plain chamber's 0 dB troughs."""
    print("Generating silencer_extended_tube.svg...")
    import warnings as _warnings

    from phonometry import expansion_chamber, extended_tube_chamber

    freqs = np.linspace(20.0, 880.0, 2400)
    kwargs = {"length": 0.4, "chamber_area": 0.04, "pipe_area": 0.01}
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore")
        plain = expansion_chamber(freqs, kwargs["length"], kwargs["chamber_area"],
                                  kwargs["pipe_area"])
        inlet = extended_tube_chamber(freqs, inlet_extension=0.1, **kwargs)
        both = extended_tube_chamber(freqs, inlet_extension=0.1,
                                     outlet_extension=0.2, **kwargs)

    _fig, ax = plt.subplots(figsize=(10, 6.2))
    ax.plot(freqs, plain.transmission_loss, color=COLOR_FG, lw=1.6, ls="--",
            label="Plain chamber, 0.4 m")
    ax.plot(freqs, inlet.transmission_loss, color=COLOR_PRIMARY, lw=1.9,
            label="Inlet extension $L/4$ = 0.10 m")
    ax.plot(freqs, both.transmission_loss, color=COLOR_SECONDARY, lw=1.7,
            ls="-.", label="Inlet $L/4$ and outlet $L/2$ = 0.20 m")
    # Each extension is a quarter-wave stub, so its length picks which trough
    # it fills: c/4(L/2) = c/2L is the first, c/4(L/4) = c/L the second.
    first = 343.0 / (2.0 * kwargs["length"])
    second = 343.0 / kwargs["length"]
    for fx in (first, second):
        ax.axvline(fx, color=COLOR_GRID, lw=1.2, ls=":")
    ax.annotate(f"$c/2L$ = {first:.0f} Hz: plain chamber 0.0 dB and the "
                "$L/4$ inlet\nstill only 0.6 dB, because $L/4$ is not tuned "
                "here.\nThe $L/2$ outlet is, and shorts the duct",
                xy=(first, 27.0), xytext=(60.0, 35.0),
                fontsize="small", color=COLOR_FG,
                arrowprops={"arrowstyle": "->", "color": COLOR_FG, "lw": 1.0})
    ax.annotate(f"$c/L$ = {second:.0f} Hz:\nthe $L/4$ inlet\nis tuned here",
                xy=(second, 27.0), xytext=(second - 175.0, 22.0),
                fontsize="small", color=COLOR_PRIMARY,
                arrowprops={"arrowstyle": "->", "color": COLOR_PRIMARY,
                            "lw": 1.0})
    ax.annotate("the inlet alone only moves\nthe surviving trough, to 444 Hz",
                xy=(444.0, 0.3), xytext=(150.0, 11.5), fontsize="small",
                color=COLOR_PRIMARY,
                arrowprops={"arrowstyle": "->", "color": COLOR_PRIMARY,
                            "lw": 1.0})
    ax.set_xlim(20.0, 880.0)
    ax.set_ylim(0.0, 46.0)
    ax.set_xlabel(LABEL_FREQ_HZ)
    ax.set_ylabel("Transmission loss [dB]")
    ax.set_title("Extended-tube chamber: quarter-wave branches buried inside",
                 pad=10)
    ax.grid(True, which="both", alpha=0.4)
    ax.legend(loc="upper right", fontsize="small")
    plt.tight_layout()
    save_figure(output_dir, "silencer_extended_tube.svg")
    plt.close()


def generate_extended_tube_geometry(output_dir: str) -> None:
    """The dimensioned cross-section of the extended-tube chamber."""
    print("Generating extended_tube_geometry.svg...")
    import warnings as _warnings

    from phonometry import extended_tube_chamber

    _fig, ax = plt.subplots(figsize=(10, 6.2))
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore")
        result = extended_tube_chamber(
            np.linspace(20.0, 880.0, 800), length=0.4, chamber_area=0.04,
            pipe_area=0.01, inlet_extension=0.1, outlet_extension=0.2,
        )
    result.plot_geometry(ax=ax, language=_LANG)
    plt.tight_layout()
    save_figure(output_dir, "extended_tube_geometry.svg")
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


def generate_reverberation_correction_terms(output_dir: str) -> None:
    """The five terms of ISO 3741 Eq. 20, on the guide's own room.

    Every value is read off the ``ReverberationSoundPowerResult`` the page's
    own call returns, so the stack adds up to the LW - Lp the guide prints.
    """
    print("Generating reverberation_correction_terms.svg...")
    from phonometry import emission

    freqs = np.array([100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000,
                      1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000,
                      10000], dtype=float)
    lp = np.linspace(80.0, 70.0, freqs.size)
    surface_area = 220.0
    res = emission.sound_power_reverberation(
        lp, np.full(freqs.size, 2.0), volume=200.0, surface_area=surface_area,
        frequencies=freqs, temperature=20.0, static_pressure=101.0)

    absorption = 10.0 * np.log10(res.absorption_area)
    eyring = 4.34 * res.absorption_area / surface_area
    meteo = np.full(freqs.size, res.c1 + res.c2)
    minus_six = np.full(freqs.size, -6.0)
    terms = (
        ("$10\\,\\mathrm{lg}(A/A_0)$", absorption, COLOR_PRIMARY),
        ("$4.34\\,A/S$", eyring, COLOR_TERTIARY),
        ("Waterhouse", res.waterhouse_correction, COLOR_SECONDARY),
        ("$C_1 + C_2$", meteo, COLOR_MUTED),
        ("−6 dB", minus_six, theme_fill(COLOR_MUTED, None)),
    )

    _fig, ax = plt.subplots(figsize=(11, 5.6))
    x = np.arange(freqs.size)
    up = np.zeros(freqs.size)
    down = np.zeros(freqs.size)
    for label, values, colour in terms:
        base = np.where(values >= 0.0, up, down)
        ax.bar(x, values, bottom=base, width=0.72, color=colour,
               edgecolor=COLOR_FG, linewidth=0.4, label=label)
        up = up + np.clip(values, 0.0, None)
        down = down + np.clip(values, None, 0.0)
    ax.plot(x, res.sound_power_level - lp, "o-", color=COLOR_FG, markersize=4,
            linewidth=1.6, label="$L_W - \\bar{L}_p$")
    ax.axhline(0.0, color=COLOR_FG, linewidth=1.0)
    ax.annotate("the Waterhouse term rules the low bands\n"
                "(1.68 dB at 100 Hz, 0.02 dB at 10 kHz)", (0.6, 14.4),
                fontsize=10, color=COLOR_SECONDARY)
    ax.annotate("$4.34\\,A/S$ is 0.32 dB in this hard room, and 0.79 dB with "
                "the same room damped to $T_{60}$ = 0.8 s",
                (0.6, -7.4), fontsize=10, color=COLOR_TERTIARY)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{f:g}" for f in freqs], rotation=60, ha="right",
                       fontsize=8)
    ax.set(xlabel=LABEL_FREQ_HZ,
           ylabel="Contribution to $L_W - \\bar{L}_p$ [dB]",
           ylim=(-8.0, 17.5))
    ax.set_title("The five terms of ISO 3741 Eq. 20",
                 pad=12)
    ax.grid(axis="y", color=COLOR_GRID, linestyle="-")
    ax.legend(loc="upper right", fontsize=9, ncol=3)
    plt.tight_layout()
    save_figure(output_dir, "reverberation_correction_terms.svg")
    plt.close()


def generate_precision_positions_arrays(output_dir: str) -> None:
    """The two ISO 3745 arrays: the 40-position hemisphere and the sphere.

    The hemisphere panel splits the array at position 20, which is where the
    standard's escalation cuts: positions 1 to 20 are used first and 21 to 40
    are added only when the band level range demands it.
    """
    print("Generating precision_positions_arrays.svg...")
    from phonometry import emission, plot_microphone_positions

    hemi = emission.precision_positions("hemisphere", radius=1.0, count=40)
    sphere = emission.precision_positions("sphere", radius=1.0, count=20)

    from matplotlib.lines import Line2D
    from matplotlib.transforms import offset_copy

    fig, (axl, axr) = plt.subplots(
        1, 2, figsize=(13, 7.0), subplot_kw={"projection": "3d"})
    # Two colours through the scatter kwargs, so the escalation the standard
    # makes at position 20 reads straight off the array.
    plot_microphone_positions(
        hemi, ax=axl, radius=1.0, language=_LANG,
        color=[COLOR_SECONDARY] * 20 + [COLOR_PRIMARY] * 20, s=36)
    # Forty numbered positions on one dome is the densest array the corpus
    # draws, and at the library's default view whole numbers disappeared under
    # their neighbours: 14 under 36, 34 under 22, 10 under 25, 26 under 30.
    # Three layout moves open the array up, none of them touching a position.
    # The dome is drawn taller and seen from lower down, because Table E.1
    # spirals its positions up in z and it is z that has to carry them apart.
    axl.set_box_aspect((1.0, 1.0, 0.78))
    axl.view_init(elev=14, azim=-24)
    # And the numbers the helper wrote on the points lean apart: a position of
    # the first set leans left of its dot, one of the escalation set leans
    # right. The array is two interleaved spirals, so nearly every stack was
    # one number of each set, and leaning the sets in opposite directions
    # stands those two side by side; which way a number leans reads off its
    # own colour. Their chips also carry no more padding than the digits need,
    # which buys the dense rings room without setting the digits smaller.
    badges = [t for t in axl.texts if t.get_text().isdigit()]
    for order, badge in enumerate(badges):
        badge.set_transform(offset_copy(
            badge.get_transform(), fig=fig, x=-3.8 if order < 20 else 3.8,
            units="points"))
        badge.get_bbox_patch().set_boxstyle("round,pad=0.12")
    # The plane's own label is anchored on the rim of the disc, which on this
    # array is exactly where position 21 sits; take it out to the edge of the
    # plane it names, clear of that chip.
    plane = next(t for t in axl.texts if not t.get_text().isdigit())
    plane.set_position_3d((1.35, 0.0, 0.0))
    axl.set_title("Hemisphere, 40 positions (Table E.1)")
    axl.legend(handles=[
        Line2D([], [], marker="o", linestyle="none", color=COLOR_SECONDARY,
               label="positions 1-20: always"),
        Line2D([], [], marker="o", linestyle="none", color=COLOR_PRIMARY,
               label="positions 21-40: on escalation"),
    ], loc="upper left", fontsize=8)

    plot_microphone_positions(sphere, ax=axr, radius=1.0, language=_LANG)
    # Nine labels to an axis ran the tick chain of the sphere's x axis into
    # itself ("−1.00" into "−0.75" into "−0.50"); the half-metre step reads
    # the same radius with room between the labels, and matches the ticks the
    # hemisphere panel carries.
    for setter in (axr.set_xticks, axr.set_yticks, axr.set_zticks):
        setter([-1.0, -0.5, 0.0, 0.5, 1.0])
    # Table D.1 puts positions 2 and 19 on opposite faces of the sphere, and
    # the default view laid one number over the other; four degrees of
    # elevation and ten of azimuth part them by half a panel.
    axr.view_init(elev=34, azim=-70)
    axr.set_title("Sphere, 20 positions (Table D.1)")

    fig.suptitle("ISO 3745 precision microphone arrays")
    # The lower eye widens the hemisphere's 3-D box on the page, and its z
    # label then reached over the sphere panel, which paints its own white
    # ground over anything of its neighbour's that arrives there. The wider
    # gutter keeps the label on its own panel.
    plt.tight_layout(w_pad=5.0)
    save_figure(output_dir, "precision_positions_arrays.svg")
    plt.close()


def generate_k1_k2_corrections(output_dir: str) -> None:
    """The shape of the two ISO 3744 corrections, and where they are capped.

    Left: K1 against the source-to-background margin, swept through
    ``sound_power_pressure`` itself so the curve is the library's. Right: K2
    against 4S/A, with the two validity limits and what a 20 % error in A
    costs at the engineering limit.
    """
    print("Generating k1_k2_corrections.svg...")
    from phonometry import emission

    # --- left: K1(delta Lp) --------------------------------------------------
    freqs = np.array([1000.0])
    margins = np.linspace(0.5, 20.0, 400)
    surface = 80.0
    k1 = np.array([
        float(emission.sound_power_pressure(
            np.full((10, 1), surface), "hemisphere", radius=2.0,
            background_levels=np.full((10, 1), surface - m),
            frequencies=freqs).background_correction[0])
        for m in margins
    ])

    # --- right: K2(4S/A) -----------------------------------------------------
    ratio = np.geomspace(0.05, 10.0, 400)
    k2 = 10.0 * np.log10(1.0 + ratio)

    _fig, (axl, axr) = plt.subplots(1, 2, figsize=(12, 5))

    axl.plot(margins, k1, color=COLOR_PRIMARY, linewidth=2.0)
    axl.axvspan(0.0, 6.0, color=theme_fill(COLOR_SECONDARY, axl), zorder=0)
    axl.axvline(6.0, color=COLOR_SECONDARY, linestyle="--", linewidth=1.4)
    axl.axvline(3.0, color=COLOR_TERTIARY, linestyle=":", linewidth=1.4)
    axl.text(6.2, 2.4, "ISO 3744 criterion\n6 dB", fontsize=9,
             color=COLOR_SECONDARY)
    axl.text(0.4, 3.4, "capped:\nupper bound", fontsize=9, color=COLOR_SECONDARY)
    axl.text(3.15, 0.9, "ISO 3746\n3 dB", fontsize=9, color=COLOR_TERTIARY)
    for m, label in ((15.0, "0.14 dB"), (10.0, "0.46 dB"), (6.0, "1.26 dB")):
        value = float(np.interp(m, margins, k1))
        axl.plot([m], [value], "o", color=COLOR_FG, markersize=5)
        axl.annotate(f"{m:g} dB → {label}", (m, value),
                     textcoords="offset points", xytext=(8, 8), fontsize=9)
    axl.set(xlabel="Source-to-background margin $\\Delta L_p$ [dB]",
            ylabel="Background correction $K_1$ [dB]", xlim=(0, 20),
            ylim=(0, 4.2))
    axl.set_title("$K_1$ is a cliff, not a slope", pad=12)
    axl.grid(color=COLOR_GRID, linestyle="-")

    axr.semilogx(ratio, k2, color=COLOR_PRIMARY, linewidth=2.0)
    # A +/- 20 % error in A moves 4S/A the other way by the same factor.
    axr.fill_between(ratio, 10.0 * np.log10(1.0 + ratio / 1.2),
                     10.0 * np.log10(1.0 + ratio / 0.8),
                     color=theme_fill(COLOR_PRIMARY, axr), zorder=0,
                     label="$A$ known to ±20 %")
    for limit, colour, name in ((4.0, COLOR_SECONDARY, "ISO 3744 limit"),
                                (7.0, COLOR_TERTIARY, "ISO 3746 limit")):
        r_limit = 10.0 ** (limit / 10.0) - 1.0
        share = 100.0 * r_limit / (1.0 + r_limit)
        axr.axhline(limit, color=colour, linestyle="--", linewidth=1.4)
        axr.annotate(f"{name}: $K_2$ = {limit:g} dB\n{share:.0f} % of the "
                     f"measured energy is room", (0.055, limit + 0.18),
                     fontsize=9, color=colour)
    axr.set(xlabel="$4S/A$", ylabel="Environmental correction $K_2$ [dB]",
            ylim=(0, 11))
    axr.set_title("$K_2$ saturates, and inherits every error in $A$",
                  pad=12)
    axr.grid(which="both", color=COLOR_GRID, linestyle="-")
    axr.legend(loc="lower right", fontsize=9)

    plt.tight_layout()
    save_figure(output_dir, "k1_k2_corrections.svg")
    plt.close()


def generate_radiation_efficiency(output_dir: str) -> None:
    """sigma(f) for the page's own plate, against the Part 1 assumption eps = 1."""
    print("Generating radiation_efficiency_sigma.svg...")
    from phonometry import vibration

    # A 3 mm steel casing panel: m'' = 23.4 kg/m2, B' from the plate stiffness,
    # so the coincidence frequency lands near 4 kHz.
    freqs = np.geomspace(50.0, 5000.0, 400)
    stiffness = vibration.plate_bending_stiffness(2.0e11, 0.003, 0.30)
    fc = vibration.coincidence_frequency(7800.0 * 0.003, stiffness)
    sigma = vibration.radiation_efficiency(freqs, 1.5, 1.25, fc)

    _fig, ax = plt.subplots(figsize=(10, 5.6))
    # The result's own .plot() draws sigma(f) on a log axis with the sigma = 1
    # line and the critical frequency already marked.
    sigma.plot(ax=ax, language=_LANG)
    values = sigma.radiation_efficiency
    ax.fill_between(freqs, values, 1.0, where=values < 1.0,
                    color=theme_fill(COLOR_SECONDARY, ax), zorder=0)
    ax.annotate("ISO/TS 7849-1 assumes $\\varepsilon = 1$ here", (60.0, 1.35),
                fontsize=10, color=COLOR_SECONDARY)
    for band in (125.0, 1000.0):
        index = float(np.interp(band, freqs, sigma.radiation_index))
        ax.annotate(f"{_fmt_minus(index, '.1f')} dB",
                    (band, float(np.interp(band, freqs, values))),
                    textcoords="offset points", xytext=(6, -16), fontsize=10,
                    color=COLOR_SECONDARY)
    ax.annotate("the gap Part 1 pays for: 25 dB at 63 Hz,\n"
                "14 dB at 2 kHz, gone only above coincidence",
                (55.0, 0.02), fontsize=10, color=COLOR_SECONDARY)
    plt.tight_layout()
    save_figure(output_dir, "radiation_efficiency_sigma.svg")
    plt.close()


def generate_partial_power_map(output_dir: str) -> None:
    """The per-segment partial powers of an ISO 9614-2 scan, over the unfolded box.

    Five faces of a measurement box laid flat, each tiled by its segments and
    coloured by its partial power level, with the one net-negative segment
    hatched: the diagnostic a band-summed LW spectrum cannot show.
    """
    print("Generating partial_power_map.svg...")
    import matplotlib.patches as mpatches

    from phonometry import emission

    # A 1,2 x 0,8 x 1,0 m machine boxed at d = 0,25 m: five faces, each cut
    # into four segments. The front face dominates; one segment on the shadow
    # side takes net inflow from a louder neighbour.
    faces = ("front", "right", "back", "left", "top")
    areas_face = np.array([1.7 * 1.5, 1.3 * 1.5, 1.7 * 1.5, 1.3 * 1.5, 1.7 * 1.3])
    seg_area = np.repeat(areas_face / 4.0, 4)
    share = np.array([
        2.6, 2.2, 2.0, 1.8,      # front
        0.9, 0.8, 0.7, 0.7,      # right
        0.5, 0.4, -0.35, 0.3,    # back, one segment net inward
        0.8, 0.7, 0.6, 0.6,      # left
        1.1, 1.0, 0.9, 0.8,      # top
    ])
    freqs = np.array([500.0])
    i_n = (share * 1.0e-5 / seg_area)[:, None]
    res = emission.sound_power_intensity(i_n, seg_area, frequencies=freqs)
    level = res.partial_power_level[:, 0]
    signed = res.partial_power[:, 0]

    # Unfolded net: the four side faces in a row with the top above the front.
    origins = {"front": (0.0, 0.0), "right": (1.7, 0.0), "back": (3.0, 0.0),
               "left": (4.7, 0.0), "top": (0.0, 1.5)}
    sizes = {"front": (1.7, 1.5), "right": (1.3, 1.5), "back": (1.7, 1.5),
             "left": (1.3, 1.5), "top": (1.7, 1.3)}

    fig, (ax, axb) = plt.subplots(
        2, 1, figsize=(11, 7.4), gridspec_kw={"height_ratios": [3.0, 1.0]})
    vmin, vmax = float(level.min()), float(level.max())
    cmap = plt.get_cmap("viridis")
    for f_index, face in enumerate(faces):
        ox, oy = origins[face]
        w, h = sizes[face]
        for k in range(4):
            cx = ox + (k % 2) * w / 2.0
            cy = oy + (1 - k // 2) * h / 2.0
            value = level[4 * f_index + k]
            shade = cmap((value - vmin) / max(vmax - vmin, 1e-9))
            ax.add_patch(mpatches.Rectangle(
                (cx, cy), w / 2.0, h / 2.0, facecolor=shade,
                edgecolor=COLOR_FG, linewidth=1.0,
                hatch="//" if signed[4 * f_index + k] < 0 else None))
            sign = "−" if signed[4 * f_index + k] < 0 else ""
            ax.text(cx + w / 4.0, cy + h / 4.0, f"{sign}{value:.0f}",
                    ha="center", va="center", fontsize=9, color="white")
        ax.add_patch(mpatches.Rectangle((ox, oy), w, h, fill=False,
                                        edgecolor=COLOR_FG, linewidth=2.0))
        if face == "top":
            ax.text(ox + w / 2.0, oy + h + 0.10, face, ha="center",
                    va="bottom", fontsize=10, fontweight="bold")
        else:
            ax.text(ox + w / 2.0, oy - 0.12, face, ha="center", va="top",
                    fontsize=10, fontweight="bold")
    ax.set_xlim(-0.2, 6.2)
    ax.set_ylim(-0.4, 3.0)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Partial power level per segment [dB re 1 pW], "
                 "the box unfolded")
    fig.colorbar(plt.cm.ScalarMappable(
        norm=Normalize(vmin, vmax), cmap=cmap), ax=ax, shrink=0.8,
        label="$L_{Wi}$ [dB]")

    per_face = np.add.reduceat(signed, np.arange(0, 20, 4))
    axb.bar(range(5), 10.0 * np.log10(np.abs(per_face) / 1e-12),
            color=[COLOR_PRIMARY] * 5, width=0.6)
    axb.axhline(float(res.sound_power_level[0]), color=COLOR_SECONDARY,
                linestyle="--", linewidth=1.6,
                label=f"$L_W$ = {float(res.sound_power_level[0]):.1f} dB")
    axb.set_xticks(range(5))
    axb.set_xticklabels(faces)
    axb.set_ylabel("Face total [dB]")
    axb.set_ylim(60, float(res.sound_power_level[0]) + 4.0)
    axb.legend(loc="upper right", fontsize=9)
    axb.grid(axis="y", color=COLOR_GRID, linestyle="-")

    plt.tight_layout()
    save_figure(output_dir, "partial_power_map.svg")
    plt.close()


def generate_spacer_bandwidth(output_dir: str) -> None:
    """What each p-p spacer costs at both ends of the band.

    Upper panel: the finite-difference bias against frequency for 6, 12 and
    50 mm, taken from ``sound_intensity`` itself. Lower panel: the residual
    index requirement each spacer inherits, which scales as 10 lg(dr/25 mm),
    so the two limits close in from opposite sides and no single spacer covers
    the audio range.
    """
    print("Generating spacer_bandwidth.svg...")
    from phonometry import sound_intensity

    fs, c = 48000, 343.0
    n = fs  # one second of white noise: enough bands to read the bias off
    rng = np.random.default_rng(61043)
    noise = rng.standard_normal(n)
    spectrum = np.fft.rfft(noise)
    freqs = np.fft.rfftfreq(n, 1 / fs)

    _fig, (axt, axb) = plt.subplots(
        2, 1, figsize=(10, 7.2), sharex=True,
        gridspec_kw={"height_ratios": [2.0, 1.0]})
    colours = (COLOR_PRIMARY, COLOR_SECONDARY, COLOR_TERTIARY)
    for (dr, colour) in zip((0.006, 0.012, 0.050), colours, strict=True):
        p2 = np.fft.irfft(spectrum * np.exp(-2j * np.pi * freqs * dr / c), n)
        res = sound_intensity(noise, p2, fs, dr, fraction=3,
                              limits=[20.0, 12500.0])
        # bias_correction is the compensating factor (k dr)/sin(k dr); the bias
        # itself is its reciprocal in decibels. Drop the bands the library
        # clamps at k dr = pi/2 so the curve stays the physics.
        assert res.frequency is not None, "band analysis returns its frequencies"
        assert res.bias_correction is not None, "the result must carry the bias"
        usable = res.frequency < c / (4.0 * dr)
        axt.semilogx(res.frequency[usable],
                     -10.0 * np.log10(res.bias_correction[usable]),
                     color=colour, linewidth=2.0,
                     label=f"$\\Delta r$ = {dr * 1000:.0f} mm")
        f_max = res.max_valid_frequency
        axt.axvline(f_max, color=colour, linestyle=":", linewidth=1.3)
        axt.annotate(f"{f_max / 1000:.2f} kHz", (f_max, -1.9), fontsize=9,
                     color=colour, rotation=90, ha="right", va="bottom")
        # The IEC 61043 Table 2 requirement moves with the spacer, by the
        # 10 lg(x/25) of its own Note 1.
        margin = 10.0 * np.log10(dr / 0.025)
        axb.semilogx(res.frequency, np.full(res.frequency.shape, margin),
                     color=colour, linewidth=2.2)
        axb.annotate(f"{_fmt_minus(margin, '+.1f')} dB",
                     (11000.0, margin + 0.3), fontsize=10,
                     color=colour, ha="right")
    axt.axhline(-0.3, color=COLOR_FG, linestyle="--", linewidth=1.3)
    axt.annotate("−0.3 dB: the bound $f_{\\mathrm{max}}$ is quoted against",
                 (24.0, -0.24), fontsize=9)
    axt.set(ylabel="Finite-difference bias [dB]", ylim=(-2.4, 0.35))
    axt.set_title("High end: the finite-difference bias, and "
                  "$f_{\\mathrm{max}} = 0.1\\,c/\\Delta r$",
                  pad=12)
    axt.grid(which="both", color=COLOR_GRID, linestyle="-")
    axt.legend(loc="lower left", fontsize=9)

    axb.axhline(0.0, color=COLOR_FG, linewidth=1.0)
    axb.annotate("25 mm: the separation Table 2 is written for", (24.0, 0.3),
                 fontsize=9)
    axb.set(xlabel=LABEL_FREQ_HZ,
            ylabel="$\\delta_{pI0}$ margin,\nre 25 mm [dB]", ylim=(-8.0, 4.6))
    axb.set_title("Low end: doubling the spacer is worth 3 dB of margin",
                  pad=10)
    axb.grid(which="both", color=COLOR_GRID, linestyle="-")
    format_frequency_axis(axb)

    plt.tight_layout()
    save_figure(output_dir, "spacer_bandwidth.svg")
    plt.close()


def generate_true_peak_intersample(output_dir: str) -> None:
    """What a sample-peak meter misses, and what oversampling recovers."""
    print("Generating true_peak_intersample.svg...")
    from phonometry import broadcast

    fs = 48000
    f0 = fs / 4.0                      # the worst case of the closed form
    phase = np.pi / 4.0                # badly phased: every sample at 1/sqrt(2)
    # The page's own signal: one second of it is metered, twelve samples drawn.
    t_full = np.arange(fs) / fs
    x_full = np.sin(2.0 * np.pi * f0 * t_full + phase)
    sample_peak = float(broadcast.true_peak_level(x_full, fs, oversample=1))
    true_peak = float(broadcast.true_peak_level(x_full, fs))

    n = 12
    t = np.arange(n) / fs
    x = x_full[:n]
    t_fine = np.linspace(0.0, (n - 1) / fs, 2000)
    fine = np.sin(2.0 * np.pi * f0 * t_fine + phase)

    _fig, (axl, axr) = plt.subplots(1, 2, figsize=(12, 5))
    axl.plot(t_fine * 1000.0, fine, color=COLOR_PRIMARY, linewidth=1.8,
             label="band-limited reconstruction")
    axl.plot(t * 1000.0, x, "o", color=COLOR_SECONDARY, markersize=7,
             label="samples at 48 kHz")
    axl.axhline(np.abs(x).max(), color=COLOR_SECONDARY, linestyle="--",
                linewidth=1.3)
    axl.axhline(1.0, color=COLOR_FG, linestyle="--", linewidth=1.3)
    axl.annotate(f"sample peak {_fmt_minus(sample_peak, '.2f')} dBFS",
                 (0.012, np.abs(x).max() + 0.04), fontsize=9,
                 color=COLOR_SECONDARY)
    axl.annotate(f"true peak {_fmt_minus(true_peak, '.2f')} dBTP",
                 (0.012, 1.04), fontsize=9)
    for tick in np.arange(0, (n - 1) / fs, 1.0 / (4 * fs)):
        axl.plot([tick * 1000.0], [-1.28], "|", color=COLOR_MUTED,
                 markersize=6)
    axl.annotate("4× oversampled grid", (0.02, -1.24), fontsize=9,
                 color=COLOR_MUTED)
    axl.set(xlabel="Time [ms]", ylabel="Amplitude [FS]", ylim=(-1.35, 1.25))
    axl.set_title("A 12 kHz tone at $\\pi/4$: every sample misses the peak",
                  pad=12)
    axl.grid(color=COLOR_GRID, linestyle="-")
    axl.legend(loc="lower right", fontsize=9)

    f_norm = np.linspace(0.0, 0.5, 400)
    for ratio, colour in zip((1, 2, 4, 8), (COLOR_SECONDARY, COLOR_TERTIARY,
                                            COLOR_PRIMARY, COLOR_MUTED),
                             strict=True):
        bound = 20.0 * np.log10(np.cos(np.pi * f_norm / ratio))
        axr.plot(f_norm, bound, color=colour, linewidth=2.0,
                 label=f"$n$ = {ratio}")
    worst = 20.0 * np.log10(np.cos(np.pi * 0.25 / 4))
    axr.plot([0.25], [worst], "o", color=COLOR_FG, markersize=6)
    axr.annotate(f"BS.1770 asks for $n \\geq 4$ at 48 kHz:\n"
                 f"{_fmt_minus(worst, '.2f')} dB left at "
                 "$f_{\\mathrm{norm}} = 1/4$", (0.25, worst),
                 textcoords="offset points", xytext=(-150, -46), fontsize=9)
    axr.axvline(0.25, color=COLOR_FG, linestyle=":", linewidth=1.2)
    axr.set(xlabel="Tone frequency / sampling rate", ylabel="Under-read [dB]",
            xlim=(0.0, 0.5), ylim=(-7.0, 0.4))
    axr.set_title("Worst-case under-read, "
                  "$20\\,\\mathrm{lg}\\,\\cos(\\pi f_{\\mathrm{norm}}/n)$",
                  pad=12)
    axr.grid(color=COLOR_GRID, linestyle="-")
    axr.legend(loc="lower left", fontsize=9)

    plt.tight_layout()
    save_figure(output_dir, "true_peak_intersample.svg")
    plt.close()


def generate_channel_weight_map(output_dir: str) -> None:
    """The BS.1770 Annex 3 channel weight as a map of the reproduction layout."""
    print("Generating channel_weight_map.svg...")
    from phonometry import broadcast

    az = np.linspace(-180.0, 180.0, 361)
    el = np.linspace(-90.0, 90.0, 181)
    grid_az, grid_el = np.meshgrid(az, el)
    weight = broadcast.channel_weight(grid_az, grid_el)
    high = float(np.max(weight))

    _fig, ax = plt.subplots(figsize=(11, 5.6))
    # contourf, not pcolormesh: the map is two-valued with straight
    # boundaries, so a filled contour keeps the SVG to a handful of paths.
    ax.contourf(grid_az, grid_el, weight, levels=[0.5 * (1.0 + high), 2.0],
                colors=[theme_fill(COLOR_PRIMARY, ax)], zorder=0)
    ax.contour(grid_az, grid_el, weight, levels=[0.5 * (1.0 + high)],
               colors=[COLOR_FG], linewidths=1.4)
    for sign in (-1.0, 1.0):
        ax.text(sign * 90.0, 20.0, f"{high:.2f}\n(+1.5 dB)", ha="center",
                va="center", fontsize=11, fontweight="bold")
    ax.text(0.0, 62.0, "1.0 everywhere else", ha="center", fontsize=11)
    ax.annotate("$60° \\leq |\\mathrm{azimuth}| \\leq 120°$,  "
                "$|\\mathrm{elevation}| < 30°$", (0.0, -50.0),
                ha="center", fontsize=10)

    speakers = (("L", -30.0, 0.0), ("R", 30.0, 0.0), ("C", 0.0, 0.0),
                ("Ls", -110.0, 0.0), ("Rs", 110.0, 0.0),
                ("U+110", 110.0, 45.0))
    for name, a, e in speakers:
        ax.plot([a], [e], "o", color=COLOR_SECONDARY, markersize=8)
        ax.annotate(f"{name}  ({broadcast.channel_weight(a, e):.2f})", (a, e),
                    textcoords="offset points", xytext=(9, -16), fontsize=9,
                    color=COLOR_SECONDARY)
    ax.set(xlabel="Azimuth [°]", ylabel="Elevation [°]",
           xlim=(-180, 180), ylim=(-90, 90))
    ax.set_xticks([-180, -120, -60, 0, 60, 120, 180])
    ax.set_yticks([-90, -30, 0, 30, 90])
    ax.set_title("BS.1770 Annex 3: the weight is a property of the "
                 "loudspeaker position", pad=12)
    ax.grid(color=COLOR_GRID, linestyle=":", alpha=0.6)
    plt.tight_layout()
    save_figure(output_dir, "channel_weight_map.svg")
    plt.close()


def generate_sound_power_grades_declaration(output_dir: str) -> None:
    """What the grade buys, and what the declaration has to absorb.

    Left: one measured LWA = 92.4 dB carried by the three accuracy grades,
    each interval computed through ``precision_uncertainty``. Right: the
    ISO 4871 Annex B example, declared value against verification level.
    """
    print("Generating sound_power_grades_declaration.svg...")
    from phonometry import emission

    lwa = 92.4
    grades = (("Grade 1\nISO 3741 / 3745", 0.5), ("Grade 2\nISO 3744 / 9614-2", 1.5),
              ("Grade 3\nISO 3746", 3.0))
    sigma_omc = 0.0

    _fig, (axl, axr) = plt.subplots(1, 2, figsize=(12, 5.2))
    for k, (name, sigma_r0) in enumerate(grades):
        u = float(emission.precision_uncertainty(sigma_r0, sigma_omc, 2.0))
        axl.errorbar([k], [lwa], yerr=[[u], [u]], fmt="o", markersize=8,
                     capsize=10, elinewidth=2.4, color=COLOR_PRIMARY)
        axl.annotate(f"$U$ = {u:.1f} dB", (k, lwa - u),
                     textcoords="offset points",
                     xytext=(0, -16), ha="center", va="top", fontsize=10)
    axl.axhline(93.0, color=COLOR_SECONDARY, linestyle="--", linewidth=1.6,
                label="declared limit 93 dB(A)")
    axl.set_xticks(range(len(grades)))
    axl.set_xticklabels([name for name, _ in grades], fontsize=9)
    axl.set_ylabel("$L_{W\\!A}$ [dB re 1 pW]")
    axl.set_ylim(84.0, 101.0)
    axl.set_title("One measurement, three grades: 92.4 dB(A) against a 93 dB "
                  "limit", pad=12)
    axl.grid(axis="y", color=COLOR_GRID, linestyle="-")
    axl.legend(loc="upper left", fontsize=9)

    modes = (("Operating mode 1", 88.0, 2.0, 89.0),
             ("Operating mode 2", 95.0, 2.0, 98.0))
    for k, (name, level, k_wa, l1) in enumerate(modes):
        axr.bar([k], [level], width=0.5, color=COLOR_PRIMARY,
                label="$L_{W\\!A}$" if k == 0 else None)
        axr.bar([k], [k_wa], width=0.5, bottom=level, color=COLOR_MUTED,
                label="$K_{W\\!A}$" if k == 0 else None)
        axr.plot([k], [l1], "D", color=COLOR_SECONDARY, markersize=10,
                 label="verification level $L_1$" if k == 0 else None)
        verdict = "verified" if l1 <= level + k_wa else "not verified"
        axr.annotate(f"$L_{{W\\!Ad}}$ = {level + k_wa:.0f} dB\n"
                     f"$L_1$ = {l1:.0f} dB\n{verdict}",
                     (k, max(l1, level + k_wa) + 1.4), ha="center", fontsize=9)
    axr.set_xticks(range(len(modes)))
    axr.set_xticklabels([name for name, *_ in modes], fontsize=9)
    axr.set_ylabel("A-weighted sound power level [dB]")
    axr.set_ylim(80.0, 106.0)
    axr.set_title("ISO 4871 Annex B: $L_{W\\!Ad} = L_{W\\!A} + K_{W\\!A}$, "
                  "verified when $L_1 \\leq L_{W\\!Ad}$",
                  pad=12)
    axr.grid(axis="y", color=COLOR_GRID, linestyle="-")
    axr.legend(loc="upper left", fontsize=9)

    plt.tight_layout()
    save_figure(output_dir, "sound_power_grades_declaration.svg")
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
                 pad=10)
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
                    label=f"$D$ = {int(diameter * 1000)} mm")
    ax.set_xlim(50.0, 2500.0)
    ax.set_ylim(bottom=0.0)
    format_frequency_axis(ax, 50.0, 2500.0)
    ax.set_xlabel(LABEL_FREQ_HZ)
    ax.set_ylabel("End reflection loss [dB]")
    ax.set_title("Duct end reflection loss (ASHRAE Table 8.14)",
                 pad=10)
    ax.grid(True, which="both", alpha=0.4)
    ax.legend(loc="upper right", fontsize="small", title="Duct diameter")
    plt.tight_layout()
    save_figure(output_dir, "hvac_end_reflection.svg")
    plt.close()


def generate_duct_attenuation_elements(output_dir: str) -> None:
    """The eight element models of section 3, drawn instead of printed."""
    print("Generating duct_attenuation_elements.svg...")
    from phonometry.noise_control import hvac

    inch, foot = 0.0254, 0.3048
    bands = np.array([63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0])
    _fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.6))

    ax = axes[0][0]
    runs = (
        ("Bare", hvac.unlined_rectangular_duct_attenuation(
            bands, 36 * inch, 24 * inch, 5 * foot), COLOR_FG, ":"),
        ("Externally wrapped", hvac.unlined_rectangular_duct_attenuation(
            bands, 36 * inch, 24 * inch, 5 * foot, wrapped=True),
         COLOR_TERTIARY, "-."),
        ("25 mm lining, insertion loss", hvac.lined_rectangular_duct_attenuation(
            bands, 36 * inch, 24 * inch, 5 * foot, 1 * inch), COLOR_SECONDARY,
         "--"),
        ("25 mm lining + side walls", hvac.lined_rectangular_duct_attenuation(
            bands, 36 * inch, 24 * inch, 5 * foot, 1 * inch,
            include_unlined=True), COLOR_PRIMARY, "-"),
    )
    for label, res, color, style in runs:
        ax.semilogx(bands, np.asarray(res.values), style, color=color, lw=1.8,
                    marker="o", ms=4, label=label)
    ax.set_title("(a) 36 × 24 in run, 5 ft", pad=8)
    ax.set_ylabel("Attenuation [dB]")

    ax = axes[0][1]
    for label, kwargs, color, style in (
        ("Square, lined", {"bend_type": "square", "lined": True},
         COLOR_PRIMARY, "-"),
        ("Square, bare", {"bend_type": "square"}, COLOR_SECONDARY, "--"),
        ("Square, vanes", {"bend_type": "square", "vanes": True},
         COLOR_TERTIARY, "-."),
        ("Round, bare", {"bend_type": "round"}, COLOR_FG, ":"),
    ):
        el = hvac.elbow_insertion_loss(bands, width=24 * inch, **kwargs)
        ax.semilogx(bands, np.asarray(el.values), style, color=color, lw=1.8,
                    marker="o", ms=4, label=label)
    ax.axvline(343.0 / (24 * inch), color=COLOR_FG, lw=1.2, ls=":", alpha=0.7)
    ax.annotate("$W/\\lambda = 1$", xy=(343.0 / (24 * inch), 1.0),
                xytext=(700.0, 1.4), fontsize="small", color=COLOR_FG)
    ax.set_title("(b) One bend, $W$ = 24 in", pad=8)

    ax = axes[1][0]
    for method, color, style in (("bies", COLOR_PRIMARY, "-"),
                                 ("long", COLOR_SECONDARY, "--")):
        er = hvac.end_reflection_loss(bands[:6], diameter=0.30, method=method)
        ax.semilogx(bands[:6], np.asarray(er.values), style, color=color,
                    lw=1.8, marker="o", ms=4, label=f'method="{method}"')
    ax.set_title("(c) Open end, 300 mm flush", pad=8)
    ax.set_ylabel("Attenuation [dB]")
    ax.set_xlabel(LABEL_FREQ_HZ)

    ax = axes[1][1]
    sil = hvac.splitter_silencer_insertion_loss(
        bands, height=24 * inch, length=5 * foot, airway_widths=[0.10] * 5,
        splitter_thickness=0.10)
    ax.semilogx(bands, np.asarray(sil.values), "-", color=COLOR_PRIMARY, lw=1.9,
                marker="o", ms=4, label="Geometry-only model, 5 ft, 5 airways")
    ax.plot([63.0, 125.0], [16.0, 21.0], "D", color=COLOR_SECONDARY, ms=9,
            mfc="none", mew=2.0, label="The unit Long's sheet specifies")
    ax.annotate("10.2 dB and 12.4 dB of low-frequency\nperformance the "
                "model cannot see", xy=(66.0, 11.0), xytext=(150.0, 5.0),
                fontsize="small", color=COLOR_SECONDARY,
                arrowprops={"arrowstyle": "->", "color": COLOR_SECONDARY,
                            "lw": 1.0})
    ax.set_title("(d) Splitter silencer", pad=8)
    ax.set_xlabel(LABEL_FREQ_HZ)

    for row in axes:
        for ax in row:
            ax.set_xlim(50.0, 10000.0)
            ax.set_ylim(bottom=0.0)
            format_frequency_axis(ax, 50.0, 10000.0)
            ax.grid(True, which="both", alpha=0.4)
            ax.legend(loc="upper left", fontsize="x-small")
    plt.tight_layout()
    save_figure(output_dir, "duct_attenuation_elements.svg")
    plt.close()


def generate_duct_sheet_verification(output_dir: str) -> None:
    """Which rows of Long's Table 14.9 the printed equations reproduce."""
    print("Generating duct_sheet_verification.svg...")
    from phonometry import fan_sound_power, nc_curve
    from phonometry.noise_control import hvac

    inch, foot = 0.0254, 0.3048
    cfm, in_wg = 0.0004719474432, 249.0
    bands = np.array([63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0])

    fan = fan_sound_power(volume_flow=5000 * cfm, static_pressure=2 * in_wg,
                          fan_type="forward_curved", relative_efficiency=80.0)
    flex = hvac.flexible_duct_insertion_loss(bands[:7], diameter=12 * inch,
                                             length=6 * foot)
    lined_small = hvac.lined_rectangular_duct_attenuation(
        bands, 18 * inch, 12 * inch, 6 * foot, 1 * inch, include_unlined=True)
    lined_large = hvac.lined_rectangular_duct_attenuation(
        bands, 36 * inch, 24 * inch, 5 * foot, 1 * inch, include_unlined=True)
    diffuser = hvac.diffuser_sound_power(bands, (24 * inch) ** 2,
                                         volume_flow=312 * cfm,
                                         pressure_drop=0.05 * in_wg)
    nc30 = np.asarray(nc_curve(30), dtype=float)[2:]   # OCTAVE_BANDS is 16 Hz up

    panels = (
        ("Fan, Eq. 13.1", [90, 86, 82, 79, 77, 75, 71, 61],
         np.asarray(fan.values)),
        ("Flexible duct, 12 in × 6 ft", [14, 14, 16, 15, 17, 22, 16],
         np.asarray(flex.values)),
        ("Lined duct, 18 × 12 in, 6 ft", [3, 3, 5, 11, 25, 22, 16, 13],
         np.asarray(lined_small.values)),
        ("Lined duct, 36 × 24 in, 5 ft", [2, 2, 3, 7, 15, 12, 11, 9],
         np.asarray(lined_large.values)),
        ("Supply diffuser, 312 cfm", [33, 32, 29, 23, 15, 4, 0, 0],
         np.asarray(diffuser.values)),
        ("NC 30 curve", [57, 48, 41, 35, 31, 29, 28, 27], nc30),
    )

    _fig, axes = plt.subplots(2, 3, figsize=(13.0, 7.6))
    for ax, (title, sheet_row, model_row) in zip(axes.ravel(), panels):
        printed = np.asarray(sheet_row, dtype=float)
        computed = np.asarray(model_row, dtype=float)
        n = min(printed.size, computed.size)
        printed, computed, axis = printed[:n], computed[:n], bands[:n]
        ax.fill_between(axis, printed, computed,
                        color=theme_fill(COLOR_SECONDARY, ax), zorder=0)
        ax.semilogx(axis, printed, "o", color=COLOR_FG, ms=6,
                    label="Long's printed row")
        ax.semilogx(axis, computed, "-", color=COLOR_PRIMARY, lw=1.9,
                    label="the library")
        # Long floors an absent row at 0 dB, so a band the sheet prints as
        # zero carries no level to disagree about.
        carries = printed > 0.0
        worst = float(np.max(np.abs(printed - computed)[carries]))
        ax.set_title(f"{title}  (worst $\\Delta$ {worst:.0f} dB)",
                     fontsize="medium", pad=7)
        ax.set_xlim(50.0, 10000.0)
        format_frequency_axis(ax, 50.0, 10000.0)
        ax.grid(True, which="both", alpha=0.4)
        ax.legend(loc="best", fontsize="x-small")
    for ax in axes[1]:
        ax.set_xlabel(LABEL_FREQ_HZ)
    for ax in axes[:, 0]:
        ax.set_ylabel("Level [dB]")
    plt.tight_layout()
    save_figure(output_dir, "duct_sheet_verification.svg")
    plt.close()


def generate_duct_regenerated_noise(output_dir: str) -> None:
    """The two velocity laws of section 4, as slopes rather than as numbers."""
    print("Generating duct_regenerated_noise.svg...")
    from phonometry import air_terminal_velocity_limit
    from phonometry.noise_control import hvac

    inch, cfm, in_wg = 0.0254, 0.0004719474432, 249.0
    bands = np.array([63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0])
    _fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.6))

    ax = axes[0]
    for velocity, color in ((5.0, COLOR_TERTIARY), (10.0, COLOR_PRIMARY),
                            (20.0, COLOR_SECONDARY), (40.0, COLOR_FG)):
        res = hvac.silencer_self_noise(bands, airway_velocity=velocity,
                                       passages=5, height=24 * inch)
        ax.semilogx(bands, np.asarray(res.values), "o-", color=color, lw=1.8,
                    ms=4, label=f"{velocity:.0f} m/s")
    ax.annotate("$55\\,\\mathrm{lg}(V/V_0)$: 16.6 dB per\ndoubling, every band",
                xy=(250.0, 55.0), xytext=(400.0, 26.0), fontsize="small",
                color=COLOR_FG,
                arrowprops={"arrowstyle": "->", "color": COLOR_FG, "lw": 1.0})
    ax.set_title("Silencer self-noise (Long Eq. 14.31)",
                 pad=10)
    ax.set_ylabel("Regenerated $L_W$ [dB re 1 pW]")
    ax.legend(loc="upper right", fontsize="small", title="Airway velocity")

    ax = axes[1]
    for flow, color in ((312.0, COLOR_TERTIARY), (624.0, COLOR_PRIMARY),
                        (1248.0, COLOR_SECONDARY)):
        res = hvac.diffuser_sound_power(bands, (24 * inch) ** 2,
                                        volume_flow=flow * cfm,
                                        pressure_drop=0.05 * in_wg)
        velocity = flow * cfm / (24 * inch) ** 2
        peak = 48.8 * velocity / 0.3048
        ax.semilogx(bands, np.asarray(res.values), "o-", color=color, lw=1.8,
                    ms=4, label=f"{flow:.0f} cfm ({velocity:.1f} m/s)")
        ax.axvline(peak, color=color, lw=1.0, ls=":")
    limit = air_terminal_velocity_limit(30, opening="supply")
    ax.annotate(f"peak band $f_\\mathrm{{P}} = 48.8\\,U_\\mathrm{{G}}$ moves up one octave per\n"
                f"doubling of the face velocity; ASHRAE Table 9 caps\nthe "
                f"neck of an RC 30 supply outlet at {limit:.1f} m/s",
                xy=(250.0, 33.0), xytext=(340.0, 12.0), fontsize="small",
                color=COLOR_FG,
                arrowprops={"arrowstyle": "->", "color": COLOR_FG, "lw": 1.0})
    ax.set_title("Diffuser sound power (Long Eqs. 13.27-13.33)",
                 pad=10)
    ax.legend(loc="upper right", fontsize="small",
              title="24 × 24 in device")

    for ax in axes:
        ax.set_xlim(50.0, 10000.0)
        format_frequency_axis(ax, 50.0, 10000.0)
        ax.set_xlabel(LABEL_FREQ_HZ)
        ax.grid(True, which="both", alpha=0.4)
    plt.tight_layout()
    save_figure(output_dir, "duct_regenerated_noise.svg")
    plt.close()


def generate_fan_sound_power(output_dir: str) -> None:
    """The source row: the efficiency staircase and the blade-tone increment."""
    print("Generating fan_sound_power.svg...")
    from phonometry import (
        fan_casing_attenuation,
        fan_efficiency_correction,
        fan_sound_power,
    )

    cfm, in_wg = 0.0004719474432, 249.0
    volume_flow, static_pressure = 5000 * cfm, 2 * in_wg
    fan_type = "forward_curved"
    _fig, ax = plt.subplots(figsize=(10, 6.4))
    for efficiency, color, style in ((90.0, COLOR_TERTIARY, ":"),
                                     (80.0, COLOR_PRIMARY, "-"),
                                     (55.0, COLOR_SECONDARY, "--")):
        fan = fan_sound_power(volume_flow, static_pressure,
                              fan_type=fan_type,
                              relative_efficiency=efficiency)
        ax.semilogx(np.asarray(fan.frequencies), np.asarray(fan.values), style,
                    color=color, lw=1.9, marker="o", ms=4,
                    label=f"{efficiency:.0f} % of peak static efficiency "
                          f"(+{fan_efficiency_correction(efficiency):.0f} dB)")
    tone = fan_sound_power(volume_flow, static_pressure, fan_type=fan_type,
                           relative_efficiency=80.0, blade_frequency=2000.0)
    # The same fan again, so the same colour: what changed is the blade tone,
    # not the fan. A shade back rather than an opacity back, which on the dark
    # page took the whole curve back to the page.
    ax.semilogx(np.asarray(tone.frequencies), np.asarray(tone.values), "-",
                color=theme_line(COLOR_PRIMARY, ax, quiet=0.55), lw=1.0,
                marker="^", ms=4,
                label="the same fan with its blade tone at 2 kHz")
    casing = fan_casing_attenuation()
    ax.semilogx(np.asarray(casing.frequencies), np.asarray(casing.values), "-.",
                color=COLOR_FG, lw=1.6, marker="s", ms=4,
                label="Casing attenuation (Table 13.8)")
    # The annotation starts clear of the efficiency inset (the committed
    # asset drew the inset over its first words).
    ax.annotate("the $C_{\\mathrm{BFI}}$ increment is 2 dB and lands whole in "
                "the\noctave containing the blade tone: 500 Hz at "
                "1200 rev/min\n× 24 blades, 2 kHz if the fan is selected "
                "faster",
                xy=(2000.0, 79.0), xytext=(460.0, 44.0), fontsize="small",
                color=COLOR_FG,
                arrowprops={"arrowstyle": "->", "color": COLOR_FG, "lw": 1.0})
    ax.set_xlim(50.0, 10000.0)
    ax.set_ylim(-4.0, 116.0)
    format_frequency_axis(ax, 50.0, 10000.0)
    ax.set_xlabel(LABEL_FREQ_HZ)
    ax.set_ylabel("Level [dB]")
    ax.set_title("The fan row: 5000 cfm at 2 in w.g., forward curved "
                 "(Long Eq. 13.1)", pad=10)
    ax.grid(True, which="both", alpha=0.4)
    ax.legend(loc="upper right", fontsize="small")

    inset = ax.inset_axes((0.07, 0.22, 0.33, 0.26))
    grid = np.linspace(40.0, 100.0, 400)
    inset.plot(grid, [fan_efficiency_correction(v) for v in grid],
               color=COLOR_PRIMARY, lw=1.8)
    inset.set_xlabel("static efficiency [% of peak]", fontsize="x-small")
    inset.set_ylabel("$C_{\\mathrm{EFF}}$ [dB]", fontsize="x-small")
    inset.tick_params(labelsize="x-small")
    inset.grid(True, alpha=0.4)
    plt.tight_layout()
    save_figure(output_dir, "fan_sound_power.svg")
    plt.close()


def generate_hvac_elbow_flow_noise(output_dir: str) -> None:
    """The bend a designer chooses, and the noise the same bend regenerates."""
    print("Generating hvac_elbow_flow_noise.svg...")
    from phonometry.noise_control import hvac

    bands = np.array([63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0])
    width = 0.3
    _fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.4))

    ax = axes[0]
    families = (
        ("Square, lined", {"bend_type": "square", "lined": True}, COLOR_PRIMARY, "-"),
        ("Square, bare", {"bend_type": "square"}, COLOR_SECONDARY, "--"),
        ("Square, vanes", {"bend_type": "square", "vanes": True}, COLOR_TERTIARY, "-."),
        ("Round, bare", {"bend_type": "round"}, COLOR_FG, ":"),
    )
    for label, kwargs, color, style in families:
        el = hvac.elbow_insertion_loss(bands, width=width, **kwargs)
        ax.semilogx(np.asarray(el.frequencies), np.asarray(el.values), style,
                    color=color, lw=1.8, marker="o", ms=4, label=label)
    f_unity = 343.0 / width
    ax.axvline(f_unity, color=COLOR_FG, lw=1.2, ls=":", alpha=0.7)
    ax.annotate("$W/\\lambda = 1$\n($W$ = 0.30 m, 1143 Hz)", xy=(f_unity, 1.6),
                xytext=(1.35 * f_unity, 1.2), fontsize="small", color=COLOR_FG)
    ax.set_xlim(50.0, 10000.0)
    ax.set_ylim(bottom=0.0)
    format_frequency_axis(ax, 50.0, 10000.0)
    ax.set_xlabel(LABEL_FREQ_HZ)
    ax.set_ylabel("Insertion loss [dB per bend]")
    ax.set_title("Elbow insertion loss (ASHRAE Table 8.11)",
                 pad=10)
    ax.grid(True, which="both", alpha=0.4)
    ax.legend(loc="upper left", fontsize="small")

    ax = axes[1]
    for velocity, color in ((5.0, COLOR_TERTIARY), (10.0, COLOR_PRIMARY),
                            (20.0, COLOR_SECONDARY)):
        fn = hvac.flow_noise_straight_duct(bands, flow_velocity=velocity,
                                           area=0.04)
        ax.semilogx(np.asarray(fn.frequencies), np.asarray(fn.values), "o-",
                    color=color, lw=1.8, ms=4,
                    label=f"Straight duct, $U$ = {velocity:.0f} m/s")
    bend = hvac.flow_noise_bend(bands, flow_velocity=10.0, area=0.04, height=0.2)
    ax.semilogx(np.asarray(bend.frequencies), np.asarray(bend.values), "s--",
                color=COLOR_FG, lw=1.8, ms=4,
                label="Mitred bend, $U$ = 10 m/s")
    ax.annotate("one bend at 10 m/s beats\na straight run at 20 m/s\nbelow 500 Hz",
                xy=(125.0, 57.8), xytext=(56.0, 73.5), va="top",
                fontsize="small", color=COLOR_FG,
                arrowprops={"arrowstyle": "->", "color": COLOR_FG, "lw": 1.0,
                            "shrinkB": 5.0})
    ax.set_xlim(50.0, 10000.0)
    ax.set_ylim(-16.0, 76.0)
    ax.set_yticks(np.arange(-10.0, 71.0, 10.0))
    format_frequency_axis(ax, 50.0, 10000.0)
    ax.set_xlabel(LABEL_FREQ_HZ)
    ax.set_ylabel("Regenerated $L_W$ [dB re 1 pW]")
    ax.set_title("Flow-generated sound power (VDI 2081, Bies Eq. 8.252)",
                 pad=10)
    ax.grid(True, which="both", alpha=0.4)
    ax.legend(loc="upper right", fontsize="small")

    plt.tight_layout()
    save_figure(output_dir, "hvac_elbow_flow_noise.svg")
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
                 pad=10)
    plt.tight_layout()
    save_figure(output_dir, "duct_path_cascade.svg")
    plt.close()


def generate_enclosure_required_tl(output_dir: str) -> None:
    """Norton problem 4.16: the panel TL an enclosure needs, band by band."""
    print("Generating enclosure_required_tl.svg...")
    from phonometry import enclosure_required_transmission_loss, mean_absorption

    bands = np.array([63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0])
    wool = [0.10, 0.20, 0.45, 0.65, 0.75, 0.80, 0.80, 0.80]
    concrete = [0.01, 0.01, 0.01, 0.02, 0.02, 0.02, 0.03, 0.03]
    lp1 = np.array([72.0, 79.0, 81.0, 84.0, 83.0, 81.0, 80.0, 75.0])
    nc45 = np.array([67.0, 60.0, 54.0, 49.0, 46.0, 44.0, 43.0, 41.0])
    radiating = 2 * (2.5 * 2.5) + 2 * (3.5 * 2.5) + 2.5 * 3.5
    machine = 2 * (1.5 * 1.5) + 2 * (2.5 * 1.5) + 1.5 * 2.5
    bare_floor = 2.5 * 3.5 - 1.5 * 2.5
    alpha = mean_absorption([(radiating, wool), (bare_floor + machine, concrete)])

    def _required(model: str) -> Any:
        return enclosure_required_transmission_loss(
            lp1 - nc45, radiating, radiating + bare_floor + machine, alpha,
            frequencies=bands, model=model,
        )

    norton, bies = _required("norton"), _required("bies")
    printed = np.array([14.4, 25.2, 28.9, 34.4, 35.2, 34.7, 34.7, 31.6])

    _fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(np.arange(bands.size) - 0.16, lp1 - nc45, width=0.32,
           color=theme_fill(COLOR_TERTIARY, ax), edgecolor=COLOR_TERTIARY,
           label="Target $\\mathrm{{IL}}$ = $L_{p1}$ − NC 45")
    ax.bar(np.arange(bands.size) + 0.16,
           np.asarray(norton.correction, dtype=float), width=0.32,
           color=theme_fill(COLOR_SECONDARY, ax), edgecolor=COLOR_SECONDARY,
           label="Interior correction $10\\,\\lg(S_\\mathrm{E}/R_\\mathrm{i})$")
    ax.plot(np.arange(bands.size),
            np.asarray(norton.panel_transmission_loss, dtype=float), "o-",
            color=COLOR_PRIMARY, lw=2.0, ms=6,
            label="Required panel $R$ (Norton)")
    ax.plot(np.arange(bands.size),
            np.asarray(bies.panel_transmission_loss, dtype=float), "--",
            color=COLOR_FG, lw=1.6, label="Required panel $R$ (Bies default)")
    ax.plot(np.arange(bands.size), printed, "D", color=COLOR_PRIMARY, ms=7,
            mfc="none", label="Norton's printed answer")
    ax.axhline(0.0, color=COLOR_GRID, lw=1.0)
    ax.annotate("the peak: the lining works and\nthe compressor is still loud",
                xy=(4.0, 35.2), xytext=(4.4, 18.0), fontsize="small",
                color=COLOR_FG,
                arrowprops={"arrowstyle": "->", "color": COLOR_FG, "lw": 1.0})
    ax.set_xticks(np.arange(bands.size))
    ax.set_xticklabels(["63", "125", "250", "500", "1k", "2k", "4k", "8k"])
    ax.set_xlabel(LABEL_FREQ_HZ)
    ax.set_ylabel("Level [dB]")
    ax.set_title("What the enclosure panels have to be: Norton problem 4.16",
                 pad=10)
    ax.set_ylim(-5.0, 45.0)
    ax.grid(True, axis="y", alpha=0.4)
    ax.legend(loc="upper left", fontsize="small", ncol=2)
    plt.tight_layout()
    save_figure(output_dir, "enclosure_required_tl.svg")
    plt.close()


def generate_room_to_room_partitions(output_dir: str) -> None:
    """Norton problem 4.21: three partitions, one room, one gap curve."""
    print("Generating room_to_room_partitions.svg...")
    from phonometry import (
        SourceRoom,
        equivalent_absorption_area,
        room_to_room_transmission,
    )

    bands = np.array([125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0])
    receiving = np.asarray(equivalent_absorption_area([
        (102.0, [0.04, 0.04, 0.09, 0.15, 0.17, 0.23]),   # walls
        (72.0, [0.02, 0.06, 0.14, 0.37, 0.60, 0.66]),    # floor
        (72.0, [0.30, 0.20, 0.15, 0.05, 0.05, 0.05]),    # ceiling
    ]), dtype=float)
    wall_area = 8.0 * 3.0
    partitions = (
        ("Two 13 mm wallboards, 64 mm gap", [18.0, 27.0, 37.0, 45.0, 43.0, 39.0],
         COLOR_PRIMARY, "o-"),
        ("125 mm plastered brick", [36.0, 36.0, 40.0, 46.0, 54.0, 57.0],
         COLOR_SECONDARY, "s--"),
        ("Double brick, 50 mm cavity", [37.0, 41.0, 48.0, 60.0, 61.0, 61.0],
         COLOR_TERTIARY, "^-."),
    )

    _fig, axes = plt.subplots(2, 1, figsize=(10, 8.4), sharex=True)

    ax = axes[0]
    for label, tl, color, style in partitions:
        res = room_to_room_transmission(
            bands, tl, wall_area, receiving,
            source=SourceRoom(level=90.0), label=label,
        )
        gap = np.asarray(res.noise_reduction, dtype=float) - np.asarray(tl)
        ax.semilogx(bands, gap, style, color=color, lw=1.8, ms=7,
                    mfc="none", label=label)
    ax.axhline(0.0, color=COLOR_FG, lw=1.2)
    ax.annotate("the three curves are one curve:\nthe gap belongs to the room, "
                "not to the wall", xy=(250.0, -0.22), xytext=(300.0, -2.6),
                fontsize="small", color=COLOR_FG,
                arrowprops={"arrowstyle": "->", "color": COLOR_FG, "lw": 1.0})
    ax.set_ylim(-4.0, 6.5)
    ax.set_ylabel("NR − TL [dB]")
    ax.set_title("What the receiving room adds to (or takes from) the wall",
                 pad=10)
    ax.grid(True, which="both", alpha=0.4)
    ax.legend(loc="upper left", fontsize="small")

    ax = axes[1]
    ax.semilogx(bands, receiving, "o-", color=COLOR_PRIMARY, lw=1.8, ms=6,
                label="Receiving-room absorption $S_2\\alpha_2$")
    ax.axhline(wall_area, color=COLOR_SECONDARY, lw=1.8, ls="--",
               label=f"Partition area $S_\\mathrm{{w}}$ = {wall_area:.0f} m²")
    ax.fill_between(bands, wall_area, receiving, where=receiving > wall_area,
                    color=theme_fill(COLOR_PRIMARY, ax), zorder=0)
    ax.annotate("crossing: below it the wall delivers less than its TL,\n"
                "above it, more", xy=(430.0, 27.0), xytext=(520.0, 8.0),
                fontsize="small", color=COLOR_FG,
                arrowprops={"arrowstyle": "->", "color": COLOR_FG, "lw": 1.0})
    ax.set_ylim(0.0, 82.0)
    format_frequency_axis(ax, 110.0, 4600.0)
    ax.set_xlim(110.0, 4600.0)
    ax.set_xlabel(LABEL_FREQ_HZ)
    ax.set_ylabel("Area [m²]")
    ax.set_title("Why: the same room, band by band",
                 pad=10)
    ax.grid(True, which="both", alpha=0.4)
    ax.legend(loc="upper left", fontsize="small")

    plt.tight_layout()
    save_figure(output_dir, "room_to_room_partitions.svg")
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
        pad=10,
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
                 pad=10)
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
    # The library legend carries "Integrated -23.0 LUFS" assembled with
    # ``format``: restate its sign before the legend below is drawn.
    for line in ax.get_lines():
        line.set_label(_ASCII_MINUS_RE.sub("−", str(line.get_label())))
    finite = res.momentary[np.isfinite(res.momentary)]
    ungated = 10.0 * np.log10(np.mean(10.0 ** (finite / 10.0)))
    ax.axhline(ungated, color=COLOR_TERTIARY, ls="-.", lw=1.6,
               label=f"Ungated mean {_fmt_minus(ungated, '.1f')} LUFS")
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
    # Its legend reads "Integrated -22.6 LUFS" assembled with ``format``:
    # restate the sign with the typographic minus of the axes.
    legend = ax.get_legend()
    if legend is not None:
        for entry in legend.get_texts():
            entry.set_text(_ASCII_MINUS_RE.sub("−", entry.get_text()))
    plt.tight_layout()
    save_figure(output_dir, "loudness_range.svg")


def generate_itu_r_468_weighting(output_dir: str) -> None:
    """The ITU-R BS.468-4 network and its CCIR-RMS renormalisation."""
    print("Generating itu_r_468_weighting...")
    from phonometry import electroacoustics

    freqs = np.geomspace(20.0, 20000.0, 600)
    w468 = np.asarray(electroacoustics.itu_r_468_weighting(freqs), dtype=float)
    ccir = w468 - 5.63

    _fig, ax = plt.subplots(figsize=(10, 6))
    fa, wa = measure_weighting_response(48000, "A")
    sel = (fa >= 20.0) & (fa <= 20000.0)
    # A wide band of the page's own ink, laid behind the networks as the
    # reference everybody knows. It is a colour, not an opacity: 18 % of the
    # ink is a pale grey on the white page and a shade of the dark page on the
    # dark one, which is where the reference used to disappear.
    ax.semilogx(fa[sel], wa[sel], color=theme_line(COLOR_FG, ax, quiet=0.18),
                linewidth=4.0, label="A-weighting (reference)")
    ax.semilogx(freqs, w468, color=COLOR_PRIMARY, linewidth=2.2,
                label="ITU-R BS.468-4 (0 dB at 1 kHz)")
    ax.semilogx(freqs, ccir, color=COLOR_SECONDARY, linewidth=1.8,
                linestyle="--", label="AES17 CCIR-RMS (0 dB at 2 kHz)")

    for f0, level, col, txt in (
        (6300.0, float(electroacoustics.itu_r_468_weighting([6300.0])[0]),
         COLOR_PRIMARY, "+12.2 dB at 6.3 kHz"),
        (1000.0, 0.0, COLOR_PRIMARY, "0 dB at 1 kHz"),
        (1000.0, -5.63, COLOR_SECONDARY, "−5.63 dB at 1 kHz"),
    ):
        ax.plot([f0], [level], "o", color=col, markersize=7, zorder=6)
        ax.annotate(txt, xy=(f0, level), xytext=(-10, 10),
                    textcoords="offset points", ha="right", fontsize=9,
                    color=col)

    ax.set_xlabel(LABEL_FREQ_HZ)
    ax.set_ylabel("Weighting [dB]")
    ax.set_title("The ITU-R BS.468-4 Network and Its CCIR-RMS Form",
                 pad=12)
    ax.set_xlim(20.0, 20000.0)
    ax.set_ylim(-45.0, 20.0)
    format_frequency_axis(ax, 20.0, 20000.0)
    ax.grid(which="both", color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", fontsize=9)
    ax.text(0.015, 0.96,
            "for a 100 Hz fundamental the network cuts $d_2$ by 13.8 dB\n"
            "and $d_3$ by 10.3 dB, and lifts the 10th order and above",
            transform=ax.transAxes, va="top", ha="left", fontsize=8.5,
            color=COLOR_FG)
    plt.tight_layout()
    save_figure(output_dir, "itu_r_468_weighting.svg")
    plt.close()


def generate_intermodulation_tests(output_dir: str) -> None:
    """Which products the three scalar intermodulation tests count."""
    print("Generating intermodulation_tests...")
    from phonometry import electroacoustics

    fs = 96000
    n = fs                              # 1 s -> 1 Hz bins, every tone on a bin
    t = np.arange(n) / fs
    a2, a3 = 0.05, 0.02                 # one weak non-linearity for all three

    def through(x: Any) -> Any:
        return x + a2 * x**2 + a3 * x**3

    def spectrum(sig: Any) -> tuple[Any, Any]:
        win = np.hanning(len(sig))
        mag = np.abs(np.fft.rfft(sig * win)) * 2.0 / np.sum(win)
        f = np.fft.rfftfreq(len(sig), d=1.0 / fs)
        return f, 20.0 * np.log10(np.maximum(mag, 1e-12) / np.max(mag))

    # (a) difference-frequency test, 13 and 14 kHz (IEC 60268-3 14.12.8).
    x_dfd = 0.5 * (np.sin(2 * np.pi * 13000.0 * t)
                   + np.sin(2 * np.pi * 14000.0 * t))
    y_dfd = through(x_dfd)
    # (b) total difference-frequency test at the default 8 / 11.95 kHz.
    x_tdfd = 0.5 * (np.sin(2 * np.pi * 8000.0 * t)
                    + np.sin(2 * np.pi * 11950.0 * t))
    y_tdfd = through(x_tdfd)
    # (c) DIM: a 15 kHz sine plus a low-passed 3.15 kHz square, 1:4 peak
    # (14.12.9). The square is built from its own band-limited Fourier series
    # so that nothing aliases before the device sees it, then taken through
    # the first-order 30 kHz low-pass the clause specifies.
    square = np.zeros_like(t)
    for k in range(1, 200, 2):
        if k * 3150.0 >= fs / 2.0:
            break
        square += np.sin(2 * np.pi * k * 3150.0 * t) / k
    square *= 4.0 / np.pi
    b_lp, a_lp = scipy_signal.butter(1, 30000.0, fs=fs)
    x_dim = 0.8 * scipy_signal.lfilter(b_lp, a_lp, square) + 0.2 * np.sin(
        2 * np.pi * 15000.0 * t)
    y_dim = through(x_dim)

    dfd2 = electroacoustics.difference_frequency_distortion(
        y_dfd, fs, 13e3, 14e3, order=2)
    tdfd = electroacoustics.total_difference_frequency_distortion(y_tdfd, fs)
    dim = electroacoustics.dynamic_intermodulation_distortion(y_dim, fs)

    # Each mark is (frequency, label, label offset in points, horizontal
    # alignment): the offsets keep every label off its own stem, off its
    # neighbours' markers and off the panel frame.
    panels = (
        (y_dfd, (0.0, 16000.0),
         (f"(a) Difference frequency, 13 / 14 kHz — "
          f"$d(d,2)$ = {dfd2 * 100:.2f} %"),
         ((1000.0, "$f_2 - f_1$", (0.0, 10.0), "center"),
          (12000.0, "$2f_1 - f_2$", (0.0, 22.0), "center"),
          (15000.0, "$2f_2 - f_1$", (0.0, 10.0), "center")),
         "denominator: $a(f_1) + a(f_2)$"),
        (y_tdfd, (0.0, 16000.0),
         (f"(b) Total difference frequency, 8 / 11.95 kHz — "
          f"$d(\\mathrm{{TDFD}})$ = {tdfd * 100:.2f} %"),
         # The two sidebands are 100 Hz apart and 17 dB apart, so the lower
         # one is labelled sideways with a leader instead of overhead.
         ((3950.0, "$f_0 - \\delta$", (0.0, 10.0), "center"),
          (4050.0, "$f_0 + \\delta$", (30.0, 0.0), "left")),
         "only the two in-band products count"),
        (y_dim, (0.0, 16000.0),
         (f"(c) Dynamic intermodulation, 15 kHz + 3.15 kHz square — "
          f"DIM = {dim * 100:.2f} %"),
         tuple((abs(k * 3150.0 - 15000.0), f"$k$ = {k}",
                (0.0, 10.0 if j % 2 == 0 else 22.0), "center")
               for j, k in enumerate((1, 2, 3, 4, 5))
               if 0.0 < abs(k * 3150.0 - 15000.0) < 15000.0),
         "denominator: the 15 kHz sine alone"),
    )

    _fig, axes = plt.subplots(3, 1, figsize=(10, 10.5), sharex=True)
    for ax, (sig, xlim, title, marks, note) in zip(axes, panels, strict=True):
        f, mag_db = spectrum(sig)
        ax.plot(f, mag_db, color=COLOR_PRIMARY, linewidth=0.9, alpha=0.85)
        for fm, lab, offset, halign in marks:
            k = int(np.argmin(np.abs(f - fm)))
            ax.plot([f[k]], [mag_db[k]], "o", color=COLOR_SECONDARY,
                    markersize=6, zorder=6)
            leader = ({"arrowstyle": "-", "color": COLOR_SECONDARY,
                       "linewidth": 0.8, "shrinkA": 1.0, "shrinkB": 4.0}
                      if offset[0] != 0.0 else None)
            ax.annotate(lab, xy=(f[k], mag_db[k]), xytext=offset,
                        textcoords="offset points", ha=halign,
                        va="center" if offset[0] != 0.0 else "baseline",
                        fontsize=8, color=COLOR_SECONDARY,
                        arrowprops=leader)
        ax.set_xlim(*xlim)
        # The top 20 dB carry no data (every spectrum is normalised to 0 dB),
        # which leaves the corner note a clear band of its own instead of
        # sitting down among the stems at the noise floor.
        ax.set_ylim(-110.0, 20.0)
        ax.set_ylabel("Level [dB]")
        ax.set_title(title, fontsize=10, pad=6)
        ax.grid(which="major", color=COLOR_GRID, linestyle="--", alpha=0.5)
        ax.set_axisbelow(True)
        ax.text(0.985, 10.0, note, transform=ax.get_yaxis_transform(),
                va="center", ha="right", fontsize=8.5, color=COLOR_FG)
    axes[-1].set_xlabel("Frequency [Hz]")
    plt.tight_layout()
    save_figure(output_dir, "intermodulation_tests.svg")
    plt.close()


def generate_feedback_stability(output_dir: str) -> None:
    """The gain structure of Long's criterion, one open microphone and four."""
    print("Generating feedback_stability...")
    from phonometry import electroacoustics as ea

    one = ea.feedback_stability(-6.0, 76.0, 80.0)
    four = ea.feedback_stability(-6.0, 76.0, 80.0, open_microphones=4)

    _fig, axes = plt.subplots(2, 1, figsize=(10, 9.2), sharey=True)
    titles = (
        f"One open microphone — headroom {_fmt_minus(one.headroom, '+.1f')} dB",
        ("Four open microphones — headroom "
        f"{_fmt_minus(four.headroom, '+.1f')} dB"),
    )
    for ax, res, title in zip(axes, (one, four), titles, strict=True):
        res.plot(ax=ax, language=_LANG)
        # The library assembles the four bar readouts with ``format``: restate
        # their negative signs with the typographic minus of the axes.
        for reading in ax.texts:
            reading.set_text(_ASCII_MINUS_RE.sub("−", reading.get_text()))
        ax.set_title(title, fontsize=11, pad=8)
    plt.tight_layout()
    save_figure(output_dir, "feedback_stability.svg")
    plt.close()


def generate_microphone_patterns(output_dir: str) -> None:
    """The first-order family and the directivity index each member returns."""
    print("Generating microphone_patterns...")
    from phonometry import MicrophoneDirectivity, microphone_characteristics

    freqs = np.geomspace(20.0, 20000.0, 200)
    response = -10.0 * np.log10(1.0 + (30.0 / freqs) ** 4)
    angles = np.linspace(0.0, 179.0, 359)
    family = (
        (0.0, "Omnidirectional"),
        (1.0 / 3.0, "Subcardioid"),
        (0.5, "Cardioid"),
        (0.63, "Supercardioid"),
        (0.75, "Hypercardioid"),
        (1.0, "Figure-of-eight"),
    )
    # Six distinguishable inks; the two that share 4.8 dB (cardioid and
    # figure-of-eight) are deliberately the two heaviest.
    colors = (COLOR_MUTED, "#9467bd", COLOR_PRIMARY, COLOR_TERTIARY,
              "#e8a838", COLOR_SECONDARY)

    _fig, ax_any = plt.subplots(figsize=(8.4, 8.4),
                                subplot_kw={"projection": "polar"})
    ax: Any = ax_any
    theta = np.radians(np.linspace(0.0, 360.0, 721))
    for (b, name), col in zip(family, colors, strict=True):
        polar_db = 20.0 * np.log10(
            np.maximum(np.abs((1.0 - b) + b * np.cos(np.radians(angles))),
                       10 ** (-30.0 / 20.0)))
        res = microphone_characteristics(
            freqs, response, 12.5, tolerance_db=3.0,
            directivity=MicrophoneDirectivity(polar=(angles, polar_db),
                                              frequency=1000.0))
        assert res.directivity_index_db is not None, "polar input gives a DI"
        di = float(res.directivity_index_db)
        mag = np.maximum(np.abs((1.0 - b) + b * np.cos(theta)),
                         10 ** (-30.0 / 20.0))
        radius = np.maximum(20.0 * np.log10(mag), -30.0)
        ax.plot(theta, radius, color=col, linewidth=2.0,
                label=f"{name} ($b$ = {b:.2f}): DI = {di:.1f} dB")
    ax.set_ylim(-30.0, 0.0)
    ax.set_yticks([-30.0, -20.0, -10.0, 0.0])
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_title("The First-Order Family and Its Directivity Index "
                 "(IEC 60268-4 13.2.2)", pad=22)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.6)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.05), fontsize=9,
              ncol=2)
    plt.tight_layout()
    save_figure(output_dir, "microphone_patterns.svg")
    plt.close()


def generate_microphone_noise_weightings(output_dir: str) -> None:
    """How much of the dB(A)-to-dB(468) gap the *weighting* accounts for.

    The customary "about 10 dB" mixes two effects: the network, which is what
    the library implements and what this figure computes, and the ITU-R 468
    quasi-peak detector, which it does not. Separating them also shows that
    the weighting half is not a constant: it depends on the shape of the
    particular capsule's noise.
    """
    print("Generating microphone_noise_weightings...")
    from phonometry import electroacoustics

    noise_f = np.geomspace(20.0, 20000.0, 31)
    falling = 6.0 + 12.0 * np.log10(1000.0 / noise_f)   # the page's capsule
    bright = 6.0 + 3.0 * np.log10(noise_f / 1000.0)     # a hissier capsule

    fa, wa_curve = measure_weighting_response(48000, "A")
    a_w = np.interp(noise_f, fa, wa_curve)
    w468 = np.asarray(electroacoustics.itu_r_468_weighting(noise_f),
                      dtype=float)

    def total(bands: Any, weight: Any) -> float:
        return float(10.0 * np.log10(np.sum(10.0 ** ((bands + weight) / 10.0))))

    _fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(10, 8.6), gridspec_kw={"height_ratios": [2.0, 1.2]})

    ax_top.semilogx(noise_f, falling, color=COLOR_FG, linewidth=2.0,
                    marker="o", markersize=4,
                    label="Inherent noise, unweighted")
    ax_top.semilogx(noise_f, falling + a_w, color=COLOR_PRIMARY, linewidth=1.8,
                    linestyle="--", label="A-weighted")
    ax_top.semilogx(noise_f, falling + w468, color=COLOR_SECONDARY,
                    linewidth=1.8, linestyle="--",
                    label="ITU-R BS.468-4 weighted")
    ax_top.set_ylabel("Band level [dB]")
    ax_top.set_xlabel(LABEL_FREQ_HZ)
    ax_top.set_title("One Noise Voltage, Two Networks (IEC 60268-4 17.2)",
                     pad=12)
    ax_top.set_xlim(20.0, 20000.0)
    format_frequency_axis(ax_top, 20.0, 20000.0)
    ax_top.grid(which="both", color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax_top.set_axisbelow(True)
    ax_top.legend(loc="lower left", fontsize=9)

    groups = (
        ("this capsule\n($1/f$, falling)", falling),
        ("a hissier capsule\n(rising to 20 kHz)", bright),
    )
    width, offset = 0.34, 0.19
    for k, (label, bands) in enumerate(groups):
        la, l468 = total(bands, a_w), total(bands, w468)
        ax_bot.bar(k - offset, la, width=width, color=COLOR_PRIMARY,
                   label="A-weighted" if k == 0 else None)
        ax_bot.bar(k + offset, l468, width=width, color=COLOR_SECONDARY,
                   label="ITU-R BS.468-4" if k == 0 else None)
        for xpos, val in ((k - offset, la), (k + offset, l468)):
            ax_bot.annotate(f"{val:.1f} dB", xy=(xpos, val), xytext=(0, 4),
                            textcoords="offset points", ha="center",
                            fontsize=9, color=COLOR_FG)
        ax_bot.annotate(f"network alone: {_fmt_minus(l468 - la, '+.1f')} dB",
                        xy=(k, max(la, l468)), xytext=(0, 22),
                        textcoords="offset points", ha="center", fontsize=9,
                        color=COLOR_FG, fontweight="bold")
    ax_bot.set_xticks([0, 1])
    ax_bot.set_xticklabels([g[0] for g in groups])
    ax_bot.set_ylabel("Band-summed level [dB]")
    ax_bot.set_ylim(0.0, 34.0)
    ax_bot.grid(axis="y", color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax_bot.set_axisbelow(True)
    ax_bot.legend(loc="upper left", fontsize=9)
    ax_bot.set_title("the quasi-peak detector of ITU-R BS.468-4 adds the rest "
                     "of the customary ~10 dB, and is not implemented here",
                     fontsize=9.5, pad=8)
    plt.tight_layout()
    save_figure(output_dir, "microphone_noise_weightings.svg")
    plt.close()


def generate_swept_sine_harmonic_responses(output_dir: str) -> None:
    """Each separated order is a full frequency response, not a number."""
    print("Generating swept_sine_harmonic_responses...")
    from phonometry import swept_sine_distortion, synchronized_sweep_signal

    fs = 48000
    f1, f2, seconds = 20.0, 6000.0, 4.0
    a2, a3 = 0.12, 0.08
    x = synchronized_sweep_signal(fs, f1, f2, seconds)
    b, a = scipy_signal.butter(2, 3000.0, fs=fs)
    y = scipy_signal.lfilter(b, a, x + a2 * x**2 + a3 * x**3)
    res = swept_sine_distortion(y, fs, f1, f2, seconds, n_harmonics=3)

    h1_ref = 1.0 + 3.0 * a3 / 4.0
    _fig, ax = plt.subplots(figsize=(10, 6))
    orders = (
        (0, "$|H_1(f)|$", COLOR_PRIMARY, h1_ref),
        (1, "$|H_2(f)|$", COLOR_SECONDARY, a2 / 2.0),
        (2, "$|H_3(f)|$", COLOR_TERTIARY, a3 / 4.0),
    )
    for idx, label, col, asymptote in orders:
        mag = np.abs(np.asarray(res.harmonic_responses[idx]))
        band = (res.frequencies >= (idx + 1) * f1) & (
            res.frequencies <= (idx + 1) * f2)
        ax.semilogx(res.frequencies[band],
                    20.0 * np.log10(np.maximum(mag[band], 1e-9)),
                    color=col, linewidth=1.8,
                    label=f"{label} over $[{idx + 1}f_1, {idx + 1}f_2]$")
        ax.axhline(20.0 * np.log10(asymptote), color=col, linestyle=":",
                   linewidth=1.2, alpha=0.8)

    ax.set_xlabel("Frequency of the harmonic itself [Hz]")
    ax.set_ylabel("Magnitude [dB]")
    ax.set_title("The Separated Harmonic Frequency Responses "
                 "(Farina / Novak)", pad=12)
    ax.set_xlim(20.0, 20000.0)
    ax.set_ylim(-70.0, 6.0)
    format_frequency_axis(ax, 20.0, 20000.0)
    ax.grid(which="both", color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", fontsize=9)
    ax.text(0.56, 0.04,
            "dotted: the Chebyshev levels $1 + 3a_3/4$, $a_2/2$ and $a_3/4$;\n"
            "each order rolls off at the 3 kHz post-filter, not at $n f_2$",
            transform=ax.transAxes, va="bottom", ha="center", fontsize=8.5,
            color=COLOR_FG)
    plt.tight_layout()
    save_figure(output_dir, "swept_sine_harmonic_responses.svg")
    plt.close()


def generate_swept_sine_methods(output_dir: str) -> None:
    """Synchronized against Farina on one recording: band and phase."""
    print("Generating swept_sine_methods...")
    from phonometry import swept_sine_distortion, synchronized_sweep_signal

    fs = 48000
    f1, f2, seconds = 20.0, 6000.0, 4.0
    a2, a3 = 0.12, 0.08
    x = synchronized_sweep_signal(fs, f1, f2, seconds)
    # The memoryless cubic, with no post-filter: the Chebyshev identities then
    # fix |H2| = a2/2 and arg H2 = -pi/2 at every frequency, so any departure
    # on the figure is the method and not the device.
    y = x + a2 * x**2 + a3 * x**3

    # The synchronized generator quantises the duration slightly; the Farina
    # analysis of the same recording has to be told the length it really is.
    actual = len(x) / fs
    sync = swept_sine_distortion(y, fs, f1, f2, actual, n_harmonics=3)
    farina = swept_sine_distortion(y, fs, f1, f2, actual, n_harmonics=3,
                                   method="farina")

    _fig, (ax_mag, ax_ph) = plt.subplots(
        2, 1, figsize=(10, 8.0), sharex=True,
        gridspec_kw={"height_ratios": [1.4, 1.0]})

    for res, label, col, style in ((sync, "synchronized", COLOR_PRIMARY, "-"),
                                   (farina, "farina", COLOR_SECONDARY, "--")):
        h2 = np.asarray(res.harmonic_responses[1])
        band = (res.frequencies >= 2 * f1) & (
            res.frequencies <= (2 * f2 if label == "synchronized" else f2))
        ax_mag.semilogx(res.frequencies[band],
                        20.0 * np.log10(np.maximum(np.abs(h2[band]), 1e-9)),
                        color=col, linewidth=1.8, linestyle=style,
                        label=f'$|H_2|$ — method="{label}"')
        ax_ph.semilogx(res.frequencies[band],
                       np.unwrap(np.angle(h2[band])), color=col,
                       linewidth=1.6, linestyle=style,
                       label=f'$\\arg H_2$ — method="{label}"')

    ax_mag.axhline(20.0 * np.log10(a2 / 2.0), color=COLOR_FG, linestyle=":",
                   linewidth=1.2, label="Chebyshev level $a_2/2$")
    ax_mag.axvline(f2, color=COLOR_FG, linestyle=":", linewidth=1.2)
    ax_mag.annotate("$f_2$ = 6 kHz: the Farina band stops here",
                    xy=(f2, -40.0), xytext=(-8, 0),
                    textcoords="offset points", ha="right", fontsize=9,
                    color=COLOR_FG)
    ax_mag.axvline(2 * f2, color=COLOR_PRIMARY, linestyle=":", linewidth=1.2)
    ax_mag.annotate("$2 f_2$ = 12 kHz", xy=(2 * f2, -40.0), xytext=(8, 0),
                    textcoords="offset points", ha="left", fontsize=9,
                    color=COLOR_PRIMARY)
    ax_mag.set_ylabel("Magnitude [dB]")
    ax_mag.set_ylim(-70.0, 0.0)
    ax_mag.set_title("Same Recording, Two Deconvolutions "
                     "(Novak et al. 2015, Fig. 6)", pad=12)
    ax_ph.axhline(-np.pi / 2.0, color=COLOR_FG, linestyle=":", linewidth=1.2)
    ax_ph.annotate("true phase of $H_2$: $-\\pi/2$ at every frequency",
                   xy=(60.0, -np.pi / 2.0), xytext=(0, 8),
                   textcoords="offset points", ha="left", fontsize=9,
                   color=COLOR_FG)
    ax_ph.set_ylabel("Unwrapped phase [rad]")
    ax_ph.set_xlabel(LABEL_FREQ_HZ)
    for axis in (ax_mag, ax_ph):
        axis.set_xlim(40.0, 20000.0)
        format_frequency_axis(axis, 40.0, 20000.0)
        axis.grid(which="both", color=COLOR_GRID, linestyle="--", alpha=0.5)
        axis.set_axisbelow(True)
        axis.legend(loc="lower left", fontsize=9)
    plt.tight_layout()
    save_figure(output_dir, "swept_sine_methods.svg")
    plt.close()
