#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Figures for the perception guides: loudness, tonality, speech and hearing.

What a listener makes of the signal: the loudness, sharpness, roughness and
fluctuation-strength models, tonal audibility and prominence, speech
intelligibility, and the hearing-threshold and hearing-loss curves. Everything
here is embedded by a page under ``perception/``.
"""

import itertools
from functools import cache

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from phonometry._plot.common import format_frequency_axis, theme_fill, theme_line

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

#: The ECMA-418-2 units carry the Sottek Hearing Model subscript
#: (Bark_HMS, sone_HMS, tu_HMS, vacil_HMS); mathtext is the only clean way
#: to set that subscript, and it stays roman on both sides: the unit name
#: and the model tag are labels, not variables.
_HMS_UNITS = {
    "Bark_HMS": r"$\mathrm{Bark}_{\mathrm{HMS}}$",
    "sone_HMS": r"$\mathrm{sone}_{\mathrm{HMS}}$",
    "tu_HMS": r"$\mathrm{tu}_{\mathrm{HMS}}$",
    "vacil_HMS": r"$\mathrm{vacil}_{\mathrm{HMS}}$",
}


def _hms(unit: str) -> str:
    """The unit with its HMS model subscript composed, others untouched."""
    return _HMS_UNITS.get(unit, unit)


def generate_equal_loudness_contours(output_dir: str) -> None:
    """Plot the ISO 226:2023 normal equal-loudness-level contours."""
    print("Generating equal_loudness_contours.png...")
    from phonometry import equal_loudness_contours

    _, ax = plt.subplots(figsize=(10, 7))
    # The result's own .plot() draws the contour family plus the hearing
    # threshold on a 1k/2k-labelled log frequency axis (ISO 226:2023 Formula 1).
    equal_loudness_contours().plot(ax=ax)
    ax.set_ylim(-10, 130)
    ax.set_title("Normal Equal-Loudness-Level Contours (ISO 226:2023)",
                 pad=12)
    save_figure(output_dir, "equal_loudness_contours.png")
    plt.close()


def generate_tonality_spectrum(output_dir: str) -> None:
    """Annotated spectrum for the tone-to-noise ratio method."""
    print("Generating tonality_spectrum.png...")
    from phonometry import tone_to_noise_ratio
    from phonometry.psychoacoustics.quality.tonality import (
        _averaged_spectrum,
        _critical_band,
    )

    fs = 48000
    rng = np.random.default_rng(21)
    tt = np.arange(fs * 30) / fs
    x = (np.sqrt(2) * 0.1 * np.sin(2 * np.pi * 1000 * tt)
         + 0.05 * rng.standard_normal(tt.size))
    result = tone_to_noise_ratio(x, fs)
    freqs, power, _ = _averaged_spectrum(x - np.mean(x), fs, 1.0)
    f1, f2, _ = _critical_band(result.frequency)

    _, ax = plt.subplots(figsize=(10, 6))
    sel_band = (freqs > 700) & (freqs < 1400)
    db = 10 * np.log10(np.maximum(power, 1e-18))
    ax.plot(freqs[sel_band], db[sel_band], color=COLOR_PRIMARY, linewidth=1.0,
            label="Averaged FFT spectrum (Hann)")
    ax.axvspan(f1, f2, color=COLOR_TERTIARY, alpha=0.15,
               label="Critical band around the tone")
    ax.axvline(result.frequency, color=COLOR_SECONDARY, linewidth=1.4,
               linestyle="--")
    ax.annotate(
        f"TNR = {result.ratio_db:.1f} dB\n(criterion {result.criterion_db:.1f} dB)",
        xy=(result.frequency, db.max() - 2), xytext=(1120, db.max() - 8),
        fontsize=11, arrowprops={"arrowstyle": "->", "lw": 1.0},
    )
    ax.set_title("Tone-to-Noise Ratio (ECMA-418-1, clause 11)",
                 pad=12)
    ax.set_xlim(700, 1400)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Bin power [dB]")
    ax.legend(loc="upper right", fontsize=9)
    save_figure(output_dir, "tonality_spectrum.png")
    plt.close()


def generate_loudness_pattern(output_dir: str) -> None:
    """Specific loudness N'(z) of a narrowband vs a broadband sound."""
    print("Generating loudness_pattern.png...")
    from phonometry import loudness_zwicker_from_spectrum

    # 28 one-third-octave band levels, 25 Hz .. 12.5 kHz (ISO 532-1
    # clause 5.3). Index 16 is the 1 kHz band.
    narrow_levels = np.full(28, -60.0)
    narrow_levels[16] = 60.0
    narrow = loudness_zwicker_from_spectrum(narrow_levels)
    flat = loudness_zwicker_from_spectrum(np.full(28, 60.0))

    z = np.arange(1, 241) * 0.1  # 0.1-Bark steps up to 24 Bark

    _, ax = plt.subplots(figsize=(10, 6))
    ax.fill_between(z, flat.specific, color=COLOR_SECONDARY, alpha=0.25)
    ax.plot(z, flat.specific, color=COLOR_SECONDARY, linewidth=1.6,
            label=f"Flat broadband 60 dB — $N$ = {flat.loudness:.1f} sone")
    ax.fill_between(z, narrow.specific, color=COLOR_PRIMARY, alpha=0.35)
    ax.plot(z, narrow.specific, color=COLOR_PRIMARY, linewidth=1.6,
            label=f"1 kHz narrowband — $N$ = {narrow.loudness:.1f} sone")

    peak_z = float(z[np.argmax(narrow.specific)])
    ax.annotate(
        "Shaded area = total loudness $N$",
        xy=(peak_z + 0.6, float(narrow.specific.max()) * 0.45),
        xytext=(12.5, float(narrow.specific.max()) * 0.75),
        fontsize=10, arrowprops={"arrowstyle": "->", "lw": 0.9},
    )
    ax.set_title("Specific Loudness Pattern (ISO 532-1 Zwicker)",
                 pad=12)
    ax.set_xlabel("Critical-band rate $z$ [Bark]")
    ax.set_ylabel(r"Specific loudness $N^{\prime}$ [sone/Bark]")
    ax.set_xlim(0, 24)
    # Headroom above the tallest pattern so the legend stays clear of it.
    ax.set_ylim(0, float(flat.specific.max()) * 1.28)
    ax.set_xticks([0, 4, 8, 12, 16, 20, 24])
    ax.legend(loc="upper right", fontsize=9)
    save_figure(output_dir, "loudness_pattern.png")
    plt.close()


def _zwicker_burst_train(fs: int) -> np.ndarray:
    """A 4.6 s train of 1 kHz bursts stepping 45 dB to 85 dB, in pascals.

    Five bursts separated by 0.35 s of silence, the loudest of them short:
    a signal with real dynamics, so the percentile loudness of ISO 532-1
    clause 6.4 has something to be a percentile *of*, and so the brief peak
    separates N5 from Nmax the way an impulsive event does.
    """
    parts: list[np.ndarray] = []
    for level, seconds in ((45.0, 0.6), (55.0, 0.6), (65.0, 0.6),
                           (75.0, 0.6), (85.0, 0.25)):
        n = int(seconds * fs)
        k = np.arange(n)
        env = np.minimum(1.0, np.minimum(k, n - 1 - k) / (0.02 * fs))
        amplitude = np.sqrt(2) * 2e-5 * 10 ** (level / 20)
        parts.append(amplitude * np.sin(2 * np.pi * 1000.0 * k / fs) * env)
        parts.append(np.zeros(int(0.35 * fs)))
    return np.concatenate(parts)


def generate_zwicker_time_varying(output_dir: str) -> None:
    """The clause 6 loudness trace and the percentiles read off it."""
    print("Generating zwicker_time_varying...")
    from phonometry import loudness_zwicker

    fs = 48000
    x = _zwicker_burst_train(fs)
    res = loudness_zwicker(x, fs)
    trace = np.asarray(res.loudness_vs_time, dtype=float)
    times = np.asarray(res.time, dtype=float)
    n5, n10 = float(res.n5 or 0.0), float(res.n10 or 0.0)
    stationary = loudness_zwicker(x, fs, stationary=True)

    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(11.5, 5.4), gridspec_kw={"width_ratios": [1.7, 1.0]})

    ax.fill_between(times, trace, color=theme_fill(COLOR_PRIMARY, ax))
    ax.plot(times, trace, color=COLOR_PRIMARY, linewidth=1.6,
            label="Loudness-vs-time $N(t)$, 2 ms steps")
    for value, color, style, label in (
        (res.loudness, COLOR_SECONDARY, "-",
         rf"$N_{{\mathrm{{max}}}}$ = {res.loudness:.1f} sone (res.loudness)"),
        (n5, COLOR_TERTIARY, "--", f"$N_5$ = {n5:.1f} sone"),
        (n10, COLOR_FG, ":", f"$N_{{10}}$ = {n10:.1f} sone"),
    ):
        ax.axhline(value, color=color, linestyle=style, linewidth=1.5,
                   label=label)
    ax.axhline(float(trace.mean()), color=COLOR_FG, linestyle="-.",
               linewidth=1.2, alpha=0.7,
               label=f"arithmetic mean = {trace.mean():.1f} sone")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Loudness $N$ [sone]")
    ax.set_xlim(0.0, float(times[-1]))
    ax.set_ylim(0.0, float(trace.max()) * 1.32)
    ax.set_title("1 kHz bursts stepping 45 to 85 dB",
                 pad=10)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=9)

    # The same numbers read as an exceedance: what a percentile of a trace is.
    grid = np.linspace(0.0, float(trace.max()), 400)
    exceeded = 100.0 * np.array([(trace >= g).mean() for g in grid])
    ax2.plot(exceeded, grid, color=COLOR_PRIMARY, linewidth=2.0)
    for pct, value, color, label in (
        (5.0, n5, COLOR_TERTIARY, "$N_5$"),
        (10.0, n10, COLOR_FG, "$N_{10}$"),
    ):
        ax2.plot([pct], [value], "o", color=color, markersize=8,
                 markerfacecolor="white", markeredgewidth=1.8, zorder=6)
        ax2.annotate(f"{label} = {value:.1f} sone", xy=(pct, value),
                     xytext=(pct + 14, value + 1.4), fontsize=10, color=color,
                     arrowprops={"arrowstyle": "->", "lw": 0.9,
                                 "color": color})
    ax2.set_xlabel("Percentage of the analysis time exceeding $N$ [%]")
    ax2.set_ylabel("Loudness $N$ [sone]")
    ax2.set_xlim(0.0, 100.0)
    ax2.set_ylim(0.0, float(trace.max()) * 1.32)
    ax2.set_title(f"Exceedance over the {times[-1]:.1f} s record",
                  pad=10)
    ax2.grid(color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax2.set_axisbelow(True)
    ax2.text(
        0.97, 0.95,
        f"stationary=True on the same record: $N$ = {stationary.loudness:.1f} sone",
        transform=ax2.transAxes, ha="right", va="top", fontsize=9,
        color=COLOR_FG,
        bbox={"boxstyle": "round,pad=0.4", "facecolor": COLOR_PANEL,
              "edgecolor": COLOR_GRID})

    fig.suptitle("Time-Varying Loudness and the Percentiles (ISO 532-1 "
                 "clause 6)")
    plt.tight_layout()
    save_figure(output_dir, "zwicker_time_varying.svg")
    plt.close()


def generate_tone_audibility_uncertainty(output_dir: str) -> None:
    """The five Annex E decisive audibilities, their U, and the Kt ladder."""
    print("Generating tone_audibility_uncertainty...")
    from phonometry import environment, psychoacoustics

    # ISO/PAS 20065 Annex E, Table E.4: the decisive audibility of each of the
    # five measured 3 s spectra of the combustion engine, with the extended
    # uncertainty U of that spectrum.
    tones = (137.3, 430.7, 137.3, 433.4, 137.3)
    decisive = np.array([9.18, 6.04, 7.46, 2.67, 7.17])
    uncertainty = np.array([3.21, 2.95, 2.44, 2.52, 2.14])
    mean = psychoacoustics.mean_audibility(decisive)
    mean_u = psychoacoustics.mean_audibility_uncertainty(decisive, uncertainty)
    k_t = environment.tonal_adjustment_from_mean_audibility(mean)
    k_low = environment.tonal_adjustment_from_mean_audibility(mean - mean_u)

    index = np.arange(1, decisive.size + 1)
    _fig, ax = plt.subplots(figsize=(10, 6.4))

    # ISO 1996-2:2017 Table J.1 read as a ladder of horizontal bands.
    edges = [-2.0, 0.0, 2.0, 4.0, 6.0, 9.0, 12.0, 14.0]
    for step, (low, high) in enumerate(itertools.pairwise(edges)):
        ax.axhline(high, color=COLOR_GRID, linewidth=1.0, linestyle=":")
        ax.text(0.46, (low + high) / 2, rf"$K_\mathrm{{t}}$ = {step}",
                fontsize=9, color=COLOR_FG, va="center", ha="left")
    ax.axhline(0.0, color=COLOR_FG, linewidth=1.2)

    ax.fill_between([0.4, 5.8], mean - mean_u, mean + mean_u,
                    color=theme_fill(COLOR_SECONDARY, ax), zorder=1)
    ax.axhline(mean, color=COLOR_SECONDARY, linewidth=1.8,
               label=f"mean audibility {mean:.2f} ± {mean_u:.2f} dB (Formula 20)")
    ax.errorbar(index, decisive, yerr=uncertainty, fmt="o",
                color=COLOR_PRIMARY, markersize=8, markerfacecolor="white",
                markeredgewidth=1.8, capsize=6, linewidth=1.6, zorder=5,
                label=r"decisive audibility of each spectrum, $\pm U$ (clause 6)")
    for x_i, y_i, f_i in zip(index, decisive, tones, strict=True):
        ax.annotate(f"{f_i:g} Hz", xy=(x_i, y_i), xytext=(x_i + 0.09, y_i + 0.5),
                    fontsize=9, color=COLOR_FG)

    ax.annotate(
        rf"$K_\mathrm{{t}}$ = {k_t} dB, but the interval reaches into "
        rf"$K_\mathrm{{t}}$ = {k_low} dB",
        xy=(3.0, mean - mean_u), xytext=(1.55, 1.1), fontsize=10,
        color=COLOR_SECONDARY,
        arrowprops={"arrowstyle": "->", "lw": 1.0, "color": COLOR_SECONDARY})

    ax.set_xlabel("Measured 3 s spectrum (Annex E, run index $j$)")
    ax.set_ylabel(r"Audibility $\Delta L$ [dB]")
    ax.set_xlim(0.4, 5.8)
    ax.set_ylim(-2.0, 14.0)
    ax.set_xticks(index)
    ax.set_title("Decisive Audibility, Its Uncertainty and the Tonal "
                 "Adjustment", pad=12)
    ax.grid(axis="x", color=COLOR_GRID, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    save_figure(output_dir, "tone_audibility_uncertainty.svg")
    plt.close()


# DIN 45692 Table A.1 critical-band edges by centre frequency, and the
# Table A.2 hearing-test target sharpness of critical-band-wide noise of the
# same loudness as the reference sound (4 sone).
_DIN45692_TABLE_A2: tuple[tuple[float, float, float, float], ...] = (
    (250.0, 200.0, 300.0, 0.38), (350.0, 300.0, 400.0, 0.49),
    (450.0, 400.0, 510.0, 0.60), (570.0, 510.0, 630.0, 0.71),
    (700.0, 630.0, 770.0, 0.82), (840.0, 770.0, 920.0, 0.93),
    (1000.0, 920.0, 1080.0, 1.00), (1170.0, 1080.0, 1270.0, 1.13),
    (1370.0, 1270.0, 1480.0, 1.26), (1600.0, 1480.0, 1720.0, 1.35),
    (1850.0, 1720.0, 2000.0, 1.49), (2150.0, 2000.0, 2320.0, 1.64),
    (2500.0, 2320.0, 2700.0, 1.78), (2900.0, 2700.0, 3150.0, 2.06),
    (3400.0, 3150.0, 3700.0, 2.40), (4000.0, 3700.0, 4400.0, 2.82),
)


def _critical_band_noise(f_low: float, f_high: float,
                         level_db: float) -> np.ndarray:
    """One critical band of noise at ``level_db``, in pascals at 48 kHz."""
    from scipy import signal as sp_signal

    rng = np.random.default_rng(7)
    white = rng.standard_normal(48000 * 2)
    sos = sp_signal.butter(8, [f_low, f_high], btype="band", fs=48000,
                           output="sos")
    band = sp_signal.sosfilt(sos, white)
    band = band / np.sqrt(np.mean(band**2))
    return np.asarray(band * 2e-5 * 10 ** (level_db / 20))


def _level_for_four_sone(f_low: float, f_high: float) -> float:
    """Band level that puts the band at the clause 6 loudness of 4 sone."""
    from phonometry import loudness_zwicker

    low, high = 30.0, 95.0
    for _ in range(14):
        mid = (low + high) / 2
        loudness = loudness_zwicker(_critical_band_noise(f_low, f_high, mid),
                                    48000, stationary=True).loudness
        low, high = (mid, high) if loudness < 4.0 else (low, mid)
    return (low + high) / 2


@cache
def _din45692_sweep() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Computed sharpness against the Table A.2 targets, at 4 sone each."""
    from phonometry import sharpness_din

    centres, targets, computed = [], [], []
    for centre, f_low, f_high, target in _DIN45692_TABLE_A2:
        level = _level_for_four_sone(f_low, f_high)
        computed.append(float(sharpness_din(
            _critical_band_noise(f_low, f_high, level), 48000)))
        centres.append(centre)
        targets.append(target)
    return np.array(centres), np.array(targets), np.array(computed)


def _sottek_am_reference(f_mod: float, seconds: float) -> np.ndarray:
    """The ECMA-418-2 reference sound: 1 kHz, 100 % AM, 60 dB overall."""
    fs = 48000
    t = np.arange(int(seconds * fs)) / fs
    x = (1.0 + np.cos(2 * np.pi * f_mod * t)) * np.sin(2 * np.pi * 1000.0 * t)
    return np.asarray(x * 2e-5 * 10 ** (60 / 20) / np.sqrt(np.mean(x**2)))


def _sottek_specific_panels(
    output_dir: str, filename: str, *, bark: np.ndarray,
    specific: np.ndarray, time: np.ndarray, trace: np.ndarray,
    single: float, unit: str, symbol: str, title: str, stimulus: str,
) -> None:
    """The two views every ECMA-418-2 running metric carries.

    Left, the average specific pattern over the Bark_HMS axis; right, the
    time-dependent value with the 90th percentile that *is* the single
    number drawn across it. Shared by the roughness and fluctuation-strength
    figures, which differ only in their labels and stimulus.
    """
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.0, 4.8))

    ax.fill_between(bark, specific, color=theme_fill(COLOR_PRIMARY, ax))
    ax.plot(bark, specific, color=COLOR_PRIMARY, linewidth=1.8)
    peak = int(np.argmax(specific))
    ax.annotate(f"the carrier's band, {bark[peak]:.1f} {_hms('Bark_HMS')}",
                xy=(bark[peak], specific[peak]),
                xytext=(bark[peak] + 2.0, specific[peak] * 0.92), fontsize=10,
                color=COLOR_FG,
                arrowprops={"arrowstyle": "->", "lw": 1.0})
    ax.set_xlabel(f"Critical-band rate $z$ [{_hms('Bark_HMS')}]")
    ax.set_ylabel(rf"Specific {title.lower()} ${symbol}^{{\prime}}(z)$ "
                  f"[{_hms(unit)}/{_hms('Bark_HMS')}]")
    ax.set_xlim(0, 24)
    ax.set_xticks([0, 4, 8, 12, 16, 20, 24])
    ax.set_title(f"Average specific {title.lower()}",
                 pad=10)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    ax2.plot(time, trace, color=COLOR_PRIMARY, linewidth=1.6,
             label=f"${symbol}(l_{{50}})$, the running value")
    ax2.axhline(single, color=COLOR_SECONDARY, linewidth=1.8, linestyle="--",
                label=f"90th percentile = {single:.4f} {_hms(unit)}")
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel(f"{title} ${symbol}$ [{_hms(unit)}]")
    ax2.set_xlim(float(time[0]), float(time[-1]))
    ax2.set_ylim(0.0, max(float(trace.max()), single) * 1.28)
    ax2.set_title("The single value is a percentile of this trace",
                  pad=10)
    ax2.grid(color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax2.set_axisbelow(True)
    ax2.legend(loc="lower right", fontsize=9)

    fig.suptitle(f"{title} of the ECMA-418-2 Reference Sound ({stimulus})",
                 )
    plt.tight_layout()
    save_figure(output_dir, filename)
    plt.close()


def generate_sottek_specific_roughness(output_dir: str) -> None:
    """R'(z) and R(l50) of the clause 7 reference sound."""
    print("Generating sottek_specific_roughness...")
    from phonometry import roughness_ecma

    result = roughness_ecma(_sottek_am_reference(70.0, 2.0), 48000,
                            field="free")
    _sottek_specific_panels(
        output_dir, "sottek_specific_roughness.svg",
        bark=np.asarray(result.bark),
        specific=np.asarray(result.specific_roughness),
        time=np.asarray(result.time),
        trace=np.asarray(result.roughness_vs_time),
        single=float(result.roughness), unit="asper", symbol="R",
        title="Roughness", stimulus="1 kHz, 100 % AM at 70 Hz, 60 dB")


def generate_sottek_specific_fluctuation(output_dir: str) -> None:
    """F'(z) and F(l50) of the clause 9 reference sound."""
    print("Generating sottek_specific_fluctuation...")
    from phonometry import fluctuation_strength_ecma

    result = fluctuation_strength_ecma(_sottek_am_reference(4.0, 8.0), 48000,
                                       field="free")
    _sottek_specific_panels(
        output_dir, "sottek_specific_fluctuation.svg",
        bark=np.asarray(result.bark),
        specific=np.asarray(result.specific_fluctuation_strength),
        time=np.asarray(result.time),
        trace=np.asarray(result.fluctuation_strength_vs_time),
        single=float(result.fluctuation_strength), unit="vacil_HMS",
        symbol="F", title="Fluctuation strength",
        stimulus="1 kHz, 100 % AM at 4 Hz, 60 dB")


def generate_sharpness_pair_and_targets(output_dir: str) -> None:
    """Sharpness as a position on the Bark axis, and the Table A.2 check."""
    print("Generating sharpness_pair_and_targets...")
    from phonometry import loudness_zwicker, sharpness_din

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.0, 5.6))

    # Left: two critical bands of noise of the same loudness, one low and one
    # high on the Bark axis.
    for (centre, f_low, f_high), color in (
            ((250.0, 200.0, 300.0), COLOR_PRIMARY),
            ((4000.0, 3700.0, 4400.0), COLOR_SECONDARY)):
        level = _level_for_four_sone(f_low, f_high)
        band = _critical_band_noise(f_low, f_high, level)
        result = loudness_zwicker(band, 48000, stationary=True)
        acum = float(sharpness_din(band, 48000))
        z = np.arange(1, result.specific.size + 1) * 0.1
        centroid = float(np.sum(result.specific * z) / np.sum(result.specific))
        ax.fill_between(z, result.specific, color=theme_fill(color, ax))
        ax.plot(z, result.specific, color=color, linewidth=1.8,
                label=(f"{centre:g} Hz critical band — "
                       f"$N$ = {result.loudness:.1f} sone, $S$ = {acum:.2f} acum"))
        ax.axvline(centroid, color=color, linestyle="--", linewidth=1.2)
        ax.annotate(rf"$\langle z\rangle$ = {centroid:.1f} Bark", xy=(centroid, 0.06),
                    xytext=(centroid + 0.7, 0.06), fontsize=9, color=color)
    ax.set_xlabel("Critical-band rate $z$ [Bark]")
    ax.set_ylabel(r"Specific loudness $N^{\prime}$ [sone/Bark]")
    ax.set_xlim(0, 24)
    ax.set_xticks([0, 4, 8, 12, 16, 20, 24])
    ax.set_title("Equally loud, seven times as sharp",
                 pad=10)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=9)

    # Right: the Table A.2 verification sweep with its tolerance band.
    centres, targets, computed = _din45692_sweep()
    tolerance = np.maximum(0.05 * targets, 0.05)
    ax2.fill_between(centres, targets - tolerance, targets + tolerance,
                     color=theme_fill(COLOR_TERTIARY, ax2),
                     label="permitted deviation: 5 % or 0.05 acum")
    ax2.semilogx(centres, computed, color=COLOR_PRIMARY, linewidth=2.0,
                 label="sharpness_din(), each band set to 4 sone")
    ax2.semilogx(centres, targets, "o", color=COLOR_SECONDARY, markersize=7,
                 markerfacecolor="white", markeredgewidth=1.6,
                 label="Table A.2 hearing-test targets")
    ax2.set_xlabel(LABEL_FREQ_HZ)
    ax2.set_ylabel("Sharpness $S$ [acum]")
    ax2.set_ylim(0.0, 3.4)
    ax2.set_title("DIN 45692 Table A.2, 250 Hz to 4 kHz",
                  pad=10)
    format_frequency_axis(ax2)
    ax2.grid(which="both", color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax2.set_axisbelow(True)
    ax2.legend(loc="upper left", fontsize=9)

    fig.suptitle("Sharpness: Where the Loudness Sits, Not How Much There Is",
                 )
    plt.tight_layout()
    save_figure(output_dir, "sharpness_pair_and_targets.svg")
    plt.close()


@cache
def _tnr_pr_neighbour_sweep() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """TNR and PR of a 1 kHz tone as a tone grows in the next band up."""
    from phonometry import prominence_ratio, tone_to_noise_ratio

    fs = 48000
    rng = np.random.default_rng(7)
    t = np.arange(8 * fs) / fs
    # A 1 kHz tone in broadband noise, comfortably prominent on its own, plus
    # a second tone at 1160 Hz: outside the 162 Hz critical band centred on
    # 1 kHz, and therefore inside the upper contiguous band the PR uses as
    # its noise estimate.
    primary = np.sqrt(2) * 0.012 * np.sin(2 * np.pi * 1000.0 * t)
    noise = 0.05 * rng.standard_normal(t.size)
    relative = np.arange(-24.0, 12.1, 3.0)
    tnr, pr = [], []
    for rel in relative:
        amplitude = np.sqrt(2) * 0.012 * 10 ** (rel / 20)
        x = primary + amplitude * np.sin(2 * np.pi * 1160.0 * t) + noise
        tnr.append(tone_to_noise_ratio(x, fs, tone_freq=1000.0).ratio_db)
        pr.append(prominence_ratio(x, fs, tone_freq=1000.0).ratio_db)
    return relative, np.array(tnr), np.array(pr)


def generate_tnr_pr_comparison(output_dir: str) -> None:
    """The two criteria side by side, and a case where they disagree."""
    print("Generating tnr_pr_comparison...")
    from phonometry import prominence_ratio, tone_to_noise_ratio

    # The 250 Hz fan tone of the assessment figure, put to both methods.
    fs = 48000
    rng = np.random.default_rng(4)
    t = np.arange(10 * fs) / fs
    fan = (np.sqrt(2) * 0.011 * np.sin(2 * np.pi * 250.0 * t)
           + 0.03 * rng.standard_normal(t.size))
    tnr_fan = tone_to_noise_ratio(fan, fs)
    pr_fan = prominence_ratio(fan, fs, tone_freq=250.0)

    freqs = np.logspace(np.log10(89.1), np.log10(11200.0), 400)
    tnr_criterion = np.where(freqs < 1000.0,
                             8.0 + 8.33 * np.log10(1000.0 / freqs), 8.0)
    pr_criterion = np.where(freqs < 1000.0,
                            9.0 + 10.0 * np.log10(1000.0 / freqs), 9.0)

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.0, 5.6))

    ax.semilogx(freqs, tnr_criterion, color=COLOR_PRIMARY, linewidth=2.2,
                label="TNR criterion (clause 11.5)")
    ax.semilogx(freqs, pr_criterion, color=COLOR_SECONDARY, linewidth=2.2,
                linestyle="--", label="PR criterion (clause 12.5)")
    for value, color, name in ((tnr_fan.ratio_db, COLOR_PRIMARY, "TNR"),
                               (pr_fan.ratio_db, COLOR_SECONDARY, "PR")):
        ax.plot([250.0], [value], "o", color=color, markersize=9,
                markerfacecolor="white", markeredgewidth=1.8, zorder=6)
        ax.annotate(f"{name} = {value:.1f} dB", xy=(250.0, value),
                    xytext=(430.0, 19.6 if name == "TNR" else 3.4),
                    fontsize=10, color=color,
                    arrowprops={"arrowstyle": "->", "lw": 1.0, "color": color})
    ax.set_xlabel(LABEL_FREQ_HZ)
    ax.set_ylabel("Ratio [dB]")
    ax.set_ylim(0.0, 24.0)
    ax.set_title("One 250 Hz fan tone, two criteria",
                 pad=10)
    format_frequency_axis(ax)
    ax.grid(which="both", color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=9)

    relative, tnr, pr = _tnr_pr_neighbour_sweep()
    ax2.plot(relative, tnr, "o-", color=COLOR_PRIMARY, linewidth=2.0,
             markersize=5, label="TNR of the 1 kHz tone")
    ax2.plot(relative, pr, "s--", color=COLOR_SECONDARY, linewidth=2.0,
             markersize=5, label="PR of the 1 kHz tone")
    ax2.axhline(8.0, color=COLOR_PRIMARY, linewidth=1.2, linestyle=":")
    ax2.axhline(9.0, color=COLOR_SECONDARY, linewidth=1.2, linestyle=":")
    ax2.text(1.0, 6.4, "TNR criterion 8 dB", fontsize=9, color=COLOR_PRIMARY)
    ax2.text(relative[0] + 0.4, 10.4, "PR criterion 9 dB", fontsize=9,
             color=COLOR_SECONDARY)
    ax2.set_ylim(-11.0, 12.5)
    ax2.set_xlabel("Level of a second tone at 1160 Hz, relative to the "
                   "1 kHz tone [dB]")
    ax2.set_ylabel("Ratio [dB]")
    ax2.set_title("A tone in the next critical band up",
                  pad=10)
    ax2.grid(color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax2.set_axisbelow(True)
    ax2.legend(loc="lower left", fontsize=9)

    fig.suptitle("Tone-to-Noise Ratio and Prominence Ratio Compared "
                 "(ECMA-418-1)")
    plt.tight_layout()
    save_figure(output_dir, "tnr_pr_comparison.svg")
    plt.close()


def generate_two_tone_separation(output_dir: str) -> None:
    """Formula 19: when two tones in one critical band are rated apart."""
    print("Generating two_tone_separation...")
    from phonometry import psychoacoustics

    freqs = np.logspace(np.log10(88.0), np.log10(1000.0), 400)
    f_d = np.array([psychoacoustics.two_tone_separation_frequency(f)
                    for f in freqs])

    _fig, ax = plt.subplots(figsize=(10, 6))
    ax.semilogx(freqs, f_d, color=COLOR_PRIMARY, linewidth=2.2,
                label=r"Threshold $f_\mathrm{D}$ (Formula 19)")
    ax.fill_between(freqs, f_d, 100.0,
                    color=theme_fill(COLOR_PRIMARY, ax), zorder=0)
    ax.plot([212.0], [21.0], "o", color=COLOR_SECONDARY, markersize=9,
            markerfacecolor="white", markeredgewidth=1.8, zorder=5)
    ax.annotate("minimum 21 Hz at 212 Hz", xy=(212.0, 21.0),
                xytext=(240.0, 11.0), fontsize=10, color=COLOR_SECONDARY,
                arrowprops={"arrowstyle": "->", "lw": 1.0,
                            "color": COLOR_SECONDARY})

    # The Annex E pair in the 137.3 Hz critical band: 118.4 and 137.3 Hz,
    # 18.9 Hz apart, evaluated at the more audible of the two.
    ax.plot([137.3], [18.9], "s", color=COLOR_TERTIARY, markersize=9,
            zorder=5)
    ax.annotate("Annex E pair: 118.4 and 137.3 Hz,\n"
                r"18.9 Hz apart — below $f_\mathrm{D}$, so combined",
                xy=(137.3, 18.9), xytext=(95.0, 44.0), fontsize=10,
                color=COLOR_TERTIARY,
                arrowprops={"arrowstyle": "->", "lw": 1.0,
                            "color": COLOR_TERTIARY})
    ax.text(600.0, 62.0, "rated separately", fontsize=11, color=COLOR_FG,
            ha="center")
    ax.text(600.0, 12.0, "energy-summed into one FG entry", fontsize=11,
            color=COLOR_FG, ha="center")

    ax.set_xlabel(r"Frequency of the more audible tone $f_\mathrm{T}$ [Hz]")
    ax.set_ylabel(r"Frequency separation $|f_{\mathrm{T}1} - f_{\mathrm{T}2}|$ [Hz]")
    ax.set_xlim(88.0, 1000.0)
    ax.set_ylim(0.0, 100.0)
    ax.set_title("Two Tones in One Critical Band: Separate or Combined "
                 "(ISO/PAS 20065 Formula 19)", pad=12)
    format_frequency_axis(ax)
    ax.grid(which="both", color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    save_figure(output_dir, "two_tone_separation.svg")
    plt.close()


def generate_sti_curve(output_dir: str) -> None:
    """STI vs reverberation time: pipeline points vs the analytic MTF."""
    print("Generating sti_vs_t60.png...")
    from phonometry import sti_from_impulse_response

    fs = 48000
    t60_points = [0.3, 0.5, 0.8, 1.2, 2.0, 3.0, 5.0]

    # The 14 full-STI modulation frequencies and the male alpha/beta
    # factors of IEC 60268-16 Ed.5 Table A.1 (phonometry.speech.sti keeps them
    # private, so they are restated here for the analytic reference).
    mod_freqs = np.array([0.63, 0.80, 1.00, 1.25, 1.60, 2.00, 2.50,
                          3.15, 4.00, 5.00, 6.30, 8.00, 10.0, 12.5])
    alpha = np.array([0.085, 0.127, 0.230, 0.233, 0.309, 0.224, 0.173])
    beta = np.array([0.085, 0.078, 0.065, 0.011, 0.047, 0.095])

    def analytic_sti(t60: float) -> float:
        # Schroeder MTF of an exponential decay: m(F) = 1/sqrt(1+(2*pi*F*T/13.8)^2)
        m = 1.0 / np.sqrt(1.0 + (2 * np.pi * mod_freqs * t60 / 13.8) ** 2)
        snr_eff = np.clip(10 * np.log10(m / (1 - m)), -15.0, 15.0)
        mti = np.full(7, ((snr_eff + 15.0) / 30.0).mean())
        return float(np.dot(alpha, mti) - np.dot(beta, np.sqrt(mti[:-1] * mti[1:])))

    rng = np.random.default_rng(2026)
    measured = []
    for t60 in t60_points:
        t = np.arange(int(2 * t60 * fs)) / fs
        ir = rng.standard_normal(t.size) * np.exp(-6.9077 * t / t60)
        measured.append(sti_from_impulse_response(ir, fs).sti)

    t_dense = np.logspace(np.log10(0.25), np.log10(6.0), 200)
    sti_dense = [analytic_sti(float(t)) for t in t_dense]

    _, ax = plt.subplots(figsize=(10, 6))
    # Annex F qualification bands (informative): edges 0.36 .. 0.76.
    edges = [0.36, 0.40, 0.44, 0.48, 0.52, 0.56, 0.60, 0.64, 0.68, 0.72, 0.76]
    letters = ["U", "J", "I", "H", "G", "F", "E", "D", "C", "B", "A", "A+"]
    y_min, y_max = 0.15, 0.95
    bounds = [y_min] + edges + [y_max]
    cmap = plt.get_cmap("RdYlGn")
    for i, letter in enumerate(letters):
        lo, hi = bounds[i], bounds[i + 1]
        ax.axhspan(lo, hi, color=theme_fill(cmap(i / (len(letters) - 1)), ax),
                   lw=0, zorder=0)
        ax.text(0.985, (lo + hi) / 2, letter, transform=ax.get_yaxis_transform(),
                ha="right", va="center", fontsize=8, color=COLOR_FG, alpha=0.7)
    ax.text(0.92, 0.985, "Annex F rating", transform=ax.transAxes,
            ha="right", va="top", fontsize=8, color=COLOR_FG, alpha=0.7)

    ax.plot(t_dense, sti_dense, color=COLOR_PRIMARY, linestyle="--",
            linewidth=1.5, label="Analytic Schroeder MTF (closed form)")
    ax.plot(t60_points, measured, "o", color=COLOR_SECONDARY, markersize=7,
            markerfacecolor="white", markeredgewidth=1.6,
            label="Measured (sti_from_impulse_response)")

    ax.set_xscale("log")
    ax.set_title("STI vs Reverberation Time (IEC 60268-16)",
                 pad=12)
    ax.set_xlabel("Reverberation time $T_{60}$ [s]")
    ax.set_ylabel("STI")
    ax.set_xlim(0.25, 6.0)
    ax.set_ylim(y_min, y_max)
    from matplotlib.ticker import NullFormatter
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xticks(t60_points)
    ax.set_xticklabels(["0.3", "0.5", "0.8", "1.2", "2", "3", "5"])
    ax.legend(loc="lower left", fontsize=9)
    save_figure(output_dir, "sti_vs_t60.png")
    plt.close()


def generate_sharpness_weighting(output_dir: str) -> None:
    """DIN 45692 sharpness weighting g(z): DIN vs Aures vs von Bismarck."""
    print("Generating sharpness_weighting.png...")
    from phonometry.psychoacoustics.quality.sharpness import (
        _Z,
        _g_aures,
        _g_bismarck,
        _g_din,
    )

    z = _Z                       # 0.1 .. 24 Bark, 0.1-Bark steps
    total_n = 4.0                # reference loudness for the Aures variant (sone)
    g_din = _g_din(z)
    g_bismarck = _g_bismarck(z)
    g_aures = _g_aures(z, total_n)

    _, ax = plt.subplots(figsize=(10, 6.5))
    ax.semilogy(z, g_din, color=COLOR_PRIMARY, linewidth=2.2,
                label="DIN 45692 $g(z)$")
    ax.semilogy(z, g_bismarck, color=COLOR_TERTIARY, linewidth=1.7,
                linestyle="--", label="von Bismarck (Annex B)")
    ax.semilogy(z, g_aures, color=COLOR_SECONDARY, linewidth=1.7,
                linestyle="-.", label=f"Aures (Annex B, $N$ = {total_n:.0f} sone)")

    # DIN weighting is flat (g = 1) up to 15.8 Bark, von Bismarck up to 15.
    # The three guides sit behind the curves, held back by shade: held back by
    # opacity, the g = 1 line and the DIN knee were a shade of the dark page.
    ax.axhline(1.0, color=theme_line(COLOR_FG, ax, quiet=0.15), linestyle="-",
               linewidth=1)
    ax.axvline(15.8, color=theme_line(COLOR_PRIMARY, ax, quiet=0.5),
               linestyle=":", linewidth=1)
    ax.axvline(15.0, color=theme_line(COLOR_TERTIARY, ax, quiet=0.5),
               linestyle=":", linewidth=1)
    ax.annotate("DIN knee\n15.8 Bark", xy=(15.8, 1.0), xytext=(10.2, 2.3),
                fontsize=9, color=COLOR_PRIMARY, ha="center",
                arrowprops={"arrowstyle": "->", "lw": 0.9, "color": COLOR_PRIMARY})
    ax.annotate("Bismarck knee\n15 Bark", xy=(15.0, 1.0), xytext=(7.0, 0.5),
                fontsize=9, color=COLOR_TERTIARY, ha="center",
                arrowprops={"arrowstyle": "->", "lw": 0.9, "color": COLOR_TERTIARY})

    ax.set_title("Sharpness Weighting $g(z)$ (DIN 45692)",
                 pad=12)
    ax.set_xlabel("Critical-band rate $z$ [Bark]")
    ax.set_ylabel("Weighting $g(z)$")
    ax.set_xlim(0, 24)
    ax.set_ylim(0.4, 30)
    ax.set_xticks([0, 4, 8, 12, 16, 20, 24])
    ax.grid(which="both", color=COLOR_GRID, linestyle="-", alpha=0.4)
    ax.legend(loc="upper right", fontsize=9)
    save_figure(output_dir, "sharpness_weighting.png")
    plt.close()


def generate_erb_bandwidth(output_dir: str) -> None:
    """Auditory-filter bandwidth against centre frequency, with the Cam scale.

    One concept: how wide the ear's analysis filter is at each frequency
    (Glasberg and Moore, 1990), against the constant-percentage one-third
    octave for scale, with the Cam axis that counts those widths.
    """
    print("Generating erb_bandwidth...")
    from phonometry import cam_from_frequency, erb_bandwidth

    f = np.geomspace(50.0, 16000.0, 400)
    erb = np.asarray(erb_bandwidth(f))
    third_octave = f * (2.0 ** (1.0 / 6.0) - 2.0 ** (-1.0 / 6.0))

    _fig, ax = plt.subplots(figsize=(10, 6))
    ax.loglog(f, erb, color=COLOR_PRIMARY, linewidth=2.2,
              label=r"$\mathrm{ERB}_\mathrm{N}$ (Glasberg & Moore, 1990)")
    ax.loglog(f, third_octave, color=COLOR_SECONDARY, linewidth=1.6,
              linestyle="--", label="One-third octave (23 % of $f$)")
    ax.set_title("Auditory-Filter Bandwidth and the Cam Scale "
                 "(Glasberg & Moore, 1990)", pad=12)
    ax.set_xlabel("Centre frequency [Hz]")
    ax.set_ylabel(
        r"Equivalent rectangular bandwidth $\mathrm{ERB}_\mathrm{N}$ [Hz]")
    ax.grid(which="both", color=COLOR_GRID, linestyle="-", alpha=0.4)
    format_frequency_axis(ax, 50.0, 16000.0)
    from matplotlib.ticker import FixedFormatter, FixedLocator
    y_ticks = [10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0]
    ax.yaxis.set_major_locator(FixedLocator(y_ticks))
    ax.yaxis.set_major_formatter(FixedFormatter([f"{v:g}" for v in y_ticks]))
    ax.legend(loc="upper left", fontsize=9)

    cam_1k = float(np.asarray(cam_from_frequency(1000.0))[()])
    erb_1k = float(np.asarray(erb_bandwidth(1000.0))[()])
    ax.plot([1000.0], [erb_1k], "o", color=COLOR_PRIMARY, markersize=8,
            markerfacecolor="white", markeredgewidth=1.6, zorder=6)
    ax.annotate(f"1 kHz = {cam_1k:.2f} Cam", xy=(1000.0, erb_1k),
                xytext=(1400.0, 0.45 * erb_1k), fontsize=10,
                arrowprops={"arrowstyle": "->", "lw": 1.0})

    # Top axis: the Cam scale, which counts ERB_N widths along frequency.
    from matplotlib.ticker import NullFormatter, NullLocator

    from phonometry import frequency_from_cam

    ax2 = ax.twiny()
    ax2.set_xscale("log")
    ax2.set_xlim(ax.get_xlim())
    cam_ticks = np.array([5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0])
    ax2.set_xticks(np.asarray(frequency_from_cam(cam_ticks)))
    ax2.set_xticklabels([f"{c:.0f}" for c in cam_ticks])
    ax2.xaxis.set_minor_locator(NullLocator())
    ax2.xaxis.set_minor_formatter(NullFormatter())
    ax2.set_xlabel(r"$\mathrm{ERB}_\mathrm{N}$ number [Cam]",
                   color=COLOR_TERTIARY)
    save_figure(output_dir, "erb_bandwidth.svg")
    plt.close()


# ---------------------------------------------------------------------------
# Advanced psychoacoustics (plan-17 block A): the ECMA-418-2 Sottek model
# (loudness, tonality, roughness) and the Moore-Glasberg ISO 532-2/-3 models.
# The heavy computations (ECMA loudness ~5 s/call, tonality ~8 s/call) are
# cached so they run once and are reused across the four themed/language
# passes rather than four times over.
# ---------------------------------------------------------------------------
_P_REF = 2e-5  # reference sound pressure [Pa]
_FS_PSY = 48000  # ECMA-418-2 / ISO 532 operate at 48 kHz


def _pure_tone(freq: float, spl_db: float, dur: float,
               fs: int = _FS_PSY) -> np.ndarray:
    """Calibrated sinusoid: sound pressure in pascals at *spl_db* dB SPL."""
    t = np.arange(round(dur * fs)) / fs
    amp = _P_REF * 10.0 ** (spl_db / 20.0) * np.sqrt(2.0)
    return np.asarray(amp * np.sin(2.0 * np.pi * freq * t))


@cache
def _loudness_models_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Total loudness (sone) vs level for the three loudness models."""
    from phonometry import (
        loudness_ecma,
        loudness_moore_glasberg_from_spectrum,
        loudness_zwicker,
    )

    levels = np.arange(20.0, 81.0, 10.0)  # 20..80 dB SPL
    zw, mg, ec = [], [], []
    for spl in levels:
        x = _pure_tone(1000.0, float(spl), 1.0)
        zw.append(loudness_zwicker(x, _FS_PSY, stationary=True).loudness)
        mg.append(
            loudness_moore_glasberg_from_spectrum([(1000.0, float(spl))]).loudness
        )
        ec.append(loudness_ecma(x, _FS_PSY).loudness)
    return levels, np.array(zw), np.array(mg), np.array(ec)


def generate_loudness_models_comparison(output_dir: str) -> None:
    """Zwicker vs Moore-Glasberg vs Sottek loudness for a 1 kHz tone."""
    print("Generating loudness_models_comparison.png...")
    levels, zw, mg, ec = _loudness_models_data()

    _, ax = plt.subplots(figsize=(10, 6))
    ax.plot(levels, zw, "o-", color=COLOR_PRIMARY, linewidth=2.0, markersize=6,
            label=f"Zwicker (ISO 532-1), $N$ = {zw[2]:.1f} sone")
    ax.plot(levels, mg, "s--", color=COLOR_TERTIARY, linewidth=1.8, markersize=6,
            label=f"Moore-Glasberg (ISO 532-2), $N$ = {mg[2]:.1f} sone")
    ax.plot(levels, ec, "^-.", color=COLOR_SECONDARY, linewidth=1.8, markersize=6,
            label=f"Sottek (ECMA-418-2), $N$ = {ec[2]:.1f} sone")

    # The three models are anchored to 1 sone at 1 kHz / 40 dB SPL.
    ax.axhline(1.0, color=COLOR_FG, linestyle=":", alpha=0.35, linewidth=1)
    ax.plot(40.0, 1.0, "o", color=COLOR_FG, markersize=9,
            markerfacecolor="none", markeredgewidth=1.6, zorder=5)
    ax.annotate("Anchor: 1 kHz / 40 dB = 1 sone",
                xy=(40.0, 1.0), xytext=(21.5, 6.5), fontsize=10,
                color=COLOR_FG,
                arrowprops={"arrowstyle": "->", "lw": 1.0, "color": COLOR_FG})
    ax.annotate("Models diverge at high levels",
                xy=(80.0, float(zw[-1])), xytext=(52.0, 13.5), fontsize=9,
                color=COLOR_FG, ha="center",
                arrowprops={"arrowstyle": "->", "lw": 0.9, "color": COLOR_FG})

    ax.set_title("Loudness Models Compared (1 kHz tone)",
                 pad=12)
    ax.set_xlabel("Sound pressure level [dB SPL]")
    ax.set_ylabel("Total loudness $N$ [sone]")
    ax.set_xlim(18, 82)
    ax.set_ylim(0, float(zw[-1]) * 1.08)
    ax.set_xticks([20, 30, 40, 50, 60, 70, 80])
    ax.legend(loc="upper left", fontsize=9)
    save_figure(output_dir, "loudness_models_comparison.png")
    plt.close()


@cache
def _sottek_specific_data() -> tuple[np.ndarray, np.ndarray, float]:
    """ECMA-418-2 specific loudness N'(z) of a 1 kHz / 60 dB tone."""
    from phonometry.psychoacoustics import loudness_ecma

    el = loudness_ecma(_pure_tone(1000.0, 60.0, 1.0), _FS_PSY)
    return el.bark.copy(), el.specific_loudness.copy(), float(el.loudness)


def generate_sottek_specific_loudness(output_dir: str) -> None:
    """ECMA-418-2 (Sottek) specific loudness N'(z) over the Bark-rate scale."""
    print("Generating sottek_specific_loudness.png...")
    bark, spec, total = _sottek_specific_data()

    _, ax = plt.subplots(figsize=(10, 6))
    ax.fill_between(bark, spec, color=COLOR_PRIMARY, alpha=0.30)
    ax.plot(bark, spec, color=COLOR_PRIMARY, linewidth=1.8,
            label=f"1 kHz tone, 60 dB ($N$ = {total:.1f} {_hms('sone_HMS')})")

    peak_i = int(np.argmax(spec))
    ax.annotate("Peak specific loudness",
                xy=(float(bark[peak_i]), float(spec[peak_i])),
                xytext=(float(bark[peak_i]) + 4.5, float(spec[peak_i]) * 0.92),
                fontsize=10, color=COLOR_FG,
                arrowprops={"arrowstyle": "->", "lw": 0.9, "color": COLOR_FG})

    ax.set_title("Sottek Specific Loudness (ECMA-418-2)",
                 pad=12)
    ax.set_xlabel("Critical-band rate $z$ [Bark]")
    ax.set_ylabel(
        rf"Specific loudness $N^{{\prime}}$ [{_hms('sone_HMS')}/Bark]")
    ax.set_xlim(0, float(bark[-1]))
    ax.set_ylim(0, float(spec.max()) * 1.25)
    ax.set_xticks([0, 4, 8, 12, 16, 20, 24])
    ax.legend(loc="upper right", fontsize=9)
    save_figure(output_dir, "sottek_specific_loudness.png")
    plt.close()


@cache
def _tonality_data() -> tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray, float]:
    """ECMA-418-2 tonality T(t) for a 1 kHz tone-in-noise vs pure noise."""
    from phonometry.psychoacoustics import tonality_ecma

    rng = np.random.default_rng(2026)
    dur = 2.0
    t = np.arange(int(dur * _FS_PSY)) / _FS_PSY
    noise = rng.standard_normal(t.size)
    noise = noise / np.sqrt(np.mean(noise ** 2)) * _P_REF * 10.0 ** (50.0 / 20.0)
    tone = _P_REF * 10.0 ** (50.0 / 20.0) * np.sqrt(2.0) * np.sin(2.0 * np.pi * 1000.0 * t)

    tin = tonality_ecma(tone + noise, _FS_PSY)
    pn = tonality_ecma(noise, _FS_PSY)
    return (tin.time.copy(), tin.tonality_vs_time.copy(), float(tin.tonality),
            pn.time.copy(), pn.tonality_vs_time.copy(), float(pn.tonality))


@cache
def _roughness_sweep_data() -> tuple[np.ndarray, np.ndarray]:
    """ECMA-418-2 roughness R vs AM frequency, 1 kHz carrier, 100 % AM, 60 dB."""
    from phonometry.psychoacoustics import roughness_ecma

    dur = 1.0
    t = np.arange(int(dur * _FS_PSY)) / _FS_PSY
    fmods = np.array([20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0,
                      100.0, 120.0, 150.0, 180.0, 200.0])
    r = []
    for fm in fmods:
        am = (1.0 + 1.0 * np.sin(2.0 * np.pi * fm * t)) * np.sin(2.0 * np.pi * 1000.0 * t)
        am = am / np.sqrt(np.mean(am ** 2)) * _P_REF * 10.0 ** (60.0 / 20.0)
        r.append(roughness_ecma(am, _FS_PSY).roughness)
    return fmods, np.array(r)


def generate_tonality_roughness_demo(output_dir: str) -> None:
    """Two-panel ECMA-418-2 sound-quality demo: tonality T(t) and roughness."""
    print("Generating tonality_roughness_demo.png...")
    (t_tin, tv_tin, t_single, _t_pn, tv_pn, pn_single) = _tonality_data()
    fmods, r = _roughness_sweep_data()

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(10, 8.5))

    # -- Top: time-dependent tonality, tone-in-noise vs pure noise -----------
    ax0.plot(t_tin, tv_tin, color=COLOR_PRIMARY, linewidth=1.8,
             label=f"Tone in noise ($T$ = {t_single:.2f} {_hms('tu_HMS')})")
    ax0.plot(_t_pn, tv_pn, color=COLOR_SECONDARY, linewidth=1.8,
             label=f"Pure noise ($T$ = {pn_single:.2f} {_hms('tu_HMS')})")
    ax0.set_title("ECMA-418-2 Tonality $T(t)$", pad=10)
    ax0.set_xlabel("Time [s]")
    ax0.set_ylabel(f"Tonality $T$ [{_hms('tu_HMS')}]")
    ax0.set_xlim(0, float(t_tin[-1]))
    ax0.set_ylim(0, max(1.0, float(tv_tin.max()) * 1.30))
    ax0.legend(loc="upper right", fontsize=9)

    # -- Bottom: roughness vs modulation frequency (peak near 70 Hz) ---------
    ax1.plot(fmods, r, "o-", color=COLOR_TERTIARY, linewidth=2.0, markersize=6,
             label="1 kHz carrier, 100 % AM")
    peak_i = int(np.argmax(r))
    ax1.plot(fmods[peak_i], r[peak_i], "o", color=COLOR_SECONDARY, markersize=9,
             markerfacecolor="none", markeredgewidth=1.6, zorder=5)
    ax1.annotate(f"Peak $R$ = {r[peak_i]:.1f} asper @ {fmods[peak_i]:.0f} Hz",
                 xy=(float(fmods[peak_i]), float(r[peak_i])),
                 xytext=(105.0, float(r[peak_i]) * 0.95), fontsize=10,
                 color=COLOR_FG,
                 arrowprops={"arrowstyle": "->", "lw": 0.9, "color": COLOR_FG})
    ax1.set_title("ECMA-418-2 Roughness vs Modulation Frequency",
                  pad=10)
    ax1.set_xlabel(r"Modulation frequency $f_{\mathrm{mod}}$ [Hz]")
    ax1.set_ylabel("Roughness $R$ [asper]")
    ax1.set_xlim(10, 210)
    ax1.set_ylim(0, float(r.max()) * 1.25)
    ax1.legend(loc="upper right", fontsize=9)

    fig.suptitle("Sound Quality Metrics (ECMA-418-2 Sottek Hearing Model)",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save_figure(output_dir, "tonality_roughness_demo.png")
    plt.close()


@cache
def _fs_ecma_sweep_data() -> tuple[np.ndarray, np.ndarray]:
    """ECMA-418-2 Clause 9 F of a 1 kHz / 60 dB / 100 %-AM tone vs f_mod.

    Cached (language/theme independent): the Sottek fluctuation-strength
    chain is run once for the modulation-frequency sweep. The overall level
    convention (60 dB SPL of the modulated signal) matches the Clause 9
    calibration.
    """
    from phonometry.psychoacoustics import fluctuation_strength_ecma

    dur = 3.0
    t = np.arange(int(dur * _FS_PSY)) / _FS_PSY
    carrier = np.sin(2.0 * np.pi * 1000.0 * t)
    fmods = np.array([0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 12.0, 16.0,
                      24.0, 32.0])
    f_vals = []
    for fm in fmods:
        am = (1.0 + np.sin(2.0 * np.pi * fm * t)) * carrier
        am = am / np.sqrt(np.mean(am ** 2)) * _P_REF * 10.0 ** (60.0 / 20.0)
        f_vals.append(
            fluctuation_strength_ecma(am, float(_FS_PSY)).fluctuation_strength
        )
    return fmods, np.array(f_vals)


def generate_hms_modulation_bandpass(output_dir: str) -> None:
    """Complementary modulation band-passes of the Sottek Hearing Model.

    Fluctuation strength (ECMA-418-2 Clause 9, maximum near 4 Hz) and
    roughness (Clause 7, maximum near 70 Hz) computed for the same
    1 kHz / 100 % AM / 60 dB signal family over the modulation rate.
    """
    print("Generating hms_modulation_bandpass...")
    fm_fs, f_vals = _fs_ecma_sweep_data()
    fm_r, r_vals = _roughness_sweep_data()

    _fig, ax = plt.subplots(figsize=(10, 6.2))
    # Per-metric identity colors (teal = fluctuation strength, brown =
    # roughness), matching the result .plot() renderers; legible on both
    # themes, kept literal on purpose.
    ax.semilogx(fm_fs, f_vals, "o-", color="#17becf", linewidth=2.2,
                markersize=6, label="Fluctuation strength $F$ (Clause 9, slow modulation)")
    ax.semilogx(fm_r, r_vals, "s-", color="#8c564b", linewidth=2.2,
                markersize=6, label="Roughness $R$ (Clause 7, fast modulation)")

    i_f = int(np.argmax(f_vals))
    i_r = int(np.argmax(r_vals))
    ax.annotate(f"$F$ = {f_vals[i_f]:.2f} {_hms('vacil_HMS')} @ {fm_fs[i_f]:.0f} Hz",
                xy=(float(fm_fs[i_f]), float(f_vals[i_f])),
                xytext=(float(fm_fs[i_f]) * 1.6, float(f_vals[i_f]) * 1.06),
                fontsize=10, color=COLOR_FG,
                arrowprops={"arrowstyle": "->", "lw": 0.9, "color": COLOR_FG})
    ax.annotate(f"$R$ = {r_vals[i_r]:.2f} asper @ {fm_r[i_r]:.0f} Hz",
                xy=(float(fm_r[i_r]), float(r_vals[i_r])),
                xytext=(float(fm_r[i_r]) * 1.5, float(r_vals[i_r]) * 1.06),
                fontsize=10, color=COLOR_FG,
                arrowprops={"arrowstyle": "->", "lw": 0.9, "color": COLOR_FG})

    ax.set_xlabel(r"Modulation frequency $f_{\mathrm{mod}}$ [Hz]")
    ax.set_ylabel(f"$F$ [{_hms('vacil_HMS')}] / $R$ [asper]")
    ax.set_title("Slow vs Fast Modulation Perception (ECMA-418-2 Sottek Hearing Model)",
                 pad=12)
    top = max(float(np.max(f_vals)), float(np.max(r_vals)))
    ax.set_ylim(0.0, top * 1.22)
    ax.set_xlim(0.4, 260.0)
    ax.grid(which="both", color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.set_xticks([0.5, 1, 2, 4, 8, 16, 32, 70, 140, 250])
    ax.set_xticklabels(["0.5", "1", "2", "4", "8", "16", "32", "70", "140", "250"])
    ax.legend(loc="upper left", fontsize=9)
    # Bottom-left: the only stretch of the frame both band-passes leave
    # clear. Right-aligned, the longer Spanish caption ran into the
    # fluctuation-strength curve as it descends to zero near 32 Hz.
    ax.text(0.015, 0.03, "1 kHz carrier, 100 % AM, overall 60 dB SPL",
            transform=ax.transAxes, va="bottom", ha="left", fontsize=8.5,
            color=COLOR_FG)
    plt.tight_layout()
    save_figure(output_dir, "hms_modulation_bandpass.svg")
    plt.close()


@cache
def _fluctuation_am_tone_sweep() -> tuple[np.ndarray, np.ndarray]:
    """Osses 2016 signal-model F of a 1 kHz / 70 dB / 100 %-AM tone vs f_mod.

    Cached (language/theme independent): the signal model is run once for the
    modulation-frequency sweep {1, 2, 4, 8, 16, 32} Hz. Reproduces the band-pass
    sensation with its maximum at 4 Hz (Osses 2016 Table 1 trend).
    """
    from phonometry.psychoacoustics import fluctuation_strength

    dur = 2.0
    t = np.arange(int(dur * _FS_PSY)) / _FS_PSY
    carrier = np.sin(2.0 * np.pi * 1000.0 * t)
    fmods = np.array([1.0, 2.0, 4.0, 8.0, 16.0, 32.0])
    f_vals = []
    for fm in fmods:
        am = (1.0 + np.sin(2.0 * np.pi * fm * t)) * carrier
        am = am / np.sqrt(np.mean(am ** 2)) * _P_REF * 10.0 ** (70.0 / 20.0)
        f_vals.append(fluctuation_strength(am, float(_FS_PSY)).fluctuation_strength)
    return fmods, np.array(f_vals)


@cache
def _fluctuation_am_noise_sweep() -> tuple[np.ndarray, np.ndarray]:
    """Osses 2016 signal-model F of AM broadband noise at 60 dB vs f_mod.

    The same stimulus the Fastl & Zwicker closed form of Eq. (10.2) is
    written for, so the two can be read against each other; cached because
    the signal model is the expensive half of the figure.
    """
    from phonometry.psychoacoustics import fluctuation_strength

    rng = np.random.default_rng(3)
    t = np.arange(int(4.0 * _FS_PSY)) / _FS_PSY
    noise = rng.standard_normal(t.size)
    fmods = np.array([0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0])
    f_vals = []
    for fm in fmods:
        am = (1.0 + np.sin(2.0 * np.pi * fm * t)) * noise
        am = am / np.sqrt(np.mean(am ** 2)) * _P_REF * 10.0 ** (60.0 / 20.0)
        f_vals.append(fluctuation_strength(am, float(_FS_PSY)).fluctuation_strength)
    return fmods, np.array(f_vals)


def generate_fluctuation_strength(output_dir: str) -> None:
    """The two models on one stimulus, and the signal model on its own."""
    print("Generating fluctuation_strength...")
    from phonometry import fluctuation_strength_am_noise

    # Exact closed form (Fastl & Zwicker Eq. 10.2) for AM broadband noise at
    # 60 dB, 100 % modulation, swept over f_mod on a log axis.
    fmod = np.logspace(np.log10(0.5), np.log10(32.0), 240)
    f_bbn = np.array([fluctuation_strength_am_noise(60.0, 1.0, fm) for fm in fmod])
    bbn_peak = int(np.argmax(f_bbn))

    # The Osses 2016 signal model on the same AM broadband noise, and on the
    # AM tone it was calibrated for.
    fm_noise, f_noise = _fluctuation_am_noise_sweep()
    fm_tone, f_tone = _fluctuation_am_tone_sweep()
    tone_peak = int(np.argmax(f_tone))
    overshoot = float(f_noise[int(np.argmin(np.abs(fm_noise - 4.0)))]
                      / fluctuation_strength_am_noise(60.0, 1.0, 4.0))

    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(12.0, 5.6), gridspec_kw={"width_ratios": [1.55, 1.0]})

    ax.semilogx(fmod, f_bbn, color=COLOR_PRIMARY, linewidth=2.4,
                label=(f"closed form, Eq. 10.2 — "
                       f"peak {f_bbn[bbn_peak]:.1f} vacil"))
    ax.plot(fmod[bbn_peak], f_bbn[bbn_peak], "o", color=COLOR_PRIMARY,
            markersize=8, markerfacecolor="white", markeredgewidth=1.6, zorder=6)
    ax.semilogx(fm_noise, f_noise, "s--", color=COLOR_SECONDARY, linewidth=2.0,
                markersize=7,
                label=(f"Osses 2016 signal model — "
                       f"peak {f_noise.max():.1f} vacil"))
    ax.axvline(4.0, color=COLOR_FG, linestyle="--", linewidth=1.0, alpha=0.7,
               label="4 Hz reference")
    ax.annotate(f"×{overshoot:.1f} at 4 Hz",
                xy=(4.0, float(f_noise[int(np.argmin(np.abs(fm_noise - 4.0)))])),
                xytext=(0.62, float(f_noise.max()) * 0.86), fontsize=10,
                color=COLOR_SECONDARY,
                arrowprops={"arrowstyle": "->", "lw": 1.0,
                            "color": COLOR_SECONDARY})
    ax.set_xlabel(r"Modulation frequency $f_{\mathrm{mod}}$ [Hz]")
    ax.set_ylabel("Fluctuation strength $F$ [vacil]")
    ax.set_ylim(0.0, max(float(f_bbn.max()), float(f_noise.max())) * 1.20)
    ax.set_title("Both models on AM broadband noise, 60 dB",
                 pad=10)
    ax.grid(which="both", color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.set_xticks([0.5, 1, 2, 4, 8, 16, 32])
    ax.set_xticklabels(["0.5", "1", "2", "4", "8", "16", "32"])
    ax.legend(loc="upper right", fontsize=9)

    info = [
        r"$F = 5.8\,(1.25\,m - 0.25)(0.05\,L - 1)$",
        r"      $/\,[(f_{\mathrm{mod}}/5)^2 + 4/f_{\mathrm{mod}} + 1.5]$  vacil",
    ]
    ax.text(0.015, 0.03, "\n".join(info), transform=ax.transAxes,
            va="bottom", ha="left", fontsize=8.5, color=COLOR_FG,
            bbox={"boxstyle": "round,pad=0.5", "facecolor": COLOR_PANEL,
                  "edgecolor": COLOR_GRID})

    # The stimulus the signal model was calibrated on, on its own axes.
    ax2.semilogx(fm_tone, f_tone, "s--", color=COLOR_TERTIARY, linewidth=2.0,
                 markersize=7,
                 label=f"peak {f_tone[tone_peak]:.2f} vacil at "
                       f"{fm_tone[tone_peak]:g} Hz")
    ax2.axvline(4.0, color=COLOR_FG, linestyle="--", linewidth=1.0, alpha=0.7)
    ax2.set_xlabel(r"Modulation frequency $f_{\mathrm{mod}}$ [Hz]")
    ax2.set_ylabel("Fluctuation strength $F$ [vacil]")
    ax2.set_ylim(0.0, float(f_tone.max()) * 1.28)
    ax2.set_title("Signal model on the AM tone, 70 dB",
                  pad=10)
    ax2.grid(which="both", color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax2.set_axisbelow(True)
    ax2.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax2.set_xticks([1, 2, 4, 8, 16, 32])
    ax2.set_xticklabels(["1", "2", "4", "8", "16", "32"])
    ax2.legend(loc="upper right", fontsize=9)

    fig.suptitle("Fluctuation Strength — the 4 Hz Band-Pass, and Which Model "
                 "to Quote")
    plt.tight_layout()
    save_figure(output_dir, "fluctuation_strength.svg")
    plt.close()


def generate_annoyance_weightings(output_dir: str) -> None:
    """The two pieces of modelling judgement in the PA formula, drawn."""
    print("Generating annoyance_weightings...")
    from phonometry import psychoacoustic_annoyance

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.0, 5.4))

    # Left: the sharpness weighting against S, for three loudnesses.
    sharpness = np.linspace(1.0, 5.0, 200)
    for n5, style in ((10.0, "-"), (30.0, "--"), (60.0, ":")):
        w_s = [psychoacoustic_annoyance(n5, float(s), 0.0, 0.0).w_s
               for s in sharpness]
        ax.plot(sharpness, w_s, style, color=COLOR_PRIMARY, linewidth=2.0,
                label=f"$N_5$ = {n5:g} sone")
    ax.axvline(1.75, color=COLOR_SECONDARY, linewidth=1.4, linestyle="--")
    ax.annotate("1.75 acum: below it sharpness costs nothing",
                xy=(1.75, 0.02), xytext=(2.1, 0.06), fontsize=10,
                color=COLOR_SECONDARY,
                arrowprops={"arrowstyle": "->", "lw": 1.0,
                            "color": COLOR_SECONDARY})
    ax.set_xlabel("Sharpness $S$ [acum]")
    # S is the sharpness quantity symbol (the x label draws it as $S$), so the
    # subscript of w stays italic: Fastl & Zwicker (16.3) prints it that way.
    ax.set_ylabel("Sharpness weighting $w_S$")
    ax.set_xlim(1.0, 5.0)
    ax.set_ylim(0.0, None)
    ax.set_title("The kink at the reference sharpness",
                 pad=10)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=9)

    # Right: the same magnitude spent on roughness or on fluctuation strength.
    magnitude = np.linspace(0.0, 1.5, 120)
    rough = [psychoacoustic_annoyance(30.0, 2.0, 0.0, float(v)).annoyance
             for v in magnitude]
    fluct = [psychoacoustic_annoyance(30.0, 2.0, float(v), 0.0).annoyance
             for v in magnitude]
    ax2.plot(magnitude, rough, color=COLOR_SECONDARY, linewidth=2.2,
             label="all roughness: $R = v$, $F = 0$")
    ax2.plot(magnitude, fluct, "--", color=COLOR_PRIMARY, linewidth=2.2,
             label="all fluctuation: $F = v$, $R = 0$")
    # The readout sits below the legend: at the legend's own height it was
    # drawn straight through its second row (visible in the shipped asset).
    ax2.annotate(f"at $v$ = 1.5, {rough[-1] - fluct[-1]:.1f} units apart",
                 xy=(1.5, (rough[-1] + fluct[-1]) / 2),
                 xytext=(0.30, rough[-1] * 0.935), fontsize=10, color=COLOR_FG,
                 arrowprops={"arrowstyle": "->", "lw": 1.0})
    ax2.set_xlabel("Sensation magnitude $v$ [asper or vacil]")
    ax2.set_ylabel("Psychoacoustic annoyance PA")
    ax2.set_xlim(0.0, 1.5)
    ax2.set_title("Roughness costs more than fluctuation (0.6 against 0.4)",
                  pad=10)
    ax2.grid(color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax2.set_axisbelow(True)
    ax2.legend(loc="upper left", fontsize=9)
    # Bottom right, not bottom left: at the left the opaque box lay on the
    # first fifth of both curves and hid where they start.
    ax2.text(0.975, 0.035, "$N_5$ = 30 sone, $S$ = 2.0 acum throughout",
             transform=ax2.transAxes, fontsize=9, color=COLOR_FG,
             ha="right",
             bbox={"boxstyle": "round,pad=0.4", "facecolor": COLOR_PANEL,
                   "edgecolor": COLOR_GRID})

    fig.suptitle("The Two Weightings of the Psychoacoustic Annoyance Model",
                 )
    plt.tight_layout()
    save_figure(output_dir, "annoyance_weightings.svg")
    plt.close()


def generate_psychoacoustic_annoyance(output_dir: str) -> None:
    """Psychoacoustic annoyance PA vs loudness N5 for three sensation profiles."""
    print("Generating psychoacoustic_annoyance...")
    from phonometry.psychoacoustics import psychoacoustic_annoyance

    n5 = np.linspace(4.0, 60.0, 200)
    # (label, sharpness [acum], fluctuation strength [vacil], roughness [asper],
    #  colour, linestyle).
    profiles = [
        ("Baseline: $S$ = 1.75 acum, $F = R = 0$", 1.75, 0.0, 0.0,
         COLOR_FG, "--"),
        ("Sharp: $S$ = 3.5 acum", 3.5, 0.0, 0.0, COLOR_PRIMARY, "-"),
        ("Rough + fluctuating: $F$ = 1.2 vacil, $R$ = 0.7 asper", 2.0, 1.2, 0.7,
         COLOR_TERTIARY, "-"),
    ]

    _fig, ax = plt.subplots(figsize=(10, 6.2))
    for label, s, f, r, color, ls in profiles:
        pa = np.array([psychoacoustic_annoyance(v, s, f, r).annoyance for v in n5])
        lw = 1.6 if ls == "--" else 2.4
        alpha = 0.7 if ls == "--" else 1.0
        ax.plot(n5, pa, color=color, linestyle=ls, linewidth=lw, alpha=alpha,
                label=label)

    # Worked example: N5 = 30 sone, S = 2.0 acum, F = 0.5 vacil, R = 0.3 asper.
    ex = psychoacoustic_annoyance(30.0, 2.0, 0.5, 0.3)
    ax.plot([30.0], [ex.annoyance], "o", color=COLOR_SECONDARY, markersize=10,
            markerfacecolor="white", markeredgewidth=2.0, zorder=6,
            label=f"Worked example (PA = {ex.annoyance:.2f})")
    ax.annotate(f"PA = {ex.annoyance:.2f}\n$w_S$ = {ex.w_s:.3f}, "
                f"$w_{{FR}}$ = {ex.w_fr:.3f}",
                xy=(30.0, ex.annoyance), xytext=(33.0, ex.annoyance * 0.72),
                fontsize=9, color=COLOR_FG,
                arrowprops={"arrowstyle": "->", "lw": 0.9, "color": COLOR_FG})

    ax.set_xlabel("Percentile loudness $N_5$ [sone]")
    ax.set_ylabel("Psychoacoustic annoyance PA")
    ax.set_xlim(0.0, 62.0)
    ax.set_ylim(0.0, None)
    ax.set_title("Psychoacoustic Annoyance vs Loudness (Fastl & Zwicker)",
                 pad=12)
    ax.grid(which="major", color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=9)

    info = [
        # S, F and R are the sharpness, fluctuation-strength and roughness
        # quantity symbols, drawn italic in the same box, so the subscripts
        # they form stay italic too (Fastl & Zwicker, Eqs 16.2 to 16.4).
        r"$\mathrm{PA} = N_5(1 + \sqrt{w_S^2 + w_{FR}^2})$",
        r"$w_S = (S - 1.75)\,0.25\,\log_{10}(N_5 + 10)$",
        r"$w_{FR} = (2.18/N_5^{0.4})(0.4\,F + 0.6\,R)$",
    ]
    ax.text(0.985, 0.03, "\n".join(info), transform=ax.transAxes,
            va="bottom", ha="right", fontsize=8.5, color=COLOR_FG,
            bbox={"boxstyle": "round,pad=0.5", "facecolor": COLOR_PANEL,
                  "edgecolor": COLOR_GRID})
    plt.tight_layout()
    save_figure(output_dir, "psychoacoustic_annoyance.svg")
    plt.close()


def generate_stoi_intelligibility(output_dir: str) -> None:
    """STOI vs ESTOI over SNR for stationary and modulated maskers."""
    print("Generating stoi_intelligibility...")
    from phonometry import stoi

    fs = 10000  # the STOI internal rate: no resampling, faster and exact
    rng = np.random.default_rng(20)
    t = np.arange(3 * fs) / fs

    # A speech-like clean signal: amplitude-modulated formant-ish tones.
    clean = np.zeros_like(t)
    for f0 in (200.0, 400.0, 700.0, 1100.0, 1800.0, 2600.0):
        depth = 0.5 * (1.0 + np.sin(2 * np.pi * rng.uniform(2.0, 6.0) * t
                                    + rng.uniform(0.0, 2 * np.pi)))
        clean += depth * np.sin(2 * np.pi * f0 * t + rng.uniform(0.0, 2 * np.pi))
    p_clean = float(np.sqrt(np.mean(clean**2)))

    # Two maskers: a stationary Gaussian noise and the same noise deeply gated
    # at 5 Hz (a modulated masker with quiet gaps, as in Jensen & Taal 2016).
    base_noise = rng.standard_normal(clean.size)
    gate = 0.5 * (1.0 + np.sign(np.sin(2 * np.pi * 5.0 * t)))  # 0/1 square gate
    modulated = base_noise * (0.05 + 0.95 * gate)

    snrs = np.arange(-15.0, 20.1, 5.0)

    def curve(masker: np.ndarray, extended: bool) -> np.ndarray:
        p_m = float(np.sqrt(np.mean(masker**2)))
        out = []
        for snr in snrs:
            g = p_clean / (p_m * 10.0 ** (snr / 20.0))
            out.append(stoi(clean, clean + g * masker, fs, extended=extended).value)
        return np.asarray(out)

    fig, (ax_stoi, ax_estoi) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, extended, title in (
        (ax_stoi, False, "STOI (Taal et al. 2011)"),
        (ax_estoi, True, "ESTOI (Jensen & Taal 2016)"),
    ):
        ax.plot(snrs, curve(base_noise, extended), "o-", color=COLOR_PRIMARY,
                linewidth=1.7, label="Stationary masker")
        ax.plot(snrs, curve(modulated, extended), "s--", color=COLOR_SECONDARY,
                linewidth=1.7, label="Modulated (5 Hz gated) masker")
        ax.set_title(title)
        ax.set_xlabel("SNR [dB]")
        ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5)
        ax.set_axisbelow(True)
        ax.set_ylim(0.0, 1.0)
        ax.legend(loc="upper left", fontsize=9)
    ax_stoi.set_ylabel("Intelligibility index")
    ax_estoi.text(0.985, 0.03,
                  "ESTOI rates the modulated masker higher: it credits the\n"
                  "speech glimpsed in the quiet gaps. STOI barely separates them.",
                  transform=ax_estoi.transAxes, va="bottom", ha="right",
                  fontsize=8.5, color=COLOR_FG)
    fig.suptitle("Short-Time Objective Intelligibility: STOI vs ESTOI",
                 )
    plt.tight_layout()
    save_figure(output_dir, "stoi_intelligibility.svg")
    plt.close()


@cache
def _time_loudness_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ISO 532-3 STL(t)/LTL(t) for a 1 kHz / 60 dB burst (on 200-400 ms)."""
    from phonometry.psychoacoustics import loudness_moore_glasberg_time

    dur = 0.8
    t = np.arange(int(dur * _FS_PSY)) / _FS_PSY
    sig = np.zeros_like(t)
    on = (t >= 0.2) & (t < 0.4)
    sig[on] = _P_REF * 10.0 ** (60.0 / 20.0) * np.sqrt(2.0) * np.sin(
        2.0 * np.pi * 1000.0 * t[on]
    )
    tv = loudness_moore_glasberg_time(sig, _FS_PSY)
    return (tv.time.copy(), tv.short_term_loudness.copy(),
            tv.long_term_loudness.copy())


def generate_moore_glasberg_time_loudness(output_dir: str) -> None:
    """ISO 532-3 short-term vs long-term loudness for a 1 kHz tone burst."""
    print("Generating moore_glasberg_time_loudness.png...")
    time, stl, ltl = _time_loudness_data()

    _, ax = plt.subplots(figsize=(10, 6))
    # Shade the burst window (200-400 ms).
    ax.axvspan(0.2, 0.4, color=theme_fill(COLOR_FG, ax), linewidth=0, zorder=0)
    ax.plot(time, stl, color=COLOR_PRIMARY, linewidth=1.8,
            label=f"Short-term loudness STL (STL peak = {stl.max():.1f} sone)")
    ax.plot(time, ltl, color=COLOR_SECONDARY, linewidth=2.0,
            label=f"Long-term loudness LTL (LTL peak = {ltl.max():.1f} sone)")

    ax.annotate("1 kHz burst, 200 ms", xy=(0.3, 0.0),
                xytext=(0.3, float(stl.max()) * 1.02), fontsize=10,
                color=COLOR_FG, ha="center")
    ax.annotate("Fast attack / release",
                xy=(float(time[int(np.argmax(stl))]), float(stl.max())),
                xytext=(0.45, float(stl.max()) * 0.82), fontsize=9,
                color=COLOR_PRIMARY,
                arrowprops={"arrowstyle": "->", "lw": 0.9, "color": COLOR_PRIMARY})
    ax.annotate("Slow integration",
                xy=(0.55, float(np.interp(0.55, time, ltl))),
                xytext=(0.58, float(ltl.max()) * 0.55), fontsize=9,
                color=COLOR_SECONDARY,
                arrowprops={"arrowstyle": "->", "lw": 0.9, "color": COLOR_SECONDARY})

    ax.set_title("Time-Varying Loudness (ISO 532-3)",
                 pad=12)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Loudness [sone]")
    ax.set_xlim(0, float(time[-1]))
    ax.set_ylim(0, float(stl.max()) * 1.18)
    ax.legend(loc="upper right", fontsize=9)
    save_figure(output_dir, "moore_glasberg_time_loudness.png")
    plt.close()


def generate_tone_audibility(output_dir: str) -> None:
    """ISO/PAS 20065 tonal audibility: per-tone ΔL of the Annex E example."""
    print("Generating tone_audibility...")
    from phonometry import assess_tones

    # Annex E combustion-engine example, spectrum 1 (Tables E.2/E.3),
    # line spacing Δf = 2.7 Hz. Each tuple is (fT, LS, LT).
    tones = [
        (118.4, 48.91, 64.56), (137.3, 49.22, 67.96), (158.8, 50.50, 68.63),
        (314.9, 52.85, 68.50), (433.4, 58.29, 73.17), (592.2, 59.53, 78.31),
        (629.8, 59.71, 75.00), (643.3, 61.98, 79.75), (1582.7, 54.16, 71.07),
    ]
    freqs = [t[0] for t in tones]
    res = assess_tones(freqs, [t[2] for t in tones], [t[1] for t in tones], 2.7)

    x = np.arange(len(freqs))
    decisive = int(np.argmax(res.audibilities))
    colors = [COLOR_PRIMARY] * len(freqs)
    colors[decisive] = COLOR_SECONDARY

    _fig, ax = plt.subplots(figsize=(10, 6.2))
    ax.bar(x, res.audibilities, width=0.7, color=colors, edgecolor=COLOR_FG,
           linewidth=0.6)
    ax.axhline(0.0, color=COLOR_FG, ls="--", lw=1.0,
               label=r"threshold $\Delta L = 0$ dB")
    ax.bar([decisive], [res.audibilities[decisive]], width=0.7,
           color=COLOR_SECONDARY, edgecolor=COLOR_FG, linewidth=0.6,
           label=(rf"decisive $\Delta L$ = {res.decisive_audibility:.1f} dB "
                  rf"@ {res.decisive_frequency:g} Hz"))

    ax.set_xticks(x)
    ax.set_xticklabels([f"{f:g}" for f in freqs], rotation=45, ha="right")
    ax.set_xlabel(LABEL_FREQ_HZ)
    ax.set_ylabel(r"Audibility $\Delta L$ [dB]")
    ax.set_title("ISO/PAS 20065 Tonal Audibility", pad=12)
    ax.grid(which="major", axis="y", color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=9)

    info = [
        r"$\Delta f_\mathrm{c} = 25 + 75\,(1 + 1.4\,(f_\mathrm{T}/1000)^2)^{0.69}$",
        (r"$L_\mathrm{G} = L_\mathrm{S} + 10\,\log_{10}(\Delta f_\mathrm{c}"
         r"/\Delta f)$,"
         r"  $a_\mathrm{v} = -2 - \log_{10}(1 + (f/502)^{2.5})$"),
        (r"$\Delta L = L_\mathrm{T} - L_\mathrm{G} - a_\mathrm{v}$"
         r"  (combustion engine, Annex E)"),
    ]
    ax.text(0.015, 0.02, "\n".join(info), transform=ax.transAxes,
            va="bottom", ha="left", fontsize=8.5, color=COLOR_FG,
            bbox={"boxstyle": "round,pad=0.5", "facecolor": COLOR_PANEL,
                  "edgecolor": COLOR_GRID})
    plt.tight_layout()
    save_figure(output_dir, "tone_audibility.svg")
    plt.close()


def generate_exposure_uncertainty(output_dir: str) -> None:
    """ISO 9612 Annex D task-based exposure with its expanded uncertainty."""
    print("Generating exposure_uncertainty.png...")
    from phonometry.hearing.occupational_exposure import Task, task_based_exposure

    tasks = [
        Task(samples=(70.0,), duration_hours=1.5, label="planning/breaks"),
        Task(samples=(80.1, 82.2, 79.6), duration_hours=5.0,
             duration_range=(4.0, 6.0), label="welding"),
        Task(samples=(86.5, 92.4, 89.3, 93.2, 87.8, 86.2), duration_hours=1.5,
             duration_range=(1.0, 2.0), label="cutting/grinding"),
    ]
    result = task_based_exposure(tasks, include_duration_uncertainty=False,
                                 warn=False)
    labels = [t.label for t in result.tasks]
    contribs = [t.lex_8h_contribution for t in result.tasks]
    lex = result.lex_8h
    upper = result.upper_limit  # LEX,8h + U

    x = np.arange(len(labels))
    _fig, ax = plt.subplots(figsize=(10, 6.3))
    ax.bar(x, contribs, color=COLOR_PRIMARY, edgecolor=COLOR_FG, linewidth=0.7,
           width=0.6, zorder=3, label="Measurement task")
    for xi, c in zip(x, contribs):
        ax.text(float(xi), c - 2.5, f"{c:.1f}", ha="center", va="top",
                fontsize=9, color="white", fontweight="bold")

    # Daily energy-summed level and its one-sided 95 % upper limit LEX,8h + U.
    ax.axhspan(lex, upper, color=COLOR_SECONDARY, alpha=0.14, zorder=0,
               label=r"$L_{\mathrm{EX,8h}} + U$ (one-sided 95 %)")
    ax.axhline(lex, color=COLOR_SECONDARY, linewidth=2.0, zorder=4,
               label=r"Daily $L_{\mathrm{EX,8h}}$")
    ax.axhline(upper, color=COLOR_SECONDARY, linewidth=1.2, linestyle="--",
               zorder=4)

    box = [
        rf"$L_{{\mathrm{{EX,8h}}}}$ = {lex:.1f} dB",
        f"$U$ = {result.expanded_uncertainty:.1f} dB ($k$ = 1.65)",
        rf"$L_{{\mathrm{{EX,8h}}}} + U$ = {upper:.1f} dB",
    ]
    ax.text(0.03, 0.78, "\n".join(box), transform=ax.transAxes, va="top",
            ha="left", fontsize=10, color=COLOR_FG,
            bbox={"boxstyle": "round,pad=0.5", "facecolor": COLOR_PANEL,
                  "edgecolor": COLOR_GRID})

    ax.set_title("ISO 9612 Task-Based Exposure (Annex D)",
                 pad=12)
    ax.set_xlabel("Measurement task")
    ax.set_ylabel(r"$L_{\mathrm{EX,8h}}$ contribution [dB]")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylim(0.0, upper + 10.0)
    ax.grid(axis="y", color=COLOR_GRID, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    save_figure(output_dir, "exposure_uncertainty.png")
    plt.close()


def generate_speech_intelligibility(output_dir: str) -> None:
    """ANSI S3.5-1997: band audibility and the SII in broadband noise."""
    print("Generating speech_intelligibility.png...")
    from phonometry import speech_intelligibility_index, standard_speech_spectrum

    # Standard normal-effort speech in a descending broadband masking noise
    # (an office/ventilation-like spectrum): the band-audibility function A_i
    # is partial across the band, and the importance-weighted contribution
    # I_i*A_i (ANSI S3.5-1997 clause 6) sums to the index SII.
    speech_spectrum = standard_speech_spectrum("normal")
    noise = np.array([38.0, 37.0, 36.0, 34.0, 32.0, 30.0, 28.0, 26.0, 24.0,
                      22.0, 20.0, 18.0, 16.0, 14.0, 12.0, 10.0, 8.0, 6.0])
    result = speech_intelligibility_index(speech_spectrum, noise)

    freqs = result.frequencies
    positions = np.arange(freqs.size)
    weighted = result.band_audibility * result.band_importance

    _fig, ax = plt.subplots(figsize=(10, 6.3))
    ax.bar(positions, result.band_audibility, width=0.8, color=COLOR_PRIMARY,
           alpha=0.35, zorder=2, label=r"Band audibility $A_i$")
    ax.bar(positions, weighted / weighted.max(), width=0.45, color=COLOR_PRIMARY,
           zorder=3, label=r"Importance-weighted $I_i\,A_i$ (scaled)")
    ax.set_title(
        f"Speech Intelligibility Index (ANSI S3.5-1997)   SII = {result.sii:.2f}",
        pad=12)
    ax.set_xlabel("One-third-octave band [Hz]")
    ax.set_ylabel("Band audibility")
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(positions)
    ax.set_xticklabels([f"{f:g}" for f in freqs], rotation=45, ha="right")
    ax.grid(which="major", axis="y", color=COLOR_GRID, linestyle="-", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right")
    plt.tight_layout()
    save_figure(output_dir, "speech_intelligibility.png")
    plt.close()


def generate_sii_vocal_efforts(output_dir: str) -> None:
    """ANSI S3.5-1997 Table 3 standard speech spectra by vocal effort."""
    print("Generating sii_vocal_efforts.png...")
    from phonometry import speech_intelligibility_index, standard_speech_spectrum
    from phonometry.speech.sii import BAND_CENTERS, VOCAL_EFFORTS

    freqs = BAND_CENTERS
    # Distinct hues (not COLOR_GRID, which blends into the gridlines and is
    # near-invisible on a light background) for the four ordered efforts.
    colours = {"normal": COLOR_TERTIARY, "raised": "#7f7f7f",
               "loud": COLOR_PRIMARY, "shout": COLOR_SECONDARY}
    _fig, (ax_s, ax_i) = plt.subplots(1, 2, figsize=(12.5, 5.6))

    # --- Left: the four standard speech spectra. ---
    for effort in VOCAL_EFFORTS:
        ax_s.plot(freqs, standard_speech_spectrum(effort), "o-",
                  color=colours[effort], label=effort.capitalize())
    ax_s.set_xscale("log")
    ax_s.set_xticks(list(freqs))
    ax_s.set_xticklabels([f"{f:g}" for f in freqs], rotation=45, ha="right")
    ax_s.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax_s.set_xlabel("One-third-octave band [Hz]")
    ax_s.set_ylabel("Speech spectrum level [dB SPL]")
    ax_s.set_title("ANSI S3.5-1997 — speech spectra by vocal effort",
                   pad=10)
    ax_s.grid(which="both", color=COLOR_GRID, linestyle="-", alpha=0.4)
    ax_s.set_axisbelow(True)
    ax_s.legend(loc="upper right")

    # --- Right: SII in a fixed broadband noise rises with vocal effort. ---
    noise = np.array([48.0, 47.0, 46.0, 44.0, 42.0, 40.0, 38.0, 36.0, 34.0,
                      32.0, 30.0, 28.0, 26.0, 24.0, 22.0, 20.0, 18.0, 16.0])
    indices = [speech_intelligibility_index(e, noise).sii for e in VOCAL_EFFORTS]
    positions = np.arange(len(VOCAL_EFFORTS))
    bar_colours = [colours[e] for e in VOCAL_EFFORTS]
    ax_i.bar(positions, indices, width=0.6, color=bar_colours, zorder=2)
    for x, v in zip(positions, indices):
        ax_i.text(x, v + 0.01, f"{v:.2f}", ha="center", va="bottom",
                  fontweight="bold")
    ax_i.set_xticks(positions)
    ax_i.set_xticklabels([e.capitalize() for e in VOCAL_EFFORTS])
    ax_i.set_ylim(0.0, 1.0)
    ax_i.set_ylabel("Speech Intelligibility Index")
    ax_i.set_title("SII vs vocal effort in a fixed noise",
                   pad=10)
    ax_i.grid(which="major", axis="y", color=COLOR_GRID, linestyle="-", alpha=0.5)
    ax_i.set_axisbelow(True)

    plt.tight_layout()
    save_figure(output_dir, "sii_vocal_efforts.png")
    plt.close()


def generate_standard_speech_spectrum(output_dir: str) -> None:
    """ANSI S3.5-1997 Table 3 standard speech spectra via StandardSpeechSpectrum.plot()."""
    print("Generating standard_speech_spectrum...")
    from phonometry import standard_speech_spectra

    _, ax = plt.subplots(figsize=(10, 6))
    # The result's own .plot() draws one line per vocal effort on the categorical
    # one-third-octave band axis (160 Hz to 8000 Hz), each higher effort lifting
    # the whole speech spectrum (ANSI S3.5-1997 Table 3).
    standard_speech_spectra().plot(ax=ax, language=_LANG)
    plt.tight_layout()
    save_figure(output_dir, "standard_speech_spectrum.svg")
    plt.close()


def generate_sii_band_procedures(output_dir: str) -> None:
    """The band-importance functions of the four ANSI S3.5-1997 procedures."""
    print("Generating sii_band_procedures...")
    from phonometry import SII_METHODS, sii_procedure

    _, ax = plt.subplots(figsize=(10, 6))
    # Each procedure's own .plot() steps Ii across its tabulated band limits on
    # a shared logarithmic frequency axis, so the 6-band octave function and
    # the 21-band critical-band function are directly comparable: the same
    # importance is spread over very different bandwidths (Tables 1 to 4).
    for method in SII_METHODS:
        sii_procedure(method).plot(ax=ax, language=_LANG, linewidth=1.8)
    plt.tight_layout()
    save_figure(output_dir, "sii_band_procedures.svg")
    plt.close()


def generate_hearing_threshold(output_dir: str) -> None:
    """ISO 7029 age-related threshold and ISO 389-7 reference threshold."""
    print("Generating hearing_threshold.png...")
    from phonometry import age_threshold, reference_threshold
    from phonometry.hearing.threshold import AUDIOMETRIC_FREQUENCIES

    freqs = AUDIOMETRIC_FREQUENCIES
    _fig, (ax_age, ax_ref) = plt.subplots(1, 2, figsize=(12.5, 5.6))

    # --- Left: ISO 7029 median threshold by age (male) + 10-90 % band @70. ---
    ages = [(20, "#9e9e9e"), (40, "#7f7f7f"), (60, COLOR_PRIMARY),
            (80, COLOR_SECONDARY)]
    for age, color in ages:
        r = age_threshold(age, "male", 0.5)
        ax_age.plot(freqs, r.median, "o-", color=color, label=f"{age} yr")
    r70 = age_threshold(70, "male", 0.5)
    z90 = 1.2816
    ax_age.fill_between(freqs, r70.median - z90 * r70.spread_lower,
                        r70.median + z90 * r70.spread_upper,
                        color=theme_fill(COLOR_PRIMARY, ax_age), zorder=0,
                        label="10-90 % band (70 yr)")
    ax_age.set_xscale("log")
    ax_age.set_xticks(list(freqs))
    ax_age.set_xticklabels([f"{f:g}" for f in freqs], rotation=45, ha="right")
    ax_age.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax_age.invert_yaxis()
    ax_age.set_xlabel("Audiometric frequency [Hz]")
    ax_age.set_ylabel("Median threshold deviation from age 18 [dB]")
    ax_age.set_title("ISO 7029 — age-related threshold (male)",
                     pad=10)
    ax_age.grid(which="both", color=COLOR_GRID, linestyle="-", alpha=0.4)
    ax_age.set_axisbelow(True)
    ax_age.legend(loc="lower left")

    # --- Right: ISO 389-7 reference threshold, free vs diffuse field. ---
    ax_ref.plot(freqs, reference_threshold("free-field"), "o-",
                color=COLOR_PRIMARY, label="Free-field (frontal)")
    ax_ref.plot(freqs, reference_threshold("diffuse-field"), "s--",
                color=COLOR_SECONDARY, label="Diffuse-field")
    ax_ref.set_xscale("log")
    ax_ref.set_xticks(list(freqs))
    ax_ref.set_xticklabels([f"{f:g}" for f in freqs], rotation=45, ha="right")
    ax_ref.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax_ref.set_xlabel("Audiometric frequency [Hz]")
    ax_ref.set_ylabel("Reference threshold [dB]")
    ax_ref.set_title("ISO 389-7 — reference threshold of hearing",
                     pad=10)
    ax_ref.grid(which="both", color=COLOR_GRID, linestyle="-", alpha=0.4)
    ax_ref.set_axisbelow(True)
    ax_ref.legend(loc="upper left")

    plt.tight_layout()
    save_figure(output_dir, "hearing_threshold.png")
    plt.close()


def generate_age_threshold_sex_and_spread(output_dir: str) -> None:
    """ISO 7029 at 4 kHz: the sex difference, and how the two spreads grow.

    The guide states both claims in prose and neither is drawn anywhere. The
    right panel is also the honest version: ``s_u`` pulls away from ``s_l``
    only up to about age 64, then falls back and crosses under it at 75 --
    which lands exactly in the range clause 4.1 marks informative-only from
    3000 Hz upwards, because the oldest subjects ran off the audiometer scale.
    """
    print("Generating age_threshold_sex_and_spread.png...")
    from phonometry import age_threshold
    from phonometry.hearing.threshold import AUDIOMETRIC_FREQUENCIES

    idx = int(np.argmin(np.abs(np.asarray(AUDIOMETRIC_FREQUENCIES) - 4000.0)))
    ages = np.arange(18.0, 81.0, 1.0)
    res = {s: [age_threshold(float(a), s, 0.5) for a in ages]
           for s in ("male", "female")}
    med = {s: np.array([r.median[idx] for r in res[s]]) for s in res}
    s_u = np.array([r.spread_upper[idx] for r in res["male"]])
    s_l = np.array([r.spread_lower[idx] for r in res["male"]])

    _fig, (ax_sex, ax_spread) = plt.subplots(1, 2, figsize=(12.5, 5.4))

    # --- Left: the median at 4 kHz for the two sexes -----------------------
    ax_sex.plot(ages, med["male"], "-", color=COLOR_PRIMARY, linewidth=2.0,
                label="Male")
    ax_sex.plot(ages, med["female"], "--", color=COLOR_SECONDARY,
                linewidth=2.0, label="Female")
    ax_sex.fill_between(ages, med["female"], med["male"],
                        color=theme_fill(COLOR_PRIMARY, ax_sex), zorder=0)
    for age, dx, dy in ((60.0, -16.0, 9.0), (80.0, -19.0, -8.0)):
        k = int(np.argmin(np.abs(ages - age)))
        gap = med["male"][k] - med["female"][k]
        mid = 0.5 * (med["male"][k] + med["female"][k])
        ax_sex.annotate(
            f"{gap:.1f} dB", xy=(age, mid), xytext=(age + dx, mid + dy),
            fontsize=11, arrowprops={"arrowstyle": "->", "lw": 1.0},
        )
    ax_sex.set_xlabel("Age [years]")
    ax_sex.set_ylabel("Median deviation from age 18 [dB]")
    ax_sex.set_title("ISO 7029 median at 4 kHz — men against women",
                     pad=10)
    ax_sex.grid(color=COLOR_GRID, linestyle="-", alpha=0.4)
    ax_sex.set_axisbelow(True)
    ax_sex.legend(loc="upper left")

    # --- Right: the two half-Gaussian spreads at the same frequency --------
    ax_spread.plot(ages, s_u, "-", color=COLOR_PRIMARY, linewidth=2.0,
                   label=r"$s_\mathrm{u}$ (worse than the median)")
    ax_spread.plot(ages, s_l, "--", color=COLOR_TERTIARY, linewidth=2.0,
                   label=r"$s_\mathrm{l}$ (better than the median)")
    ax_spread.fill_between(ages, s_l, s_u, where=s_u >= s_l,
                           color=theme_fill(COLOR_PRIMARY, ax_spread),
                           zorder=0,
                           label=r"$s_\mathrm{u} - s_\mathrm{l}$: the asymmetry")
    # The three-line clause note is far wider than the ten-year band it
    # describes, so the top of the panel keeps a clear strip for it: the note
    # is anchored to the right edge in axes coordinates (Spanish grows to the
    # left from there, still well inside the frame) and the 70 yr marker stops
    # below the strip instead of running down through the words. The floor
    # sits below zero so that headroom does not push the start of the curves,
    # at 18 yr, down onto the legend.
    ax_spread.set_ylim(-1.5, max(s_u.max(), s_l.max()) + 10.0)
    peak = int(np.argmax(s_u))
    ax_spread.annotate(
        rf"$s_\mathrm{{u}}$ peaks at {ages[peak]:.0f} yr ({s_u[peak]:.1f} dB)",
        xy=(ages[peak], s_u[peak]), xytext=(24.0, s_u[peak] + 4.0),
        fontsize=11, arrowprops={"arrowstyle": "->", "lw": 1.0},
    )
    cross = ages[np.argmax(s_u < s_l)]
    ax_spread.axvline(70.0, ymax=0.84, color=COLOR_SECONDARY, linewidth=1.4,
                      linestyle=":")
    ax_spread.axvspan(70.0, ages[-1], color=theme_fill(COLOR_SECONDARY,
                                                       ax_spread), zorder=0)
    ax_spread.text(0.985, 0.98,
                   "above 70 yr:\ninformative only\n(clause 4.1, $f \\geq 3$ kHz)",
                   transform=ax_spread.transAxes,
                   fontsize=10, ha="right", va="top", color=COLOR_SECONDARY)
    ax_spread.annotate(
        f"they cross at {cross:.0f} yr", xy=(cross, s_l[int(cross - 18)]),
        xytext=(81.5, 3.5), ha="right", fontsize=11,
        arrowprops={"arrowstyle": "->", "lw": 1.0},
    )
    ax_spread.set_xlabel("Age [years]")
    ax_spread.set_ylabel("Standard deviation at 4 kHz [dB]")
    ax_spread.set_title("The spread around the median (male)",
                        pad=10)
    ax_spread.grid(color=COLOR_GRID, linestyle="-", alpha=0.4)
    ax_spread.set_axisbelow(True)
    ax_spread.legend(loc="lower left", fontsize=10)

    plt.tight_layout()
    save_figure(output_dir, "age_threshold_sex_and_spread.png")
    plt.close()


def generate_nipts_level_growth(output_dir: str) -> None:
    """ISO 1999 NIPTS against exposure level: the cut-off and the square law."""
    print("Generating nipts_level_growth.png...")
    from phonometry import hearing
    from phonometry.hearing.noise_induced_hearing_loss import NIPTS_FREQUENCIES

    freqs = NIPTS_FREQUENCIES
    levels = np.arange(75.0, 100.5, 0.5)
    med = np.array([hearing.nipts(float(x), 40.0, 0.5).median for x in levels])

    _fig, (ax_f, ax_t) = plt.subplots(1, 2, figsize=(12.5, 5.4))

    # --- Left: one curve per audiometric frequency, 40 years --------------
    colors = ["#9e9e9e", "#7f7f7f", COLOR_TERTIARY, "#ff7f0e", COLOR_SECONDARY,
              COLOR_PRIMARY]
    for k, (f, color) in enumerate(zip(freqs, colors)):
        ax_f.plot(levels, med[:, k], "-", color=color, linewidth=2.0,
                  label=f"{f:g} Hz")
        # The cut-off L0 is where the curve leaves zero.
        lift = int(np.argmax(med[:, k] > 0.0))
        ax_f.plot(levels[lift], 0.0, "o", color=color, markersize=6)
    ax_f.axhline(0.0, color=COLOR_GRID, linewidth=1.0)
    ax_f.annotate(
        "each dot is that band's cut-off $L_0$\n(93, 89, 80, 77, 75, 77 dB)",
        xy=(76.5, 1.0), xytext=(76.0, 22.0), fontsize=11,
        arrowprops={"arrowstyle": "->", "lw": 1.0},
    )
    ax_f.set_xlabel(r"$L_{\mathrm{EX,8h}}$ [dB(A)]")
    ax_f.set_ylabel("Median NIPTS $N_{50}$ [dB]")
    ax_f.set_title("ISO 1999 median NIPTS against level (40 years)",
                   pad=10)
    ax_f.grid(color=COLOR_GRID, linestyle="-", alpha=0.4)
    ax_f.set_axisbelow(True)
    ax_f.legend(loc="upper left", fontsize=10)

    # --- Right: 4 kHz alone, the duration family --------------------------
    idx = int(np.argmin(np.abs(np.asarray(freqs) - 4000.0)))
    for years, color in ((10, "#9e9e9e"), (20, "#7f7f7f"), (30, COLOR_PRIMARY),
                         (40, COLOR_SECONDARY)):
        curve = [hearing.nipts(float(x), float(years), 0.5).median[idx]
                 for x in levels]
        ax_t.plot(levels, curve, "-", color=color, linewidth=2.0,
                  label=f"{years} yr")
    forty = med[:, idx]
    for lo, hi in ((85.0, 88.0), (95.0, 98.0)):
        a = float(np.interp(lo, levels, forty))
        b = float(np.interp(hi, levels, forty))
        ax_t.annotate(
            "", xy=(hi, b), xytext=(lo, a),
            arrowprops={"arrowstyle": "<->", "lw": 1.4, "color": COLOR_FG},
        )
        ax_t.text(lo - 0.6, 0.5 * (a + b), f"+{b - a:.1f} dB", fontsize=11,
                  ha="right", va="center", color=COLOR_FG)
    ax_t.set_xlabel(r"$L_{\mathrm{EX,8h}}$ [dB(A)]")
    ax_t.set_ylabel("Median NIPTS at 4 kHz [dB]")
    ax_t.set_title("The same 3 dB is worth more the louder the job",
                   pad=10)
    ax_t.grid(color=COLOR_GRID, linestyle="-", alpha=0.4)
    ax_t.set_axisbelow(True)
    ax_t.legend(loc="upper left", fontsize=10)

    plt.tight_layout()
    save_figure(output_dir, "nipts_level_growth.png")
    plt.close()


def generate_htlan_compression(output_dir: str) -> None:
    """The ISO 1999 Formula (1) compression term over the (H, N) plane."""
    print("Generating htlan_compression.png...")
    from phonometry import hearing

    grid = np.linspace(0.0, 60.0, 121)
    hh, nn = np.meshgrid(grid, grid)
    removed = hh + nn - hearing.combine_age_and_noise(hh, nn)

    # The plotting area is a light colormap in both themes, so everything
    # drawn on top of it is pinned dark rather than following the theme.
    ink = "#1a1a1a"
    _fig, ax = plt.subplots(figsize=(8.2, 6.4))
    levels = np.arange(0.0, 31.0, 2.0)
    cs = ax.contourf(hh, nn, removed, levels=levels, cmap="YlOrRd")
    lines = ax.contour(hh, nn, removed, levels=levels[::2], colors=ink,
                       linewidths=0.7, alpha=0.55)
    ax.clabel(lines, fmt="%.0f dB", fontsize=9, colors=ink)
    ax.grid(False)
    cbar = _fig.colorbar(cs, ax=ax)
    cbar.set_label("Decibels removed by $HN/120$")
    ax.plot([0.0, 40.0], [40.0, 0.0], "--", color="#1f77b4", linewidth=2.2,
            label="$H + N$ = 40 dB")
    ax.plot(20.2, 24.8, "o", color="#1f77b4", markersize=9,
            markeredgecolor=ink)
    ax.annotate("the worked case: $H$ = 20.2, $N$ = 24.8\n45.0 dB sum → 40.8 dB "
                "HTLAN, 4.2 dB removed",
                xy=(20.2, 24.8), xytext=(2.0, 47.0), fontsize=11, color=ink,
                arrowprops={"arrowstyle": "->", "lw": 1.0, "color": ink},
                bbox={"boxstyle": "round,pad=0.35", "fc": "#ffffff",
                      "ec": ink, "alpha": 0.88})
    ax.set_xlabel("Age component $H$ (HTLA) [dB]")
    ax.set_ylabel("Noise component $N$ (NIPTS) [dB]")
    ax.set_title("ISO 1999 Formula (1): what the compression term removes",
                 pad=10)
    ax.legend(loc="lower right")
    plt.tight_layout()
    save_figure(output_dir, "htlan_compression.png")
    plt.close()


def generate_exposure_budget(output_dir: str) -> None:
    """The ISO 9612 Annex C budget: where the variance comes from, and what
    reduces it. Left, the Annex D day term by term; right, the expanded
    uncertainty against the sample count for the two sampling models."""
    print("Generating exposure_budget.png...")
    from phonometry import hearing

    tasks = [
        hearing.Task(samples=(70.0,), duration_hours=1.5,
                     label="planning/\nbreaks"),
        hearing.Task(samples=(80.1, 82.2, 79.6), duration_hours=5.0,
                     duration_range=(4.0, 6.0), label="welding"),
        hearing.Task(samples=(86.5, 92.4, 89.3, 93.2, 87.8, 86.2),
                     duration_hours=1.5, duration_range=(1.0, 2.0),
                     label="cutting/\ngrinding"),
    ]
    res = hearing.task_based_exposure(tasks, warn=False)

    _fig, (ax_b, ax_n) = plt.subplots(1, 2, figsize=(12.8, 5.4))

    # --- Left: the Annex D variance budget, term by term ------------------
    parts = [
        (r"sampling  $(c_{1\mathrm{a}}u_{1\mathrm{a}})^2$", COLOR_PRIMARY,
         [(t.c1a * t.u1a) ** 2 for t in res.tasks]),
        (r"duration  $(c_{1\mathrm{b}}u_{1\mathrm{b}})^2$", COLOR_SECONDARY,
         [(t.c1b * t.u1b) ** 2 for t in res.tasks]),
        (r"instrument  $(c_{1\mathrm{a}}u_2)^2$", COLOR_TERTIARY,
         [(t.c1a * t.u2) ** 2 for t in res.tasks]),
        (r"position  $(c_{1\mathrm{a}}u_3)^2$", "#ff7f0e",
         [(t.c1a * t.u3) ** 2 for t in res.tasks]),
    ]
    labels = [t.label for t in res.tasks] + ["whole day"]
    xs = np.arange(len(labels), dtype=float)
    bottom = np.zeros(len(labels))
    for name, color, vals in parts:
        col = np.array([*vals, float(np.sum(vals))])
        ax_b.bar(xs, col, 0.62, bottom=bottom, color=color, label=name,
                 edgecolor=COLOR_PANEL, linewidth=0.6)
        bottom += col
    total = float(bottom[-1])
    ax_b.text(xs[-1], total + 0.12,
              f"$u$ = {np.sqrt(total):.2f} dB → $U$ = {1.65 * np.sqrt(total):.1f} dB",
              ha="center", fontsize=11, fontweight="bold")
    ax_b.annotate(
        "1.5 h of the day,\n91 % of the variance",
        xy=(xs[2], bottom[2] * 0.5), xytext=(0.15, 3.0), fontsize=11,
        arrowprops={"arrowstyle": "->", "lw": 1.0},
    )
    ax_b.set_xticks(xs)
    ax_b.set_xticklabels(labels)
    ax_b.set_ylim(0.0, total + 0.9)
    ax_b.set_ylabel("Contribution to $u^2$ [dB²]")
    ax_b.set_title("ISO 9612 Annex D: the Annex C budget, term by term",
                   pad=10)
    ax_b.grid(axis="y", color=COLOR_GRID, linestyle="-", alpha=0.4)
    ax_b.set_axisbelow(True)
    ax_b.legend(loc="upper left", fontsize=10)

    # --- Right: U against the sample count, both sampling models ----------
    counts = np.arange(3, 31, dtype=float)
    u2, u3 = 1.5, 1.0
    scatter = 3.0

    def expanded(term: np.ndarray | float, inst: float = u2) -> np.ndarray:
        return 1.65 * np.sqrt(np.asarray(term) ** 2 + inst**2 + u3**2)

    ax_n.plot(counts, expanded(scatter / np.sqrt(counts)), "-",
              color=COLOR_PRIMARY, linewidth=2.0,
              label=r"task-based, $u_{1\mathrm{a}}$ (Eq. C.6)")
    ax_n.plot(counts,
              expanded(np.asarray([hearing.table_c4_contribution(int(n), scatter)
                                   for n in counts])),
              "--", color=COLOR_SECONDARY, linewidth=2.0,
              label="job-based, $c_1u_1$ (Table C.4)")
    ax_n.axhline(float(expanded(0.0)), color=COLOR_TERTIARY, linewidth=1.4,
                 linestyle=":")
    ax_n.axhline(float(expanded(0.0, 0.7)), color="#ff7f0e", linewidth=1.4,
                 linestyle=":")
    ax_n.text(3.4, float(expanded(0.0)) + 0.10,
              "floor with a personal exposimeter or class 2 meter", fontsize=10,
              ha="left", color=COLOR_TERTIARY)
    ax_n.text(3.4, float(expanded(0.0, 0.7)) + 0.10,
              "floor with a class 1 meter", fontsize=10, ha="left",
              color="#ff7f0e")
    ax_n.set_ylim(1.6, 7.0)
    ax_n.set_xlabel("Samples per task ($I$) or per group ($N$)")
    ax_n.set_ylabel("Expanded uncertainty $U$ [dB]")
    ax_n.set_title("A 3 dB sample scatter: what more samples buy",
                   pad=10)
    ax_n.grid(color=COLOR_GRID, linestyle="-", alpha=0.4)
    ax_n.set_axisbelow(True)
    ax_n.legend(loc="upper right", fontsize=10)

    plt.tight_layout()
    save_figure(output_dir, "exposure_budget.png")
    plt.close()


def generate_stoi_segment_scores(output_dir: str) -> None:
    """The time marginal of STOI, and what the front end throws away."""
    print("Generating stoi_segment_scores.png...")
    from phonometry import stoi

    fs = 10000
    rng = np.random.default_rng(11)
    n = 4 * fs
    t = np.arange(n) / fs
    freq = np.fft.rfftfreq(n, 1.0 / fs)
    spectrum = np.fft.rfft(rng.standard_normal(n))
    spectrum[(freq < 200.0) | (freq > 4000.0)] = 0.0
    carrier = np.fft.irfft(spectrum, n)
    # Deeper syllabic pauses than the guide's own example - silent between
    # words - so the 40 dB silent-frame rule actually has frames to remove.
    envelope = np.abs(np.sin(2 * np.pi * 3.5 * t)) ** 2
    envelope = np.where(envelope < 0.08, 0.0, envelope)
    clean = carrier * envelope

    noise = rng.standard_normal(n)
    noise *= np.sqrt(np.mean(clean**2) / np.mean(noise**2))
    steady = stoi(clean, clean + noise, fs)

    # The same pair with a 0.35 s dropout instead of steady noise.
    gate = np.ones(n)
    gate[int(1.6 * fs):int(1.95 * fs)] = 0.0
    dropout = stoi(clean, clean * gate, fs)

    _fig, (ax_w, ax_s) = plt.subplots(
        2, 1, figsize=(11.0, 6.6), sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.2]})

    # --- Top: the clean waveform, with the frames the 40 dB rule drops ----
    hop = 128                                   # 50 % overlap of 256 samples
    frames = np.lib.stride_tricks.sliding_window_view(clean, 256)[::hop]
    energy = np.sum(frames**2, axis=1)
    rms = np.sqrt(np.maximum(energy, 1e-24))
    keep = 20.0 * np.log10(rms / rms.max()) > -40.0
    ax_w.plot(t, clean, color=COLOR_PRIMARY, linewidth=0.5)
    for k in np.flatnonzero(~keep):
        ax_w.axvspan(k * hop / fs, (k * hop + 256) / fs,
                     color=theme_fill(COLOR_SECONDARY, ax_w), lw=0, zorder=0)
    ax_w.axvspan(1.6, 1.95, color=theme_fill(COLOR_TERTIARY, ax_w), lw=0,
                 zorder=0)
    ax_w.text(1.775, 0.92 * float(np.max(clean)), "dropout", fontsize=10,
              ha="center", color=COLOR_TERTIARY)
    dropped = int(np.count_nonzero(~keep))
    ax_w.text(0.02, 0.94, f"{dropped} of {keep.size} frames fall 40 dB below "
              "the loudest and are discarded",
              transform=ax_w.transAxes, fontsize=10, va="top",
              color=COLOR_SECONDARY)
    ax_w.set_ylabel("Clean reference")
    ax_w.set_title("The frames the 40 dB rule drops (shaded), and the segment "
                   "scores under them", pad=10)
    ax_w.grid(color=COLOR_GRID, linestyle="-", alpha=0.4)
    ax_w.set_axisbelow(True)

    # --- Bottom: the two segment-score series -----------------------------
    for res, color, style, label in (
        (steady, COLOR_PRIMARY, "-",
         f"steady noise at 0 dB (STOI = {steady.value:.3f})"),
        (dropout, COLOR_TERTIARY, "--",
         f"a 0.35 s dropout (STOI = {dropout.value:.3f})"),
    ):
        scores = np.asarray(res.segment_scores, dtype=float)
        axis = np.linspace(0.0, float(t[-1]), scores.size)
        ax_s.step(axis, scores, style, where="mid", color=color, linewidth=1.8,
                  label=label)
        ax_s.axhline(res.value, color=color, linewidth=1.2, linestyle=":")
    ax_s.set_xlabel("Time [s]")
    ax_s.set_ylabel("Segment score")
    ax_s.set_ylim(-0.1, 1.05)
    ax_s.grid(color=COLOR_GRID, linestyle="-", alpha=0.4)
    ax_s.set_axisbelow(True)
    ax_s.legend(loc="lower left", fontsize=10)

    plt.tight_layout()
    save_figure(output_dir, "stoi_segment_scores.png")
    plt.close()


def _sii_low_band_noise(frequencies: np.ndarray, level: float) -> np.ndarray:
    """Spectrum level *level* below 450 Hz, nothing above: a fan or a duct."""
    return np.where(np.asarray(frequencies) <= 450.0, level, 0.0)


def generate_sii_masking_chain(output_dir: str) -> None:
    """The ANSI S3.5 clause 5 chain the guide states in equations only."""
    print("Generating sii_masking_chain.png...")
    from phonometry import speech

    proc = speech.sii_procedure("one-third-octave")
    freqs = np.asarray(proc.frequencies)
    noise = _sii_low_band_noise(freqs, 60.0)
    res = speech.speech_intelligibility_index("normal", noise)
    e = np.asarray(res.speech_spectrum)
    d = np.asarray(res.disturbance)

    _fig, ax = plt.subplots(figsize=(10.4, 6.0))
    pos = np.arange(freqs.size, dtype=float)
    # The 30 dB window the audibility function lives inside.
    ax.fill_between(pos, d - 15.0, d + 15.0,
                    color=theme_fill(COLOR_TERTIARY, ax), zorder=0,
                    label="the 30 dB window: $D_i - 15$ to $D_i + 15$")
    ax.plot(pos, e, "o-", color=COLOR_PRIMARY, linewidth=2.0,
            label=r"speech $E_i^{\prime}$")
    ax.plot(pos, noise, "s--", color=COLOR_FG, linewidth=1.6,
            label=r"external noise $N_i^{\prime}$")
    ax.plot(pos, res.masking, "^-", color=COLOR_SECONDARY, linewidth=1.8,
            label="equivalent masking $Z_i$")
    ax.plot(pos, d, ":", color=COLOR_TERTIARY, linewidth=2.2,
            label="equivalent disturbance $D_i$")
    k = 8                                   # the 1 kHz band
    ax.annotate(
        f"at 1 kHz there is no noise in this band,\n"
        f"yet $Z_i$ = {res.masking[k]:.0f} dB: the masking is spread up\n"
        "from the low bands, not made here",
        xy=(pos[k], res.masking[k]), xytext=(pos[k] + 0.6, 54.0), fontsize=10,
        arrowprops={"arrowstyle": "->", "lw": 1.0},
    )
    ax2 = ax.twinx()
    ax2.plot(pos, res.band_audibility, "-", color="#ff7f0e", linewidth=2.4,
             alpha=0.9)
    ax2.set_ylabel("Band audibility $A_i$", color="#ff7f0e")
    ax2.set_ylim(0.0, 1.0)
    ax2.tick_params(axis="y", colors="#ff7f0e")
    ax.set_xticks(pos)
    ax.set_xticklabels([f"{f:g}" for f in freqs], rotation=45, ha="right")
    ax.set_xlabel("One-third-octave band [Hz]")
    ax.set_ylabel("Equivalent spectrum level [dB]")
    ax.set_ylim(-25.0, 72.0)
    ax.set_title(f"The clause 5 chain under a low-frequency masker "
                 f"(SII = {res.sii:.2f})", pad=10)
    ax.grid(color=COLOR_GRID, linestyle="-", alpha=0.4)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", fontsize=9)

    plt.tight_layout()
    save_figure(output_dir, "sii_masking_chain.png")
    plt.close()


def generate_sii_octave_masking_blindness(output_dir: str) -> None:
    """What the octave procedure cannot see: the upward spread of masking."""
    print("Generating sii_octave_masking_blindness.png...")
    from phonometry import speech

    levels = np.arange(45.0, 80.1, 1.0)
    styles = {
        "critical-band": ("-", COLOR_PRIMARY),
        "equally-contributing": ("-", COLOR_TERTIARY),
        "one-third-octave": ("-", COLOR_FG),
        "octave": ("--", COLOR_SECONDARY),
    }
    curves: dict[str, np.ndarray] = {}
    for method, (style, color) in styles.items():
        proc = speech.sii_procedure(method)
        freqs = np.asarray(proc.frequencies)
        curves[method] = np.array([
            speech.speech_intelligibility_index(
                proc.speech_spectrum, _sii_low_band_noise(freqs, x),
                method=method).sii
            for x in levels
        ])

    _, ax = plt.subplots(figsize=(9.6, 5.8))
    for method, (style, color) in styles.items():
        width = 2.6 if method == "octave" else 1.8
        ax.plot(levels, curves[method], style, color=color, linewidth=width,
                label=method)
    gap = float(curves["octave"][-1] - curves["one-third-octave"][-1])
    ax.annotate(
        f"{gap:.2f} index units apart",
        xy=(levels[-1], 0.5 * (curves["octave"][-1]
                               + curves["one-third-octave"][-1])),
        xytext=(62.0, 0.52), fontsize=11,
        arrowprops={"arrowstyle": "->", "lw": 1.0},
    )
    ax.annotate("", xy=(levels[-1], curves["octave"][-1]),
                xytext=(levels[-1], curves["one-third-octave"][-1]),
                arrowprops={"arrowstyle": "<->", "lw": 1.6,
                            "color": COLOR_FG})
    ax.set_xlabel("Spectrum level of the low-frequency noise below 450 Hz [dB]")
    ax.set_ylabel("SII")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("The octave procedure carries no spread of masking",
                 pad=10)
    ax.grid(color=COLOR_GRID, linestyle="-", alpha=0.4)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", fontsize=10)

    plt.tight_layout()
    save_figure(output_dir, "sii_octave_masking_blindness.png")
    plt.close()


def _sti_decay(t60: float, fs: int = 48000) -> np.ndarray:
    """White noise under an exponential decay: the guide's own test channel."""
    rng = np.random.default_rng(0)
    n = np.arange(int(max(1.0, 2.5 * t60) * fs))
    return rng.standard_normal(n.size) * np.exp(-6.9078 * n / fs / t60)


def generate_sti_mtf_curves(output_dir: str) -> None:
    """The two ways a channel destroys modulation: a corner, and a scaling."""
    print("Generating sti_mtf_curves.png...")
    from phonometry import speech
    from phonometry.speech.sti import _MOD_FREQS, _STIPA_F1, _STIPA_F2

    fs = 48000
    band = 3                     # the 1 kHz octave band
    mod = np.asarray(_MOD_FREQS)

    _fig, (ax_t, ax_n) = plt.subplots(1, 2, figsize=(12.6, 5.2))

    # --- Left: reverberation moves the low-pass corner --------------------
    for t60, color in ((0.3, COLOR_TERTIARY), (0.9, COLOR_PRIMARY),
                       (2.5, COLOR_SECONDARY)):
        res = speech.sti_from_impulse_response(_sti_decay(t60), fs)
        ax_t.semilogx(mod, res.mtf[band], "o-", color=color, linewidth=1.8,
                      markersize=5, label=f"$T_{{60}}$ = {t60:g} s")
        closed = 1.0 / np.sqrt(1.0 + (2.0 * np.pi * mod * t60 / 13.8) ** 2)
        ax_t.semilogx(mod, closed, "--", color=color, linewidth=1.2, alpha=0.8)
    for f_stipa in (_STIPA_F1[band], _STIPA_F2[band]):
        ax_t.axvline(f_stipa, color=COLOR_FG, linewidth=1.0, linestyle=":")
    ax_t.text(np.sqrt(_STIPA_F1[band] * _STIPA_F2[band]), 0.04,
              "STIPA probes only these two $F$ in this band", fontsize=10,
              ha="center", va="bottom", color=COLOR_FG)
    ax_t.set_title("Reverberation: a low-pass corner that moves",
                   pad=10)
    ax_t.legend(loc="lower left", fontsize=10)

    # --- Right: noise scales the whole curve ------------------------------
    ir = _sti_decay(0.9)
    for snr, color in ((20.0, COLOR_TERTIARY), (10.0, COLOR_PRIMARY),
                       (0.0, COLOR_SECONDARY)):
        res = speech.sti_from_impulse_response(ir, fs, snr=snr)
        ax_n.semilogx(mod, res.mtf[band], "o-", color=color, linewidth=1.8,
                      markersize=5, label=f"SNR = {snr:g} dB")
    quiet = speech.sti_from_impulse_response(ir, fs)
    ax_n.semilogx(mod, quiet.mtf[band], "--", color=COLOR_FG, linewidth=1.2,
                  label="noise-free ($T_{60}$ = 0.9 s)")
    ax_n.annotate(r"$\times\,1/(1 + 10^{-\mathrm{SNR}/10})$: flat in $F$",
                  xy=(1.25, 0.47), xytext=(5.0, 0.10), fontsize=11,
                  ha="center", arrowprops={"arrowstyle": "->", "lw": 1.0})
    ax_n.set_title("Steady noise: the same curve, scaled down",
                   pad=10)
    ax_n.legend(loc="lower left", fontsize=10)

    for ax in (ax_t, ax_n):
        ax.set_xlabel("Modulation frequency $F$ [Hz]")
        ax.set_ylabel("Modulation transfer $m$ (1 kHz band)")
        ax.set_ylim(0.0, 1.05)
        ax.set_xticks([0.63, 1.0, 2.0, 4.0, 8.0, 12.5])
        ax.set_xticklabels(["0.63", "1", "2", "4", "8", "12.5"])
        ax.xaxis.set_minor_formatter(mticker.NullFormatter())
        ax.grid(which="both", color=COLOR_GRID, linestyle="-", alpha=0.4)
        ax.set_axisbelow(True)

    plt.tight_layout()
    save_figure(output_dir, "sti_mtf_curves.png")
    plt.close()


def generate_sti_level_dependence(output_dir: str) -> None:
    """The STI of one impulse response against the level it is played at."""
    print("Generating sti_level_dependence.png...")
    from phonometry import speech
    from phonometry.speech.sti import _SPEECH_SPECTRUM_ED5

    fs = 48000
    ir = _sti_decay(0.9)
    # A quiet but real hall: ventilation-dominated ambient octave-band levels.
    ambient = np.array([45.0, 40.0, 35.0, 30.0, 28.0, 25.0, 22.0])
    shape = np.asarray(_SPEECH_SPECTRUM_ED5)
    shape_total = 10.0 * np.log10(np.sum(10.0 ** (shape / 10.0)))

    totals = np.arange(40.0, 100.5, 2.5)
    corrected = np.array([
        speech.sti_from_impulse_response(
            ir, fs, level=shape - shape_total + t, ambient=ambient).sti
        for t in totals
    ])
    plain = speech.sti_from_impulse_response(ir, fs).sti

    _, ax = plt.subplots(figsize=(9.2, 5.6))
    ax.plot(totals, corrected, "-", color=COLOR_PRIMARY, linewidth=2.2,
            label="with level= and ambient= (Tables A.2/A.3)")
    ax.axhline(plain, color=COLOR_SECONDARY, linewidth=2.0, linestyle="--",
               label=f"without them: a flat {plain:.3f}")
    peak = int(np.argmax(corrected))
    ax.plot(totals[peak], corrected[peak], "o", color=COLOR_PRIMARY,
            markersize=8)
    ax.annotate("reception threshold:\nthe speech is barely above\nthe room's "
                "own noise",
                xy=(45.0, float(np.interp(45.0, totals, corrected))),
                xytext=(41.0, 0.06), fontsize=10, ha="left",
                arrowprops={"arrowstyle": "->", "lw": 1.0})
    ax.annotate("auditory masking:\nloud low bands mask\nthe high ones",
                xy=(97.0, float(np.interp(97.0, totals, corrected))),
                xytext=(80.0, 0.30), fontsize=10,
                arrowprops={"arrowstyle": "->", "lw": 1.0})
    ax.axvline(60.0, color=COLOR_TERTIARY, linewidth=1.4, linestyle=":")
    ax.text(60.8, 0.10, "the standard's fallback level:\n60 dB(A) at 1 m "
            "from the source", fontsize=10, color=COLOR_TERTIARY, ha="left")
    ax.set_xlabel("Overall speech level at the listener [dB SPL]")
    ax.set_ylabel("STI")
    ax.set_ylim(0.0, 0.75)
    ax.set_title("One impulse response against the level it is played at",
                 pad=10)
    ax.grid(color=COLOR_GRID, linestyle="-", alpha=0.4)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", fontsize=10)

    plt.tight_layout()
    save_figure(output_dir, "sti_level_dependence.png")
    plt.close()


def generate_noise_induced_hearing_loss(output_dir: str) -> None:
    """ISO 1999 noise-induced permanent threshold shift and HTLAN combination."""
    print("Generating noise_induced_hearing_loss.png...")
    from phonometry import htlan, nipts
    from phonometry.hearing.noise_induced_hearing_loss import NIPTS_FREQUENCIES

    freqs = NIPTS_FREQUENCIES
    _fig, (ax_n, ax_h) = plt.subplots(1, 2, figsize=(12.5, 5.6))

    # --- Left: median NIPTS growth with exposure duration at 95 dB. ---
    durations = [(10, "#9e9e9e"), (20, "#7f7f7f"), (30, COLOR_PRIMARY),
                 (40, COLOR_SECONDARY)]
    for years, color in durations:
        r = nipts(95.0, years, 0.5)
        ax_n.plot(freqs, r.median, "o-", color=color, label=f"{years} yr")
    r40 = nipts(95.0, 40.0, 0.5)
    z90 = 1.2816
    ax_n.fill_between(freqs, np.maximum(r40.median - z90 * r40.spread_lower, 0.0),
                      r40.median + z90 * r40.spread_upper,
                      color=theme_fill(COLOR_SECONDARY, ax_n), zorder=0,
                      label="10-90 % band (40 yr)")
    ax_n.set_xscale("log")
    ax_n.set_xticks(list(freqs))
    ax_n.set_xticklabels([f"{f:g}" for f in freqs], rotation=45, ha="right")
    ax_n.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax_n.invert_yaxis()
    ax_n.set_xlabel("Audiometric frequency [Hz]")
    ax_n.set_ylabel("Median NIPTS [dB]")
    ax_n.set_title(r"ISO 1999 — NIPTS at $L_{\mathrm{EX,8h}}$ = 95 dB",
                   pad=10)
    ax_n.grid(which="both", color=COLOR_GRID, linestyle="-", alpha=0.4)
    ax_n.set_axisbelow(True)
    ax_n.legend(loc="lower left")

    # --- Right: HTLAN = age + noise for a 60-year-old worker, 95 dB / 30 yr. ---
    h = htlan(60, "male", 95.0, 30.0, 0.5)
    ax_h.plot(freqs, h.htla, "o-", color=COLOR_PRIMARY,
              label="Age (HTLA, ISO 7029)")
    ax_h.plot(freqs, h.nipts, "^-", color="#ff7f0e", label="Noise (NIPTS)")
    ax_h.plot(freqs, h.threshold, "s--", color=COLOR_SECONDARY,
              label="Age + noise (HTLAN)")
    ax_h.set_xscale("log")
    ax_h.set_xticks(list(freqs))
    ax_h.set_xticklabels([f"{f:g}" for f in freqs], rotation=45, ha="right")
    ax_h.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax_h.invert_yaxis()
    ax_h.set_xlabel("Audiometric frequency [Hz]")
    ax_h.set_ylabel("Hearing threshold level [dB]")
    ax_h.set_title("ISO 1999 — HTLAN (male, age 60, 95 dB / 30 yr)",
                   pad=10)
    ax_h.grid(which="both", color=COLOR_GRID, linestyle="-", alpha=0.4)
    ax_h.set_axisbelow(True)
    ax_h.legend(loc="lower left")

    plt.tight_layout()
    save_figure(output_dir, "noise_induced_hearing_loss.png")
    plt.close()
def generate_tone_prominence_assessment(output_dir: str) -> None:
    """ECMA-418-1 TNR of a fan tone against the clause 11.5 criterion."""
    print("Generating tone_prominence_assessment...")
    from phonometry import psychoacoustics

    # A 250 Hz fan tone recorded in broadband machinery noise: 10 s at 48 kHz,
    # the tone 15.1 dB above the masking noise of its critical band against a
    # 13.0 dB criterion, so it is prominent with about 2 dB to spare.
    fs = 48000
    rng = np.random.default_rng(4)
    t = np.arange(10 * fs) / fs
    x = (np.sqrt(2) * 0.011 * np.sin(2 * np.pi * 250.0 * t)
         + 0.03 * rng.standard_normal(t.size))
    res = psychoacoustics.tone_to_noise_ratio(x, fs)

    _fig, ax = plt.subplots(figsize=(10, 6))
    res.plot(ax=ax, language=_LANG)
    plt.tight_layout()
    save_figure(output_dir, "tone_prominence_assessment.svg")
    plt.close()


def generate_tone_audibility_levels(output_dir: str) -> None:
    """ISO/PAS 20065 tone levels above the critical-band masking noise."""
    print("Generating tone_audibility_levels...")
    from phonometry import psychoacoustics

    # ISO/PAS 20065 Annex E combustion-engine spectrum 1 (Delta f = 2.7 Hz):
    # the levels view of the same assessment the audibility bars summarise.
    ft = [118.4, 137.3, 158.8, 314.9, 433.4, 592.2, 629.8, 643.3, 1582.7]
    lt = [64.56, 67.96, 68.63, 68.50, 73.17, 78.31, 75.00, 79.75, 71.07]
    ls = [48.91, 49.22, 50.50, 52.85, 58.29, 59.53, 59.71, 61.98, 54.16]
    res = psychoacoustics.assess_tones(ft, lt, ls, 2.7)

    _fig, ax = plt.subplots(figsize=(10, 6))
    res.plot(ax=ax, view="levels", language=_LANG)
    plt.tight_layout()
    save_figure(output_dir, "tone_audibility_levels.svg")
    plt.close()


def generate_moore_glasberg_specific_loudness(output_dir: str) -> None:
    """ISO 532-2 specific loudness over the ERB-number (Cam) scale."""
    print("Generating moore_glasberg_specific_loudness...")
    from phonometry import psychoacoustics

    # The definitional anchor of the sone: a 1 kHz tone at 40 dB SPL, free
    # field, binaural -> N = 1 sone, with the excitation pattern spreading
    # around the tone's ERB.
    fs = 48000
    x = (np.sqrt(2) * 2e-5 * 10 ** (40 / 20)
         * np.sin(2 * np.pi * 1000 * np.arange(fs) / fs))
    res = psychoacoustics.loudness_moore_glasberg(
        x, fs, field="free", presentation="binaural")

    _fig, ax = plt.subplots(figsize=(10, 6))
    res.plot(ax=ax, language=_LANG)
    plt.tight_layout()
    save_figure(output_dir, "moore_glasberg_specific_loudness.svg")
    plt.close()


def generate_sottek_specific_tonality(output_dir: str) -> None:
    """ECMA-418-2 average specific tonality T'(z) of a 1 kHz tone."""
    print("Generating sottek_specific_tonality...")
    from phonometry import psychoacoustics

    fs = 48000
    t = np.arange(int(1.2 * fs)) / fs
    x = np.sqrt(2) * 2e-5 * 10 ** (40 / 20) * np.sin(2 * np.pi * 1000 * t)
    res = psychoacoustics.tonality_ecma(x, fs, field="free")

    _fig, ax = plt.subplots(figsize=(10, 6))
    # A supplied axes draws the specific-tonality panel alone (the tonality
    # concentrated in the tone's critical band), not the T(l) trace.
    res.plot(ax=ax, language=_LANG)
    plt.tight_layout()
    save_figure(output_dir, "sottek_specific_tonality.svg")
    plt.close()


def generate_fluctuation_strength_specific(output_dir: str) -> None:
    """Specific fluctuation strength over the Bark axis (Osses 2016 model)."""
    print("Generating fluctuation_strength_specific...")
    from phonometry import psychoacoustics

    # The reference-like stimulus: a 1 kHz tone at 70 dB SPL, fully amplitude
    # modulated at 4 Hz, where the sensation peaks.
    fs = 48000
    t = np.arange(int(2.0 * fs)) / fs
    am = (1.0 + np.sin(2 * np.pi * 4.0 * t)) * np.sin(2 * np.pi * 1000 * t)
    am = am / np.sqrt(np.mean(am**2)) * 2e-5 * 10 ** (70 / 20)
    res = psychoacoustics.fluctuation_strength(am, float(fs))

    _fig, ax = plt.subplots(figsize=(10, 6))
    res.plot(ax=ax, language=_LANG)
    plt.tight_layout()
    save_figure(output_dir, "fluctuation_strength_specific.svg")
    plt.close()


def generate_sii_hearing_loss(output_dir: str) -> None:
    """ANSI S3.5-1997 band audibility with a sloping hearing loss."""
    print("Generating sii_hearing_loss...")
    from phonometry import speech_intelligibility_index, standard_speech_spectrum

    # The same speech and office noise as the reference SII figure, heard by a
    # listener with a sloping high-frequency loss: the consonant-bearing bands
    # fall below the raised internal noise and the index drops from 0.46 to
    # 0.36.
    speech_spectrum = standard_speech_spectrum("normal")
    noise = np.array([38.0, 37.0, 36.0, 34.0, 32.0, 30.0, 28.0, 26.0, 24.0,
                      22.0, 20.0, 18.0, 16.0, 14.0, 12.0, 10.0, 8.0, 6.0])
    threshold = np.array([5.0, 5.0, 5.0, 5.0, 8.0, 10.0, 12.0, 15.0, 18.0,
                          22.0, 28.0, 35.0, 42.0, 48.0, 55.0, 60.0, 65.0, 70.0])
    res = speech_intelligibility_index(speech_spectrum, noise, threshold=threshold)

    _fig, ax = plt.subplots(figsize=(10, 6))
    res.plot(ax=ax, language=_LANG)
    plt.tight_layout()
    save_figure(output_dir, "sii_hearing_loss.svg")
    plt.close()


def generate_age_threshold_fractiles(output_dir: str) -> None:
    """ISO 7029 age-related threshold with its 10-90 % fractile band."""
    print("Generating age_threshold_fractiles...")
    from phonometry import hearing

    # A 70-year-old man at the worst-hearing decile: the median presbycusis
    # slope with the population spread around it.
    res = hearing.age_threshold(70, "male", fractile=0.9)

    _fig, ax = plt.subplots(figsize=(10, 6))
    res.plot(ax=ax, language=_LANG)
    plt.tight_layout()
    save_figure(output_dir, "age_threshold_fractiles.svg")
    plt.close()


def generate_nipts_audiogram(output_dir: str) -> None:
    """ISO 1999 NIPTS spectrum of a long, loud exposure with its spread."""
    print("Generating nipts_audiogram...")
    from phonometry import hearing

    # 40 years at an 8 h-normalised 95 dB(A), most-susceptible tenth: the
    # 4 kHz notch of noise damage.
    res = hearing.nipts(95.0, 40.0, fractile=0.9)

    _fig, ax = plt.subplots(figsize=(10, 6))
    res.plot(ax=ax, language=_LANG)
    plt.tight_layout()
    save_figure(output_dir, "nipts_audiogram.svg")
    plt.close()


def generate_stoi_band_scores(output_dir: str) -> None:
    """Per-band intermediate correlation behind a STOI index."""
    print("Generating stoi_band_scores...")
    from scipy import signal as sp_signal

    from phonometry import stoi

    # Speech-like material (band-limited noise with a 3.5 Hz syllabic
    # envelope) in a flat masker at 0 dB SNR: the low bands lose most of the
    # envelope correlation, the consonant bands keep it.
    fs = 10000
    rng = np.random.default_rng(11)
    t = np.arange(4 * fs) / fs
    b, a = sp_signal.butter(2, [200 / (fs / 2), 4000 / (fs / 2)], btype="band")
    carrier = sp_signal.lfilter(b, a, rng.standard_normal(t.size))
    clean = carrier * (0.15 + 0.85 * np.abs(np.sin(2 * np.pi * 3.5 * t)) ** 2)
    masker = rng.standard_normal(clean.size)
    gain = np.sqrt(np.mean(clean**2)) / np.sqrt(np.mean(masker**2))
    res = stoi(clean, clean + gain * masker, fs)

    _fig, ax = plt.subplots(figsize=(10, 6))
    res.plot(ax=ax, language=_LANG)
    plt.tight_layout()
    save_figure(output_dir, "stoi_band_scores.svg")
    plt.close()


def generate_sti_band_mti(output_dir: str) -> None:
    """IEC 60268-16 per-band modulation transfer index of a real room."""
    print("Generating sti_band_mti...")
    from phonometry import sti_from_impulse_response

    # A reverberant hall (T60 = 0.9 s) with a 15 dB speech-to-noise ratio:
    # the per-band MTI bars behind the single STI number and its rating.
    fs = 48000
    rng = np.random.default_rng(0)
    n = np.arange(fs)
    ir = rng.standard_normal(fs) * np.exp(-6.9078 * n / fs / 0.9)
    res = sti_from_impulse_response(ir, fs, snr=15.0)

    _fig, ax = plt.subplots(figsize=(10, 6))
    res.plot(ax=ax, language=_LANG)
    plt.tight_layout()
    save_figure(output_dir, "sti_band_mti.svg")
