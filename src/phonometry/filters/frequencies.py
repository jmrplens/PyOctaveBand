#  Copyright (c) 2026. Jose Manuel Requena Plens
"""
Frequency calculation logic according to ANSI/IEC standards.
"""

from __future__ import annotations

import warnings
from functools import lru_cache

import numpy as np

from .._internal.warnings import PhonometryWarning


def nominal_frequencies(
    fraction: float,
    limits: list[float] | None = None,
) -> tuple[list[float], list[float], list[float], list[str]]:
    """
    Calculate frequencies according to ANSI/IEC standards.

    :param fraction: Bandwidth fraction (e.g., 1, 3).
    :param limits: [f_min, f_max] limits.
    :return: Tuple of (center_freqs, lower_edges, upper_edges, nominal_labels).
    """
    if limits is None:
        limits = [12, 20000]

    g = 10 ** (3 / 10)
    fr = 1000

    x = _initindex(limits[0], fr, g, fraction)
    freq_list = [_ratio(g, x, fraction) * fr]

    while freq_list[-1] * _bandedge(g, fraction) < limits[1]:
        x += 1
        freq_list.append(_ratio(g, x, fraction) * fr)
    freq = np.array(freq_list)

    freq_d = freq / _bandedge(g, fraction)
    freq_u = freq * _bandedge(g, fraction)

    labels = [_format_nominal_freq(_nominal_freq_for_band(f, fraction)) for f in freq.tolist()]
    return freq.tolist(), freq_d.tolist(), freq_u.tolist(), labels


def _initindex(f: float, fr: float, g: float, b: float) -> int:
    """
    Calculate starting index for band generation.

    :param f: Frequency.
    :param fr: Reference frequency.
    :param g: Base ratio.
    :param b: Bandwidth fraction.
    :return: Index integer.
    """
    if round(b) % 2:
        return int(np.round((b * np.log(f / fr) + 30 * np.log(g)) / np.log(g)))
    return int(np.round((2 * b * np.log(f / fr) + 59 * np.log(g)) / (2 * np.log(g))))


def _ratio(g: float, x: int, b: float) -> float:
    """
    Calculate ratio for center frequency.

    :param g: Base ratio.
    :param x: Index.
    :param b: Bandwidth fraction.
    :return: Frequency ratio.
    """
    if round(b) % 2:
        return float(g ** ((x - 30) / b))
    return float(g ** ((2 * x - 59) / (2 * b)))


def _bandedge(g: float, b: float) -> float:
    """
    Calculate band-edge ratio.

    :param g: Base ratio.
    :param b: Bandwidth fraction.
    :return: Edge ratio.
    """
    return float(g ** (1 / (2 * b)))


def _deleteouters(
    freq: list[float], freq_d: list[float], freq_u: list[float], fs: int
) -> tuple[list[float], list[float], list[float]]:
    """
    Remove bands exceeding the Nyquist frequency.

    :param freq: Center frequencies.
    :param freq_d: Lower edges.
    :param freq_u: Upper edges.
    :param fs: Sample rate.
    :return: Filtered (center, lower, upper) frequencies.
    """
    freq_arr = np.array(freq)
    freq_d_arr = np.array(freq_d)
    freq_u_arr = np.array(freq_u)

    idx = np.nonzero(freq_u_arr > fs / 2)[0]
    if len(idx) > 0:
        warnings.warn(
            "Low sampling rate: frequencies above fs/2 removed",
            PhonometryWarning,
            stacklevel=3,
        )
        freq_arr = np.delete(freq_arr, idx)
        freq_d_arr = np.delete(freq_d_arr, idx)
        freq_u_arr = np.delete(freq_u_arr, idx)

    return freq_arr.tolist(), freq_d_arr.tolist(), freq_u_arr.tolist()


def _genfreqs(
    limits: list[float], fraction: float, fs: int
) -> tuple[list[float], list[float], list[float], list[str]]:
    """
    Determine band frequencies within limits.

    :param limits: [f_min, f_max].
    :param fraction: Bandwidth fraction.
    :param fs: Sample rate.
    :return: Tuple of center, lower, upper frequencies, and nominal labels.
    """
    freq, freq_d, freq_u, labels = nominal_frequencies(fraction, limits)
    freq, freq_d, freq_u = _deleteouters(freq, freq_d, freq_u, fs)
    # _deleteouters only removes trailing bands above Nyquist, so slice labels
    labels = labels[: len(freq)]
    return freq, freq_d, freq_u, labels


def _iec_e3_round(f: float) -> float:
    """IEC 61260-1 Annex E.3: 3 sig figs if MSD 1–4, 2 sig figs if MSD 5–9."""
    if f <= 0:
        return f
    exponent = int(np.floor(np.log10(f)))
    msd = f / (10.0 ** exponent)
    step = 10.0 ** (exponent - 2) if msd < 5.0 else 10.0 ** (exponent - 1)
    return round(f / step) * step


@lru_cache(maxsize=4)
def _extended_preferred(frac: int) -> list[float]:
    """Cached expansion of the IEC preferred frequency table across decades."""
    base = normalized_frequencies(frac)
    return [f * (10 ** d) for d in range(-3, 4) for f in base]


def _nominal_freq_for_band(exact_freq: float, fraction: float) -> float:
    """Return IEC 61260-1 nominal frequency (float) for an exact mid-band frequency.

    For standard fractions (1, 3), snaps to the IEC preferred table via
    ``normalized_frequencies``.  For non-standard fractions, falls back to Annex E.3
    significant-figure rounding (``_iec_e3_round``).
    """
    frac = round(fraction)
    if np.isclose(fraction, frac) and frac in (1, 3):
        extended = _extended_preferred(frac)
        return min(extended, key=lambda f: abs(np.log(f / exact_freq)))
    return _iec_e3_round(exact_freq)


def _infer_band_fraction(frequency: np.ndarray) -> int:
    """Octave (1) or one-third-octave (3) from adjacent band-centre ratios.

    Distinguishes the two IEC 61260 band structures by the ratio of adjacent
    mid-band centres (2 for octaves, 2**(1/3) ~ 1.26 for one-third octaves).
    Fewer than two bands cannot be told apart, so octave (1) is returned as the
    conventional default. Shared by the room-acoustics plot renderer and the
    report fiche so they classify a band set identically.
    """
    freq = np.asarray(frequency, dtype=np.float64)
    if freq.size < 2:
        return 1
    return 1 if float(freq[1] / freq[0]) > 1.5 else 3


def _format_nominal_freq(f: float) -> str:
    """Format a nominal frequency as a human-readable label string."""
    if f >= 1000:
        return f"{f / 1000:g}k"
    return f"{f:g}"


def normalized_frequencies(fraction: int) -> list[float]:
    """
    Get standardized IEC center frequencies.

    :param fraction: 1 or 3 (Octave or 1/3 Octave).
    :return: List of standard frequencies.
    """
    predefined = {
        1: [16, 31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000],
        3: [
            12.5, 16, 20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500,
            630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000,
            12500, 16000, 20000,
        ],
    }
    if fraction not in predefined:
        raise ValueError("Normalized frequencies only available for fraction=1 or 3")
    return predefined[fraction]
