#  Copyright (c) 2026. Jose M. Requena-Plens
"""
Frequency calculation logic according to ANSI/IEC standards.
"""

from __future__ import annotations

import warnings
from typing import List, Tuple

import numpy as np


def getansifrequencies(
    fraction: float,
    limits: List[float] | None = None,
) -> Tuple[List[float], List[float], List[float]]:
    """
    Calculate frequencies according to ANSI/IEC standards.

    :param fraction: Bandwidth fraction (e.g., 1, 3).
    :param limits: [f_min, f_max] limits.
    :return: Tuple of (center_freqs, lower_edges, upper_edges).
    """
    if limits is None:
        limits = [12, 20000]

    g = 10 ** (3 / 10)
    fr = 1000

    x = _initindex(limits[0], fr, g, fraction)
    freq = np.array([_ratio(g, x, fraction) * fr])

    freq_x = freq[0]
    while freq_x * _bandedge(g, fraction) < limits[1]:
        x += 1
        freq_x = _ratio(g, x, fraction) * fr
        freq = np.append(freq, freq_x)

    freq_d = freq / _bandedge(g, fraction)
    freq_u = freq * _bandedge(g, fraction)

    return freq.tolist(), freq_d.tolist(), freq_u.tolist()


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
    freq: List[float], freq_d: List[float], freq_u: List[float], fs: int
) -> Tuple[List[float], List[float], List[float]]:
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
        warnings.warn("Low sampling rate: frequencies above fs/2 removed", stacklevel=3)
        freq_arr = np.delete(freq_arr, idx)
        freq_d_arr = np.delete(freq_d_arr, idx)
        freq_u_arr = np.delete(freq_u_arr, idx)

    return freq_arr.tolist(), freq_d_arr.tolist(), freq_u_arr.tolist()


def _genfreqs(limits: List[float], fraction: float, fs: int) -> Tuple[List[float], List[float], List[float]]:
    """
    Determine band frequencies within limits.

    :param limits: [f_min, f_max].
    :param fraction: Bandwidth fraction.
    :param fs: Sample rate.
    :return: Tuple of center, lower, and upper frequencies.
    """
    freq, freq_d, freq_u = getansifrequencies(fraction, limits)
    return _deleteouters(freq, freq_d, freq_u, fs)


def normalizedfreq(fraction: int) -> List[float]:
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