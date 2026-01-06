#  Copyright (c) 2026. Jose M. Requena-Plens
"""
Filter design and visualization for pyoctaveband.
"""

from __future__ import annotations

from typing import List

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal


def _design_sos_filter(
    freq: List[float],
    freq_d: List[float],
    freq_u: List[float],
    fs: int,
    order: int,
    factor: np.ndarray,
    filter_type: str,
    ripple: float,
    attenuation: float,
    show: bool = False,
    plot_file: str | None = None,
) -> List[np.ndarray]:
    """
    Generate SOS coefficients for the filter bank.

    :param freq: Center frequencies.
    :param freq_d: Lower edge frequencies.
    :param freq_u: Upper edge frequencies.
    :param fs: Original sample rate.
    :param order: Filter order.
    :param factor: Downsampling factors per band.
    :param filter_type: Type of filter.
    :param ripple: Passband ripple (dB).
    :param attenuation: Stopband attenuation (dB).
    :param show: If True, plot response.
    :param plot_file: Path to save plot.
    :return: List of SOS coefficient arrays.
    """
    sos = [np.array([]) for _ in range(len(freq))]

    for idx, (lower, upper) in enumerate(zip(freq_d, freq_u)):
        fsd = fs / factor[idx]
        wn = np.array([lower, upper]) / (fsd / 2)
        
        if filter_type == "butter":
            sos[idx] = signal.butter(N=order, Wn=wn, btype="bandpass", output="sos")
        elif filter_type == "cheby1":
            sos[idx] = signal.cheby1(N=order, rp=ripple, Wn=wn, btype="bandpass", output="sos")
        elif filter_type == "cheby2":
            sos[idx] = signal.cheby2(N=order, rs=attenuation, Wn=wn, btype="bandpass", output="sos")
        elif filter_type == "ellip":
            sos[idx] = signal.ellip(N=order, rp=ripple, rs=attenuation, Wn=wn, btype="bandpass", output="sos")
        elif filter_type == "bessel":
            sos[idx] = signal.bessel(N=order, Wn=wn, btype="bandpass", norm="phase", output="sos")

    if show or plot_file:
        _showfilter(sos, freq, freq_u, freq_d, fs, factor, show, plot_file)

    return sos

def _showfilter(
    sos: List[np.ndarray],
    freq: List[float],
    freq_u: List[float],
    freq_d: List[float],
    fs: int,
    factor: np.ndarray,
    show: bool = False,
    plot_file: str | None = None,
) -> None:
    """
    Visualize filter bank frequency response.

    :param sos: List of SOS coefficients.
    :param freq: Center frequencies.
    :param freq_u: Upper edges.
    :param freq_d: Lower edges.
    :param fs: Original sample rate.
    :param factor: Downsampling factors.
    :param show: If True, show the plot.
    :param plot_file: Path to save the plot.
    """
    wn = 8192
    w = np.zeros([wn, len(freq)])
    h: np.ndarray = np.zeros([wn, len(freq)], dtype=np.complex128)

    for idx in range(len(freq)):
        fsd = fs / factor[idx]
        w[:, idx], h[:, idx] = signal.sosfreqz(sos[idx], worN=wn, whole=False, fs=fsd)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.semilogx(w, 20 * np.log10(abs(h) + np.finfo(float).eps), color="#1f77b4", linewidth=1.2)
    ax.axhline(-3, color="#d62728", linestyle="--", alpha=0.5, linewidth=1, label="-3 dB")

    ax.set_title("Filter Bank Frequency Response", fontweight="bold", pad=15)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Amplitude [dB]")
    ax.grid(which="major", color="#e0e0e0", linestyle="-")
    ax.grid(which="minor", color="#e0e0e0", linestyle=":", alpha=0.4)

    plt.xlim(freq_d[0] * 0.8, freq_u[-1] * 1.2)
    plt.ylim(-4, 1)

    xticks = [16, 31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]
    xticklabels = ["16", "31.5", "63", "125", "250", "500", "1k", "2k", "4k", "8k", "16k"]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)

    if plot_file:
        plt.savefig(plot_file, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)