#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Band-filter design and visualization: the SOS designer behind the octave and
fractional-octave banks, and the response plot the banks draw with it.
"""

from __future__ import annotations

import numpy as np
from scipy import signal

from .frequencies import _format_nominal_freq

# Fewest standard-frequency ticks that must land inside the plotted range
# before _showfilter falls back to plain powers-of-10 decade ticks.
_MIN_STANDARD_TICKS = 3


def _cheby2_transition_ratio(order: int, attenuation: float) -> float:
    """Ratio between stopband-edge and -3 dB frequencies for a Chebyshev II prototype."""
    # Below ~3.01 dB the -3 dB point does not exist (arccosh argument < 1).
    if attenuation <= 10 * np.log10(2):
        msg = "cheby2 'attenuation' must be greater than 3.01 dB."
        raise ValueError(msg)
    eps_term = np.sqrt(10 ** (attenuation / 10) - 1)
    return float(np.cosh(np.arccosh(eps_term) / order))


def _cheby2_stopband_edges(
    fd: float, fu: float, order: int, attenuation: float, fs: float | None = None
) -> tuple[float, float]:
    """Map desired -3 dB band edges (fd, fu) to the Chebyshev II stopband
    edges that scipy expects as ``Wn``.

    Uses the analog lowpass-to-bandpass transform: the stopband keeps the
    geometric center (f1*f2 = fd*fu) and widens the bandwidth by the
    prototype transition ratio.

    When ``fs`` is given, the mapping is done in the pre-warped analog
    domain of the bilinear transform. Decimated bands sit at a large
    fraction of Nyquist, where the tan() frequency warping would otherwise
    shift the -3 dB points well away from the band edges.
    """
    ratio = _cheby2_transition_ratio(order, attenuation)

    if fs is not None:
        ad = (fs / np.pi) * np.tan(np.pi * fd / fs)
        au = (fs / np.pi) * np.tan(np.pi * fu / fs)
    else:
        ad, au = fd, fu

    bw_stop = (au - ad) * ratio
    a1 = (-bw_stop + np.sqrt(bw_stop**2 + 4 * ad * au)) / 2
    a2 = a1 + bw_stop

    if fs is not None:
        f1 = (fs / np.pi) * np.arctan(np.pi * a1 / fs)
        f2 = (fs / np.pi) * np.arctan(np.pi * a2 / fs)
        return float(f1), float(f2)
    return float(a1), float(a2)


def _cheby2_headroom(fraction: float, order: int, attenuation: float) -> float:
    """Headroom factor ``f2_stop / f_upper_edge`` needed above the band's upper
    edge. Constant across bands of the same fraction (bands are geometric).
    """
    g = 10 ** (3 / 10)
    edge = g ** (1 / (2 * fraction))
    fd, fu = 1.0 / edge, edge  # normalized band around fc=1
    _, f2 = _cheby2_stopband_edges(fd, fu, order, attenuation)
    return float(f2 / fu)


def _design_sos_filter(
    freq: list[float],
    freq_d: list[float],
    freq_u: list[float],
    fs: int,
    order: int,
    factor: np.ndarray,
    filter_type: str,
    ripple: float,
    attenuation: float,
    show: bool = False,
    plot_file: str | None = None,
) -> list[np.ndarray]:
    """Generate SOS coefficients for the filter bank.

    :param freq: Center frequencies.
    :param freq_d: Lower edge frequencies.
    :param freq_u: Upper edge frequencies.
    :param fs: Original sample rate.
    :param order: Filter order.
    :param factor: Downsampling factors per band.
    :param filter_type: Type of filter.
    :param ripple: Passband ripple (dB).
    :param attenuation: Stopband attenuation (dB). For ``cheby2`` this is also
        the equiripple deep-stopband floor, so it must be >= 70 dB to meet the
        IEC 61260-1:2014 class 1 deep-stopband limit (the bank default is 72).
    :param show: If True, plot response.
    :param plot_file: Path to save plot.
    :return: List of SOS coefficient arrays.
    """
    sos = [np.array([]) for _ in range(len(freq))]

    for idx, (lower, upper) in enumerate(zip(freq_d, freq_u, strict=True)):
        fsd = fs / factor[idx]
        wn = np.array([lower, upper]) / (fsd / 2)

        if filter_type == "butter":
            sos[idx] = signal.butter(N=order, Wn=wn, btype="bandpass", output="sos")
        elif filter_type == "cheby1":
            sos[idx] = signal.cheby1(
                N=order, rp=ripple, Wn=wn, btype="bandpass", output="sos"
            )
        elif filter_type == "cheby2":
            # Wn in cheby2 is the STOPBAND edge; map the desired -3 dB band
            # edges to stopband edges so the passband matches the ANSI band.
            f1_stop, f2_stop = _cheby2_stopband_edges(
                lower, upper, order, attenuation, fsd
            )
            # No Nyquist clamp is needed here. Called with a sample rate, the
            # mapping returns through (fsd/pi)*arctan(...), and arctan is
            # bounded by pi/2, so f2_stop < fsd/2 strictly however wide the
            # prototype's transition or however close the band sits to Nyquist.
            nyq = fsd / 2
            wn_stop = np.array([f1_stop, f2_stop]) / nyq
            sos[idx] = signal.cheby2(
                N=order, rs=attenuation, Wn=wn_stop, btype="bandpass", output="sos"
            )
        elif filter_type == "ellip":
            sos[idx] = signal.ellip(
                N=order,
                rp=ripple,
                rs=attenuation,
                Wn=wn,
                btype="bandpass",
                output="sos",
            )
        elif filter_type == "bessel":
            # norm="mag" places the -3 dB point at Wn (band edges);
            # norm="phase" would shift it to ~-10 dB at the edges.
            sos[idx] = signal.bessel(
                N=order, Wn=wn, btype="bandpass", norm="mag", output="sos"
            )

    if show or plot_file:
        _showfilter(sos, freq, freq_u, freq_d, fs, factor, show, plot_file)

    return sos


def _showfilter(
    sos: list[np.ndarray],
    freq: list[float],
    freq_u: list[float],
    freq_d: list[float],
    fs: int,
    factor: np.ndarray,
    show: bool = False,
    plot_file: str | None = None,
    close: bool = True,
) -> None:
    """Visualize filter bank frequency response.

    :param sos: List of SOS coefficients.
    :param freq: Center frequencies.
    :param freq_u: Upper edges.
    :param freq_d: Lower edges.
    :param fs: Original sample rate.
    :param factor: Downsampling factors.
    :param show: If True, show the plot.
    :param plot_file: Path to save the plot.
    :param close: Close the figure before returning; pass ``False`` when the
        caller post-processes and saves the live figure itself.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        msg = "Plotting requires matplotlib. Install it with: pip install phonometry[plot]"
        raise ImportError(msg) from exc

    wn = 8192
    w = np.zeros([wn, len(freq)])
    h: np.ndarray = np.zeros([wn, len(freq)], dtype=np.complex128)

    for idx in range(len(freq)):
        fsd = fs / factor[idx]
        w[:, idx], h[:, idx] = signal.sosfreqz(sos[idx], worN=wn, whole=False, fs=fsd)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.semilogx(
        w, 20 * np.log10(abs(h) + np.finfo(float).eps), color="#1f77b4", linewidth=1.2
    )
    # The -3 dB line is a reference, so it is drawn quieter than the responses
    # it is read against -- but quieter by *colour*, not by opacity. Half
    # opacity of this red is a soft pink over the white page and lands within
    # a few levels of the dark one, so the same call reads on one theme and
    # vanishes on the other; theme_line keeps the softness where there is room
    # for it and raises the colour where there is not.
    from .._plot.common import theme_line

    ax.axhline(
        -3,
        color=theme_line("#d62728", ax, quiet=0.5),
        linestyle="--",
        linewidth=1,
        label="-3 dB",
    )

    ax.set_title("Filter Bank Frequency Response", pad=15)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Amplitude [dB]")
    ax.grid(which="major", color="#e0e0e0", linestyle="-")
    ax.grid(which="minor", color="#e0e0e0", linestyle=":", alpha=0.4)

    plt.xlim(freq_d[0] * 0.8, freq_u[-1] * 1.2)
    plt.ylim(-4, 1)

    # Dynamic ticks based on range
    f_min = freq_d[0] * 0.8
    f_max = freq_u[-1] * 1.2

    # Standard frequencies for ticks
    all_ticks = [
        0.1,
        0.2,
        0.5,
        1,
        2,
        5,
        10,
        20,
        50,
        100,
        200,
        500,
        1000,
        2000,
        5000,
        10000,
        20000,
        50000,
        100000,
        200000,
        500000,
    ]
    xticks = [f for f in all_ticks if f_min <= f <= f_max]

    # If too few ticks, use simpler logic
    if len(xticks) < _MIN_STANDARD_TICKS:
        # Fallback to powers of 10
        p_min = int(np.floor(np.log10(f_min)))
        p_max = int(np.ceil(np.log10(f_max)))
        xticks = [10**p for p in range(p_min, p_max + 1)]

    xticklabels = [_format_nominal_freq(f) for f in xticks]

    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)

    if plot_file:
        plt.savefig(plot_file, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    if close:
        plt.close(fig)
