#  Copyright (c) 2020. Jose M. Requena-Plens
"""
Octave-Band and Fractional Octave-Band filter for signals in the time domain.
Implementation according to ANSI s1.11-2004 and IEC 61260-1-2014.
"""

import warnings
from typing import List, Optional, Tuple, Union, cast

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

# Use non-interactive backend for plots
matplotlib.use("Agg")

# Public methods
__all__ = ["octavefilter", "getansifrequencies", "normalizedfreq"]


def octavefilter(
    x: Union[List[float], np.ndarray],
    fs: int,
    fraction: float = 1,
    order: int = 6,
    limits: Optional[List[float]] = None,
    show: bool = False,
    sigbands: bool = False,
    plot_file: Optional[str] = None,
) -> Union[Tuple[np.ndarray, List[float]], Tuple[np.ndarray, List[float], List[np.ndarray]]]:
    """
    Filter a signal with octave or fractional octave filter bank.

    This method uses a Butterworth filter with Second-Order Sections (SOS)
    coefficients. To obtain the correct coefficients, automatic subsampling
    is applied to the signal in each filtered band.

    Multichannel support: If x is 2D (channels, samples), each channel is filtered.

    :param x: Input signal (1D array or 2D array [channels, samples]).
    :type x: Union[List[float], np.ndarray]
    :param fs: Sample rate in Hz.
    :type fs: int
    :param fraction: Bandwidth 'b'. Examples: 1/3-octave b=3, 1-octave b=1, 2/3-octave b=1.5. Default: 1.
    :type fraction: float
    :param order: Order of Butterworth filter. Default: 6.
    :type order: int
    :param limits: Minimum and maximum limit frequencies [f_min, f_max]. Default [12, 20000].
    :type limits: Optional[List[float]]
    :param show: If True, plot and show the filter response.
    :type show: bool
    :param sigbands: If True, also return the signal in the time domain divided into bands.
    :type sigbands: bool
    :param plot_file: Path to save the filter response plot.
    :type plot_file: Optional[str]
    :return: A tuple containing (SPL_array, Frequencies_list) if sigbands is False,
             or (SPL_array, Frequencies_list, List_of_filtered_signals) if sigbands is True.
    :rtype: Union[Tuple[np.ndarray, List[float]], Tuple[np.ndarray, List[float], List[np.ndarray]]]
    """

    if limits is None:
        limits = [12, 20000]

    # Convert input to numpy array
    x_proc = _typesignal(x)

    # Handle multichannel detection
    is_multichannel = x_proc.ndim > 1
    if not is_multichannel:
        x_proc = x_proc[np.newaxis, :]  # Standardize to 2D

    num_channels = x_proc.shape[0]

    # Generate band center and edge frequencies
    freq, freq_d, freq_u = _genfreqs(limits, fraction, fs)
    num_bands = len(freq)

    # Calculate required downsampling factor for stability
    factor = _downsamplingfactor(freq_u, fs)

    # Design the Butterworth filter bank
    sos = _buttersosfilter(freq, freq_d, freq_u, fs, order, factor, show, plot_file)

    # Process signal across all bands and channels
    spl, xb = _process_bands(x_proc, num_channels, num_bands, factor, sos, sigbands)

    # Format output based on input dimensionality
    if not is_multichannel:
        spl = spl[0]
        if sigbands and xb is not None:
            xb = [band[0] for band in xb]

    if sigbands and xb is not None:
        return spl, freq, xb
    else:
        return spl, freq


def _process_bands(
    x_proc: np.ndarray,
    num_channels: int,
    num_bands: int,
    factor: np.ndarray,
    sos: List[np.ndarray],
    sigbands: bool,
) -> Tuple[np.ndarray, Optional[List[np.ndarray]]]:
    """Process signal through each frequency band."""
    spl = np.zeros([num_channels, num_bands])
    xb: Optional[List[np.ndarray]] = [np.array([]) for _ in range(num_bands)] if sigbands else None

    for idx in range(num_bands):
        for ch in range(num_channels):
            # Resample signal for this specific band to improve accuracy
            sd = signal.resample(x_proc[ch], round(len(x_proc[ch]) / factor[idx]))
            y = signal.sosfilt(sos[idx], sd)

            # Standard reference pressure for SPL calculation: 2e-5 Pa
            spl[ch, idx] = 20 * np.log10(np.max([np.std(y), np.finfo(float).eps]) / 2e-5)

            if sigbands and xb is not None:
                # Restore original length
                y_resampled = _resample_to_length(y, int(factor[idx]), x_proc.shape[1])
                if ch == 0:
                    xb[idx] = np.zeros([num_channels, x_proc.shape[1]])
                xb[idx][ch] = y_resampled
    return spl, xb


def _resample_to_length(y: np.ndarray, factor: int, target_length: int) -> np.ndarray:
    """Resample signal and ensure the output matches target_length exactly."""
    y_resampled = signal.resample_poly(y, factor, 1)
    if len(y_resampled) > target_length:
        y_resampled = y_resampled[:target_length]
    elif len(y_resampled) < target_length:
        y_resampled = np.pad(y_resampled, (0, target_length - len(y_resampled)))
    return cast(np.ndarray, y_resampled)


def _typesignal(x: Union[List[float], np.ndarray, Tuple[float, ...]]) -> np.ndarray:
    """Ensure signal is a numpy array."""
    if isinstance(x, np.ndarray):
        return x
    return cast(np.ndarray, np.atleast_1d(np.array(x)))


def _buttersosfilter(
    freq: List[float],
    freq_d: List[float],
    freq_u: List[float],
    fs: int,
    order: int,
    factor: np.ndarray,
    show: bool = False,
    plot_file: Optional[str] = None,
) -> List[np.ndarray]:
    """Generate SOS coefficients for the filter bank."""
    sos = [np.array([]) for _ in range(len(freq))]

    for idx, (lower, upper) in enumerate(zip(freq_d, freq_u)):
        fsd = fs / factor[idx]
        sos[idx] = signal.butter(
            N=order, Wn=np.array([lower, upper]) / (fsd / 2), btype="bandpass", analog=False, output="sos"
        )

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
    plot_file: Optional[str] = None,
) -> None:
    """Visualize filter bank frequency response."""
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


def _genfreqs(limits: List[float], fraction: float, fs: int) -> Tuple[List[float], List[float], List[float]]:
    """Determine band frequencies within limits."""
    freq, freq_d, freq_u = getansifrequencies(fraction, limits)
    return _deleteouters(freq, freq_d, freq_u, fs)


def normalizedfreq(fraction: int) -> List[float]:
    """Get standardized IEC center frequencies."""
    predefined = {
        1: [16, 31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000],
        3: [
            12.5,
            16,
            20,
            25,
            31.5,
            40,
            50,
            63,
            80,
            100,
            125,
            160,
            200,
            250,
            315,
            400,
            500,
            630,
            800,
            1000,
            1250,
            1600,
            2000,
            2500,
            3150,
            4000,
            5000,
            6300,
            8000,
            10000,
            12500,
            16000,
            20000,
        ],
    }
    if fraction not in predefined:
        raise ValueError("Normalized frequencies only available for fraction=1 or 3")
    return predefined[fraction]


def _deleteouters(
    freq: List[float], freq_d: List[float], freq_u: List[float], fs: int
) -> Tuple[List[float], List[float], List[float]]:
    """Remove bands exceeding the Nyquist frequency."""
    freq_arr = np.array(freq)
    freq_d_arr = np.array(freq_d)
    freq_u_arr = np.array(freq_u)

    idx = np.nonzero(freq_u_arr > fs / 2)[0]
    if len(idx) > 0:
        warnings.warn("Low sampling rate: frequencies above fs/2 removed", stacklevel=2)
        freq_arr = np.delete(freq_arr, idx)
        freq_d_arr = np.delete(freq_d_arr, idx)
        freq_u_arr = np.delete(freq_u_arr, idx)

    return freq_arr.tolist(), freq_d_arr.tolist(), freq_u_arr.tolist()


def getansifrequencies(
    fraction: float,
    limits: Optional[List[float]] = None,
) -> Tuple[List[float], List[float], List[float]]:
    """Calculate frequencies according to ANSI/IEC standards."""
    if limits is None:
        limits = [12, 20000]

    g = 10 ** (3 / 10)
    fr = 1000

    x = _init_index(limits[0], fr, g, fraction)
    freq = np.array([_ratio(g, x, fraction) * fr])

    freq_x = freq[0]
    while freq_x * _bandedge(g, fraction) < limits[1]:
        x += 1
        freq_x = _ratio(g, x, fraction) * fr
        freq = np.append(freq, freq_x)

    freq_d = freq / _bandedge(g, fraction)
    freq_u = freq * _bandedge(g, fraction)

    return freq.tolist(), freq_d.tolist(), freq_u.tolist()


def _init_index(f: float, fr: float, g: float, b: float) -> int:
    """Calculate starting index for band generation."""
    if round(b) % 2:
        return int(np.round((b * np.log(f / fr) + 30 * np.log(g)) / np.log(g)))
    return int(np.round((2 * b * np.log(f / fr) + 59 * np.log(g)) / (2 * np.log(g))))


def _ratio(g: float, x: int, b: float) -> float:
    """Calculate ratio for center frequency."""
    if round(b) % 2:
        return float(g ** ((x - 30) / b))
    return float(g ** ((2 * x - 59) / (2 * b)))


def _bandedge(g: float, b: float) -> float:
    """Calculate band-edge ratio."""
    return float(g ** (1 / (2 * b)))


def _downsamplingfactor(freq: List[float], fs: int) -> np.ndarray:
    """Compute optimal downsampling factors for filter stability."""
    guard = 0.10
    factor = (np.floor((fs / (2 + guard)) / np.array(freq))).astype("int")
    return cast(np.ndarray, np.clip(factor, 1, 50))
