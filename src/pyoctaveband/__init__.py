#  Copyright (c) 2020. Jose M. Requena-Plens
"""
Octave-Band and Fractional Octave-Band filter.
"""

from typing import List, Optional, Tuple, Union, cast

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

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

    This method uses a Butterworth filter with Second-Order Sections coefficients.
    To obtain the correct coefficients, a subsampling is applied to the signal
    in each filtered band.

    Multichannel support: If x is 2D (channels, samples), each channel is filtered.

    :param x: Input signal.
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
             SPL_array shape: (channels, bands) or (bands,) for 1D input.
             List_of_filtered_signals: List of arrays, each with shape (channels, samples)
             or (samples,) for 1D input.
    :rtype: Union[Tuple[np.ndarray, List[float]], Tuple[np.ndarray, List[float], List[np.ndarray]]]
    """

    if limits is None:
        limits = [12, 20000]

    # Ensure signal is in a suitable format (numpy array)
    x_proc = _typesignal(x)

    # Handle multichannel
    is_multichannel = x_proc.ndim > 1
    if not is_multichannel:
        x_proc = x_proc[np.newaxis, :]  # Add channel dimension

    num_channels = x_proc.shape[0]

    # Generate frequency array
    freq, freq_d, freq_u = _genfreqs(limits, fraction, fs)
    num_bands = len(freq)

    # Calculate the downsampling factor (array of integers with size [freq])
    factor = _downsamplingfactor(freq_u, fs)

    # Get SOS filter coefficients (3D - matrix with size: [freq,order,6])
    sos = _buttersosfilter(freq, freq_d, freq_u, fs, order, factor, show, plot_file)

    # Create array with SPL for each frequency band and channel
    spl = np.zeros([num_channels, num_bands])
    xb: Optional[List[np.ndarray]] = [np.array([]) for _ in range(num_bands)] if sigbands else None

    for idx in range(num_bands):
        for ch in range(num_channels):
            # Downsampling to improve filter coefficients accuracy at low frequencies
            sd = signal.resample(x_proc[ch], round(len(x_proc[ch]) / factor[idx]))
            y = signal.sosfilt(sos[idx], sd)

            # Calculate Sound Pressure Level (SPL)
            # Using standard reference pressure of 2e-5 Pa
            spl[ch, idx] = 20 * np.log10(np.max([np.std(y), np.finfo(float).eps]) / 2e-5)

            if sigbands and xb is not None:
                # Resample back to original sample rate
                y_resampled = signal.resample_poly(y, factor[idx], 1)
                # Ensure the length matches original (resample_poly might differ by 1)
                if len(y_resampled) > x_proc.shape[1]:
                    y_resampled = y_resampled[: x_proc.shape[1]]
                elif len(y_resampled) < x_proc.shape[1]:
                    y_resampled = np.pad(y_resampled, (0, x_proc.shape[1] - len(y_resampled)))

                if ch == 0:
                    xb[idx] = np.zeros([num_channels, x_proc.shape[1]])
                xb[idx][ch] = y_resampled

    # Format output
    if not is_multichannel:
        spl = spl[0]
        if sigbands and xb is not None:
            xb = [band[0] for band in xb]

    if sigbands and xb is not None:
        return spl, freq, xb
    else:
        return spl, freq


def _typesignal(x: Union[List[float], np.ndarray, Tuple[float, ...]]) -> np.ndarray:
    """
    Ensure the input signal is a numpy array and has at least one dimension.

    :param x: Input signal.
    :type x: Union[List[float], np.ndarray, Tuple[float, ...]]
    :return: Signal as a numpy array.
    :rtype: np.ndarray
    """
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
    """
    Generate Second-Order Sections (SOS) coefficients for each frequency band.

    :param freq: List of center frequencies.
    :type freq: List[float]
    :param freq_d: List of lower edge frequencies.
    :type freq_d: List[float]
    :param freq_u: List of upper edge frequencies.
    :type freq_u: List[float]
    :param fs: Sample rate in Hz.
    :type fs: int
    :param order: Order of the Butterworth filter.
    :type order: int
    :param factor: Downsampling factors for each band.
    :type factor: np.ndarray
    :param show: If True, plot and show the filter response.
    :type show: bool
    :param plot_file: Path to save the filter response plot.
    :type plot_file: Optional[str]
    :return: List of SOS coefficients for each band.
    :rtype: List[np.ndarray]
    """
    # Initialize coefficients matrix
    sos = [np.array([]) for _ in range(len(freq))]

    # Generate coefficients for each frequency band
    for idx, (lower, upper) in enumerate(zip(freq_d, freq_u)):
        # Downsampling to improve filter coefficients
        fsd = fs / factor[idx]  # New sampling rate
        # Butterworth Filter with SOS coefficients
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
    """
    Plot the frequency response of the filter bank.

    :param sos: List of SOS coefficients.
    :type sos: List[np.ndarray]
    :param freq: List of center frequencies.
    :type freq: List[float]
    :param freq_u: List of upper edge frequencies.
    :type freq_u: List[float]
    :param freq_d: List of lower edge frequencies.
    :type freq_d: List[float]
    :param fs: Sample rate in Hz.
    :type fs: int
    :param factor: Downsampling factors for each band.
    :type factor: np.ndarray
    :param show: If True, show the plot.
    :type show: bool
    :param plot_file: Path to save the plot.
    :type plot_file: Optional[str]
    """
    wn = 8192
    w = np.zeros([wn, len(freq)])
    h: np.ndarray = np.zeros([wn, len(freq)], dtype=np.complex128)

    for idx in range(len(freq)):
        fsd = fs / factor[idx]  # New sampling rate
        w[:, idx], h[:, idx] = signal.sosfreqz(sos[idx], worN=wn, whole=False, fs=fsd)

    fig, ax = plt.subplots()
    ax.semilogx(w, 20 * np.log10(abs(h) + np.finfo(float).eps), "b")
    ax.grid(which="major")
    ax.grid(which="minor", linestyle=":")
    ax.set_xlabel(r"Frequency [Hz]")
    ax.set_ylabel("Amplitude [dB]")
    ax.set_title("Second-Order Sections - Butterworth Filter")
    plt.xlim(freq_d[0] * 0.8, freq_u[-1] * 1.2)
    plt.ylim(-4, 1)
    ax.set_xticks([16, 31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000])
    ax.set_xticklabels(["16", "31.5", "63", "125", "250", "500", "1k", "2k", "4k", "8k", "16k"])
    if plot_file:
        plt.savefig(plot_file)
    if show:
        plt.show()
    plt.close(fig)


def _genfreqs(limits: List[float], fraction: float, fs: int) -> Tuple[List[float], List[float], List[float]]:
    """
    Generate center, lower, and upper frequencies based on limits and fraction.

    :param limits: Minimum and maximum frequencies [f_min, f_max].
    :type limits: List[float]
    :param fraction: Bandwidth fraction (e.g., 1 for octave, 3 for 1/3 octave).
    :type fraction: float
    :param fs: Sample rate in Hz.
    :type fs: int
    :return: Tuple of (center_freqs, lower_edges, upper_edges).
    :rtype: Tuple[List[float], List[float], List[float]]
    """
    # Generate frequencies
    freq, freq_d, freq_u = getansifrequencies(fraction, limits)

    # Remove outer frequency to prevent filter error (fs/2 < freq)
    freq, freq_d, freq_u = _deleteouters(freq, freq_d, freq_u, fs)

    return freq, freq_d, freq_u


def normalizedfreq(fraction: int) -> List[float]:
    """
    Get normalized frequencies for one-octave and third-octave band according to IEC 61260-1-2014.

    :param fraction: Octave type, 1 for one-octave, 3 for third-octave.
    :type fraction: int
    :return: List of predefined frequencies.
    :rtype: List[float]
    :raises ValueError: If fraction is not 1 or 3.
    """
    predefined = {
        1: _oneoctave(),
        3: _thirdoctave(),
    }
    if fraction not in predefined:
        raise ValueError("Normalized frequencies only available for fraction=1 or fraction=3")
    return predefined[fraction]


def _thirdoctave() -> List[float]:
    """
    Return the standard 1/3 octave band center frequencies.

    :return: List of frequencies.
    :rtype: List[float]
    """
    # IEC 61260 - 1 - 2014 (added 12.5, 16, 20 Hz)
    return [
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
    ]


def _oneoctave() -> List[float]:
    """
    Return the standard 1 octave band center frequencies.

    :return: List of frequencies.
    :rtype: List[float]
    """
    # IEC 61260 - 1 - 2014 (added 16 Hz)
    return [16, 31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]


def _deleteouters(
    freq: List[float], freq_d: List[float], freq_u: List[float], fs: int
) -> Tuple[List[float], List[float], List[float]]:
    """
    Remove frequencies above the Nyquist frequency (fs/2).

    :param freq: List of center frequencies.
    :type freq: List[float]
    :param freq_d: List of lower edge frequencies.
    :type freq_d: List[float]
    :param freq_u: List of upper edge frequencies.
    :type freq_u: List[float]
    :param fs: Sample rate in Hz.
    :type fs: int
    :return: Tuple of (freq, freq_d, freq_u) with high frequencies removed.
    :rtype: Tuple[List[float], List[float], List[float]]
    """
    freq_arr = np.array(freq)
    freq_d_arr = np.array(freq_d)
    freq_u_arr = np.array(freq_u)

    idx = np.where(freq_u_arr > fs / 2)[0]
    if len(idx) > 0:
        _printwarn("Low sampling rate, frequencies above fs/2 will be removed")
        freq_arr = np.delete(freq_arr, idx)
        freq_d_arr = np.delete(freq_d_arr, idx)
        freq_u_arr = np.delete(freq_u_arr, idx)

    return freq_arr.tolist(), freq_d_arr.tolist(), freq_u_arr.tolist()


def getansifrequencies(
    fraction: float, limits: Optional[List[float]] = None
) -> Tuple[List[float], List[float], List[float]]:
    """
    Calculate array of frequencies and its edges according to ANSI s1.11-2004 & IEC 61260-1-2014.

    :param fraction: Bandwidth 'b'. Examples: 1/3-octave b=3, 1-octave b=1, 2/3-octave b=1.5.
    :type fraction: float
    :param limits: Minimum and maximum frequencies [f_min, f_max]. Default: [12, 20000].
    :type limits: Optional[List[float]]
    :return: Tuple containing (center_frequencies, lower_edges, upper_edges).
    :rtype: Tuple[List[float], List[float], List[float]]
    """

    if limits is None:
        limits = [12, 20000]

    # Octave ratio g (ANSI s1.11, 3.2, pg. 2)
    g = 10 ** (3 / 10)  # Or g = 2
    # Reference frequency (ANSI s1.11, 3.4, pg. 2)
    fr = 1000

    # Get starting index 'x' and first center frequency
    x = _initindex(limits[0], fr, g, fraction)
    freq = np.array([_ratio(g, x, fraction) * fr])

    # Get each frequency until reach maximum frequency
    freq_x = freq[0]
    while freq_x * _bandedge(g, fraction) < limits[1]:
        # Increase index
        x = x + 1
        # New frequency
        freq_x = _ratio(g, x, fraction) * fr
        # Store new frequency
        freq = np.append(freq, freq_x)

    # Get band-edges
    freq_d = freq / _bandedge(g, fraction)
    freq_u = freq * _bandedge(g, fraction)

    return freq.tolist(), freq_d.tolist(), freq_u.tolist()


def _initindex(f: float, fr: float, g: float, b: float) -> int:
    """
    Calculate the starting index 'x' for frequency generation.

    :param f: Starting frequency.
    :type f: float
    :param fr: Reference frequency (1000 Hz).
    :type fr: float
    :param g: Octave ratio.
    :type g: float
    :param b: Bandwidth fraction.
    :type b: float
    :return: Index 'x'.
    :rtype: int
    """
    # We use a threshold to determine if b is effectively an odd integer
    # fraction=3 (1/3 octave) -> b=3 (odd)
    # fraction=1 (1 octave) -> b=1 (odd)
    # fraction=1.5 (2/3 octave) -> b=1.5 (not integer, but let's follow the standard)
    if round(b) % 2:  # ODD ('x' solve from ANSI s1.11, eq. 3)
        return int(np.round((b * np.log(f / fr) + 30 * np.log(g)) / np.log(g)))
    else:  # EVEN ('x' solve from ANSI s1.11, eq. 4)
        return int(np.round((2 * b * np.log(f / fr) + 59 * np.log(g)) / (2 * np.log(g))))


def _ratio(g: float, x: int, b: float) -> float:
    """
    Calculate the ratio for center frequency based on index and bandwidth.

    :param g: Octave ratio.
    :type g: float
    :param x: Index.
    :type x: int
    :param b: Bandwidth fraction.
    :type b: float
    :return: Frequency ratio.
    :rtype: float
    """
    if round(b) % 2:  # ODD (ANSI s1.11, eq. 3)
        return float(g ** ((x - 30) / b))
    else:  # EVEN (ANSI s1.11, eq. 4)
        return float(g ** ((2 * x - 59) / (2 * b)))


def _bandedge(g: float, b: float) -> float:
    """
    Calculate the band-edge ratio.

    :param g: Octave ratio.
    :type g: float
    :param b: Bandwidth fraction.
    :type b: float
    :return: Band-edge ratio.
    :rtype: float
    """
    # Band-edge ratio (ANSI s1.11, 3.7, pg. 3)
    return float(g ** (1 / (2 * b)))


def _printwarn(msg: str) -> None:
    """
    Print a warning message to the console.

    :param msg: Message to print.
    :type msg: str
    """
    print("*********\n" + msg + "\n*********")


def _downsamplingfactor(freq: List[float], fs: int) -> np.ndarray:
    """
    Calculate downsampling factor to improve filter accuracy at low frequencies.

    :param freq: List of center frequencies.
    :type freq: List[float]
    :param fs: Sample rate in Hz.
    :type fs: int
    :return: Array of downsampling factors.
    :rtype: np.ndarray
    """
    guard = 0.10
    factor = (np.floor((fs / (2 + guard)) / np.array(freq))).astype("int")
    # Clamp factor between 1 and 50
    return cast(np.ndarray, np.clip(factor, 1, 50))
