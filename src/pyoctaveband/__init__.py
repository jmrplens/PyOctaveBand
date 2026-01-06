#  Copyright (c) 2020. Jose M. Requena-Plens
"""
Octave-Band and Fractional Octave-Band filter for signals in the time domain.
Implementation according to ANSI s1.11-2004 and IEC 61260-1-2014.
"""

from typing import List, Optional, Tuple, Union

import matplotlib
import numpy as np

from .calibration import calculate_sensitivity
from .core import OctaveFilterBank
from .frequencies import getansifrequencies, normalizedfreq
from .parametric_filters import linkwitz_riley, time_weighting, weighting_filter

# Use non-interactive backend for plots
matplotlib.use("Agg")

# Public methods
__all__ = [
    "octavefilter",
    "getansifrequencies",
    "normalizedfreq",
    "OctaveFilterBank",
    "weighting_filter",
    "time_weighting",
    "linkwitz_riley",
    "calculate_sensitivity",
]


def octavefilter(
    x: Union[List[float], np.ndarray],
    fs: int,
    fraction: float = 1,
    order: int = 6,
    limits: Optional[List[float]] = None,
    show: bool = False,
    sigbands: bool = False,
    plot_file: Optional[str] = None,
    filter_type: str = "butter",
    ripple: float = 0.1,
    attenuation: float = 60.0,
    calibration_factor: float = 1.0,
    dbfs: bool = False,
) -> Union[Tuple[np.ndarray, List[float]], Tuple[np.ndarray, List[float], List[np.ndarray]]]:
    """
    Filter a signal with octave or fractional octave filter bank.

    This method uses a filter bank with Second-Order Sections (SOS) coefficients.
    To obtain the correct coefficients, automatic subsampling is applied to the
    signal in each filtered band.

    Multichannel support: If x is 2D (channels, samples), each channel is filtered.

    :param x: Input signal (1D array or 2D array [channels, samples]).
    :type x: Union[List[float], np.ndarray]
    :param fs: Sample rate in Hz.
    :type fs: int
    :param fraction: Bandwidth 'b'. Examples: 1/3-octave b=3, 1-octave b=1, 2/3-octave b=1.5. Default: 1.
    :type fraction: float
    :param order: Order of the filter. Default: 6.
    :type order: int
    :param limits: Minimum and maximum limit frequencies [f_min, f_max]. Default [12, 20000].
    :type limits: Optional[List[float]]
    :param show: If True, plot and show the filter response.
    :type show: bool
    :param sigbands: If True, also return the signal in the time domain divided into bands.
    :type sigbands: bool
    :param plot_file: Path to save the filter response plot.
    :type plot_file: Optional[str]
    :param filter_type: Type of filter ('butter', 'cheby1', 'cheby2', 'ellip', 'bessel'). Default: 'butter'.
    :type filter_type: str
    :param ripple: Passband ripple in dB (for cheby1, ellip). Default: 0.1.
    :type ripple: float
    :param attenuation: Stopband attenuation in dB (for cheby2, ellip). Default: 60.0.
    :type attenuation: float
    :param calibration_factor: Sensitivity multiplier to map digital units to Pascals. Default: 1.0.
    :type calibration_factor: float
    :param dbfs: If True, return results in dB relative to Full Scale (0 dB = RMS 1.0). Default: False.
    :type dbfs: bool
    :return: A tuple containing (SPL_array, Frequencies_list) if sigbands is False,
             or (SPL_array, Frequencies_list, List_of_filtered_signals) if sigbands is True.
    :rtype: Union[Tuple[np.ndarray, List[float]], Tuple[np.ndarray, List[float], List[np.ndarray]]]
    """
    
    # Use the class-based implementation
    filter_bank = OctaveFilterBank(
        fs=fs,
        fraction=fraction,
        order=order,
        limits=limits,
        filter_type=filter_type,
        ripple=ripple,
        attenuation=attenuation,
        show=show,
        plot_file=plot_file,
        calibration_factor=calibration_factor,
        dbfs=dbfs
    )
    
    return filter_bank.filter(x, sigbands=sigbands)
