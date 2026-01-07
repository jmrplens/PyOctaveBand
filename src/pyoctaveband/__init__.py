#  Copyright (c) 2020. Jose M. Requena-Plens
"""
Octave-Band and Fractional Octave-Band filter for signals in the time domain.
Implementation according to ANSI s1.11-2004 and IEC 61260-1-2014.
"""

from __future__ import annotations

from typing import List, Tuple, cast, overload

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

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


@overload
def octavefilter(
    x: List[float] | np.ndarray,
    fs: int,
    fraction: float = 1,
    order: int = 6,
    limits: List[float] | None = None,
    show: bool = False,
    sigbands: Literal[False] = False,
    plot_file: str | None = None,
    detrend: bool = True,
    **kwargs: str | float | bool
) -> Tuple[np.ndarray, List[float]]: ...


@overload
def octavefilter(
    x: List[float] | np.ndarray,
    fs: int,
    fraction: float = 1,
    order: int = 6,
    limits: List[float] | None = None,
    show: bool = False,
    sigbands: Literal[True] = True,
    plot_file: str | None = None,
    detrend: bool = True,
    **kwargs: str | float | bool
) -> Tuple[np.ndarray, List[float], List[np.ndarray]]: ...


def octavefilter(
    x: List[float] | np.ndarray,
    fs: int,
    fraction: float = 1,
    order: int = 6,
    limits: List[float] | None = None,
    show: bool = False,
    sigbands: bool = False,
    plot_file: str | None = None,
    detrend: bool = True,
    **kwargs: str | float | bool
) -> Tuple[np.ndarray, List[float]] | Tuple[np.ndarray, List[float], List[np.ndarray]]:
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
    :param detrend: If True, remove DC offset before filtering. Default: True.
    :type detrend: bool
    :param filter_type: (Optional) Type of filter ('butter', 'cheby1', 'cheby2', 'ellip', 'bessel'). Default: 'butter'.
    :param ripple: (Optional) Passband ripple in dB (for cheby1, ellip). Default: 0.1.
    :param attenuation: (Optional) Stopband attenuation in dB (for cheby2, ellip). Default: 60.0.
    :param calibration_factor: (Optional) Sensitivity multiplier. Default: 1.0.
    :param dbfs: (Optional) If True, return results in dBFS. Default: False.
    :param mode: (Optional) 'rms' or 'peak'. Default: 'rms'.
    :return: A tuple containing (SPL_array, Frequencies_list) or (SPL_array, Frequencies_list, signals).
    :rtype: Union[Tuple[np.ndarray, List[float]], Tuple[np.ndarray, List[float], List[np.ndarray]]]
    """
    
    # Use the class-based implementation
    filter_bank = OctaveFilterBank(
        fs=fs,
        fraction=fraction,
        order=order,
        limits=limits,
        filter_type=cast(str, kwargs.get("filter_type", "butter")),
        ripple=cast(float, kwargs.get("ripple", 0.1)),
        attenuation=cast(float, kwargs.get("attenuation", 60.0)),
        show=show,
        plot_file=plot_file,
        calibration_factor=cast(float, kwargs.get("calibration_factor", 1.0)),
        dbfs=cast(bool, kwargs.get("dbfs", False))
    )
    
    if sigbands:
        return filter_bank.filter(x, sigbands=True, mode=cast(str, kwargs.get("mode", "rms")), detrend=detrend)
    else:
        return filter_bank.filter(x, sigbands=False, mode=cast(str, kwargs.get("mode", "rms")), detrend=detrend)