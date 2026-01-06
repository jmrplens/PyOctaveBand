#  Copyright (c) 2026. Jose M. Requena-Plens
"""
Signal processing utilities for pyoctaveband.
"""

from __future__ import annotations

from typing import List, Tuple, cast

import numpy as np
from scipy import signal


def _typesignal(x: List[float] | np.ndarray | Tuple[float, ...]) -> np.ndarray:
    """
    Ensure signal is a numpy array.

    :param x: Input signal.
    :return: Numpy array.
    """
    if isinstance(x, np.ndarray):
        return x
    return np.atleast_1d(np.array(x))


def _resample_to_length(y: np.ndarray, factor: int, target_length: int) -> np.ndarray:
    """
    Resample signal and ensure the output matches target_length exactly.

    :param y: Input signal.
    :param factor: Resampling factor.
    :param target_length: Target length.
    :return: Resampled signal.
    """
    y_resampled = signal.resample_poly(y, factor, 1)
    if len(y_resampled) > target_length:
        y_resampled = y_resampled[:target_length]
    elif len(y_resampled) < target_length:
        y_resampled = np.pad(y_resampled, (0, target_length - len(y_resampled)))
    return cast(np.ndarray, y_resampled)


def _downsamplingfactor(freq: List[float], fs: int) -> np.ndarray:
    """
    Compute optimal downsampling factors for filter stability.

    :param freq: Frequencies.
    :param fs: Sample rate.
    :return: Array of factors.
    """
    guard = 0.50
    factor = (np.floor((fs / (2 + guard)) / np.array(freq))).astype("int")
    return cast(np.ndarray, np.clip(factor, 1, 500))