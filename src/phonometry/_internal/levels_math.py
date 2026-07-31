#  Copyright (c) 2026. Jose Manuel Requena Plens
"""
Shared decibel-energy arithmetic helpers (private).

Seeded by the library audit: many modules restated the same
``10 lg(sum/mean 10^(L/10))`` energy combinations inline. New code
should combine levels through these helpers; existing modules migrate
as they are touched.
"""

from __future__ import annotations

from typing import overload

import numpy as np
from numpy.typing import ArrayLike


@overload
def energy_mean(levels: ArrayLike, axis: None = None) -> float: ...


@overload
def energy_mean(levels: ArrayLike, axis: int) -> np.ndarray: ...


def energy_mean(levels: ArrayLike, axis: int | None = None) -> np.ndarray | float:
    """Energy (mean-square) average of decibel levels.

    Computes ``10 lg( (1/n) sum_i 10^(Li/10) )`` along *axis*.

    :param levels: Levels to average, in decibels.
    :param axis: Axis over which to average; ``None`` averages everything.
    :return: A ``float`` when *axis* is ``None``, otherwise an array.
    """
    arr = np.asarray(levels, dtype=np.float64)
    out = 10.0 * np.log10(np.mean(10.0 ** (arr / 10.0), axis=axis))
    if axis is None:
        return float(out)
    return np.asarray(out, dtype=np.float64)


@overload
def energy_sum(levels: ArrayLike, axis: None = None) -> float: ...


@overload
def energy_sum(levels: ArrayLike, axis: int) -> np.ndarray: ...


def energy_sum(levels: ArrayLike, axis: int | None = None) -> np.ndarray | float:
    """Energetic sum of decibel levels.

    Computes ``10 lg( sum_i 10^(Li/10) )`` along *axis*.

    :param levels: Levels to combine, in decibels.
    :param axis: Axis over which to sum; ``None`` sums everything.
    :return: A ``float`` when *axis* is ``None``, otherwise an array.
    """
    arr = np.asarray(levels, dtype=np.float64)
    out = 10.0 * np.log10(np.sum(10.0 ** (arr / 10.0), axis=axis))
    if axis is None:
        return float(out)
    return np.asarray(out, dtype=np.float64)


@overload
def weighted_energy_mean(
    levels: ArrayLike, weights: ArrayLike, axis: None = None
) -> float: ...


@overload
def weighted_energy_mean(
    levels: ArrayLike, weights: ArrayLike, axis: int
) -> np.ndarray: ...


def weighted_energy_mean(
    levels: ArrayLike, weights: ArrayLike, axis: int | None = None
) -> np.ndarray | float:
    """Weighted energy average of decibel levels.

    Computes ``10 lg( sum_i wi 10^(Li/10) / sum_i wi )`` along *axis*.
    The weights must broadcast against *levels* along the reduced axis
    (e.g. pass ``areas[:, None]`` for per-row weights of a 2-D array).

    :param levels: Levels to average, in decibels.
    :param weights: Non-negative weights (e.g. sub-areas or durations).
    :param axis: Axis over which to average; ``None`` averages everything.
    :return: A ``float`` when *axis* is ``None``, otherwise an array.
    """
    arr = np.asarray(levels, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    # Normalize by the weights summed over the SAME reduction, so weights
    # that vary along a non-reduced axis stay correct.
    denominator = np.sum(np.broadcast_to(w, arr.shape), axis=axis)
    out = 10.0 * np.log10(np.sum(w * 10.0 ** (arr / 10.0), axis=axis) / denominator)
    if axis is None:
        return float(out)
    return np.asarray(out, dtype=np.float64)
