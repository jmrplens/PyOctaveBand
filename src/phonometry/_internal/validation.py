#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Shared input-validation helpers (private).

Seeded by the library audit: modules used to hand-roll the same checks with
diverging NaN semantics and error-message styles. New code should validate
through these helpers; existing modules migrate as they are touched.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import ArrayLike


def require_positive(value: float, name: str) -> float:
    """Require a positive finite number (rejects NaN and infinities).

    :param value: The value to validate.
    :param name: Parameter name used in the error message.
    :return: The validated value as a ``float``.
    :raises ValueError: for a non-finite or non-positive value.
    """
    if not math.isfinite(value) or value <= 0.0:
        msg = f"'{name}' must be positive."
        raise ValueError(msg)
    return float(value)


def require_non_negative(value: float, name: str) -> float:
    """Require a non-negative finite number (rejects NaN and infinities).

    :param value: The value to validate.
    :param name: Parameter name used in the error message.
    :return: The validated value as a ``float``.
    :raises ValueError: for a non-finite or negative value.
    """
    if not math.isfinite(value) or value < 0.0:
        msg = f"'{name}' must be non-negative."
        raise ValueError(msg)
    return float(value)


def require_fraction(value: float, name: str) -> float:
    """Require a finite fraction in ``[0, 1)``.

    :param value: The value to validate.
    :param name: Parameter name used in the error message.
    :return: The validated value as a ``float``.
    :raises ValueError: for a non-finite value or one outside ``[0, 1)``.
    """
    if not math.isfinite(value) or value < 0.0 or value >= 1.0:
        msg = f"'{name}' must be in the range [0, 1)."
        raise ValueError(msg)
    return float(value)


def require_choice(value: str, name: str, options: tuple[str, ...]) -> str:
    """Require *value* to be one of *options*.

    :param value: The value to validate.
    :param name: Parameter name used in the error message.
    :param options: The accepted values.
    :return: The validated value.
    :raises ValueError: for a value not in *options*.
    """
    if value not in options:
        msg = f"'{name}' must be one of {options}; got {value!r}."
        raise ValueError(msg)
    return value


def require_positive_array(x: ArrayLike, name: str) -> np.ndarray:
    """Coerce *x* to a 1-D float64 array of strictly positive, finite values.

    :param x: The input (scalar or 1-D array-like).
    :param name: Parameter name used in the error message.
    :return: The validated ``float64`` array (at least 1-D).
    :raises ValueError: for an empty, multi-dimensional, non-finite or
        non-positive input.
    """
    arr = np.atleast_1d(np.asarray(x, dtype=np.float64))
    if arr.ndim != 1 or arr.size == 0:
        msg = f"'{name}' must be a non-empty 1-D array."
        raise ValueError(msg)
    if not np.all(np.isfinite(arr)):
        msg = f"'{name}' must be finite."
        raise ValueError(msg)
    if np.any(arr <= 0.0):
        msg = f"'{name}' must be strictly positive."
        raise ValueError(msg)
    return arr


def require_finite_array(x: ArrayLike, name: str) -> np.ndarray:
    """Coerce *x* to a non-empty 1-D float64 array of finite values.

    The sign-agnostic sibling of :func:`require_positive_array`, for the
    level-like quantities (dB values, corrections) that may legitimately be
    negative but never NaN.

    :param x: The input (scalar or 1-D array-like).
    :param name: Parameter name used in the error message.
    :return: The validated ``float64`` array (at least 1-D).
    :raises ValueError: for an empty, multi-dimensional or non-finite input.
    """
    arr = np.atleast_1d(np.asarray(x, dtype=np.float64))
    if arr.ndim != 1 or arr.size == 0:
        msg = f"'{name}' must be a non-empty 1-D array."
        raise ValueError(msg)
    if not np.all(np.isfinite(arr)):
        msg = f"'{name}' must contain only finite values."
        raise ValueError(msg)
    return arr


def require_per_band(
    x: ArrayLike, name: str, bands: np.ndarray, bands_name: str = "frequency"
) -> np.ndarray:
    """Broadcast *x* over the band axis, saying whose length disagrees.

    A scalar applies to every band and an array of the band count is taken
    band by band. Nothing else is: broadcasting alone would also stretch a
    one-element array across every band, which is the shape a caller lands on
    by computing one value where the bands needed several, and the answer that
    comes back from repeating it looks perfectly ordinary.

    ``np.broadcast_to`` is left to do the stretching once the shape is known
    to be one of the two, because what it cannot do is name the culprit: it
    reports the two shapes it failed to reconcile, from inside its own C code,
    without saying which argument carried which.

    :param x: The per-band input (a scalar applies to every band).
    :param name: Parameter name used in the error message.
    :param bands: The band axis to match, already coerced and 1-D.
    :param bands_name: Parameter name of the band axis, for the message.
    :return: The broadcast ``float64`` array, of ``bands.shape``.
    :raises ValueError: if *x* is neither a scalar nor of ``bands``' length.
    """
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 0 and arr.shape != bands.shape:
        msg = (
            f"'{name}' must be a scalar or carry one value per band "
            f"({bands.size} in '{bands_name}'); got shape {arr.shape}."
        )
        raise ValueError(msg)
    return np.broadcast_to(arr, bands.shape).astype(np.float64)


def require_1d_signal(x: ArrayLike, name: str = "signal") -> np.ndarray:
    """Coerce *x* to a float64 array and require a 1-D time series.

    Multichannel input is rejected rather than silently flattened (a raveled
    2-D signal concatenates the channels into one wrong series).

    :param x: The input signal.
    :param name: Parameter name used in the error message.
    :return: The validated ``float64`` array.
    :raises ValueError: for a non-1-D input.
    """
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 1:
        msg = (
            f"{name} must be a 1-D time series; got shape {arr.shape}. "
            "Process multichannel signals one channel at a time."
        )
        raise ValueError(msg)
    return arr
