#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Signal processing utilities for phonometry."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
from scipy import signal

from .validation import _as_float64

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

# Rank of the cached SOS filter state ``zi`` as ``_sos_initial_state`` lays
# it out: (n_sections, 2) for 1-D input, (n_sections, n_channels, 2) for 2-D.
_ZI_NDIM_MONO = 2
_ZI_NDIM_MULTICHANNEL = 3


def _typesignal(x: ArrayLike, *, name: str = "x") -> np.ndarray:
    """Ensure signal is a float64 numpy array.

    Integer inputs (e.g. int16 audio from ``scipy.io.wavfile.read``) are
    converted to float64 to prevent silent overflow when the signal is
    squared internally. Float64 arrays are passed through without copying.

    Every entry point that consumes a recording resolves its samples here or
    through :func:`phonometry.io._resolve.resolve_samples`, which is why the
    three refusals below live in one place rather than in each of them.

    ``np.asarray`` on its own says too little and admits too much. A string
    or a ragged list of lists comes back in numpy's own words, naming no
    parameter, and a dict, a complex number or a plain object comes back as
    a ``TypeError``, which is not the exception these entry points document;
    both are re-raised as a ``ValueError`` that says which argument was
    wrong, exactly as :func:`~phonometry._internal.validation._as_float64`
    does for the result fields one layer up.

    ``None`` it does not refuse at all. ``None`` converts, to a ``NaN``
    array of one sample, so a call that lost its signal on the way in came
    back with a whole spectrum of ``NaN`` and refused nothing. It is not a
    ``Signal``, not a sequence of floats and not an array, which is what
    ``ArrayLike`` above says and what every caller's own annotation says.

    A non-finite sample is refused for the same reason one layer further
    down: a single ``NaN`` survives every filter and poisons every band
    computed from it, and it arrives by two routes that are the same value
    afterwards -- a ``None`` sitting inside the sequence, and a ``NaN`` the
    caller passed. The entry points that refused it by hand did so each in
    its own words, and only some of them refused it at all; the check
    belongs where every one of them passes, and they now share this one.

    :param x: Input signal.
    :param name: Parameter name used in the error messages.
    :return: Numpy float64 array.
    :raises ValueError: if *x* is ``None``, cannot be read as numbers, or
        carries a ``NaN`` or an infinity.
    """
    if x is None:
        msg = f"'{name}' must be a signal, not None."
        raise ValueError(msg)
    samples = np.atleast_1d(_as_float64(x, name))
    if not np.all(np.isfinite(samples)):
        msg = (
            f"'{name}' must contain only finite samples; a None inside the "
            "sequence arrives as a NaN and is refused the same way."
        )
        raise ValueError(msg)
    return samples


def _resample_to_length(y: np.ndarray, factor: int, target_length: int) -> np.ndarray:
    """Resample signal and ensure the output matches target_length exactly.
    Handles both 1D and 2D (channels, samples) arrays.

    :param y: Input signal.
    :param factor: Resampling factor.
    :param target_length: Target length.
    :return: Resampled signal.
    """
    if factor == 1:
        # Nothing to resample: fall through to the slice/pad logic only.
        y_resampled = y
    else:
        y_resampled = cast(np.ndarray, signal.resample_poly(y, factor, 1, axis=-1))
    current_length = y_resampled.shape[-1]

    if current_length > target_length:
        # Slice along the last axis (works for both 1D and 2D)
        y_resampled = y_resampled[..., :target_length]

    elif current_length < target_length:
        diff = target_length - current_length
        # Pad only the last axis. This works for both 1D and 2D arrays.
        # For 1D, pad_width becomes `[(0, diff)]`.
        # For 2D, pad_width becomes `[(0, 0), (0, diff)]`.
        pad_width: list[tuple[int, int]] = [(0, 0)] * (y_resampled.ndim - 1) + [
            (0, diff)
        ]

        y_resampled = np.pad(y_resampled, pad_width, mode="constant")

    return y_resampled


def _downsamplingfactor(
    freq: list[float], fs: int, headroom: float = 1.25
) -> np.ndarray:
    """Compute optimal downsampling factors for filter stability.

    :param freq: Band upper-edge frequencies.
    :param fs: Sample rate.
    :param headroom: Required ratio between the decimated Nyquist and the
        band's upper edge. 1.25 reproduces the classic ``fs / (2 + 0.5)``
        guard; filter types whose design extends above the upper edge
        (cheby2 stopband) need more.
    :return: Array of factors.
    """
    factor = (np.floor((fs / 2) / (headroom * np.array(freq)))).astype("int")
    return cast(np.ndarray, np.clip(factor, 1, 500))


def _sos_initial_state(
    sos: np.ndarray, x_proc: np.ndarray, steady_ic: bool
) -> np.ndarray:
    """Initial ``zi`` for an SOS cascade, sized to match the input shape."""
    n_sections = sos.shape[0]
    if x_proc.ndim == 1:
        if steady_ic:
            return cast(np.ndarray, signal.sosfilt_zi(sos))
        return np.zeros((n_sections, 2))
    n_channels = x_proc.shape[0]
    if steady_ic:
        zi_base = signal.sosfilt_zi(sos)
        return np.tile(zi_base[:, np.newaxis, :], (1, n_channels, 1))
    return np.zeros((n_sections, n_channels, 2))


def _sos_state_mismatch(zi: np.ndarray, x_proc: np.ndarray) -> bool:
    """Whether ``zi`` must be (re)allocated for *x_proc*."""
    if zi.size == 0:
        return True
    if x_proc.ndim == 1:
        return zi.ndim != _ZI_NDIM_MONO
    return zi.ndim != _ZI_NDIM_MULTICHANNEL or zi.shape[1] != x_proc.shape[0]
