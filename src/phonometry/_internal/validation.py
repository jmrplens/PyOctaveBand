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
    from collections.abc import Mapping

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


def require_axis_count(
    value: object, owner: str, name: str, axis: str = "band", *, rank: int | None
) -> int:
    """The number of entries along the first axis of *value*.

    ``len()``, not ``np.shape()``: the first axis of a tuple of per-band
    arrays is its length whatever the arrays hold, while ``np.shape`` would
    try to stack them and raise about an inhomogeneous shape for a bank whose
    bands carry filters of different orders.

    *rank* has no default and must be stated, because a count on its own says
    nothing about how many axes there are: an array of shape ``(bands, 2)``
    counts the right number of bands on its first axis and carries the second
    one onward untouched. Pass ``None`` only where the rank is pinned
    elsewhere, as :func:`require_ranks` does for a whole result at once.

    :param value: The field whose entries are counted.
    :param owner: Name of the type being checked, used in the error message.
    :param name: Field name used in the error message.
    :param axis: What the entries measure, singular, for the error message.
    :param rank: The number of axes the field must have, or ``None`` when that
        is checked elsewhere.
    :return: The number of entries.
    :raises ValueError: if *value* is a single number, or has another rank.
    """
    if rank is not None:
        require_axis_rank(value, owner, name, rank)
    try:
        return len(value)  # type: ignore[arg-type]
    except TypeError:
        msg = f"{owner}: '{name}' must carry one value per {axis}, not a single number."
        raise ValueError(msg) from None


def require_axis_rank(value: object, owner: str, name: str, rank: int) -> None:
    """Require *value* to have exactly *rank* axes.

    Counting an axis is not enough on its own: an array of shape
    ``(paths, bands, 2)`` has the right number of paths on its first axis and
    the right number of bands on its second, so every count agrees and the
    extra axis travels on into whatever indexes the field next.

    Only arrays are measured. A tuple of per-band entries is one axis of
    entries whatever each entry holds, and asking numpy for its rank would
    stack them and raise about an inhomogeneous shape.

    :param value: The field whose axes are counted.
    :param owner: Name of the type being checked, used in the error message.
    :param name: Field name used in the error message.
    :param rank: The number of axes the field must have.
    :raises ValueError: if *value* has a different number of axes.
    """
    actual = value.ndim if isinstance(value, np.ndarray) else 1
    if actual != rank:
        axes = "one axis" if rank == 1 else f"{rank} axes"
        msg = (
            f"{owner}: '{name}' must have {axes}; got {actual}. "
            "An extra axis passes every count and reaches the reader intact."
        )
        raise ValueError(msg)


def require_ranks(owner: object, **ranks: int) -> None:
    """Require each named field of *owner* to have the given number of axes.

    The companion of :func:`require_same_length`, which pins how long an axis
    is but not how many there are. ``None`` fields are skipped.

    :param owner: The instance whose fields are measured.
    :param ranks: Field name to the number of axes it must have.
    :raises ValueError: if a field has a different number of axes.
    """
    for name, rank in ranks.items():
        value = getattr(owner, name)
        if value is None:
            continue
        require_axis_rank(value, type(owner).__name__, name, rank)


def require_equal_counts(
    owner: str, counts: Mapping[str, int], axis: str = "band"
) -> None:
    """Require every measured count to agree, naming each one that does not.

    The sibling of :func:`require_same_length` for the counts a caller has
    already measured: entries of a nested sequence, an axis picked out of a
    two-dimensional field, anything whose label is not a plain attribute name.

    The message lists every count it was given rather than picking out an odd
    one, because among peers no count is authoritative and a majority winner
    would be a guess. Where one quantity *is* authoritative, pass it and the
    single suspect: two entries make the shortest true message there is.

    :param owner: Name of the type being checked, used in the error message.
    :param counts: Label to count, in the order they should be reported.
    :param axis: What the counts measure, singular, for the error message.
    :raises ValueError: if two of the counts disagree.
    """
    if len(set(counts.values())) <= 1:
        return
    listed = ", ".join(f"'{label}' ({n})" for label, n in counts.items())
    msg = f"{owner}: {listed} must each carry one value per {axis}."
    raise ValueError(msg)


def require_same_length(
    owner: object, *fields: str | tuple[str, int], axis: str = "band"
) -> None:
    """Require the named fields of *owner* to share one length.

    Written for the ``__post_init__`` of a result dataclass, so that a result
    whose per-band arrays disagree cannot be built at all. Validating here
    rather than where the values are read covers the readers that do not
    exist yet, and both directions of the mistake: an array one short raises
    somewhere downstream, but an array one long is silently truncated by the
    table that prints it, under a total that summed the whole of it.

    A field may be given as ``"name"`` for a length along the first axis, or
    as ``("name", i)`` when the axis in question is not the first one, as it
    is not for a per-path-per-band array whose bands run along axis 1.

    ``None`` fields are skipped: an absent optional quantity is not a
    disagreement.

    :param owner: The instance whose fields are compared.
    :param fields: Field names, or ``(name, axis_index)`` pairs.
    :param axis: What the lengths measure, singular, for the error message.
    :raises ValueError: if two of the fields disagree, or one is a scalar.
    """
    counts: dict[str, int] = {}
    for field in fields:
        name, index = field if isinstance(field, tuple) else (field, 0)
        value = getattr(owner, name)
        if value is None:
            continue
        if index == 0:
            counts[name] = require_axis_count(
                value, type(owner).__name__, name, axis, rank=None
            )
            continue
        shape = np.shape(value)
        if len(shape) <= index:
            msg = (
                f"{type(owner).__name__}: '{name}' must have an axis {index} "
                f"to carry one value per {axis}; got shape {shape}."
            )
            raise ValueError(msg)
        counts[f"{name} (axis {index})"] = shape[index]
    require_equal_counts(type(owner).__name__, counts, axis)


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
