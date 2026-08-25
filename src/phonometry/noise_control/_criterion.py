#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Shared plumbing of the composed noise-control workflows.

:mod:`phonometry.noise_control.duct_path` and
:mod:`phonometry.noise_control.room_to_room` are the two calculations in the
library that walk a whole path and end by laying the received octave-band
spectrum against a room criterion. They differ in everything between the ends
-- one cascades duct elements, the other crosses a partition -- but the ends
themselves are the same: coerce a per-band decibel spectrum, validate a
criterion family and its target, and answer the four questions a design sheet
asks of the result. Those live here so both results answer them identically.

Private module: nothing here is part of the public API.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import numpy as np

from .._internal.validation import require_choice

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

    from ..room.noise_criteria import NCResult, RCResult

#: The room-criterion families a composed workflow can be rated against:
#: the NC curves of ANSI/ASA S12.2-2019 Table 1 and the RC Mark II curves of
#: its Annex D.
CRITERION_FAMILIES: tuple[str, ...] = ("NC", "RC")


def validate_target(criterion: str, target: float | None) -> tuple[str, float | None]:
    """Validate a criterion family and its target value.

    :param criterion: ``"NC"`` or ``"RC"``.
    :param target: The curve designation (e.g. ``45``), or ``None`` for none.
    :return: The validated ``(family, target)``.
    :raises ValueError: If the family is unknown or the target is not finite.
    """
    family = require_choice(criterion, "criterion", CRITERION_FAMILIES)
    if target is None:
        return family, None
    value = float(target)
    if not np.isfinite(value):
        msg = "'target' must be a finite criterion value."
        raise ValueError(msg)
    return family, value


class PerBandValues(Protocol):
    """Anything carrying a per-band ``values`` array.

    Structural stand-in for the spectrum results a workflow forwards whole,
    such as :class:`~phonometry.noise_control.hvac.HvacSpectrumResult`.
    """

    @property
    def values(self) -> ArrayLike:
        """The per-band quantity."""


def as_band_spectrum(
    value: ArrayLike | PerBandValues | None, n: int, name: str, *, default: float = 0.0
) -> NDArray[np.float64]:
    """Coerce a scalar, array or spectrum result to a finite ``n``-band array.

    :param value: A scalar (broadcast over the bands), a per-band sequence, an
        object exposing per-band ``values`` (such as an
        :class:`~phonometry.noise_control.hvac.HvacSpectrumResult`), or ``None``
        for a constant ``default``.
    :param n: Number of analysis bands.
    :param name: Parameter name, for the error messages.
    :param default: The value ``None`` expands to.
    :return: The validated ``(n,)`` array.
    :raises ValueError: If the shape does not match or a value is not finite.
    """
    if value is None:
        return np.full(n, default, dtype=np.float64)
    values = getattr(value, "values", value)
    arr = np.atleast_1d(np.asarray(values, dtype=np.float64))
    if arr.size == 1:
        arr = np.full(n, float(arr[0]), dtype=np.float64)
    if arr.shape != (n,):
        msg = (
            f"'{name}' must be a scalar or have one value per band ({n}); "
            f"got shape {arr.shape}."
        )
        raise ValueError(msg)
    if not np.all(np.isfinite(arr)):
        msg = f"'{name}' must be finite."
        raise ValueError(msg)
    return arr


class CriterionCarrier(Protocol):
    """A composed-workflow result that ends against a room criterion.

    Structural contract of the four helpers below: the analysis bands, the
    spectrum that arrives at the receiver, the criterion family and the design
    target. :class:`~phonometry.noise_control.duct_path.DuctPathResult` and
    :class:`~phonometry.noise_control.room_to_room.RoomToRoomResult` both
    satisfy it.
    """

    @property
    def frequencies(self) -> np.ndarray:
        """Analysis band centre frequencies, Hz."""

    @property
    def received_level(self) -> np.ndarray:
        """The spectrum arriving at the receiver, dB."""

    @property
    def criterion(self) -> str:
        """The criterion family, one of :data:`CRITERION_FAMILIES`."""

    @property
    def target(self) -> float | None:
        """The design curve designation, or ``None``."""


def rating_of(result: CriterionCarrier) -> NCResult | RCResult:
    """Rate the received spectrum against its criterion family."""
    from ..room.noise_criteria import noise_criterion, room_criterion

    if result.criterion == "NC":
        return noise_criterion(result.received_level, result.frequencies)
    return room_criterion(result.received_level, result.frequencies)


def curve_of(result: CriterionCarrier) -> np.ndarray | None:
    """The design criterion curve at the analysis bands, or ``None``."""
    if result.target is None:
        return None
    from ..room.noise_criteria import _criterion_curve_at

    return _criterion_curve_at(result.criterion, result.target, result.frequencies)


def exceedance_of(result: CriterionCarrier) -> np.ndarray | None:
    """Band-by-band excess of the received level over the criterion curve."""
    curve = curve_of(result)
    return None if curve is None else result.received_level - curve


def meets_target_of(result: CriterionCarrier) -> bool | None:
    """``True`` when no band of the received spectrum exceeds the curve."""
    excess = exceedance_of(result)
    return None if excess is None else bool(np.all(excess <= 0.0))
