#  Copyright (c) 2026. Jose Manuel Requena Plens
r"""Sound-speed profiles over a column of water.

A profile is a description of a place rather than of a substance, so it stays
with the marchers that consume it while the point state of sea water lives in
:mod:`phonometry.fluids.water`. The four sound-speed equations moved there with
it; this module builds a column out of them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from ..._internal.validation import (
    require_above_absolute_zero_array,
    require_ranks,
    require_same_length,
)
from ...fluids.water import sea_water_sound_speed

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from numpy.typing import NDArray

#: The minimum number of nodes a depth polyline can have.
_MIN_POLYLINE_NODES = 2


@dataclass(frozen=True)
class SoundSpeedProfile:
    """Sound-speed profile ``c(z)`` over a column of water.

    :ivar depth: Depths, in metres (increasing downward).
    :ivar sound_speed: Sound speed at each depth, in m/s.
    :ivar gradient: Vertical sound-speed gradient ``dc/dz``, in (m/s)/m.
    :ivar model: The equation used.
    """

    depth: NDArray[np.float64]
    sound_speed: NDArray[np.float64]
    gradient: NDArray[np.float64]
    model: str

    def __post_init__(self) -> None:
        """Reject a profile whose three columns are not one water column.

        The three arrays are one depth axis written down three times:
        :func:`sound_speed_profile` broadcasts the temperature and the salinity
        onto ``depths``, evaluates the equation there, and hands the same array
        to ``np.gradient``. Nothing else in the library builds this result.

        The loud half of a mismatch is ``sound_speed``, which the figure draws
        against ``depth``: matplotlib stops both a short and a long one, but
        with its own ``x and y must have same first dimension, but have shapes
        (120,) and (121,)``, which names neither column and arrives from the
        drawing rather than from the profile that was wrong all along.

        The quiet half is ``gradient``, which no figure reads: short or long,
        the profile still plots without complaint. It is the column a reader
        takes the result apart for, since the sound-channel axis is where
        ``dc/dz`` turns from negative to positive and the ray curvature radius
        is ``c`` over its magnitude, and both of those read an entry of
        ``gradient`` beside the depth of the same index. Carrying the gradient
        of the same water column sampled on 25 depths onto a 121-depth profile
        leaves +0.017 (m/s)/m standing at 500 m where this water's own value
        there is -0.032: rays bending up where they in fact bend down, on a
        radius of 89 km instead of 46 km.

        The ranks are pinned as well because a count alone passes an extra
        axis on. Two scenarios stacked column-wise into ``sound_speed`` as an
        ``(n, 2)`` array carry one value per depth by every measure, and the
        figure quietly draws two curves under the one legend entry that names
        a single model.

        :raises ValueError: if ``sound_speed`` or ``gradient`` disagrees with
            ``depth``, or if a column carries more than one axis.
        """
        require_ranks(self, depth=1, sound_speed=1, gradient=1)
        require_same_length(self, "depth", "sound_speed", "gradient", axis="depth")

    def plot(
        self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any
    ) -> Axes:
        """Plot the sound-speed profile (speed vs depth, depth increasing down)."""
        from ..._i18n import check_language
        from ..._plot.underwater import plot_sound_speed_profile

        return plot_sound_speed_profile(
            self, ax=ax, language=check_language(language), **kwargs
        )


def sound_speed_profile(
    depths: NDArray[np.float64] | list[float],
    temperatures: NDArray[np.float64] | list[float] | float,
    salinities: NDArray[np.float64] | list[float] | float,
    *,
    model: str = "unesco",
    latitude: float = 45.0,
) -> SoundSpeedProfile:
    """Evaluate a sound-speed profile over a depth column.

    :param depths: Depths, in metres (1-D, non-negative, increasing).
    :param temperatures: Temperature per depth, in °C (array or a scalar
        broadcast to every depth).
    :param salinities: Salinity per depth, in PSU (array or scalar).
    :param model: Sound-speed equation (see :func:`sea_water_sound_speed`).
    :param latitude: Latitude for the depth→pressure conversion, in degrees.
    :return: A :class:`SoundSpeedProfile`.
    :raises ValueError: If the inputs are invalid.
    """
    z = np.asarray(depths, dtype=np.float64)
    if z.ndim != 1 or z.size < _MIN_POLYLINE_NODES:
        msg = "'depths' must be a 1-D array of at least two depths."
        raise ValueError(msg)
    if np.any(z < 0.0) or not np.all(np.isfinite(z)):
        msg = "'depths' must be finite and non-negative."
        raise ValueError(msg)
    if np.any(np.diff(z) <= 0.0):
        msg = "'depths' must be strictly increasing."
        raise ValueError(msg)
    temp = np.broadcast_to(np.asarray(temperatures, dtype=np.float64), z.shape)
    sal = np.broadcast_to(np.asarray(salinities, dtype=np.float64), z.shape)
    if not (np.all(np.isfinite(temp)) and np.all(np.isfinite(sal))):
        msg = "'temperatures' and 'salinities' must be finite."
        raise ValueError(msg)
    require_above_absolute_zero_array(temp, "temperatures")
    if np.any(sal < 0.0):
        msg = "'salinities' must be non-negative."
        raise ValueError(msg)
    # One column, one call: the four fits live with the medium now, and this
    # asks them for the whole profile at once rather than restating the
    # dispatch and the pressure conversion a second time.
    c = np.asarray(
        sea_water_sound_speed(temp, sal, z, model=model, latitude=latitude),
        dtype=np.float64,
    )
    gradient = np.gradient(c, z)
    return SoundSpeedProfile(
        depth=z, sound_speed=c, gradient=gradient, model=model.strip().lower()
    )
