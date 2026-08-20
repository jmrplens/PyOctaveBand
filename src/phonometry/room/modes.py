#  Copyright (c) 2026. Jose Manuel Requena Plens
r"""Normal modes of a rectangular room: frequencies, kinds, count and density.

Below the Schroeder frequency a room is not a diffuse field but a handful of
discrete standing waves. For the idealised rigid-walled rectangular room of
dimensions :math:`l_x \times l_y \times l_z` the wave equation separates and
the eigen-
frequencies are known in closed form (Long, *Architectural Acoustics* 2nd ed.,
Equation (8.43); Kuttruff, *Room Acoustics* 6th ed., 3.1):

.. math::

   f(n_x, n_y, n_z) = \frac{c_0}{2}
   \sqrt{ \left(\frac{n_x}{l_x}\right)^2 + \left(\frac{n_y}{l_y}\right)^2
   + \left(\frac{n_z}{l_z}\right)^2 }

with non-negative integer orders ``nx, ny, nz`` counting the nodal planes
perpendicular to each axis. Each mode is classified by how many of its orders
are non-zero:

* **axial** (one non-zero order): a wave bouncing between one pair of walls,
  the strongest and most audible family, e.g. the ``1,0,0`` fundamental;
* **tangential** (two): a wave grazing four walls, about 3 dB weaker;
* **oblique** (three): a wave involving all six walls, weaker still.

**How many modes.** Counting lattice points of the ``k``-space grid inside the
positive octant of a sphere of radius :math:`k = 2 \pi f / c_0`, with the
half- and
quarter-weight corrections for the points lying on the coordinate planes and
axes, gives the integrated mode count (Long Equation (8.45), after Morse 1948
and Pierce 1981):

.. math::

   N(f) = \frac{4 \pi}{3} V \left(\frac{f}{c_0}\right)^3
   + \frac{\pi}{4} S \left(\frac{f}{c_0}\right)^2
   + \frac{L}{8} \frac{f}{c_0}

with the room volume ``V``, the total wall area ``S`` and the sum ``L`` of the
twelve edge lengths. Its derivative is the **modal density** in modes per hertz
(Long Equation (8.46)):

.. math::

   \frac{dN}{df} = \frac{4 \pi V f^2}{c_0^3}
   + \frac{\pi}{2} \frac{S f}{c_0^2} + \frac{L}{8 c_0}

The smooth count is an asymptotic estimate: at low frequency, where only a few
modes exist, the exact enumeration of :func:`room_modes` is the honest answer,
while above the **Schroeder frequency**
(:func:`phonometry.room.schroeder_frequency`) the modes overlap so densely
that only the statistical description of
:mod:`phonometry.room.steady_field` remains useful. Marking that frequency on
the mode ladder is the point of :meth:`RoomModesResult.plot`.

This module describes the *eigenfrequencies* of an empty rectangular box. It is
not a prediction of the level at a receiver: for that, use the image-source
model (:mod:`phonometry.room.image_source`) or the steady-state statistical
field (:mod:`phonometry.room.steady_field`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from .._internal.types import as_float_or_array
from .._internal.validation import require_positive
from .steady_field import schroeder_frequency

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from numpy.typing import ArrayLike, NDArray

#: Speed of sound in air used when none is given, in m/s (20 degC).
DEFAULT_SPEED_OF_SOUND = 343.0

#: Mode kinds, ordered from the strongest (one axis) to the weakest (three).
MODE_KINDS: tuple[str, str, str] = ("axial", "tangential", "oblique")

#: Largest order lattice :func:`room_modes` will enumerate. The count grows as
#: ``f_max^3``, so a whole-audio-band request on an ordinary room asks for
#: hundreds of millions of triples; refusing with an explanation beats
#: exhausting memory. Enumeration is a low-frequency tool anyway: above the
#: Schroeder frequency use :func:`room_mode_count` and
#: :func:`room_modal_density`.
MAX_ENUMERATED_LATTICE = 20_000_000


def _require_dimensions(dimensions: ArrayLike) -> NDArray[np.float64]:
    """Validate the three room dimensions and return them as an array."""
    dims = np.asarray(dimensions, dtype=np.float64).ravel()
    if dims.size != 3:
        msg = "'dimensions' must hold the three room dimensions (lx, ly, lz) in m."
        raise ValueError(msg)
    if not np.all(np.isfinite(dims)) or np.any(dims <= 0.0):
        msg = "'dimensions' must be positive and finite."
        raise ValueError(msg)
    return dims


def _room_geometry(dims: NDArray[np.float64]) -> tuple[float, float, float]:
    """Volume ``V``, total wall area ``S`` and total edge length ``L``."""
    lx, ly, lz = (float(v) for v in dims)
    volume = lx * ly * lz
    surface = 2.0 * (lx * ly + lx * lz + ly * lz)
    edges = 4.0 * (lx + ly + lz)
    return volume, surface, edges


def room_mode_frequency(
    orders: ArrayLike,
    dimensions: ArrayLike,
    *,
    speed_of_sound: float = DEFAULT_SPEED_OF_SOUND,
) -> np.ndarray | float:
    r"""Eigenfrequency of a rectangular-room mode (Long Equation (8.43)).

    .. math::

       f = \frac{c_0}{2}
       \sqrt{ \left(\frac{n_x}{l_x}\right)^2
       + \left(\frac{n_y}{l_y}\right)^2
       + \left(\frac{n_z}{l_z}\right)^2 }

    :param orders: Mode orders ``(nx, ny, nz)``, non-negative integers; either
        a single triple or an ``(N, 3)`` array of triples.
    :param dimensions: Room dimensions ``(lx, ly, lz)``, m.
    :param speed_of_sound: Speed of sound ``c0``, m/s (default
        :data:`DEFAULT_SPEED_OF_SOUND`).
    :return: The modal frequency in Hz; a float for a single triple, otherwise
        one frequency per row.
    """
    dims = _require_dimensions(dimensions)
    c0 = require_positive(speed_of_sound, "speed_of_sound")
    n = np.asarray(orders, dtype=np.float64)
    if n.ndim == 1:
        n = n.reshape(1, -1)
        single = True
    else:
        single = False
    if n.ndim != 2 or n.shape[1] != 3:
        msg = "'orders' must be a triple (nx, ny, nz) or an (N, 3) array."
        raise ValueError(msg)
    if np.any(n < 0.0) or np.any(n != np.floor(n)):
        msg = "'orders' must be non-negative integers."
        raise ValueError(msg)
    freqs = 0.5 * c0 * np.sqrt(np.sum((n / dims) ** 2, axis=1))
    return float(freqs[0]) if single else freqs


def room_mode_count(
    frequency: ArrayLike,
    dimensions: ArrayLike,
    *,
    speed_of_sound: float = DEFAULT_SPEED_OF_SOUND,
) -> np.ndarray | float:
    r"""Integrated number of modes below ``frequency`` (Long Equation (8.45)).

    .. math::

       N(f) = \frac{4 \pi}{3} V \left(\frac{f}{c_0}\right)^3
       + \frac{\pi}{4} S \left(\frac{f}{c_0}\right)^2
       + \frac{L}{8} \frac{f}{c_0}

    This is the
    Morse/Pierce lattice count with its wall-plane and axis corrections. It is
    a smooth asymptotic estimate, so it is only meaningful once several modes
    fit below ``f``; the exact enumeration is :func:`room_modes`.

    :param frequency: Upper frequency ``f``, Hz (scalar or array).
    :param dimensions: Room dimensions ``(lx, ly, lz)``, m.
    :param speed_of_sound: Speed of sound ``c0``, m/s.
    :return: The (non-integer) mode count.
    """
    dims = _require_dimensions(dimensions)
    c0 = require_positive(speed_of_sound, "speed_of_sound")
    f = np.asarray(frequency, dtype=np.float64)
    if np.any(f < 0.0) or not np.all(np.isfinite(f)):
        msg = "'frequency' must be non-negative and finite."
        raise ValueError(msg)
    volume, surface, edges = _room_geometry(dims)
    x = f / c0
    n = (
        (4.0 * np.pi / 3.0) * volume * x**3
        + (np.pi / 4.0) * surface * x**2
        + (edges / 8.0) * x
    )
    return as_float_or_array(n)


def room_modal_density(
    frequency: ArrayLike,
    dimensions: ArrayLike,
    *,
    speed_of_sound: float = DEFAULT_SPEED_OF_SOUND,
) -> np.ndarray | float:
    r"""Modal density :math:`dN/df` in modes per hertz (Long Equation (8.46)).

    .. math::

       \frac{dN}{df} = \frac{4 \pi V f^2}{c_0^3}
       + \frac{\pi}{2} \frac{S f}{c_0^2} + \frac{L}{8 c_0}

    This is the
    derivative of :func:`room_mode_count`. Its reciprocal is the mean spacing
    between adjacent eigenfrequencies.

    :param frequency: Frequency ``f``, Hz (scalar or array).
    :param dimensions: Room dimensions ``(lx, ly, lz)``, m.
    :param speed_of_sound: Speed of sound ``c0``, m/s.
    :return: The modal density, modes/Hz.
    """
    dims = _require_dimensions(dimensions)
    c0 = require_positive(speed_of_sound, "speed_of_sound")
    f = np.asarray(frequency, dtype=np.float64)
    if np.any(f < 0.0) or not np.all(np.isfinite(f)):
        msg = "'frequency' must be non-negative and finite."
        raise ValueError(msg)
    volume, surface, edges = _room_geometry(dims)
    density = (
        4.0 * np.pi * volume * f**2 / c0**3
        + (np.pi / 2.0) * surface * f / c0**2
        + edges / (8.0 * c0)
    )
    return as_float_or_array(density)


@dataclass(frozen=True)
class RoomModesResult:
    """The normal modes of a rectangular room up to a frequency limit.

    :ivar orders: Mode orders, an ``(N, 3)`` integer array of ``(nx, ny, nz)``
        sorted by frequency.
    :ivar frequencies: Modal frequencies in ascending order, Hz.
    :ivar kinds: Mode kind per row, one of :data:`MODE_KINDS`.
    :ivar dimensions: Room dimensions ``(lx, ly, lz)``, m.
    :ivar speed_of_sound: Speed of sound ``c0`` used, m/s.
    :ivar max_frequency: Frequency limit of the enumeration, Hz.
    :ivar schroeder_frequency: Schroeder frequency, Hz, when a reverberation
        time was supplied; otherwise ``None``.
    """

    orders: np.ndarray
    frequencies: np.ndarray
    kinds: np.ndarray
    dimensions: tuple[float, float, float]
    speed_of_sound: float
    max_frequency: float
    schroeder_frequency: float | None = None

    @property
    def volume(self) -> float:
        """Room volume ``V``, m3."""
        lx, ly, lz = self.dimensions
        return lx * ly * lz

    @property
    def surface_area(self) -> float:
        """Total area ``S`` of the six walls, m2."""
        lx, ly, lz = self.dimensions
        return 2.0 * (lx * ly + lx * lz + ly * lz)

    @property
    def edge_length(self) -> float:
        """Sum ``L`` of the twelve edge lengths, m."""
        lx, ly, lz = self.dimensions
        return 4.0 * (lx + ly + lz)

    def count_by_kind(self) -> dict[str, int]:
        """Number of enumerated modes of each kind, keyed by :data:`MODE_KINDS`."""
        return {kind: int(np.count_nonzero(self.kinds == kind)) for kind in MODE_KINDS}

    def density(self, frequency: ArrayLike) -> np.ndarray | float:
        """Modal density at ``frequency`` for this room (Long Equation (8.46))."""
        return room_modal_density(
            frequency, self.dimensions, speed_of_sound=self.speed_of_sound
        )

    def plot(
        self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any
    ) -> Axes | np.ndarray:
        """Plot the mode ladder by kind and the modal density.

        The upper panel stems every enumerated mode at its frequency, coloured
        axial / tangential / oblique; the lower panel draws the smooth modal
        density (Long Equation (8.46)). The Schroeder frequency, when known,
        is marked on both. Requires matplotlib
        (``pip install phonometry[plot]``).
        """
        from .._i18n import check_language
        from .._plot.room import plot_room_modes

        check_language(language)
        return plot_room_modes(self, ax=ax, language=language, **kwargs)


def room_modes(
    dimensions: ArrayLike,
    *,
    max_frequency: float = 200.0,
    speed_of_sound: float = DEFAULT_SPEED_OF_SOUND,
    reverberation_time: float | None = None,
) -> RoomModesResult:
    r"""Enumerate the normal modes of a rectangular room (Long Chapter 8).

    Every order triple ``(nx, ny, nz)`` whose frequency
    :math:`(c_0/2) \sqrt{(n_x/l_x)^2 + (n_y/l_y)^2 + (n_z/l_z)^2}` does not
    exceed
    ``max_frequency`` is listed, sorted by frequency and classified as axial,
    tangential or oblique. The trivial ``(0, 0, 0)`` mode (the static pressure
    solution) is excluded.

    :param dimensions: Room dimensions ``(lx, ly, lz)``, m.
    :param max_frequency: Highest modal frequency to enumerate, Hz.
    :param speed_of_sound: Speed of sound ``c0``, m/s (default
        :data:`DEFAULT_SPEED_OF_SOUND`).
    :param reverberation_time: Optional reverberation time ``T``, s. When
        given, the Schroeder frequency :math:`2000 \sqrt{T/V}` is computed and
        carried in the result (and marked by :meth:`RoomModesResult.plot`).
    :return: A :class:`RoomModesResult`.
    """
    dims = _require_dimensions(dimensions)
    c0 = require_positive(speed_of_sound, "speed_of_sound")
    f_max = require_positive(max_frequency, "max_frequency")

    # The highest order along each axis is set by that axis alone:
    # f(n,0,0) = n c0 / (2 lx) <= f_max.
    n_max = np.floor(2.0 * f_max * dims / c0).astype(int)
    lattice = float(np.prod(n_max + 1, dtype=np.float64))
    if lattice > MAX_ENUMERATED_LATTICE:
        msg = (
            f"Enumerating the modes of a "
            f"{dims[0]:g} x {dims[1]:g} x {dims[2]:g} m room up to "
            f"{f_max:g} Hz needs {lattice:.3g} order triples, above the "
            f"{MAX_ENUMERATED_LATTICE:g} limit. Lower 'max_frequency', or use "
            f"room_mode_count / room_modal_density, which are accurate "
            f"wherever the count is this large."
        )
        raise ValueError(msg)
    grids = np.meshgrid(*(np.arange(n + 1) for n in n_max), indexing="ij")
    orders = np.stack([g.ravel() for g in grids], axis=1)
    orders = orders[np.any(orders > 0, axis=1)]  # drop (0, 0, 0)

    freqs = 0.5 * c0 * np.sqrt(np.sum((orders / dims) ** 2, axis=1))
    keep = freqs <= f_max
    orders, freqs = orders[keep], freqs[keep]

    # Sort by frequency, breaking ties on the order triple so the listing is
    # deterministic for the degenerate modes of a cubic or 1:2 room.
    idx = np.lexsort((orders[:, 2], orders[:, 1], orders[:, 0], freqs))
    orders, freqs = orders[idx], freqs[idx]

    non_zero = np.count_nonzero(orders, axis=1)
    kinds = np.array(MODE_KINDS, dtype="<U10")[non_zero - 1]

    fs: float | None = None
    if reverberation_time is not None:
        volume, _, _ = _room_geometry(dims)
        fs = float(np.asarray(schroeder_frequency(reverberation_time, volume))[()])

    return RoomModesResult(
        orders=orders.astype(int),
        frequencies=freqs,
        kinds=kinds,
        dimensions=(float(dims[0]), float(dims[1]), float(dims[2])),
        speed_of_sound=float(c0),
        max_frequency=float(f_max),
        schroeder_frequency=fs,
    )
