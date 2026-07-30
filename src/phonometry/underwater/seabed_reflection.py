#  Copyright (c) 2026. Jose Manuel Requena Plens
"""
Plane-wave reflection at the seabed (fluid-fluid Rayleigh model).

A plane wave in the water column (density ``ρ1``, sound speed ``c1``) striking a
fluid sediment half-space (``ρ2``, ``c2``) reflects with the Rayleigh pressure
reflection coefficient (Medwin & Clay, Eq. 2.6.11a). Using the grazing angle
``φ`` (measured from the interface), the angle of incidence from the normal is
``θ1 = 90° − φ``, Snell's law gives ``sinθ2 = (c2/c1)·cosφ`` and
``R = (ρ2·c2·sinφ − ρ1·c1·cosθ2) / (ρ2·c2·sinφ + ρ1·c1·cosθ2)``.

* :func:`critical_angle` -- the critical grazing angle ``φc = arccos(c1/c2)``
  (only when ``c2 > c1``); below it the wave is totally reflected (``|R| = 1``).
* :func:`reflection_coefficient` -- the complex ``R`` per grazing angle.
* :func:`seabed_reflection` -- the complex ``R``, its magnitude ``|R|`` and the
  bottom loss bundled with the interface parameters into a
  :class:`SeabedReflection` whose ``.plot()`` draws ``|R|`` versus grazing angle.
* :func:`bottom_reflection_loss` -- the bottom loss ``BL = −20·lg|R|`` (dB),
  returned as a :class:`BottomLossResult` with a ``.plot()``.

Lossless fluid-fluid model (real ``ρ``/``c``); sediment attenuation is out of
scope. Densities enter only through the impedance ratio, so any consistent unit
works (kg/m³ by convention).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from .._internal.validation import require_positive

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from numpy.typing import NDArray


def critical_angle(c1: float, c2: float) -> float:
    """Critical grazing angle ``φc = arccos(c1/c2)``, in degrees.

    Defined only when the sediment is faster than the water (``c2 > c1``); at and
    below this grazing angle the wave is totally reflected.

    :param c1: Sound speed in the water, in m/s.
    :param c2: Sound speed in the sediment, in m/s.
    :return: The critical grazing angle, in degrees.
    :raises ValueError: If ``c2 <= c1`` (no critical angle exists).
    """
    cw = require_positive(c1, "c1")
    cs = require_positive(c2, "c2")
    if cs <= cw:
        raise ValueError("A critical angle exists only when c2 > c1 (faster sediment).")
    return float(np.degrees(np.arccos(cw / cs)))


def reflection_coefficient(
    grazing_angle: NDArray[np.float64] | list[float] | float,
    *,
    rho1: float,
    c1: float,
    rho2: float,
    c2: float,
) -> NDArray[np.complex128]:
    """Complex plane-wave pressure reflection coefficient at the seabed.

    :param grazing_angle: Grazing angle(s) ``φ`` from the interface, in degrees
        (``0`` grazing to ``90`` normal incidence).
    :param rho1: Water density ``ρ1`` (any consistent unit; kg/m³ by convention).
    :param c1: Sound speed in the water ``c1``, in m/s.
    :param rho2: Sediment density ``ρ2``.
    :param c2: Sound speed in the sediment ``c2``, in m/s.
    :return: The complex reflection coefficient per grazing angle.
    :raises ValueError: If the inputs are invalid.
    """
    phi = np.atleast_1d(np.asarray(grazing_angle, dtype=np.float64))
    if phi.size == 0 or not np.all(np.isfinite(phi)):
        raise ValueError("'grazing_angle' must be finite and non-empty.")
    if np.any(phi < 0.0) or np.any(phi > 90.0):
        raise ValueError("'grazing_angle' must be within [0, 90] degrees.")
    r1 = require_positive(rho1, "rho1")
    cw = require_positive(c1, "c1")
    r2 = require_positive(rho2, "rho2")
    cs = require_positive(c2, "c2")
    phi_rad = np.radians(phi)
    cos_t1 = np.sin(phi_rad)  # θ1 from normal = 90° − φ
    sin_t2 = (cs / cw) * np.cos(phi_rad)
    cos_t2 = np.sqrt(1.0 - sin_t2.astype(np.complex128) ** 2)
    z1 = r1 * cw
    z2 = r2 * cs
    num = z2 * cos_t1 - z1 * cos_t2
    den = z2 * cos_t1 + z1 * cos_t2
    with np.errstate(divide="ignore", invalid="ignore"):
        r = num / den
    # Singular limit: with no sound-speed contrast (c1 == c2) at exactly grazing
    # incidence, cos_t1 = cos_t2 = 0 gives 0/0 (NaN). The angle-independent limit
    # is the normal-incidence coefficient (z2 − z1)/(z2 + z1).
    r = np.where(np.isnan(r), (z2 - z1) / (z2 + z1), r)
    return np.asarray(r, dtype=np.complex128)


@dataclass(frozen=True)
class BottomLossResult:
    """Bottom reflection loss versus grazing angle (fluid-fluid Rayleigh model).

    :ivar grazing_angle: Grazing angles, in degrees.
    :ivar reflection_loss: Bottom loss ``BL = −20·lg|R|`` per angle, in dB.
    :ivar reflection_coefficient: Complex reflection coefficient per angle.
    :ivar critical_angle: The critical grazing angle, in degrees, or ``None`` if
        the sediment is not faster than the water.
    """

    grazing_angle: NDArray[np.float64]
    reflection_loss: NDArray[np.float64]
    reflection_coefficient: NDArray[np.complex128]
    critical_angle: float | None

    def plot(self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any) -> Axes:
        """Plot the bottom loss versus grazing angle with the critical angle."""
        from .._i18n import check_language
        from .._plot.underwater import plot_bottom_loss

        return plot_bottom_loss(self, ax=ax, language=check_language(language), **kwargs)


def bottom_reflection_loss(
    grazing_angle: NDArray[np.float64] | list[float] | float,
    *,
    rho1: float = 1000.0,
    c1: float = 1500.0,
    rho2: float,
    c2: float,
) -> BottomLossResult:
    """Bottom reflection loss ``BL = −20·lg|R|`` versus grazing angle (dB).

    At an angle of intromission ``R = 0`` and the loss is ``inf`` (total
    transmission into the sediment); that is a legitimate bottom-loss value,
    returned without warning.

    :param grazing_angle: Grazing angle(s) ``φ`` from the interface, in degrees.
    :param rho1: Water density ``ρ1`` (default 1000 kg/m³).
    :param c1: Sound speed in the water ``c1``, in m/s (default 1500).
    :param rho2: Sediment density ``ρ2``.
    :param c2: Sound speed in the sediment ``c2``, in m/s.
    :return: A :class:`BottomLossResult`.
    :raises ValueError: If the inputs are invalid.
    """
    phi = np.atleast_1d(np.asarray(grazing_angle, dtype=np.float64))
    r = reflection_coefficient(phi, rho1=rho1, c1=c1, rho2=rho2, c2=c2)
    # At an intromission angle |R| = 0 and the bottom loss is legitimately
    # infinite (total transmission); silence the log10(0) warning.
    with np.errstate(divide="ignore"):
        loss = -20.0 * np.log10(np.abs(r))
    phi_c = critical_angle(c1, c2) if float(c2) > float(c1) else None
    return BottomLossResult(
        grazing_angle=phi,
        reflection_loss=np.asarray(loss, dtype=np.float64),
        reflection_coefficient=r,
        critical_angle=phi_c,
    )


@dataclass(frozen=True)
class SeabedReflection:
    """Plane-wave seabed reflection coefficient versus grazing angle.

    Bundles the complex Rayleigh reflection coefficient ``R`` over a
    grazing-angle grid with its magnitude ``|R|``, the bottom loss
    ``BL = −20·lg|R|`` and the fluid-fluid interface parameters, so the classic
    ``|R|`` versus grazing-angle curve can be drawn with :meth:`plot`. Build it
    with :func:`seabed_reflection`; the frozen instance is a thin, plottable
    wrapper and re-runs none of the maths.

    :ivar grazing_angle: Grazing angles ``φ`` from the interface, in degrees.
    :ivar reflection_coefficient: Complex pressure reflection coefficient ``R``
        per grazing angle.
    :ivar magnitude: Reflection-coefficient magnitude ``|R|`` per grazing angle
        (``1`` below the critical angle for a faster sediment).
    :ivar bottom_loss: Bottom loss ``BL = −20·lg|R|`` per grazing angle, in dB.
    :ivar critical_angle: The critical grazing angle, in degrees, or ``None`` if
        the sediment is not faster than the water.
    :ivar rho1: Water density ``ρ1``.
    :ivar c1: Sound speed in the water ``c1``, in m/s.
    :ivar rho2: Sediment density ``ρ2``.
    :ivar c2: Sound speed in the sediment ``c2``, in m/s.
    """

    grazing_angle: NDArray[np.float64]
    reflection_coefficient: NDArray[np.complex128]
    magnitude: NDArray[np.float64]
    bottom_loss: NDArray[np.float64]
    critical_angle: float | None
    rho1: float
    c1: float
    rho2: float
    c2: float

    def plot(self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any) -> Axes:
        """Plot the reflection-coefficient magnitude ``|R|`` versus grazing angle.

        Draws ``|R|`` on a linear grazing-angle axis (0..90°), marking the
        critical angle when the sediment is faster than the water. Requires
        matplotlib (``pip install phonometry[plot]``); returns the
        :class:`~matplotlib.axes.Axes` and never calls ``plt.show``.

        :param ax: Existing axes, or ``None`` to create a figure.
        :param language: ``"en"`` (default) or ``"es"``.
        :param kwargs: Forwarded to the ``|R|`` curve ``plot`` call.
        :return: The axes.
        """
        from .._i18n import check_language
        from .._plot.underwater import plot_seabed_reflection

        return plot_seabed_reflection(self, ax=ax, language=check_language(language), **kwargs)


def seabed_reflection(
    grazing_angle: NDArray[np.float64] | list[float] | float,
    *,
    rho1: float = 1000.0,
    c1: float = 1500.0,
    rho2: float,
    c2: float,
) -> SeabedReflection:
    """Build a plottable seabed reflection-coefficient result (Rayleigh model).

    Evaluates :func:`reflection_coefficient` at ``grazing_angle`` for the given
    fluid-fluid interface and bundles the complex ``R``, its magnitude ``|R|``,
    the bottom loss ``BL = −20·lg|R|`` and the interface parameters into a
    :class:`SeabedReflection` that exposes ``.plot()``. The maths is unchanged;
    this is a thin, plottable wrapper around the existing function (the same
    ``ValueError`` cases apply). At an angle of intromission ``R = 0`` and the
    bottom loss is ``inf`` (total transmission), returned without warning.

    :param grazing_angle: Grazing angle(s) ``φ`` from the interface, in degrees
        (``0`` grazing to ``90`` normal incidence).
    :param rho1: Water density ``ρ1`` (default 1000 kg/m³).
    :param c1: Sound speed in the water ``c1``, in m/s (default 1500).
    :param rho2: Sediment density ``ρ2``.
    :param c2: Sound speed in the sediment ``c2``, in m/s.
    :return: A frozen :class:`SeabedReflection`.
    :raises ValueError: If the inputs are invalid.
    """
    phi = np.atleast_1d(np.asarray(grazing_angle, dtype=np.float64))
    r = reflection_coefficient(phi, rho1=rho1, c1=c1, rho2=rho2, c2=c2)
    magnitude = np.abs(r)
    # At an intromission angle |R| = 0 and the bottom loss is legitimately
    # infinite (total transmission); silence the log10(0) warning.
    with np.errstate(divide="ignore"):
        loss = -20.0 * np.log10(magnitude)
    phi_c = critical_angle(c1, c2) if float(c2) > float(c1) else None
    return SeabedReflection(
        grazing_angle=phi,
        reflection_coefficient=r,
        magnitude=np.asarray(magnitude, dtype=np.float64),
        bottom_loss=np.asarray(loss, dtype=np.float64),
        critical_angle=phi_c,
        rho1=float(rho1),
        c1=float(c1),
        rho2=float(rho2),
        c2=float(c2),
    )
