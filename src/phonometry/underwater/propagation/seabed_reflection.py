#  Copyright (c) 2026. Jose Manuel Requena Plens
r"""Plane-wave reflection at the seabed (fluid-fluid Rayleigh model).

A plane wave in the water column (density :math:`\rho_1`, sound speed
``c1``) striking a fluid sediment half-space (:math:`\rho_2`, ``c2``)
reflects with the Rayleigh pressure reflection coefficient (Medwin & Clay,
Eq. 2.6.11a). Using the grazing angle :math:`\varphi` (measured from the
interface), the angle of incidence from the normal is
:math:`\theta_1 = 90^\circ - \varphi`, Snell's law gives
:math:`\sin\theta_2 = (c_2/c_1) \cos\varphi` and

.. math::

   R = \frac{\rho_2 c_2 \sin\varphi - \rho_1 c_1 \cos\theta_2}
   {\rho_2 c_2 \sin\varphi + \rho_1 c_1 \cos\theta_2}

* :func:`critical_angle` -- the critical grazing angle
  :math:`\varphi_\mathrm{c} = \arccos(c_1/c_2)` (only when ``c2 > c1``); below it the
  wave is totally reflected (:math:`\lvert R \rvert = 1`).
* :func:`reflection_coefficient` -- the complex ``R`` per grazing angle.
* :func:`seabed_reflection` -- the complex ``R``, its magnitude ``|R|`` and the
  bottom loss bundled with the interface parameters into a
  :class:`SeabedReflection` whose ``.plot()`` draws ``|R|`` versus grazing angle.
* :func:`bottom_reflection_loss` -- the bottom loss
  :math:`\mathrm{BL} = -20 \log_{10} \lvert R \rvert` (dB), returned as a
  :class:`BottomLossResult` with a ``.plot()``.

Lossless fluid-fluid model (real :math:`\rho`/``c``); sediment attenuation is
out of scope. Densities enter only through the impedance ratio, so any
consistent unit works (kg/m³ by convention).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from ..._internal.validation import (
    require_positive,
    require_ranks,
    require_same_length,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from numpy.typing import NDArray

# Upper edge of the grazing-angle domain: 90° is normal incidence.
_NORMAL_INCIDENCE_DEG = 90.0


def critical_angle(c1: float, c2: float) -> float:
    r"""Critical grazing angle :math:`\varphi_\mathrm{c} = \arccos(c_1/c_2)`, in
    degrees.

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
        msg = "A critical angle exists only when c2 > c1 (faster sediment)."
        raise ValueError(msg)
    return float(np.degrees(np.arccos(cw / cs)))


def reflection_coefficient(
    grazing_angle: NDArray[np.float64] | list[float] | float,
    *,
    rho1: float,
    c1: float,
    rho2: float,
    c2: float,
) -> NDArray[np.complex128]:
    r"""Complex plane-wave pressure reflection coefficient at the seabed.

    :param grazing_angle: Grazing angle(s) :math:`\varphi` from the
        interface, in degrees (``0`` grazing to ``90`` normal incidence).
    :param rho1: Water density :math:`\rho_1` (any consistent unit; kg/m³ by
        convention).
    :param c1: Sound speed in the water ``c1``, in m/s.
    :param rho2: Sediment density :math:`\rho_2`.
    :param c2: Sound speed in the sediment ``c2``, in m/s.
    :return: The complex reflection coefficient per grazing angle.
    :raises ValueError: If the inputs are invalid.
    """
    phi = np.atleast_1d(np.asarray(grazing_angle, dtype=np.float64))
    if phi.size == 0 or not np.all(np.isfinite(phi)):
        msg = "'grazing_angle' must be finite and non-empty."
        raise ValueError(msg)
    if np.any(phi < 0.0) or np.any(phi > _NORMAL_INCIDENCE_DEG):
        msg = "'grazing_angle' must be within [0, 90] degrees."
        raise ValueError(msg)
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
    r"""Bottom reflection loss versus grazing angle (fluid-fluid Rayleigh
    model).

    :ivar grazing_angle: Grazing angles, in degrees.
    :ivar reflection_loss: Bottom loss
        :math:`\mathrm{BL} = -20 \log_{10} \lvert R \rvert` per angle, in dB.
    :ivar reflection_coefficient: Complex reflection coefficient per angle.
    :ivar critical_angle: The critical grazing angle, in degrees, or ``None`` if
        the sediment is not faster than the water.
    """

    grazing_angle: NDArray[np.float64]
    reflection_loss: NDArray[np.float64]
    reflection_coefficient: NDArray[np.complex128]
    critical_angle: float | None

    def __post_init__(self) -> None:
        r"""Reject a loss curve that does not run over its own angles.

        The three arrays are one angle sweep written down three times:
        :func:`bottom_reflection_loss` evaluates ``R`` once at the angles it
        was handed and takes :math:`-20 \log_{10} \lvert R \rvert` of that
        very array, so entry ``i`` of each is the same grazing angle. Nothing
        in the result records which angle an entry belongs to except its
        position.

        Only two of the three are drawn: :func:`plot_bottom_loss` calls
        ``ax.plot(grazing_angle, reflection_loss)``, so a disagreement
        between those two stops the figure with matplotlib's own complaint,
        ``x and y must have same first dimension, but have shapes (5,) and
        (91,)`` -- shapes only, neither field named, and the reader is left to
        work out which of the two was short. ``reflection_coefficient`` is
        never read by the figure at all, so a short one is drawn straight past
        in silence: the picture comes out complete over 91 angles while the
        coefficient beside it holds 5, waiting for whoever pairs ``R`` with
        the angles next. A single-entry ``R`` is the quietest of the lot,
        because numpy broadcasts it: one angle's reflection coefficient
        spreads over all 91 and the arithmetic returns a full-length answer
        with no complaint anywhere.

        The ranks are pinned because a count cannot see an extra axis. A
        ``reflection_loss`` of shape ``(91, 2)`` -- two sediments stacked, or
        ``(angle, value)`` pairs kept in one array -- has 91 on its first axis
        like everything else, passes the length check, and reaches matplotlib,
        which quietly draws both columns: the figure comes back with two
        curves and the legend row ``Bottom loss`` printed twice.

        :raises ValueError: if the three arrays disagree, or one carries an
            axis beyond the angle sweep.
        """
        require_ranks(
            self,
            grazing_angle=1,
            reflection_loss=1,
            reflection_coefficient=1,
        )
        require_same_length(
            self,
            "grazing_angle",
            "reflection_loss",
            "reflection_coefficient",
            axis="grazing angle",
        )

    def plot(
        self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any
    ) -> Axes:
        """Plot the bottom loss versus grazing angle with the critical angle."""
        from ..._i18n import check_language
        from ..._plot.underwater import plot_bottom_loss

        return plot_bottom_loss(
            self, ax=ax, language=check_language(language), **kwargs
        )


def bottom_reflection_loss(
    grazing_angle: NDArray[np.float64] | list[float] | float,
    *,
    rho1: float = 1000.0,
    c1: float = 1500.0,
    rho2: float,
    c2: float,
) -> BottomLossResult:
    r"""Bottom reflection loss :math:`\mathrm{BL} = -20 \log_{10} \lvert R \rvert`
    versus grazing angle (dB).

    At an angle of intromission :math:`R = 0` and the loss is ``inf`` (total
    transmission into the sediment); that is a legitimate bottom-loss value,
    returned without warning.

    :param grazing_angle: Grazing angle(s) :math:`\varphi` from the
        interface, in degrees.
    :param rho1: Water density :math:`\rho_1` (default 1000 kg/m³).
    :param c1: Sound speed in the water ``c1``, in m/s (default 1500).
    :param rho2: Sediment density :math:`\rho_2`.
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
    r"""Plane-wave seabed reflection coefficient versus grazing angle.

    Bundles the complex Rayleigh reflection coefficient ``R`` over a
    grazing-angle grid with its magnitude ``|R|``, the bottom loss
    :math:`\mathrm{BL} = -20 \log_{10} \lvert R \rvert` and the fluid-fluid
    interface parameters, so the classic ``|R|`` versus grazing-angle curve
    can be drawn with :meth:`plot`. Build it with :func:`seabed_reflection`;
    the frozen instance is a thin, plottable wrapper and re-runs none of the
    maths.

    :ivar grazing_angle: Grazing angles :math:`\varphi` from the interface,
        in degrees.
    :ivar reflection_coefficient: Complex pressure reflection coefficient ``R``
        per grazing angle.
    :ivar magnitude: Reflection-coefficient magnitude ``|R|`` per grazing angle
        (``1`` below the critical angle for a faster sediment).
    :ivar bottom_loss: Bottom loss
        :math:`\mathrm{BL} = -20 \log_{10} \lvert R \rvert` per grazing angle, in
        dB.
    :ivar critical_angle: The critical grazing angle, in degrees, or ``None`` if
        the sediment is not faster than the water.
    :ivar rho1: Water density :math:`\rho_1`.
    :ivar c1: Sound speed in the water ``c1``, in m/s.
    :ivar rho2: Sediment density :math:`\rho_2`.
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

    def __post_init__(self) -> None:
        r"""Reject a reflection record whose four columns are not one sweep.

        :func:`seabed_reflection` evaluates ``R`` once over the angles it was
        given and derives ``|R|`` and the bottom loss from that same array, so
        the four arrays are one grazing-angle sweep recorded four times and
        entry ``i`` of each belongs to the same angle. The class carries the
        interface parameters precisely so that the record is self-contained;
        an angle axis that fits none of the three quantities beside it makes
        it self-contained about the wrong sweep.

        :func:`plot_seabed_reflection` draws only ``|R|``: against a
        mismatched ``grazing_angle`` it stops with matplotlib's ``x and y must
        have same first dimension, but have shapes (91,) and (5,)`` (the same
        both ways round, and either way the message names shapes rather than
        fields). The other two never reach the figure, and that is where the
        silence is: a ``bottom_loss`` of 5 entries beside 91 angles draws a
        complete ``|R|`` curve over all 91, and the loss in decibels -- the
        quantity a sonar budget actually spends -- is wrong and unmentioned.

        The ranks are pinned because a count passes an extra axis through. A
        ``magnitude`` of shape ``(91, 2)`` counts 91 on its first axis exactly
        as a single sweep does, and matplotlib draws every column it is given:
        the figure comes back carrying two ``|R|`` curves and repeating the
        ``$|R|$`` legend row, with nothing to say which sediment is which.

        :raises ValueError: if the four arrays disagree, or one carries an
            axis beyond the angle sweep.
        """
        require_ranks(
            self,
            grazing_angle=1,
            reflection_coefficient=1,
            magnitude=1,
            bottom_loss=1,
        )
        require_same_length(
            self,
            "grazing_angle",
            "reflection_coefficient",
            "magnitude",
            "bottom_loss",
            axis="grazing angle",
        )

    def plot(
        self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any
    ) -> Axes:
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
        from ..._i18n import check_language
        from ..._plot.underwater import plot_seabed_reflection

        return plot_seabed_reflection(
            self, ax=ax, language=check_language(language), **kwargs
        )


def seabed_reflection(
    grazing_angle: NDArray[np.float64] | list[float] | float,
    *,
    rho1: float = 1000.0,
    c1: float = 1500.0,
    rho2: float,
    c2: float,
) -> SeabedReflection:
    r"""Build a plottable seabed reflection-coefficient result (Rayleigh
    model).

    Evaluates :func:`reflection_coefficient` at ``grazing_angle`` for the given
    fluid-fluid interface and bundles the complex ``R``, its magnitude ``|R|``,
    the bottom loss :math:`\mathrm{BL} = -20 \log_{10} \lvert R \rvert` and the
    interface parameters into a :class:`SeabedReflection` that exposes
    ``.plot()``. The maths is unchanged; this is a thin, plottable wrapper
    around the existing function (the same ``ValueError`` cases apply). At an
    angle of intromission :math:`R = 0` and the bottom loss is ``inf`` (total
    transmission), returned without warning.

    :param grazing_angle: Grazing angle(s) :math:`\varphi` from the
        interface, in degrees (``0`` grazing to ``90`` normal incidence).
    :param rho1: Water density :math:`\rho_1` (default 1000 kg/m³).
    :param c1: Sound speed in the water ``c1``, in m/s (default 1500).
    :param rho2: Sediment density :math:`\rho_2`.
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
