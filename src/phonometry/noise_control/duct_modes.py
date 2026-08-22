#  Copyright (c) 2026. Jose Manuel Requena Plens
r"""Higher-order acoustic modes in ducts, with the mean-flow cut-on shift.

Every plane-wave duct method has the same expiry date: the frequency at which
the first higher-order mode cuts on. Below it a duct carries only plane waves,
the four-pole (transfer-matrix) silencer algebra of
:mod:`phonometry.noise_control.silencers` is exact and a single sound pressure
describes the whole cross section. Above it several modes propagate at once,
each with its own axial wavenumber, and a plane-wave prediction quietly stops
being right.

This module implements the cut-on analysis of Norton & Karczub, *Fundamentals
of Noise and Vibration Analysis for Engineers* 2nd ed., section 7.3:

* circular ducts (Eq. 7.6),
  :math:`(f_\mathrm{co})_{pq} = \pi \alpha_{pq} c / (2 \pi a_\mathrm{i})`, with the
  :math:`\pi \alpha_{pq}` eigenvalues of Table 7.1 that solve
  :math:`J'_p(\kappa_{pq} a_\mathrm{i}) = 0` (the first is 1.8412, which sets the
  plane-wave limit :math:`k a_\mathrm{i} < 1.8412`);
* rectangular ducts (Eq. 7.10),
  :math:`(f_\mathrm{co})_{pq} = (c / 2) \sqrt{(p / a)^2 + (q / b)^2}`;
* the **mean-flow correction** (Eq. 7.8): a uniform axial flow of Mach number
  ``M`` lowers every cut-on frequency by :math:`\sqrt{1 - M^2}` and moves the
  cut-on from :math:`k_x = 0` to
  :math:`k_x = -M \kappa_{pq} / \sqrt{1 - M^2}` (Eq. 7.9), so the
  dispersion curve, symmetric about the frequency axis in still air, becomes
  asymmetric. For a turbulent rather than uniform profile Norton notes that
  replacing ``M`` by the centre-line Mach number ``M_0`` represents the
  convective effect adequately.

The practical consequence in a ventilation duct is blunt: a 0.65 m x 0.4 m
air-conditioning duct cuts on at 264 Hz, so plane-wave silencer algebra is
valid only in the 63 Hz to 250 Hz octaves. :func:`plane_wave_limit` returns
that frequency and :func:`warn_above_plane_wave_limit` issues the
:class:`PlaneWaveWarning` the silencer and duct-path results raise on the
caller's behalf.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from .._internal.validation import (
    require_non_negative,
    require_positive,
    require_ranks,
    require_same_length,
)
from .._internal.warnings import PhonometryWarning

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from numpy.typing import ArrayLike, NDArray

_C_AIR = 343.0

#: Norton & Karczub Table 7.1 -- the ``pi alpha_pq`` eigenvalues solving
#: ``J'_p(kappa_pq a_i) = 0`` for the first twelve higher-order acoustic modes
#: of a rigid circular duct, keyed by the mode order ``(p, q)``: ``p`` plane
#: diametral nodal surfaces and ``q`` cylindrical nodal surfaces.
CIRCULAR_EIGENVALUES: dict[tuple[int, int], float] = {
    (1, 0): 1.8412,
    (2, 0): 3.0542,
    (0, 1): 3.8317,
    (3, 0): 4.2012,
    (4, 0): 5.3175,
    (1, 1): 5.3314,
    (5, 0): 6.4156,
    (2, 1): 6.7061,
    (0, 2): 7.0156,
    (6, 0): 7.5013,
    (3, 1): 8.0152,
    (1, 2): 8.5363,
}

#: The plane-wave eigenvalue ``k a_i = 1.8412`` of the ``(1, 0)`` mode: below it
#: a rigid circular duct carries plane waves only.
PLANE_WAVE_EIGENVALUE = 1.8412


class PlaneWaveWarning(PhonometryWarning):
    """A plane-wave duct method is evaluated above the first cut-on frequency.

    Raised by the results that assume one-dimensional propagation (the reactive
    silencers, the duct-path cascade) when the analysis reaches frequencies at
    which higher-order modes propagate in the duct cross section. The numbers
    are still returned: above cut-on they describe the plane-wave mode only,
    and a measurement will show the extra modes.
    """


@dataclass(frozen=True)
class DuctModeResult:
    r"""Cut-on frequencies of the higher-order acoustic modes of a duct.

    :ivar modes: Mode orders ``(p, q)``, ordered by ascending no-flow cut-on
        frequency.
    :ivar cut_on: Cut-on frequency of each mode with the mean flow applied, Hz.
    :ivar cut_on_no_flow: Cut-on frequency of each mode in still air, Hz.
    :ivar axial_wavenumber: Axial wavenumber ``k_x`` at cut-on, 1/m (zero
        without flow, negative with it).
    :ivar mach: Mean-flow Mach number :math:`M = U / c`.
    :ivar section: ``"circular"`` or ``"rectangular"``.
    :ivar label: A short human label of the duct.
    """

    modes: tuple[tuple[int, int], ...]
    cut_on: np.ndarray
    cut_on_no_flow: np.ndarray
    axial_wavenumber: np.ndarray
    mach: float
    section: str
    label: str

    def __post_init__(self) -> None:
        """Reject a ladder whose curves do not all describe the same modes.

        The cut-on chart labels one tick per entry of ``modes`` and plots both
        cut-on curves against those ticks, so the mode order a frequency is
        read under is the order standing at its position in ``modes`` and
        nothing else. A count that disagrees mislabels nothing: the chart is
        the only reader that pairs the two, and it stops with matplotlib's
        complaint about an x and a y whose first dimensions differ, which
        names neither the curve nor the result it came from. There is no
        fiche behind the chart to truncate a column quietly, and no reader
        pairs :attr:`axial_wavenumber` with the modes at all, so a ladder
        that is never plotted carries its disagreement to the caller with
        nothing raised anywhere. The check is what names the field, and what
        names it where the ladder was built rather than where it is drawn.

        The rank half is the silent one. An extra axis still counts one entry
        per mode on its first, so it passes every length above: a
        ``(mode, 2)`` stack of two flow states draws one extra curve per
        column, all of them under the same legend entry, and
        :attr:`plane_wave_limit` minimises over the whole stack instead of
        over the ladder it claims to describe.

        :raises ValueError: if the mode orders and the per-mode arrays
            disagree, or a per-mode array carries an extra axis.
        """
        require_ranks(self, cut_on=1, cut_on_no_flow=1, axial_wavenumber=1)
        require_same_length(
            self,
            "modes",
            "cut_on",
            "cut_on_no_flow",
            "axial_wavenumber",
            axis="mode",
        )

    @property
    def plane_wave_limit(self) -> float:
        """The lowest cut-on frequency, Hz: plane waves only below it."""
        return float(np.min(self.cut_on))

    def plot(
        self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any
    ) -> Axes:
        """Plot the cut-on ladder, with and without flow, over the mode order.

        Requires matplotlib (``pip install phonometry[plot]``).

        :param ax: Existing axes, or ``None`` to create a figure.
        :param language: Label language, ``"en"`` (default) or ``"es"``.
        :param kwargs: Forwarded to the with-flow ``Axes.plot``.
        :return: The axes.
        """
        from .._i18n import check_language
        from .._plot.noise_control import plot_duct_modes

        check_language(language)
        return plot_duct_modes(self, ax=ax, language=language, **kwargs)


def _mach(flow_velocity: float, speed_of_sound: float) -> tuple[float, float]:
    """Validate the flow condition and return ``(mach, sqrt(1 - M^2))``."""
    c = require_positive(speed_of_sound, "speed_of_sound")
    u = require_non_negative(flow_velocity, "flow_velocity")
    mach = u / c
    if mach >= 1.0:
        msg = (
            "'flow_velocity' must be subsonic; the cut-on correction "
            "sqrt(1 - M^2) is undefined at and above M = 1."
        )
        raise ValueError(msg)
    return mach, float(np.sqrt(1.0 - mach**2))


def circular_duct_cut_on(
    diameter: float,
    *,
    flow_velocity: float = 0.0,
    speed_of_sound: float = _C_AIR,
    count: int = 6,
) -> DuctModeResult:
    r"""Cut-on frequencies of the higher-order modes of a circular duct.

    Norton & Karczub Eq. 7.6 with the Table 7.1 eigenvalues, corrected for a
    uniform axial mean flow by Eqs. 7.8 and 7.9:

    .. math::

       (f_\mathrm{co})_{pq} = \frac{\pi \alpha_{pq} c \sqrt{1 - M^2}}{2 \pi a_\mathrm{i}}

       k_x = \frac{-M \kappa_{pq}}{\sqrt{1 - M^2}}, \qquad
       \kappa_{pq} = \frac{\pi \alpha_{pq}}{a_\mathrm{i}}

    The flow lowers every cut-on frequency and shifts the cut-on away from
    :math:`k_x = 0`: the mode is already travelling upstream, against the
    flow, at
    the frequency at which it appears. Only the first twelve modes are
    tabulated by Norton, so ``count`` cannot exceed twelve.

    :param diameter: Internal duct diameter, m (``a_i`` is half of it).
    :param flow_velocity: Mean axial flow speed ``U``, m/s (0 for still air;
        use the centre-line speed for a turbulent profile).
    :param speed_of_sound: Speed of sound ``c`` in the duct fluid, m/s.
    :param count: How many higher-order modes to return, 1 to 12.
    :return: A :class:`DuctModeResult`.
    :raises ValueError: For a non-positive diameter, a sonic or supersonic
        flow, or a ``count`` outside 1 to 12.
    """
    d = require_positive(diameter, "diameter")
    mach, beta = _mach(flow_velocity, speed_of_sound)
    if not 1 <= count <= len(CIRCULAR_EIGENVALUES):
        msg = f"'count' must be between 1 and {len(CIRCULAR_EIGENVALUES)}."
        raise ValueError(msg)
    radius = 0.5 * d
    ordered = sorted(CIRCULAR_EIGENVALUES.items(), key=lambda item: item[1])[:count]
    modes = tuple(mode for mode, _ in ordered)
    kappa = np.array([value / radius for _, value in ordered])
    no_flow = kappa * speed_of_sound / (2.0 * np.pi)
    return DuctModeResult(
        modes=modes,
        cut_on=no_flow * beta,
        cut_on_no_flow=no_flow,
        axial_wavenumber=-mach * kappa / beta,
        mach=mach,
        section="circular",
        label=f"Circular duct (D = {d * 1000:.0f} mm, M = {mach:.3f})",
    )


def rectangular_duct_cut_on(
    width: float,
    height: float,
    *,
    flow_velocity: float = 0.0,
    speed_of_sound: float = _C_AIR,
    count: int = 3,
) -> DuctModeResult:
    r"""Cut-on frequencies of the higher-order modes of a rectangular duct.

    Norton & Karczub Eq. 7.10, with the same :math:`\sqrt{1 - M^2}`
    convective factor as the circular case:

    .. math::

       (f_\mathrm{co})_{pq} = \frac{c}{2}
       \sqrt{\left(\frac{p}{a}\right)^2 + \left(\frac{q}{b}\right)^2}
       \sqrt{1 - M^2}

    where ``a`` and ``b`` are the cross-sectional dimensions and ``(p, q)``
    the mode order; ``(0, 0)`` is the plane wave and is excluded. The modes are
    returned in ascending cut-on order, so the first is always the half
    wavelength across the wider side.

    :param width: Cross-sectional dimension ``a``, m.
    :param height: Cross-sectional dimension ``b``, m.
    :param flow_velocity: Mean axial flow speed ``U``, m/s.
    :param speed_of_sound: Speed of sound ``c`` in the duct fluid, m/s.
    :param count: How many higher-order modes to return (at least 1).
    :return: A :class:`DuctModeResult`.
    :raises ValueError: For non-positive dimensions, a sonic or supersonic
        flow, or a non-positive ``count``.
    """
    a = require_positive(width, "width")
    b = require_positive(height, "height")
    mach, beta = _mach(flow_velocity, speed_of_sound)
    if count < 1:
        msg = "'count' must be at least 1."
        raise ValueError(msg)
    # A (count+1)-square grid of orders always contains the `count` lowest.
    orders = [
        (p, q) for p in range(count + 1) for q in range(count + 1) if (p, q) != (0, 0)
    ]
    kappa = np.array([np.pi * np.hypot(p / a, q / b) for p, q in orders])
    order = np.argsort(kappa, kind="stable")[:count]
    modes = tuple(orders[i] for i in order)
    kappa = kappa[order]
    no_flow = kappa * speed_of_sound / (2.0 * np.pi)
    return DuctModeResult(
        modes=modes,
        cut_on=no_flow * beta,
        cut_on_no_flow=no_flow,
        axial_wavenumber=-mach * kappa / beta,
        mach=mach,
        section="rectangular",
        label=(
            f"Rectangular duct ({a * 1000:.0f} x {b * 1000:.0f} mm, M = {mach:.3f})"
        ),
    )


def plane_wave_limit(
    *,
    diameter: float | None = None,
    width: float | None = None,
    height: float | None = None,
    area: float | None = None,
    flow_velocity: float = 0.0,
    speed_of_sound: float = _C_AIR,
) -> float:
    r"""The first cut-on frequency of a duct: plane waves only below it.

    A convenience over :func:`circular_duct_cut_on` and
    :func:`rectangular_duct_cut_on` that takes whichever description of the
    cross section is at hand. Give either ``diameter``, or both ``width`` and
    ``height``, or ``area`` (treated as a circular duct of the equivalent
    diameter :math:`\sqrt{4S/\pi}`).

    :param diameter: Internal diameter of a circular duct, m.
    :param width: Cross-sectional dimension ``a`` of a rectangular duct, m.
    :param height: Cross-sectional dimension ``b`` of a rectangular duct, m.
    :param area: Cross-sectional area, m2, for a duct described by its area.
    :param flow_velocity: Mean axial flow speed ``U``, m/s.
    :param speed_of_sound: Speed of sound ``c``, m/s.
    :return: The lowest cut-on frequency, Hz.
    :raises ValueError: If the cross section is not described exactly once.
    """
    if width is not None and height is not None:
        return rectangular_duct_cut_on(
            width,
            height,
            flow_velocity=flow_velocity,
            speed_of_sound=speed_of_sound,
            count=1,
        ).plane_wave_limit
    if diameter is None and area is not None:
        diameter = float(np.sqrt(4.0 * require_positive(area, "area") / np.pi))
    if diameter is None:
        msg = "give 'diameter', or both 'width' and 'height', or 'area'."
        raise ValueError(msg)
    return circular_duct_cut_on(
        diameter,
        flow_velocity=flow_velocity,
        speed_of_sound=speed_of_sound,
        count=1,
    ).plane_wave_limit


def warn_above_plane_wave_limit(
    frequencies: ArrayLike, limit: float, context: str, *, stacklevel: int = 3
) -> NDArray[np.bool_]:
    """Warn when analysis frequencies exceed a duct's first cut-on frequency.

    :param frequencies: The analysis frequencies, Hz.
    :param limit: The first cut-on frequency of the duct, Hz (from
        :func:`plane_wave_limit`).
    :param context: What the warning is about, e.g. ``"expansion chamber"``.
    :param stacklevel: Passed to :func:`warnings.warn` so the warning points at
        the caller of the public function rather than at this helper.
    :return: A boolean mask of the frequencies that are above the limit.
    """
    f = np.atleast_1d(np.asarray(frequencies, dtype=np.float64))
    above = np.asarray(f > limit, dtype=np.bool_)
    if bool(np.any(above)):
        warnings.warn(
            f"{context}: {int(np.count_nonzero(above))} of {f.size} frequencies "
            f"are above the first duct cut-on frequency ({limit:.0f} Hz), where "
            "higher-order modes propagate and the plane-wave result describes "
            "the plane-wave mode only.",
            PlaneWaveWarning,
            stacklevel=stacklevel,
        )
    return above
