#  Copyright (c) 2026. Jose Manuel Requena Plens
r"""
Wall ties in masonry cavity walls: the structural bridge across the cavity
(Hopkins 2007, Sections 3.11.3.2 and 4.3.5.4.1).

A masonry cavity wall is a double leaf, and the textbook double-leaf prediction
(:func:`phonometry.building.double_wall_transmission_loss`) treats the cavity
as pure air. Real cavity walls are stitched together by **wall ties** every few
courses. Those ties do two things the air-only model cannot see: they add a
mechanical spring in parallel with the air spring, which pushes the
mass-spring-mass resonance up, and they open a structure-borne path from one
leaf to the other, which caps the insulation the pair can reach.

**Dynamic stiffness of a tie (Section 3.11.3.2).** A tie is characterised by a
single number ``sX mm``, its dynamic stiffness at a cavity width ``X``,
measured on two nominally identical 100 mm concrete cubes from the
mass-spring-mass resonance of the pair,
:math:`s_{X\,\mathrm{mm}} = 2 \pi^2 f_\mathrm{msm}^2 m_\mathrm{av}` (Eq. 3.202).
:data:`WALL_TIE_STIFFNESS` carries Hopkins' Table A4, whose 50 mm
rows come from Hopkins, Wilson & Craik (1999) and whose 100 mm row comes from
Hall & Hopkins (2001).

**The tie array as a spring in parallel with the air (Eq. 4.89).** ``N`` ties
over a plate of area ``S`` add :math:`N k / S` to the cavity air stiffness
``s_a``:

.. math::

   f_\mathrm{msm} = \frac{1}{2\pi} \sqrt{ \frac{s_\mathrm{a} + N k / S}
   {\rho_{\mathrm{s}1} \rho_{\mathrm{s}2} / (\rho_{\mathrm{s}1} + \rho_{\mathrm{s}2})} } \tag{Eq. 4.89}

Below ``fmsm`` the two leaves move as one plate of the combined mass. Stiff ties
are therefore doubly bad: they raise the resonance into the rating range *and*
bridge the cavity. Use :func:`wall_tie_stiffness_per_area` and pass the result
as ``tie_stiffness_per_area`` to
:func:`phonometry.building.mass_spring_mass_resonance` or
:func:`phonometry.building.double_wall_transmission_loss`.

**The structure-borne path (Eqs. 4.84 to 4.88).** Each tie is a point
connection between two plates. With the driving-point mobilities ``Yi``,
``Yj`` of the two leaves (infinite thin plates,
:math:`Y = 1/(8 \sqrt{B' m''})`, Eq. 2.190) and the connector mobility of a
linear spring :math:`Y_\mathrm{c} = i \omega / k` (Eq. 4.88), ``N`` identical
uncorrelated connections give the coupling loss factor

.. math::

   \eta_{ij} = \frac{N}{\omega m_i}
   \frac{\operatorname{Re}\{Y_j\}}{| Y_i + Y_j + Y_\mathrm{c} |^2} \tag{Eq. 4.87}

The plate area cancels (:math:`N/m_i = n/\rho_{\mathrm{s}1}` with ``n`` ties per m2),
so :func:`wall_tie_coupling_loss_factor` needs only the tie density. A rigid
connection (screw, nail, bolt, or a tie so stiff it never yields) is the
limit :math:`Y_\mathrm{c} = 0`, where the only frequency dependence left is the
:math:`1/\omega` and :math:`\eta_{ij}` falls as :math:`1/f`. Once
:math:`|Y_\mathrm{c}| = \omega/k` overtakes the plate mobilities a resilient tie adds
two more powers, so :math:`\eta_{ij}` falls as :math:`1/f^3` and the *ratio*
to the rigid ceiling as :math:`1/f^2`. That is exactly why a butterfly tie at
1.7 MN/m and a vertical-twist tie at 94 MN/m behave so differently: the soft
one enters that regime inside the building acoustics range, the stiff one
stays on the rigid ceiling for another two octaves.

.. note::

   The *inputs* of this model are printed data: Table A4 here, confirmed
   value for value by Hopkins, Wilson & Craik (1999) Table 1, which prints the
   same 1.7 / 16.1 / 94.0 MN/m at a 50 mm cavity. Craik & Wilson (1995)
   Table 1 measures the same *tie types* at an 85 mm cavity and reports
   1.1 and 4.3 MN/m for the butterfly and double-triangle ties, so it
   corroborates the ordering and the order of magnitude but not the values;
   the dynamic stiffness is defined at a given cavity width and changes with
   it. The *output* is not printed anywhere: every published sound reduction
   index of a bridged masonry cavity wall is a figure, so the per-band
   transmission-loss penalty from wall ties has no numeric oracle. The
   resonance shift does: Hopkins Fig. 4.35 prints :math:`f_\mathrm{msm} = 26` Hz
   without ties and :math:`f_\mathrm{msm} = 50` Hz with them for the same wall.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike

from ..._internal.validation import require_finite_array, require_positive
from ...vibration.structural.point_mobility import infinite_plate_mobility

if TYPE_CHECKING:
    from matplotlib.axes import Axes

__all__ = [
    "WALL_TIE_STIFFNESS",
    "WallTieCouplingResult",
    "wall_tie_coupling_loss_factor",
    "wall_tie_stiffness",
    "wall_tie_stiffness_per_area",
]

#: Dynamic stiffness of wall ties, transcribed digit-for-digit from Hopkins
#: (2007) **Table A4** (printed p. 610), as ``(cavity width in m, dynamic
#: stiffness ``sX mm`` in N/m)``. The three 50 mm ties are those of
#: BS 1243:1978 measured by Hopkins, Wilson & Craik (1999) (their Table 1
#: prints the same 1,7 / 16,1 / 94,0 MN/m, with a standard deviation only for
#: the two butterfly rows, the other two being single samples); the 100 mm
#: proprietary vertical-twist tie is from Hall & Hopkins (2001). The stiffness
#: is a property of the tie *at that cavity width*: the same vertical-twist
#: design measures 94,0 MN/m at 50 mm and 43,4 MN/m at 100 mm, so pass the
#: matching ``gap`` to the double-wall functions.
WALL_TIE_STIFFNESS: dict[str, tuple[float, float]] = {
    "butterfly": (0.050, 1.7e6),
    "double_triangle": (0.050, 16.1e6),
    "vertical_twist": (0.050, 94.0e6),
    "vertical_twist_100mm": (0.100, 43.4e6),
}


def wall_tie_stiffness(tie: str) -> tuple[float, float]:
    """Look up a wall tie in Hopkins Table A4.

    :param tie: One of ``"butterfly"``, ``"double_triangle"``,
        ``"vertical_twist"`` (all at a 50 mm cavity) or
        ``"vertical_twist_100mm"``.
    :return: ``(cavity_width, stiffness)``: the cavity width in m at which the
        tie was measured, and its dynamic stiffness ``sX mm`` in N/m.
    :raises ValueError: for an unknown tie name.
    """
    try:
        return WALL_TIE_STIFFNESS[tie]
    except KeyError:
        raise ValueError(
            f"Unknown wall tie {tie!r}; choose one of "
            f"{sorted(WALL_TIE_STIFFNESS)}."
        ) from None


def wall_tie_stiffness_per_area(
    ties_per_area: float, tie: str | float
) -> float:
    r"""Stiffness per unit area of a tie array, :math:`N k / S` (Hopkins
    Eq. 4.89).

    The term that acts in parallel with the cavity air stiffness ``s_a`` in the
    mass-spring-mass resonance. Feed it to
    :func:`phonometry.building.mass_spring_mass_resonance` or
    :func:`phonometry.building.double_wall_transmission_loss` as
    ``tie_stiffness_per_area``.

    :param ties_per_area: Number of ties per unit area :math:`n = N/S`, in
        1/m^2 (> 0).
    :param tie: A name from :data:`WALL_TIE_STIFFNESS`, or an explicit dynamic
        stiffness ``k`` of one tie, in N/m (> 0).
    :return: The stiffness per unit area :math:`n k`, in N/m^3.
    :raises ValueError: for a non-positive input or an unknown tie name.
    """
    n = require_positive(ties_per_area, "ties_per_area")
    k = wall_tie_stiffness(tie)[1] if isinstance(tie, str) else require_positive(
        float(tie), "tie"
    )
    return float(n * k)


@dataclass(frozen=True)
class WallTieCouplingResult:
    r"""Structure-borne coupling of a wall-tie array (Hopkins Eqs. 4.87/4.88).

    :ivar frequencies: Frequencies, in hertz.
    :ivar coupling_loss_factor: Coupling loss factor ``eta_ij`` from leaf 1 to
        leaf 2 per frequency (dimensionless).
    :ivar mobility1: Driving-point mobility ``Yi`` of leaf 1, in m/(N.s).
    :ivar mobility2: Driving-point mobility ``Yj`` of leaf 2, in m/(N.s).
    :ivar connector_mobility: Magnitude
        :math:`\lvert Y_\mathrm{c} \rvert = \omega/k` of the tie mobility
        per frequency, in m/(N.s); all zeros for a rigid connection.
    :ivar ties_per_area: Number of ties per unit area ``n``, in 1/m^2.
    :ivar tie_stiffness: Dynamic stiffness ``k`` of one tie, in N/m, or ``None``
        for a rigid connection.
    :ivar rigid_coupling_loss_factor: The :math:`Y_\mathrm{c} = 0` coupling loss
        factor per frequency, the ceiling a resilient tie is measured
        against.
    """

    frequencies: np.ndarray
    coupling_loss_factor: np.ndarray
    mobility1: float
    mobility2: float
    connector_mobility: np.ndarray
    ties_per_area: float
    tie_stiffness: float | None
    rigid_coupling_loss_factor: np.ndarray

    def plot(self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any) -> Axes:
        """Plot ``eta_ij(f)`` against the rigid-connection ceiling.

        Requires matplotlib (``pip install phonometry[plot]``); returns the
        :class:`~matplotlib.axes.Axes`.
        """
        from ..._i18n import check_language
        from ..._plot.building import plot_wall_tie_coupling

        check_language(language)
        return plot_wall_tie_coupling(self, ax=ax, language=language, **kwargs)


def wall_tie_coupling_loss_factor(
    frequency: ArrayLike,
    mass1: float,
    mass2: float,
    bending_stiffness1: float,
    bending_stiffness2: float,
    *,
    ties_per_area: float,
    tie: str | float | None = None,
) -> WallTieCouplingResult:
    r"""Coupling loss factor of a wall-tie array (Hopkins Eqs. 4.87 and 4.88).

    .. math::

       \eta_{ij} = \frac{N}{\omega m_i}
       \frac{\operatorname{Re}\{Y_j\}}{| Y_i + Y_j + Y_\mathrm{c} |^2} \tag{Eq. 4.87}

    for ``N`` identical, uncorrelated point connections, with the leaves
    modelled as infinite thin plates (:math:`Y = 1/(8 \sqrt{B' m''})`,
    Eq. 2.190) and each tie as a linear spring (:math:`Y_\mathrm{c} = i \omega / k`,
    Eq. 4.88). Since :math:`m_i = \rho_{\mathrm{s}1} S` and :math:`N = n S`, the plate
    area cancels and only the tie density ``n`` enters.

    :param frequency: Frequencies ``f``, in hertz (array, > 0).
    :param mass1: Surface density ``rho_s1`` of the excited leaf, in kg/m^2 (> 0).
    :param mass2: Surface density ``rho_s2`` of the receiving leaf, in kg/m^2 (> 0).
    :param bending_stiffness1: Bending stiffness per unit width ``B'`` of leaf 1,
        in N.m (> 0); see
        :func:`phonometry.vibration.structural.point_mobility.plate_bending_stiffness`.
    :param bending_stiffness2: Bending stiffness per unit width of leaf 2, in N.m.
    :param ties_per_area: Number of ties per unit area ``n``, in 1/m^2 (> 0).
    :param tie: A name from :data:`WALL_TIE_STIFFNESS`, an explicit dynamic
        stiffness ``k`` in N/m, or ``None`` for a rigid connection
        (:math:`Y_\mathrm{c} = 0`, the screw/nail/bolt limit).
    :return: A :class:`WallTieCouplingResult`.
    :raises ValueError: for a non-positive input or an unknown tie name.
    """
    f = require_finite_array(frequency, "frequency")
    if np.any(f <= 0.0):
        raise ValueError("'frequency' must be positive and finite.")
    m1 = require_positive(mass1, "mass1")
    require_positive(mass2, "mass2")
    n = require_positive(ties_per_area, "ties_per_area")
    y1 = infinite_plate_mobility(bending_stiffness1, m1)
    y2 = infinite_plate_mobility(bending_stiffness2, mass2)
    omega = 2.0 * np.pi * f

    k: float | None = None
    if tie is not None:
        k = wall_tie_stiffness(tie)[1] if isinstance(tie, str) else require_positive(
            float(tie), "tie"
        )
    yc = np.zeros_like(f) if k is None else omega / k
    # Yi and Yj are real for a thin plate, Yc purely imaginary.
    denominator = (y1 + y2) ** 2 + yc**2
    eta = n / (omega * m1) * y2 / denominator
    rigid = n / (omega * m1) * y2 / (y1 + y2) ** 2
    return WallTieCouplingResult(
        frequencies=f,
        coupling_loss_factor=np.asarray(eta, dtype=np.float64),
        mobility1=y1,
        mobility2=y2,
        connector_mobility=np.asarray(yc, dtype=np.float64),
        ties_per_area=n,
        tie_stiffness=k,
        rigid_coupling_loss_factor=np.asarray(rigid, dtype=np.float64),
    )
