#  Copyright (c) 2026. Jose Manuel Requena Plens
"""
Bending-wave transmission coefficients for rigid plate junctions
(Hopkins 2007, *Sound Insulation*, Section 5.2.1.3; Cremer et al. 1973;
Craik 1981, 1996).

The **wave approach** models a plane bending wave that is incident on a rigid
junction of thin plates at an angle ``theta`` and, assuming the junction beam is
simply supported (pinned so it can rotate but not translate), produces only
reflected and transmitted **bending** waves (no in-plane conversion). The
resulting angle-resolved transmission coefficients are *frequency independent*,
which is what makes them convenient closed-form building blocks for
statistical-energy-analysis (SEA) and the EN 12354 flanking model. This module
implements the rigid X, T, L and in-line junctions of two thin, homogeneous,
isotropic plates.

**Wave parameters (Hopkins Eqs 5.10 and 5.11, after Cremer et al. 1973).** With
plate ``i`` of thickness ``h_i``, quasi-longitudinal wave speed ``cL_i``,
surface density ``rho_s,i`` (kg/m^2), bending stiffness per unit width ``B_i``
and critical frequency ``fc_i``::

    chi = kB2 / kB1 = (rho_s2 B1 / (rho_s1 B2))**0.25
        = sqrt(h1 cL1 / (h2 cL2)) = sqrt(fc2 / fc1)                    (5.10)

    psi = B2 kB2**2 / (B1 kB1**2)
        = (h2 cL2 rho_s2) / (h1 cL1 rho_s1) = (rho_s2 fc1) / (rho_s1 fc2)  (5.11)

``chi`` is the ratio of bending wavenumbers (it fixes the total-internal-
reflection cut-off ``theta_co = arcsin(chi)``) and ``psi`` is the ratio of the
plates' bending-moment mobilities.

**Transmission around a corner (Hopkins Eq. 5.12, Craik 1981/1996).** For an
incident wave on plate 1, if ``chi >= sin(theta)``::

                     0.5 J1 J2 psi cos(theta) sqrt(chi**2 - sin**2(theta))
    tau12(theta) = --------------------------------------------------------
                    (J2 psi)**2 + chi**2 + J2 psi ( sqrt((1 + sin**2 theta)
                    (chi**2 + sin**2 theta)) + sqrt((1 - sin**2 theta)
                    (chi**2 - sin**2 theta)) )

and ``tau12(theta) = 0`` for ``chi < sin(theta)`` (no propagating transmitted
wave beyond the cut-off angle).

**Transmission across a straight section (Hopkins Eq. 5.13, Craik 1981/1996).**
Only the X-junction and T-junction (1) have an in-line (straight-through)
section. If ``chi >= sin(theta)``::

                          0.5 chi**2 cos**2(theta)
    tau13(theta) = -----------------------------------------  (same denominator
                    (J3 psi)**2 + chi**2 + J3 psi ( ... )       shape as 5.12)

and for ``chi < sin(theta)``::

                                    cos**2(theta)
    tau13(theta) = ------------------------------------------------------
                    2 + (J3 psi)**2 C**2 / chi**4
                        + (2 J3 psi C / chi**2) sqrt(1 + sin**2 theta)

with ``C = sqrt(chi**2 + sin**2 theta) + sqrt(sin**2 theta - chi**2)``.

**Junction constants.** ``J1``, ``J2`` set the corner coefficient and ``J3`` the
straight one:

===============  ====  =====  =====
Junction         J1    J2     J3
===============  ====  =====  =====
X                1     1      1
T-junction (1)   2     0.5    0.5
T-junction (2)   2     2      --
L                4     1      --
===============  ====  =====  =====

For T-junction (1) plates 1 and 3 are identical; for T-junction (2) plates 2
and 4 are identical. The straight section is undefined for T-junction (2) and
for the L-junction.

**In-line junction (Hopkins Eq. 5.14, Cremer et al. 1973).** Two collinear
plates (a change of section). Only normal incidence is used; it is within 1 dB
of the angular average when ``chi >= 1``::

                             2 (1 + chi)(1 + psi) sqrt(chi psi)     2
    tau12 ~= tau12(0 deg) = [ ------------------------------------ ]   (5.14)
                              chi (1 + psi)**2 + 2 psi (1 + chi**2)

**Angular average (Hopkins Eq. 5.6).** In a diffuse vibration field every angle
of incidence is equally probable and the incident intensity carries a
``cos(theta)`` obliquity factor, so the average transmission coefficient is::

    tau_bar_ij = integral_0^(pi/2) tau_ij(theta) cos(theta) d(theta)   (5.6)

(the ``cos(theta)`` weight already normalises the average, since
``integral_0^(pi/2) cos(theta) d(theta) = 1``).

**Coupling loss factor (Hopkins Eq. 2.154).** For a source plate ``i`` of area
``S_i``, bending-wave group velocity ``cg_i`` and junction length ``L_ij``::

    eta_ij = cg_i L_ij tau_ij / (2 pi**2 f S_i)                        (2.154)

**Vibration reduction index (Hopkins Eq. 5.116).** The wave-approach value of
the EN 12354 junction descriptor, with ``fc_j`` the critical frequency of the
**receiving** plate and the reference frequency ``f_ref = 1000 Hz``::

    K_ij = 10 lg(1 / tau_ij) + 5 lg(fc_j / f_ref)                      (5.116)

Combined with the reciprocity relationship below (``tau_bar_12 = chi
tau_bar_21`` with ``chi = sqrt(fc_2 / fc_1)``) this form is symmetric,
``K_ij = K_ji``, as EN 12354 and ISO 10848 require of the junction descriptor.

**Reciprocity (Hopkins Eq. 5.7, the SEA consistency relationship).** The angular
averages of the two directions are linked by
``tau_bar_ij = tau_bar_ji sqrt(h_i cL_i / (h_j cL_j)) = tau_bar_ji sqrt(fc_j /
fc_i)``, i.e. ``tau_bar_12 = chi tau_bar_21``.

**Two shortcuts for right-angle joints (Norton & Karczub 2003, Section
6.6.1).** Alongside the angle-resolved wave approach above, the SEA literature
uses a pair of closed forms that need no integration, which is what the
experimental SEA of :mod:`phonometry.vibration.experimental_sea` is normally
compared against:

* :func:`right_angle_transmission_coefficient` (Norton Eqs. 6.53 to 6.55, after
  Bies & Hamid and Cremer et al.). The normal-incidence coefficient of two
  plates at right angles is ``tau12(0) = 2 (psi_N**0.5 + psi_N**-0.5)**-2``
  with ``psi_N = rho_1 cL1**1.5 h1**2.5 / (rho_2 cL2**1.5 h2**2.5)`` (note
  this is a *different* ``psi`` from Hopkins Eq. 5.11 above), and the random
  incidence value follows from the empirical factor ``2,754 X / (1 + 3,24 X)``
  with ``X = h1/h2``. Feeding it to :func:`coupling_loss_factor` reproduces
  Norton Eq. (6.52), ``eta_12 = 2 cB L tau12 / (pi omega S1)``, identically:
  that equation and Hopkins Eq. (2.154) are the same expression once
  ``cg = 2 cB``.
* :func:`point_connection_coupling_loss_factor` (Norton Eq. 6.56, after
  Clarkson & Ranky). Plates joined at ``N`` discrete points (bolts, rivets,
  spot welds) rather than along a line. Use it when the bending wavelength is
  shorter than the joint length, and the line-junction route above when it is
  longer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import quad

from .._internal.validation import require_choice, require_positive

if TYPE_CHECKING:
    from matplotlib.axes import Axes

#: Reference frequency ``f_ref`` of Hopkins Eq. 5.116, in Hz.
_REFERENCE_FREQUENCY: float = 1000.0

#: Speed of sound in air ``c0`` used for the plate critical frequencies, m/s.
_SPEED_OF_SOUND: float = 343.0

#: Junction constants ``(J1, J2, J3)`` for the perpendicular junctions
#: (Hopkins Eq. 5.12/5.13). ``J3 is None`` where no straight section exists.
_JUNCTIONS: dict[str, tuple[float, float, float | None]] = {
    "X": (1.0, 1.0, 1.0),
    "T1": (2.0, 0.5, 0.5),
    "T2": (2.0, 2.0, None),
    "L": (4.0, 1.0, None),
}

__all__ = [
    "JunctionTransmissionResult",
    "angular_average_transmission_coefficient",
    "corner_transmission_coefficient",
    "coupling_loss_factor",
    "inline_transmission_coefficient",
    "junction_transmission",
    "junction_wave_parameters",
    "point_connection_coupling_loss_factor",
    "right_angle_transmission_coefficient",
    "straight_transmission_coefficient",
    "wave_vibration_reduction_index",
]


def junction_wave_parameters(
    thickness1: float,
    wave_speed1: float,
    surface_density1: float,
    thickness2: float,
    wave_speed2: float,
    surface_density2: float,
) -> tuple[float, float]:
    """Wave parameters ``chi`` and ``psi`` of a plate pair (Hopkins 5.10/5.11).

    ``chi = sqrt(h1 cL1 / (h2 cL2))`` (Eq. 5.10) and
    ``psi = (h2 cL2 rho_s2) / (h1 cL1 rho_s1)`` (Eq. 5.11), with plate 1 the
    plate carrying the incident wave.

    :param thickness1: Thickness ``h1`` of plate 1, in m (> 0).
    :param wave_speed1: Quasi-longitudinal wave speed ``cL1`` of plate 1, in m/s
        (> 0).
    :param surface_density1: Surface density ``rho_s1`` of plate 1, in kg/m^2
        (> 0).
    :param thickness2: Thickness ``h2`` of plate 2, in m (> 0).
    :param wave_speed2: Quasi-longitudinal wave speed ``cL2`` of plate 2, in m/s
        (> 0).
    :param surface_density2: Surface density ``rho_s2`` of plate 2, in kg/m^2
        (> 0).
    :return: The pair ``(chi, psi)``.
    :raises ValueError: for a non-positive input.
    """
    h1 = require_positive(thickness1, "thickness1")
    c1 = require_positive(wave_speed1, "wave_speed1")
    r1 = require_positive(surface_density1, "surface_density1")
    h2 = require_positive(thickness2, "thickness2")
    c2 = require_positive(wave_speed2, "wave_speed2")
    r2 = require_positive(surface_density2, "surface_density2")
    hc1 = h1 * c1
    hc2 = h2 * c2
    chi = math.sqrt(hc1 / hc2)
    psi = (hc2 * r2) / (hc1 * r1)
    return chi, psi


def _critical_frequency(thickness: float, wave_speed: float) -> float:
    """Thin-plate critical frequency ``fc = sqrt(12) c0**2 / (2 pi h cL)``.

    Hopkins Eq. 2.201 (``fc = (c0**2 / 2 pi) sqrt(m'' / B')``) written for a
    homogeneous isotropic plate, where ``B' = m'' cL**2 h**2 / 12``, with the
    speed of sound in air fixed at ``c0 = 343 m/s``.
    """
    return math.sqrt(12.0) * _SPEED_OF_SOUND**2 / (
        2.0 * math.pi * thickness * wave_speed
    )


def _junction(junction: str) -> tuple[float, float, float | None]:
    """Validated ``(J1, J2, J3)`` for a perpendicular junction name."""
    name = require_choice(str(junction).upper(), "junction", tuple(_JUNCTIONS))
    return _JUNCTIONS[name]


def _obliquity_sum(chi: float, sin2: NDArray[np.float64]) -> NDArray[np.float64]:
    """Shared denominator obliquity term of Eqs 5.12/5.13.

    ``sqrt((1 + s2)(chi**2 + s2)) + sqrt((1 - s2)(chi**2 - s2))`` where the
    second radicand is clipped to zero beyond the cut-off (it is never used
    there because the coefficient itself is zero for the corner and switches
    branch for the straight section).
    """
    chi2 = chi * chi
    first = np.sqrt((1.0 + sin2) * (chi2 + sin2))
    second = np.sqrt(np.clip((1.0 - sin2) * (chi2 - sin2), 0.0, None))
    return np.asarray(first + second, dtype=np.float64)


def _check_angle(angle: ArrayLike) -> NDArray[np.float64]:
    """Incidence angle(s) in radians, restricted to ``[0, pi/2]``."""
    theta = np.asarray(angle, dtype=np.float64)
    if not np.all(np.isfinite(theta)):
        raise ValueError("'angle' must be finite.")
    if np.any(theta < 0.0) or np.any(theta > math.pi / 2.0 + 1e-9):
        raise ValueError("'angle' must lie in [0, pi/2] radians.")
    return theta


def corner_transmission_coefficient(
    angle: ArrayLike, chi: float, psi: float, junction: str = "X"
) -> NDArray[np.float64]:
    """Transmission around a corner ``tau12(theta)`` (Hopkins Eq. 5.12).

    Returns ``0`` for angles beyond the cut-off ``arcsin(chi)`` (only reached
    when ``chi < 1``).

    :param angle: Incidence angle ``theta``, in **radians** (scalar or array,
        ``0 <= theta <= pi/2``).
    :param chi: Wave parameter ``chi`` (Eq. 5.10, > 0).
    :param psi: Wave parameter ``psi`` (Eq. 5.11, > 0).
    :param junction: ``"X"``, ``"T1"``, ``"T2"`` or ``"L"``.
    :return: ``tau12(theta)`` (same shape as *angle*).
    :raises ValueError: for a non-positive ``chi``/``psi``, an out-of-range
        angle or an unknown junction.
    """
    theta = _check_angle(angle)
    c = require_positive(chi, "chi")
    p = require_positive(psi, "psi")
    j1, j2, _ = _junction(junction)
    sin = np.sin(theta)
    sin2 = sin * sin
    disc = c * c - sin2
    num = 0.5 * j1 * j2 * p * np.cos(theta) * np.sqrt(np.clip(disc, 0.0, None))
    den = (j2 * p) ** 2 + c * c + j2 * p * _obliquity_sum(c, sin2)
    tau = np.where(disc >= 0.0, num / den, 0.0)
    return np.asarray(tau, dtype=np.float64)


def straight_transmission_coefficient(
    angle: ArrayLike, chi: float, psi: float, junction: str = "X"
) -> NDArray[np.float64]:
    """Transmission across a straight section ``tau13(theta)`` (Hopkins 5.13).

    Defined only for the X-junction and T-junction (1); both incidence regimes
    ``chi >= sin(theta)`` and ``chi < sin(theta)`` are covered.

    :param angle: Incidence angle ``theta``, in **radians** (scalar or array,
        ``0 <= theta <= pi/2``).
    :param chi: Wave parameter ``chi`` (Eq. 5.10, > 0).
    :param psi: Wave parameter ``psi`` (Eq. 5.11, > 0).
    :param junction: ``"X"`` or ``"T1"`` (the only junctions with a straight
        section).
    :return: ``tau13(theta)`` (same shape as *angle*).
    :raises ValueError: for a non-positive ``chi``/``psi``, an out-of-range
        angle, or a junction without a straight section.
    """
    theta = _check_angle(angle)
    c = require_positive(chi, "chi")
    p = require_positive(psi, "psi")
    _, _, j3 = _junction(junction)
    if j3 is None:
        raise ValueError(
            f"junction {junction!r} has no straight section (use 'X' or 'T1')."
        )
    sin = np.sin(theta)
    sin2 = sin * sin
    cos2 = np.cos(theta) ** 2
    # chi >= sin(theta): same denominator shape as the corner coefficient.
    den_ge = (j3 * p) ** 2 + c * c + j3 * p * _obliquity_sum(c, sin2)
    tau_ge = 0.5 * c * c * cos2 / den_ge
    # chi < sin(theta): Eq. 5.13 lower branch.
    big_c = np.sqrt(c * c + sin2) + np.sqrt(np.clip(sin2 - c * c, 0.0, None))
    den_lt = (
        2.0
        + (j3 * p) ** 2 * big_c**2 / c**4
        + (2.0 * j3 * p * big_c / (c * c)) * np.sqrt(1.0 + sin2)
    )
    tau_lt = cos2 / den_lt
    tau = np.where(c >= sin, tau_ge, tau_lt)
    return np.asarray(tau, dtype=np.float64)


def inline_transmission_coefficient(chi: float, psi: float) -> float:
    """Normal-incidence transmission across an in-line junction (Hopkins 5.14).

    ``tau12 = [2 (1 + chi)(1 + psi) sqrt(chi psi) / (chi (1 + psi)**2 +
    2 psi (1 + chi**2))]**2`` (Cremer et al. 1973). For identical plates
    (``chi = psi = 1``) this is 1 (a continuous plate transmits fully).

    :param chi: Wave parameter ``chi`` (Eq. 5.10, > 0).
    :param psi: Wave parameter ``psi`` (Eq. 5.11, > 0).
    :return: ``tau12(0 deg)``.
    :raises ValueError: for a non-positive ``chi``/``psi``.
    """
    c = require_positive(chi, "chi")
    p = require_positive(psi, "psi")
    ratio = 2.0 * (1.0 + c) * (1.0 + p) * math.sqrt(c * p) / (
        c * (1.0 + p) ** 2 + 2.0 * p * (1.0 + c * c)
    )
    return float(ratio * ratio)


def angular_average_transmission_coefficient(
    chi: float, psi: float, junction: str = "X", *, section: str = "corner"
) -> float:
    """Diffuse-field angular average of a transmission coefficient (Hopkins 5.6).

    ``tau_bar = integral_0^(pi/2) tau(theta) cos(theta) d(theta)``, evaluated by
    adaptive quadrature.

    :param chi: Wave parameter ``chi`` (Eq. 5.10, > 0).
    :param psi: Wave parameter ``psi`` (Eq. 5.11, > 0).
    :param junction: ``"X"``, ``"T1"``, ``"T2"`` or ``"L"``.
    :param section: ``"corner"`` (``tau12``, default) or ``"straight"``
        (``tau13``; only for ``"X"``/``"T1"``).
    :return: The angular-average transmission coefficient ``tau_bar``.
    :raises ValueError: for a non-positive ``chi``/``psi``, an unknown junction
        or section, or a straight section that does not exist.
    """
    section = require_choice(str(section).lower(), "section", ("corner", "straight"))
    if section == "corner":
        def coeff(t: float) -> float:
            return float(corner_transmission_coefficient(t, chi, psi, junction))
    else:
        def coeff(t: float) -> float:
            return float(straight_transmission_coefficient(t, chi, psi, junction))

    # For chi < 1 the coefficient cuts off at theta = arcsin(chi), where the
    # integrand has a derivative kink; hand that break point to quad so it
    # splits the interval there instead of fighting the slope discontinuity.
    break_points = [math.asin(chi)] if chi < 1.0 else None
    value, _ = quad(
        lambda t: coeff(t) * math.cos(t), 0.0, math.pi / 2.0, points=break_points
    )
    return float(value)


def coupling_loss_factor(
    transmission_coefficient: ArrayLike,
    group_velocity: float,
    junction_length: float,
    frequency: ArrayLike,
    plate_area: float,
) -> NDArray[np.float64]:
    """Coupling loss factor from a transmission coefficient (Hopkins Eq. 2.154).

    ``eta_ij = cg_i L_ij tau_ij / (2 pi**2 f S_i)`` with the source-plate
    bending-wave group velocity ``cg_i``, the junction length ``L_ij``, the
    frequency ``f`` and the source-plate area ``S_i``.

    :param transmission_coefficient: Angular-average ``tau_ij`` (scalar/array).
    :param group_velocity: Source-plate bending-wave group velocity ``cg_i``,
        in m/s (> 0). For a thin plate ``cg = 2 cB`` with the bending phase
        speed ``cB`` (see
        :func:`phonometry.vibration.point_mobility.plate_bending_wave_speed`).
    :param junction_length: Junction length ``L_ij``, in m (> 0).
    :param frequency: Frequency ``f``, in hertz (scalar or array, > 0).
    :param plate_area: Source-plate area ``S_i``, in m^2 (> 0).
    :return: The coupling loss factor ``eta_ij`` (broadcast of the inputs).
    :raises ValueError: for a non-positive input.
    """
    tau = np.asarray(transmission_coefficient, dtype=np.float64)
    if np.any(tau < 0.0):
        raise ValueError("'transmission_coefficient' must be non-negative.")
    cg = require_positive(group_velocity, "group_velocity")
    lij = require_positive(junction_length, "junction_length")
    freq = np.asarray(frequency, dtype=np.float64)
    if np.any(freq <= 0.0):
        raise ValueError("'frequency' must be positive.")
    area = require_positive(plate_area, "plate_area")
    eta = cg * lij * tau / (2.0 * math.pi**2 * freq * area)
    return np.asarray(eta, dtype=np.float64)


def right_angle_transmission_coefficient(
    thickness1: float,
    thickness2: float,
    *,
    density1: float,
    density2: float,
    wave_speed1: float,
    wave_speed2: float,
    incidence: str = "random",
) -> float:
    """Right-angle plate junction ``tau12`` (Norton Eqs. 6.53 to 6.55).

    The closed form used throughout the SEA literature for two flat plates
    coupled at right angles along a line, with no angular integration::

        psi_N = rho1 cL1**1.5 h1**2.5 / (rho2 cL2**1.5 h2**2.5)          (6.54)
        tau12(0) = 2 (sqrt(psi_N) + 1/sqrt(psi_N))**-2                   (6.53)
        tau12    = tau12(0) 2,754 X / (1 + 3,24 X),  X = h1/h2            (6.55)

    ``tau12(0)`` is symmetric in the two plates (swapping them inverts
    ``psi_N``, which the expression does not see); the random-incidence factor
    is not, so ``tau12`` and ``tau21`` differ. Pass the result to
    :func:`coupling_loss_factor` with the source plate's group velocity
    ``cg = 2 cB`` to obtain Norton Eq. (6.52).

    ``rho`` here is the **volume** density in kg/m^3, not the surface density
    of :func:`junction_wave_parameters`.

    :param thickness1: Thickness ``h1`` of the source plate, in m (> 0).
    :param thickness2: Thickness ``h2`` of the receiving plate, in m (> 0).
    :param density1: Density ``rho1`` of the source plate, in kg/m^3 (> 0).
    :param density2: Density ``rho2`` of the receiving plate, in kg/m^3 (> 0).
    :param wave_speed1: Quasi-longitudinal wave speed ``cL1``, in m/s (> 0).
    :param wave_speed2: Quasi-longitudinal wave speed ``cL2``, in m/s (> 0).
    :param incidence: ``"random"`` (Default, Eq. 6.55) or ``"normal"``
        (Eq. 6.53 alone).
    :return: The transmission coefficient ``tau12``.
    :raises ValueError: for a non-positive input or an unknown incidence.
    """
    h_1 = require_positive(thickness1, "thickness1")
    h_2 = require_positive(thickness2, "thickness2")
    rho_1 = require_positive(density1, "density1")
    rho_2 = require_positive(density2, "density2")
    c_1 = require_positive(wave_speed1, "wave_speed1")
    c_2 = require_positive(wave_speed2, "wave_speed2")
    mode = require_choice(incidence, "incidence", ("random", "normal"))

    psi_n = (rho_1 * c_1**1.5 * h_1**2.5) / (rho_2 * c_2**1.5 * h_2**2.5)
    root = math.sqrt(psi_n)
    tau_0 = 2.0 / (root + 1.0 / root) ** 2
    if mode == "normal":
        return float(tau_0)
    x = h_1 / h_2
    return float(tau_0 * 2.754 * x / (1.0 + 3.24 * x))


def point_connection_coupling_loss_factor(
    frequency: ArrayLike,
    n_connections: int,
    *,
    thickness1: float,
    thickness2: float,
    surface_density1: float,
    surface_density2: float,
    wave_speed1: float,
    wave_speed2: float,
    plate_area1: float,
) -> NDArray[np.float64]:
    """Coupling loss factor of a point-connected plate pair (Norton Eq. 6.56).

    Two homogeneous plates joined by ``N`` bolts, rivets or spot welds rather
    than along a continuous line::

                 4 N h1 cL1        A1 A2
        eta_12 = -------------- ------------ ,   Ai = rho_si**2 hi**2 cLi**2
                 sqrt(3) w S1   (A1 + A2)**2

    with ``rho_si`` the surface density in kg/m^2, ``w = 2 pi f`` and ``S1``
    the source-plate area. Norton recommends it when the bending wavelength in
    the plates is shorter than the length of the connected edge, and the line
    junction (:func:`coupling_loss_factor` on
    :func:`right_angle_transmission_coefficient`) when it is longer. The
    result falls as ``1/f``, unlike the line junction's ``1/sqrt(f)``.

    .. note::

       Equation (6.56) is printed in the book with the ``(A1 + A2)`` bracket
       *not* squared, which is dimensionally inconsistent (the coupling loss
       factor would not be dimensionless). The squared form implemented here
       is the one that reproduces the book's own answer to problem 6.13; see
       ``docs/ERRATA.md``.

    :param frequency: Band centre frequency ``f``, in hertz (scalar/array, >0).
    :param n_connections: Number of point connections ``N`` (integer >= 1).
    :param thickness1: Thickness ``h1`` of the source plate, in m (> 0).
    :param thickness2: Thickness ``h2`` of the receiving plate, in m (> 0).
    :param surface_density1: ``rho_s1``, in kg/m^2 (> 0).
    :param surface_density2: ``rho_s2``, in kg/m^2 (> 0).
    :param wave_speed1: Quasi-longitudinal wave speed ``cL1``, in m/s (> 0).
    :param wave_speed2: Quasi-longitudinal wave speed ``cL2``, in m/s (> 0).
    :param plate_area1: Source-plate area ``S1``, in m^2 (> 0).
    :return: The coupling loss factor ``eta_12`` per band.
    :raises ValueError: for a non-positive or non-integer input.
    """
    freq = np.asarray(frequency, dtype=np.float64)
    if np.any(freq <= 0.0):
        raise ValueError("'frequency' must be positive.")
    n = int(n_connections)
    if n < 1:
        raise ValueError("'n_connections' must be a positive integer.")
    h_1 = require_positive(thickness1, "thickness1")
    h_2 = require_positive(thickness2, "thickness2")
    rs_1 = require_positive(surface_density1, "surface_density1")
    rs_2 = require_positive(surface_density2, "surface_density2")
    c_1 = require_positive(wave_speed1, "wave_speed1")
    c_2 = require_positive(wave_speed2, "wave_speed2")
    area = require_positive(plate_area1, "plate_area1")

    a_1 = (rs_1 * h_1 * c_1) ** 2
    a_2 = (rs_2 * h_2 * c_2) ** 2
    omega = 2.0 * math.pi * freq
    eta = (
        4.0 * n * h_1 * c_1 / (math.sqrt(3.0) * omega * area)
        * a_1 * a_2 / (a_1 + a_2) ** 2
    )
    return np.asarray(eta, dtype=np.float64)


def wave_vibration_reduction_index(
    transmission_coefficient: ArrayLike,
    critical_frequency_receiver: float,
) -> NDArray[np.float64]:
    """Vibration reduction index from a transmission coefficient (Hopkins 5.116).

    ``K_ij = 10 lg(1 / tau_ij) + 5 lg(fc_j / f_ref)`` with ``fc_j`` the
    critical frequency of the **receiving** plate and the reference frequency
    ``f_ref = 1000 Hz``. Because the angular-average transmission coefficients
    satisfy the reciprocity relationship ``tau_bar_ij = tau_bar_ji
    sqrt(fc_j / fc_i)`` (Eq. 5.7), this form is symmetric: ``K_ij = K_ji``.

    :param transmission_coefficient: ``tau_ij`` (scalar or array, > 0).
    :param critical_frequency_receiver: Critical frequency ``fc_j`` of the
        receiving plate, in hertz (> 0).
    :return: The vibration reduction index ``K_ij``, in dB.
    :raises ValueError: for a non-positive ``tau`` or ``fc_j``.
    """
    tau = np.asarray(transmission_coefficient, dtype=np.float64)
    if np.any(tau <= 0.0):
        raise ValueError("'transmission_coefficient' must be positive.")
    fcj = require_positive(critical_frequency_receiver, "critical_frequency_receiver")
    kij = 10.0 * np.log10(1.0 / tau) + 5.0 * math.log10(fcj / _REFERENCE_FREQUENCY)
    return np.asarray(kij, dtype=np.float64)


@dataclass(frozen=True)
class JunctionTransmissionResult:
    """Bending-wave transmission across a rigid plate junction (Hopkins 5.2.1.3).

    :ivar junction: Junction type (``"X"``, ``"T1"``, ``"T2"`` or ``"L"``).
    :ivar chi: Wave parameter ``chi`` (Eq. 5.10).
    :ivar psi: Wave parameter ``psi`` (Eq. 5.11).
    :ivar critical_frequency1: Critical frequency ``fc_1`` of the source
        plate, in hertz (thin plate, ``c0 = 343 m/s``).
    :ivar critical_frequency2: Critical frequency ``fc_2`` of the receiving
        plate, in hertz (thin plate, ``c0 = 343 m/s``).
    :ivar angles_deg: Incidence-angle grid, in degrees.
    :ivar corner: Corner transmission coefficient ``tau12(theta)`` on the grid.
    :ivar straight: Straight-section coefficient ``tau13(theta)`` on the grid,
        or ``None`` when the junction has no straight section.
    :ivar corner_average: Diffuse-field angular average ``tau_bar_12`` (Eq. 5.6).
    :ivar straight_average: Angular average ``tau_bar_13``, or ``None``.
    :ivar thickness1: Plate 1 thickness, in metres, retained (with
        ``thickness2``) so :meth:`plot_geometry` can draw the junction;
        appended after the original fields and ``None`` for hand-built
        results.
    :ivar thickness2: Plate 2 thickness, in metres, or ``None``.
    """

    junction: str
    chi: float
    psi: float
    critical_frequency1: float
    critical_frequency2: float
    angles_deg: np.ndarray
    corner: np.ndarray
    straight: np.ndarray | None
    corner_average: float
    straight_average: float | None
    thickness1: float | None = None
    thickness2: float | None = None

    @property
    def corner_reduction_index(self) -> float:
        """Wave-approach ``K_12`` of the corner path, in dB (Hopkins Eq. 5.116).

        ``K_12 = 10 lg(1 / tau_bar_12) + 5 lg(fc_2 / 1000)`` with the receiving
        plate's critical frequency ``fc_2``. The value is symmetric: building
        the reverse junction (plates swapped, and for a T-junction the matching
        constants ``T1`` <-> ``T2``) gives the same ``K_21 = K_12``.
        """
        kij = wave_vibration_reduction_index(
            self.corner_average, self.critical_frequency2
        )
        return float(kij)

    def plot(self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any) -> Axes:
        """Plot ``tau(theta)`` versus incidence angle for this junction.

        Requires matplotlib (``pip install phonometry[plot]``); returns the
        :class:`~matplotlib.axes.Axes`.
        """
        from .._i18n import check_language
        from .._plot.junction import plot_junction_transmission

        return plot_junction_transmission(
            self, ax=ax, language=check_language(language), **kwargs
        )

    def plot_geometry(
        self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any
    ) -> Axes:
        """Draw the plate-junction cross-section to scale.

        Requires matplotlib (``pip install phonometry[plot]``); returns the
        :class:`~matplotlib.axes.Axes`.

        :raises ValueError: If the result does not retain its geometry.
        """
        from .._i18n import check_language
        from .._plot.geometry import plot_junction_result_geometry

        check_language(language)
        return plot_junction_result_geometry(self, ax=ax, language=language, **kwargs)


def junction_transmission(
    junction: str,
    thickness1: float,
    wave_speed1: float,
    surface_density1: float,
    thickness2: float,
    wave_speed2: float,
    surface_density2: float,
    *,
    angles_deg: ArrayLike | None = None,
) -> JunctionTransmissionResult:
    """Bending-wave transmission of a rigid perpendicular plate junction.

    Builds the angle-resolved corner (and, for X / T-junction (1), straight)
    transmission coefficients of Hopkins Eqs 5.12/5.13 and their diffuse-field
    angular averages (Eq. 5.6) from the two plates' properties, together with
    the thin-plate critical frequencies ``fc = sqrt(12) c0**2 / (2 pi h cL)``
    (``c0 = 343 m/s``) used by the Eq. 5.116 vibration reduction index. For the
    in-line junction (normal incidence only) use
    :func:`inline_transmission_coefficient`.

    :param junction: ``"X"``, ``"T1"``, ``"T2"`` or ``"L"``.
    :param thickness1: Thickness ``h1`` of the source plate, in m (> 0).
    :param wave_speed1: Quasi-longitudinal wave speed ``cL1`` of the source
        plate, in m/s (> 0).
    :param surface_density1: Surface density ``rho_s1`` of the source plate, in
        kg/m^2 (> 0).
    :param thickness2: Thickness ``h2`` of the receiving plate, in m (> 0).
    :param wave_speed2: Quasi-longitudinal wave speed ``cL2`` of the receiving
        plate, in m/s (> 0).
    :param surface_density2: Surface density ``rho_s2`` of the receiving plate,
        in kg/m^2 (> 0).
    :param angles_deg: Incidence-angle grid in degrees (Default: 0 to 90 in 91
        one-degree steps).
    :return: A :class:`JunctionTransmissionResult`.
    :raises ValueError: for a non-positive input or an unknown junction.
    """
    name = require_choice(str(junction).upper(), "junction", tuple(_JUNCTIONS))
    chi, psi = junction_wave_parameters(
        thickness1, wave_speed1, surface_density1,
        thickness2, wave_speed2, surface_density2,
    )
    if angles_deg is None:
        grid = np.linspace(0.0, 90.0, 91)
    else:
        grid = np.atleast_1d(np.asarray(angles_deg, dtype=np.float64))
        if grid.ndim != 1 or grid.size == 0:
            raise ValueError("'angles_deg' must be a non-empty 1-D array.")
    theta = np.radians(grid)
    corner = corner_transmission_coefficient(theta, chi, psi, name)
    corner_avg = angular_average_transmission_coefficient(chi, psi, name, section="corner")

    _, _, j3 = _JUNCTIONS[name]
    straight: np.ndarray | None
    straight_avg: float | None
    if j3 is None:
        straight = None
        straight_avg = None
    else:
        straight = straight_transmission_coefficient(theta, chi, psi, name)
        straight_avg = angular_average_transmission_coefficient(
            chi, psi, name, section="straight"
        )

    return JunctionTransmissionResult(
        junction=name,
        chi=chi,
        psi=psi,
        critical_frequency1=_critical_frequency(thickness1, wave_speed1),
        critical_frequency2=_critical_frequency(thickness2, wave_speed2),
        angles_deg=grid,
        corner=corner,
        straight=straight,
        corner_average=corner_avg,
        straight_average=straight_avg,
        thickness1=float(thickness1),
        thickness2=float(thickness2),
    )
