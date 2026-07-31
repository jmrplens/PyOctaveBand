#  Copyright (c) 2026. Jose Manuel Requena Plens
"""
Point mobilities and impedances of infinite structures (Cremer, Heckl &
Petersson 2005, Chapter 5, Table 5.1).

The **point mobility** ``Y`` of a structure is the complex ratio of the
velocity response at a driving point to the point force that produces it, and
its reciprocal is the **point impedance** ``Z = 1/Y`` (the same
motion-per-force / force-per-motion pair as ISO 7626-1 mechanical mobility, so
these theoretical values slot straight into :class:`~phonometry.MobilityResult`
and :func:`~phonometry.convert_frf`). For an *infinite* structure the driving
point never sees a reflected wave, so the mobility is the free-field value that
sets the vibrational power a source injects (Cremer 5.5): with a point force of
amplitude ``F`` the time-averaged injected power is (Cremer Eq. 5.23)::

    W = 0.5 * |F|**2 * Re{Y}                                    [W]

These are the theoretical companions of the *measured* driving-point mobilities
of ISO 7626 and the isolator transfer stiffnesses of ISO 10846, and they supply
the receiver mobility that the installed structure-borne prediction of
EN 12354-5 needs when no measurement is available.

**Compilation (Cremer Table 5.1).** With ``m'`` the mass per unit length
(kg/m), ``m''`` the mass per unit area (kg/m^2), ``B`` the bending stiffness of
a beam (N.m^2) and ``B'`` the bending stiffness of a plate *per unit width*
(N.m):

===============================  =========================  =============
Structure (point force)          Impedance ``Z``            Mobility ``Y``
===============================  =========================  =============
Longitudinal rod                 ``rho cL S``               ``1/(rho cL S)``
Slender beam, bending, centre    ``2 m' cB (1 + j)``        ``(1 - j)/(4 m' cB)``
Slender beam, bending, end       ``(m' cB / 2)(1 + j)``     ``(1 - j)/(m' cB)``
Thin plate, bending, centre      ``8 sqrt(B' m'')``         ``1/(8 sqrt(B' m''))``
Thin plate, bending, edge        ``3.5 sqrt(B' m'')``       ``1/(3.5 sqrt(B' m''))``
===============================  =========================  =============

The thin-plate driving-point impedance ``Z = 8 sqrt(B' m'')`` is real and
frequency independent (the plate behaves as a pure resistance to a point
force), so a plate absorbs power like a matched resistance. The beam impedance
grows as ``cB = (B omega**2 / m')**(1/4)`` (the bending wave speed), so its
mobility falls as ``omega**(-1/2)``; the ``(1 - j)`` factor means half the
input goes into a reactive near field. A moment excitation of the beam has the
mobility (Cremer Eq. 5.75) ``Y_M = omega (1 + j) / (4 B kB)`` with
``kB = omega / cB`` the bending wavenumber.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .._internal.validation import require_positive
from .mechanical_mobility import MobilityResult

__all__ = [
    "beam_bending_wave_speed",
    "infinite_beam_impedance",
    "infinite_beam_mobility",
    "infinite_beam_moment_mobility",
    "infinite_beam_point_mobility",
    "infinite_plate_impedance",
    "infinite_plate_mobility",
    "infinite_plate_point_mobility",
    "injected_power",
    "longitudinal_rod_impedance",
    "longitudinal_rod_mobility",
    "plate_bending_stiffness",
    "plate_bending_wave_speed",
]

#: Cremer Table 5.1 point-impedance constants for a thin plate in bending.
_PLATE_CONSTANT: dict[str, float] = {"centre": 8.0, "edge": 3.5}
#: Cremer Table 5.1 point-impedance constants for a slender beam in bending
#: (the ``(1 + j)`` prefactor is applied separately).
_BEAM_CONSTANT: dict[str, float] = {"centre": 2.0, "end": 0.5}


def plate_bending_stiffness(
    youngs_modulus: float, thickness: float, poisson_ratio: float = 0.0
) -> float:
    """Bending stiffness of a thin plate per unit width (Cremer Eq. 4.22).

    ``B' = E h**3 / (12 (1 - nu**2))``, the plate bending stiffness ``B'`` in
    N.m used throughout this module and by the coincidence frequency of
    :func:`phonometry.vibration.radiation_efficiency.coincidence_frequency`.

    :param youngs_modulus: Young's modulus ``E`` of the plate material, in Pa.
    :param thickness: Plate thickness ``h``, in m.
    :param poisson_ratio: Poisson's ratio ``nu`` (Default: 0.0).
    :return: The bending stiffness per unit width ``B'``, in N.m.
    :raises ValueError: for a non-positive modulus/thickness or ``|nu| >= 1``.
    """
    e = require_positive(youngs_modulus, "youngs_modulus")
    h = require_positive(thickness, "thickness")
    if not -1.0 < poisson_ratio < 1.0:
        raise ValueError("'poisson_ratio' must lie in (-1, 1).")
    return float(e * h**3 / (12.0 * (1.0 - poisson_ratio**2)))


def beam_bending_wave_speed(
    frequency: ArrayLike, bending_stiffness: float, mass_per_length: float
) -> NDArray[np.float64]:
    """Free bending wave speed of a beam ``cB = (B omega**2 / m')**(1/4)``.

    :param frequency: Frequency ``f``, in hertz (scalar or array, > 0).
    :param bending_stiffness: Beam bending stiffness ``B = E I``, in N.m^2.
    :param mass_per_length: Mass per unit length ``m'``, in kg/m.
    :return: The bending phase speed ``cB``, in m/s.
    :raises ValueError: for a non-positive stiffness, mass or frequency.
    """
    b = require_positive(bending_stiffness, "bending_stiffness")
    m1 = require_positive(mass_per_length, "mass_per_length")
    omega = _omega(frequency)
    return np.asarray((b * omega**2 / m1) ** 0.25, dtype=np.float64)


def plate_bending_wave_speed(
    frequency: ArrayLike, bending_stiffness: float, mass_per_area: float
) -> NDArray[np.float64]:
    """Free bending wave speed of a plate ``cB = (B' omega**2 / m'')**(1/4)``.

    :param frequency: Frequency ``f``, in hertz (scalar or array, > 0).
    :param bending_stiffness: Plate bending stiffness per unit width ``B'``,
        in N.m.
    :param mass_per_area: Mass per unit area ``m''``, in kg/m^2.
    :return: The bending phase speed ``cB``, in m/s.
    :raises ValueError: for a non-positive stiffness, mass or frequency.
    """
    b = require_positive(bending_stiffness, "bending_stiffness")
    m2 = require_positive(mass_per_area, "mass_per_area")
    omega = _omega(frequency)
    return np.asarray((b * omega**2 / m2) ** 0.25, dtype=np.float64)


def _omega(frequency: ArrayLike) -> NDArray[np.float64]:
    """Angular frequency ``omega = 2 pi f`` (rad/s); rejects non-positive f."""
    f = np.asarray(frequency, dtype=np.float64)
    if np.any(f <= 0.0):
        raise ValueError("'frequency' must be positive.")
    return np.asarray(2.0 * np.pi * f, dtype=np.float64)


# ---------------------------------------------------------------------------
# Thin plate in bending (frequency independent, real).
# ---------------------------------------------------------------------------


def infinite_plate_impedance(
    bending_stiffness: float, mass_per_area: float, *, location: str = "centre"
) -> float:
    """Point impedance of an infinite thin plate (Cremer Table 5.1).

    ``Z = C sqrt(B' m'')`` with ``C = 8`` for a force at the plate centre and
    ``C = 3.5`` for a force at a free edge. The impedance is purely real and
    frequency independent: an infinite plate presents a matched resistance to a
    point force.

    :param bending_stiffness: Plate bending stiffness per unit width ``B'``,
        in N.m (see :func:`plate_bending_stiffness`).
    :param mass_per_area: Mass per unit area ``m''``, in kg/m^2.
    :param location: ``"centre"`` (``C = 8``) or ``"edge"`` (``C = 3.5``).
    :return: The point impedance ``Z``, in N.s/m.
    :raises ValueError: for a non-positive stiffness/mass or unknown location.
    """
    b = require_positive(bending_stiffness, "bending_stiffness")
    m2 = require_positive(mass_per_area, "mass_per_area")
    if location not in _PLATE_CONSTANT:
        raise ValueError("'location' must be 'centre' or 'edge'.")
    return float(_PLATE_CONSTANT[location] * np.sqrt(b * m2))


def infinite_plate_mobility(
    bending_stiffness: float, mass_per_area: float, *, location: str = "centre"
) -> float:
    """Point mobility of an infinite thin plate ``Y = 1 / (C sqrt(B' m''))``.

    The reciprocal of :func:`infinite_plate_impedance` (real, frequency
    independent).

    :param bending_stiffness: Plate bending stiffness per unit width ``B'``,
        in N.m.
    :param mass_per_area: Mass per unit area ``m''``, in kg/m^2.
    :param location: ``"centre"`` (``C = 8``) or ``"edge"`` (``C = 3.5``).
    :return: The point mobility ``Y``, in m/(N.s).
    :raises ValueError: for a non-positive stiffness/mass or unknown location.
    """
    return 1.0 / infinite_plate_impedance(
        bending_stiffness, mass_per_area, location=location
    )


# ---------------------------------------------------------------------------
# Slender beam in bending (frequency dependent, complex).
# ---------------------------------------------------------------------------


def infinite_beam_impedance(
    frequency: ArrayLike,
    bending_stiffness: float,
    mass_per_length: float,
    *,
    location: str = "centre",
) -> NDArray[np.complex128]:
    """Point impedance of an infinite beam in bending (Cremer Table 5.1).

    ``Z = 2 m' cB (1 + j)`` for a force at the centre of the beam and
    ``Z = (m' cB / 2)(1 + j)`` for a force at a free end, with the bending wave
    speed ``cB = (B omega**2 / m')**(1/4)`` (:func:`beam_bending_wave_speed`).

    :param frequency: Frequency ``f``, in hertz (scalar or array, > 0).
    :param bending_stiffness: Beam bending stiffness ``B = E I``, in N.m^2.
    :param mass_per_length: Mass per unit length ``m'``, in kg/m.
    :param location: ``"centre"`` or ``"end"``.
    :return: The complex point impedance ``Z``, in N.s/m.
    :raises ValueError: for a non-positive input or unknown location.
    """
    if location not in _BEAM_CONSTANT:
        raise ValueError("'location' must be 'centre' or 'end'.")
    m1 = require_positive(mass_per_length, "mass_per_length")
    c_b = beam_bending_wave_speed(frequency, bending_stiffness, m1)
    z = _BEAM_CONSTANT[location] * m1 * c_b * (1.0 + 1j)
    return np.asarray(z, dtype=np.complex128)


def infinite_beam_mobility(
    frequency: ArrayLike,
    bending_stiffness: float,
    mass_per_length: float,
    *,
    location: str = "centre",
) -> NDArray[np.complex128]:
    """Point mobility of an infinite beam in bending (Cremer Table 5.1).

    ``Y = (1 - j) / (4 m' cB)`` for a force at the centre and
    ``Y = (1 - j) / (m' cB)`` for a force at a free end, the reciprocal of
    :func:`infinite_beam_impedance`. The mobility falls as ``omega**(-1/2)``.

    :param frequency: Frequency ``f``, in hertz (scalar or array, > 0).
    :param bending_stiffness: Beam bending stiffness ``B = E I``, in N.m^2.
    :param mass_per_length: Mass per unit length ``m'``, in kg/m.
    :param location: ``"centre"`` or ``"end"``.
    :return: The complex point mobility ``Y``, in m/(N.s).
    :raises ValueError: for a non-positive input or unknown location.
    """
    z = infinite_beam_impedance(
        frequency, bending_stiffness, mass_per_length, location=location
    )
    return np.asarray(1.0 / z, dtype=np.complex128)


def infinite_beam_moment_mobility(
    frequency: ArrayLike, bending_stiffness: float, mass_per_length: float
) -> NDArray[np.complex128]:
    """Moment (rotational) mobility of an infinite beam (Cremer Eq. 5.75).

    ``Y_M = omega (1 + j) / (4 B kB)`` with the bending wavenumber
    ``kB = omega / cB``, the angular velocity per unit applied moment at the
    driving point.

    :param frequency: Frequency ``f``, in hertz (scalar or array, > 0).
    :param bending_stiffness: Beam bending stiffness ``B = E I``, in N.m^2.
    :param mass_per_length: Mass per unit length ``m'``, in kg/m.
    :return: The complex moment mobility ``Y_M``, in rad/(N.m.s).
    :raises ValueError: for a non-positive input.
    """
    b = require_positive(bending_stiffness, "bending_stiffness")
    m1 = require_positive(mass_per_length, "mass_per_length")
    omega = _omega(frequency)
    c_b = beam_bending_wave_speed(frequency, b, m1)
    k_b = omega / c_b
    y = omega * (1.0 + 1j) / (4.0 * b * k_b)
    return np.asarray(y, dtype=np.complex128)


# ---------------------------------------------------------------------------
# Longitudinal rod (real).
# ---------------------------------------------------------------------------


def longitudinal_rod_impedance(
    density: float, longitudinal_wave_speed: float, cross_section_area: float
) -> float:
    """Point impedance of an infinite rod in longitudinal motion (Table 5.1).

    ``Z = rho cL S``, real and frequency independent.

    :param density: Material density ``rho``, in kg/m^3.
    :param longitudinal_wave_speed: Longitudinal wave speed ``cL``, in m/s.
    :param cross_section_area: Cross-section area ``S``, in m^2.
    :return: The point impedance ``Z``, in N.s/m.
    :raises ValueError: for a non-positive input.
    """
    rho = require_positive(density, "density")
    c_l = require_positive(longitudinal_wave_speed, "longitudinal_wave_speed")
    s = require_positive(cross_section_area, "cross_section_area")
    return float(rho * c_l * s)


def longitudinal_rod_mobility(
    density: float, longitudinal_wave_speed: float, cross_section_area: float
) -> float:
    """Point mobility of an infinite rod ``Y = 1 / (rho cL S)`` (Table 5.1).

    :param density: Material density ``rho``, in kg/m^3.
    :param longitudinal_wave_speed: Longitudinal wave speed ``cL``, in m/s.
    :param cross_section_area: Cross-section area ``S``, in m^2.
    :return: The point mobility ``Y``, in m/(N.s).
    :raises ValueError: for a non-positive input.
    """
    return 1.0 / longitudinal_rod_impedance(
        density, longitudinal_wave_speed, cross_section_area
    )


# ---------------------------------------------------------------------------
# Injected power and bundled results.
# ---------------------------------------------------------------------------


def injected_power(force: ArrayLike, mobility: ArrayLike) -> NDArray[np.float64]:
    """Time-averaged vibrational power injected by a point force (Cremer 5.23).

    ``W = 0.5 |F|**2 Re{Y}``: only the real part (conductance) of the mobility
    carries power; the reactive part stores near-field energy.

    :param force: Point-force amplitude ``F`` (peak, scalar or array), in N.
    :param mobility: Complex point mobility ``Y`` (broadcast with *force*),
        in m/(N.s).
    :return: The injected power ``W``, in W.
    """
    f = np.asarray(force, dtype=np.complex128)
    y = np.asarray(mobility, dtype=np.complex128)
    return np.asarray(0.5 * np.abs(f) ** 2 * y.real, dtype=np.float64)


def infinite_plate_point_mobility(
    frequency: ArrayLike,
    bending_stiffness: float,
    mass_per_area: float,
    *,
    location: str = "centre",
) -> MobilityResult:
    """Infinite-plate point mobility bundled as a :class:`MobilityResult`.

    The plate mobility is frequency independent, so the returned spectrum is
    constant across *frequency*; bundling it lets it be plotted and converted
    with the ISO 7626 mobility machinery.

    :param frequency: Frequencies ``f``, in hertz (array, > 0).
    :param bending_stiffness: Plate bending stiffness per unit width ``B'``,
        in N.m.
    :param mass_per_area: Mass per unit area ``m''``, in kg/m^2.
    :param location: ``"centre"`` or ``"edge"``.
    :return: The :class:`MobilityResult` (driving point).
    """
    freq = np.atleast_1d(np.asarray(frequency, dtype=np.float64))
    if np.any(freq <= 0.0):
        raise ValueError("'frequency' must be positive.")
    y = infinite_plate_mobility(bending_stiffness, mass_per_area, location=location)
    mob = np.full(freq.shape, y, dtype=np.complex128)
    return MobilityResult(frequencies=freq, mobility=mob, driving_point=True)


def infinite_beam_point_mobility(
    frequency: ArrayLike,
    bending_stiffness: float,
    mass_per_length: float,
    *,
    location: str = "centre",
) -> MobilityResult:
    """Infinite-beam point mobility bundled as a :class:`MobilityResult`.

    :param frequency: Frequencies ``f``, in hertz (array, > 0).
    :param bending_stiffness: Beam bending stiffness ``B = E I``, in N.m^2.
    :param mass_per_length: Mass per unit length ``m'``, in kg/m.
    :param location: ``"centre"`` or ``"end"``.
    :return: The :class:`MobilityResult` (driving point).
    """
    freq = np.atleast_1d(np.asarray(frequency, dtype=np.float64))
    mob = infinite_beam_mobility(
        freq, bending_stiffness, mass_per_length, location=location
    )
    return MobilityResult(frequencies=freq, mobility=mob, driving_point=True)
