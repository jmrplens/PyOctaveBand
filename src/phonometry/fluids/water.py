#  Copyright (c) 2026. Jose Manuel Requena Plens
r"""Sea water: its speed of sound, and its density.

Four coexisting equations for the sound speed ``c`` as a function of
temperature, salinity and depth or pressure, selectable through
``sound_speed_model``:

* ``"unesco"`` -- the UNESCO / Chen & Millero (1977) algorithm, the
  international standard, in the Wong & Zhu (1995) ITS-90 recalculation.
  Default.
* ``"del_grosso"`` -- the Del Grosso (1974) equation (Wong & Zhu 1995 form),
  a high-accuracy alternative over a narrower domain.
* ``"mackenzie"`` -- the Mackenzie (1981) nine-term depth-based equation.
* ``"medwin"`` -- the Medwin (1975) six-term short formula.

Unlike air, sea water has four competing fits to one quantity rather than one
model of a substance, which is why this constructor takes a model and
:func:`~phonometry.fluids.air` does not: those four are answers to the same
question, while the air formulas scattered through the library are clauses of
different measurement standards and must never be substitutable.

The density comes from Ainslie, *Principles of Sonar Performance Modelling*
(Springer 2010), Equation (4.6) on printed folio 127, which the book attributes
to Pierce (1989, p. 34). Its pressure argument is **absolute**, defined by
Equation (4.4) as the atmosphere plus the water column above.

Sea water has no ratio of specific heats or thermal diffusivity here: no source
in this library prints them, so a :class:`~phonometry.fluids.Fluid` built from
this module raises :class:`~phonometry.fluids.FluidPropertyUnavailable` for
them rather than inventing a number.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, overload

import numpy as np

from .._internal.validation import (
    require_above_absolute_zero,
    require_above_absolute_zero_array,
    require_finite_array,
)
from ._state import Fluid

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

#: Bar per megapascal (1 MPa = 10 bar).
_BAR_PER_MPA = 10.0
#: kg/cm² per bar (100 kPa = 1.019716 kg/cm²; 1 bar = 100 kPa).
_KGCM2_PER_BAR = 1.019716
#: Minimum depth samples in a profile: two points are the floor for a
#: piecewise-linear profile and for ``np.gradient``'s finite differences.
_MIN_POLYLINE_NODES = 2

_MODELS = ("unesco", "del_grosso", "mackenzie", "medwin")
#: Models that take a depth directly instead of a pressure.
_DEPTH_MODELS = ("mackenzie", "medwin")


def _positive(value: float, name: str) -> float:
    scalar = float(value)
    if not np.isfinite(scalar) or scalar <= 0.0:
        msg = f"'{name}' must be a positive, finite number."
        raise ValueError(msg)
    return scalar


def _finite(value: float, name: str) -> float:
    scalar = float(value)
    if not np.isfinite(scalar):
        msg = f"'{name}' must be a finite number."
        raise ValueError(msg)
    return scalar


def depth_to_gauge_pressure_mpa(*, depth_m: float, latitude_deg: float = 45.0) -> float:
    r"""Gauge pressure at an ocean depth (Leroy & Parthiot 1998), in MPa.

    The standard-ocean formula, for an ideal medium of 0 degC and 35 ppt; no
    local corrections are applied. This is the pressure the UNESCO and Del
    Grosso sound speeds want, and it is **gauge**: zero at the surface.

    Its companion :func:`depth_to_absolute_pressure_pa` answers the other
    question, in the other unit and from the other datum, which is why both
    names carry theirs. Handing one to what wants the other is a factor of a
    million and an offset of an atmosphere.

    :param depth_m: Depth below the surface ``Z``, in metres (``>= 0``).
    :param latitude_deg: Latitude :math:`\varphi`, in degrees (default 45).
    :return: Gauge pressure, in megapascals.
    :raises ValueError: If the depth is negative or non-finite.
    """
    z = _finite(depth_m, "depth_m")
    if z < 0.0:
        msg = "'depth_m' must be non-negative."
        raise ValueError(msg)
    return float(_pressure_mpa(z, _finite(latitude_deg, "latitude_deg")))


def _pressure_mpa(
    z: float | NDArray[np.float64],
    latitude_deg: float,
) -> float | NDArray[np.float64]:
    """Leroy & Parthiot standard-ocean pressure, array-capable kernel."""
    phi = np.radians(latitude_deg)
    h45 = 1.00818e-2 * z + 2.465e-8 * z**2 - 1.25e-13 * z**3 + 2.8e-19 * z**4
    g = 9.7803 * (1.0 + 5.3e-3 * np.sin(phi) ** 2)
    k = (g - 2e-5 * z) / (9.80612 - 2e-5 * z)
    result: float | NDArray[np.float64] = h45 * k
    return result


# ---------------------------------------------------------------------------
# Mackenzie (1981)
# ---------------------------------------------------------------------------


def _mackenzie(
    t: float | NDArray[np.float64],
    s: float | NDArray[np.float64],
    depth: float | NDArray[np.float64],
) -> float | NDArray[np.float64]:
    return (
        1448.96
        + 4.591 * t
        - 5.304e-2 * t**2
        + 2.374e-4 * t**3
        + 1.340 * (s - 35.0)
        + 1.630e-2 * depth
        + 1.675e-7 * depth**2
        - 1.025e-2 * t * (s - 35.0)
        - 7.139e-13 * t * depth**3
    )


# ---------------------------------------------------------------------------
# Medwin (1975)
# ---------------------------------------------------------------------------


def _medwin(
    t: float | NDArray[np.float64],
    s: float | NDArray[np.float64],
    depth: float | NDArray[np.float64],
) -> float | NDArray[np.float64]:
    r"""Medwin's short formula, Ainslie (2010) Equation (1.2), printed p. 20.

    .. math::

       c = 1449.2 + 4.6 T + 0.016 z - 0.055 T^2
       + \left[ (1.34 - 0.010 T)(S - 35)
       + 2.9 \times 10^{-4} T^3 \right]

    The bracketed pair are the salinity and cubic-temperature
    corrections, negligible under typical ocean conditions.
    """
    return (
        1449.2
        + 4.6 * t
        + 0.016 * depth
        - 0.055 * t**2
        + (1.34 - 0.010 * t) * (s - 35.0)
        + 2.9e-4 * t**3
    )


# ---------------------------------------------------------------------------
# UNESCO / Chen & Millero (1977), Wong & Zhu (1995) ITS-90 coefficients
# ---------------------------------------------------------------------------

#: Cw(T,P) coefficients, indexed ``_C[power_of_T][power_of_P]``.
_C = (
    (1402.388, 0.153563, 3.1260e-5, -9.7729e-9),
    (5.03830, 6.8999e-4, -1.7111e-6, 3.8513e-10),
    (-5.81090e-2, -8.1829e-6, 2.5986e-8, -2.3654e-12),
    (3.3432e-4, 1.3632e-7, -2.5353e-10, 0.0),
    (-1.47797e-6, -6.1260e-10, 1.0415e-12, 0.0),
    (3.1419e-9, 0.0, 0.0, 0.0),
)
#: A(T,P) coefficients, indexed ``_A[power_of_T][power_of_P]``.
_A = (
    (1.389, 9.4742e-5, -3.9064e-7, 1.100e-10),
    (-1.262e-2, -1.2583e-5, 9.1061e-9, 6.651e-12),
    (7.166e-5, -6.4928e-8, -1.6009e-10, -3.391e-13),
    (2.008e-6, 1.0515e-8, 7.994e-12, 0.0),
    (-3.21e-8, -2.0142e-10, 0.0, 0.0),
)


def _unesco(
    t: float | NDArray[np.float64],
    s: float | NDArray[np.float64],
    pressure_bar: float | NDArray[np.float64],
) -> float | NDArray[np.float64]:
    p = pressure_bar
    cw = (
        sum(_C[i][0] * t**i for i in range(6))
        + sum(_C[i][1] * t**i for i in range(5)) * p
        + sum(_C[i][2] * t**i for i in range(5)) * p**2
        + sum(_C[i][3] * t**i for i in range(3)) * p**3
    )
    a = (
        sum(_A[i][0] * t**i for i in range(5))
        + sum(_A[i][1] * t**i for i in range(5)) * p
        + sum(_A[i][2] * t**i for i in range(4)) * p**2
        + sum(_A[i][3] * t**i for i in range(3)) * p**3
    )
    b = -1.922e-2 - 4.42e-5 * t + (7.3637e-5 + 1.7950e-7 * t) * p
    d = 1.727e-3 - 7.9836e-6 * p
    return cw + a * s + b * s**1.5 + d * s**2


# ---------------------------------------------------------------------------
# Del Grosso (1974), Wong & Zhu (1995) form
# ---------------------------------------------------------------------------


def _del_grosso(
    t: float | NDArray[np.float64],
    s: float | NDArray[np.float64],
    pressure_kgcm2: float | NDArray[np.float64],
) -> float | NDArray[np.float64]:
    p = pressure_kgcm2
    c000 = 1402.392
    d_ct = 0.5012285e1 * t - 0.551184e-1 * t**2 + 0.221649e-3 * t**3
    d_cs = 0.1329530e1 * s + 0.1288598e-3 * s**2
    d_cp = 0.1560592 * p + 0.2449993e-4 * p**2 - 0.8833959e-8 * p**3
    d_cstp = (
        0.6353509e-2 * t * p
        - 0.4383615e-6 * t**3 * p
        - 0.1593895e-5 * t * p**2
        + 0.2656174e-7 * t**2 * p**2
        + 0.5222483e-9 * t * p**3
        - 0.1275936e-1 * s * t
        + 0.9688441e-4 * s * t**2
        - 0.3406824e-3 * s * t * p
        + 0.4857614e-5 * s**2 * t * p
        - 0.1616745e-8 * s**2 * p**2
    )
    return c000 + d_ct + d_cs + d_cp + d_cstp


@overload
def sea_water_sound_speed(
    temperature: float,
    salinity: float,
    depth: float,
    *,
    model: str = ...,
    latitude: float = ...,
) -> float: ...


@overload
def sea_water_sound_speed(
    temperature: ArrayLike,
    salinity: ArrayLike,
    depth: ArrayLike,
    *,
    model: str = ...,
    latitude: float = ...,
) -> float | NDArray[np.float64]: ...


def sea_water_sound_speed(
    temperature: ArrayLike,
    salinity: ArrayLike,
    depth: ArrayLike,
    *,
    model: str = "unesco",
    latitude: float = 45.0,
) -> float | NDArray[np.float64]:
    """Speed of sound in sea water, in metres per second.

    :param temperature: Temperature ``T``, in degrees Celsius.
    :param salinity: Salinity ``S``, in parts per thousand (PSU).
    :param depth: Depth below the surface, in metres (``>= 0``).
    :param model: ``"unesco"`` (default), ``"del_grosso"``, ``"mackenzie"`` or
        ``"medwin"``.
    :param latitude: Latitude for the depth→pressure conversion, in degrees
        (used by ``"unesco"`` and ``"del_grosso"``; default 45°).
    :return: The sound speed ``c``, in m/s.
    :raises ValueError: If ``model`` is unknown or an input is non-finite.

    .. note::
        Each equation is a fit over a bounded oceanographic domain and
        **extrapolates silently outside it** (e.g. Del Grosso abused at
        T = 40 °C, S = 0, z = 11 km returns an unphysical ~1995 m/s).
        Published validity domains: UNESCO/Chen-Millero T 0-40 °C, S 0-40,
        P 0-1000 bar; Del Grosso T 0-30 °C, S 30-40, P 0-1000 kg/cm²;
        Mackenzie T 2-30 °C, S 25-40, depth 0-8000 m. Medwin is a
        deliberately simplified fit ("not accurate by modern standards", in
        Ainslie's words) and drifts by a few m/s against the UNESCO standard
        away from mid-range temperatures and shallow depths.
    """
    scalar = np.isscalar(temperature) and np.isscalar(salinity) and np.isscalar(depth)
    t = require_above_absolute_zero_array(temperature, "temperature")
    s = require_finite_array(salinity, "salinity")
    z = require_finite_array(depth, "depth")
    t, s, z = np.broadcast_arrays(t, s, z)
    if np.any(s < 0.0):
        msg = "'salinity' must be non-negative."
        raise ValueError(msg)
    if np.any(z < 0.0):
        msg = "'depth' must be non-negative."
        raise ValueError(msg)
    key = model.strip().lower()
    if key == "mackenzie":
        c = np.asarray(_mackenzie(t, s, z), dtype=np.float64)
    elif key == "medwin":
        c = np.asarray(_medwin(t, s, z), dtype=np.float64)
    elif key in ("unesco", "del_grosso"):
        pressure_bar = (
            np.asarray(_pressure_mpa(z, _finite(latitude, "latitude"))) * _BAR_PER_MPA
        )
        c = np.asarray(
            _unesco(t, s, pressure_bar)
            if key == "unesco"
            else _del_grosso(t, s, pressure_bar * _KGCM2_PER_BAR),
            dtype=np.float64,
        )
    else:
        msg = f"'model' must be one of {_MODELS}, got {model!r}."
        raise ValueError(msg)
    # The validators return at least 1-D, so a scalar call comes back as a
    # one-element array; unwrap it so the point case keeps its old type.
    return float(np.ravel(c)[0]) if scalar else c


# ---------------------------------------------------------------------------
# Density (Ainslie 2010, Eq. (4.6), printed folio 127)
# ---------------------------------------------------------------------------
#: Reference density of Eq. (4.6), in kg/m3, at 10 degC, salinity 35 and zero
#: gauge pressure.
_AINSLIE_RHO_REF = 1027.0
#: The four printed coefficients of Eq. (4.6), in the order they are printed.
_AINSLIE_PRESSURE = 4.3e-7
_AINSLIE_SALINITY = 0.75
_AINSLIE_TEMPERATURE = -0.16
_AINSLIE_TEMPERATURE2 = -0.004
#: The reference state the departures of Eq. (4.6) are measured from.
_AINSLIE_T_REF = 10.0
_AINSLIE_S_REF = 35.0
#: Eq. (4.11), printed folio 128: the absolute pressure of the standard ocean.
_AINSLIE_P_SCALE = 98066.5
_AINSLIE_P_SURFACE = 1.04
_AINSLIE_P_LINEAR = 0.102506
_AINSLIE_P_LATITUDE = 5.28e-3
_AINSLIE_P_QUADRATIC = 2.524e-7

#: What Ainslie states for the validity of the sound-speed fits, in prose.
_VALIDITY = (
    "The sound-speed fits are bounded: UNESCO/Chen-Millero 0 degC to 40 degC, "
    "salinity 0 to 40, 0 to 1000 bar; Del Grosso 0 degC to 30 degC, salinity "
    "30 to 40, 0 to 1000 kg/cm2; Mackenzie -2 degC to 30 degC, salinity 25 to "
    "40, 0 to 8000 m. Each extrapolates silently outside its own. Eq. (4.6) "
    "for the density states no domain."
)


def depth_to_absolute_pressure_pa(
    *, depth_m: float, latitude_deg: float = 45.0
) -> float:
    r"""Absolute static pressure at an ocean depth, in pascals.

    Ainslie Equation (4.11), printed folio 128. **Absolute**, not gauge:
    Equation (4.4) defines the static pressure as the atmosphere plus the
    weight of the water above, so this returns one atmosphere at the surface
    rather than zero.

    That is the pressure :func:`sea_water`'s density wants. The sound speeds
    want the other one, from :func:`depth_to_gauge_pressure_mpa`, and the two
    names say which is which because the difference is a factor of a million
    and an offset of an atmosphere.

    :param depth_m: Depth below the surface ``z``, in metres (``>= 0``).
    :param latitude_deg: Latitude :math:`\varphi`, in degrees (default 45).
    :return: Absolute static pressure, in pascals.
    :raises ValueError: If the depth is negative or non-finite.
    """
    z = _finite(depth_m, "depth_m")
    if z < 0.0:
        msg = "'depth_m' must be non-negative."
        raise ValueError(msg)
    phi = math.radians(_finite(latitude_deg, "latitude_deg"))
    return float(
        _AINSLIE_P_SCALE
        * (
            _AINSLIE_P_SURFACE
            + _AINSLIE_P_LINEAR * (1.0 + _AINSLIE_P_LATITUDE * math.sin(phi) ** 2) * z
            + _AINSLIE_P_QUADRATIC * z**2
        )
    )


def sea_water_density(
    *, temperature_c: float, salinity_psu: float, absolute_pressure_pa: float
) -> float:
    r"""Density of sea water, in kilograms per cubic metre.

    Ainslie Equation (4.6), printed folio 127, attributed there to Pierce
    (1989, p. 34):

    .. math::

       \rho = 1027 + 4{,}3\times10^{-7} P_\mathrm{w}
              + 0{,}75\,(S - 35) - 0{,}16\,(T - 10) - 0{,}004\,(T - 10)^2

    The pressure is **absolute**, in pascals, as Equations (4.4) and (4.7) to
    (4.10) define it. Use :func:`depth_to_absolute_pressure_pa` to get one from
    a depth.

    :param temperature_c: Temperature ``T``, in degrees Celsius.
    :param salinity_psu: Practical salinity ``S``, dimensionless.
    :param absolute_pressure_pa: Absolute static pressure ``P_w``, in pascals.
    :return: Density ``rho``, in kg/m3.
    :raises ValueError: For a temperature at or below absolute zero, a negative
        salinity, or a non-positive pressure.
    """
    t = require_above_absolute_zero(
        _finite(temperature_c, "temperature_c"), "temperature_c"
    )
    s = _finite(salinity_psu, "salinity_psu")
    if s < 0.0:
        msg = "'salinity_psu' must be non-negative."
        raise ValueError(msg)
    p = _positive(absolute_pressure_pa, "absolute_pressure_pa")
    return float(
        _AINSLIE_RHO_REF
        + _AINSLIE_PRESSURE * p
        + _AINSLIE_SALINITY * (s - _AINSLIE_S_REF)
        + _AINSLIE_TEMPERATURE * (t - _AINSLIE_T_REF)
        + _AINSLIE_TEMPERATURE2 * (t - _AINSLIE_T_REF) ** 2
    )


def sea_water(
    *,
    temperature_c: float,
    salinity_psu: float = 35.0,
    depth_m: float = 0.0,
    latitude_deg: float = 45.0,
    sound_speed_model: str = "unesco",
) -> Fluid:
    """Sea water at one point of the ocean.

    :param temperature_c: Temperature ``T``, in degrees Celsius.
    :param salinity_psu: Practical salinity ``S``, dimensionless (default 35,
        the salinity of the standard ocean).
    :param depth_m: Depth below the surface, in metres (default 0).
    :param latitude_deg: Latitude, in degrees, for the depth-to-pressure
        conversions (default 45).
    :param sound_speed_model: Which of the four fits to use; see the module
        docstring.
    :return: The :class:`~phonometry.fluids.Fluid` at that point.

    The density and the speed of sound come from different sources, which is
    why both are named in the result's ``model``. Neither source in this
    library prints a ratio of specific heats, a viscosity or a thermal
    diffusivity for sea water, so reading one raises rather than guessing.
    """
    speed = sea_water_sound_speed(
        temperature_c,
        salinity_psu,
        depth_m,
        model=sound_speed_model,
        latitude=latitude_deg,
    )
    absolute_pa = depth_to_absolute_pressure_pa(
        depth_m=depth_m, latitude_deg=latitude_deg
    )
    density = sea_water_density(
        temperature_c=temperature_c,
        salinity_psu=salinity_psu,
        absolute_pressure_pa=absolute_pa,
    )
    return Fluid(
        temperature_c=float(temperature_c),
        static_pressure_pa=absolute_pa,
        composition={
            "salinity_psu": float(salinity_psu),
            "depth_m": float(depth_m),
            "latitude_deg": float(latitude_deg),
        },
        model=(
            f"sound speed {sound_speed_model!r}; density Ainslie (2010) "
            f"Eq. (4.6) after Pierce (1989)"
        ),
        validity=_VALIDITY,
        properties={"density": density, "speed_of_sound": speed},
    )
