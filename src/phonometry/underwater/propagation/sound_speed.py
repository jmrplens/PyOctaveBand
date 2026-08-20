#  Copyright (c) 2026. Jose Manuel Requena Plens
r"""Speed of sound in sea water (empirical equations).

Four coexisting equations for the sound speed ``c`` as a function of
temperature, salinity and depth/pressure, selectable through ``model``:

* ``"unesco"`` -- the UNESCO / Chen & Millero (1977) algorithm, the
  international standard, in the Wong & Zhu (1995) ITS-90 recalculation. Default.
* ``"del_grosso"`` -- the Del Grosso (1974) equation (Wong & Zhu 1995 form),
  a high-accuracy alternative over a narrower domain.
* ``"mackenzie"`` -- the Mackenzie (1981) nine-term depth-based equation.
* ``"medwin"`` -- the Medwin (1975) six-term short formula, the simplest member
  of the family, whose partial derivatives are the classic rules of thumb
  :math:`\partial c/\partial T \approx 4.6 - 0.110 T` m/s per °C and
  :math:`\partial c/\partial z \approx 0.016` m/s per m.

The UNESCO and Del Grosso equations use pressure, not depth, so a depth is first
converted with the Leroy & Parthiot (1998) standard-ocean formula
(:func:`depth_to_pressure`). :class:`SoundSpeedProfile` evaluates ``c`` over a
depth profile and exposes the sound-speed gradient.

Sources (clean-room, implemented from the equations, validated by cross-model
agreement and the canonical Mackenzie check value 1550.744 m/s at 25 °C, 35 ppt,
1000 m): NPL Technical Guide "Speed of Sound in Sea-Water" (Wong & Zhu 1995
coefficient tables), Mackenzie (1981) JASA 70, Del Grosso (1974) JASA 56,
Leroy & Parthiot (1998) JASA 103, Medwin (1975) as printed in Ainslie,
*Principles of Sonar Performance Modelling* (Springer 2010), Equations
(1.2)-(1.4), printed p. 20.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from numpy.typing import NDArray

#: Bar per megapascal (1 MPa = 10 bar).
_BAR_PER_MPA = 10.0
#: kg/cm² per bar (100 kPa = 1.019716 kg/cm²; 1 bar = 100 kPa).
_KGCM2_PER_BAR = 1.019716

_MODELS = ("unesco", "del_grosso", "mackenzie", "medwin")
#: Models that take a depth directly instead of a pressure.
_DEPTH_MODELS = ("mackenzie", "medwin")


def _positive(value: float, name: str) -> float:
    scalar = float(value)
    if not np.isfinite(scalar) or scalar <= 0.0:
        raise ValueError(f"'{name}' must be a positive, finite number.")
    return scalar


def _finite(value: float, name: str) -> float:
    scalar = float(value)
    if not np.isfinite(scalar):
        raise ValueError(f"'{name}' must be a finite number.")
    return scalar


def depth_to_pressure(depth: float, latitude: float = 45.0) -> float:
    r"""Gauge pressure at a given ocean depth (Leroy & Parthiot 1998), in
    MPa.

    Standard-ocean formula (an ideal medium of 0 °C and 35 ppt); no local
    corrections are applied.

    :param depth: Depth below the surface ``Z``, in metres (``>= 0``).
    :param latitude: Latitude :math:`\varphi`, in degrees (default 45°).
    :return: Gauge pressure, in megapascals.
    :raises ValueError: If the depth is negative or non-finite.
    """
    z = _finite(depth, "depth")
    if z < 0.0:
        raise ValueError("'depth' must be non-negative.")
    return float(_pressure_mpa(z, _finite(latitude, "latitude")))


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


def sea_water_sound_speed(
    temperature: float,
    salinity: float,
    depth: float,
    *,
    model: str = "unesco",
    latitude: float = 45.0,
) -> float:
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
    t = _finite(temperature, "temperature")
    s = _finite(salinity, "salinity")
    if s < 0.0:
        raise ValueError("'salinity' must be non-negative.")
    z = _finite(depth, "depth")
    if z < 0.0:
        raise ValueError("'depth' must be non-negative.")
    key = model.strip().lower()
    if key == "mackenzie":
        return float(_mackenzie(t, s, z))
    if key == "medwin":
        return float(_medwin(t, s, z))
    pressure_bar = depth_to_pressure(z, latitude) * _BAR_PER_MPA
    if key == "unesco":
        return float(_unesco(t, s, pressure_bar))
    if key == "del_grosso":
        return float(_del_grosso(t, s, pressure_bar * _KGCM2_PER_BAR))
    raise ValueError(f"'model' must be one of {_MODELS}, got {model!r}.")


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
    if z.ndim != 1 or z.size < 2:
        raise ValueError("'depths' must be a 1-D array of at least two depths.")
    if np.any(z < 0.0) or not np.all(np.isfinite(z)):
        raise ValueError("'depths' must be finite and non-negative.")
    if np.any(np.diff(z) <= 0.0):
        raise ValueError("'depths' must be strictly increasing.")
    temp = np.broadcast_to(np.asarray(temperatures, dtype=np.float64), z.shape)
    sal = np.broadcast_to(np.asarray(salinities, dtype=np.float64), z.shape)
    if not (np.all(np.isfinite(temp)) and np.all(np.isfinite(sal))):
        raise ValueError("'temperatures' and 'salinities' must be finite.")
    if np.any(sal < 0.0):
        raise ValueError("'salinities' must be non-negative.")
    key = model.strip().lower()
    lat = _finite(latitude, "latitude")
    if key in _DEPTH_MODELS:
        c_any = (
            _mackenzie(temp, sal, z) if key == "mackenzie" else _medwin(temp, sal, z)
        )
    elif key in ("unesco", "del_grosso"):
        pressure_bar = np.asarray(_pressure_mpa(z, lat)) * _BAR_PER_MPA
        c_any = (
            _unesco(temp, sal, pressure_bar)
            if key == "unesco"
            else _del_grosso(temp, sal, pressure_bar * _KGCM2_PER_BAR)
        )
    else:
        raise ValueError(f"'model' must be one of {_MODELS}, got {model!r}.")
    c = np.asarray(c_any, dtype=np.float64)
    gradient = np.gradient(c, z)
    return SoundSpeedProfile(
        depth=z, sound_speed=c, gradient=gradient, model=model.strip().lower()
    )
