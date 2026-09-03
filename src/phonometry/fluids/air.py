#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Humid air (IEC 61094-2:2009, Annex F).

Annex F carries the CIPM-2007 formulation for the density of moist air together
with fits for the speed of sound, the ratio of specific heats, the viscosity and
the thermal diffusivity. It is the only model of air in this library that is
computed rather than quoted: every other air in the tree is a value some
standard printed for its own procedure, and those stay where their clause prints
them.

Table F.1 (printed folio 40) prints five quantities at two condition sets, and
every one of them reproduces here from the Table F.2 coefficients (printed folio
41) to better than 1,5e-7 relative, which is inside the rounding of the last
printed figure. Thermal conductivity and specific heat capacity come from the
two expressions of Clause F.6 (printed folio 39); the annex prints their formulae
and units but tabulates only the diffusivity they form, so those two are anchored
in closed form rather than against a printed number.
"""

from __future__ import annotations

import math
import warnings

from ._state import Fluid, FluidAssumptionWarning, FluidWarning

__all__ = [
    "DEFAULT_CO2_MOLE_FRACTION",
    "DEFAULT_RELATIVE_HUMIDITY_PERCENT",
    "DEFAULT_STATIC_PRESSURE_PA",
    "air",
]

#: One standard atmosphere, in pascals. Assumed when no pressure is supplied.
DEFAULT_STATIC_PRESSURE_PA = 101325.0
#: Relative humidity assumed when none is supplied, in per cent. There is no
#: standard humidity: Annex F's own two examples use 50 % and 65 %.
DEFAULT_RELATIVE_HUMIDITY_PERCENT = 50.0
#: Carbon dioxide mole fraction assumed when none is supplied. Clause F.2:
#: "The composition of standard air is based upon a carbon dioxide mole fraction
#: of 0,000 314. It is generally accepted that under laboratory conditions a
#: higher value is found and in the absence of actual measurements a value of
#: x_c = 0,000 4 is recommended."
DEFAULT_CO2_MOLE_FRACTION = 0.0004

#: Relative humidity is saturation at 100 %, so that is the bound, not a choice.
_SATURATED_PERCENT = 100.0

#: Celsius-to-kelvin offset, in kelvin. Annex F writes some of its expressions
#: in kelvin and others in degrees Celsius; each is converted at its own use.
_ABSOLUTE_ZERO_C_OFFSET = 273.15

#: The domain Annex F states for itself, printed folio 38: "The equations used
#: for the calculations are considered valid for environmental conditions within
#: the ranges: temperature 15 °C - 27 °C / static pressure 60 kPa - 110 kPa /
#: relative humidity 10 % - 90 %". No domain is printed for the CO2 fraction.
_VALID_TEMPERATURE_C = (15.0, 27.0)
_VALID_STATIC_PRESSURE_PA = (60_000.0, 110_000.0)
_VALID_RELATIVE_HUMIDITY_PERCENT = (10.0, 90.0)

_VALIDITY = (
    "IEC 61094-2:2009 Annex F states 15 degC to 27 degC, 60 kPa to 110 kPa and "
    "10 % to 90 % relative humidity; it states no range for the carbon dioxide "
    "mole fraction."
)

# --- Table F.2 coefficients, printed folio 41 -------------------------------
# The table prints no units column and mixes its temperature convention across
# columns, so each set records the argument its expression takes. Table F.1
# pins every one of them, which is what makes the reproduction test a guard
# rather than a restatement.
_PSV = (1.2378847e-5, -1.9121316e-2, 33.93711047, -6.3431645e3)  # T in kelvin
_ENHANCEMENT = (1.00062, 3.14e-8, 5.6e-7)  # p_s in pascals, t in degrees Celsius
_COMPRESSIBILITY = (
    1.58123e-6, -2.9331e-8, 1.1043e-10, 5.707e-6, -2.051e-8,
    1.9898e-4, -2.376e-6, 1.83e-11, -0.765e-8,
)  # fmt: skip
_SPEED_OF_SOUND = (
    331.5024, 0.603055, -0.000528, 51.471935, 0.1495874, -0.000782,
    -1.82e-7, 3.73e-8, -2.93e-10, -85.20931, -0.228525, 5.91e-5,
    -2.835149, -2.15e-13, 29.179762, 0.000486,
)  # fmt: skip
_HEAT_CAPACITY_RATIO = (
    1.400822, -1.75e-5, -1.73e-7, -0.0873629, -1.665e-4, -3.26e-6,
    2.047e-8, -1.26e-10, 5.939e-14, -0.1199717, -8.693e-4, 1.979e-6,
    -0.01104, -3.478e-16, 0.0450616, 1.82e-6,
)  # fmt: skip
_VISCOSITY = (84.986, 7.0, 113.157, -1.0, -3.7501e-3, -100.015)  # T in kelvin
_THERMAL_CONDUCTIVITY = (60.054, 1.846, 2.06e-6, 40.0, -1.775e-4)  # T in kelvin
_SPECIFIC_HEAT = (
    0.251625, -9.2525e-5, 2.1334e-7, -1.0043e-10, 0.12477, -2.283e-5,
    1.267e-7, 0.01116, 4.61e-6, 1.74e-8,
)  # fmt: skip
#: Calories to joules. Clause F.6 writes both thermal expressions scaled by it.
_CALORIE_J = 4186.8

_MODEL = "IEC 61094-2:2009 Annex F (CIPM-2007)"


def _saturation_vapour_pressure(kelvin: float) -> float:
    """``p_sv``, in pascals (Annex F, printed folio 38)."""
    a = _PSV
    return math.exp(a[0] * kelvin**2 + a[1] * kelvin + a[2] + a[3] / kelvin)


def _enhancement_factor(static_pressure_pa: float, temperature_c: float) -> float:
    """``f(p_s, t)``, dimensionless (Annex F, printed folio 38)."""
    a = _ENHANCEMENT
    return a[0] + a[1] * static_pressure_pa + a[2] * temperature_c**2


def _compressibility(
    static_pressure_pa: float, temperature_c: float, water_fraction: float
) -> float:
    """``Z``, dimensionless (Annex F, printed folio 38)."""
    a = _COMPRESSIBILITY
    t, p, xw = temperature_c, static_pressure_pa, water_fraction
    kelvin = _ABSOLUTE_ZERO_C_OFFSET + t
    bracket = (
        a[0]
        + a[1] * t
        + a[2] * t**2
        + (a[3] + a[4] * t) * xw
        + (a[5] + a[6] * t) * xw**2
    )
    return 1.0 - (p / kelvin) * bracket + (p**2 / kelvin**2) * (a[7] + a[8] * xw**2)


def _quadratic_in_t(
    a: tuple[float, ...],
    temperature_c: float,
    static_pressure_pa: float,
    water_fraction: float,
    co2_mole_fraction: float,
) -> float:
    """The sixteen-coefficient form Annex F uses for ``c_0`` and ``kappa``.

    Both are written as the same polynomial in temperature, water-vapour mole
    fraction, static pressure and carbon dioxide mole fraction, so the shape is
    spelled once and the two coefficient sets are passed in.
    """
    t, p, xw, xc = temperature_c, static_pressure_pa, water_fraction, co2_mole_fraction
    return (
        a[0]
        + a[1] * t
        + a[2] * t**2
        + (a[3] + a[4] * t + a[5] * t**2) * xw
        + (a[6] + a[7] * t + a[8] * t**2) * p
        + (a[9] + a[10] * t + a[11] * t**2) * xc
        + a[12] * xw**2
        + a[13] * p**2
        + a[14] * xc**2
        + a[15] * xw * p * xc
    )


def _warn_assumptions(
    static_pressure_pa: float | None, relative_humidity_percent: float | None
) -> None:
    """Announce, once, whichever of the two conditions was not supplied.

    One warning rather than two, because a caller who supplied neither has one
    thing to fix, not two, and because the two matter for opposite reasons: the
    humidity is the one nobody knows without measuring it, and the pressure is
    the one that costs most when it is wrong.
    """
    assumed = []
    if static_pressure_pa is None:
        assumed.append(f"{DEFAULT_STATIC_PRESSURE_PA:.0f} Pa")
    if relative_humidity_percent is None:
        assumed.append(f"{DEFAULT_RELATIVE_HUMIDITY_PERCENT:.0f} % relative humidity")
    if not assumed:
        return
    msg = (
        f"air() assumed {' and '.join(assumed)}. The pressure is the one worth "
        f"measuring: a site 1000 m up sits near 90 kPa, and taking it for one "
        f"standard atmosphere puts the density about 13 % high. The whole span "
        f"of humidity, 0 % to 100 %, is worth about 1 % of the density. Pass "
        f"the conditions to silence this."
    )
    warnings.warn(msg, FluidAssumptionWarning, stacklevel=3)


def _warn_outside_domain(
    temperature_c: float, static_pressure_pa: float, relative_humidity_percent: float
) -> None:
    """Announce a state outside the domain Annex F states for itself."""
    outside = []
    for value, (low, high), what, unit in (
        (temperature_c, _VALID_TEMPERATURE_C, "temperature", "degC"),
        (static_pressure_pa, _VALID_STATIC_PRESSURE_PA, "static pressure", "Pa"),
        (
            relative_humidity_percent,
            _VALID_RELATIVE_HUMIDITY_PERCENT,
            "relative humidity",
            "%",
        ),
    ):
        if not low <= value <= high:
            outside.append(f"{what} {value:g} {unit} (stated {low:g} to {high:g})")
    if not outside:
        return
    msg = (
        f"Air state outside the domain IEC 61094-2:2009 Annex F states for its "
        f"equations, printed folio 38: {'; '.join(outside)}. The result is an "
        f"extrapolation of a fit, not a refusal: the annex states where it was "
        f"validated, not what air can be."
    )
    warnings.warn(msg, FluidWarning, stacklevel=3)


def air(
    *,
    temperature_c: float,
    static_pressure_pa: float | None = None,
    relative_humidity_percent: float | None = None,
    co2_mole_fraction: float | None = None,
) -> Fluid:
    r"""Humid air at one state (IEC 61094-2:2009, Annex F).

    Returns the density, speed of sound, ratio of specific heats, viscosity and
    thermal diffusivity Table F.1 tabulates, and the thermal conductivity and
    specific heat capacity Clause F.6 gives expressions for.

    :param temperature_c: Air temperature ``t``, in **degrees Celsius**.
        Required: there is no defensible default for the one condition the
        caller actually measured.
    :param static_pressure_pa: Absolute static pressure ``p_s``, in **pascals**.
        ``None`` assumes :data:`DEFAULT_STATIC_PRESSURE_PA` and warns.
    :param relative_humidity_percent: Relative humidity ``H``, in **per cent**.
        ``None`` assumes :data:`DEFAULT_RELATIVE_HUMIDITY_PERCENT` and warns.
    :param co2_mole_fraction: Carbon dioxide mole fraction ``x_c``.
        ``None`` takes :data:`DEFAULT_CO2_MOLE_FRACTION`, the value Clause F.2
        recommends for laboratory conditions, and does **not** warn: unlike the
        other two it is a value the annex names, and it reaches the fifth figure
        of the density at most.
    :return: The :class:`~phonometry.fluids.Fluid` at that state.
    :raises ValueError: if the temperature is at or below -273,15 degC, the
        pressure is not positive, the humidity is outside 0 % to 100 %, or the
        carbon dioxide mole fraction is outside 0 to 1.

    Nothing else is refused. Annex F states a domain for its equations and this
    warns outside it, because a fit past its range is still arithmetic; what it
    refuses is a state that cannot exist.
    """
    kelvin = _ABSOLUTE_ZERO_C_OFFSET + float(temperature_c)
    if not math.isfinite(kelvin) or kelvin <= 0.0:
        msg = "'temperature_c' must be a finite temperature above -273.15 degC."
        raise ValueError(msg)

    _warn_assumptions(static_pressure_pa, relative_humidity_percent)
    pressure = (
        DEFAULT_STATIC_PRESSURE_PA
        if static_pressure_pa is None
        else float(static_pressure_pa)
    )
    humidity = (
        DEFAULT_RELATIVE_HUMIDITY_PERCENT
        if relative_humidity_percent is None
        else float(relative_humidity_percent)
    )
    carbon_dioxide = (
        DEFAULT_CO2_MOLE_FRACTION
        if co2_mole_fraction is None
        else float(co2_mole_fraction)
    )
    if not math.isfinite(pressure) or pressure <= 0.0:
        msg = "'static_pressure_pa' must be a finite positive pressure."
        raise ValueError(msg)
    if not math.isfinite(humidity) or not 0.0 <= humidity <= _SATURATED_PERCENT:
        msg = "'relative_humidity_percent' must be between 0 and 100."
        raise ValueError(msg)
    if not math.isfinite(carbon_dioxide) or not 0.0 <= carbon_dioxide <= 1.0:
        msg = "'co2_mole_fraction' must be a mole fraction between 0 and 1."
        raise ValueError(msg)

    _warn_outside_domain(float(temperature_c), pressure, humidity)

    t = float(temperature_c)
    saturation = _saturation_vapour_pressure(kelvin)
    enhancement = _enhancement_factor(pressure, t)
    water = (humidity / 100.0) * (saturation / pressure) * enhancement
    compressibility = _compressibility(pressure, t, water)

    density = (
        (3.483740 + 1.4446 * (carbon_dioxide - 0.0004))
        * 1e-3
        * pressure
        / (compressibility * kelvin)
        * (1.0 - 0.3780 * water)
    )
    speed_of_sound = _quadratic_in_t(
        _SPEED_OF_SOUND, t, pressure, water, carbon_dioxide
    )
    heat_capacity_ratio = _quadratic_in_t(
        _HEAT_CAPACITY_RATIO, t, pressure, water, carbon_dioxide
    )
    v = _VISCOSITY
    viscosity = (
        v[0] + v[1] * kelvin + (v[2] + v[3] * kelvin) * water + v[4] * kelvin**2
        + v[5] * water**2
    ) * 1e-8  # fmt: skip
    k = _THERMAL_CONDUCTIVITY
    thermal_conductivity = (
        _CALORIE_J
        * (k[0] + k[1] * kelvin + k[2] * kelvin**2 + (k[3] + k[4] * kelvin) * water)
        * 1e-8
    )
    cp = _SPECIFIC_HEAT
    specific_heat = _CALORIE_J * (
        cp[0]
        + cp[1] * kelvin
        + cp[2] * kelvin**2
        + cp[3] * kelvin**3
        + (cp[4] + cp[5] * kelvin + cp[6] * kelvin**2) * water
        + (cp[7] + cp[8] * kelvin + cp[9] * kelvin**2) * water**2
    )

    return Fluid(
        temperature_c=t,
        static_pressure_pa=pressure,
        composition={
            "relative_humidity_percent": humidity,
            "co2_mole_fraction": carbon_dioxide,
            "water_vapour_mole_fraction": water,
        },
        model=_MODEL,
        validity=_VALIDITY,
        properties={
            "density": density,
            "speed_of_sound": speed_of_sound,
            "heat_capacity_ratio": heat_capacity_ratio,
            "viscosity": viscosity,
            "thermal_diffusivity": thermal_conductivity / (density * specific_heat),
            "thermal_conductivity": thermal_conductivity,
            "specific_heat_capacity": specific_heat,
        },
    )
