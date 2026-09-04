#  Copyright (c) 2026. Jose Manuel Requena Plens
"""The state of the medium (IEC 61094-2:2009, Annex F).

Annex F carries the CIPM-2007 formulation for humid air and tabulates five
quantities at two condition sets in Table F.1 (printed folio 40). Those ten
figures are the whole oracle, and the tolerance is read off the print rather
than chosen: a value given to seven significant figures is stated to within half
of its last figure, so reproducing it means landing inside that. Each row
reports the worst fraction of that allowance any of the five quantities used, so
a passing row means every printed figure came back.

The two quantities Clause F.6 gives expressions for but Table F.1 does not
tabulate, the thermal conductivity and the specific heat capacity, have no
printed value to check against. What can be checked is that they close the
identity Formula (F.5) states with the diffusivity that is printed, which is the
guard against transcribing either expression wrongly in a way their ratio would
hide.
"""

from __future__ import annotations

import warnings
from decimal import Decimal
from typing import TYPE_CHECKING

import phonometry as ph

from ..registry import Outcome, numeric, register

if TYPE_CHECKING:
    from phonometry.fluids import Fluid

_FLUIDS = "Humid air (IEC 61094-2:2009 Annex F)"

#: Table F.1 as printed, kept as strings so the allowance stays derivable.
_SET_A = {
    "density": "1.1860848",
    "speed_of_sound": "345.86652",
    "heat_capacity_ratio": "1.4007573",
    "viscosity": "1.826566e-5",
    "thermal_diffusivity": "2.115317e-5",
}
_SET_B = {
    "density": "0.9441589",
    "speed_of_sound": "344.38267",
    "heat_capacity_ratio": "1.4000266",
    "viscosity": "1.811295e-5",
    "thermal_diffusivity": "2.627024e-5",
}


def _half_of_the_last_printed_figure(printed: str) -> float:
    """Half a unit in the last place the annex printed, as an absolute bound."""
    exponent = int(Decimal(printed).as_tuple().exponent)
    return float(Decimal(5) * Decimal(10) ** (exponent - 1))


def _air(temperature_c: float, static_pressure_pa: float, humidity: float) -> Fluid:
    """Air at a fully stated condition, so nothing is assumed and nothing warns."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        return ph.fluids.air(
            temperature_c=temperature_c,
            static_pressure_pa=static_pressure_pa,
            relative_humidity_percent=humidity,
        )


def _worst_fraction_of_the_allowance(
    conditions: tuple[float, float, float], printed: dict[str, str]
) -> float:
    """The largest share of its printed rounding any of the five quantities used."""
    fluid = _air(*conditions)
    return max(
        abs(float(getattr(fluid, quantity)) - float(value))
        / _half_of_the_last_printed_figure(value)
        for quantity, value in printed.items()
    )


@register(
    _FLUIDS,
    "IEC 61094-2:2009 Table F.1",
    "Set A (23 C, 101 325 Pa, 50 % RH): rho, c0, kappa, eta and alpha_t, as a "
    "fraction of the rounding of the last printed figure",
)
def _chk_annex_f_set_a() -> Outcome:
    used = _worst_fraction_of_the_allowance((23.0, 101325.0, 50.0), _SET_A)
    return numeric(0.0, used, 1.0, places=3)


@register(
    _FLUIDS,
    "IEC 61094-2:2009 Table F.1",
    "Set B (20 C, 80 000 Pa, 65 % RH): rho, c0, kappa, eta and alpha_t, as a "
    "fraction of the rounding of the last printed figure",
)
def _chk_annex_f_set_b() -> Outcome:
    used = _worst_fraction_of_the_allowance((20.0, 80000.0, 65.0), _SET_B)
    return numeric(0.0, used, 1.0, places=3)


@register(
    _FLUIDS,
    "IEC 61094-2:2009 Formula (F.5)",
    "Thermal conductivity and specific heat capacity close the printed thermal "
    "diffusivity, alpha_t = k_a / (rho C_P)",
)
def _chk_annex_f_diffusivity_closes() -> Outcome:
    fluid = _air(23.0, 101325.0, 50.0)
    closed = fluid.thermal_conductivity / (fluid.density * fluid.specific_heat_capacity)
    return numeric(
        float(fluid.thermal_diffusivity),
        float(closed),
        1e-12,
        rel=True,
        unit="m2/s",
        places=9,
    )


# ---------------------------------------------------------------------------
# Sea water (Ainslie 2010)
# ---------------------------------------------------------------------------
_SEA = "Sea water (Ainslie 2010)"

#: Eq. (4.11) at the surface: 98 066,5 x 1,04, one atmosphere and not zero.
_SURFACE_ABSOLUTE_PA = 101989.16


@register(
    _SEA,
    "Ainslie (2010) Eq. (4.6), printed folio 127",
    "Density of the standard ocean: 10 C, salinity 35, at the surface",
)
def _chk_ainslie_standard_ocean() -> Outcome:
    rho = ph.fluids.sea_water_density(
        temperature_c=10.0,
        salinity_psu=35.0,
        absolute_pressure_pa=_SURFACE_ABSOLUTE_PA,
    )
    # Folio 28 prints 1027 kg/m3 for these conditions, to four figures, so half
    # a unit in the last place is what reproducing it means.
    return numeric(1027.0, float(rho), 0.5, unit="kg/m3", places=4)


@register(
    _SEA,
    "Ainslie (2010) Eq. (4.11), printed folio 128",
    "Absolute static pressure at the surface is one atmosphere, not zero",
)
def _chk_ainslie_surface_pressure() -> Outcome:
    return numeric(
        _SURFACE_ABSOLUTE_PA,
        float(ph.fluids.depth_to_absolute_pressure_pa(depth_m=0.0)),
        1e-6,
        unit="Pa",
        places=2,
    )


@register(
    _SEA,
    "Ainslie (2010) Eq. (4.6) vs printed folio 177",
    "The pressure term the book's own folio 177 drops: 4,3e-7 per pascal times "
    "one atmosphere",
)
def _chk_ainslie_folio_177_gap() -> Outcome:
    with_pressure = ph.fluids.sea_water_density(
        temperature_c=23.0, salinity_psu=35.0, absolute_pressure_pa=_SURFACE_ABSOLUTE_PA
    )
    without_pressure = ph.fluids.sea_water_density(
        temperature_c=23.0, salinity_psu=35.0, absolute_pressure_pa=1.0
    )
    # 4,3e-7 per pascal over the atmosphere Eq. (4.11) puts at the surface. The
    # expected value is that product rather than a rounded quotation of it, so
    # the row measures the implementation and not the rounding.
    expected = 4.3e-7 * (_SURFACE_ABSOLUTE_PA - 1.0)
    return numeric(
        expected,
        float(with_pressure) - float(without_pressure),
        1e-12,
        unit="kg/m3",
        places=7,
    )
