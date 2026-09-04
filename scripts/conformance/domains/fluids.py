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
