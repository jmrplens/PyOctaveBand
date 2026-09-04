#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Humid air against the two condition sets IEC 61094-2:2009 Annex F prints.

Table F.1 (printed folio 40) is the whole oracle: five quantities at two states.
The tolerance is not chosen, it is read off the print. A value given to seven
significant figures is stated to within half of its last figure, so that is what
reproducing it means, and asserting anything tighter would fail on the standard's
own rounding rather than on the library.
"""

from __future__ import annotations

import math
import warnings
from decimal import Decimal

import pytest

from phonometry.fluids import (
    DEFAULT_RELATIVE_HUMIDITY_PERCENT,
    DEFAULT_STATIC_PRESSURE_PA,
    Fluid,
    FluidAssumptionWarning,
    FluidWarning,
    air,
)

#: Table F.1, printed folio 40 (PDF page 42) of BS EN 61094-2:2009. The values
#: are kept as the strings the annex prints so the tolerance can be derived from
#: the precision rather than guessed.
_TABLE_F1: tuple[tuple[str, tuple[float, float, float], dict[str, str]], ...] = (
    (
        "A",
        (23.0, 101325.0, 50.0),
        {
            "density": "1.1860848",
            "speed_of_sound": "345.86652",
            "heat_capacity_ratio": "1.4007573",
            "viscosity": "1.826566e-5",
            "thermal_diffusivity": "2.115317e-5",
        },
    ),
    (
        "B",
        (20.0, 80000.0, 65.0),
        {
            "density": "0.9441589",
            "speed_of_sound": "344.38267",
            "heat_capacity_ratio": "1.4000266",
            "viscosity": "1.811295e-5",
            "thermal_diffusivity": "2.627024e-5",
        },
    ),
)


def _half_of_the_last_printed_figure(printed: str) -> float:
    """Half a unit in the last place the annex printed, as an absolute bound."""
    exponent = Decimal(printed).as_tuple().exponent
    return float(Decimal(5) * Decimal(10) ** (int(exponent) - 1))


def _quiet_air(temperature_c: float, pressure_pa: float, humidity: float) -> Fluid:
    """Build air at a stated condition, with nothing assumed and no warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        return air(
            temperature_c=temperature_c,
            static_pressure_pa=pressure_pa,
            relative_humidity_percent=humidity,
        )


@pytest.mark.parametrize(("label", "conditions", "printed"), _TABLE_F1)
def test_table_f1_reproduces_every_printed_figure(
    label: str, conditions: tuple[float, float, float], printed: dict[str, str]
) -> None:
    """All five printed quantities, at both printed states, to the last figure."""
    fluid = _quiet_air(*conditions)
    for quantity, value in printed.items():
        assert getattr(fluid, quantity) == pytest.approx(
            float(value), abs=_half_of_the_last_printed_figure(value)
        ), f"set {label}: {quantity}"


def test_the_two_unprinted_constituents_close_the_printed_diffusivity() -> None:
    """Clause F.6 prints k_a and C_P as expressions, and alpha_t as their ratio.

    Table F.1 tabulates only the ratio, so the two constituents have no printed
    check value of their own. What can be verified is that they close the
    identity Formula (F.5) states, which is the guard against transcribing
    either expression wrongly in a way the ratio would hide.
    """
    fluid = _quiet_air(23.0, 101325.0, 50.0)
    closed = fluid.thermal_conductivity / (fluid.density * fluid.specific_heat_capacity)
    assert closed == pytest.approx(fluid.thermal_diffusivity, rel=1e-15)


def test_the_specific_heat_is_a_possible_heat_capacity_of_air() -> None:
    """A sanity bound the ratio alone cannot give.

    ISO 9053-2:2020 Annex A.3 prints a C_P of 938,7 J/(kg K) for this same air
    and credits it to this document; it is 27,19 J/(mol K), below the
    rigid-rotor diatomic floor of (7/2)R, so no diatomic gas has it. This pins
    that the library computes the annex's own value and not that one.
    """
    molar_mass_air = 0.0289647  # kg/mol
    diatomic_floor = 3.5 * 8.314462618  # J/(mol K)
    molar = _quiet_air(23.0, 101325.0, 50.0).specific_heat_capacity * molar_mass_air
    assert molar > diatomic_floor


# --- what is assumed, and what is refused -----------------------------------


def test_omitting_a_condition_says_which_and_what_it_is_worth() -> None:
    """One warning, naming both assumptions, not one warning each."""
    with pytest.warns(FluidAssumptionWarning) as caught:
        air(temperature_c=20.0)
    assert len(caught) == 1
    message = str(caught[0].message)
    assert f"{DEFAULT_STATIC_PRESSURE_PA:.0f} Pa" in message
    assert f"{DEFAULT_RELATIVE_HUMIDITY_PERCENT:.0f} % relative humidity" in message


def test_omitting_only_the_humidity_names_only_the_humidity() -> None:
    with pytest.warns(FluidAssumptionWarning, match="relative humidity") as caught:
        air(temperature_c=20.0, static_pressure_pa=90_000.0)
    assert "Pa." not in str(caught[0].message).split("assumed")[1].split(".")[0]


def test_supplying_both_conditions_is_silent() -> None:
    """The property a caller who measured their air should get: nothing."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        air(
            temperature_c=20.0,
            static_pressure_pa=101_000.0,
            relative_humidity_percent=45.0,
        )


def test_the_carbon_dioxide_default_does_not_warn() -> None:
    """Clause F.2 names 0,000 4 for laboratory conditions; that is not a guess."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        air(
            temperature_c=20.0,
            static_pressure_pa=101_000.0,
            relative_humidity_percent=45.0,
        )


@pytest.mark.parametrize(
    ("conditions", "expected"),
    [
        ((5.0, 101325.0, 50.0), "temperature"),
        ((20.0, 40_000.0, 50.0), "static pressure"),
        ((20.0, 101325.0, 95.0), "relative humidity"),
    ],
)
def test_outside_the_printed_domain_warns_and_still_answers(
    conditions: tuple[float, float, float], expected: str
) -> None:
    """Annex F states where its equations were validated, not what air can be.

    So a state outside that box is an extrapolation the caller is told about,
    never a refusal: the printed domain bounds the fit, not the physics.
    """
    with pytest.warns(FluidWarning, match=expected):
        fluid = air(
            temperature_c=conditions[0],
            static_pressure_pa=conditions[1],
            relative_humidity_percent=conditions[2],
        )
    assert math.isfinite(fluid.density)
    assert math.isfinite(fluid.speed_of_sound)


def test_inside_the_printed_domain_is_silent() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        air(
            temperature_c=23.0,
            static_pressure_pa=101_325.0,
            relative_humidity_percent=50.0,
        )


def test_only_the_impossible_is_refused() -> None:
    """The bound is what cannot exist, not what is unusual.

    Air at 60 degC in a duct and air at -60 degC in a cold chamber are both
    outside Annex F's stated domain and both real; they warn. Absolute zero is
    not a state, and neither is a negative pressure.

    200 degC used to be in this list at 50 % relative humidity, which is not a
    state either: saturation there is 1 592 kPa, so at one atmosphere the most
    the air can hold is 6,4 %, and 50 % asks for more water vapour than total
    pressure. It stays, at a humidity it can actually have.
    """
    for temperature_c, relative_humidity_percent in (
        (-60.0, 50.0),
        (60.0, 50.0),
        (200.0, 6.0),
    ):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FluidWarning)
            assert math.isfinite(
                air(
                    temperature_c=temperature_c,
                    static_pressure_pa=101_325.0,
                    relative_humidity_percent=relative_humidity_percent,
                ).density
            )
    with pytest.raises(ValueError, match="'temperature_c' must be"):
        air(temperature_c=-273.15, static_pressure_pa=101_325.0)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"static_pressure_pa": 0.0}, "'static_pressure_pa' must be"),
        ({"static_pressure_pa": -1.0}, "'static_pressure_pa' must be"),
        ({"relative_humidity_percent": -1.0}, "'relative_humidity_percent' must be"),
        ({"relative_humidity_percent": 101.0}, "'relative_humidity_percent' must be"),
        ({"co2_mole_fraction": 1.5}, "'co2_mole_fraction' must be"),
    ],
)
def test_impossible_conditions_are_refused(
    kwargs: dict[str, float], match: str
) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError, match=match):
            air(temperature_c=20.0, **kwargs)


def test_conditions_that_cannot_hold_together_are_refused() -> None:
    """Each argument can be a state and the combination still not be one.

    At 20 degC and 1 kPa, 50 % relative humidity asks for a water vapour mole
    fraction of 1,17, and a mole fraction cannot reach 1: there is more water
    vapour than total pressure. Every argument passes its own guard, so nothing
    but a guard on the fraction catches it, and without one the CIPM equations
    carry on and return a density and a speed of sound that look like
    measurements.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError, match="cannot hold together"):
            air(
                temperature_c=20.0,
                static_pressure_pa=1_000.0,
                relative_humidity_percent=50.0,
            )


@pytest.mark.parametrize(
    ("temperature_c", "static_pressure_pa", "relative_humidity_percent"),
    [
        (27.0, 60_000.0, 90.0),  # the corner of the printed domain
        (20.0, 101_325.0, 100.0),  # saturated at one atmosphere
        (60.0, 101_325.0, 100.0),  # saturated and hot: x = 0,198
        (50.0, 60_000.0, 100.0),  # saturated and thin: x = 0,207
    ],
)
def test_air_that_can_exist_is_not_refused_by_that_guard(
    temperature_c: float, static_pressure_pa: float, relative_humidity_percent: float
) -> None:
    """The guard has to let saturated, hot and thin air through.

    Saturation carries a fifth of the mole fraction at 60 degC, so a guard set
    anywhere below 1 would start refusing air that exists.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fluid = air(
            temperature_c=temperature_c,
            static_pressure_pa=static_pressure_pa,
            relative_humidity_percent=relative_humidity_percent,
        )
    assert 0.0 < fluid.composition["water_vapour_mole_fraction"] < 1.0
    assert fluid.density > 0.0


def test_the_refusal_comes_before_the_assumption_warning() -> None:
    """A caller who promotes the warning must still learn which argument failed.

    The assumption warning used to be raised before the optional values were
    validated, so `simplefilter("error", FluidAssumptionWarning)` turned it into
    the exception a bad humidity raised, and the ValueError naming the argument
    never arrived.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error", FluidAssumptionWarning)
        with pytest.raises(ValueError, match="'relative_humidity_percent' must be"):
            air(temperature_c=20.0, relative_humidity_percent=101.0)


def test_air_refuses_a_positional_argument() -> None:
    """Every condition is named at the call site, so no unit can be dropped."""
    with pytest.raises(TypeError, match="positional"):
        air(20.0)  # type: ignore[misc]
