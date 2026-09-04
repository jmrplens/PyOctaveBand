#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Sea water: the density this library had nowhere before, and its two pressures.

The oracle is Ainslie, *Principles of Sonar Performance Modelling* (Springer
2010): Equation (4.6) on printed folio 127 for the density, Equation (4.11) on
folio 128 for the absolute pressure, and the 1027 kg/m3 the book gives for the
standard ocean on folio 28.
"""

from __future__ import annotations

import warnings

import pytest

from phonometry.fluids import (
    Fluid,
    FluidPropertyUnavailable,
    depth_to_absolute_pressure_pa,
    depth_to_gauge_pressure_mpa,
    sea_water,
    sea_water_density,
)

#: Eq. (4.11) at the surface: 98 066,5 x 1,04, one atmosphere rather than zero.
_SURFACE_ABSOLUTE_PA = 101989.16


def test_the_two_pressures_answer_different_questions() -> None:
    """One is gauge in megapascals, the other absolute in pascals.

    They differ by a factor of a million and an offset of an atmosphere, which
    is why each name carries its unit and its datum. At the surface the gauge
    one is zero by definition and the absolute one is an atmosphere.
    """
    assert depth_to_gauge_pressure_mpa(depth_m=0.0) == pytest.approx(0.0, abs=1e-12)
    assert depth_to_absolute_pressure_pa(depth_m=0.0) == pytest.approx(
        _SURFACE_ABSOLUTE_PA, abs=1e-6
    )


def test_density_reproduces_the_standard_ocean() -> None:
    """Folio 28: the density of sea water under the book's representative
    conditions, 10 degC and salinity 35 at the surface, is 1027 kg/m3.
    """
    rho = sea_water_density(
        temperature_c=10.0,
        salinity_psu=35.0,
        absolute_pressure_pa=depth_to_absolute_pressure_pa(depth_m=0.0),
    )
    assert round(rho) == 1027


def test_the_pressure_term_is_small_but_present() -> None:
    """4,3e-7 per pascal: an atmosphere is worth 0,044 kg/m3.

    Small enough to be invisible against any tolerance in this library, and the
    reason the book's own folio 177 disagrees with its Equation (4.6) by
    exactly that much.
    """
    at_surface = sea_water_density(
        temperature_c=23.0, salinity_psu=35.0, absolute_pressure_pa=_SURFACE_ABSOLUTE_PA
    )
    without_the_term = sea_water_density(
        temperature_c=23.0, salinity_psu=35.0, absolute_pressure_pa=1.0
    )
    assert at_surface - without_the_term == pytest.approx(0.0439, abs=1e-4)


def test_density_falls_with_temperature_and_rises_with_salt_and_depth() -> None:
    """The three signs Equation (4.6) prints."""
    base = {"salinity_psu": 35.0, "absolute_pressure_pa": _SURFACE_ABSOLUTE_PA}
    assert sea_water_density(temperature_c=25.0, **base) < sea_water_density(
        temperature_c=5.0, **base
    )
    assert sea_water_density(
        temperature_c=10.0, salinity_psu=38.0, absolute_pressure_pa=_SURFACE_ABSOLUTE_PA
    ) > sea_water_density(temperature_c=10.0, **base)
    deep = depth_to_absolute_pressure_pa(depth_m=4000.0)
    assert sea_water_density(
        temperature_c=10.0, salinity_psu=35.0, absolute_pressure_pa=deep
    ) > sea_water_density(temperature_c=10.0, **base)


def test_sea_water_gives_a_fluid_that_says_where_it_came_from() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        water = sea_water(temperature_c=10.0)
    assert isinstance(water, Fluid)
    assert round(water.density) == 1027
    assert water.speed_of_sound == pytest.approx(1489.832, abs=1e-3)
    assert water.characteristic_impedance == pytest.approx(
        water.density * water.speed_of_sound, rel=1e-15
    )
    assert "Ainslie" in water.model
    assert water.composition["salinity_psu"] == pytest.approx(35.0)


@pytest.mark.parametrize(
    "quantity",
    ["heat_capacity_ratio", "viscosity", "thermal_diffusivity", "thermal_conductivity"],
)
def test_sea_water_refuses_what_no_source_here_prints(quantity: str) -> None:
    """Air has these and water does not, and the type says so rather than
    returning a number nobody printed. This is the case the accessor exists
    for.
    """
    water = sea_water(temperature_c=10.0)
    with pytest.raises(FluidPropertyUnavailable, match="Ainslie"):
        getattr(water, quantity)


@pytest.mark.parametrize("model", ["unesco", "del_grosso", "mackenzie", "medwin"])
def test_every_sound_speed_model_still_reaches_the_fluid(model: str) -> None:
    """The four fits are competing answers to one question, so the constructor
    takes a model where air's does not.
    """
    water = sea_water(temperature_c=10.0, sound_speed_model=model)
    assert 1400.0 < water.speed_of_sound < 1600.0


def test_impossible_states_are_refused() -> None:
    with pytest.raises(ValueError, match="'temperature_c' must be"):
        sea_water_density(
            temperature_c=-300.0, salinity_psu=35.0, absolute_pressure_pa=101325.0
        )
    with pytest.raises(ValueError, match="'salinity_psu' must be non-negative"):
        sea_water_density(
            temperature_c=10.0, salinity_psu=-1.0, absolute_pressure_pa=101325.0
        )
    with pytest.raises(ValueError, match="'depth_m' must be non-negative"):
        depth_to_absolute_pressure_pa(depth_m=-1.0)
