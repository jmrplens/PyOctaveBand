#  Copyright (c) 2026. Jose Manuel Requena Plens
"""The fluid state type: what it closes by identity, and what it refuses to guess."""

from __future__ import annotations

import warnings

import pytest

from phonometry import PhonometryWarning
from phonometry.fluids import (
    Fluid,
    FluidAssumptionWarning,
    FluidPropertyUnavailable,
    FluidWarning,
    air,
)


@pytest.fixture
def reference_air() -> Fluid:
    """Annex F's first printed condition set, with nothing assumed."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        return air(
            temperature_c=23.0,
            static_pressure_pa=101_325.0,
            relative_humidity_percent=50.0,
        )


def test_characteristic_impedance_is_rho_c(reference_air: Fluid) -> None:
    assert reference_air.characteristic_impedance == pytest.approx(
        reference_air.density * reference_air.speed_of_sound, rel=1e-15
    )


def test_prandtl_number_closes_from_the_three_it_is_made_of(
    reference_air: Fluid,
) -> None:
    """Pr = eta / (rho alpha_t), against the same ratio of the printed figures.

    Annex F tabulates the three, so the Prandtl number is theirs and needs no
    constant of its own here. It comes to 0,728 02 at the reference state.

    Worth pinning because a published porous model may carry its own Prandtl
    number as a fitted constant. That constant belongs to the model, not to the
    air: the Johnson-Champoux-Allard 0,71 differs from this by two and a half
    per cent, and it stays frozen where it is published.
    """
    printed = 1.826566e-5 / (1.1860848 * 2.115317e-5)
    assert reference_air.prandtl_number == pytest.approx(printed, rel=5e-7)
    assert reference_air.prandtl_number == pytest.approx(0.72802, abs=5e-6)


def test_kinematic_viscosity_closes_from_the_two_it_is_made_of(
    reference_air: Fluid,
) -> None:
    assert reference_air.kinematic_viscosity == pytest.approx(
        reference_air.viscosity / reference_air.density, rel=1e-15
    )


def test_a_quantity_the_model_did_not_fix_names_the_model(reference_air: Fluid) -> None:
    """Nothing is invented for a quantity no model determined.

    The message has to carry the model, because the same accessor is available
    on every fluid and the answer to "why not" is which model built this one.
    """
    lean = Fluid(
        temperature_c=reference_air.temperature_c,
        static_pressure_pa=reference_air.static_pressure_pa,
        composition={},
        model="a model that prints only a speed of sound",
        validity="",
        properties={"speed_of_sound": 1500.0},
    )
    with pytest.raises(FluidPropertyUnavailable, match="only a speed of sound"):
        _ = lean.density
    assert lean.speed_of_sound == pytest.approx(1500.0)


def test_a_lean_fluid_cannot_close_an_identity_it_lacks_a_term_for() -> None:
    """The derived accessors fail the same way, naming the missing term."""
    lean = Fluid(
        temperature_c=20.0,
        static_pressure_pa=101_325.0,
        composition={},
        model="speed only",
        validity="",
        properties={"speed_of_sound": 343.0},
    )
    with pytest.raises(FluidPropertyUnavailable, match="'density'"):
        _ = lean.characteristic_impedance


def test_the_state_carries_what_was_assumed(reference_air: Fluid) -> None:
    """A result can always say what air it was computed for."""
    assert reference_air.composition["relative_humidity_percent"] == pytest.approx(50.0)
    assert "61094-2" in reference_air.model
    assert "15 degC to 27 degC" in reference_air.validity


def test_the_fluid_is_frozen(reference_air: Fluid) -> None:
    with pytest.raises(AttributeError):
        reference_air.temperature_c = 0.0  # type: ignore[misc]


@pytest.mark.parametrize("warning", [FluidWarning, FluidAssumptionWarning])
def test_every_diagnostic_is_filterable_as_one(warning: type[Warning]) -> None:
    """One filterwarnings rule reaches every diagnostic the library raises."""
    assert issubclass(warning, PhonometryWarning)
