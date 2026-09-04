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
    characteristic_impedance,
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


@pytest.mark.parametrize("mapping", ["properties", "composition"])
def test_the_mappings_are_frozen_too(reference_air: Fluid, mapping: str) -> None:
    """``frozen=True`` stops the rebinding, not the writing through it.

    The shared states are what make this matter: a module-level air that every
    visco-thermal model defaults to would have moved for the whole process on
    one ``air.properties["density"] = 999`` anywhere in it.
    """
    with pytest.raises(TypeError, match="does not support item assignment"):
        getattr(reference_air, mapping)["density"] = 999.0


def test_the_mapping_passed_in_cannot_reach_back() -> None:
    """The mappings are copied before they are wrapped.

    Wrapping without copying would leave the caller's own dict as a live handle
    on a fluid that reports itself frozen.
    """
    mine = {"density": 1.0, "speed_of_sound": 2.0}
    fluid = Fluid(
        temperature_c=0.0,
        static_pressure_pa=1.0,
        composition={},
        model="two numbers",
        validity="",
        properties=mine,
    )
    mine["density"] = 999.0
    assert fluid.density == pytest.approx(1.0, abs=0.0)


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
@pytest.mark.parametrize("position", ["density", "speed_of_sound"])
def test_characteristic_impedance_refuses_what_is_not_a_positive_number(
    position: str, bad: float
) -> None:
    """Every argument that is not a positive finite number raises, NaN included.

    A bare ``<= 0.0`` would let NaN through, because every comparison with NaN
    is false, and the product would come back as NaN rather than raising. The
    message has to name the argument that was wrong, not both of them.
    """
    kwargs = {"density": 1.2, "speed_of_sound": 343.0} | {position: bad}
    with pytest.raises(ValueError, match=f"'{position}' must be positive"):
        characteristic_impedance(**kwargs)


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


def test_a_printed_prandtl_number_wins_over_the_identity() -> None:
    """A published fit keeps the constant it was fitted with.

    Johnson-Champoux-Allard carries 0,71. Air at that state has 0,728. Closing
    the identity from the better air would not correct the model, it would
    change it, and it moves what the model computes by 1,5 parts in a thousand.
    So a carried value wins, and a model that prints none has it closed from
    the three quantities that it did print.
    """
    published = Fluid(
        temperature_c=20.0,
        static_pressure_pa=101_325.0,
        composition={},
        model="a fit that printed its own Prandtl number",
        validity="",
        properties={"viscosity": 1.84e-5, "density": 1.205, "prandtl_number": 0.71},
    )
    assert published.prandtl_number == pytest.approx(0.71, abs=1e-15)


def test_without_a_printed_one_it_closes_from_the_three(reference_air: Fluid) -> None:
    assert "prandtl_number" not in reference_air.properties
    assert reference_air.prandtl_number == pytest.approx(
        reference_air.viscosity
        / (reference_air.density * reference_air.thermal_diffusivity),
        rel=1e-15,
    )


@pytest.mark.parametrize(
    "fraction", [-1e-12, -0.5, float("nan"), float("inf"), float("-inf")]
)
def test_a_composition_fraction_that_cannot_exist_is_refused(fraction: float) -> None:
    """A fraction may be nought; it may not be negative or absent.

    Dry air really does carry no water vapour, so nought passes where a
    property would be refused. Everything below it, and every non-number,
    describes a mixture that does not exist, and the refusal happens on
    construction so that no model downstream has to ask.
    """
    with pytest.raises(
        ValueError, match=r"'composition\['relative_humidity_percent'\]'"
    ):
        Fluid(
            temperature_c=20.0,
            static_pressure_pa=101_325.0,
            composition={"relative_humidity_percent": fraction},
            model="an air with an impossible amount of water in it",
            validity="",
            properties={"speed_of_sound": 343.0, "density": 1.2},
        )


def test_a_composition_fraction_of_nought_is_accepted() -> None:
    """Dry air is a real air, and its water vapour fraction is nought."""
    dry = Fluid(
        temperature_c=20.0,
        static_pressure_pa=101_325.0,
        composition={"relative_humidity_percent": 0.0},
        model="dry air",
        validity="",
        properties={"speed_of_sound": 343.0, "density": 1.2},
    )
    assert dry.composition["relative_humidity_percent"] == 0.0
