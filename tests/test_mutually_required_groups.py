#  Copyright (c) 2026. Jose Manuel Requena Plens
"""
Every parameter group that has to travel together is refused when it does not.

Twenty public functions take a group of arguments that the signature used to
present as independently optional: some are "exactly one of", some are "all or
none", and three are required only for a particular value of another argument.
Each of them now states the group in its signature through ``typing.overload``,
so a type checker rejects the invalid call before it runs, and each still
raises at run time for callers who do not type-check.

This module pins the run-time half. The static half is exercised by ``mypy``
over ``src`` and cannot be asserted from inside a test: a file that made mypy
fail would fail the gate rather than pass this suite.

One of the twenty did not raise before this was written: giving
``floating_floor_improvement_spectrum`` one half of its pair returned
``delta_lw=None``, silently dropping the very quantity the pair is for.
"""

from __future__ import annotations

import numpy as np
import pytest

from phonometry.building.measurement.insulation import airborne_insulation
from phonometry.building.measurement.intensity_insulation import adaptation_term_kc
from phonometry.building.measurement.survey_insulation import (
    survey_airborne_insulation,
)
from phonometry.building.prediction.ceiling_plenum import (
    plenum_flanking_reduction_index,
)
from phonometry.building.prediction.linings import lining_resonance_frequency
from phonometry.building.prediction.panel_transmission import (
    plateau_transmission_loss,
)
from phonometry.building.prediction.resilient_layers import (
    floating_floor_improvement_spectrum,
)
from phonometry.building.regulation.spain import db_hr_global_index
from phonometry.emission.sound_power_anechoic import sound_power_anechoic
from phonometry.environment.assessment.measurement import gaussian_residual_level
from phonometry.environment.propagation.ground_barriers import ground_effect
from phonometry.materials.diffusers.design import predict_diffuser_polar_response
from phonometry.metrology.calibration import sensitivity
from phonometry.noise_control.room_to_room import SourceRoom
from phonometry.speech.sti import sti_from_impulse_response, stipa
from phonometry.vibration.human.multiple_shock import multiple_shock_assessment

BANDS = np.array(
    [50.0, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000]
)
OCTAVES = np.array([125.0, 250.0, 500.0, 1000.0])


# ---------------------------------------------------------------------------
# Exactly one of
# ---------------------------------------------------------------------------


def test_lining_resonance_takes_one_formula_not_both() -> None:
    """D.1 and D.2 are two constructions, not two ways to say one number."""
    for kwargs in ({}, {"dynamic_stiffness": 1e7, "cavity_depth": 0.05}):
        with pytest.raises(ValueError, match="exactly one"):
            lining_resonance_frequency(100.0, 10.0, **kwargs)  # type: ignore[call-overload]
    assert lining_resonance_frequency(100.0, 10.0, dynamic_stiffness=1e7) > 0.0
    assert lining_resonance_frequency(100.0, 10.0, cavity_depth=0.05) > 0.0


def test_ground_effect_takes_one_ground_description() -> None:
    both = {"flow_resistivity": 200e3, "impedance": np.ones(4, dtype=complex)}
    for kwargs in ({}, both):
        with pytest.raises(ValueError, match="exactly one"):
            ground_effect(  # type: ignore[call-overload]
                OCTAVES, 1.0, 1.5, 10.0, **kwargs
            )


def test_gaussian_residual_takes_one_percentile() -> None:
    for kwargs in ({}, {"l90": 45.0, "l95": 44.0}):
        with pytest.raises(ValueError, match="exactly one"):
            gaussian_residual_level(50.0, **kwargs)  # type: ignore[call-overload]


def test_diffuser_polar_takes_one_description() -> None:
    for kwargs in ({}, {"depths": BANDS, "reflection": BANDS}):
        with pytest.raises(ValueError, match="exactly one"):
            predict_diffuser_polar_response(0.1, 500.0, **kwargs)  # type: ignore[call-overload]


# ---------------------------------------------------------------------------
# All or none
# ---------------------------------------------------------------------------


def test_airborne_insulation_wants_area_and_volume_together() -> None:
    l1, l2, t2 = np.full(16, 80.0), np.full(16, 40.0), np.full(16, 0.5)
    with pytest.raises(ValueError, match="together"):
        airborne_insulation(l1, l2, t2, area=10.0)  # type: ignore[call-overload]
    with pytest.raises(ValueError, match="together"):
        airborne_insulation(l1, l2, t2, volume=50.0)  # type: ignore[call-overload]


def test_survey_insulation_area_needs_volume() -> None:
    l1, l2 = np.full(5, 80.0), np.full(5, 40.0)
    k = np.full(5, 5.0)
    with pytest.raises(ValueError, match="requires 'volume'"):
        survey_airborne_insulation(l1, l2, k, area=10.0)  # type: ignore[call-overload]


def test_adaptation_term_wants_both_or_neither() -> None:
    """Neither is not an error here: it selects the Formula (B.2) approximation."""
    with pytest.raises(ValueError, match="both"):
        adaptation_term_kc(OCTAVES, boundary_area=50.0)  # type: ignore[call-overload]
    assert adaptation_term_kc(OCTAVES).shape == OCTAVES.shape


def test_plenum_attenuations_travel_together() -> None:
    r = np.full(4, 30.0)
    with pytest.raises(ValueError, match="together"):
        plenum_flanking_reduction_index(  # type: ignore[call-overload]
            r, r, ceiling_length=3.0, plenum_height=1.0, attenuation_source=r
        )


def test_plateau_needs_a_complete_construction() -> None:
    with pytest.raises(ValueError, match="mass_per_area"):
        plateau_transmission_loss(OCTAVES, mass_per_area=20.0)  # type: ignore[call-overload]


def test_floating_floor_pair_is_not_dropped_in_silence() -> None:
    """Half the pair used to return ``delta_lw=None`` and say nothing.

    The pair exists only to produce the weighted improvement, so answering
    ``None`` to a caller who asked for it is the failure that matters here.
    """
    both = floating_floor_improvement_spectrum(
        BANDS, resonance_frequency=52.8, mass_per_area=120.0, dynamic_stiffness=1e7
    )
    assert both.delta_lw is not None
    neither = floating_floor_improvement_spectrum(BANDS, resonance_frequency=52.8)
    assert neither.delta_lw is None
    for half in ({"mass_per_area": 120.0}, {"dynamic_stiffness": 1e7}):
        with pytest.raises(ValueError, match="both or neither"):
            floating_floor_improvement_spectrum(  # type: ignore[call-overload]
                BANDS, resonance_frequency=52.8, **half
            )


def test_cremer_hammer_needs_its_limiting_frequency() -> None:
    with pytest.raises(ValueError, match="limiting_frequency"):
        floating_floor_improvement_spectrum(  # type: ignore[call-overload]
            BANDS, resonance_frequency=52.8, model="cremer_hammer"
        )


def test_background_levels_need_their_frequencies() -> None:
    levels = np.full((20, 8), 80.0)
    background = np.full(8, 60.0)
    with pytest.raises(ValueError, match="'frequencies' are required"):
        sound_power_anechoic(  # type: ignore[call-overload]
            levels, "hemisphere", radius=1.0, background_levels=background
        )


def test_sti_ambient_needs_the_speech_levels() -> None:
    fs = 48000
    x = np.random.default_rng(0).standard_normal(fs * 2)
    ambient = np.full(7, 30.0)
    with pytest.raises(ValueError, match="'ambient' requires"):
        stipa(x, fs, ambient=ambient)  # type: ignore[call-overload]
    ir = np.zeros(4096)
    ir[0] = 1.0
    with pytest.raises(ValueError, match="'ambient' requires"):
        sti_from_impulse_response(ir, fs, ambient=ambient)  # type: ignore[call-overload]


def test_sti_snr_and_ambient_are_two_ways_of_saying_one_thing() -> None:
    """A group the first sweep missed, found in review of this change.

    The run time refused the pair already; nothing in the signature said so,
    which is the whole defect this change is about.
    """
    ir = np.zeros(4096)
    ir[0] = 1.0
    level, ambient = np.full(7, 60.0), np.full(7, 30.0)
    with pytest.raises(ValueError, match="not both"):
        sti_from_impulse_response(  # type: ignore[call-overload]
            ir, 48000, 15.0, level, ambient
        )


def test_shock_exposure_and_measurement_times_travel_together() -> None:
    a = np.random.default_rng(0).standard_normal(4096)
    with pytest.raises(ValueError, match="both"):
        multiple_shock_assessment(  # type: ignore[call-overload]
            a, 1000.0, start_age=20.0, years=10, days_per_year=200.0,
            exposure_time=1.0,
        )


def test_source_room_refuses_itself_at_construction() -> None:
    """The one group that lives between two fields of a bundle, not two arguments.

    No overload of ``room_to_room_transmission`` can constrain the inside of
    its own argument, so the bundle checks itself, as
    ``OperatingModeDeclaration`` already does. Checking it here rather than at
    the call means the complaint arrives at the line that built the object.
    """
    for kwargs in ({}, {"level": 80.0, "power_level": 90.0}):
        with pytest.raises(ValueError, match="exactly one"):
            SourceRoom(**kwargs)
    with pytest.raises(ValueError, match="'room_constant' is required"):
        SourceRoom(power_level=90.0)
    assert SourceRoom(level=80.0).level == 80.0
    assert SourceRoom(power_level=90.0, room_constant=25.0).room_constant == 25.0


# ---------------------------------------------------------------------------
# The two the type system cannot state, checked at run time only
# ---------------------------------------------------------------------------


def test_db_hr_long_spectrum_needs_its_frequencies() -> None:
    """The condition is the *length* of the input, which no signature can say."""
    too_long = np.full(30, 40.0)
    with pytest.raises(ValueError, match="eighteen DB-HR"):
        db_hr_global_index(too_long, "pink")


def test_narrowband_sensitivity_needs_its_rate() -> None:
    tone = np.sin(2 * np.pi * 1000.0 * np.arange(48000) / 48000)
    with pytest.raises(ValueError, match="requires 'fs'"):
        sensitivity(tone, target_spl=94.0, narrowband=True)  # type: ignore[call-overload]
