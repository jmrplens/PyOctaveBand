#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Tests for the VDI 2081 fan model, against the guideline's own worked example.

Oracle: VDI 2081 Part 2:2005-05, *Air-conditioning: noise generation and noise
reduction. Examples*, Table 1 on printed folio 12, element 1 (the supply air
fan). Every expected value below is a cell of that printed table.

The method it exercises is VDI 2081 Part 1:2001-07, Section 4.3, printed folios
18 to 25: Equation (13) for the level, Equation (15) with Figures 10 to 12 for
the shape, Section 4.3.3 for the three assembly types and Figures 13 and 14 for
the allowance away from the best duty point.

Both prints are superseded, by VDI 2081 Part 1:2022-04 and Part 2:2022-10, and
neither successor is held. The pair in hand is self-consistent: Part 2:2005 was
written against Part 1:2001 and every cross-reference in its tables resolves
there, which is what makes the example usable as an oracle for this edition.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from phonometry.noise_control import hvac

#: Table 1, the supply air fan: 16 000 m3/h against a 600 Pa total pressure
#: rise, a double-inlet radial fan with rearwards curved blades (assembly RR)
#: turning at 1250 min^-1.
FAN_VOLUME_FLOW_M3_S = 16000.0 / 3600.0
FAN_TOTAL_PRESSURE_PA = 600.0
FAN_SPEED_RPM = 1250.0

#: Table 1, row "Schallleistungspegel L_W4": the overall level, dB.
PRINTED_OVERALL_DB = 96.0
#: Table 1, row "Strouhalzahl": the Strouhal number of each octave.
PRINTED_STROUHAL = (0.963, 1.91, 3.82, 7.639, 15.28, 30.56, 61.12, 122.2)
#: Table 1, row "Relatives Frequenzspektrum": the relative spectrum, dB.
PRINTED_RELATIVE_SPECTRUM_DB = (-5.7, -7.3, -9.8, -13.2, -17.5, -22.8, -28.9, -35.9)
#: Table 1, row "Ventilatorspektrum": the fan spectrum, dB re 1e-12 W.
PRINTED_FAN_SPECTRUM_DB = (90.4, 88.8, 86.3, 82.9, 78.6, 73.4, 67.2, 60.2)
#: Table 1, the two summed columns of that row: L_W and L_WA, dB.
PRINTED_SUM_DB = 94.1
PRINTED_SUM_A_DB = 84.5
#: Table 1, row "A-Korrektur": the A-weighting of each octave, dB.
PRINTED_A_WEIGHTING_DB = (-26.2, -16.1, -8.6, -3.2, 0.0, 1.2, 1.0, -1.1)

#: The table prints one decimal, so a cell is met when the computed value
#: rounds onto it.
PRINTED_TOLERANCE_DB = 0.05


def _supply_fan() -> hvac.HvacSpectrumResult:
    """Element 1 of Table 1, with nothing but the printed service data."""
    return hvac.fan_sound_power(
        FAN_VOLUME_FLOW_M3_S,
        model="vdi2081",
        fan_total_pressure_pa=FAN_TOTAL_PRESSURE_PA,
        assembly="rr",
        fan_speed_rpm=FAN_SPEED_RPM,
    )


def test_the_supply_fan_reproduces_every_printed_octave() -> None:
    """Table 1, element 1, band by band and then both of its sums."""
    res = _supply_fan()
    assert res.quantity == "sound_power_level"
    assert res.values == pytest.approx(
        PRINTED_FAN_SPECTRUM_DB, abs=PRINTED_TOLERANCE_DB
    )

    total = 10.0 * math.log10(float(np.sum(10.0 ** (res.values / 10.0))))
    weighted = 10.0 * math.log10(
        float(np.sum(10.0 ** ((res.values + np.array(PRINTED_A_WEIGHTING_DB)) / 10.0)))
    )
    assert total == pytest.approx(PRINTED_SUM_DB, abs=PRINTED_TOLERANCE_DB)
    assert weighted == pytest.approx(PRINTED_SUM_A_DB, abs=PRINTED_TOLERANCE_DB)


def test_equation_13_sets_the_overall_level() -> None:
    """``L_W4 = L_WSM + 10 lg V + 20 lg dp_t``, with L_WSM = 34 dB for RR.

    The factor on the pressure is 20 and not the 5 (gamma - 1) of Equation
    (11), because Section 4.3.2 takes the Mach number exponent as 5 for every
    ventilation fan. Checked here against the printed 96,0 dB.
    """
    overall = (
        34.0
        + 10.0 * math.log10(FAN_VOLUME_FLOW_M3_S)
        + 20.0 * math.log10(FAN_TOTAL_PRESSURE_PA)
    )
    assert overall == pytest.approx(PRINTED_OVERALL_DB, abs=PRINTED_TOLERANCE_DB)

    # And the module's own spectrum sums back to it: the shape of Equation (15)
    # is a relative spectrum, so subtracting it from each band returns the one
    # level, once the 0,1 dB of the best-duty allowance is taken off.
    res = _supply_fan()
    shape = np.array(PRINTED_RELATIVE_SPECTRUM_DB)
    recovered = res.values - shape
    assert recovered == pytest.approx(overall + 0.1, abs=PRINTED_TOLERANCE_DB)


def test_the_strouhal_number_carries_no_diameter() -> None:
    """``St = f 60 / (pi n)`` of Equation (15), the eight printed values.

    The impeller diameter cancels between the tip speed and the impeller
    circumference, so the printed 0,6 m of Table 1 is a nomogram result rather
    than an input to the spectrum.
    """
    strouhal = hvac.OCTAVE_BANDS * 60.0 / (math.pi * FAN_SPEED_RPM)
    assert strouhal == pytest.approx(PRINTED_STROUHAL, rel=5e-4)


def test_equation_15_reproduces_the_relative_spectrum() -> None:
    """``dL = -5 - 5 (lg St + c3)^2`` with c3 = 0,4 for assembly RR."""
    strouhal = np.array(PRINTED_STROUHAL)
    shape = -5.0 - 5.0 * (np.log10(strouhal) + 0.4) ** 2
    assert shape == pytest.approx(
        PRINTED_RELATIVE_SPECTRUM_DB, abs=PRINTED_TOLERANCE_DB
    )


def test_the_best_duty_point_is_worth_a_tenth_of_a_decibel() -> None:
    """Figure 13 at ``V/V_opt = 1``, which Table 1 prints as 0,1 dB in every band.

    The cubic does not pass exactly through nought at the best duty point; the
    guideline prints what it is worth there rather than rounding it away, and
    so does this.
    """
    at_best = 18.9 - 46.6 + 33.0 - 5.2
    assert at_best == pytest.approx(0.1, abs=1e-9)

    away = hvac.fan_sound_power(
        FAN_VOLUME_FLOW_M3_S,
        model="vdi2081",
        fan_total_pressure_pa=FAN_TOTAL_PRESSURE_PA,
        assembly="rr",
        fan_speed_rpm=FAN_SPEED_RPM,
        relative_flow=1.4,
    )
    # Throttling away from the optimum makes a fan louder, not quieter.
    assert np.all(away.values > _supply_fan().values)


def test_each_assembly_has_its_own_level_and_shape() -> None:
    """RR, T and AM differ in both L_WSM and the spectral parameter c3.

    Section 4.3.3 prints 34, 36 and 42 dB, so at one duty the three separate by
    exactly those differences before the shape is applied; and c3 moves the
    peak of the spectrum, so the three do not merely translate.
    """
    common = {
        "model": "vdi2081",
        "fan_total_pressure_pa": FAN_TOTAL_PRESSURE_PA,
        "fan_speed_rpm": FAN_SPEED_RPM,
    }
    rr = hvac.fan_sound_power(FAN_VOLUME_FLOW_M3_S, assembly="rr", **common)  # type: ignore[arg-type]
    t = hvac.fan_sound_power(FAN_VOLUME_FLOW_M3_S, assembly="t", **common)  # type: ignore[arg-type]
    am = hvac.fan_sound_power(FAN_VOLUME_FLOW_M3_S, assembly="am", **common)  # type: ignore[arg-type]

    # A pure level difference would keep the band-to-band differences equal.
    assert not np.allclose(t.values - rr.values, (t.values - rr.values)[0])
    # AM peaks higher up the Strouhal axis than RR, so it is the brighter fan.
    assert am.values[-1] - am.values[0] > rr.values[-1] - rr.values[0]


def test_the_blade_allowance_lands_in_the_octave_that_holds_it() -> None:
    """A downstream-diffuser axial fan takes 4 dB, and only in its own band.

    Section 4.3.4: the allowance is 0 dB for RR and T built to the state of the
    art and 4 dB for AM, added to the octave containing ``f = n z / 60``.
    """
    plain = hvac.fan_sound_power(
        FAN_VOLUME_FLOW_M3_S,
        model="vdi2081",
        fan_total_pressure_pa=FAN_TOTAL_PRESSURE_PA,
        assembly="am",
        fan_speed_rpm=FAN_SPEED_RPM,
    )
    bladed = hvac.fan_sound_power(
        FAN_VOLUME_FLOW_M3_S,
        model="vdi2081",
        fan_total_pressure_pa=FAN_TOTAL_PRESSURE_PA,
        assembly="am",
        fan_speed_rpm=FAN_SPEED_RPM,
        blade_count=12,
    )
    passing = FAN_SPEED_RPM * 12.0 / 60.0  # 250 Hz
    difference = bladed.values - plain.values
    holds = np.abs(np.log2(hvac.OCTAVE_BANDS / passing)) < 0.5
    assert difference[holds] == pytest.approx(4.0, abs=1e-9)
    assert difference[~holds] == pytest.approx(0.0, abs=1e-9)

    # Assembly RR is built to take none of it.
    rr_bladed = hvac.fan_sound_power(
        FAN_VOLUME_FLOW_M3_S,
        model="vdi2081",
        fan_total_pressure_pa=FAN_TOTAL_PRESSURE_PA,
        assembly="rr",
        fan_speed_rpm=FAN_SPEED_RPM,
        blade_count=12,
    )
    assert rr_bladed.values == pytest.approx(_supply_fan().values, abs=1e-9)


def test_a_measured_specific_level_replaces_the_representative_one() -> None:
    """The printed 34 dB is an assembly average; a fan of one's own overrides it.

    Section 4.3.3 says the averages can rise by up to 7 dB at the optimum duty
    point, so a manufacturer's own value is the better input where it exists.
    """
    louder = hvac.fan_sound_power(
        FAN_VOLUME_FLOW_M3_S,
        model="vdi2081",
        fan_total_pressure_pa=FAN_TOTAL_PRESSURE_PA,
        assembly="rr",
        fan_speed_rpm=FAN_SPEED_RPM,
        specific_sound_power_level=41.0,
    )
    assert louder.values == pytest.approx(_supply_fan().values + 7.0, abs=1e-9)


def test_the_two_models_do_not_share_a_pressure() -> None:
    """VDI 2081 takes the total pressure rise, the ASHRAE law the static one.

    They are different quantities, so each model asks for its own and neither
    silently reads the other's. Refusing here is what stops a static pressure
    from being scaled by the factor 20 that only the total-pressure form has.
    """
    with pytest.raises(ValueError, match=r"model='vdi2081' needs"):
        hvac.fan_sound_power(
            FAN_VOLUME_FLOW_M3_S,
            model="vdi2081",  # type: ignore[call-overload]
            fan_static_pressure_pa=FAN_TOTAL_PRESSURE_PA,
            assembly="rr",
            fan_speed_rpm=FAN_SPEED_RPM,
        )

    with pytest.raises(ValueError, match=r"model='ashrae' needs"):
        hvac.fan_sound_power(FAN_VOLUME_FLOW_M3_S)  # type: ignore[call-overload]


def test_the_default_model_is_still_the_ashrae_one() -> None:
    """Nothing that took the old signature changes its answer."""
    default = hvac.fan_sound_power(2.0, fan_static_pressure_pa=500.0)
    named = hvac.fan_sound_power(2.0, fan_static_pressure_pa=500.0, model="ashrae")
    assert default.values == pytest.approx(named.values, abs=1e-12)
    assert "VDI" not in default.label


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"assembly": "radial"}, r"'assembly' must be one of"),
        ({"fan_speed_rpm": 0.0}, r"'fan_speed_rpm' must be positive"),
        ({"relative_flow": -1.0}, r"'relative_flow' must be positive"),
        (
            {"specific_sound_power_level": 0.0},
            r"'specific_sound_power_level' must be positive",
        ),
        ({"blade_count": 0}, r"'blade_count' must be positive"),
    ],
)
def test_the_vdi_model_refuses_what_is_not_a_fan(
    kwargs: dict[str, object], match: str
) -> None:
    base: dict[str, object] = {
        "model": "vdi2081",
        "fan_total_pressure_pa": FAN_TOTAL_PRESSURE_PA,
        "assembly": "rr",
        "fan_speed_rpm": FAN_SPEED_RPM,
    }
    base.update(kwargs)
    with pytest.raises(ValueError, match=match):
        hvac.fan_sound_power(FAN_VOLUME_FLOW_M3_S, **base)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Duct elements: Table 1 again, this time its attenuation rows
# ---------------------------------------------------------------------------
#: Table 1, element 5: a 500 x 400 mm rectangular duct, 4 m long, and the
#: "delta L_W (dB/Okt)" row it prints for that run.
RECTANGULAR_RUN = (0.500, 0.400, 4.000)
PRINTED_RECTANGULAR_DB = (2.4, 2.4, 1.2, 0.6, 0.6, 0.6, 0.6, 0.6)
#: Table 1, elements 13 and 17: a 160 mm round duct, 1 m and 2 m long.
PRINTED_ROUND_1M_DB = (0.1, 0.1, 0.15, 0.15, 0.3, 0.3, 0.3, 0.3)
PRINTED_ROUND_2M_DB = (0.2, 0.2, 0.3, 0.3, 0.6, 0.6, 0.6, 0.6)
#: Table 1, element 14: a 160 mm round bend, and the limit frequency and
#: spectrum the table prints beside it. The example works in air at 340 m/s.
BEND_DIAMETER_M = 0.160
EXAMPLE_SPEED_OF_SOUND = 340.0
PRINTED_LIMIT_FREQUENCY_HZ = 1245.0
PRINTED_BEND_DB = (0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 3.0)
#: Table 1, elements 3, 7 and 16: the three junctions, as (fed branch area,
#: all branch areas, printed reduction in dB).
PRINTED_JUNCTIONS = (
    (0.30, (0.30, 0.36, 0.42), 5.6),
    (0.049, (0.049, 0.049, 0.049), 4.8),
    (0.020, (0.020, 0.020), 3.0),
)


def test_the_straight_runs_reproduce_table_5() -> None:
    """Elements 5, 13 and 17, which between them cover both duct shapes.

    The rectangular row is selected by the **largest** side: 500 mm puts the
    duct in the 0,40 to 0,80 m band, and reading it by the 400 mm side instead
    would give 0,45 dB/m at 250 Hz where the table gives 0,3.
    """
    width, height, length = RECTANGULAR_RUN
    rect = hvac.unlined_rectangular_duct_attenuation(
        hvac.OCTAVE_BANDS, width, height, length, model="vdi2081"
    )
    assert rect.values == pytest.approx(PRINTED_RECTANGULAR_DB, abs=1e-9)

    for run_length, printed in (
        (1.000, PRINTED_ROUND_1M_DB),
        (2.000, PRINTED_ROUND_2M_DB),
    ):
        round_duct = hvac.unlined_circular_duct_attenuation(
            hvac.OCTAVE_BANDS, run_length, diameter=0.160, model="vdi2081"
        )
        assert round_duct.values == pytest.approx(printed, abs=1e-9)


def test_the_bend_shifts_table_7_onto_its_own_limit_frequency() -> None:
    """Element 14: a 160 mm round bend, whose limit frequency is 1245 Hz.

    Equation (34) puts ``f_G = 0,586 c / d`` in the 1 kHz octave, three octaves
    above the 125 Hz the table is printed for, so the whole row moves three
    octaves up. That is what turns the printed 0, 1, 2, 3, 3, ... into the
    0, 0, 0, 1, 2, 3, 3, 3 of the example.
    """
    limit = 0.586 * EXAMPLE_SPEED_OF_SOUND / BEND_DIAMETER_M
    assert limit == pytest.approx(PRINTED_LIMIT_FREQUENCY_HZ, abs=0.5)

    bend = hvac.elbow_insertion_loss(
        hvac.OCTAVE_BANDS,
        BEND_DIAMETER_M,
        bend_type="round",
        speed_of_sound=EXAMPLE_SPEED_OF_SOUND,
        model="vdi2081",
    )
    assert bend.values == pytest.approx(PRINTED_BEND_DB, abs=1e-9)


def test_a_wider_bend_moves_the_same_row_further_down() -> None:
    """The shift is the whole point of Table 7 being printed only once.

    A 1250 mm rectangular duct is the size the table is tabulated for, so its
    row comes back unshifted; a duct eight times narrower has a limit frequency
    three octaves higher and takes the row three octaves up with it.
    """
    tabulated = hvac.elbow_insertion_loss(
        hvac.OCTAVE_BANDS, 1.250, speed_of_sound=340.0, model="vdi2081"
    )
    # c / (2 a) = 136 Hz, inside the 125 Hz octave, so nothing moves: the row's
    # 63 Hz, 125 Hz and 250 Hz columns arrive where they are printed.
    assert tabulated.values[:3] == pytest.approx((3.0, 7.0, 6.0), abs=1e-9)

    narrow = hvac.elbow_insertion_loss(
        hvac.OCTAVE_BANDS, 1.250 / 8.0, speed_of_sound=340.0, model="vdi2081"
    )
    assert narrow.values[3:6] == pytest.approx((3.0, 7.0, 6.0), abs=1e-9)
    assert narrow.values[:3] == pytest.approx((0.0, 0.0, 0.0), abs=1e-9)


def test_the_junction_is_the_area_split_alone() -> None:
    """Elements 3, 7 and 16, and the term VDI 2081 does not put here.

    Equation (35) is the share of the total branch area and nothing else. Long
    folds a reflection from the change of total section into the same function,
    which is why the third junction, whose branches sum to twice its feeder,
    differs by half a decibel between the two.
    """
    for fed, branches, printed in PRINTED_JUNCTIONS:
        areas = list(branches)
        index = areas.index(fed)
        vdi = hvac.split_loss(sum(areas), areas, branch=index, model="vdi2081")
        assert vdi == pytest.approx(printed, abs=0.05)

    _, mismatched, _ = PRINTED_JUNCTIONS[2]
    feeder = 0.020
    areas = list(mismatched)
    reflection = hvac.split_loss(feeder, areas) - hvac.split_loss(
        feeder, areas, model="vdi2081"
    )
    assert reflection == pytest.approx(0.512, abs=1e-3)


def test_the_vdi_duct_models_refuse_what_they_cannot_answer() -> None:
    """Each one asks for exactly what its own table needs."""
    with pytest.raises(ValueError, match=r"model='vdi2081' needs 'diameter'"):
        hvac.unlined_circular_duct_attenuation(None, 1.0, model="vdi2081")

    with pytest.raises(ValueError, match=r"'wrapped' has no meaning"):
        hvac.unlined_rectangular_duct_attenuation(
            hvac.OCTAVE_BANDS, 0.5, 0.4, 1.0, wrapped=True, model="vdi2081"
        )

    with pytest.raises(ValueError, match=r"outside VDI 2081 Table 5"):
        hvac.unlined_circular_duct_attenuation(None, 1.0, diameter=1.5, model="vdi2081")

    with pytest.raises(ValueError, match=r"prints no row for a bend lined on one"):
        hvac.elbow_insertion_loss(
            hvac.OCTAVE_BANDS,
            0.5,
            vanes=True,
            lined=True,
            lined_side="one",
            model="vdi2081",
        )


def test_lining_one_side_of_a_corner_is_its_own_row() -> None:
    """Table 7 tabulates lining before, after and on both sides separately.

    The ASHRAE table does not make that distinction, so it is a place where the
    German method answers a question the other cannot.
    """
    common = {
        "bend_type": "square",
        "lined": True,
        "speed_of_sound": 340.0,
        "model": "vdi2081",
    }
    both = hvac.elbow_insertion_loss(hvac.OCTAVE_BANDS, 1.250, **common)  # type: ignore[arg-type]
    one = hvac.elbow_insertion_loss(
        hvac.OCTAVE_BANDS,
        1.250,
        lined_side="one",
        **common,  # type: ignore[arg-type]
    )
    assert np.all(one.values <= both.values)
    assert not np.allclose(one.values, both.values)


# ---------------------------------------------------------------------------
# Flow noise: Section 5.2, and the rows of Table 1 that carry it
# ---------------------------------------------------------------------------
#: Table 1, element 3: the junction's flow noise, band by band. Its approach
#: velocity is the whole system's 16 000 m3/h over the 0,90 m2 feeder, not the
#: 4200 m3/h of the branch that carries on.
JUNCTION_APPROACH_VELOCITY = (16000.0 / 3600.0) / 0.90
JUNCTION_BRANCH_VELOCITY = (4200.0 / 3600.0) / 0.30
JUNCTION_BRANCH_DIAMETER = 0.62
JUNCTION_ROUNDING_RATIO = 0.025
PRINTED_JUNCTION_NOISE_DB = (39.1, 33.5, 27.4, 20.7, 13.7, 6.2, -1.5, -9.6)
PRINTED_JUNCTION_STROUHAL = (10.01, 19.87, 39.73, 79.46, 158.9, 317.8, 635.7, 1271.0)
PRINTED_JUNCTION_NORMALISED_DB = (
    -4.8,
    -12.9,
    -21.5,
    -30.6,
    -40.2,
    -50.1,
    -60.3,
    -70.9,
)
PRINTED_JUNCTION_K_DB = (4.2, 3.7, 3.2, 2.7, 2.1, 1.6, 1.1, 0.6)
#: Table 1, element 14: the bend's flow noise. A bend is the same law with the
#: two velocities equal, and the example applies no rounding correction to it.
BEND_AREA_M2 = math.pi * 0.08**2
BEND_VELOCITY = (280.0 / 3600.0) / BEND_AREA_M2
PRINTED_BEND_NOISE_DB = (26.9, 23.0, 18.1, 12.5, 6.5, -0.1, -7.0, -14.4)
#: Table 1, element 5: the straight run's flow noise, printed only as the two
#: overall levels, which is all the example carries forward.
STRAIGHT_AREA_M2 = 0.5 * 0.4
STRAIGHT_VELOCITY = (4200.0 / 3600.0) / STRAIGHT_AREA_M2
PRINTED_STRAIGHT_OVERALL_DB = 38.0
PRINTED_STRAIGHT_OVERALL_A_DB = 22.0


def test_the_straight_run_flow_noise_is_equations_16_and_17() -> None:
    """Element 5, and element 13 in the round duct, to one decibel as printed.

    The table rounds these to whole decibels and prints the velocity rounded
    too: reading 5,8 m/s off the table instead of the 5,8333 the duty gives
    would move the level by a tenth.
    """
    overall = hvac.flow_noise_straight_duct_overall(STRAIGHT_VELOCITY, STRAIGHT_AREA_M2)
    weighted = hvac.flow_noise_straight_duct_overall(
        STRAIGHT_VELOCITY, STRAIGHT_AREA_M2, weighting="A"
    )
    assert round(overall) == PRINTED_STRAIGHT_OVERALL_DB
    assert round(weighted) == PRINTED_STRAIGHT_OVERALL_A_DB

    round_area = math.pi * 0.08**2
    round_velocity = (280.0 / 3600.0) / round_area
    assert (
        round(hvac.flow_noise_straight_duct_overall(round_velocity, round_area)) == 19
    )
    assert (
        round(
            hvac.flow_noise_straight_duct_overall(
                round_velocity, round_area, weighting="A"
            )
        )
        == -1
    )


def test_the_two_overall_forms_are_the_printed_closed_forms() -> None:
    """Equation (16) and Equation (17), each against its own expression.

    They are not one number weighted: the A-weighted form carries a seventieth
    power of the speed and the unweighted one a fiftieth, because raising the
    speed also moves the spectrum into the part of the curve the weighting
    stops attenuating. Doubling the speed is worth 15,05 dB unweighted and
    21,07 dB A-weighted, and the two forms cross at 8,7 m/s in a square metre.
    """
    for velocity, area in ((2.0, 0.1), (5.8333333, 0.2), (12.0, 1.5)):
        assert hvac.flow_noise_straight_duct_overall(velocity, area) == pytest.approx(
            7.0 + 50.0 * math.log10(velocity) + 10.0 * math.log10(area)
        )
        assert hvac.flow_noise_straight_duct_overall(
            velocity, area, weighting="A"
        ) == pytest.approx(
            -25.0 + 70.0 * math.log10(velocity) + 10.0 * math.log10(area)
        )
    doubled = hvac.flow_noise_straight_duct_overall(
        4.0, 1.0
    ) - hvac.flow_noise_straight_duct_overall(2.0, 1.0)
    assert doubled == pytest.approx(50.0 * math.log10(2.0))
    doubled_a = hvac.flow_noise_straight_duct_overall(
        4.0, 1.0, weighting="A"
    ) - hvac.flow_noise_straight_duct_overall(2.0, 1.0, weighting="A")
    assert doubled_a == pytest.approx(70.0 * math.log10(2.0))


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"flow_velocity": 0.0}, "flow_velocity"),
        ({"area": -1.0}, "area"),
        ({"weighting": "C"}, "'weighting' must be one of"),
    ],
)
def test_the_overall_forms_refuse_what_the_guideline_does_not_print(
    kwargs: dict[str, object], match: str
) -> None:
    call: dict[str, object] = {"flow_velocity": 5.0, "area": 0.2}
    call.update(kwargs)
    with pytest.raises(ValueError, match=match):
        hvac.flow_noise_straight_duct_overall(**call)  # type: ignore[arg-type]


def test_the_limit_frequency_is_the_ducts_own_first_cut_on() -> None:
    """Equations (33) and (34) are the first cut-on, and are called as such.

    The rectangular form is exactly ``c / (2 a)``, and the round one is the
    guideline's rounding of the first circular mode: ``0,586`` for
    ``1,8412 / pi``, which agrees to four figures. Written twice, the two
    could drift; called once, they cannot.
    """
    from phonometry import noise_control as nc

    assert hvac._vdi2081_limit_frequency("rectangular", 1.25, 343.0) == pytest.approx(
        343.0 / (2.0 * 1.25)
    )
    assert hvac._vdi2081_limit_frequency("round", 0.16, 340.0) == pytest.approx(
        0.586 * 340.0 / 0.16, rel=2e-4
    )
    assert hvac._vdi2081_limit_frequency("round", 0.16, 340.0) == pytest.approx(
        nc.plane_wave_limit(diameter=0.16, speed_of_sound=340.0)
    )


def test_the_straight_run_model_is_already_the_german_one() -> None:
    """`flow_noise_straight_duct` is Equation (16) with the Figure 16 shape.

    Bies Eq. (8.251) reproduces VDI 2081 Part 1 Section 5.2.1 and says so, so
    there is no second model to add here, only the oracle it never had. The
    spectrum does not sum back to the overall level of Equation (16): the
    relative shape is a fit, and the guideline prints the two separately.
    """
    spectrum = hvac.flow_noise_straight_duct(
        hvac.OCTAVE_BANDS, STRAIGHT_VELOCITY, STRAIGHT_AREA_M2
    )
    overall = (
        7.0 + 50.0 * math.log10(STRAIGHT_VELOCITY) + 10.0 * math.log10(STRAIGHT_AREA_M2)
    )
    shape = -2.0 - 26.0 * np.log10(1.14 + 0.02 * hvac.OCTAVE_BANDS / STRAIGHT_VELOCITY)
    assert spectrum.values == pytest.approx(overall + shape, abs=1e-9)


def test_the_junction_flow_noise_reproduces_element_3() -> None:
    """Equation (18) with Figures 17 and 18, band by band.

    The approach velocity is the one in the duct **ahead** of the junction, so
    it comes from the flow the feeder carries and not from the branch: reading
    it off the element's own 4200 m3/h would put the ratio at one and lose five
    decibels at 63 Hz.
    """
    strouhal = hvac.OCTAVE_BANDS * JUNCTION_BRANCH_DIAMETER / JUNCTION_BRANCH_VELOCITY
    assert strouhal == pytest.approx(PRINTED_JUNCTION_STROUHAL, rel=5e-3)

    lg_st = np.log10(strouhal)
    ratio = math.log10(JUNCTION_APPROACH_VELOCITY / JUNCTION_BRANCH_VELOCITY)
    normalised = 12.0 - 21.5 * lg_st**1.268 + (32.0 + 13.0 * lg_st) * ratio
    assert normalised == pytest.approx(PRINTED_JUNCTION_NORMALISED_DB, abs=0.1)

    correction = 13.9 * (3.43 - lg_st) * (0.15 - JUNCTION_ROUNDING_RATIO)
    assert correction == pytest.approx(PRINTED_JUNCTION_K_DB, abs=0.05)

    noise = hvac.flow_noise_bend(
        hvac.OCTAVE_BANDS,
        JUNCTION_BRANCH_VELOCITY,
        0.30,
        0.6,
        model="vdi2081",
        branch_diameter=JUNCTION_BRANCH_DIAMETER,
        approach_velocity=JUNCTION_APPROACH_VELOCITY,
        rounding_ratio=JUNCTION_ROUNDING_RATIO,
    )
    assert noise.values == pytest.approx(PRINTED_JUNCTION_NOISE_DB, abs=0.05)


def test_a_bend_is_the_same_law_with_one_velocity() -> None:
    """Element 14: the two velocities equal, and no rounding correction.

    Figure 18 is drawn for the rounding of a junction, and all its curves cross
    zero at ``r / d_a = 0,15``, so leaving it out is not the same as passing
    nought: a sharp-cornered junction earns over 6 dB there.
    """
    bend = hvac.flow_noise_bend(
        hvac.OCTAVE_BANDS,
        BEND_VELOCITY,
        BEND_AREA_M2,
        0.16,
        model="vdi2081",
        branch_diameter=0.160,
    )
    assert bend.values == pytest.approx(PRINTED_BEND_NOISE_DB, abs=0.05)

    sharp = hvac.flow_noise_bend(
        hvac.OCTAVE_BANDS,
        BEND_VELOCITY,
        BEND_AREA_M2,
        0.16,
        model="vdi2081",
        branch_diameter=0.160,
        rounding_ratio=0.0,
    )
    # K falls with frequency, from 6,3 dB at 63 Hz to 1,9 dB at 8 kHz, so a
    # sharp corner is louder in every band and most of all in the lowest.
    assert np.all(sharp.values > bend.values)
    assert sharp.values[0] - bend.values[0] > 6.0

    # A junction rounded to 0,15 of its branch diameter is the crossing point,
    # so it lands back on the uncorrected law.
    crossing = hvac.flow_noise_bend(
        hvac.OCTAVE_BANDS,
        BEND_VELOCITY,
        BEND_AREA_M2,
        0.16,
        model="vdi2081",
        branch_diameter=0.160,
        rounding_ratio=0.15,
    )
    assert crossing.values == pytest.approx(bend.values, abs=1e-9)


def test_below_a_strouhal_number_of_one_the_fit_does_not_apply() -> None:
    """Both figures say so, so those bands carry no contribution at all.

    Returning an extrapolation would be worse than returning nothing: the fit
    turns over below one and its fractional power of ``lg St`` is not real
    there.
    """
    slow = hvac.flow_noise_bend(
        hvac.OCTAVE_BANDS,
        20.0,
        0.05,
        0.2,
        model="vdi2081",
        branch_diameter=0.05,
    )
    below = hvac.OCTAVE_BANDS * 0.05 / 20.0 <= 1.0
    assert np.all(np.isneginf(slow.values[below]))
    assert np.all(np.isfinite(slow.values[~below]))


def test_the_vdi_flow_noise_asks_for_the_branch_it_is_written_on() -> None:
    """Equation (18) is written on the branch diameter, which Bies does not take."""
    with pytest.raises(ValueError, match=r"model='vdi2081' needs 'branch_diameter'"):
        hvac.flow_noise_bend(hvac.OCTAVE_BANDS, 5.0, 0.2, 0.4, model="vdi2081")


# ---------------------------------------------------------------------------
# The silencer and the nozzle: Sections 7.2.4.2 and 6.6
# ---------------------------------------------------------------------------
#: Table 1, element 2: a splitter silencer, 1500 x 600 mm over 2 m, five
#: 200 mm splitters with 100 mm gaps, and the two quantities the table prints
#: for it beside its manufacturer's attenuation.
SILENCER_GAP_VELOCITY = 14.81
SILENCER_PRESSURE_DROP_PA = 145.0
SILENCER_APPROACH_AREA_M2 = 1.5 * 0.6
SILENCER_GAP_M = 0.100
PRINTED_SILENCER_LWA_DB = 52.0
PRINTED_SILENCER_NOISE_DB = (62.7, 58.3, 53.7, 49.4, 45.4, 41.9, 38.6, 35.6)
#: Table 1, element 2 again: the Strouhal row, which is what shows the printed
#: hydraulic diameter is not the one the example computes with.
PRINTED_SILENCER_STROUHAL = (0.9, 1.7, 3.4, 6.8, 13.5, 27.0, 54.0, 108.0)
PRINTED_SILENCER_HYDRAULIC_DIAMETER_M = 0.171
#: Table 2, element 18: the end reflection of a 200 mm nozzle in a ceiling,
#: printed both as computed and capped at 15 dB.
NOZZLE_DIAMETER_M = 0.200
PRINTED_NOZZLE_DB = (15.8, 10.2, 5.3, 2.1, 0.7, 0.2, 0.1, 0.1)
PRINTED_NOZZLE_CAPPED_DB = (15.0, 10.2, 5.3, 2.1, 0.7, 0.2, 0.1, 0.1)


def test_the_silencer_self_noise_reproduces_element_2() -> None:
    """Equations (49), (46), (50) and (51), band by band."""
    weighted = (
        56.6 * math.log10(SILENCER_GAP_VELOCITY)
        - 0.5 * math.log10(SILENCER_PRESSURE_DROP_PA)
        + 10.0 * math.log10(SILENCER_APPROACH_AREA_M2)
        - 12.7
    )
    assert weighted == pytest.approx(PRINTED_SILENCER_LWA_DB, abs=0.05)

    noise = hvac.silencer_self_noise(
        hvac.OCTAVE_BANDS,
        SILENCER_GAP_VELOCITY,
        5,
        0.6,
        model="vdi2081",
        pressure_drop_pa=SILENCER_PRESSURE_DROP_PA,
        approach_area=SILENCER_APPROACH_AREA_M2,
        airway_width=SILENCER_GAP_M,
    )
    assert noise.values == pytest.approx(PRINTED_SILENCER_NOISE_DB, abs=0.05)


def test_the_example_computes_on_twice_the_gap_not_on_the_diameter_it_prints() -> None:
    """Element 2 prints one hydraulic diameter and works with another.

    Section 7.2.4.2 sets ``St = f d_h / v_i``, so the printed ``d_h`` and the
    printed Strouhal row determine each other. They disagree: with the printed
    0,171 m not one of the eight rounds onto the row, and with ``2 s`` all
    eight do. Both are defensible diameters for the gap, ``4 A / P`` and the
    parallel-plate limit, and ``docs/ERRATA.md`` records which the example
    used.
    """
    printed = np.array(PRINTED_SILENCER_STROUHAL)
    from_printed_diameter = (
        hvac.OCTAVE_BANDS
        * PRINTED_SILENCER_HYDRAULIC_DIAMETER_M
        / SILENCER_GAP_VELOCITY
    )
    from_twice_the_gap = (
        hvac.OCTAVE_BANDS * 2.0 * SILENCER_GAP_M / SILENCER_GAP_VELOCITY
    )
    # The row is printed to one decimal, so the test is whether each value
    # rounds onto its cell, which is the only comparison the print supports.
    assert np.round(from_twice_the_gap, 1) == pytest.approx(printed, abs=1e-9)
    assert not np.any(np.round(from_printed_diameter, 1) == printed)


def test_the_nozzle_reflection_reproduces_element_18() -> None:
    """Figure 28 in closed form, and the flat 15 dB ceiling of Section 6.6.

    The area comes from the nozzle's own bore, so a 200 mm outlet gives the
    0,0314 m2 the example prints, and the ceiling is what turns its computed
    15,8 dB at 63 Hz into the 15,0 it carries forward.
    """
    uncapped = hvac.end_reflection_loss(
        hvac.OCTAVE_BANDS,
        NOZZLE_DIAMETER_M,
        termination="wall",
        method="vdi2081",
        speed_of_sound=EXAMPLE_SPEED_OF_SOUND,
        maximum_reduction_db=None,
    )
    assert uncapped.values == pytest.approx(PRINTED_NOZZLE_DB, abs=0.05)

    capped = hvac.end_reflection_loss(
        hvac.OCTAVE_BANDS,
        NOZZLE_DIAMETER_M,
        termination="wall",
        method="vdi2081",
        speed_of_sound=EXAMPLE_SPEED_OF_SOUND,
    )
    assert capped.values == pytest.approx(PRINTED_NOZZLE_CAPPED_DB, abs=0.05)


def test_the_nozzle_knows_all_four_solid_angles() -> None:
    """Figure 28 tabulates a nozzle in the room, in a wall, on an edge and in a corner.

    Halving the solid angle doubles the pressure the same power makes, so each
    step towards a corner is worth 3 dB of reflection at low frequency, where
    the piston term dominates.
    """
    levels = [
        hvac.end_reflection_loss(
            np.array([63.0]),
            NOZZLE_DIAMETER_M,
            termination=where,
            method="vdi2081",
            speed_of_sound=EXAMPLE_SPEED_OF_SOUND,
            maximum_reduction_db=None,
        ).values[0]
        for where in ("room", "wall", "edge", "corner")
    ]
    steps = np.diff(levels)
    assert np.all(steps < 0.0)
    # Not exactly 3 dB: the piston term is 10 lg(1 + x) rather than 10 lg(x),
    # so the leading one holds each step a little short of the halving.
    assert steps == pytest.approx([-3.0, -3.0, -3.0], abs=0.25)
    assert np.all(steps > -3.0)


def test_the_vdi_silencer_and_nozzle_refuse_what_they_cannot_answer() -> None:
    """Each asks for what its own equation needs, and nothing else."""
    with pytest.raises(ValueError, match=r"model='vdi2081' needs"):
        hvac.silencer_self_noise(None, 14.0, 5, 0.6, model="vdi2081")

    with pytest.raises(ValueError, match=r"'termination' must be one of"):
        hvac.end_reflection_loss(
            hvac.OCTAVE_BANDS, 0.2, termination="duct", method="vdi2081"
        )


# ---------------------------------------------------------------------------
# The chain, and the assessment curve it is measured against
# ---------------------------------------------------------------------------
#: Section 1.1 of Blatt 2: the correction that turns an A-weighted room
#: requirement into a per-octave one.
PRINTED_KA_DB = (21.0, 11.0, 4.0, -2.0, -5.0, -6.0, -6.0, -4.0)
#: Table 1, element 1 again: the note under the fan row recommends adding the
#: difference of the two summed levels to every octave, because the relative
#: spectrum is fitted to the shape rather than to the total.
PRINTED_FAN_CORRECTED_DB = (92.3, 90.7, 88.2, 84.8, 80.5, 75.3, 69.2, 62.1)
#: Table 1, elements 2 and 3: the running total after each one.
PRINTED_AFTER_SILENCER_DB = (86.3, 73.9, 54.4, 50.5, 45.7, 45.1, 49.5, 44.7)
PRINTED_AFTER_JUNCTION_DB = (80.8, 68.3, 48.9, 44.9, 40.2, 39.5, 44.0, 39.1)
#: Table 1, element 2: the splitter attenuation, which is manufacturer's data
#: rather than anything the guideline computes.
SILENCER_ATTENUATION_DB = (6.0, 17.0, 42.0, 41.0, 47.0, 33.0, 20.0, 18.0)


def _energy_sum(*spectra: object) -> np.ndarray:
    """Add sound power levels band by band."""
    total = np.zeros(len(hvac.OCTAVE_BANDS))
    for spectrum in spectra:
        total += 10.0 ** (np.asarray(spectrum, dtype=float) / 10.0)
    return 10.0 * np.log10(total)


def test_the_assessment_curve_is_the_inverse_a_weighting_less_five() -> None:
    """Section 1.1: ``K_A = -A - 5``, and Equation (1) applies it.

    The 5 dB is what the guideline allows for the sum of eight octave bands. A
    spectrum flat in A-weighted terms would earn 9 dB; 5 is taken because the
    noise of an air-conditioning system does not follow the inverse A curve.
    """
    assert hvac.VDI2081_SPECTRAL_CORRECTION == pytest.approx(PRINTED_KA_DB, abs=1e-9)
    inverse = -np.array(PRINTED_A_WEIGHTING_DB) - 5.0
    assert np.round(inverse) == pytest.approx(PRINTED_KA_DB, abs=1e-9)

    limits = hvac.octave_band_limits(25.0)
    assert limits.values == pytest.approx(25.0 + np.array(PRINTED_KA_DB), abs=1e-9)

    # The same requirement read the other way round, which is what the worked
    # example does: weight the spectrum and compare against a flat L_A - 5.
    weighted_limit = 25.0 - 5.0
    assert limits.values + np.array(PRINTED_A_WEIGHTING_DB) == pytest.approx(
        weighted_limit, abs=0.5
    )


def test_the_fan_spectrum_is_corrected_onto_its_own_total() -> None:
    """The note under element 1, applied as the table applies it.

    Equation (15) fits the shape of the spectrum, not its total, so the eight
    bands do not sum back to Equation (13). The guideline recommends adding the
    difference to every octave, which is what its own next row carries forward.
    """
    fan = _supply_fan().values
    total = 10.0 * math.log10(float(np.sum(10.0 ** (fan / 10.0))))
    corrected = fan + (PRINTED_OVERALL_DB - total)
    assert corrected == pytest.approx(PRINTED_FAN_CORRECTED_DB, abs=0.1)


def test_the_chain_carries_the_example_from_the_fan_to_the_second_junction() -> None:
    """Elements 1 to 3 in sequence: attenuate, then add what the element makes.

    Each of the three pieces has been checked against its own printed row
    above; this is the check that they compose. The example rounds to one
    decimal at every step and carries the rounded value forward, so a tenth of
    a decibel is the most the running total can be held to.
    """
    fan = _supply_fan().values
    total = 10.0 * math.log10(float(np.sum(10.0 ** (fan / 10.0))))
    running = fan + (PRINTED_OVERALL_DB - total)

    silencer_noise = hvac.silencer_self_noise(
        hvac.OCTAVE_BANDS,
        SILENCER_GAP_VELOCITY,
        5,
        0.6,
        model="vdi2081",
        pressure_drop_pa=SILENCER_PRESSURE_DROP_PA,
        approach_area=SILENCER_APPROACH_AREA_M2,
        airway_width=SILENCER_GAP_M,
    ).values
    running = _energy_sum(running - np.array(SILENCER_ATTENUATION_DB), silencer_noise)
    assert running == pytest.approx(PRINTED_AFTER_SILENCER_DB, abs=0.1)

    split = hvac.split_loss(
        0.30 + 0.36 + 0.42, [0.30, 0.36, 0.42], branch=0, model="vdi2081"
    )
    junction_noise = hvac.flow_noise_bend(
        hvac.OCTAVE_BANDS,
        JUNCTION_BRANCH_VELOCITY,
        0.30,
        0.6,
        model="vdi2081",
        branch_diameter=JUNCTION_BRANCH_DIAMETER,
        approach_velocity=JUNCTION_APPROACH_VELOCITY,
        rounding_ratio=JUNCTION_ROUNDING_RATIO,
    ).values
    running = _energy_sum(running - split, junction_noise)
    assert running == pytest.approx(PRINTED_AFTER_JUNCTION_DB, abs=0.1)


def test_the_octave_limits_refuse_a_level_that_is_not_one() -> None:
    with pytest.raises(ValueError, match=r"'a_weighted_limit_db' must be finite"):
        hvac.octave_band_limits(float("nan"))
