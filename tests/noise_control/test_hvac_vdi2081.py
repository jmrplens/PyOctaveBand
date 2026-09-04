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
