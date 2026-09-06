#  Copyright (c) 2026. Jose Manuel Requena Plens
"""IEC 60534-8-4:2005 against the three examples of Annex A.

Table A.1 prints every intermediate of three operating points of one DN 100
globe valve on water: one turbulent, one cavitating, and a third that repeats
the second with the characteristic pressure ratio shifted by 0,1 to show what
that uncertainty costs. Those printed columns are the oracle here, and each
test names the equation whose value it pins.

Three printed intermediates do not follow from the intermediates printed
beside them, one is printed with the wrong sign, and Equations (18a) and (18b)
carry conditions on two different thresholds; all of them are recorded in
``docs/ERRATA.md`` and each has a test here that says what the equations give
and why the printed figure cannot be met exactly.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from phonometry.noise_control import ValveNoiseWarning
from phonometry.noise_control import valves_hydrodynamic as hydro

if TYPE_CHECKING:  # pragma: no cover - typing only
    from numpy.typing import NDArray

#: A.1's given data, shared by the three columns, in SI units.
COMMON: dict[str, Any] = {
    "inlet_pressure": 1.0e6,
    "vapour_pressure": 2.32e3,
    "liquid_density": 997.0,
    "liquid_sound_speed": 1400.0,
    "flow_coefficient": 90.0,
    "style_modifier": 0.42,
    "pressure_recovery": 0.92,
    "power_ratio": 0.25,
    "valve_diameter": 0.1,
    "seat_diameter": 0.1,
    "internal_diameter": 0.1071,
    "wall_thickness": 0.0036,
    "pipe_density": 7800.0,
}

#: Equation (3a) for this valve, which the annex prints as 0,2543.
INCIPIENT = hydro.incipient_cavitation_ratio(90.0, 0.42, 0.92)

#: The per-column data of Table A.1. The third column is the second with
#: "Calculation with x_Fz = x_Fz + 0,1", as the annex prints it.
EXAMPLES: list[dict[str, Any]] = [
    {"example": 1, "mass_flow": 30.0, "outlet_pressure": 8.0e5, "shift": 0.0},
    {"example": 2, "mass_flow": 40.0, "outlet_pressure": 6.5e5, "shift": 0.0},
    {"example": 3, "mass_flow": 40.0, "outlet_pressure": 6.5e5, "shift": 0.1},
]

#: The band Table A.1 evaluates the frequency route at.
BAND_HZ = 8000.0


def _run(index: int) -> hydro.HydrodynamicValveNoise:
    """Column ``index`` of Table A.1, one to three."""
    case = dict(EXAMPLES[index - 1])
    shift = case.pop("shift")
    case.pop("example")
    return hydro.valve_hydrodynamic_noise(
        **COMMON, **case, incipient_ratio=INCIPIENT + shift
    )


def _band(result: hydro.HydrodynamicValveNoise, values: NDArray[np.float64]) -> float:
    """``values`` at the 8 kHz band, which is the one the annex prints."""
    index = int(np.argmin(np.abs(result.frequency - BAND_HZ)))
    return float(values[index])


class TestPreliminaryCalculations:
    """Clause 4: Equations (1) to (6)."""

    @pytest.mark.parametrize(
        ("index", "expected"), [(1, 0.2005), (2, 0.3508), (3, 0.3508)]
    )
    def test_the_differential_pressure_ratio_matches(
        self, index: int, expected: float
    ) -> None:
        assert _run(index).pressure_ratio == pytest.approx(expected, abs=5e-5)

    @pytest.mark.parametrize(
        ("index", "expected"), [(1, 2.0e5), (2, 3.5e5), (3, 3.5e5)]
    )
    def test_the_cavitation_differential_matches(
        self, index: int, expected: float
    ) -> None:
        assert _run(index).cavitation_differential == pytest.approx(expected, abs=1.0)

    def test_the_differential_is_capped_at_the_choking_point(self) -> None:
        # The annex's own second candidate, F_L^2 (p_1 - p_v) = 8,44e5 Pa,
        # is above all three printed differentials, so Equation (2) returns
        # p_1 - p_2 in every column. Push p_2 down and the cap binds.
        choked = hydro.cavitation_differential(
            inlet_pressure=1.0e6,
            outlet_pressure=1.0e5,
            vapour_pressure=2.32e3,
            pressure_recovery=0.92,
        )
        assert choked == pytest.approx(0.92**2 * (1.0e6 - 2.32e3), rel=1e-12)
        assert choked < 1.0e6 - 1.0e5

    def test_the_chain_caps_it_too(self) -> None:
        # Annex A never chokes, so the cap has to be exercised on a column of
        # our own: 9 bar across the same valve, against the 8,444 bar of
        # F_L^2 (p_1 - p_v). Equation (5) then runs on the cap and not on the
        # differential, which is 3 % less velocity than the full drop would
        # give and 8 % less stream power.
        result = hydro.valve_hydrodynamic_noise(
            **COMMON, mass_flow=60.0, outlet_pressure=1.0e5, incipient_ratio=INCIPIENT
        )
        assert result.differential == pytest.approx(9.0e5)
        assert result.cavitation_differential == pytest.approx(844436.35, abs=0.1)
        assert result.velocity == pytest.approx(44.737, abs=5e-3)
        uncapped = hydro.vena_contracta_velocity(9.0e5, 997.0, 0.92)
        assert result.velocity < uncapped
        # Equation (9) carries (p_1 - p_2)/Delta p_c, which is 1 in every
        # printed column and 1,066 here, so this is also the one case that
        # pins that factor.
        found = result.cavitation_efficiency
        assert found is not None
        without = found / math.sqrt(
            result.differential / result.cavitation_differential
        )
        assert found == pytest.approx(1.0328 * without, rel=1e-3)

    def test_the_incipient_ratio_matches_equation_3a(self) -> None:
        assert INCIPIENT == pytest.approx(0.2543, abs=5e-5)

    @pytest.mark.parametrize(
        ("index", "expected"), [(1, 0.2386), (2, 0.2386), (3, 0.3324)]
    )
    def test_the_corrected_ratio_matches_equation_3c(
        self, index: int, expected: float
    ) -> None:
        assert _run(index).corrected_ratio == pytest.approx(expected, abs=5e-5)

    def test_the_correction_leaves_the_reference_pressure_alone(self) -> None:
        # Equation (3c) is drawn at 6e5 Pa, so there it does nothing.
        ratio = hydro.corrected_incipient_ratio(
            0.2543, hydro.REFERENCE_INLET_PRESSURE_PA
        )
        assert ratio == pytest.approx(0.2543, rel=1e-12)

    def test_a_higher_inlet_pressure_lowers_the_threshold(self) -> None:
        low = hydro.corrected_incipient_ratio(0.2543, 2.0e5)
        high = hydro.corrected_incipient_ratio(0.2543, 5.0e6)
        assert high < 0.2543 < low

    def test_the_multihole_form_answers_on_hole_area_alone(self) -> None:
        # Equation (3b) sees N_o d_H^2, so twice the holes at 1/sqrt(2) the
        # diameter is the same trim as far as the threshold is concerned.
        one = hydro.multihole_incipient_cavitation_ratio(60, 0.004, 0.92)
        two = hydro.multihole_incipient_cavitation_ratio(
            120, 0.004 / math.sqrt(2), 0.92
        )
        assert one == pytest.approx(two, rel=1e-12)

    def test_the_multihole_form_matches_its_own_arithmetic(self) -> None:
        # 1/sqrt(4,5 + 1 650 * 120 * 0,004^2 / 0,92): the two printed
        # constants are 4,5 and 1 650, and neither is free.
        assert hydro.multihole_incipient_cavitation_ratio(
            120, 0.004, 0.92
        ) == pytest.approx(0.35481, abs=5e-5)
        # With no holes at all the offset alone would give 1/sqrt(4,5).
        assert hydro.multihole_incipient_cavitation_ratio(
            1, 1e-9, 0.92
        ) == pytest.approx(1.0 / math.sqrt(4.5), abs=1e-6)

    def test_a_multihole_trim_stays_quiet_longer_than_a_single_port(self) -> None:
        multihole = hydro.multihole_incipient_cavitation_ratio(120, 0.004, 0.92)
        single = hydro.incipient_cavitation_ratio(90.0, 1.0, 0.92)
        assert multihole > single

    @pytest.mark.parametrize("index", [1, 2, 3])
    def test_the_jet_diameter_matches_equation_4(self, index: int) -> None:
        assert _run(index).jet_diameter == pytest.approx(0.01758, abs=5e-6)

    @pytest.mark.parametrize(
        ("index", "expected"), [(1, 21.772), (2, 28.801), (3, 28.801)]
    )
    def test_the_vena_contracta_velocity_matches(
        self, index: int, expected: float
    ) -> None:
        assert _run(index).velocity == pytest.approx(expected, abs=5e-4)

    @pytest.mark.parametrize(
        ("index", "expected"), [(1, 6018.05), (2, 14042.1), (3, 14042.1)]
    )
    def test_the_stream_power_matches(self, index: int, expected: float) -> None:
        assert _run(index).stream_power == pytest.approx(expected, abs=0.05)


class TestEfficiencies:
    """Equations (7a), (7b), (8) and (9)."""

    @pytest.mark.parametrize(
        ("index", "expected"), [(1, 1.555e-6), (2, 2.057e-6), (3, 2.057e-6)]
    )
    def test_the_turbulent_efficiency_matches(
        self, index: int, expected: float
    ) -> None:
        assert _run(index).turbulent_efficiency == pytest.approx(expected, abs=5e-10)

    def test_the_turbulent_efficiency_reaches_its_constant_at_the_sound_speed(
        self,
    ) -> None:
        assert hydro.turbulent_efficiency(1400.0, 1400.0) == pytest.approx(1e-4)

    def test_the_turbulent_column_has_no_cavitation_term(self) -> None:
        result = _run(1)
        assert result.regime == "turbulent"
        assert result.cavitation_efficiency is None
        assert result.cavitation_peak is None
        assert result.cavitation_transmission_loss is None

    @pytest.mark.parametrize(("index", "expected"), [(2, 1.243e-6), (3, 1.992e-8)])
    def test_the_cavitation_efficiency_matches(
        self, index: int, expected: float
    ) -> None:
        found = _run(index).cavitation_efficiency
        assert found is not None
        assert found == pytest.approx(expected, rel=1e-3)

    def test_the_two_regimes_meet_without_a_step(self) -> None:
        # (x_F - x_Fzp1)^1,5 puts Equation (9) at exactly zero on the
        # threshold, so the cavitating branch starts where the turbulent one
        # ends rather than jumping.
        efficiency = hydro.cavitation_efficiency(
            turbulent=2.057e-6,
            differential=3.5e5,
            choked_differential=3.5e5,
            pressure_ratio=0.2386,
            corrected_ratio=0.2386,
        )
        assert efficiency == 0.0

    def test_it_climbs_steeply_once_the_threshold_is_passed(self) -> None:
        near = hydro.cavitation_efficiency(
            turbulent=2.057e-6,
            differential=3.5e5,
            choked_differential=3.5e5,
            pressure_ratio=0.30,
            corrected_ratio=0.2386,
        )
        far = hydro.cavitation_efficiency(
            turbulent=2.057e-6,
            differential=3.5e5,
            choked_differential=3.5e5,
            pressure_ratio=0.60,
            corrected_ratio=0.2386,
        )
        assert far > 100.0 * near

    @pytest.mark.parametrize(
        ("index", "expected"), [(1, 0.00234), (2, 0.01158), (3, 0.00729)]
    )
    def test_the_sound_power_matches(self, index: int, expected: float) -> None:
        assert _run(index).sound_power == pytest.approx(expected, abs=5e-6)

    def test_the_power_ratio_scales_the_sound_power(self) -> None:
        # Table 2 is the only place r_W enters, and it enters linearly.
        louder = hydro.valve_hydrodynamic_noise(
            **{**COMMON, "power_ratio": 0.5},
            **{k: v for k, v in EXAMPLES[0].items() if k not in ("example", "shift")},
            incipient_ratio=INCIPIENT,
        )
        assert louder.sound_power == pytest.approx(2.0 * _run(1).sound_power, rel=1e-12)
        assert louder.external_level == pytest.approx(
            _run(1).external_level + 10.0 * math.log10(2.0), abs=1e-9
        )


class TestInternalLevel:
    """Equation (10)."""

    @pytest.mark.parametrize(
        ("index", "expected"), [(1, 149.596), (2, 156.543), (3, 154.532)]
    )
    def test_the_internal_level_matches(self, index: int, expected: float) -> None:
        assert _run(index).internal_level == pytest.approx(expected, abs=5e-3)

    def test_the_liquid_impedance_is_in_the_equation(self) -> None:
        # The printed page loses the Greek base of rho_L and leaves a bare
        # subscript; Table A.1 prints it intact. Without the density the
        # level would be 30 dB low for water.
        with_density = hydro.internal_sound_pressure_level(
            sound_power=0.00234,
            density=997.0,
            sound_speed=1400.0,
            internal_diameter=0.1071,
        )
        assert with_density == pytest.approx(149.596, abs=5e-3)
        assert with_density - 10.0 * math.log10(997.0) == pytest.approx(119.6, abs=0.1)


class TestPeakFrequencies:
    """Equations (11), (12) and (13)."""

    @pytest.mark.parametrize(
        ("index", "expected"), [(1, 0.399), (2, 0.399), (3, 0.243)]
    )
    def test_the_strouhal_number_matches_the_annex_form(
        self, index: int, expected: float
    ) -> None:
        assert _run(index).strouhal_number == pytest.approx(expected, abs=5e-4)

    def test_the_two_diameters_of_equation_12_are_not_interchangeable(
        self,
    ) -> None:
        # Annex A's valve has d = d_o = 0,1 m, which is exactly the case that
        # cannot tell the denominator's two diameters apart. Make them
        # differ: the product d*d_o is what the equation divides by.
        shared: dict[str, Any] = {
            "flow_coefficient": 90.0,
            "style_modifier": 0.42,
            "pressure_recovery": 0.92,
            "corrected_ratio": 0.2386,
            "inlet_pressure": 1.0e6,
            "vapour_pressure": 2.32e3,
        }
        wide = hydro.jet_strouhal_number(
            **shared, valve_diameter=0.1, seat_diameter=0.05
        )
        narrow = hydro.jet_strouhal_number(
            **shared, valve_diameter=0.05, seat_diameter=0.1
        )
        assert wide == pytest.approx(narrow, rel=1e-12)
        assert wide == pytest.approx(2.0 * 0.399, abs=1e-3)

    def test_the_clause_form_is_a_different_number(self) -> None:
        # Clause 5.1 prints 0,02 and no F_d; Table A.1 prints 0,036 and
        # F_d^0,75. See ``docs/ERRATA.md``: only the annex form reproduces
        # the annex's own 0,399.
        shared: dict[str, Any] = {
            "flow_coefficient": 90.0,
            "style_modifier": 0.42,
            "pressure_recovery": 0.92,
            "corrected_ratio": 0.2386,
            "valve_diameter": 0.1,
            "seat_diameter": 0.1,
            "inlet_pressure": 1.0e6,
            "vapour_pressure": 2.32e3,
        }
        annex = hydro.jet_strouhal_number(**shared, form="annex")
        clause = hydro.jet_strouhal_number(**shared, form="clause")
        assert annex == pytest.approx(0.399, abs=5e-4)
        assert clause == pytest.approx(0.4253, abs=5e-4)
        assert clause / annex == pytest.approx(0.02 / (0.036 * 0.42**0.75), rel=1e-12)

    def test_the_two_forms_agree_only_at_one_style_modifier(self) -> None:
        crossing = (0.02 / 0.036) ** (1.0 / 0.75)
        shared: dict[str, Any] = {
            "flow_coefficient": 90.0,
            "style_modifier": crossing,
            "pressure_recovery": 0.92,
            "corrected_ratio": 0.2386,
            "valve_diameter": 0.1,
            "seat_diameter": 0.1,
            "inlet_pressure": 1.0e6,
            "vapour_pressure": 2.32e3,
        }
        assert hydro.jet_strouhal_number(**shared, form="annex") == pytest.approx(
            hydro.jet_strouhal_number(**shared, form="clause"), rel=1e-12
        )

    @pytest.mark.parametrize(("index", "expected"), [(2, 654.35), (3, 397.93)])
    def test_the_turbulent_peak_matches(self, index: int, expected: float) -> None:
        assert _run(index).turbulent_peak == pytest.approx(expected, abs=0.05)

    def test_the_first_column_prints_a_peak_its_own_chain_misses(self) -> None:
        # Table A.1 prints 494,5 Hz where the unrounded chain gives 494,64;
        # columns 2 and 3 reproduce to the last printed digit. Recorded in
        # ``docs/ERRATA.md``; nothing downstream moves at the printed
        # precision.
        assert _run(1).turbulent_peak == pytest.approx(494.64, abs=0.05)
        assert _run(1).turbulent_peak != pytest.approx(494.5, abs=0.05)

    @pytest.mark.parametrize(("index", "expected"), [(2, 1088.94), (3, 1973.43)])
    def test_the_cavitation_peak_matches(self, index: int, expected: float) -> None:
        found = _run(index).cavitation_peak
        assert found is not None
        assert found == pytest.approx(expected, abs=0.05)

    def test_the_cavitation_peak_starts_at_six_times_the_turbulent_one(self) -> None:
        # On the threshold both brackets of Equation (13) are exactly 1.
        peak = hydro.cavitation_peak_frequency(500.0, 0.2386, 0.2386)
        assert peak == pytest.approx(3000.0, rel=1e-12)

    def test_it_falls_as_the_valve_goes_further_into_cavitation(self) -> None:
        near = hydro.cavitation_peak_frequency(500.0, 0.30, 0.2386)
        far = hydro.cavitation_peak_frequency(500.0, 0.60, 0.2386)
        assert far < near


class TestTransmissionLoss:
    """Equations (14) to (17), (22a) and (22b)."""

    @pytest.mark.parametrize("index", [1, 2, 3])
    def test_the_ring_frequency_matches(self, index: int) -> None:
        assert _run(index).pipe_ring_frequency == pytest.approx(14860.406, abs=5e-3)

    @pytest.mark.parametrize("index", [1, 2, 3])
    def test_the_reference_loss_matches(self, index: int) -> None:
        assert _run(index).reference_transmission_loss == pytest.approx(
            -44.71, abs=5e-3
        )

    def test_the_reference_loss_is_negative_and_added(self) -> None:
        # Both terms of Equation (15) are minus signs, which is why the
        # external level of Equation (18a) *adds* the transmission loss.
        result = _run(1)
        assert result.reference_transmission_loss < 0.0
        assert result.external_level < result.internal_level

    @pytest.mark.parametrize(
        ("index", "expected"), [(1, -29.56), (2, -27.13), (3, -31.45)]
    )
    def test_the_peak_correction_matches_equation_16b(
        self, index: int, expected: float
    ) -> None:
        result = _run(index)
        correction = (
            result.turbulent_transmission_loss - result.reference_transmission_loss
        )
        assert correction == pytest.approx(expected, abs=5e-3)

    @pytest.mark.parametrize(
        ("index", "expected"), [(1, -74.27), (2, -71.84), (3, -76.16)]
    )
    def test_the_turbulent_transmission_loss_matches(
        self, index: int, expected: float
    ) -> None:
        assert _run(index).turbulent_transmission_loss == pytest.approx(
            expected, abs=0.01
        )

    def test_the_correction_is_worst_far_from_the_ring_frequency(self) -> None:
        ring = 14860.406
        near = float(hydro.transmission_loss_correction(ring, ring)[0])
        low = float(hydro.transmission_loss_correction(50.0, ring)[0])
        high = float(hydro.transmission_loss_correction(20000.0, ring)[0])
        assert low < high < near

    def test_it_takes_a_whole_band_set_at_once(self) -> None:
        bands = np.array([500.0, 1000.0, 2000.0])
        found = hydro.transmission_loss_correction(bands, 14860.406)
        assert found.shape == bands.shape
        assert np.all(np.diff(found) > 0.0)

    @pytest.mark.parametrize(("index", "expected"), [(2, -62.86), (3, -74.92)])
    def test_the_cavitating_transmission_loss_matches_its_own_equation(
        self, index: int, expected: float
    ) -> None:
        # Table A.1 prints -62,917 and -75,006. Equation (17) with the
        # intermediates printed beside them gives these values instead, a
        # 0,06 to 0,08 dB offset whose origin is not visible on the page.
        # Recorded in ``docs/ERRATA.md``.
        found = _run(index).cavitation_transmission_loss
        assert found is not None
        assert found == pytest.approx(expected, abs=5e-3)
        printed = {2: -62.917, 3: -75.006}[index]
        assert abs(found - printed) < 0.1

    def test_the_cavitating_loss_is_the_one_the_external_level_uses(self) -> None:
        result = _run(2)
        assert result.transmission_loss == result.cavitation_transmission_loss
        assert _run(1).transmission_loss == _run(1).turbulent_transmission_loss

    def test_cavitation_makes_the_wall_more_transparent(self) -> None:
        result = _run(2)
        assert result.cavitation_transmission_loss is not None
        assert result.cavitation_transmission_loss > result.turbulent_transmission_loss

    def test_the_note_floors_the_efficiency_ratio_near_the_threshold(self) -> None:
        # Inside x_Fzp1 < x_F < x_Fzp1 + 0,1 the NOTE floors the ratio at
        # the value that makes the bracket exactly 1, so the cavitating loss
        # cannot fall below the turbulent one.
        floored = hydro.cavitation_transmission_loss(
            -71.84,
            turbulent_peak=654.35,
            cavitation_peak=1088.94,
            efficiency_ratio=1e-9,
            pressure_ratio=0.30,
            corrected_ratio=0.2386,
        )
        assert floored == pytest.approx(-71.84, abs=1e-9)

    def test_without_the_ratios_the_equation_is_read_as_printed(self) -> None:
        bare = hydro.cavitation_transmission_loss(
            -71.84,
            turbulent_peak=654.35,
            cavitation_peak=1088.94,
            efficiency_ratio=1e-9,
        )
        assert bare < -71.84

    def test_the_floor_does_not_apply_beyond_its_band(self) -> None:
        # 0,34 is just past 0,2386 + 0,1, so the NOTE has stopped applying
        # one hundredth of a ratio earlier.
        outside = hydro.cavitation_transmission_loss(
            -71.84,
            turbulent_peak=654.35,
            cavitation_peak=1088.94,
            efficiency_ratio=1e-9,
            pressure_ratio=0.34,
            corrected_ratio=0.2386,
        )
        assert outside < -71.84
        inside = hydro.cavitation_transmission_loss(
            -71.84,
            turbulent_peak=654.35,
            cavitation_peak=1088.94,
            efficiency_ratio=1e-9,
            pressure_ratio=0.33,
            corrected_ratio=0.2386,
        )
        assert inside == pytest.approx(-71.84, abs=1e-9)

    def test_the_third_column_stays_just_inside_the_floor_band(self) -> None:
        # x_F = 0,3508 against x_Fzp1 + 0,1 = 0,4324, and the ratio 0,00959
        # is above the floor 0,00722, which is what the annex prints twice.
        result = _run(3)
        assert result.corrected_ratio < result.pressure_ratio
        assert hydro.CAVITATION_FLOOR_WIDTH == pytest.approx(0.1)
        assert result.pressure_ratio < result.corrected_ratio + 0.1
        assert result.cavitation_efficiency is not None
        ratio = result.cavitation_efficiency / (
            result.turbulent_efficiency + result.cavitation_efficiency
        )
        assert ratio == pytest.approx(0.00959, abs=5e-5)
        floor = result.turbulent_peak**2 / (250.0 * result.cavitation_peak**1.5)
        assert floor == pytest.approx(0.00722, abs=5e-5)
        assert ratio > floor

    @pytest.mark.parametrize("index", [1, 2, 3])
    def test_the_band_loss_at_8_kHz_is_negative_51_76(self, index: int) -> None:
        # Table A.1 prints TL(8 000 Hz) = 51,76 dB with no minus sign in all
        # three columns, where its own inputs sum to -51,763. Recorded in
        # ``docs/ERRATA.md``; row (21) only closes with the negative value.
        assert _band(_run(index), _run(index).band_transmission_loss) == pytest.approx(
            -51.76, abs=5e-3
        )


class TestBandRoute:
    """Equations (19a), (19b), (20a), (20b) and (21)."""

    def test_the_default_bands_are_the_printed_range(self) -> None:
        bands = _run(1).frequency
        assert bands[0] == pytest.approx(50.0)
        assert bands[-1] == pytest.approx(20000.0)

    def test_a_band_set_can_be_given_instead(self) -> None:
        result = hydro.valve_hydrodynamic_noise(
            **COMMON,
            mass_flow=30.0,
            outlet_pressure=8.0e5,
            incipient_ratio=INCIPIENT,
            frequency=[63.0, 125.0, 250.0],
        )
        assert result.frequency.tolist() == [63.0, 125.0, 250.0]
        assert result.band_external_level.shape == (3,)

    @pytest.mark.parametrize(("index", "expected"), [(1, -33.34), (2, -29.69)])
    def test_the_turbulent_distribution_matches(
        self, index: int, expected: float
    ) -> None:
        result = _run(index)
        found = float(hydro.turbulent_distribution(BAND_HZ, result.turbulent_peak)[0])
        assert found == pytest.approx(expected, abs=0.01)

    def test_the_third_column_prints_a_distribution_its_peak_does_not_give(
        self,
    ) -> None:
        # -36,24 needs f_p,turb = 396,0 Hz; the column's own printed
        # 397,93 Hz gives -36,18. Recorded in ``docs/ERRATA.md``.
        found = float(hydro.turbulent_distribution(BAND_HZ, 397.93)[0])
        assert found == pytest.approx(-36.18, abs=5e-3)
        assert float(hydro.turbulent_distribution(BAND_HZ, 396.0)[0]) == pytest.approx(
            -36.24, abs=5e-3
        )

    def test_the_turbulent_distribution_peaks_near_its_peak_frequency(self) -> None:
        bands = np.array([100.0, 500.0, 1000.0, 8000.0])
        found = hydro.turbulent_distribution(bands, 500.0)
        assert float(np.argmax(found)) == 1.0

    def test_it_falls_faster_above_the_peak_than_below_it(self) -> None:
        # Equation (20a) is f^-1 below the peak and f^3 above it: away from
        # the peak an octave up costs three times what an octave down does.
        peak = 500.0
        below = float(hydro.turbulent_distribution(peak / 4.0, peak)[0])
        above = float(hydro.turbulent_distribution(peak * 4.0, peak)[0])
        centre = float(hydro.turbulent_distribution(peak, peak)[0])
        assert (centre - above) > 2.0 * (centre - below)

    @pytest.mark.parametrize(("index", "expected"), [(2, -10.51), (3, -6.85)])
    def test_the_cavitation_distribution_matches(
        self, index: int, expected: float
    ) -> None:
        peak = _run(index).cavitation_peak
        assert peak is not None
        found = float(hydro.cavitation_distribution(BAND_HZ, peak)[0])
        assert found == pytest.approx(expected, abs=5e-3)

    def test_the_cavitation_distribution_is_symmetric_about_its_maximum(self) -> None:
        # Both exponents are 1,5, so the hump is symmetric — but about
        # cbrt(4) f_p,cav, not about f_p,cav itself, because the quarter in
        # front of the rising branch shifts the maximum half an octave up.
        peak = 1000.0
        top = peak * 4.0 ** (1.0 / 3.0)
        low = float(hydro.cavitation_distribution(top / 4.0, peak)[0])
        high = float(hydro.cavitation_distribution(top * 4.0, peak)[0])
        assert low == pytest.approx(high, rel=1e-12)
        bands = np.geomspace(peak / 8.0, peak * 8.0, 401)
        found = hydro.cavitation_distribution(bands, peak)
        assert bands[int(np.argmax(found))] == pytest.approx(top, rel=0.02)

    def test_it_is_broader_than_the_turbulent_one(self) -> None:
        peak = 1000.0
        turbulent = float(hydro.turbulent_distribution(peak * 4.0, peak)[0])
        cavitating = float(hydro.cavitation_distribution(peak * 4.0, peak)[0])
        assert cavitating > turbulent

    @pytest.mark.parametrize(
        ("index", "expected"), [(1, 116.3), (2, 141.9), (3, 128.0)]
    )
    def test_the_internal_band_level_matches(self, index: int, expected: float) -> None:
        result = _run(index)
        assert _band(result, result.band_internal_level) == pytest.approx(
            expected, abs=0.05
        )

    @pytest.mark.parametrize(("index", "expected"), [(1, 51.8), (2, 77.4), (3, 63.6)])
    def test_the_external_band_level_matches(self, index: int, expected: float) -> None:
        result = _run(index)
        assert _band(result, result.band_external_level) == pytest.approx(
            expected, abs=0.05
        )

    def test_the_cavitating_spectrum_carries_both_humps(self) -> None:
        # Equation (19b) weights the two distributions by their share of the
        # sound power, so with a fraction of 0,377 the cavitation hump at
        # 1 089 Hz stands above the turbulent one at 654 Hz.
        bands = np.array([654.35, 1088.94])
        mixed = hydro.band_internal_levels(
            bands,
            156.543,
            turbulent_peak=654.35,
            cavitation_peak=1088.94,
            cavitation_fraction=0.377,
        )
        turbulent_only = hydro.band_internal_levels(
            bands, 156.543, turbulent_peak=654.35
        )
        assert mixed[1] > turbulent_only[1]

    def test_a_zero_fraction_is_the_turbulent_equation(self) -> None:
        bands = np.array([500.0, 1000.0])
        one = hydro.band_internal_levels(
            bands, 150.0, turbulent_peak=654.35, cavitation_peak=1088.94
        )
        two = hydro.band_internal_levels(bands, 150.0, turbulent_peak=654.35)
        assert np.allclose(one, two)


class TestWholeChain:
    """Equations (18a) and (18b), and what the three columns are for."""

    @pytest.mark.parametrize(
        ("index", "regime"),
        [(1, "turbulent"), (2, "cavitating"), (3, "cavitating")],
    )
    def test_each_column_lands_in_its_printed_regime(
        self, index: int, regime: str
    ) -> None:
        assert _run(index).regime == regime

    @pytest.mark.parametrize(("index", "expected"), [(1, 62.7), (2, 81.0), (3, 66.9)])
    def test_the_external_level_matches(self, index: int, expected: float) -> None:
        assert _run(index).external_level == pytest.approx(expected, abs=0.05)

    def test_the_shifted_threshold_costs_the_annexs_own_14_dB(self) -> None:
        # A.1's closing prose: shifting x_Fz by 0,1 moves the answer by about
        # 14 dB, which is why 4.2 asks for a measured value.
        assert _run(3).external_level - _run(2).external_level == pytest.approx(
            -14.1, abs=0.1
        )

    def test_the_spreading_term_is_the_printed_12_7_dB(self) -> None:
        result = _run(1)
        spreading = (
            result.internal_level + result.transmission_loss - result.external_level
        )
        assert spreading == pytest.approx(12.67, abs=5e-3)

    def test_a_thicker_wall_lets_less_out(self) -> None:
        thick = hydro.valve_hydrodynamic_noise(
            **{**COMMON, "wall_thickness": 0.010},
            mass_flow=30.0,
            outlet_pressure=8.0e5,
            incipient_ratio=INCIPIENT,
        )
        assert thick.external_level < _run(1).external_level

    def test_the_regime_is_decided_on_the_corrected_ratio(self) -> None:
        # (18a) is printed "for x_F <= x_Fz" and (18b) "for
        # x_Fzp1 < x_F <= 1", which divide the domain between them only at
        # the 6e5 Pa where those two thresholds are equal. 5.1 tests the
        # corrected one, and so does this chain. See ``docs/ERRATA.md``.
        ratio = 0.5 * (INCIPIENT + hydro.corrected_incipient_ratio(INCIPIENT, 1.0e6))
        outlet = 1.0e6 - ratio * (1.0e6 - 2.32e3)
        result = hydro.valve_hydrodynamic_noise(
            **COMMON,
            mass_flow=35.0,
            outlet_pressure=outlet,
            incipient_ratio=INCIPIENT,
        )
        assert result.corrected_ratio < result.pressure_ratio < INCIPIENT
        assert result.regime == "cavitating"

    def test_the_chain_stops_at_flashing(self) -> None:
        with pytest.raises(ValueError, match="flashes"):
            hydro.valve_hydrodynamic_noise(
                **{**COMMON, "vapour_pressure": 8.0e5},
                mass_flow=30.0,
                outlet_pressure=7.0e5,
                incipient_ratio=INCIPIENT,
            )


class TestMultistageTrim:
    """Clause 6: Equations (23a) to (29) and the seat diameter of 6.3.2 b)."""

    def test_equal_stages_split_the_differential_evenly(self) -> None:
        # The series law 1/C^2 = sum(1/C_i^2) makes three equal stages of a
        # C = 90 valve C_i = 90 sqrt(3), and each then takes a third of the
        # 6 bar the valve drops.
        stages = hydro.stage_conditions(
            inlet_pressure=1.0e6,
            outlet_pressure=4.0e5,
            vapour_pressure=2.32e3,
            stage_coefficients=[90.0 * math.sqrt(3.0)] * 3,
            flow_coefficient=90.0,
        )
        assert [s.inlet_pressure for s in stages] == pytest.approx(
            [1.0e6, 8.0e5, 6.0e5], abs=1.0
        )
        assert [s.outlet_pressure for s in stages] == pytest.approx(
            [8.0e5, 6.0e5, 4.0e5], abs=1.0
        )

    def test_the_first_stage_starts_at_the_valve_inlet(self) -> None:
        stages = hydro.stage_conditions(
            inlet_pressure=1.0e6,
            outlet_pressure=4.0e5,
            vapour_pressure=2.32e3,
            stage_coefficients=[110.0, 156.5],
            flow_coefficient=90.0,
        )
        assert stages[0].inlet_pressure == 1.0e6
        assert stages[-1].outlet_pressure == 4.0e5

    def test_the_pressure_falls_along_the_trim(self) -> None:
        # Equation (23b) is printed with p_1,i+1 on the right, which would
        # run the pressure the other way. See ``docs/ERRATA.md``.
        stages = hydro.stage_conditions(
            inlet_pressure=1.0e6,
            outlet_pressure=4.0e5,
            vapour_pressure=2.32e3,
            stage_coefficients=[130.0, 160.0, 199.1],
            flow_coefficient=90.0,
        )
        inlets = [s.inlet_pressure for s in stages]
        assert inlets == sorted(inlets, reverse=True)

    def test_an_increasing_flow_area_leaves_the_last_stage_the_least_to_do(
        self,
    ) -> None:
        # Three stages that satisfy the series law with increasing
        # capacities, which is the device of Figure 2.
        stages = hydro.stage_conditions(
            inlet_pressure=1.0e6,
            outlet_pressure=4.0e5,
            vapour_pressure=2.32e3,
            stage_coefficients=[130.0, 160.0, 199.1],
            flow_coefficient=90.0,
        )
        drops = [s.inlet_pressure - s.outlet_pressure for s in stages]
        assert drops[0] > drops[1] > drops[2]
        # The last stage takes whatever is left, and with the series law
        # satisfied that is exactly its own share.
        assert drops[2] == pytest.approx(6.0e5 / (199.1 / 90.0) ** 2, rel=1e-3)

    def test_each_stage_carries_its_own_pressure_ratio(self) -> None:
        stages = hydro.stage_conditions(
            inlet_pressure=1.0e6,
            outlet_pressure=4.0e5,
            vapour_pressure=2.32e3,
            stage_coefficients=[90.0 * math.sqrt(3.0)] * 3,
            flow_coefficient=90.0,
        )
        first = stages[0]
        assert first.pressure_ratio == pytest.approx(
            (1.0e6 - 8.0e5) / (1.0e6 - 2.32e3), rel=1e-9
        )
        # Equal drops, falling inlet pressures: the last stage works at the
        # highest ratio and is the one that cavitates first.
        assert stages[-1].pressure_ratio > first.pressure_ratio

    def test_it_refuses_stages_that_take_more_than_the_valve_has(self) -> None:
        with pytest.raises(ValueError, match="series law"):
            hydro.stage_conditions(
                inlet_pressure=1.0e6,
                outlet_pressure=4.0e5,
                vapour_pressure=2.32e3,
                stage_coefficients=[95.0, 95.0, 95.0],
                flow_coefficient=90.0,
            )

    def test_it_refuses_a_single_stage(self) -> None:
        with pytest.raises(ValueError, match="at least 2 values"):
            hydro.stage_conditions(
                inlet_pressure=1.0e6,
                outlet_pressure=4.0e5,
                vapour_pressure=2.32e3,
                stage_coefficients=[200.0],
                flow_coefficient=90.0,
            )

    def test_the_stage_levels_add_in_energy(self) -> None:
        assert hydro.combine_stage_levels(80.0, 80.0) == pytest.approx(
            83.0103, abs=5e-4
        )
        assert hydro.combine_stage_levels(80.0, 60.0, 60.0) == pytest.approx(
            80.0864, abs=5e-4
        )

    def test_the_sum_needs_more_than_one_stage(self) -> None:
        with pytest.raises(ValueError, match="at least 2 levels"):
            hydro.combine_stage_levels(80.0)

    def test_the_last_stage_differential_is_capped_by_its_own_threshold(self) -> None:
        # Equation (28) caps with x_Fzp1,n, not with F_L^2 as Equation (2)
        # does, and the cap is the smaller of the two.
        capped = hydro.last_stage_differential(
            inlet_pressure=6.0e5,
            outlet_pressure=4.0e5,
            vapour_pressure=2.32e3,
            corrected_ratio=0.25,
        )
        assert capped == pytest.approx(0.25 * (6.0e5 - 2.32e3), rel=1e-12)
        assert capped < 6.0e5 - 4.0e5

    def test_a_last_stage_below_its_threshold_keeps_its_differential(self) -> None:
        kept = hydro.last_stage_differential(
            inlet_pressure=6.0e5,
            outlet_pressure=5.5e5,
            vapour_pressure=2.32e3,
            corrected_ratio=0.25,
        )
        assert kept == pytest.approx(5.0e4, rel=1e-12)

    def test_the_style_modifier_of_uniform_openings(self) -> None:
        assert hydro.uniform_passage_style_modifier(1) == pytest.approx(1.0)
        assert hydro.uniform_passage_style_modifier(16) == pytest.approx(0.25)

    def test_the_seat_diameter_comes_out_in_millimetres(self) -> None:
        # Clause 3 declares d_o in metres and this formula returns tens for
        # any real last stage. See ``docs/ERRATA.md``: it is millimetres.
        assert hydro.last_stage_seat_diameter_mm(90.0) == pytest.approx(53.36, abs=5e-3)
        assert hydro.last_stage_seat_diameter_mm(
            90.0, coefficient="Kv"
        ) == pytest.approx(49.332, abs=5e-3)

    def test_the_seat_diameter_is_of_the_order_of_the_pipe_in_millimetres(self) -> None:
        # A DN 100 valve's last stage cannot be 53 m across; in millimetres
        # it is half the bore, which is what a last stage looks like.
        assert 0.2 < hydro.last_stage_seat_diameter_mm(90.0) / 100.0 < 1.0


class TestGuards:
    """The ranges each equation is written for."""

    def test_it_refuses_a_valve_that_raises_the_pressure(self) -> None:
        with pytest.raises(ValueError, match="drops pressure"):
            hydro.differential_pressure_ratio(
                inlet_pressure=8.0e5, outlet_pressure=1.0e6, vapour_pressure=2.32e3
            )

    def test_it_refuses_an_inlet_at_the_vapour_pressure(self) -> None:
        with pytest.raises(ValueError, match="vapour pressure"):
            hydro.differential_pressure_ratio(
                inlet_pressure=1.0e6, outlet_pressure=8.0e5, vapour_pressure=1.0e6
            )

    def test_the_differential_refuses_the_same_two_things(self) -> None:
        with pytest.raises(ValueError, match="drops pressure"):
            hydro.cavitation_differential(
                inlet_pressure=8.0e5,
                outlet_pressure=1.0e6,
                vapour_pressure=2.32e3,
                pressure_recovery=0.92,
            )
        with pytest.raises(ValueError, match="vapour pressure"):
            hydro.cavitation_differential(
                inlet_pressure=1.0e6,
                outlet_pressure=8.0e5,
                vapour_pressure=1.2e6,
                pressure_recovery=0.92,
            )

    @pytest.mark.parametrize("bad", [0.0, -0.5, 1.5, math.nan, math.inf])
    def test_it_refuses_a_recovery_factor_outside_its_range(self, bad: float) -> None:
        with pytest.raises(ValueError, match="pressure_recovery"):
            hydro.vena_contracta_velocity(2.0e5, 997.0, bad)

    def test_the_cavitating_efficiency_refuses_the_turbulent_side(self) -> None:
        with pytest.raises(ValueError, match="cavitating branch"):
            hydro.cavitation_efficiency(
                turbulent=2.0e-6,
                differential=1.0e5,
                choked_differential=1.0e5,
                pressure_ratio=0.10,
                corrected_ratio=0.2386,
            )

    def test_the_cavitating_efficiency_refuses_flashing(self) -> None:
        with pytest.raises(ValueError, match="divides by 1 - x_F"):
            hydro.cavitation_efficiency(
                turbulent=2.0e-6,
                differential=1.0e5,
                choked_differential=1.0e5,
                pressure_ratio=1.0,
                corrected_ratio=0.2386,
            )

    def test_the_cavitating_peak_refuses_flashing(self) -> None:
        # Equation (13) is finite there; the sound power it belongs to is
        # not, which is the reason the message gives.
        with pytest.raises(ValueError, match="divides by 1 - x_F"):
            hydro.cavitation_peak_frequency(500.0, 1.2, 0.2386)

    @pytest.mark.parametrize("bad", ["cv", "Kv2", ""])
    def test_it_refuses_a_flow_coefficient_it_has_no_constant_for(
        self, bad: str
    ) -> None:
        with pytest.raises(ValueError, match="'coefficient' must be one of"):
            hydro.incipient_cavitation_ratio(90.0, 0.42, 0.92, coefficient=bad)

    def test_it_refuses_a_printing_of_equation_12_that_does_not_exist(self) -> None:
        with pytest.raises(ValueError, match="'form' must be one of"):
            hydro.jet_strouhal_number(
                flow_coefficient=90.0,
                style_modifier=0.42,
                pressure_recovery=0.92,
                corrected_ratio=0.2386,
                valve_diameter=0.1,
                seat_diameter=0.1,
                inlet_pressure=1.0e6,
                vapour_pressure=2.32e3,
                form="table",
            )

    @pytest.mark.parametrize("bad", [0, -3, 2.5])
    def test_it_refuses_a_passage_count_that_is_not_a_count(self, bad: float) -> None:
        with pytest.raises(ValueError, match="whole number"):
            hydro.uniform_passage_style_modifier(bad)  # type: ignore[arg-type]

    def test_it_refuses_a_cavitation_fraction_outside_zero_to_one(self) -> None:
        with pytest.raises(ValueError, match="between 0 and 1"):
            hydro.band_internal_levels(
                [500.0], 150.0, turbulent_peak=500.0, cavitation_fraction=1.5
            )

    def test_it_refuses_the_cavitating_branch_without_its_peak(self) -> None:
        with pytest.raises(ValueError, match="'cavitation_peak'"):
            hydro.band_internal_levels(
                [500.0], 150.0, turbulent_peak=500.0, cavitation_fraction=0.3
            )

    def test_the_floor_needs_both_ratios_or_neither(self) -> None:
        with pytest.raises(ValueError, match="or neither"):
            hydro.cavitation_transmission_loss(
                -70.0,
                turbulent_peak=650.0,
                cavitation_peak=1090.0,
                efficiency_ratio=0.3,
                pressure_ratio=0.3,
            )

    @pytest.mark.parametrize("bad", [25.0, 100.0])
    def test_it_refuses_a_power_ratio_read_as_a_percentage(self, bad: float) -> None:
        # Table 2 prints 0,25 and 0,5. Read as percentages they would add
        # 20 dB to the answer without a word.
        with pytest.raises(ValueError, match="not percentages"):
            hydro.valve_hydrodynamic_noise(
                **{**COMMON, "power_ratio": bad},
                mass_flow=30.0,
                outlet_pressure=8.0e5,
                incipient_ratio=INCIPIENT,
            )

    def test_the_cavitating_loss_refuses_a_ratio_above_one(self) -> None:
        with pytest.raises(ValueError, match="not percentages"):
            hydro.cavitation_transmission_loss(
                -71.84,
                turbulent_peak=654.35,
                cavitation_peak=1088.94,
                efficiency_ratio=37.7,
            )

    def test_it_refuses_a_jet_faster_than_sound_in_the_liquid(self) -> None:
        # The arguments the other way round: 1 400 m/s of jet in a liquid
        # that carries sound at 28,8.
        with pytest.raises(ValueError, match="speed of sound in the liquid"):
            hydro.turbulent_efficiency(1400.0, 28.801)

    def test_the_stages_are_expected_to_close_the_series_law(self) -> None:
        # Three stages of C_i = 900 on a C = 90 valve account for 3 % of its
        # resistance, and Equation (24b) then hands the whole remainder to
        # the last stage, whose pressure ratio is the number 6.3 tests.
        with pytest.warns(ValveNoiseWarning, match="series law"):
            hydro.stage_conditions(
                inlet_pressure=1.0e6,
                outlet_pressure=4.0e5,
                vapour_pressure=2.32e3,
                stage_coefficients=[900.0, 900.0, 900.0],
                flow_coefficient=90.0,
            )

    @pytest.mark.parametrize("bad", [math.nan, math.inf])
    def test_it_refuses_levels_that_are_not_levels(self, bad: float) -> None:
        with pytest.raises(ValueError, match="finite"):
            hydro.band_internal_levels([500.0], bad, turbulent_peak=500.0)
        with pytest.raises(ValueError, match="finite"):
            hydro.cavitation_transmission_loss(
                bad,
                turbulent_peak=650.0,
                cavitation_peak=1090.0,
                efficiency_ratio=0.3,
            )

    @pytest.mark.parametrize("bad", [1.0, 1.5, 25.0])
    def test_the_correction_refuses_a_threshold_at_or_above_one(
        self, bad: float
    ) -> None:
        # Equations (3a) and (3b) cap x_Fz at 0,90 and 0,47, and (3c) only
        # lowers it. A value of 1 or more is a percentage that lost its
        # division, and it would take the bracket (1 - x_Fzp1)/(1 - x_F) of
        # Equation (9) negative.
        with pytest.raises(ValueError, match="not a percentage"):
            hydro.corrected_incipient_ratio(bad, 1.0e6)

    def test_the_cavitating_peak_refuses_one_too(self) -> None:
        with pytest.raises(ValueError, match="not a percentage"):
            hydro.cavitation_peak_frequency(500.0, 0.5, 1.5)

    def test_the_chain_refuses_it_too(self) -> None:
        with pytest.raises(ValueError, match="not a percentage"):
            hydro.valve_hydrodynamic_noise(
                **COMMON, mass_flow=30.0, outlet_pressure=8.0e5, incipient_ratio=25.0
            )
