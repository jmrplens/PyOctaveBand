#  Copyright (c) 2026. Jose Manuel Requena Plens
"""IEC 60534-8-3:2010, Clause 5, against the seven examples of Annex A.

Annex A prints every intermediate of six operating points of one valve, one
per regime with regime V used twice, and a seventh example on a different
valve whose pipe exercises the transmission loss over all 33 bands. Those
printed columns are the oracle here: each test names the equation whose value
it pins.

Two of the annex's own printed values do not reproduce themselves and are
recorded in ``docs/ERRATA.md``; the fixtures below carry the values that make
the annex self-consistent and say so where they do.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from phonometry.noise_control import valves

#: The annex prints the piping geometry factor as 0,98, but every printed
#: vena contracta pressure needs 0,984. See ``docs/ERRATA.md``.
PRESSURE_RECOVERY = 0.792 / 0.984

#: Table 4, "Globe, ported cage design": the valve the six examples use.
EFFICIENCY_CORRECTION = -3.8
STROUHAL = 0.2

#: The shared given data of A.2, in SI units.
COMMON: dict[str, Any] = {
    "inlet_pressure": 1.0e6,
    "inlet_density": 5.3,
    "inlet_temperature": 450.0,
    "specific_heat_ratio": 1.22,
    "molecular_mass": 19.8,
    "pressure_recovery": PRESSURE_RECOVERY,
    "wall_thickness": 0.008,
    "pipe_density": 8000.0,
    "efficiency_correction": EFFICIENCY_CORRECTION,
    "strouhal_number": STROUHAL,
}

#: The per-example given data of Table A.1, and the regime each one lands in.
EXAMPLES = [
    {
        "example": 1,
        "mass_flow": 2.22,
        "outlet_pressure": 7.2e5,
        "flow_coefficient": 90.0,
        "valve_outlet_diameter": 0.1,
        "internal_diameter": 0.2031,
        "regime": 1,
    },
    {
        "example": 2,
        "mass_flow": 2.29,
        "outlet_pressure": 6.9e5,
        "flow_coefficient": 90.0,
        "valve_outlet_diameter": 0.1,
        "internal_diameter": 0.2031,
        "regime": 2,
    },
    {
        "example": 3,
        "mass_flow": 2.59,
        "outlet_pressure": 4.8e5,
        "flow_coefficient": 90.0,
        "valve_outlet_diameter": 0.1,
        "internal_diameter": 0.2031,
        "regime": 3,
    },
    {
        "example": 4,
        "mass_flow": 1.18,
        "outlet_pressure": 4.2e5,
        "flow_coefficient": 40.0,
        "valve_outlet_diameter": 0.2031,
        "internal_diameter": 0.2031,
        "regime": 4,
    },
    {
        "example": 5,
        "mass_flow": 1.19,
        "outlet_pressure": 5.0e4,
        "flow_coefficient": 40.0,
        "valve_outlet_diameter": 0.2031,
        "internal_diameter": 0.2031,
        "regime": 5,
    },
    {
        "example": 6,
        "mass_flow": 0.89,
        "outlet_pressure": 5.0e4,
        "flow_coefficient": 30.0,
        "valve_outlet_diameter": 0.1,
        "internal_diameter": 0.15,
        "regime": 5,
    },
]


def _style_modifier() -> float:
    """A.2's cage: six passages of 0,00137 m² with a 0,181 m wetted perimeter."""
    return valves.valve_style_modifier(0.00137, 0.181, 6)


def _run(index: int) -> valves.AerodynamicValveNoise:
    """Example ``index`` of Table A.1, one to six."""
    case = dict(EXAMPLES[index - 1])
    case.pop("example")
    case.pop("regime")
    return valves.valve_aerodynamic_noise(
        **COMMON, **case, style_modifier=_style_modifier()
    )


class TestRegimeBoundaries:
    """Equations (3) to (7), which are the same for all six examples."""

    def test_the_four_boundaries_are_the_printed_ones(self) -> None:
        found = valves.pressure_ratio_boundaries(1.22, PRESSURE_RECOVERY)
        assert found.vena_contracta == pytest.approx(0.439, abs=5e-4)
        assert found.critical == pytest.approx(0.285, abs=5e-4)
        assert found.recovery == pytest.approx(0.784, abs=5e-4)
        assert found.break_point == pytest.approx(0.576, abs=5e-4)
        assert found.constant_efficiency == pytest.approx(0.942, abs=5e-4)

    def test_they_run_in_the_order_the_clause_lists_them(self) -> None:
        found = valves.pressure_ratio_boundaries(1.22, PRESSURE_RECOVERY)
        assert found.critical < found.vena_contracta < found.break_point
        assert found.break_point < found.constant_efficiency < 1.0

    def test_the_fifth_regime_starts_above_its_boundary_and_not_at_it(
        self,
    ) -> None:
        # Clause 5.2 prints "x_CE < x" and Table 3 prints "x_CE <= x", which
        # would put the boundary itself in two regimes. The clause governs.
        found = valves.pressure_ratio_boundaries(1.22, PRESSURE_RECOVERY)
        assert valves.flow_regime(found.constant_efficiency, found) == 4
        assert valves.flow_regime(found.constant_efficiency + 1e-9, found) == 5

    def test_a_higher_recovery_factor_chokes_the_valve_later(self) -> None:
        low = valves.pressure_ratio_boundaries(1.22, 0.6)
        high = valves.pressure_ratio_boundaries(1.22, 0.9)
        assert high.critical > low.critical

    @pytest.mark.parametrize("bad", [0.0, -1.0, 1.0, math.nan, math.inf])
    def test_it_refuses_a_heat_ratio_that_is_not_above_one(self, bad: float) -> None:
        with pytest.raises(ValueError, match="above 1|must be positive"):
            valves.pressure_ratio_boundaries(bad, 0.8)

    @pytest.mark.parametrize("bad", [0.0, -0.5, 1.5, math.nan])
    def test_it_refuses_a_pressure_ratio_outside_the_open_unit_interval(
        self, bad: float
    ) -> None:
        found = valves.pressure_ratio_boundaries(1.22, PRESSURE_RECOVERY)
        with pytest.raises(ValueError, match="strictly between 0 and 1"):
            valves.flow_regime(bad, found)


class TestGeometry:
    """Equations (8a) to (9)."""

    def test_the_style_modifier_is_the_printed_three_tenths(self) -> None:
        # (8b) gives d_H = 0,030 m and (8c) gives d_o = 0,102 m, which the
        # annex prints as 0,010 m: see docs/ERRATA.md. F_d = 0,30 either way
        # only if d_o is the larger value.
        assert _style_modifier() == pytest.approx(0.30, abs=5e-3)

    def test_the_orifice_diameter_is_the_one_that_makes_the_annex_close(
        self,
    ) -> None:
        orifice = math.sqrt(4.0 * 6 * 0.00137 / math.pi)
        assert orifice == pytest.approx(0.102, abs=5e-4)
        assert 4.0 * 0.00137 / 0.181 / orifice == pytest.approx(0.30, abs=5e-3)

    @pytest.mark.parametrize(
        ("index", "expected"),
        [(1, 0.012), (2, 0.012), (3, 0.012), (4, 0.008), (5, 0.008), (6, 0.007)],
    )
    def test_the_jet_diameter_matches_every_column(
        self, index: int, expected: float
    ) -> None:
        assert _run(index).jet_diameter == pytest.approx(expected, abs=5e-4)

    def test_the_two_flow_coefficients_agree_to_the_rounding_of_table_one(
        self,
    ) -> None:
        # The same valve rated in K_v and in C_v does not give exactly the
        # same jet: Table 1 rounds both constants to two digits, and
        # 4,9/4,6 = 1,065 against the sqrt(1,156) = 1,075 the conversion
        # asks for. The gap is 1 %, which is the rounding and not a choice.
        as_cv = valves.jet_diameter(90.0, 0.3, 0.8, coefficient="Cv")
        as_kv = valves.jet_diameter(90.0 / 1.156, 0.3, 0.8, coefficient="Kv")
        assert as_kv == pytest.approx(as_cv, rel=0.01)
        assert as_kv < as_cv

    def test_it_refuses_a_coefficient_the_table_does_not_print(self) -> None:
        with pytest.raises(ValueError, match="coefficient"):
            valves.jet_diameter(90.0, 0.3, 0.8, coefficient="Av")

    @pytest.mark.parametrize("bad", [0, -3, 2.5])
    def test_it_refuses_a_passage_count_that_is_not_a_whole_number(
        self, bad: float
    ) -> None:
        with pytest.raises(ValueError, match="whole number"):
            valves.valve_style_modifier(0.00137, 0.181, bad)  # type: ignore[arg-type]


class TestPrintedExamples:
    """Table A.1, column by column and equation by equation."""

    @pytest.mark.parametrize("case", EXAMPLES, ids=lambda c: f"example{c['example']}")
    def test_each_example_lands_in_its_printed_regime(
        self, case: dict[str, float]
    ) -> None:
        assert _run(int(case["example"])).regime == case["regime"]

    @pytest.mark.parametrize(
        ("index", "expected"),
        [(1, 0.28), (2, 0.31), (3, 0.52), (4, 0.58), (5, 0.95), (6, 0.95)],
    )
    def test_the_pressure_ratio_matches(self, index: int, expected: float) -> None:
        assert _run(index).pressure_ratio == pytest.approx(expected, abs=5e-3)

    @pytest.mark.parametrize(
        ("index", "expected"),
        [
            (1, 567787.0),
            (2, 521478.0),
            (3, 197319.0),
            (4, 104702.0),
            (5, -466437.0),
            (6, -466437.0),
        ],
    )
    def test_the_vena_contracta_pressure_matches(
        self, index: int, expected: float
    ) -> None:
        # Examples 5 and 6 print it negative: Equation (2) is being read far
        # past the choking point, where it has stopped meaning a pressure.
        assert _run(index).vena_contracta_pressure == pytest.approx(expected, abs=2.0)

    @pytest.mark.parametrize(
        ("index", "expected"),
        [(1, 22.3), (2, 30.4), (3, 141.3), (4, 86.1), (5, 291.9), (6, 218.3)],
    )
    def test_the_sound_power_matches(self, index: int, expected: float) -> None:
        assert _run(index).sound_power == pytest.approx(expected, abs=0.05)

    @pytest.mark.parametrize(
        ("index", "expected"),
        [(1, 0.26), (2, 0.29), (3, 0.47), (4, 0.24), (5, 2.4), (6, 2.5)],
    )
    def test_the_velocity_correction_matches(self, index: int, expected: float) -> None:
        assert _run(index).velocity_correction == pytest.approx(expected, abs=0.05)

    @pytest.mark.parametrize(
        ("index", "expected"),
        [(1, 155.3), (2, 156.5), (3, 161.7), (4, 158.8), (5, 157.0), (6, 158.4)],
    )
    def test_the_internal_level_matches(self, index: int, expected: float) -> None:
        assert _run(index).internal_level == pytest.approx(expected, abs=0.05)

    def test_the_sixth_example_runs_past_the_outlet_limit(self) -> None:
        # M_o = 0,89 against the 0,3 of NOTE 1 to Equation (15): Clause 5 is
        # outside its own validity and the annex prints the result anyway.
        assert _run(6).outlet_mach == pytest.approx(0.89, abs=5e-3)
        assert _run(6).outlet_mach > valves.MACH_LIMIT_STANDARD_TRIM

    def test_the_pipe_mach_number_is_clipped_before_the_correction(self) -> None:
        # Example 6 computes M_2 = 0,4 and NOTE 2 limits it to 0,3, which is
        # the value the printed L_g of 2,5 dB comes from.
        result = _run(6)
        assert result.pipe_mach == pytest.approx(0.4, abs=5e-3)
        clipped = 16.0 * math.log10(1.0 / (1.0 - valves.PIPE_WALL_MACH_LIMIT))
        assert result.velocity_correction == pytest.approx(clipped, abs=1e-9)

    def test_a_regime_five_valve_is_louder_than_a_subsonic_one(self) -> None:
        assert _run(5).acoustical_efficiency > _run(1).acoustical_efficiency


class TestInternalSpectrum:
    """Equation (19)."""

    def test_the_band_levels_peak_near_the_peak_frequency(self) -> None:
        result = _run(1)
        loudest = result.frequency[int(np.argmax(result.band_internal_level))]
        assert loudest == pytest.approx(result.peak_frequency, rel=0.35)

    def test_the_spectrum_falls_off_faster_above_the_peak_than_below(self) -> None:
        # The two exponents are 2,5 above and 1,7 below, so a band an octave
        # above the peak loses more than one an octave below it.
        peak = 1000.0
        bands = np.array([500.0, 2000.0])
        levels = valves.internal_spectrum(100.0, peak, bands)
        assert levels[1] < levels[0]

    def test_it_takes_eight_decibels_off_at_the_peak(self) -> None:
        at_peak = valves.internal_spectrum(150.0, 1000.0, np.array([1000.0]))
        shape = 10.0 * math.log10((1.0 + 0.5**2.5) * (1.0 + 0.5**1.7))
        assert at_peak[0] == pytest.approx(150.0 - 8.0 - shape, abs=1e-9)

    @pytest.mark.parametrize("bad", [0.0, -1.0, math.nan, math.inf])
    def test_it_refuses_a_peak_frequency_that_is_not_a_frequency(
        self, bad: float
    ) -> None:
        with pytest.raises(ValueError, match="peak_frequency"):
            valves.internal_spectrum(150.0, bad, np.array([1000.0]))


class TestPipeTransmission:
    """Equations (20a) to (23), against example 7's pipe.

    Example 7 runs a DN 200 pipe with 8 mm walls and a downstream sonic
    velocity of 408 m/s, and prints the transmission loss in all 33 bands.
    The loss depends on the pipe and on the fluid leaving the valve, not on
    the trim, so it can be checked on its own.
    """

    PIPE = {
        "internal_diameter": 0.200,
        "wall_thickness": 0.008,
        "valve_outlet_diameter": 0.200,
        "downstream_density": 11.1,
        "downstream_sound_speed": 408.0,
        "pipe_density": 8000.0,
    }

    #: Table A.2's printed TL, bands 1 to 24, in dB.
    PRINTED = (
        -94.1, -92.0, -90.0, -88.1, -86.1, -84.1, -82.2, -80.2,
        -78.1, -76.2, -74.3, -72.2, -70.4, -68.5, -66.5, -64.5,
        -62.6, -60.7, -58.7, -56.9, -55.1, -53.0, -51.2, -49.4,
    )  # fmt: skip

    def test_the_three_pipe_frequencies_are_the_printed_ones(self) -> None:
        pipe = valves.coincidence_frequencies(0.200, 0.008, 408.0)
        assert pipe.ring == pytest.approx(7958.0, abs=1.0)
        assert pipe.internal_coincidence == pytest.approx(2365.0, abs=1.5)
        assert pipe.external_coincidence == pytest.approx(1622.0, abs=1.0)

    def test_the_transmission_loss_matches_every_printed_band(self) -> None:
        bands = np.asarray(
            valves.valve_aerodynamic_noise(
                **COMMON,
                mass_flow=2.22,
                outlet_pressure=7.2e5,
                flow_coefficient=90.0,
                valve_outlet_diameter=0.1,
                internal_diameter=0.2031,
                style_modifier=_style_modifier(),
            ).frequency
        )
        loss = valves.pipe_transmission_loss(bands, **self.PIPE)
        for index, expected in enumerate(self.PRINTED):
            assert loss[index] == pytest.approx(expected, abs=0.1), bands[index]

    def test_a_wide_pipe_takes_no_damping_and_a_narrow_one_takes_nine(
        self,
    ) -> None:
        bands = np.array([1000.0])
        wide = valves.pipe_transmission_loss(
            bands, **{**self.PIPE, "valve_outlet_diameter": 0.2}
        )
        narrow = valves.pipe_transmission_loss(
            bands, **{**self.PIPE, "valve_outlet_diameter": 0.04}
        )
        assert wide[0] - narrow[0] == pytest.approx(9.0, abs=1e-9)

    def test_the_damping_cubic_meets_its_floor_and_not_its_ceiling(self) -> None:
        # Equation (20b) is a cubic between 0,05 m and 0,15 m and a constant
        # on each side. At the bottom the pieces meet, to within the rounding
        # of the printed coefficients: the cubic gives 8,99 dB against the
        # 9 dB below it. At the top they do not: the cubic gives 0,95 dB and
        # the branch above it gives nothing at all, so a 0,15 m outlet and a
        # 0,151 m one are a decibel apart.
        bands = np.array([1000.0])
        base = valves.pipe_transmission_loss(
            bands, **{**self.PIPE, "valve_outlet_diameter": 0.2}
        )[0]

        def damping(diameter: float) -> float:
            return float(
                base
                - valves.pipe_transmission_loss(
                    bands, **{**self.PIPE, "valve_outlet_diameter": diameter}
                )[0]
            )

        assert damping(0.05) == pytest.approx(8.99, abs=0.02)
        assert damping(0.15) == pytest.approx(0.95, abs=0.02)
        assert damping(0.1501) == pytest.approx(0.0, abs=1e-9)

    @pytest.mark.parametrize("bad", [0.0, -1.0])
    def test_it_refuses_a_wall_thickness_that_is_not_a_thickness(
        self, bad: float
    ) -> None:
        with pytest.raises(ValueError, match="wall_thickness"):
            valves.pipe_transmission_loss(
                np.array([1000.0]), **{**self.PIPE, "wall_thickness": bad}
            )


class TestWholeChain:
    """What the caller of :func:`valve_aerodynamic_noise` gets."""

    def test_the_result_carries_all_thirty_three_bands(self) -> None:
        result = _run(1)
        assert result.frequency.shape == (33,)
        assert result.band_internal_level.shape == (33,)
        assert result.band_transmission_loss.shape == (33,)
        assert result.band_external_level.shape == (33,)
        assert result.frequency[0] == pytest.approx(12.5)
        assert result.frequency[-1] == pytest.approx(20000.0)

    def test_the_a_weighting_table_covers_the_same_bands(self) -> None:
        assert len(valves.AERODYNAMIC_A_WEIGHTING_DB) == 33
        assert valves.AERODYNAMIC_A_WEIGHTING_DB[19] == 0.0

    def test_the_external_level_is_far_below_the_internal_one(self) -> None:
        result = _run(1)
        assert result.external_level < result.internal_level - 40.0

    @pytest.mark.parametrize(
        ("index", "expected"),
        [(1, 92), (2, 93), (3, 98), (4, 94), (5, 97)],
    )
    def test_the_level_at_one_metre_matches_the_printed_answer(
        self, index: int, expected: int
    ) -> None:
        # Equation (25), the end of the chain. The sixth column is missing on
        # purpose: its valve outlet is narrower than its pipe, so the annex
        # adds the expander noise of Clause 7 to it, which Clause 5 does not
        # know about and this module does not implement.
        assert round(_run(index).external_level) == expected

    def test_the_sixth_example_is_the_one_with_an_expander(self) -> None:
        # 93 dB(A) from the valve alone against the 94 the annex prints, which
        # is the valve and the expander together.
        assert round(_run(6).external_level) == 93
        assert EXAMPLES[5]["valve_outlet_diameter"] < EXAMPLES[5]["internal_diameter"]

    def test_the_pipe_wall_is_what_the_level_outside_depends_on(self) -> None:
        thin = valves.valve_aerodynamic_noise(
            **{**COMMON, "wall_thickness": 0.004},
            mass_flow=2.22,
            outlet_pressure=7.2e5,
            flow_coefficient=90.0,
            valve_outlet_diameter=0.1,
            internal_diameter=0.2031,
            style_modifier=_style_modifier(),
        )
        assert thin.external_level > _run(1).external_level

    def test_the_sound_power_level_is_the_power_in_decibels(self) -> None:
        result = _run(1)
        assert result.sound_power_level == pytest.approx(
            10.0 * math.log10(result.sound_power / 1e-12), abs=1e-9
        )

    def test_it_refuses_a_valve_that_does_not_drop_pressure(self) -> None:
        with pytest.raises(ValueError, match="drops pressure"):
            valves.valve_aerodynamic_noise(
                **COMMON,
                mass_flow=2.22,
                outlet_pressure=1.2e6,
                flow_coefficient=90.0,
                valve_outlet_diameter=0.1,
                internal_diameter=0.2031,
                style_modifier=_style_modifier(),
            )


class TestPrintedTables:
    """The tables the module carries verbatim."""

    def test_table_four_carries_every_printed_style(self) -> None:
        assert len(valves.VALVE_ACOUSTIC_STYLES) == 13
        assert valves.VALVE_ACOUSTIC_STYLES["globe ported cage"] == (-3.8, 0.2)
        assert valves.VALVE_ACOUSTIC_STYLES["expander"] == (-3.0, 0.2)

    def test_every_correction_factor_is_negative(self) -> None:
        # A_eta is the exponent of a small number: -4 is the pure dipole of a
        # free jet, and every printed valve sits between -4,8 and -3,0.
        corrections = [pair[0] for pair in valves.VALVE_ACOUSTIC_STYLES.values()]
        assert max(corrections) == pytest.approx(-3.0)
        assert min(corrections) == pytest.approx(-4.8)

    def test_the_strouhal_numbers_sit_in_the_printed_range(self) -> None:
        # The clause puts St_p between 0,1 and 0,3 for free jets.
        numbers = [pair[1] for pair in valves.VALVE_ACOUSTIC_STYLES.values()]
        assert min(numbers) >= 0.1
        assert max(numbers) <= 0.3

    def test_table_one_holds_both_flow_coefficients(self) -> None:
        assert valves.FLOW_COEFFICIENT_CONSTANTS == {"Cv": 4.6e-3, "Kv": 4.9e-3}
