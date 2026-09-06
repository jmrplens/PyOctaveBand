#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Tests for the printed tables and reporting rules of ISO 3382-1:2009.

Validation strategy: these are transcribed tables and averaging rules, so
the tests are exact lookups and exact arithmetic. Three of them are about
the traps rather than the values: Table A.1 does not average every quantity
over the same bands, only one of its rows is energy averaged, and only one
of its just-noticeable differences is relative. A fourth pins that Table A.2
lies on a straight line in the logarithm of the seat count, which is what
lets a hall between its rows be answered at all.
"""

from __future__ import annotations

import numpy as np
import pytest

from phonometry import room

OCTAVES = np.array([125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0])


class TestTableA1:
    """The seven rows, and what differs between them."""

    def test_it_holds_the_seven_printed_quantities(self) -> None:
        assert set(room.TABLE_A1) == {
            "G",
            "EDT",
            "C80",
            "D50",
            "Ts",
            "J_LF",
            "L_J",
        }

    def test_the_five_listener_aspects_are_the_printed_grouping(self) -> None:
        # Five aspects over seven quantities: "Perceived clarity of sound"
        # carries three of them and the other four carry one each.
        aspects = [row.aspect for row in room.TABLE_A1.values()]
        assert len(set(aspects)) == 5
        assert aspects.count("Perceived clarity of sound") == 3

    def test_the_averaging_bands_are_not_the_same_for_every_row(self) -> None:
        # Two bands for the five monaural quantities, four for the lateral
        # ones. An accessor that hard-codes the mid pair is wrong for half
        # the table, which is why A.5 prints both cases as examples.
        assert room.TABLE_A1["G"].averaging_bands_hz == (500.0, 1000.0)
        assert room.TABLE_A1["Ts"].averaging_bands_hz == (500.0, 1000.0)
        assert room.TABLE_A1["J_LF"].averaging_bands_hz == (
            125.0,
            250.0,
            500.0,
            1000.0,
        )
        assert room.TABLE_A1["L_J"].averaging_bands_hz == (
            125.0,
            250.0,
            500.0,
            1000.0,
        )

    def test_only_the_late_lateral_level_is_energy_averaged(self) -> None:
        energetic = [s for s, row in room.TABLE_A1.items() if row.energy_averaged]
        assert energetic == ["L_J"]

    def test_only_the_early_decay_time_has_a_relative_jnd(self) -> None:
        relative = [s for s, row in room.TABLE_A1.items() if row.relative_jnd]
        assert relative == ["EDT"]
        assert room.TABLE_A1["EDT"].just_noticeable_difference == 0.05

    def test_the_late_lateral_level_has_no_jnd_at_all(self) -> None:
        # The table prints "Not known", which is neither zero nor 1 dB by
        # analogy with the other decibel quantities.
        assert room.TABLE_A1["L_J"].just_noticeable_difference is None

    @pytest.mark.parametrize(
        ("symbol", "jnd", "low", "high"),
        [
            ("G", 1.0, -2.0, 10.0),
            ("C80", 1.0, -5.0, 5.0),
            ("D50", 0.05, 0.3, 0.7),
            ("Ts", 0.010, 0.060, 0.260),
            ("J_LF", 0.05, 0.05, 0.35),
        ],
    )
    def test_the_printed_values_are_transcribed(
        self, symbol: str, jnd: float, low: float, high: float
    ) -> None:
        row = room.TABLE_A1[symbol]
        assert row.just_noticeable_difference == pytest.approx(jnd)
        assert row.typical_range == pytest.approx((low, high))


class TestSingleNumberAverage:
    """Footnote a of Table A.1, and the two examples of A.5."""

    def test_the_strength_averages_the_two_mid_octaves(self) -> None:
        values = np.array([2.0, 3.0, 4.0, 6.0, 5.0, 4.5])
        assert room.single_number_average("G", values, OCTAVES) == pytest.approx(
            5.0, abs=1e-12
        )

    def test_the_cosine_weighting_shares_the_row_it_is_printed_in(self) -> None:
        # Table A.1 prints one row, "J_LF or J_LFC", so the cosine-weighted
        # variant of (A.15) is averaged and judged by the same numbers.
        values = np.array([0.2, 0.3, 0.4, 0.5, 9.0, 9.0])
        assert room.single_number_average("J_LFC", values, OCTAVES) == pytest.approx(
            room.single_number_average("J_LF", values, OCTAVES)
        )
        assert room.perceptibly_different("J_LFC", 0.20, 0.26)
        assert not room.perceptibly_different("J_LFC", 0.20, 0.24)

    def test_it_names_both_printed_symbols_when_it_refuses_one(self) -> None:
        with pytest.raises(ValueError, match="J_LFC"):
            room.single_number_average("J_XX", np.zeros(6), OCTAVES)

    @pytest.mark.parametrize("bad", [np.nan, np.inf])
    def test_it_refuses_a_band_axis_that_is_not_finite(self, bad: float) -> None:
        # A NaN centre made np.argmin hand back index 0 for every band, and
        # the average that came out was a real reading from the wrong band.
        axis = OCTAVES.copy()
        axis[0] = bad
        with pytest.raises(ValueError, match="finite centre frequencies"):
            room.single_number_average("G", np.arange(6.0), axis)

    def test_the_lateral_fraction_averages_four_bands(self) -> None:
        # EXAMPLE 2 of A.5: J_LFm is averaged in the 125 Hz to 1 kHz bands,
        # so the two top octaves take no part in it.
        values = np.array([0.2, 0.3, 0.4, 0.5, 9.0, 9.0])
        assert room.single_number_average("J_LF", values, OCTAVES) == pytest.approx(
            0.35, abs=1e-12
        )

    def test_the_late_lateral_level_takes_the_energy_route(self) -> None:
        values = np.array([-14.0, -8.0, -5.0, 1.0, 0.0, 0.0])
        energetic = room.single_number_average("L_J", values, OCTAVES)
        assert energetic == pytest.approx(
            room.late_lateral_average(values[:4]), abs=1e-12
        )
        assert energetic > float(np.mean(values[:4]))
        assert energetic == pytest.approx(-3.5324, abs=1e-4)

    def test_it_refuses_a_band_axis_that_lacks_a_band_it_averages(self) -> None:
        with pytest.raises(ValueError, match="does not carry the 125 Hz band"):
            room.single_number_average(
                "J_LF", np.zeros(4), np.array([500.0, 1000.0, 2000.0, 4000.0])
            )

    def test_it_refuses_a_quantity_the_table_does_not_print(self) -> None:
        with pytest.raises(ValueError, match="Table A.1 prints"):
            room.single_number_average("C50", np.zeros(6), OCTAVES)


class TestOctavePairAverages:
    """The low, mid and high presentation of A.5."""

    def test_each_pair_is_the_arithmetic_mean_of_two_octaves(self) -> None:
        values = np.array([2.0, 3.0, 4.0, 6.0, 5.0, 4.5])
        pairs = room.octave_pair_averages(values, OCTAVES)
        assert pairs == pytest.approx({"low": 2.5, "mid": 5.0, "high": 4.75})

    def test_the_mid_pair_and_the_single_number_agree_only_sometimes(self) -> None:
        # They are the same two bands for the five monaural quantities and
        # different band sets for the lateral ones, so they are two products
        # and not one.
        values = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        mid = room.octave_pair_averages(values, OCTAVES)["mid"]
        assert mid == pytest.approx(room.single_number_average("G", values, OCTAVES))
        assert mid != pytest.approx(room.single_number_average("J_LF", values, OCTAVES))

    def test_it_refuses_a_band_axis_that_does_not_span_all_six(self) -> None:
        with pytest.raises(ValueError, match="does not carry the 4000 Hz band"):
            room.octave_pair_averages(np.zeros(5), OCTAVES[:5])


class TestPerceptiblyDifferent:
    """The just-noticeable differences, absolute and relative."""

    def test_an_absolute_jnd_is_compared_as_a_difference(self) -> None:
        assert room.perceptibly_different("G", 4.0, 5.0)
        assert not room.perceptibly_different("G", 4.0, 4.9)

    def test_the_early_decay_time_is_compared_as_a_fraction(self) -> None:
        # 5 % of a 2 s decay is 0,1 s and 5 % of a 0,5 s decay is 0,025 s,
        # so the same 0,06 s difference is imperceptible in one hall and
        # plain in the other.
        assert not room.perceptibly_different("EDT", 2.0, 2.06)
        assert room.perceptibly_different("EDT", 0.5, 0.56)

    def test_the_late_lateral_level_refuses_the_comparison(self) -> None:
        with pytest.raises(ValueError, match="'Not known'"):
            room.perceptibly_different("L_J", -8.0, -2.0)

    def test_it_compares_whole_bands_at_once(self) -> None:
        assert room.perceptibly_different("C80", [0.0, 1.0], [2.0, 3.0])
        assert not room.perceptibly_different("C80", [0.0, 1.0], [2.0, 1.5])


class TestMinimumReceiverPositions:
    """Table A.2, and the line its three rows lie on."""

    @pytest.mark.parametrize(
        ("seats", "positions"), [(500, 6.0), (1000, 8.0), (2000, 10.0)]
    )
    def test_the_three_printed_rows_reproduce_exactly(
        self, seats: int, positions: float
    ) -> None:
        assert room.minimum_receiver_positions(seats) == pytest.approx(
            positions, abs=1e-12
        )

    def test_the_rows_are_two_positions_per_doubling(self) -> None:
        # Each printed row doubles the seats and adds two positions, with no
        # residual at any of the three, which is what makes a hall between
        # them answerable at all.
        assert room.minimum_receiver_positions(1000) - room.minimum_receiver_positions(
            500
        ) == pytest.approx(2.0, abs=1e-12)
        assert room.minimum_receiver_positions(500 * np.sqrt(2.0)) == pytest.approx(
            7.0, abs=1e-12
        )

    def test_it_does_not_extrapolate_past_the_bracket_a4_prints(self) -> None:
        # A.4 asks for "a minimum of between 6 and 10", so a 200-seat studio
        # does not get four positions and a 5 000-seat arena does not get
        # thirteen on the strength of three rows.
        assert room.minimum_receiver_positions(200) == pytest.approx(6.0)
        assert room.minimum_receiver_positions(5000) == pytest.approx(10.0)

    def test_it_keeps_the_shape_it_is_given(self) -> None:
        out = room.minimum_receiver_positions([500, 1000, 2000])
        assert isinstance(out, np.ndarray)
        assert out == pytest.approx([6.0, 8.0, 10.0])

    def test_it_refuses_a_hall_with_no_seats(self) -> None:
        with pytest.raises(ValueError, match="positive, finite number of seats"):
            room.minimum_receiver_positions(0)

    @pytest.mark.parametrize("seats", [np.nan, np.inf, [500.0, np.nan]])
    def test_it_refuses_a_seat_count_that_is_not_a_number(
        self, seats: float | list[float]
    ) -> None:
        # A NaN used to come back as a NaN, and an infinite hall used to be
        # clipped to the ten positions Table A.2 prints for a 2 000-seat one.
        with pytest.raises(ValueError, match="positive, finite number of seats"):
            room.minimum_receiver_positions(seats)


class TestSourceDirectivity:
    """Table 1 and the gliding average of 4.2.1."""

    def test_the_six_printed_limits_are_transcribed(self) -> None:
        assert room.MAX_SOURCE_DIRECTIVITY_DEVIATION_DB == {
            125.0: 1.0,
            250.0: 1.0,
            500.0: 1.0,
            1000.0: 3.0,
            2000.0: 5.0,
            4000.0: 6.0,
        }

    def test_a_band_outside_the_table_has_no_limit_rather_than_the_nearest(
        self,
    ) -> None:
        assert room.source_directivity_limit(4000.0) == 6.0
        with pytest.raises(ValueError, match="and no others"):
            room.source_directivity_limit(8000.0)

    def test_a_perfectly_omnidirectional_source_deviates_by_nothing(self) -> None:
        deviations = room.gliding_directivity_deviation(np.full(72, 94.0))
        assert deviations == pytest.approx(np.zeros(72), abs=1e-12)

    def test_the_survey_the_clause_prints_gives_one_arc_per_bearing(self) -> None:
        bearings = round(360.0 / room.DIRECTIVITY_STEP_DEG)
        assert bearings == 72
        deviations = room.gliding_directivity_deviation(np.zeros(bearings))
        assert deviations.size == bearings

    def test_the_average_is_energetic_and_the_reference_is_the_whole_turn(
        self,
    ) -> None:
        # A cosine pattern has an energy mean of exactly half its on-axis
        # value over any whole turn, so the arc that straddles the axis
        # stands above that reference and the one across the null falls
        # below it, by amounts that sum to nothing in energy.
        angles = np.arange(72) * 2.0 * np.pi / 72.0
        levels = 20.0 * np.log10(np.abs(np.cos(angles)) + 1e-9)
        deviations = room.gliding_directivity_deviation(levels)
        assert float(np.max(deviations)) > 0.0
        assert float(np.min(deviations)) < 0.0
        assert float(np.mean(10.0 ** (deviations / 10.0))) == pytest.approx(
            1.0, abs=1e-9
        )

    def test_a_scaled_survey_deviates_the_same(self) -> None:
        # The deviation is against the survey's own reference, so a change
        # of source level moves neither it nor the verdict.
        angles = np.arange(72) * 2.0 * np.pi / 72.0
        levels = 20.0 * np.log10(np.abs(np.cos(angles)) + 1e-9)
        assert room.gliding_directivity_deviation(levels + 12.0) == pytest.approx(
            room.gliding_directivity_deviation(levels), abs=1e-12
        )

    def test_it_refuses_a_survey_the_arc_does_not_divide(self) -> None:
        # 50 bearings put 7,2 degrees between them, and a 30 degree arc is
        # not a whole number of those.
        with pytest.raises(ValueError, match="whole number"):
            room.gliding_directivity_deviation(np.zeros(50))


class TestReportingConstants:
    """Clause 9.1's two routes and Clause 9.2's contract."""

    def test_the_two_routes_of_clause_91_are_both_carried(self) -> None:
        assert room.MID_FREQUENCY_OCTAVES_HZ == (500.0, 1000.0)
        assert room.MID_FREQUENCY_THIRD_OCTAVES_HZ == (
            400.0,
            500.0,
            630.0,
            800.0,
            1000.0,
            1250.0,
        )

    def test_the_third_octave_route_really_is_six_bands(self) -> None:
        # 9.1 says "the six one-third-octave bands from 400 Hz to 1 250 Hz".
        # There are six of them, and they are consecutive: each nominal
        # centre is the next preferred number, within the rounding that
        # series carries against the exact 2^(1/3).
        bands = np.asarray(room.MID_FREQUENCY_THIRD_OCTAVES_HZ)
        assert bands.size == 6
        assert bands[0] == 400.0
        assert bands[-1] == 1250.0
        assert bands[1:] / bands[:-1] == pytest.approx(2.0 ** (1.0 / 3.0), rel=0.02)

    def test_the_report_contract_has_the_fifteen_lettered_items(self) -> None:
        assert len(room.TEST_REPORT_ITEMS) == 15
        assert room.TEST_REPORT_ITEMS[0].startswith("a statement that")
        assert room.TEST_REPORT_ITEMS[-1].startswith("date of measurement")
