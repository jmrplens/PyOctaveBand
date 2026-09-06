#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Tests for the stage support of ISO 3382-1:2009, Annex C.

Validation strategy: both supports are ratios of energies over printed
windows, so a response made of arrivals placed inside and outside those
windows has an answer that closes in one line. The two windows also have a
gap and a ceiling the prose of C.2.1 and C.2.2 do not mention, and an
arrival dropped into either shows that the equations, not the prose, are
what is implemented.
"""

from __future__ import annotations

import numpy as np
import pytest

from phonometry import room

FS = 48000

#: 6*ln(10): decay-rate constant so that exp(-A60*t/T) falls 60 dB in T s.
A60 = 6.0 * np.log(10.0)

#: Sample the direct sound sits at.
DIRECT = 100


def platform_response(
    arrivals: list[tuple[float, float]], seconds: float = 2.0, fs: int = FS
) -> np.ndarray:
    """A unit direct arrival plus ``(time, amplitude)`` reflections."""
    x = np.zeros(round(seconds * fs))
    x[DIRECT] = 1.0
    for time, amplitude in arrivals:
        x[DIRECT + round(time * fs)] += amplitude
    return x


class TestStageSupport:
    """Equations (C.1) and (C.2)."""

    def test_one_reflection_in_each_window_closes_in_one_line(self) -> None:
        # The denominator is the unit direct sound, so each support is just
        # 20 lg of the amplitude that landed in its own window.
        response = platform_response([(0.050, 0.2), (0.400, 0.1)])
        result = room.stage_support(response, FS, limits=None)
        assert result.early[0] == pytest.approx(20.0 * np.log10(0.2), abs=1e-9)
        assert result.late[0] == pytest.approx(20.0 * np.log10(0.1), abs=1e-9)

    def test_the_two_supports_share_a_denominator(self) -> None:
        # Louder direct sound, same reflections: both supports drop by the
        # same amount and their difference does not move.
        quiet = room.stage_support(
            platform_response([(0.050, 0.2), (0.400, 0.1)]), FS, limits=None
        )
        loud = platform_response([(0.050, 0.2), (0.400, 0.1)])
        loud[DIRECT] = 2.0
        louder = room.stage_support(loud, FS, limits=None)
        shift = 20.0 * np.log10(0.5)
        assert louder.early[0] == pytest.approx(quiet.early[0] + shift, abs=1e-9)
        assert louder.late[0] == pytest.approx(quiet.late[0] + shift, abs=1e-9)
        assert louder.late[0] - louder.early[0] == pytest.approx(
            quiet.late[0] - quiet.early[0], abs=1e-12
        )

    def test_the_gap_the_prose_does_not_mention_is_really_there(self) -> None:
        # C.2.1 calls it "the reflected energy within the first 0,1 s", but
        # (C.1) starts at 20 ms. A reflection at 15 ms belongs to neither
        # the direct window nor the early one, and counts for nothing.
        inside = room.stage_support(
            platform_response([(0.050, 0.2), (0.400, 0.1)]), FS, limits=None
        )
        in_the_gap = room.stage_support(
            platform_response([(0.015, 5.0), (0.050, 0.2), (0.400, 0.1)]),
            FS,
            limits=None,
        )
        assert in_the_gap.early[0] == pytest.approx(inside.early[0], abs=1e-9)
        assert in_the_gap.late[0] == pytest.approx(inside.late[0], abs=1e-9)

    def test_the_ceiling_the_prose_does_not_mention_is_really_there(self) -> None:
        # C.2.2 calls it "the reflected energy after the first 0,1 s" with no
        # upper bound, but (C.2) stops at one second, which is inside the
        # decay of any hall worth measuring.
        response = platform_response([(0.050, 0.2), (0.400, 0.1)])
        beyond = response.copy()
        beyond[DIRECT + round(1.5 * FS)] = 5.0
        assert room.stage_support(beyond, FS, limits=None).late[0] == pytest.approx(
            room.stage_support(response, FS, limits=None).late[0], abs=1e-9
        )

    def test_it_defaults_to_the_four_bands_c24_averages(self) -> None:
        rng = np.random.default_rng(3382)
        t = np.arange(round(2.0 * FS)) / FS
        response = rng.standard_normal(t.size) * np.exp(-0.5 * A60 * t / 1.6) * 0.02
        response[DIRECT] += 1.0
        result = room.stage_support(response, FS)
        assert result.frequency is not None
        assert result.frequency.size == len(room.STAGE_SUPPORT_BANDS_HZ)
        assert result.frequency[0] == pytest.approx(250.0, rel=0.01)
        assert result.frequency[-1] == pytest.approx(2000.0, rel=0.01)

    def test_it_refuses_a_response_that_cannot_reach_one_second(self) -> None:
        with pytest.raises(ValueError, match="integrates to 1 s"):
            room.stage_support(
                platform_response([(0.050, 0.2)], seconds=0.5), FS, limits=None
            )

    def test_the_printed_standard_deviations_close_on_twelve_readings(self) -> None:
        # C.2.4 estimates 1 dB for one band in one position and 0,3 dB for
        # the single number, which is that divided by the root of the four
        # bands times three positions it averages.
        readings = len(room.STAGE_SUPPORT_BANDS_HZ) * room.STAGE_SUPPORT_POSITIONS
        assert readings == 12
        naive = room.STAGE_SUPPORT_STANDARD_DEVIATION_DB / np.sqrt(readings)
        assert naive == pytest.approx(0.2887, abs=1e-4)
        assert round(float(naive), 1) == pytest.approx(
            room.STAGE_SUPPORT_SINGLE_NUMBER_STANDARD_DEVIATION_DB
        )

    def test_the_plot_draws_both_supports_and_both_ranges(self) -> None:
        pytest.importorskip("matplotlib")
        import matplotlib as mpl
        from matplotlib import pyplot as plt

        mpl.use("Agg")
        rng = np.random.default_rng(11)
        t = np.arange(round(2.0 * FS)) / FS
        response = rng.standard_normal(t.size) * np.exp(-0.5 * A60 * t / 1.6) * 0.02
        response[DIRECT] += 1.0
        result = room.stage_support(response, FS)
        ax = result.plot()
        curves = [line for line in ax.get_lines() if np.size(line.get_ydata()) == 4]
        assert len(curves) == 2
        assert curves[0].get_ydata() == pytest.approx(result.early)
        assert curves[1].get_ydata() == pytest.approx(result.late)
        spans = [p for p in ax.patches if "Table C.1" in p.get_label()]
        assert len(spans) == 2
        plt.close("all")

    def test_the_spanish_edition_translates_the_plot(self) -> None:
        pytest.importorskip("matplotlib")
        import matplotlib as mpl
        from matplotlib import pyplot as plt

        mpl.use("Agg")
        response = platform_response([(0.050, 0.2), (0.400, 0.1)])
        ax = room.stage_support(response, FS).plot(language="es")
        assert "Soporte de escenario" in ax.get_title()
        labels = [t.get_text() for t in ax.get_legend().get_texts()]
        assert any("Soporte temprano" in text for text in labels)
        assert any("Tabla C.1" in text for text in labels)
        plt.close("all")


class TestReverberationTimeUncertainty:
    """Clause 7.1, Equations (4) and (5)."""

    def test_it_reproduces_the_printed_form_by_hand(self) -> None:
        # T30 = 2,0 s in the 1 kHz octave band, ten decays in each of twelve
        # positions.
        sigma = room.reverberation_time_standard_deviation(
            2.0, 710.0, evaluation_range=30.0, decays=10, positions=12
        )
        expected = 0.55 * 2.0 * np.sqrt((1.0 + 1.52 / 10.0) / (12.0 * 710.0 * 2.0))
        assert sigma == pytest.approx(expected, abs=1e-12)

    def test_the_two_ranges_differ_by_the_printed_coefficients(self) -> None:
        # 0,88/0,55 is exactly 1,6, and the rest of the ratio is the two
        # decay terms.
        common = {"bandwidth": 710.0, "decays": 10, "positions": 12}
        wide = room.reverberation_time_standard_deviation(
            2.0, evaluation_range=20.0, **common
        )
        narrow = room.reverberation_time_standard_deviation(
            2.0, evaluation_range=30.0, **common
        )
        assert wide / narrow == pytest.approx(1.6 * np.sqrt(1.19 / 1.152), abs=1e-12)

    def test_it_grows_as_the_square_root_of_the_decay_time(self) -> None:
        # The prefactor's T and the T under the radical leave half a power,
        # so four times the decay time is twice the standard deviation.
        short = room.reverberation_time_standard_deviation(0.5, 710.0)
        long = room.reverberation_time_standard_deviation(2.0, 710.0)
        assert long / short == pytest.approx(2.0, abs=1e-12)

    def test_the_integrated_response_default_is_the_printed_ten(self) -> None:
        assert room.INTEGRATED_RESPONSE_DECAYS == 10
        assert room.reverberation_time_standard_deviation(2.0, 710.0) == pytest.approx(
            room.reverberation_time_standard_deviation(2.0, 710.0, decays=10), abs=1e-15
        )

    def test_ten_decays_is_not_the_same_as_infinitely_many(self) -> None:
        # 7.2 says the theory gives infinity and the practice gives ten; the
        # gap is 7 % on T30, which is not a rounding.
        ten = room.reverberation_time_standard_deviation(2.0, 710.0, decays=10)
        many = room.reverberation_time_standard_deviation(2.0, 710.0, decays=10**9)
        assert ten / many == pytest.approx(np.sqrt(1.152), rel=1e-6)
        assert ten / many == pytest.approx(1.0733, abs=1e-4)

    def test_it_refuses_an_evaluation_range_the_clause_does_not_print(self) -> None:
        with pytest.raises(ValueError, match="20 dB and 30 dB"):
            room.reverberation_time_standard_deviation(
                2.0, 710.0, evaluation_range=10.0
            )

    def test_it_refuses_counts_below_one(self) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            room.reverberation_time_standard_deviation(2.0, 710.0, positions=0)


class TestFilterBandwidth:
    """Clause 7.1's two printed bandwidths."""

    @pytest.mark.parametrize(("fraction", "expected"), [(1, 710.0), (3, 230.0)])
    def test_it_is_the_printed_fraction_of_the_centre(
        self, fraction: int, expected: float
    ) -> None:
        assert room.filter_bandwidth(1000.0, fraction) == pytest.approx(
            expected, abs=1e-12
        )

    def test_the_printed_fractions_round_from_the_iec_band_edges(self) -> None:
        # 2^(1/2) - 2^(-1/2) = 0,7071 and 2^(1/6) - 2^(-1/6) = 0,2316, which
        # is what the clause rounds to two figures.
        octave = 2.0 ** (1.0 / 2.0) - 2.0 ** (-1.0 / 2.0)
        third = 2.0 ** (1.0 / 6.0) - 2.0 ** (-1.0 / 6.0)
        assert octave == pytest.approx(0.70711, abs=1e-5)
        assert third == pytest.approx(0.23156, abs=1e-5)
        assert round(octave, 2) == room.FILTER_BANDWIDTH_FRACTION[1]
        assert round(third, 2) == room.FILTER_BANDWIDTH_FRACTION[3]

    def test_it_keeps_the_shape_it_is_given(self) -> None:
        out = room.filter_bandwidth([125.0, 1000.0])
        assert isinstance(out, np.ndarray)
        assert out == pytest.approx([88.75, 710.0], abs=1e-12)

    def test_it_refuses_a_fraction_the_clause_does_not_print(self) -> None:
        with pytest.raises(ValueError, match="must be one of"):
            room.filter_bandwidth(1000.0, 6)


class TestMinimumReliableReverberationTime:
    """Clause 7.3, Equations (6) and (7)."""

    def test_the_filter_limit_is_sixteen_over_the_bandwidth(self) -> None:
        assert room.minimum_reliable_reverberation_time(710.0) == pytest.approx(
            16.0 / 710.0, abs=1e-15
        )

    def test_the_detector_takes_over_when_it_is_the_slower_of_the_two(self) -> None:
        # 2 T_det = 0,1 s against 16/B = 0,0225 s.
        assert room.minimum_reliable_reverberation_time(710.0, 0.05) == pytest.approx(
            0.1, abs=1e-15
        )

    def test_a_narrow_band_binds_harder_than_a_wide_one(self) -> None:
        low = room.minimum_reliable_reverberation_time(room.filter_bandwidth(125.0))
        high = room.minimum_reliable_reverberation_time(room.filter_bandwidth(4000.0))
        assert low > high
        assert low == pytest.approx(16.0 / 88.75, abs=1e-12)

    def test_it_refuses_a_negative_detector_time(self) -> None:
        with pytest.raises(ValueError, match="must not be negative"):
            room.minimum_reliable_reverberation_time(710.0, -0.1)
