#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Tests for the lateral and binaural measures of ISO 3382-1:2009.

Validation strategy (closed forms, and the three characters the text layer
of the standard deletes):

- A single reflection at a known angle makes Equations (A.14) and (A.15)
  exact: J_LF is the square of the cosine times that reflection's share of
  the early energy, and J_LFC is the cosine itself times it.
- The modulus of (A.15) is printed. Two mirror-image reflections have
  figure-of-eight responses of opposite sign, so without it their
  contributions cancel to exactly zero; with it they add. The test pins the
  printed answer and states the other one.
- The 0,25 of (A.17) is one quarter, so four equal band values return that
  value unchanged, and an energy average of unequal ones sits above their
  arithmetic mean.
- The square root of (B.1) is printed: it is what makes two identical
  channels give exactly 1. The modulus of (B.2) is printed: it is what makes
  two anti-phase channels give 1 rather than -1.
"""

from __future__ import annotations

import numpy as np
import pytest

from phonometry import room

FS = 48000

#: 6*ln(10): decay-rate constant so that exp(-A60*t/T) falls 60 dB in T s.
A60 = 6.0 * np.log(10.0)

#: Sample the direct sound sits at, comfortably inside every band filter's
#: settling and well before the 5 ms lower limit of Equation (A.14).
DIRECT = 100


def reflection_pair(
    angles_deg: list[float],
    amplitudes: list[float],
    times_s: list[float],
    seconds: float = 0.3,
    fs: int = FS,
) -> tuple[np.ndarray, np.ndarray]:
    """An omnidirectional and a figure-of-eight response of known geometry.

    The direct sound is a unit arrival that the figure-of-eight microphone
    does not see at all, because A.2.4 points its null at the source. Each
    reflection reaches the omnidirectional microphone at its own amplitude
    and the figure-of-eight one at that amplitude times the cosine of its
    angle of incidence.
    """
    n = round(seconds * fs)
    omni, lateral = np.zeros(n), np.zeros(n)
    omni[DIRECT] = 1.0
    for angle, amplitude, time in zip(angles_deg, amplitudes, times_s, strict=True):
        index = DIRECT + round(time * fs)
        omni[index] += amplitude
        lateral[index] += amplitude * np.cos(np.deg2rad(angle))
    return omni, lateral


def decaying_noise(t60: float, seconds: float, seed: int, fs: int = FS) -> np.ndarray:
    t = np.arange(round(seconds * fs)) / fs
    rng = np.random.default_rng(seed)
    return np.asarray(rng.standard_normal(t.size) * np.exp(-0.5 * A60 * t / t60))


class TestEarlyLateralEnergyFraction:
    """Equations (A.14) and (A.15)."""

    @pytest.mark.parametrize("angle", [0.0, 45.0, 60.0])
    def test_one_reflection_gives_the_squared_cosine_share(self, angle: float) -> None:
        # Total early energy is the direct 1 plus the reflection's 0,25; the
        # lateral numerator is that 0,25 times cos^2(theta).
        omni, lateral = reflection_pair([angle], [0.5], [0.020])
        share = 0.25 * np.cos(np.deg2rad(angle)) ** 2 / 1.25
        result = room.early_lateral_energy_fraction(omni, lateral, FS, limits=None)
        assert result.energy_fraction[0] == pytest.approx(share, abs=1e-12)
        assert result.weighting == "squared"

    @pytest.mark.parametrize("angle", [0.0, 45.0, 60.0])
    def test_the_cosine_weighting_gives_the_plain_cosine_share(
        self, angle: float
    ) -> None:
        omni, lateral = reflection_pair([angle], [0.5], [0.020])
        share = 0.25 * abs(np.cos(np.deg2rad(angle))) / 1.25
        result = room.early_lateral_energy_fraction(
            omni, lateral, FS, weighting="cosine", limits=None
        )
        assert result.energy_fraction[0] == pytest.approx(share, abs=1e-12)
        assert result.weighting == "cosine"

    def test_the_printed_modulus_stops_mirror_reflections_cancelling(self) -> None:
        # Two reflections of equal amplitude at +45 and -45 degrees. The
        # figure-of-eight sees them with opposite sign, so the numerator of
        # (A.15) without its printed modulus would be exactly zero.
        omni, lateral = reflection_pair([45.0, 135.0], [0.5, 0.5], [0.020, 0.030])
        result = room.early_lateral_energy_fraction(
            omni, lateral, FS, weighting="cosine", limits=None
        )
        total = 1.0 + 0.25 + 0.25
        with_modulus = 2.0 * 0.25 * np.cos(np.deg2rad(45.0)) / total
        assert result.energy_fraction[0] == pytest.approx(with_modulus, abs=1e-12)
        assert result.energy_fraction[0] > 0.2

        signed = (
            0.25 * np.cos(np.deg2rad(45.0)) + 0.25 * np.cos(np.deg2rad(135.0))
        ) / total
        assert signed == pytest.approx(0.0, abs=1e-15)

    def test_a_reflection_before_five_milliseconds_is_outside_the_numerator(
        self,
    ) -> None:
        # (A.14) integrates the lateral energy from 5 ms, not from 0: a
        # reflection at 3 ms adds to the denominator and to nothing else.
        early = reflection_pair([45.0], [0.5], [0.003])
        late = reflection_pair([45.0], [0.5], [0.020])
        assert room.early_lateral_energy_fraction(
            *early, FS, limits=None
        ).energy_fraction[0] == pytest.approx(0.0, abs=1e-15)
        assert (
            room.early_lateral_energy_fraction(*late, FS, limits=None).energy_fraction[
                0
            ]
            > 0.0
        )

    def test_a_reflection_after_eighty_milliseconds_counts_for_neither(self) -> None:
        inside = reflection_pair([45.0], [0.5], [0.070])
        outside = reflection_pair([45.0], [0.5], [0.090])
        assert room.early_lateral_energy_fraction(
            *outside, FS, limits=None
        ).energy_fraction[0] == pytest.approx(0.0, abs=1e-15)
        assert (
            room.early_lateral_energy_fraction(
                *inside, FS, limits=None
            ).energy_fraction[0]
            > 0.0
        )

    def test_it_refuses_a_weighting_the_standard_does_not_print(self) -> None:
        omni, lateral = reflection_pair([45.0], [0.5], [0.020])
        with pytest.raises(ValueError, match="'weighting' must be one of"):
            room.early_lateral_energy_fraction(
                omni, lateral, FS, weighting="linear", limits=None
            )

    def test_it_refuses_two_responses_of_different_lengths(self) -> None:
        omni, lateral = reflection_pair([45.0], [0.5], [0.020])
        with pytest.raises(ValueError, match="share a time axis"):
            room.early_lateral_energy_fraction(omni, lateral[:-1], FS, limits=None)

    def test_it_refuses_a_response_too_short_for_the_early_window(self) -> None:
        # 50 ms of recording cannot carry an 80 ms integral, and a window
        # quietly shortened to what was recorded reports a larger fraction
        # than the printed one, because the missing tail is denominator only.
        omni, lateral = reflection_pair([45.0], [0.5], [0.020], seconds=0.05)
        with pytest.raises(ValueError, match="integrates to 0.08 s"):
            room.early_lateral_energy_fraction(omni, lateral, FS, limits=None)

    def test_it_returns_one_value_per_band(self) -> None:
        omni = decaying_noise(1.0, 2.0, 11)
        lateral = 0.5 * decaying_noise(1.0, 2.0, 12)
        result = room.early_lateral_energy_fraction(omni, lateral, FS)
        assert result.energy_fraction.shape == (6,)
        assert result.frequency is not None


class TestLateLateralSoundLevel:
    """Equations (A.16) and (A.17)."""

    def test_it_is_the_level_of_the_energy_after_eighty_milliseconds(self) -> None:
        # One late arrival of amplitude a at 120 ms against a unit reference
        # impulse: L_J = 10 lg a^2 = 20 lg a exactly.
        omni, lateral = reflection_pair([90.0], [0.5], [0.020], seconds=0.5)
        lateral[DIRECT + round(0.120 * FS)] = 0.25
        reference = np.zeros(round(0.2 * FS))
        reference[480] = 1.0
        result = room.late_lateral_sound_level(
            omni, lateral, reference, FS, limits=None
        )
        assert result.level[0] == pytest.approx(20.0 * np.log10(0.25), abs=1e-9)

    def test_energy_before_eighty_milliseconds_does_not_count(self) -> None:
        omni, lateral = reflection_pair([90.0], [0.5], [0.020], seconds=0.5)
        lateral[DIRECT + round(0.120 * FS)] = 0.25
        loud_early = lateral.copy()
        loud_early[DIRECT + round(0.050 * FS)] = 10.0
        reference = np.zeros(round(0.2 * FS))
        reference[480] = 1.0
        quiet = room.late_lateral_sound_level(omni, lateral, reference, FS, limits=None)
        noisy = room.late_lateral_sound_level(
            omni, loud_early, reference, FS, limits=None
        )
        assert noisy.level[0] == pytest.approx(quiet.level[0], abs=1e-12)

    def test_the_two_reference_routes_agree(self) -> None:
        omni = decaying_noise(1.0, 2.0, 21)
        lateral = 0.5 * decaying_noise(1.0, 2.0, 22)
        reference = np.zeros(round(0.2 * FS))
        reference[480] = 1.0
        from_ir = room.late_lateral_sound_level(omni, lateral, reference, FS)
        levels = room.sound_pressure_exposure_level(reference, FS)
        from_level = room.late_lateral_sound_level(
            omni, lateral, fs=FS, reference_level=levels
        )
        assert from_level.level == pytest.approx(from_ir.level, abs=1e-12)

    def test_it_wants_the_reference_exactly_once(self) -> None:
        omni = decaying_noise(1.0, 2.0, 23)
        lateral = 0.5 * decaying_noise(1.0, 2.0, 24)
        with pytest.raises(ValueError, match="exactly once"):
            room.late_lateral_sound_level(omni, lateral, fs=FS)


class TestLateLateralAverage:
    """Equation (A.17), the one energy average in Table A.1."""

    def test_four_equal_values_return_that_value(self) -> None:
        assert room.late_lateral_average([-8.0] * 4) == pytest.approx(-8.0, abs=1e-12)

    def test_it_is_the_printed_quarter_and_nothing_else(self) -> None:
        # 0, 0, 0, 6,0206 dB: the energies are 1, 1, 1, 4, so the quarter of
        # their sum is 1,75 and the level is 10 lg 1,75.
        levels = [0.0, 0.0, 0.0, 20.0 * np.log10(2.0)]
        assert room.late_lateral_average(levels) == pytest.approx(
            10.0 * np.log10(1.75), abs=1e-12
        )
        assert room.late_lateral_average(levels) == pytest.approx(2.4304, abs=1e-4)

    def test_it_sits_above_the_arithmetic_mean_of_the_same_bands(self) -> None:
        levels = np.array([-14.0, -8.0, -5.0, 1.0])
        assert room.late_lateral_average(levels) > float(np.mean(levels))

    def test_it_refuses_any_count_but_four(self) -> None:
        with pytest.raises(ValueError, match="octave bands"):
            room.late_lateral_average([-8.0, -8.0, -8.0])
        with pytest.raises(ValueError, match="octave bands"):
            room.late_lateral_average([-8.0] * 5)


class TestInterauralCrossCorrelation:
    """Equations (B.1) and (B.2)."""

    def test_identical_channels_correlate_exactly_one(self) -> None:
        x = decaying_noise(1.0, 0.5, 31)
        result = room.interaural_cross_correlation(x, x, FS, limits=None)
        assert result.coefficient[0] == pytest.approx(1.0, abs=1e-12)
        assert result.delay[0] == pytest.approx(0.0, abs=1e-12)

    def test_the_printed_modulus_makes_anti_phase_channels_correlate_one(self) -> None:
        # Two ears that hear the same thing with opposite sign are as
        # dissimilar as two signals can be in sign alone, and (B.2) takes the
        # maximum of the magnitude, so the answer is 1 and not -1 or 0.
        x = decaying_noise(1.0, 0.5, 32)
        result = room.interaural_cross_correlation(x, -x, FS, limits=None)
        assert result.coefficient[0] == pytest.approx(1.0, abs=1e-12)
        # The function itself reaches -1, never +1: it is the modulus of
        # (B.2) that turns that into a coefficient of one.
        assert float(np.min(result.correlation[0])) == pytest.approx(-1.0, abs=1e-12)
        assert float(np.max(result.correlation[0])) < 0.5

    def test_the_printed_square_root_is_what_bounds_it_by_one(self) -> None:
        # Scale one ear by 4: the normalisation absorbs it exactly. Without
        # the root the denominator would be the plain product of the two
        # energies and the coefficient would collapse by that factor.
        x = decaying_noise(1.0, 0.5, 33)
        plain = room.interaural_cross_correlation(x, x, FS, limits=None)
        scaled = room.interaural_cross_correlation(x, 4.0 * x, FS, limits=None)
        assert scaled.coefficient[0] == pytest.approx(plain.coefficient[0], abs=1e-12)

    def test_it_finds_a_delay_the_search_window_can_reach(self) -> None:
        x = decaying_noise(1.0, 0.5, 34)
        shift = round(0.0005 * FS)
        delayed = np.concatenate([np.zeros(shift), x])[: x.size]
        result = room.interaural_cross_correlation(x, delayed, FS, limits=None)
        assert result.delay[0] == pytest.approx(0.0005, abs=0.5 / FS)
        assert result.coefficient[0] > 0.99

    def test_the_search_window_is_one_millisecond_either_side(self) -> None:
        x = decaying_noise(1.0, 0.5, 35)
        result = room.interaural_cross_correlation(x, x, FS, limits=None)
        assert result.lag[0] == pytest.approx(-0.001, abs=0.5 / FS)
        assert result.lag[-1] == pytest.approx(0.001, abs=0.5 / FS)
        assert result.correlation.shape == (1, result.lag.size)

    def test_a_delay_outside_the_window_is_not_found(self) -> None:
        x = decaying_noise(1.0, 0.5, 36)
        shift = round(0.004 * FS)
        delayed = np.concatenate([np.zeros(shift), x])[: x.size]
        result = room.interaural_cross_correlation(x, delayed, FS, limits=None)
        assert result.coefficient[0] < 0.5

    def test_the_early_window_and_the_late_one_differ(self) -> None:
        x = decaying_noise(1.0, 0.5, 37)
        y = decaying_noise(1.0, 0.5, 38)
        early = room.interaural_cross_correlation(
            x, y, FS, window=room.IACC_EARLY_WINDOW_S, limits=None
        )
        late = room.interaural_cross_correlation(
            x, y, FS, window=(room.IACC_LATE_START_S, None), limits=None
        )
        assert early.coefficient[0] != late.coefficient[0]

    def test_it_refuses_a_window_that_runs_backwards(self) -> None:
        x = decaying_noise(1.0, 0.5, 39)
        with pytest.raises(ValueError, match="forwards in time"):
            room.interaural_cross_correlation(x, x, FS, window=(0.2, 0.1), limits=None)

    def test_it_refuses_two_channels_of_different_lengths(self) -> None:
        x = decaying_noise(1.0, 0.5, 40)
        with pytest.raises(ValueError, match="share a time axis"):
            room.interaural_cross_correlation(x, x[:-1], FS, limits=None)


class TestPlots:
    """The three renderers, checked for content rather than looked at."""

    @staticmethod
    def _responses() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        omni = decaying_noise(1.0, 1.0, 51)
        lateral = 0.5 * decaying_noise(1.0, 1.0, 52)
        reference = np.zeros(round(0.2 * FS))
        reference[480] = 1.0
        return omni, lateral, reference

    def test_the_lateral_plot_draws_the_fraction_and_the_range(self) -> None:
        pytest.importorskip("matplotlib")
        import matplotlib as mpl
        from matplotlib import pyplot as plt

        mpl.use("Agg")
        omni, lateral, _ = self._responses()
        result = room.early_lateral_energy_fraction(omni, lateral, FS)
        ax = result.plot()
        curve = next(
            line
            for line in ax.get_lines()
            if np.size(line.get_ydata()) == result.energy_fraction.size
        )
        assert curve.get_ydata() == pytest.approx(result.energy_fraction)
        span = next(
            p for p in ax.patches if p.get_label().startswith(("Typical", "Rango"))
        )
        bottom = float(span.get_xy()[1])
        assert (bottom, bottom + float(span.get_height())) == (0.05, 0.35)
        plt.close("all")

    def test_the_lateral_plot_names_the_weighting_it_drew(self) -> None:
        pytest.importorskip("matplotlib")
        import matplotlib as mpl
        from matplotlib import pyplot as plt

        mpl.use("Agg")
        omni, lateral, _ = self._responses()
        squared = room.early_lateral_energy_fraction(omni, lateral, FS).plot()
        cosine = room.early_lateral_energy_fraction(
            omni, lateral, FS, weighting="cosine"
        ).plot()
        assert any("{LF}" in line.get_label() for line in squared.get_lines())
        assert any("{LFm}" in line.get_label() for line in squared.get_lines())
        assert any("{LFC}" in line.get_label() for line in cosine.get_lines())
        assert any("{LFCm}" in line.get_label() for line in cosine.get_lines())
        plt.close("all")

    def test_the_late_lateral_plot_marks_the_energy_average(self) -> None:
        pytest.importorskip("matplotlib")
        import matplotlib as mpl
        from matplotlib import pyplot as plt

        mpl.use("Agg")
        omni, lateral, reference = self._responses()
        result = room.late_lateral_sound_level(omni, lateral, reference, FS)
        ax = result.plot()
        expected = room.late_lateral_average(result.level[:4])
        flat = [
            line
            for line in ax.get_lines()
            if len(set(np.round(line.get_ydata(), 9))) == 1
        ]
        assert any(
            float(line.get_ydata()[0]) == pytest.approx(expected, abs=1e-9)
            for line in flat
        )
        plt.close("all")

    def test_the_correlation_plot_draws_one_curve_per_band(self) -> None:
        pytest.importorskip("matplotlib")
        import matplotlib as mpl
        from matplotlib import pyplot as plt

        mpl.use("Agg")
        omni, lateral, _ = self._responses()
        result = room.interaural_cross_correlation(omni, lateral, FS)
        ax = result.plot()
        curves = [line for line in ax.get_lines() if line.get_ydata().size > 1]
        assert len(curves) == result.correlation.shape[0]
        assert curves[0].get_ydata() == pytest.approx(result.correlation[0])
        plt.close("all")

    def test_the_spanish_edition_translates_the_three_titles(self) -> None:
        pytest.importorskip("matplotlib")
        import matplotlib as mpl
        from matplotlib import pyplot as plt

        mpl.use("Agg")
        omni, lateral, reference = self._responses()
        assert (
            "lateral temprana"
            in room.early_lateral_energy_fraction(omni, lateral, FS)
            .plot(language="es")
            .get_title()
        )
        assert (
            "lateral tardío"
            in room.late_lateral_sound_level(omni, lateral, reference, FS)
            .plot(language="es")
            .get_title()
        )
        assert (
            "interaural"
            in room.interaural_cross_correlation(omni, lateral, FS)
            .plot(language="es")
            .get_title()
        )
        plt.close("all")
