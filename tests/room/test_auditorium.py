#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Tests for the sound strength G of ISO 3382-1:2009, A.2.1.

Validation strategy (closed forms and printed constants, never
self-consistency):

- Equation (A.1) is a ratio of energies, so scaling one response by k and
  leaving the other alone must give exactly 20 lg k, whatever the waveform.
- Equation (A.2) is an absolute level: a rectangular burst of known
  pressure and duration has an exposure level that closes in one line.
- Equations (A.4) and (A.8) are the inverse-square law, exact at every
  distance, and the identity at 10 m.
- The printed offsets of Equations (A.5) and (A.9), 37 dB and 31 dB, are
  the rounded values of 10 lg(1600 pi) and 10 lg(400 pi). Both roundings
  are correct on their own and the pair cannot close better than
  10 lg 4 - 6 = 0,0206 dB, which is pinned here rather than left for a
  future test to trip over.
- The energy mean of a cosine directivity is exactly 1/2, so its level is
  10 lg(1/2) below the on-axis level for every bearing count.
"""

from __future__ import annotations

import numpy as np
import pytest

from phonometry import room
from phonometry.room.auditorium import AuditoriumWarning

FS = 48000

#: 6*ln(10): decay-rate constant so that exp(-A60*t/T) falls 60 dB in T s.
A60 = 6.0 * np.log(10.0)

#: 10 lg(4 pi * 10^2), the exact free-field spread from a sound power level
#: to the pressure level at 10 m, of which Equation (A.9) prints 31 dB.
EXACT_POWER_OFFSET_DB = 10.0 * np.log10(4.0 * np.pi * 100.0)

#: 10 lg(16 pi * 10^2), the exact diffuse-to-free-field ratio at 10 m, of
#: which Equation (A.5) prints 37 dB.
EXACT_DIFFUSE_OFFSET_DB = 10.0 * np.log10(16.0 * np.pi * 100.0)


def exponential_ir(t60: float, seconds: float, fs: int = FS) -> np.ndarray:
    """Pressure IR whose energy envelope is exactly exp(-A60*t/t60)."""
    t = np.arange(round(seconds * fs)) / fs
    return np.asarray(np.exp(-0.5 * A60 * t / t60))


def noisy_decay(t60: float, seconds: float, fs: int = FS) -> np.ndarray:
    """A broadband decay with energy in every octave band of the default range."""
    t = np.arange(round(seconds * fs)) / fs
    rng = np.random.default_rng(20260906)
    return np.asarray(rng.standard_normal(t.size) * np.exp(-0.5 * A60 * t / t60))


def anechoic_ir(seconds: float = 0.3, fs: int = FS) -> np.ndarray:
    """A clean direct arrival, the free-field reference of Equation (A.3).

    The arrival sits a tenth of the way in, so a short window is short of
    the tail rather than of the arrival itself.
    """
    x = np.zeros(round(seconds * fs))
    x[round(0.1 * x.size)] = 1.0
    return x


class TestSoundExposureLevel:
    """Equation (A.2): the absolute level the strength is a difference of."""

    def test_it_closes_for_a_rectangular_burst(self) -> None:
        # A burst of 1 Pa held for tau seconds carries an energy of tau Pa^2 s,
        # so L_pE = 10 lg(tau / (1 s * (20 uPa)^2)) with no integration error
        # to speak of: every sample contributes exactly 1/fs.
        tau = 0.5
        burst = np.ones(round(tau * FS))
        expected = 10.0 * np.log10(tau / (1.0 * (2.0e-5) ** 2))
        assert room.sound_pressure_exposure_level(
            burst, FS, limits=None
        ) == pytest.approx(expected, abs=1e-9)

    def test_it_is_a_level_so_doubling_the_pressure_adds_six_decibels(self) -> None:
        burst = np.ones(round(0.5 * FS))
        quiet = room.sound_pressure_exposure_level(burst, FS, limits=None)
        loud = room.sound_pressure_exposure_level(2.0 * burst, FS, limits=None)
        assert loud - quiet == pytest.approx(20.0 * np.log10(2.0), abs=1e-12)

    def test_it_returns_one_value_per_band(self) -> None:
        levels = room.sound_pressure_exposure_level(noisy_decay(1.0, 2.0), FS)
        assert np.asarray(levels).shape == (6,)

    def test_each_band_reports_its_own_band(self) -> None:
        # A tone in one octave only: that band's level has to be the loud
        # one, which is what pins the band axis to the values on it rather
        # than to some other ordering of the same six numbers.
        seconds = 2.0
        t = np.arange(round(seconds * FS)) / FS
        loud = np.sin(2.0 * np.pi * 1000.0 * t) * np.exp(-0.5 * A60 * t / 1.0)
        levels = np.asarray(room.sound_pressure_exposure_level(loud, FS))
        assert int(np.argmax(levels)) == 3
        assert levels[3] - float(np.max(np.delete(levels, 3))) > 25.0

    def test_it_integrates_from_the_direct_sound_and_not_from_sample_zero(
        self,
    ) -> None:
        # The A.3.4 trigger discards what sits before the direct sound, so a
        # response padded with noise ahead of its arrival reports the same
        # level as the unpadded one. Without the trim the padding is
        # integrated and the level rises with the length of the pad.
        rng = np.random.default_rng(4242)
        clean = noisy_decay(1.0, 2.0)
        padded = np.concatenate([1e-2 * rng.standard_normal(FS), clean])
        assert room.sound_pressure_exposure_level(
            padded, FS, limits=None
        ) == pytest.approx(
            room.sound_pressure_exposure_level(clean, FS, limits=None), abs=0.01
        )

    def test_it_returns_a_float_for_a_broadband_response(self) -> None:
        value = room.sound_pressure_exposure_level(
            noisy_decay(1.0, 2.0), FS, limits=None
        )
        assert isinstance(value, float)

    def test_it_refuses_a_silent_response(self) -> None:
        with pytest.raises(ValueError, match="silent"):
            room.sound_pressure_exposure_level(np.zeros(1024), FS, limits=None)


class TestSoundStrength:
    """Equation (A.1), the ratio the annex defines G as."""

    @pytest.mark.parametrize("factor", [2.0, 10.0, 0.5])
    def test_a_scaled_response_gives_exactly_twenty_lg_of_the_factor(
        self, factor: float
    ) -> None:
        # The waveform cancels: only the factor survives the ratio, which is
        # the whole point of writing G as an energy ratio.
        reference = anechoic_ir()
        result = room.sound_strength(factor * reference, reference, FS, limits=None)
        assert result.strength[0] == pytest.approx(20.0 * np.log10(factor), abs=1e-12)

    def test_the_two_reference_routes_agree(self) -> None:
        room_ir = noisy_decay(1.0, 2.0)
        reference = anechoic_ir()
        from_ir = room.sound_strength(room_ir, reference, FS)
        levels = room.sound_pressure_exposure_level(reference, FS)
        from_level = room.sound_strength(room_ir, fs=FS, reference_level=levels)
        assert from_level.strength == pytest.approx(from_ir.strength, abs=1e-12)

    def test_strength_is_the_difference_of_the_two_levels_it_reports(self) -> None:
        result = room.sound_strength(noisy_decay(1.0, 2.0), anechoic_ir(), FS)
        assert result.strength == pytest.approx(
            result.exposure_level - result.reference_level, abs=1e-12
        )

    def test_a_common_gain_on_both_responses_cancels(self) -> None:
        room_ir, reference = noisy_decay(1.0, 2.0), anechoic_ir()
        plain = room.sound_strength(room_ir, reference, FS)
        gained = room.sound_strength(7.5 * room_ir, 7.5 * reference, FS)
        assert gained.strength == pytest.approx(plain.strength, abs=1e-12)

    def test_it_carries_the_band_centres(self) -> None:
        result = room.sound_strength(noisy_decay(1.0, 2.0), anechoic_ir(), FS)
        assert result.frequency is not None
        assert result.frequency.shape == (6,)
        assert result.frequency[0] == pytest.approx(125.0, rel=0.01)

    def test_a_broadband_measurement_has_no_band_axis(self) -> None:
        result = room.sound_strength(
            noisy_decay(1.0, 2.0), anechoic_ir(), FS, limits=None
        )
        assert result.frequency is None
        assert result.strength.shape == (1,)

    def test_it_wants_the_reference_exactly_once(self) -> None:
        room_ir = noisy_decay(1.0, 2.0)
        reference = anechoic_ir()
        with pytest.raises(ValueError, match="exactly once"):
            room.sound_strength(room_ir, fs=FS)
        with pytest.raises(ValueError, match="exactly once"):
            room.sound_strength(room_ir, reference, FS, reference_level=0.0)

    def test_it_refuses_a_reference_level_of_the_wrong_width(self) -> None:
        room_ir = noisy_decay(1.0, 2.0)
        with pytest.raises(ValueError, match="does not broadcast"):
            room.sound_strength(room_ir, fs=FS, reference_level=[1.0, 2.0, 3.0])

    def test_the_decay_range_the_warning_asks_for_is_the_printed_thirty(
        self,
    ) -> None:
        # A.2.1 asks the integral to reach the point where the decay curve
        # has fallen 30 dB, and the module's own constant has to be that
        # number and be used as a power ratio, not as an amplitude one.
        from phonometry.room import auditorium

        assert auditorium._MINIMUM_DECAY_RANGE_DB == 30.0
        assert 10.0 ** (auditorium._MINIMUM_DECAY_RANGE_DB / 10.0) == 1000.0

    def test_it_warns_when_the_room_response_was_cut_short(self) -> None:
        # A 2 s decay recorded for 0,1 s never reaches the 30 dB of decay
        # A.2.1 asks the integral to run to.
        room_ir = exponential_ir(2.0, 0.1)
        reference = anechoic_ir()
        with pytest.warns(AuditoriumWarning, match="cut short"):
            room.sound_strength(room_ir, reference, FS, limits=None)

    def test_it_does_not_grow_with_the_length_of_the_recording(self) -> None:
        # Same event, three lengths of tape over a noise floor. Without the
        # truncation of 5.3.3, Equation (3), every extra second of noise
        # would add energy that is not the source's, and G with it.
        fs, seconds = FS, 8.0
        rng = np.random.default_rng(3382)
        t = np.arange(round(seconds * fs)) / fs
        event = rng.standard_normal(t.size) * np.exp(-0.5 * A60 * t / 1.2)
        recorded = event + 3e-3 * rng.standard_normal(t.size)
        reference = anechoic_ir(0.3)

        short, long = (
            room.sound_strength(recorded[: round(cut * fs)], reference, fs).strength
            for cut in (2.0, 8.0)
        )
        assert long == pytest.approx(short, abs=0.01)

    def test_it_warns_when_a_response_cannot_hold_its_lowest_band(self) -> None:
        # A 5 ms anechoic window stops long before the 125 Hz octave filter
        # has rung down, so that band's exposure level is 24 dB light.
        room_ir = noisy_decay(1.0, 3.0)
        reference = anechoic_ir(0.006)
        with pytest.warns(AuditoriumWarning, match="ring down"):
            room.sound_strength(room_ir, reference, FS)

    def test_it_does_not_warn_about_the_free_field_reference(self) -> None:
        # The reference has no reverberant decay to reach 30 dB of; what
        # falls away after its peak is the band filter's own ring-down.
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error", AuditoriumWarning)
            room.sound_strength(noisy_decay(1.0, 3.0), anechoic_ir(0.3), FS)


class TestFreeFieldReferenceLevel:
    """Equations (A.4) and (A.8), the inverse-square correction to 10 m."""

    def test_ten_metres_is_the_identity(self) -> None:
        assert room.free_field_reference_level(83.4, 10.0) == pytest.approx(
            83.4, abs=1e-12
        )

    @pytest.mark.parametrize(
        ("distance", "expected"),
        [(5.0, -6.020599913279624), (3.0, -10.457574905606752), (20.0, 6.0205999132)],
    )
    def test_it_follows_the_inverse_square_law(
        self, distance: float, expected: float
    ) -> None:
        assert room.free_field_reference_level(0.0, distance) == pytest.approx(
            expected, abs=1e-9
        )

    def test_it_keeps_the_shape_it_is_given(self) -> None:
        out = room.free_field_reference_level([80.0, 81.0], 5.0)
        assert isinstance(out, np.ndarray)
        assert out.shape == (2,)

    @pytest.mark.parametrize(
        ("call", "scalar"),
        [
            (lambda: room.free_field_reference_level(80.0, 5.0), True),
            (lambda: room.free_field_reference_level([80.0], 5.0), False),
            (lambda: room.reverberation_room_reference_level(80.0, 10.0), True),
            (lambda: room.reverberation_room_reference_level([80.0], 10.0), False),
            (lambda: room.sound_strength_from_power(80.0, 100.0), True),
            (lambda: room.sound_strength_from_power([80.0], 100.0), False),
        ],
    )
    def test_a_scalar_in_gives_a_float_out(self, call, scalar: bool) -> None:  # noqa: ANN001
        # The level-domain functions keep the shape they are handed, so a
        # caller working in plain numbers is not handed a one-element array
        # to unwrap.
        assert isinstance(call(), float) is scalar

    def test_it_warns_inside_the_printed_minimum_distance(self) -> None:
        with pytest.warns(AuditoriumWarning, match="at least 3 m"):
            room.free_field_reference_level(90.0, 1.5)

    def test_it_refuses_a_distance_that_is_not_a_length(self) -> None:
        with pytest.raises(ValueError, match="positive, finite"):
            room.free_field_reference_level(90.0, 0.0)


class TestReverberationRoomReferenceLevel:
    """Equation (A.5), the diffuse-field route to the same reference."""

    def test_it_reproduces_a_hand_calculation(self) -> None:
        # A = 0,16 * 200 / 3,2 = 10 m^2 exactly, so 10 lg(A/S0) = 10 dB and
        # the reference sits 27 dB below the room level.
        area = 0.16 * 200.0 / 3.2
        assert room.reverberation_room_reference_level(80.0, area) == pytest.approx(
            53.0, abs=1e-12
        )

    def test_the_printed_offset_is_the_rounded_closed_form(self) -> None:
        assert room.DIFFUSE_FIELD_REFERENCE_OFFSET_DB == pytest.approx(
            round(EXACT_DIFFUSE_OFFSET_DB), abs=0.0
        )
        assert EXACT_DIFFUSE_OFFSET_DB == pytest.approx(37.012698553, abs=1e-9)

    def test_it_broadcasts_a_per_band_absorption_area(self) -> None:
        out = room.reverberation_room_reference_level([80.0, 80.0], [10.0, 100.0])
        assert np.asarray(out) == pytest.approx([53.0, 63.0], abs=1e-12)

    def test_it_refuses_a_non_positive_absorption_area(self) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            room.reverberation_room_reference_level(80.0, 0.0)


class TestSoundStrengthFromPower:
    """Equation (A.9), the route that needs no free-field measurement."""

    def test_it_is_the_printed_sum(self) -> None:
        assert room.sound_strength_from_power(70.0, 90.0) == pytest.approx(
            -20.0 + 31.0, abs=1e-12
        )

    def test_the_printed_offset_is_the_rounded_closed_form(self) -> None:
        assert room.SOUND_STRENGTH_POWER_OFFSET_DB == pytest.approx(
            round(EXACT_POWER_OFFSET_DB), abs=0.0
        )
        assert EXACT_POWER_OFFSET_DB == pytest.approx(30.992098640, abs=1e-9)

    def test_the_two_printed_routes_cannot_close_better_than_this(self) -> None:
        # Same source, same room, both printed routes. A reverberation room
        # of V = 200 m^3 and T = 2,0 s has A = 16 m^2 by (A.6), where a
        # source of L_W = 100 dB gives a diffuse level of L_W + 10 lg(4/A).
        power_level = 100.0
        area = 0.16 * 200.0 / 2.0
        diffuse_level = power_level + 10.0 * np.log10(4.0 / area)
        pressure_level = 80.0

        via_room = pressure_level - room.reverberation_room_reference_level(
            diffuse_level, area
        )
        via_power = room.sound_strength_from_power(pressure_level, power_level)

        # The exact offsets differ by 10 lg 4; the printed integers by 6.
        assert via_power - via_room == pytest.approx(
            10.0 * np.log10(4.0) - 6.0, abs=1e-12
        )
        assert via_power - via_room == pytest.approx(0.0205999132, abs=1e-9)

    def test_the_exact_offsets_do_close(self) -> None:
        assert EXACT_DIFFUSE_OFFSET_DB - EXACT_POWER_OFFSET_DB == pytest.approx(
            10.0 * np.log10(4.0), abs=1e-12
        )


class TestDirectivityEnergyAverage:
    """The note under Equation (A.4), and the step that does not divide 360."""

    def test_it_averages_one_band_per_row(self) -> None:
        # The rest of the module works band by band, so a survey of several
        # bands is one array, and the mean is taken along the bearings.
        angles = np.arange(29) * 2.0 * np.pi / 29.0
        one = 20.0 * np.log10(np.abs(np.cos(angles)) + 1e-300)
        banded = np.vstack([one, one + 10.0])
        assert room.directivity_energy_average(banded) == pytest.approx(
            [10.0 * np.log10(0.5), 10.0 * np.log10(0.5) + 10.0], abs=1e-12
        )
        assert room.directivity_energy_average(banded.T, axis=0) == pytest.approx(
            [10.0 * np.log10(0.5), 10.0 * np.log10(0.5) + 10.0], abs=1e-12
        )

    @pytest.mark.parametrize("bearings", [29, 30, 36, 72])
    def test_a_cosine_pattern_averages_to_exactly_half_its_on_axis_energy(
        self, bearings: int
    ) -> None:
        # sum cos^2(2 pi i / N) = N/2 for every N >= 3, so the energy mean
        # of a cosine directivity is 1/2 whatever the bearing count. That
        # makes this a test of the averaging, not of the sampling.
        angles = np.arange(bearings) * 2.0 * np.pi / bearings
        levels = 20.0 * np.log10(np.abs(np.cos(angles)) + 1e-300)
        assert room.directivity_energy_average(levels) == pytest.approx(
            10.0 * np.log10(0.5), abs=1e-12
        )

    def test_it_is_an_energy_mean_and_not_an_arithmetic_one(self) -> None:
        levels = np.full(29, 80.0)
        levels[0] = 100.0
        arithmetic = float(np.mean(levels))
        energetic = room.directivity_energy_average(levels)
        assert energetic > arithmetic
        assert energetic == pytest.approx(
            10.0 * np.log10((10.0**10.0 + 28.0 * 10.0**8.0) / 29.0), abs=1e-12
        )

    def test_it_refuses_a_turn_sampled_coarser_than_the_printed_step(self) -> None:
        # 28 bearings 12,857 degrees apart is coarser than the 12,5 the note
        # prints; 29 is the first count that is not.
        with pytest.raises(ValueError, match="at least 29"):
            room.directivity_energy_average(np.zeros(28))
        room.directivity_energy_average(np.zeros(29))


class TestPlot:
    """The renderer, checked for content rather than looked at."""

    def test_it_draws_the_strength_and_the_two_levels(self) -> None:
        pytest.importorskip("matplotlib")
        import matplotlib as mpl

        mpl.use("Agg")
        result = room.sound_strength(noisy_decay(1.0, 2.0), anechoic_ir(), FS)
        axes = result.plot()
        assert len(axes) == 2
        strength_line = next(
            line
            for line in axes[0].get_lines()
            if np.size(line.get_ydata()) == result.strength.size
        )
        assert strength_line.get_ydata() == pytest.approx(result.strength)
        assert len(axes[1].get_lines()) == 2
        assert axes[1].get_lines()[0].get_ydata() == pytest.approx(
            result.exposure_level
        )
        assert axes[1].get_lines()[1].get_ydata() == pytest.approx(
            result.reference_level
        )
        mpl.pyplot.close("all")

    def test_a_single_axes_gets_the_strength_panel_only(self) -> None:
        pytest.importorskip("matplotlib")
        import matplotlib as mpl
        from matplotlib import pyplot as plt

        mpl.use("Agg")
        result = room.sound_strength(noisy_decay(1.0, 2.0), anechoic_ir(), FS)
        _, ax = plt.subplots()
        assert result.plot(ax) is ax
        curve = next(
            line
            for line in ax.get_lines()
            if np.size(line.get_ydata()) == result.strength.size
        )
        assert curve.get_ydata() == pytest.approx(result.strength)
        plt.close("all")

    def test_the_band_axis_carries_the_nominal_band_labels(self) -> None:
        pytest.importorskip("matplotlib")
        import matplotlib as mpl
        from matplotlib import pyplot as plt

        mpl.use("Agg")
        result = room.sound_strength(noisy_decay(1.0, 2.0), anechoic_ir(), FS)
        _, ax = plt.subplots()
        result.plot(ax)
        labels = [tick.get_text() for tick in ax.get_xticklabels()]
        assert labels == ["125", "250", "500", "1k", "2k", "4k"]
        plt.close("all")

    def test_the_caller_can_restyle_the_curve(self) -> None:
        pytest.importorskip("matplotlib")
        import matplotlib as mpl
        from matplotlib import pyplot as plt

        mpl.use("Agg")
        result = room.sound_strength(noisy_decay(1.0, 2.0), anechoic_ir(), FS)
        _, ax = plt.subplots()
        result.plot(ax, color="k", label="mine")
        curve = next(line for line in ax.get_lines() if line.get_label() == "mine")
        assert curve.get_color() == "k"
        assert curve.get_ydata() == pytest.approx(result.strength)
        plt.close("all")

    def test_it_shades_the_table_a1_typical_range(self) -> None:
        pytest.importorskip("matplotlib")
        import matplotlib as mpl
        from matplotlib import pyplot as plt

        mpl.use("Agg")
        result = room.sound_strength(noisy_decay(1.0, 2.0), anechoic_ir(), FS)
        _, ax = plt.subplots()
        result.plot(ax)
        spans = [
            p for p in ax.patches if p.get_label().startswith(("Typical", "Rango"))
        ]
        assert len(spans) == 1
        bottom = float(spans[0].get_xy()[1])
        assert (bottom, bottom + float(spans[0].get_height())) == (-2.0, 10.0)
        plt.close("all")

    def test_the_spanish_edition_translates_every_visible_string(self) -> None:
        pytest.importorskip("matplotlib")
        import matplotlib as mpl
        from matplotlib import pyplot as plt

        mpl.use("Agg")
        result = room.sound_strength(noisy_decay(1.0, 2.0), anechoic_ir(), FS)
        axes = result.plot(language="es")
        assert "Fuerza sonora" in axes[0].get_title()
        assert "Frecuencia" in axes[1].get_xlabel()
        labels = [t.get_text() for t in axes[0].get_legend().get_texts()]
        assert any("Rango habitual del número único" in text for text in labels)
        assert any("Número único" in text for text in labels)
        plt.close("all")
