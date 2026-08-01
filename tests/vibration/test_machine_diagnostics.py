#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Tests for the kinematic machine fault frequencies.

M. P. Norton and D. G. Karczub, *Fundamentals of Noise and Vibration Analysis
for Engineers* (2nd ed., Cambridge University Press, 2003), Section 8.4
(Eqs. 8.3 to 8.20), with the published answers to problems 8.5, 8.6 and 8.7.

**Clean-room oracle.** Every expected number below is either a *printed answer*
from the book (Answers to problems, printed p. 619) or an exact kinematic
identity that the equations imply and the implementation does not:

* **Problem 8.5** (printed p. 597; answer p. 619). Rolling-contact bearing with
  a rotating inner race and a stationary outer race: fifteen rollers, 34 mm
  pitch diameter, 6 mm roller diameter, 12,96 degrees contact angle, 2000
  r/min. Printed answers ``fs = 33,33 Hz``, ``f_bcsor = 13,80 Hz``,
  ``f_re = 91,6 Hz``, ``f_repfo = 207 Hz``, ``f_resf = 183,3 Hz``,
  ``f_rciso = 19,53 Hz``, ``f_recri = 293 Hz``.
* **Problem 8.6** (printed p. 597; answer p. 619). Six-blade, four-vane axial
  fan at 3500 r/min: blade-passing frequency 350 Hz, first lobed interaction
  patterns at 35 Hz and 175 Hz.
* **Problem 8.7** (printed pp. 597-598; answer p. 619). Induction motor with
  sixty rotor bars, six magnetic poles, 3600 r/min and zero slip: 60 Hz,
  120 Hz, 360 Hz and 3600 Hz.
* **Identities.** ``BPFO + BPFI = Z fs`` for any geometry (adding Eqs. 8.8 and
  8.9 cancels the ``(d/D) cos phi`` term); ``BPFO = Z FTF`` with a stationary
  outer race; ``BDF = 2 BSF`` (Eq. 8.10 is twice Eq. 8.7); ``FTF + FTF_rel =
  fs`` (Eq. 8.5 plus Eq. 8.11); and the slot harmonic of Eq. (8.20) at
  ``n = 1`` equals the rotor-bar passing rate ``R fs`` at any slip.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from phonometry.vibration.machine_diagnostics import (
    FaultFrequencyResult,
    bearing_fault_frequencies,
    blade_pass_frequencies,
    combine_fault_lines,
    gear_mesh_frequencies,
    induction_motor_frequencies,
    shaft_rate,
)

#: Complete input geometry of Norton problem 8.5.
_P85 = {
    "speed_rpm": 2000.0,
    "n_elements": 15,
    "element_diameter": 6.0,
    "pitch_diameter": 34.0,
    "contact_angle_deg": 12.96,
}


class TestNortonProblem85:
    """Bearing fault frequencies against the printed answers to problem 8.5."""

    @pytest.fixture(scope="class")
    def result(self) -> FaultFrequencyResult:
        return bearing_fault_frequencies(**_P85)  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        ("name", "expected", "decimals"),
        [
            ("shaft", 33.33, 2),  # fs
            ("FTF", 13.80, 2),  # f_bcsor
            ("BSF", 91.6, 1),  # f_re (91,65 rounded down in the book)
            ("BPFO", 207.0, 0),  # f_repfo
            ("BDF", 183.3, 1),  # f_resf
            ("FTF_rel", 19.53, 2),  # f_rciso
            ("BPFI", 293.0, 0),  # f_recri
        ],
    )
    def test_printed_answer(
        self, result: FaultFrequencyResult, name: str, expected: float, decimals: int
    ) -> None:
        """Each line matches the printed answer at its printed precision."""
        # The book prints f_re = 91.6 and f_resf = 183.3 = 2 x 91.65, so its own
        # rounding of f_re is one unit low; allow the half-digit there only.
        tolerance = 0.55 * 10.0**-decimals if name == "BSF" else 0.5 * 10.0**-decimals
        assert result[name] == pytest.approx(expected, abs=tolerance)

    def test_line_names_and_families(self, result: FaultFrequencyResult) -> None:
        """The seven Shahan & Kamperman lines are present and classified."""
        assert result.names == (
            "shaft",
            "FTF",
            "FTF_rel",
            "BSF",
            "BDF",
            "BPFO",
            "BPFI",
        )
        assert result.source == "rolling-contact bearing"
        families = {line.name: line.family for line in result.lines}
        assert families["shaft"] == "shaft"
        assert all(families[n] == "bearing" for n in result.names if n != "shaft")

    def test_orders(self, result: FaultFrequencyResult) -> None:
        """Shaft orders are the frequencies divided by the shaft rate."""
        np.testing.assert_allclose(
            result.orders, result.frequencies / result.shaft_rate, rtol=1e-12
        )
        assert result["BPFO"] / result.shaft_rate == pytest.approx(6.21, abs=5e-3)


class TestBearingIdentities:
    """Exact kinematic identities implied by Eqs. 8.4 to 8.14."""

    @pytest.mark.parametrize(
        ("z", "d", "dm", "phi"),
        [(15, 6.0, 34.0, 12.96), (8, 7.94, 39.0, 0.0), (12, 11.0, 71.5, 40.0)],
    )
    def test_pass_frequencies_sum_to_z_times_shaft(
        self, z: int, d: float, dm: float, phi: float
    ) -> None:
        """``BPFO + BPFI = Z fs`` exactly (Eqs. 8.8 + 8.9)."""
        res = bearing_fault_frequencies(
            1770.0, z, d, dm, contact_angle_deg=phi
        )
        assert res["BPFO"] + res["BPFI"] == pytest.approx(z * res.shaft_rate, rel=1e-12)

    def test_pass_outer_is_z_times_cage(self) -> None:
        """``BPFO = Z FTF`` with a stationary outer race (Eq. 8.8 = Z x 8.5)."""
        res = bearing_fault_frequencies(**_P85)  # type: ignore[arg-type]
        assert res["BPFO"] == pytest.approx(15 * res["FTF"], rel=1e-12)

    def test_spin_is_twice_element_rotation(self) -> None:
        """``BDF = 2 BSF`` (Eq. 8.10 is twice Eq. 8.7)."""
        res = bearing_fault_frequencies(**_P85)  # type: ignore[arg-type]
        assert res["BDF"] == pytest.approx(2.0 * res["BSF"], rel=1e-12)

    def test_cage_rates_add_to_shaft_rate(self) -> None:
        """``FTF + FTF_rel = fs`` (Eq. 8.5 + Eq. 8.11)."""
        res = bearing_fault_frequencies(**_P85)  # type: ignore[arg-type]
        assert res["FTF"] + res["FTF_rel"] == pytest.approx(res.shaft_rate, rel=1e-12)

    def test_rotating_race_moves_only_the_cage(self) -> None:
        """Eqs. 8.8/8.14 and 8.9/8.13 are identical, so BPFO/BPFI do not move."""
        inner = bearing_fault_frequencies(**_P85)  # type: ignore[arg-type]
        outer = bearing_fault_frequencies(rotating_race="outer", **_P85)  # type: ignore[arg-type]
        assert outer["BPFO"] == pytest.approx(inner["BPFO"], rel=1e-12)
        assert outer["BPFI"] == pytest.approx(inner["BPFI"], rel=1e-12)
        assert outer["FTF"] == pytest.approx(inner["FTF_rel"], rel=1e-12)

    def test_zero_contact_angle_radial_bearing(self) -> None:
        """With ``phi = 0`` the ratio ``d/D`` alone sets the lines (Eq. 8.5)."""
        res = bearing_fault_frequencies(600.0, 9, 5.0, 25.0)
        g = 5.0 / 25.0
        assert res["FTF"] == pytest.approx(0.5 * 10.0 * (1.0 - g), rel=1e-12)

    def test_only_the_diameter_ratio_enters(self) -> None:
        """Millimetres or metres give the same frequencies."""
        mm = bearing_fault_frequencies(**_P85)  # type: ignore[arg-type]
        m = bearing_fault_frequencies(
            2000.0, 15, 0.006, 0.034, contact_angle_deg=12.96
        )
        np.testing.assert_allclose(m.frequencies, mm.frequencies, rtol=1e-12)


class TestNortonProblem86:
    """Axial-fan blade-pass and lobed interaction patterns (problem 8.6)."""

    def test_blade_passing_frequency(self) -> None:
        """Printed answer: 350 Hz for six blades at 3500 r/min (Eq. 8.15)."""
        res = blade_pass_frequencies(3500.0, 6)
        assert res["BPF"] == pytest.approx(350.0, abs=0.05)
        assert res["2xBPF"] == pytest.approx(700.0, abs=0.1)
        assert res["3xBPF"] == pytest.approx(1050.0, abs=0.1)

    def test_lobed_interaction_patterns(self) -> None:
        """Printed answers: 35 Hz and 175 Hz (Eqs. 8.16 and 8.17).

        With ``N = 6`` blades and ``V = 4`` vanes the first patterns have
        ``mL = 6 - 4 = 2`` and ``mL = 6 + 4 = 10`` lobes, rotating at
        ``n N (r/min) / mL`` = 10 500 and 2100 r/min, i.e. 175 Hz and 35 Hz.
        """
        res = blade_pass_frequencies(3500.0, 6, harmonics=1, n_vanes=4)
        assert res["lobe n=1 m=2"] == pytest.approx(175.0, abs=0.05)
        assert res["lobe n=1 m=10"] == pytest.approx(35.0, abs=0.05)

    def test_equal_lobe_counts_from_different_harmonics_both_survive(self) -> None:
        """Eq. (8.17) carries ``n``, so ``mL`` alone does not identify a pattern.

        With ``N = 4`` blades and ``V = 2`` vanes, ``n = 1, k = +1`` and
        ``n = 2, k = -1`` both give ``mL = 6``, but Eq. (8.17) puts the first
        at ``1 x 4 fs / 6 = 0,667 fs`` and the second at ``2 x 4 fs / 6 =
        1,333 fs``. Two distinct patterns, two distinct speeds.
        """
        res = blade_pass_frequencies(60.0, 4, harmonics=2, n_vanes=2)
        fs = res.shaft_rate
        assert res["lobe n=1 m=6"] == pytest.approx(4.0 * fs / 6.0, rel=1e-12)
        assert res["lobe n=2 m=6"] == pytest.approx(8.0 * fs / 6.0, rel=1e-12)
        assert res["lobe n=2 m=6"] != pytest.approx(res["lobe n=1 m=6"])
        # Eq. (8.16) with n <= 2 and |k| = 1 has exactly these four patterns
        # (mL = 2, 6, 6 and 10), and none of them is silently dropped.
        lobes = [line.name for line in res.lines if line.name.startswith("lobe")]
        assert lobes == [
            "lobe n=1 m=2", "lobe n=1 m=6", "lobe n=2 m=6", "lobe n=2 m=10",
        ]

    def test_rotary_blower_repeats_four_times_per_revolution(self) -> None:
        """Norton's rule: use Eq. 8.15 with ``4 x r/min`` (Section 8.4.4)."""
        blower = blade_pass_frequencies(4.0 * 900.0, 1, harmonics=1)
        assert blower["BPF"] == pytest.approx(60.0, rel=1e-12)


class TestNortonProblem87:
    """Induction-motor discrete frequencies (problem 8.7)."""

    def test_printed_answers(self) -> None:
        """60 Hz, 120 Hz, 360 Hz and 3600 Hz at zero slip."""
        res = induction_motor_frequencies(3600.0, 6, 60, slip=0.0)
        assert res["1x"] == pytest.approx(60.0, rel=1e-12)
        assert res["2x"] == pytest.approx(120.0, rel=1e-12)
        assert res["2fe"] == pytest.approx(360.0, rel=1e-12)
        assert res["fsh"] == pytest.approx(3600.0, rel=1e-12)

    def test_zero_slip_matches_norton_eq_819(self) -> None:
        """``fe = fs p / 2`` when ``s = 0`` (Eq. 8.19)."""
        res = induction_motor_frequencies(3600.0, 6, 60, slip=0.0)
        assert res["fe"] == pytest.approx(60.0 * 6 / 2.0, rel=1e-12)
        assert "FP" not in res

    @pytest.mark.parametrize("slip", [0.0, 0.02, 0.05])
    def test_slot_harmonic_is_the_rotor_bar_passing_rate(self, slip: float) -> None:
        """Eq. (8.20) at ``n = 1`` reduces to ``R fs`` for any slip."""
        res = induction_motor_frequencies(1750.0, 4, 52, slip=slip)
        assert res["fsh"] == pytest.approx(52.0 * 1750.0 / 60.0, rel=1e-10)

    def test_supply_frequency_and_slip_are_interchangeable(self) -> None:
        """Giving ``fe`` recovers the same lines as giving the matching slip."""
        by_supply = induction_motor_frequencies(
            1750.0, 4, 52, supply_frequency=60.0
        )
        implied_slip = 1.0 - (1750.0 / 60.0) / (2.0 * 60.0 / 4.0)
        by_slip = induction_motor_frequencies(1750.0, 4, 52, slip=implied_slip)
        np.testing.assert_allclose(
            by_supply.frequencies, by_slip.frequencies, rtol=1e-12
        )
        assert by_supply["fe"] == pytest.approx(60.0, rel=1e-12)

    def test_pole_pass_and_slip_lines_under_load(self) -> None:
        """``FP = p f_slip`` with the slip frequency ``f_sync - fs``."""
        res = induction_motor_frequencies(1750.0, 4, 52, supply_frequency=60.0)
        f_slip = 2.0 * 60.0 / 4.0 - 1750.0 / 60.0
        assert res["f_slip"] == pytest.approx(f_slip, rel=1e-12)
        assert res["FP"] == pytest.approx(4.0 * f_slip, rel=1e-12)

    def test_slot_sidebands_straddle_the_slot_harmonic(self) -> None:
        """Dynamic eccentricity: sidebands at +/- fs and +/- the slip rate."""
        res = induction_motor_frequencies(
            1750.0, 4, 52, supply_frequency=60.0, sidebands=1
        )
        fsh, fs = res["fsh"], res.shaft_rate
        assert res["fsh+1x"] == pytest.approx(fsh + fs, rel=1e-12)
        assert res["fsh-1x"] == pytest.approx(fsh - fs, rel=1e-12)
        assert res["fsh+1s"] + res["fsh-1s"] == pytest.approx(2.0 * fsh, rel=1e-12)


class TestGearMesh:
    """Gear-meshing frequency and sidebands (Eq. 8.3)."""

    def test_mesh_frequency_and_harmonics(self) -> None:
        """``GMF = N x r/min / 60`` with integer harmonics."""
        res = gear_mesh_frequencies(1500.0, 28, harmonics=3)
        assert res["GMF"] == pytest.approx(28 * 25.0, rel=1e-12)
        np.testing.assert_allclose(
            res.harmonics("GMF", 3), [700.0, 1400.0, 2100.0], rtol=1e-12
        )

    def test_sidebands_are_spaced_by_the_modulating_rate(self) -> None:
        """A once-per-revolution fault puts sidebands at ``k GMF +/- m fs``."""
        res = gear_mesh_frequencies(1500.0, 28, harmonics=2, sidebands=2)
        assert res["GMF+1x"] - res["GMF"] == pytest.approx(25.0, rel=1e-12)
        assert res["GMF"] - res["GMF-2x"] == pytest.approx(50.0, rel=1e-12)
        assert res["2xGMF+1x"] == pytest.approx(1425.0, rel=1e-12)

    def test_mating_wheel_modulation_rate(self) -> None:
        """*sideband_rate* modulates at the mating wheel's shaft rate."""
        res = gear_mesh_frequencies(
            1500.0, 28, harmonics=1, sidebands=1, sideband_rate=17.5
        )
        assert res["GMF+1x"] - res["GMF"] == pytest.approx(17.5, rel=1e-12)

    def test_non_positive_sidebands_are_dropped(self) -> None:
        """Lines at or below 0 Hz never enter the family."""
        res = gear_mesh_frequencies(60.0, 2, harmonics=1, sidebands=3)
        assert all(line.frequency > 0.0 for line in res.lines)


class TestResultHelpers:
    """The result container's selection and merging helpers."""

    def test_within_keeps_the_analysis_span(self) -> None:
        res = bearing_fault_frequencies(**_P85)  # type: ignore[arg-type]
        kept = res.within(100.0, 250.0)
        assert set(kept.names) == {"BDF", "BPFO"}
        assert kept.shaft_rate == res.shaft_rate

    def test_as_dict_and_membership(self) -> None:
        res = bearing_fault_frequencies(**_P85)  # type: ignore[arg-type]
        assert res.as_dict()["BPFO"] == res["BPFO"]
        assert "BPFO" in res
        assert "GMF" not in res

    def test_combine_merges_families(self) -> None:
        """A gearbox bearing carries bearing, gear and shaft lines at once."""
        bearing = bearing_fault_frequencies(1500.0, 9, 7.94, 39.0)
        gear = gear_mesh_frequencies(1500.0, 28, harmonics=1)
        both = combine_fault_lines(bearing, gear)
        assert both["BPFO"] == bearing["BPFO"]
        assert both["GMF"] == gear["GMF"]
        # The duplicated "shaft" line is disambiguated, never dropped.
        assert "shaft (gear pair)" in both.names
        assert len(both.lines) == len(bearing.lines) + len(gear.lines)

    def test_shaft_rate_helper(self) -> None:
        assert shaft_rate(1500.0) == pytest.approx(25.0, rel=1e-12)


class TestValidation:
    """Every invalid input is rejected with a ValueError."""

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"speed_rpm": 0.0},
            {"n_elements": 0},
            {"element_diameter": 0.0},
            {"pitch_diameter": 0.0},
            {"contact_angle_deg": -1.0},
            {"contact_angle_deg": 90.0},
            {"rotating_race": "middle"},
        ],
    )
    def test_bearing_rejects(self, kwargs: dict[str, object]) -> None:
        with pytest.raises(ValueError):
            bearing_fault_frequencies(**{**_P85, **kwargs})  # type: ignore[arg-type]

    def test_bearing_rejects_element_larger_than_pitch(self) -> None:
        with pytest.raises(ValueError, match="smaller than"):
            bearing_fault_frequencies(2000.0, 15, 40.0, 34.0)

    @pytest.mark.parametrize(
        "kwargs",
        [{"n_teeth": 0}, {"harmonics": 0}, {"sidebands": -1}, {"sideband_rate": 0.0}],
    )
    def test_gear_rejects(self, kwargs: dict[str, object]) -> None:
        with pytest.raises(ValueError):
            gear_mesh_frequencies(1500.0, **{"n_teeth": 28, **kwargs})  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"poles": 3},
            {"poles": 0},
            {"rotor_bars": 0},
            {"slip": 1.0},
            {"slip": -0.1},
            {"slot_harmonics": 0},
            {"sidebands": -1},
        ],
    )
    def test_motor_rejects(self, kwargs: dict[str, object]) -> None:
        with pytest.raises(ValueError):
            induction_motor_frequencies(1750.0, **{"poles": 4, "rotor_bars": 52, **kwargs})  # type: ignore[arg-type]

    def test_motor_rejects_supply_below_shaft_rate(self) -> None:
        """A shaft turning at or above synchronous speed is not an induction motor."""
        with pytest.raises(ValueError, match="synchronous"):
            induction_motor_frequencies(3600.0, 4, 52, supply_frequency=60.0)

    @pytest.mark.parametrize(
        "kwargs",
        [{"n_blades": 0}, {"harmonics": 0}, {"n_vanes": 0}, {"lobe_orders": 0}],
    )
    def test_blade_rejects(self, kwargs: dict[str, object]) -> None:
        with pytest.raises(ValueError):
            blade_pass_frequencies(3500.0, **{"n_blades": 6, **kwargs})  # type: ignore[arg-type]

    def test_unknown_line_name(self) -> None:
        res = bearing_fault_frequencies(**_P85)  # type: ignore[arg-type]
        with pytest.raises(KeyError, match="BPFO"):
            _ = res["not-a-line"]

    def test_within_rejects_inverted_span(self) -> None:
        res = bearing_fault_frequencies(**_P85)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="greater than"):
            res.within(500.0, 100.0)

    def test_harmonics_rejects_zero_count(self) -> None:
        res = bearing_fault_frequencies(**_P85)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="positive integer"):
            res.harmonics("BPFO", 0)

    def test_combine_rejects_no_results(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            combine_fault_lines()

    def test_combine_rejects_mismatched_shaft_rates(self) -> None:
        bearing = bearing_fault_frequencies(1500.0, 9, 7.94, 39.0)
        gear = gear_mesh_frequencies(1800.0, 28)
        with pytest.raises(ValueError, match="same shaft rate"):
            combine_fault_lines(bearing, gear)


class TestEnvelopeChainIntegration:
    """The predicted lines land on a synthesised outer-race defect signature.

    The oracle is the *signal*, not the implementation: an impulse train at
    Norton's printed ``f_repfo = 207 Hz`` amplitude-modulates a 3 kHz
    structural resonance. The envelope spectrum of that record must show its
    largest line at the predicted ``BPFO``, which is the whole point of the
    module.
    """

    def test_envelope_spectrum_peaks_on_the_predicted_bpfo(self) -> None:
        from phonometry.signals.envelope import envelope_spectrum

        res = bearing_fault_frequencies(**_P85)  # type: ignore[arg-type]
        bpfo = res["BPFO"]
        fs_rate, duration = 20000.0, 2.0
        t = np.arange(int(fs_rate * duration)) / fs_rate
        # Impacts at 1/BPFO, each ringing a 3 kHz resonance with a 3 ms decay.
        impacts = np.zeros_like(t)
        for k in range(int(duration * bpfo)):
            idx = round(k / bpfo * fs_rate)
            if idx < impacts.size:
                impacts[idx] = 1.0
        tau = np.arange(int(0.003 * fs_rate)) / fs_rate
        ring = np.exp(-tau / 6e-4) * np.sin(2.0 * math.pi * 3000.0 * tau)
        signal = np.convolve(impacts, ring)[: t.size]

        spec = envelope_spectrum(signal, fs_rate, band=(2000.0, 4000.0))
        band = spec.frequencies <= 3.0 * bpfo
        peak = float(spec.frequencies[band][np.argmax(spec.amplitude[band])])
        assert peak == pytest.approx(bpfo, rel=0.01)
        # The prediction is within one analysis bin of the measured line.
        assert abs(peak - bpfo) <= 2.0 * float(np.diff(spec.frequencies)[0])
