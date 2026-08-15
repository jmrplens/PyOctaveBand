#  Copyright (c) 2026. Jose Manuel Requena Plens
"""End-to-end tests over Ainslie's seven numeric sonar worked examples.

Oracle: Ainslie, *Principles of Sonar Performance Modelling* (Springer 2010).
Each example prints a table of sonar-equation terms in both decibel and linear
form and states the resulting figure of merit; the linear column is the
unrounded one, so it is what these tests feed in.

* §3.2.3.8 narrowband passive, Table 3.1 (printed p. 76) -- FOM 78.0 dB re m².
* §3.2.4.8 broadband passive, Table 3.2 (printed p. 90) -- FOM 79.0 dB re m².
* §3.3.3.8 active CW with a Doppler filter, Table 3.3 (printed p. 107) --
  FOM 82.7 dB re m², detection range 1.3 km, minimum detectable sphere
  a = 6.10 m (TS_E = 9.7 dB re m²).
* §3.3.4.8 the same problem with an energy detector, Table 3.4 (printed p. 122)
  -- DT 13.0 dB, detection range 0.9 km.
* §11.2.8 "NWP winter" narrowband passive, Table 11.3 (printed p. 587) --
  FOM 73.8 dB re m² through Equation (11.31).
* §11.3.7 "shallow-water sand" broadband passive, Table 11.5 (printed p. 604).
* §11.4.6 active broadband, orca versus salmon, Tables 11.6 and 11.7 (printed
  pp. 620 and 624) -- FOM_HT 59.0 and FOM_NL 51.0 dB re m², detection ranges
  241 / 149 / 117 m at wind speeds 2 / 6 / 10 m/s.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from phonometry.underwater.bioacoustics.audiograms import orca_audiogram
from phonometry.underwater.sonar_equation import (
    DetectionRangeResult,
    active_sonar_equation,
    detection_range,
    detection_range_from_curve,
    passive_sonar_equation,
)


def _db(linear: float) -> float:
    return float(10.0 * np.log10(linear))


# ---------------------------------------------------------------------------
# Chapter 3
# ---------------------------------------------------------------------------


def test_example_3_2_3_8_narrowband_passive_fom() -> None:
    """Table 3.1 linear column -> FOM_NB = SL + (AG − BW) − NLf − DT = 78.0 dB re m².

    S0 = 2.44e13 µPa²m², 3/2·c·W_Af^N = 9.26e5 µPa²/Hz, GA = 14.2, R50 = 23.8,
    Δf/N_FFT = 0.250 Hz (Equation 3.48, printed p. 63).
    """
    sl, nl_f, ag, dt, bw = _db(2.44e13), _db(9.26e5), _db(14.2), _db(23.8), _db(0.250)
    # In-band noise level = spectrum level + bandwidth term.
    res = passive_sonar_equation(sl, [0.0], nl_f + bw, directivity_index=ag,
                                 detection_threshold=dt)
    assert res.figure_of_merit == pytest.approx(sl + (ag - bw) - nl_f - dt, rel=1e-12)
    assert res.figure_of_merit == pytest.approx(78.0, abs=0.05)


def test_example_3_2_3_8_rounded_terms_reproduce_the_table() -> None:
    """The printed decibel column: 133.9, 59.7, 11.5, 13.8 and −6.0 dB."""
    assert _db(2.44e13) == pytest.approx(133.9, abs=0.05)
    assert _db(9.26e5) == pytest.approx(59.7, abs=0.05)
    assert _db(14.2) == pytest.approx(11.5, abs=0.05)
    assert _db(23.8) == pytest.approx(13.8, abs=0.05)
    assert _db(0.250) == pytest.approx(-6.0, abs=0.05)


def test_example_3_2_4_8_broadband_passive_fom() -> None:
    """Table 3.2 -> FOM_BB = SLf − (NLf − AGm) − DT = 79.0 dB re m² (Eq. 3.145)."""
    sl_f, nl_f, ag_m, dt = _db(1.22e10), _db(2.09e5), _db(19.0), _db(1.38e-2)
    assert dt == pytest.approx(-18.6, abs=0.05)
    res = passive_sonar_equation(sl_f, [0.0], nl_f, directivity_index=ag_m,
                                 detection_threshold=dt)
    assert res.figure_of_merit == pytest.approx(79.0, abs=0.06)


def test_example_3_3_3_8_active_cw_fom_and_minimum_sphere() -> None:
    """Table 3.3 -> FOM = (SL_E + TS_E + AG − NLf − DT)/2 = 82.7 dB re m².

    The minimum detectable target is a rigid sphere of radius a = 6.10 m,
    TS = 10·lg(a²/4) = 9.7 dB re m² (Equation 3.227, printed p. 106).
    """
    ts_e = 10.0 * np.log10(6.10**2 / 4.0)
    assert ts_e == pytest.approx(9.7, abs=0.05)
    sl_e, nl_f, ag, dt = _db(2.00e19), _db(1.97e4), _db(99.1), _db(26.8)
    res = active_sonar_equation(sl_e, [0.0], ts_e, nl_f, directivity_index=ag,
                                detection_threshold=dt)
    assert res.figure_of_merit == pytest.approx(82.7, abs=0.05)
    # The FOM is by definition the propagation loss at r50 = 1.3 km, and the
    # printed propagation factor there is F = 5.34e-9 m⁻².
    assert -10.0 * np.log10(5.34e-9) == pytest.approx(82.7, abs=0.05)


def test_example_3_3_3_8_detection_range() -> None:
    """r50 = 1.3 km at 50 kHz for FOM = 82.7 dB re m² (printed p. 107).

    Ainslie's own propagation model adds refraction detail this closed form
    does not carry, so the agreement is a few per cent rather than exact.
    """
    res = detection_range(82.7, 50e3)
    assert isinstance(res, DetectionRangeResult)
    assert res.detection_range == pytest.approx(1300.0, rel=0.05)


def test_example_3_3_4_8_energy_detector() -> None:
    """Table 3.4 -> DT = 13.0 dB (R50 = 20.1), 1.2 dB below the Doppler filter.

    "the in-beam noise level is 20 dB higher" and "the incoherent processing
    results in ... a decrease in DT50 of 1.2 dB", giving a signal-excess drop of
    about 19 dB (printed p. 117) and a detection range of 0.9 km.
    """
    dt_energy, dt_doppler = _db(20.1), _db(26.8)
    assert dt_energy == pytest.approx(13.0, abs=0.05)
    assert dt_doppler - dt_energy == pytest.approx(1.2, abs=0.05)
    signal_excess_drop = 20.0 - (dt_doppler - dt_energy)
    assert signal_excess_drop == pytest.approx(19.0, abs=0.3)
    # A signal-excess drop of 18.8 dB on a two-way path is 9.4 dB of one-way FOM.
    res = detection_range(82.7 - signal_excess_drop / 2.0, 50e3)
    assert res.detection_range == pytest.approx(900.0, rel=0.05)


# ---------------------------------------------------------------------------
# Chapter 11
# ---------------------------------------------------------------------------


def test_example_11_2_8_nwp_winter_fom() -> None:
    """Table 11.3 -> FOM_NB = 133.9 + (11.1 − (−6.0)) − 64.2 − 13.0 = 73.8 dB re m²."""
    res = passive_sonar_equation(133.9, [0.0], 64.2 + (-6.0), directivity_index=11.1,
                                 detection_threshold=13.0)
    assert res.figure_of_merit == pytest.approx(73.8, abs=1e-9)


def test_example_11_3_7_shallow_water_sand_fom() -> None:
    """Table 11.5 -> FOM_BB = 100.9 − (51.7 − 14.0) − (−6.6) = 69.8 dB re m²."""
    res = passive_sonar_equation(100.9, [0.0], 51.7, directivity_index=14.0,
                                 detection_threshold=-6.6)
    assert res.figure_of_merit == pytest.approx(69.8, abs=1e-9)
    # The source spectrum level is unchanged from the Chapter 3 broadband example.
    assert _db(1.22e10) == pytest.approx(100.9, abs=0.05)


def test_example_11_4_6_orca_hearing_limited_fom() -> None:
    """Table 11.6 -> FOM_HT = (SL + TS − HT)/2 = (198.2 − 29.0 − 51.2)/2 = 59.0."""
    ht = float(orca_audiogram(50e3).threshold[0])
    res = active_sonar_equation(198.2, [0.0], -29.0, ht)
    assert res.figure_of_merit == pytest.approx(59.0, abs=0.05)


def test_example_11_4_6_target_strength_of_the_salmon() -> None:
    """TS ≈ 10·lg L² − 27.1 with L = 0.8 m gives −29.0 dB re m² (Eq. 11.158)."""
    ts = 10.0 * np.log10(0.8**2) - 27.1
    assert ts == pytest.approx(-29.0, abs=0.05)


def test_example_11_4_6_noise_limited_fom() -> None:
    """Table 11.7 -> FOM = (SL + TS − NL + AG − DT)/2 = 51.0 dB re m² (Eq. 11.175)."""
    res = active_sonar_equation(198.2, [0.0], -29.0, 75.0, directivity_index=16.5,
                                detection_threshold=8.7)
    assert res.figure_of_merit == pytest.approx(51.0, abs=1e-9)


@pytest.mark.parametrize(
    ("wind_speed", "printed_range"),
    [(2.0, 241.0), (6.0, 149.0), (10.0, 117.0)],
)
def test_example_11_4_6_detection_ranges_versus_wind_speed(
    wind_speed: float, printed_range: float
) -> None:
    """Figure 11.22 caption: 117, 149 and 241 m for v10 = 10, 6 and 2 m/s.

    The wind-noise source factor scales as ``v̂^2.24`` (Equation 11.165), so
    ``NL`` rises by ``22.4·lg(v/2)`` dB above the 75.0 dB re µPa² of Table 11.7
    and the noise-limited figure of merit falls by half of that. Inverting the
    50 kHz propagation loss at that figure of merit reproduces the three
    published ranges to within a few per cent, the closed form standing in for
    Ainslie's broadband propagation model.
    """
    noise = 75.0 + 22.4 * np.log10(wind_speed / 2.0)
    fom = active_sonar_equation(198.2, [0.0], -29.0, noise, directivity_index=16.5,
                                detection_threshold=8.7).figure_of_merit
    res = detection_range(fom, 50e3)
    assert res.detection_range == pytest.approx(printed_range, rel=0.05)


# ---------------------------------------------------------------------------
# The detection-range solver itself
# ---------------------------------------------------------------------------


def test_detection_range_inverts_spherical_spreading_exactly() -> None:
    """With the absorption switched off by a very low frequency, r = 10^(FOM/20)."""
    res = detection_range(60.0, 1.0, model="thorp")
    assert res.detection_range == pytest.approx(1000.0, rel=1e-6)
    assert res.absorption_coefficient == pytest.approx(0.0, abs=1e-6)


def test_detection_range_matches_the_loss_it_inverts() -> None:
    res = detection_range(78.0, 300.0)
    from phonometry.underwater.propagation import propagation_loss

    at_root = propagation_loss(res.detection_range, 300.0).pl[0]
    assert at_root == pytest.approx(78.0, abs=1e-6)


def test_detection_range_is_infinite_when_the_loss_never_reaches_the_fom() -> None:
    assert detection_range(300.0, 100e3, max_range=1000.0).detection_range == float("inf")


def test_detection_range_saturates_at_zero_below_the_search_floor() -> None:
    """Spherical spreading is −60 dB at 1 mm, so any lower FOM has no root."""
    assert detection_range(-10.0, 1000.0).detection_range == pytest.approx(0.3162, abs=1e-3)
    assert detection_range(-100.0, 1000.0).detection_range == 0.0


def test_detection_range_from_curve_interpolates_the_crossing() -> None:
    ranges = np.array([100.0, 200.0, 300.0])
    losses = np.array([40.0, 50.0, 60.0])
    assert detection_range_from_curve(45.0, ranges, losses) == pytest.approx(150.0)


def test_detection_range_from_curve_handles_multiple_crossings() -> None:
    """Ainslie §11.2.8 warns that an oscillatory loss crosses the FOM repeatedly."""
    ranges = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    losses = np.array([40.0, 60.0, 40.0, 60.0, 70.0])
    assert detection_range_from_curve(50.0, ranges, losses, crossing="first") == pytest.approx(1.5)
    assert detection_range_from_curve(50.0, ranges, losses, crossing="last") == pytest.approx(3.5)


def test_detection_range_from_curve_without_a_crossing() -> None:
    """A loss that stays below the figure of merit is detectable past the grid."""
    assert detection_range_from_curve(
        60.0, [1.0, 2.0], [40.0, 50.0]
    ) == float("inf")
    # Descending through the figure of merit: still below it at the far end.
    assert detection_range_from_curve(
        60.0, [1.0, 2.0], [70.0, 50.0]
    ) == float("inf")


def test_both_solvers_return_zero_when_the_target_is_undetectable_everywhere() -> None:
    """The loss exceeds the figure of merit at every range: the range is 0, not ``inf``.

    ``inf`` is this module's value for "detectable at any range", so the
    opposite situation must not collapse onto it. The two solvers are pinned to
    the same answer on the same physical case: the closed form inverted below
    its own loss at the search floor, and the very curve that inversion
    returns fed back into the curve reader.
    """
    res = detection_range(-100.0, 1000.0)
    assert res.detection_range == 0.0
    assert float(res.propagation_loss.min()) > res.figure_of_merit
    assert detection_range_from_curve(
        res.figure_of_merit, res.range_m, res.propagation_loss
    ) == 0.0
    # And directly, on a plain rising curve entirely above the figure of merit.
    assert detection_range_from_curve(20.0, [1.0, 2.0, 3.0], [50.0, 58.0, 70.0]) == 0.0


def test_detection_range_from_curve_bridges_a_numerical_model() -> None:
    """The solver reads a detection range off a normal-mode propagation loss."""
    from phonometry.underwater.propagation.numerical import normal_modes

    depths = np.array([0.0, 100.0])
    speeds = np.array([1500.0, 1500.0])
    ranges = np.linspace(100.0, 20_000.0, 800)
    nm = normal_modes(200.0, depths, speeds, source_depth=40.0, receiver_depth=60.0,
                      ranges_m=ranges)
    got = detection_range_from_curve(60.0, nm.ranges, nm.propagation_loss)
    assert 100.0 < got < 20_000.0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_range": 0.5}, "max_range"),
        ({"n_points": 1}, "n_points"),
    ],
)
def test_detection_range_validates_its_arguments(
    kwargs: dict[str, float], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        detection_range(60.0, 1000.0, **kwargs)  # type: ignore[arg-type]


def test_detection_range_from_curve_validates_its_arguments() -> None:
    with pytest.raises(ValueError, match="propagation_loss"):
        detection_range_from_curve(60.0, [1.0, 2.0], [10.0])
    with pytest.raises(ValueError, match="range_m"):
        detection_range_from_curve(60.0, [2.0, 1.0], [10.0, 20.0])
    with pytest.raises(ValueError, match="crossing"):
        detection_range_from_curve(60.0, [1.0, 2.0], [10.0, 20.0], crossing="middle")


def test_detection_range_plot_returns_axes() -> None:
    res = detection_range(78.0, 300.0)
    assert res.plot() is not None
    assert res.plot(language="es") is not None
    plt.close("all")
