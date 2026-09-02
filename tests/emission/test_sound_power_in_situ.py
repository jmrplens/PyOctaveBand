#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Sound power and sound energy in situ by comparison: ISO 3747:2010.

Normative anchors (ISO 3747:2010):
- Background correction, Eq. (7): K1i = -10 lg(1 - 10^(-0,1 dLpi)), with the
  8.1 rules: dL > 15 dB -> K1 = 0; 6 dB <= dL <= 15 dB -> Eq. (7); dL < 6 dB ->
  the correction is capped at 1,3 dB and the result is an upper bound.
- Mean corrected levels, Eq. (8)-(10): energy means over the positions.
- Sound power level, Eq. (11): LW = LW(RSS) - Lp(RSS) + Lp(ST); m locations,
  Eq. (12): energy means over j of LWj(RSS) and of Lpj(RSS).
- Single events, Eq. (13)-(18); sound energy level Eq. (19)-(20).
- Uncertainty, Eq. (22)-(23): sigma_tot = sqrt(sigma_R0^2 + sigma_omc^2),
  U = k sigma_tot; Table 2: sigma_R0 = 1,5 dB (grade 2), 4,0 dB (grade 3).
- 9.5 EXAMPLE: grade 2, sigma_omc = 2,0 dB, k = 2 -> U = 2 sqrt(1,5^2 + 2^2)
  = 5 dB.
- Annex A, Eq. (A.1): dLf(r) = Lp(RSS),r - LW(RSS) + 11 dB + 20 lg(r/r0).
- Annex C: C2 = -10 lg(ps/ps0) + 15 lg((273,15 + theta)/296); Eq. (C.2):
  ps = ps0 (1 - a Ha)^b, a = 2,2560e-5 1/m, b = 5,2553.
- Annex D, Table D.1: Ck = -26,2 -16,1 -8,6 -3,2 0,0 1,2 1,0 -1,1 dB for the
  octaves 63 Hz to 8 kHz; Eq. (D.1): LWA = 10 lg sum 10^(0,1 (LWk + Ck)).
"""

from __future__ import annotations

import dataclasses
import warnings

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from phonometry import emission

FREQS = np.array([125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0])
#: Calibrated octave-band sound power of the reference source, dB re 1 pW.
LW_RSS = np.array([87.0, 90.5, 92.5, 93.8, 94.0, 93.0, 90.0])
#: Time-averaged levels of the source under test at four positions, dB.
ST = np.array(
    [
        [80.1, 83.4, 85.0, 84.2, 81.0, 76.5, 70.2],
        [79.0, 82.8, 84.6, 83.9, 80.4, 75.8, 69.5],
        [81.2, 84.0, 85.9, 85.0, 81.9, 77.1, 70.9],
        [80.5, 83.1, 85.3, 84.5, 81.3, 76.2, 70.0],
    ]
)
#: Time-averaged levels of the reference source at the same positions, dB.
RSS = np.array(
    [
        [78.5, 81.9, 83.7, 84.9, 84.8, 83.5, 79.8],
        [77.9, 81.2, 83.1, 84.3, 84.1, 82.9, 79.2],
        [79.3, 82.6, 84.4, 85.5, 85.4, 84.1, 80.3],
        [78.8, 82.1, 83.9, 85.0, 85.0, 83.7, 79.9],
    ]
)
#: Background at every position, 10 dB or more below both sources, dB.
BACKGROUND = np.array([68.0, 70.0, 72.0, 71.0, 66.0, 60.0, 55.0])
#: Table D.1, k = 2..8 (the 63 Hz row is not in this example's range).
CK_TABLE_D1 = np.array([-16.1, -8.6, -3.2, 0.0, 1.2, 1.0, -1.1])


def _energy_mean(levels: np.ndarray, axis: int) -> np.ndarray:
    """10 lg[(1/n) sum 10^(0,1 L)] along ``axis`` (Eq. 8 to 10, 12, 15, 18)."""
    return np.asarray(10.0 * np.log10(np.mean(10.0 ** (0.1 * levels), axis=axis)))


def _k1(delta: float) -> float:
    """Eq. (7) evaluated by hand for a margin inside 6 dB to 15 dB."""
    return float(-10.0 * np.log10(1.0 - 10.0 ** (-0.1 * delta)))


def _power(**kwargs: object) -> emission.InSituSoundPowerResult:
    return emission.sound_power_in_situ(ST, RSS, LW_RSS, FREQS, **kwargs)  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# Eq. (7) and the 8.1 rules
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("margin", "expected"),
    [
        # -10 lg(1 - 10^-0,6) = -10 lg(1 - 0,251 189) = -10 lg 0,748 811
        (6.0, 1.2563),
        # -10 lg(1 - 0,1) = -10 lg 0,9
        (10.0, 0.4576),
        # -10 lg(1 - 10^-1,5) = -10 lg(1 - 0,031 623) = -10 lg 0,968 377
        (15.0, 0.1396),
    ],
)
def test_background_correction_follows_eq7_between_6_and_15_db(
    margin: float, expected: float
) -> None:
    """K1i at one position and band is Eq. (7) of the margin at that position."""
    levels = ST.copy()
    background = np.full_like(ST, 40.0)
    background[1, 3] = levels[1, 3] - margin
    res = _power(background_levels=background)
    assert res.background_correction[1, 3] == pytest.approx(expected, abs=5e-5)
    assert res.background_correction[1, 3] == pytest.approx(_k1(margin), abs=1e-9)
    assert bool(np.all(res.background_requirement_met))


def test_background_margin_above_15_db_needs_no_correction() -> None:
    """8.1: dL > 15 dB -> K1i = 0 at that position, and the band stays valid."""
    background = np.full_like(ST, 40.0)
    background[2, 0] = ST[2, 0] - 15.5
    res = _power(background_levels=background)
    assert res.background_correction[2, 0] == 0.0
    assert bool(np.all(res.background_requirement_met))


def test_background_margin_below_6_db_is_capped_and_flagged() -> None:
    """8.1: dL < 6 dB -> the correction is at most 1,3 dB and the band is an
    upper bound; a margin of 5,9 dB is still below the cap (Eq. 7 gives
    1,290 dB there) and a margin of 3 dB, or none at all, takes the cap.
    """
    background = np.full_like(ST, 40.0)
    background[0, 1] = ST[0, 1] - 5.9
    background[3, 2] = ST[3, 2] - 3.0
    background[1, 4] = ST[1, 4] + 2.0  # louder than the source
    res = _power(background_levels=background)
    assert res.background_correction[0, 1] == pytest.approx(_k1(5.9), abs=1e-9)
    assert res.background_correction[0, 1] < 1.3
    assert res.background_correction[3, 2] == pytest.approx(1.3)
    assert res.background_correction[1, 4] == pytest.approx(1.3)
    assert list(res.background_requirement_met) == [
        True,
        False,
        False,
        True,
        False,
        True,
        True,
    ]
    assert np.all(np.isfinite(res.sound_power_level))


def test_reference_source_background_is_corrected_per_location_and_position() -> None:
    """Eq. (10) prints K1i(RSS) without a location index, but its margin
    depends on L'pji(RSS): each (j, i) carries its own correction, and the
    source's background is reused when the reference brought none (7.5).
    """
    background = np.full_like(ST, 40.0)
    background[0, 5] = RSS[0, 5] - 8.0
    two = np.stack([RSS, RSS + 2.0])
    res = emission.sound_power_in_situ(
        ST, two, LW_RSS, FREQS, background_levels=background
    )
    assert res.background_correction_ref.shape == (2, 4, 7)
    assert res.background_correction_ref[0, 0, 5] == pytest.approx(_k1(8.0))
    assert res.background_correction_ref[1, 0, 5] == pytest.approx(_k1(10.0))
    assert res.background_correction_ref[0, 1, 5] == 0.0
    own = emission.sound_power_in_situ(
        ST,
        two,
        LW_RSS,
        FREQS,
        background_levels=background,
        background_levels_ref=40.0 + np.zeros(7),
    )
    assert np.all(own.background_correction_ref == 0.0)


# --------------------------------------------------------------------------
# Eq. (8) to (12)
# --------------------------------------------------------------------------
def test_sound_power_level_is_eq11_of_the_energy_means() -> None:
    """LW = LW(RSS) - Lp(RSS) + Lp(ST) with Eq. (8) and (9) re-derived here."""
    res = _power()
    lp_st = _energy_mean(ST, axis=0)
    lp_rss = _energy_mean(RSS, axis=0)
    np.testing.assert_allclose(res.mean_source_level, lp_st, atol=1e-12)
    np.testing.assert_allclose(res.mean_reference_level, lp_rss, atol=1e-12)
    np.testing.assert_allclose(res.reference_power_level, LW_RSS, atol=1e-12)
    np.testing.assert_allclose(
        res.sound_power_level, LW_RSS - lp_rss + lp_st, atol=1e-12
    )
    assert res.quantity == "power"
    assert np.all(np.isnan(res.sound_energy_level))
    assert np.isnan(res.sound_energy_level_a)


def test_eq12_with_identical_locations_collapses_to_eq11() -> None:
    """The energy mean of m equal terms is the term: Eq. (12) with the RSS at
    m indistinguishable locations is Eq. (11), whichever way lw_ref is given.
    """
    one = _power(background_levels=BACKGROUND)
    three = emission.sound_power_in_situ(
        ST,
        np.stack([RSS, RSS, RSS]),
        np.stack([LW_RSS, LW_RSS, LW_RSS]),
        FREQS,
        background_levels=BACKGROUND,
    )
    np.testing.assert_allclose(
        three.sound_power_level, one.sound_power_level, atol=1e-12
    )
    assert three.reference_levels.shape == (3, 7)
    np.testing.assert_allclose(three.reference_levels[1], one.reference_levels[0])


def test_eq12_energy_averages_the_locations_separately() -> None:
    """Two locations 6 dB apart in both the calibrated power and the measured
    level: the two energy means are taken before the subtraction, so LW is
    the single-location value (the 6 dB cancels term by term), not the value
    of either location's arithmetic.
    """
    res = emission.sound_power_in_situ(
        ST, np.stack([RSS, RSS + 6.0]), np.stack([LW_RSS, LW_RSS + 6.0]), FREQS
    )
    one = _power()
    np.testing.assert_allclose(res.sound_power_level, one.sound_power_level, atol=1e-12)
    # and the two energy means themselves are the printed 10 lg[(1 + 10^0,6)/2]
    lift = 10.0 * np.log10((1.0 + 10.0**0.6) / 2.0)
    np.testing.assert_allclose(res.reference_power_level, LW_RSS + lift, atol=1e-12)
    np.testing.assert_allclose(
        res.mean_reference_level, one.mean_reference_level + lift, atol=1e-12
    )


def test_eq11_agrees_with_the_iso3741_comparison_method() -> None:
    """ISO 3741:2010 Eq. (21), LW = LW(RSS) + (Lp(ST) - Lp(RSS) + C2), is
    Eq. (11) plus the Annex C correction; with no background the two
    determinations share every other term, so the reference-condition level
    of this module must equal the reverberation-room result exactly.
    """
    with warnings.catch_warnings():  # four positions trip the ISO 3741 advisory
        warnings.simplefilter("ignore", emission.SoundPowerWarning)
        room = emission.sound_power_comparison(
            ST, RSS, LW_RSS, frequencies=FREQS, temperature=20.0, static_pressure=100.0
        )
    res = _power(temperature=20.0, static_pressure=100.0)
    np.testing.assert_allclose(
        res.sound_power_level_ref, room.sound_power_level, atol=1e-12
    )
    assert res.c2 == pytest.approx(room.c2)


# --------------------------------------------------------------------------
# Single events, Eq. (13) to (20)
# --------------------------------------------------------------------------
def test_identical_events_one_at_a_time_give_their_own_level() -> None:
    """Eq. (15) over N equal levels is that level; Eq. (18) then averages the
    positions and Eq. (19) is Eq. (11) with LE(ST) in place of Lp(ST).
    """
    events = np.repeat(ST[:, None, :], 5, axis=1)
    res = emission.sound_energy_in_situ(events, RSS, LW_RSS, FREQS)
    lp_rss = _energy_mean(RSS, axis=0)
    np.testing.assert_allclose(
        res.mean_source_level, _energy_mean(ST, axis=0), atol=1e-12
    )
    np.testing.assert_allclose(
        res.sound_energy_level, LW_RSS - lp_rss + _energy_mean(ST, axis=0), atol=1e-12
    )
    assert res.quantity == "energy"
    assert np.all(np.isnan(res.sound_power_level))
    assert np.isnan(res.sound_power_level_a)


def test_n_events_in_one_measurement_reduce_by_10_lg_n() -> None:
    """Eq. (17): one measurement encompassing N events at L + 10 lg N per
    position is the same determination as N events one at a time at L.
    """
    one_at_a_time = emission.sound_energy_in_situ(
        np.repeat(ST[:, None, :], 8, axis=1), RSS, LW_RSS, FREQS
    )
    encompassing = emission.sound_energy_in_situ(
        ST + 10.0 * np.log10(8.0), RSS, LW_RSS, FREQS, events=8
    )
    np.testing.assert_allclose(
        encompassing.sound_energy_level, one_at_a_time.sound_energy_level, atol=1e-12
    )


def test_event_levels_are_corrected_event_by_event() -> None:
    """Eq. (13)/(14): with two events 10 dB apart over the same background,
    the quieter one carries the larger K1; the per-position value reported
    is the shift the corrections leave on the Eq. (15) mean.
    """
    loud = ST[0] + 10.0
    quiet = ST[0]
    events = np.stack([np.stack([quiet, loud])] * 4)  # (4, 2, 7)
    background = np.stack([ST[0] - 8.0] * 4)  # margins 8 dB and 18 dB
    with warnings.catch_warnings():  # two events trip the N >= 5 advisory
        warnings.simplefilter("ignore", emission.SoundPowerWarning)
        res = emission.sound_energy_in_situ(
            events, RSS, LW_RSS, FREQS, background_levels=background
        )
    corrected = np.stack([quiet - _k1(8.0), loud])  # 18 dB: no correction
    expected_mean = _energy_mean(corrected, axis=0)
    raw_mean = _energy_mean(np.stack([quiet, loud]), axis=0)
    np.testing.assert_allclose(res.background_correction[0], raw_mean - expected_mean)
    np.testing.assert_allclose(res.mean_source_level, expected_mean, atol=1e-12)


def test_integration_time_carries_the_background_to_the_event_interval() -> None:
    """Eq. (14) as printed subtracts the time-averaged background; with the
    event integrated over T = 10 s the same background holds 10 lg 10 = 10 dB
    more energy (3.4 NOTE 1), so a printed margin of 16 dB is really 6 dB.
    """
    events = np.repeat(ST[:, None, :], 5, axis=1)
    background = ST - 16.0
    printed = emission.sound_energy_in_situ(
        events, RSS, LW_RSS, FREQS, background_levels=background
    )
    unit = emission.sound_energy_in_situ(
        events, RSS, LW_RSS, FREQS, background_levels=background, integration_time=1.0
    )
    ten = emission.sound_energy_in_situ(
        events, RSS, LW_RSS, FREQS, background_levels=background, integration_time=10.0
    )
    assert np.all(printed.background_correction == 0.0)
    np.testing.assert_array_equal(
        unit.background_correction, printed.background_correction
    )
    np.testing.assert_allclose(ten.background_correction, _k1(6.0), atol=1e-9)
    assert bool(np.all(ten.background_requirement_met))


def test_eq20_with_identical_locations_collapses_to_eq19() -> None:
    events = np.repeat(ST[:, None, :], 5, axis=1)
    one = emission.sound_energy_in_situ(events, RSS, LW_RSS, FREQS)
    two = emission.sound_energy_in_situ(events, np.stack([RSS, RSS]), LW_RSS, FREQS)
    np.testing.assert_allclose(
        two.sound_energy_level, one.sound_energy_level, atol=1e-12
    )


# --------------------------------------------------------------------------
# Clause 9 and Table 2
# --------------------------------------------------------------------------
def test_clause_9_5_example_expanded_uncertainty_is_5_db() -> None:
    """9.5 EXAMPLE: grade 2, sigma_omc = 2,0 dB, k = 2, sigma_R0 = 1,5 dB from
    Table 2: U = 2 sqrt(1,5^2 + 2^2) dB = 2 sqrt(6,25) dB = 5 dB, exactly.
    """
    res = _power(
        excess_levels=[8.0, 9.5, 7.2, 8.8], directivity_range=3.0, sigma_omc=2.0
    )
    assert res.grade == "engineering"
    assert res.sigma_r0 == 1.5
    assert res.sigma_omc == 2.0
    assert res.sigma_tot == pytest.approx(2.5)
    assert res.expanded_uncertainty == pytest.approx(5.0)
    assert res.coverage_factor == 2.0


@pytest.mark.parametrize(
    ("sigma_omc", "expected"),
    [(0.5, 1.6), (2.0, 2.5), (4.0, 4.3)],
)
def test_table_e1_grade_2_row(sigma_omc: float, expected: float) -> None:
    """Table E.1, sigma_R0 = 1,5 dB row: sqrt(2,25 + 0,25) = 1,58 -> 1,6;
    sqrt(2,25 + 4) = 2,5; sqrt(2,25 + 16) = 4,27 -> 4,3 (printed to 0,1 dB).
    """
    res = _power(
        excess_levels=[8.0, 9.5, 7.2, 8.8], directivity_range=3.0, sigma_omc=sigma_omc
    )
    assert round(res.sigma_tot, 1) == expected


def test_one_sided_coverage_factor_scales_eq23() -> None:
    """9.1: k = 1,6 for the one-sided 95 % comparison with a limit value."""
    res = _power(
        excess_levels=[8.0, 9.5, 7.2, 8.8],
        directivity_range=3.0,
        sigma_omc=2.0,
        coverage_factor=1.6,
    )
    assert res.expanded_uncertainty == pytest.approx(1.6 * 2.5)


def test_uncertainty_is_nan_without_sigma_omc() -> None:
    res = _power(excess_levels=[8.0, 9.5, 7.2, 8.8], directivity_range=3.0)
    assert res.sigma_r0 == 1.5
    assert np.isnan(res.sigma_omc)
    assert np.isnan(res.sigma_tot)
    assert np.isnan(res.expanded_uncertainty)


@pytest.mark.parametrize(
    ("excess", "directivity", "grade", "sigma_r0"),
    [
        ([7.0, 7.0, 7.0, 7.0], 7.0, "engineering", 1.5),  # both at the limit
        ([7.0, 6.9, 7.0, 7.0], 7.0, "survey", 4.0),  # one position short
        ([7.0, 7.0, 7.0, 7.0], 7.1, "survey", 4.0),  # too directional
        (None, 3.0, "survey", 4.0),  # dLfA not determined
        ([9.0, 9.0, 9.0, 9.0], None, "survey", 4.0),  # directivity not surveyed
    ],
)
def test_table_2_grades(
    excess: list[float] | None, directivity: float | None, grade: str, sigma_r0: float
) -> None:
    """Table 2: grade 2 needs dLfA >= 7 dB at every position and a directivity
    range within +/-7 dB; anything else, or an indicator not determined, is
    grade 3 with sigma_R0 = 4,0 dB.
    """
    res = _power(excess_levels=excess, directivity_range=directivity)
    assert res.grade == grade
    assert res.sigma_r0 == sigma_r0


# --------------------------------------------------------------------------
# Annex A, C, D
# --------------------------------------------------------------------------
def test_excess_level_is_zero_in_a_free_field_and_the_reverberant_surplus() -> None:
    """Eq. (A.1) with Lp = LW - 11 - 20 lg(r/r0) (the spherical free field)
    returns 0; 7 dB more pressure returns the 7 dB the method requires.
    """
    lw, r = 92.0, 4.0
    free = lw - 11.0 - 20.0 * np.log10(r)
    assert emission.excess_sound_pressure_level(free, lw, r) == pytest.approx(
        0.0, abs=1e-12
    )
    assert emission.excess_sound_pressure_level(free + 7.0, lw, r) == pytest.approx(7.0)
    traverse = emission.excess_sound_pressure_level([free, free + 3.0], lw, [r, r])
    assert isinstance(traverse, np.ndarray)
    np.testing.assert_allclose(traverse, [0.0, 3.0], atol=1e-12)


def test_c2_at_reference_conditions_is_the_296_k_residual() -> None:
    """Annex C prints theta_ref = 296 K beside a 23,0 °C reference: at exactly
    101,325 kPa and 23,0 °C, C2 = 15 lg(296,15/296) = 15 x 0,000 220 03
    = 0,003 300 4 dB, not zero; the pressure term vanishes.
    """
    res = _power()
    assert res.c2 == pytest.approx(0.0033004, abs=5e-8)
    np.testing.assert_allclose(
        res.sound_power_level_ref, res.sound_power_level + res.c2
    )


def test_static_pressure_from_altitude_eq_c2() -> None:
    """Eq. (C.2) at 500 m: 1 - 2,2560e-5 x 500 = 0,988 72; ln = -0,011 344 1;
    x 5,2553 = -0,059 617; exp = 0,942 126; ps = 95,461 kPa, and the Annex C
    pressure term -10 lg(ps/ps0) is then 0,258 9 dB. At sea level ps = ps0.
    """
    assert emission.static_pressure_from_altitude(0.0) == pytest.approx(101.325)
    ps = emission.static_pressure_from_altitude(500.0)
    assert ps == pytest.approx(95.461, abs=5e-4)
    assert -10.0 * np.log10(ps / 101.325) == pytest.approx(0.2589, abs=5e-5)
    below = emission.static_pressure_from_altitude(-400.0)
    assert below > 101.325


def test_a_weighted_total_uses_table_d1() -> None:
    """Eq. (D.1) with the printed Ck of Table D.1 typed in here."""
    res = _power(background_levels=BACKGROUND)
    expected = 10.0 * np.log10(
        np.sum(10.0 ** (0.1 * (res.sound_power_level + CK_TABLE_D1)))
    )
    assert res.sound_power_level_a == pytest.approx(expected, abs=1e-12)
    events = np.repeat(ST[:, None, :], 5, axis=1)
    energy = emission.sound_energy_in_situ(events, RSS, LW_RSS, FREQS)
    expected_j = 10.0 * np.log10(
        np.sum(10.0 ** (0.1 * (energy.sound_energy_level + CK_TABLE_D1)))
    )
    assert energy.sound_energy_level_a == pytest.approx(expected_j, abs=1e-12)


def test_63_hz_row_of_table_d1_is_accepted() -> None:
    freqs = np.array([63.0, 125.0])
    res = emission.sound_power_in_situ(ST[:, :2], RSS[:, :2], LW_RSS[:2], freqs)
    expected = 10.0 * np.log10(
        np.sum(10.0 ** (0.1 * (res.sound_power_level + np.array([-26.2, -16.1]))))
    )
    assert res.sound_power_level_a == pytest.approx(expected, abs=1e-12)


# --------------------------------------------------------------------------
# Advisories
# --------------------------------------------------------------------------
def test_conforming_shapes_raise_no_warning() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _power(background_levels=BACKGROUND)
        emission.sound_energy_in_situ(
            np.repeat(ST[:, None, :], 5, axis=1), RSS, LW_RSS, FREQS
        )


def test_fewer_than_three_positions_warns() -> None:
    with pytest.warns(emission.SoundPowerWarning, match="7.4.1"):
        emission.sound_power_in_situ(ST[:2], RSS[:2], LW_RSS, FREQS)


def test_fewer_than_five_events_warns() -> None:
    with pytest.warns(emission.SoundPowerWarning, match="7.6"):
        emission.sound_energy_in_situ(ST, RSS, LW_RSS, FREQS, events=4)


# --------------------------------------------------------------------------
# Refusals
# --------------------------------------------------------------------------
def test_levels_must_be_a_2d_grid() -> None:
    with pytest.raises(ValueError, match="'levels' must be a non-empty 2D"):
        emission.sound_power_in_situ(ST[0], RSS, LW_RSS, FREQS)


def test_levels_must_be_finite() -> None:
    bad = ST.copy()
    bad[1, 1] = np.nan
    with pytest.raises(ValueError, match="'levels' must be finite"):
        emission.sound_power_in_situ(bad, RSS, LW_RSS, FREQS)


def test_levels_ref_must_match_positions_and_bands() -> None:
    with pytest.raises(ValueError, match="'levels_ref' must be measured at the same"):
        emission.sound_power_in_situ(ST, RSS[:3], LW_RSS, FREQS)


def test_lw_ref_must_match_bands_or_locations() -> None:
    with pytest.raises(ValueError, match="'lw_ref' must carry one value per band"):
        emission.sound_power_in_situ(ST, RSS, LW_RSS[:5], FREQS)
    two_locations = np.stack([RSS, RSS])
    three_powers = np.stack([LW_RSS] * 3)
    with pytest.raises(ValueError, match="'lw_ref' must be one spectrum, or one per"):
        emission.sound_power_in_situ(ST, two_locations, three_powers, FREQS)


def test_frequencies_must_be_table_d1_octaves() -> None:
    with pytest.raises(ValueError, match="'frequencies' must carry one value per band"):
        emission.sound_power_in_situ(ST, RSS, LW_RSS, FREQS[:5])
    thirds = FREQS.copy()
    thirds[2] = 630.0
    with pytest.raises(ValueError, match="'frequencies' must be nominal octave"):
        emission.sound_power_in_situ(ST, RSS, LW_RSS, thirds)


def test_background_shapes_are_checked_by_name() -> None:
    with pytest.raises(
        ValueError, match="'background_levels' must carry one value per band"
    ):
        _power(background_levels=BACKGROUND[:4])
    three_rows = np.stack([BACKGROUND] * 3)
    five_rows = np.stack([BACKGROUND] * 5)
    with pytest.raises(
        ValueError, match="'background_levels' must be one spectrum or one per"
    ):
        _power(background_levels=three_rows)
    with pytest.raises(
        ValueError, match="'background_levels_ref' must be one spectrum or one per"
    ):
        _power(background_levels_ref=five_rows)


@pytest.mark.parametrize("theta", [-273.0, -300.0, float("nan")])
def test_temperature_is_validated(theta: float) -> None:
    with pytest.raises(ValueError, match="'temperature' must be finite and greater"):
        _power(temperature=theta)


def test_static_pressure_is_validated() -> None:
    with pytest.raises(
        ValueError, match="'static_pressure' must be finite and positive"
    ):
        _power(static_pressure=0.0)


def test_grade_indicators_are_validated() -> None:
    with pytest.raises(
        ValueError, match="'excess_levels' must carry one finite value per"
    ):
        _power(excess_levels=[8.0, 9.0], directivity_range=3.0)
    with pytest.raises(
        ValueError, match="'directivity_range' must be finite and non-negative"
    ):
        _power(excess_levels=[8.0, 9.0, 8.0, 8.0], directivity_range=-1.0)


def test_uncertainty_inputs_are_validated() -> None:
    with pytest.raises(ValueError, match="'sigma_omc' must be finite and non-negative"):
        _power(sigma_omc=-0.5)
    with pytest.raises(ValueError, match="'coverage_factor' must be positive"):
        _power(sigma_omc=2.0, coverage_factor=0.0)


def test_events_argument_is_checked_against_the_form() -> None:
    three_d = np.repeat(ST[:, None, :], 5, axis=1)
    with pytest.raises(ValueError, match="'events' is counted from the second axis"):
        emission.sound_energy_in_situ(three_d, RSS, LW_RSS, FREQS, events=5)
    with pytest.raises(ValueError, match="'events' must be a positive integer"):
        emission.sound_energy_in_situ(ST, RSS, LW_RSS, FREQS)
    with pytest.raises(ValueError, match="'events' must be a positive integer"):
        emission.sound_energy_in_situ(ST, RSS, LW_RSS, FREQS, events=0)
    with pytest.raises(ValueError, match="'event_levels' must be a non-empty 2D or 3D"):
        emission.sound_energy_in_situ(ST[0], RSS, LW_RSS, FREQS, events=5)


def test_integration_time_must_be_positive() -> None:
    three_d = np.repeat(ST[:, None, :], 5, axis=1)
    with pytest.raises(ValueError, match="'integration_time' must be positive"):
        emission.sound_energy_in_situ(
            three_d,
            RSS,
            LW_RSS,
            FREQS,
            background_levels=BACKGROUND,
            integration_time=0.0,
        )


def test_altitude_is_validated() -> None:
    with pytest.raises(ValueError, match="'altitude' must be finite and below"):
        emission.static_pressure_from_altitude(50_000.0)
    not_a_number = float("nan")
    with pytest.raises(ValueError, match="'altitude' must be finite and below"):
        emission.static_pressure_from_altitude(not_a_number)


def test_excess_level_inputs_are_validated() -> None:
    with pytest.raises(ValueError, match="'distance' must be finite and positive"):
        emission.excess_sound_pressure_level(70.0, 92.0, 0.0)
    not_a_number = float("nan")
    unbounded = float("inf")
    with pytest.raises(ValueError, match="'level' must be finite"):
        emission.excess_sound_pressure_level(not_a_number, 92.0, 2.0)
    with pytest.raises(ValueError, match="'lw_ref' must be finite"):
        emission.excess_sound_pressure_level(70.0, unbounded, 2.0)


def test_result_refuses_disagreeing_shapes() -> None:
    res = _power()
    with pytest.raises(ValueError, match="'sound_power_level'"):
        dataclasses.replace(res, sound_power_level=res.sound_power_level[:-1])
    with pytest.raises(ValueError, match="'background_correction_ref'"):
        dataclasses.replace(
            res, background_correction_ref=res.background_correction_ref[:, :3, :]
        )
    two_locations = np.stack([res.reference_levels[0]] * 2)
    with pytest.raises(ValueError, match="'reference_levels'"):
        dataclasses.replace(res, reference_levels=two_locations)
    with pytest.raises(ValueError, match="'quantity' must be one of"):
        dataclasses.replace(res, quantity="intensity")
    with pytest.raises(ValueError, match="'grade' must be one of"):
        dataclasses.replace(res, grade="precision")


# --------------------------------------------------------------------------
# Plot
# --------------------------------------------------------------------------
def test_plot_draws_one_bar_per_band_and_hatches_upper_bounds() -> None:
    background = np.full_like(ST, 40.0)
    background[3, 2] = ST[3, 2] - 3.0
    res = _power(background_levels=background)
    ax = res.plot()
    heights = [p.get_height() for p in ax.patches]
    np.testing.assert_allclose(heights, res.sound_power_level)
    hatched = [p.get_hatch() for p in ax.patches]
    assert hatched[2] == "//"
    assert hatched[0] is None
    assert f"{res.sound_power_level_a:.1f}" in ax.get_title()
    assert ax.get_legend() is not None
    plt.close("all")


def test_plot_of_an_energy_result_labels_lj_in_spanish() -> None:
    events = np.repeat(ST[:, None, :], 5, axis=1)
    res = emission.sound_energy_in_situ(events, RSS, LW_RSS, FREQS)
    ax = res.plot(language="es")
    assert "$L_J$" in ax.get_ylabel()
    assert "energía" in ax.get_ylabel()
    assert "ISO 3747" in ax.get_title()
    assert ax.get_legend() is None
    heights = [p.get_height() for p in ax.patches]
    np.testing.assert_allclose(heights, res.sound_energy_level)
    plt.close("all")


def test_plot_rejects_an_unknown_language() -> None:
    res = _power()
    with pytest.raises(ValueError, match="Unknown language"):
        res.plot(language="xx")
