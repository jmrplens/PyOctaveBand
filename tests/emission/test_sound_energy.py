#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Sound energy level of noise bursts and transient emissions: ISO 3744:2010
clause 8.3 (ISO 3746:2010 clause 8.4) over an enveloping surface and
ISO 3741:2010 clause 9.2 in a reverberation room, with the Annex E/F
A-weighting and octave sums and the ISO 3744 Annex G reference atmosphere.

Neither standard prints a worked example with L_E or L_J (the only EXAMPLEs
are the uncertainty ones of ISO 3744 9.5 and ISO 3741 10.5), so the oracle is
closed form, derived from clause 3 of each standard and written out here:

- ISO 3744 3.3/3.4 (= ISO 3741 3.3/3.4): L_{p,T} = 10 lg[(1/T) int p^2 dt / p0^2]
  and L_E = 10 lg[int p^2 dt / E0], E0 = (20 uPa)^2 s = p0^2 * 1 s. For p^2
  constant over T the integral is T p^2, so L_E = L_{p,T} + 10 lg(T/T0),
  T0 = 1 s, which is what 3.4 NOTE 1 prints.
- ISO 3744 3.22/3.23 (= ISO 3747 Eq. 5/6): J = int P(t) dt, L_J = 10 lg(J/J0),
  J0 = 1 pJ, and P0 = 1 pW (3.21), so a steady source of power P radiating for
  T seconds has L_J = L_W + 10 lg(T/T0).
- Every equation of ISO 3744 8.3 is its 8.2 twin with L_E in place of L_p
  ((19)<->(12), (21)<->(16), (22)<->(17), (23)<->(18)), and every equation of
  ISO 3741 9.2 its 9.1 twin ((22)/(24)<->(13), (25)<->(14), (26)<->(15),
  (27)<->(16), (30)<->(20), (31)<->(21)); the corrections K1, K2, A, C1, C2 and
  the Waterhouse term are the same, so with the same margins L_J = L_W + 10 lg(T/T0).

Arithmetic anchors, computed by hand and not by the library:
10 lg(2 pi 2^2) = 14.0024 dB; 10 lg 5 = 6.9897 dB; 10 lg 3 = 4.7712 dB;
K1(6 dB) = -10 lg(1 - 10^-0.6) = 1.2563 dB, K1(10 dB) = 0.4576 dB,
K1(3 dB) = 3.0206 dB; C1(23 C, 101.325 kPa) = 5 lg(296.15/314) = -0.1271 dB;
C2(23 C) = 15 lg(296.15/296) = +0.0033 dB; Eq. (G.2) at 500 m:
101.325 (1 - 2.2560e-5 * 500)^5.2553 = 95.46 kPa.
"""

from __future__ import annotations

import warnings

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from phonometry import emission

_FREQS = np.array([250.0, 500.0, 1000.0, 2000.0])
#: ISO 3744 Table E.1 A-weighting band corrections Ck at the four bands above.
_CK = np.array([-8.6, -3.2, 0.0, 1.2])


def _energy_mean(levels: np.ndarray, axis: int = 0) -> np.ndarray:
    """10 lg[(1/N) sum 10^(0.1 L)] written out, the printed Eq. (12)/(19)."""
    return 10.0 * np.log10(np.mean(10.0 ** (0.1 * np.asarray(levels)), axis=axis))


def _k1(delta: float) -> float:
    """-10 lg(1 - 10^(-0.1 dL)), the printed Eq. (16)/(21)."""
    return -10.0 * np.log10(1.0 - 10.0 ** (-0.1 * delta))


# --------------------------------------------------------------------------
# ISO 3744 8.3: the identity with the sound power chain
# --------------------------------------------------------------------------
def test_steady_source_identity_lj_equals_lw_plus_10lg_t() -> None:
    """A steady source measured for T = 10 s: L_J = L_W + 10 lg(T/T0) exactly.

    Every position level is raised by 10 lg(T/T0) = 10 dB (3.4 NOTE 1), the
    background is the same time-averaged spectrum in both chains (measured
    over the same T, 8.3.1) and the room the same; K1, K2, the A-weighted
    total and the uncertainty must then coincide and the levels differ by
    exactly 10 dB, band by band.
    """
    rng = np.random.default_rng(3744)
    lp = np.array([70.0, 74.0, 78.0, 75.0]) + rng.normal(0.0, 0.5, (10, 4))
    bg = np.array([58.0, 60.0, 62.0, 61.0])
    room = emission.RoomEnvironment(absorption_area=150.0)
    lw = emission.sound_power_pressure(
        lp, "hemisphere", radius=2.0, background_levels=bg, frequencies=_FREQS,
        room=room, omc_uncertainty=1.0,
    )  # fmt: skip
    lj = emission.sound_energy_pressure(
        lp + 10.0, "hemisphere", radius=2.0, background_levels=bg,
        integration_time=10.0, frequencies=_FREQS, room=room, omc_uncertainty=1.0,
    )  # fmt: skip
    assert isinstance(lj, emission.SoundEnergyResult)
    np.testing.assert_allclose(lj.sound_energy_level, lw.sound_power_level + 10.0)
    np.testing.assert_allclose(lj.background_correction, lw.background_correction)
    np.testing.assert_allclose(lj.environmental_correction, lw.environmental_correction)
    np.testing.assert_allclose(lj.surface_event_level, lw.surface_pressure_level + 10.0)
    np.testing.assert_allclose(lj.mean_event_level, lw.mean_pressure_level + 10.0)
    np.testing.assert_allclose(lj.directivity_index, lw.directivity_index)
    assert lj.sound_energy_level_a == pytest.approx(lw.sound_power_level_a + 10.0)
    assert lj.uncertainty == pytest.approx(lw.uncertainty)
    assert lj.surface_area == pytest.approx(lw.surface_area)
    assert lj.grade == "engineering"
    assert lj.events is None
    assert lj.integration_time == pytest.approx(10.0)


def test_monopole_hemisphere_recovers_lj_eq23() -> None:
    """Eq. (23) inverted: L_E = L_J - 10 lg(2 pi r^2) on r = 2 m gives L_J back.

    10 lg(2 pi 4) = 14.0024 dB by hand.
    """
    lj_true = 95.0
    surface_term = 10.0 * np.log10(2.0 * np.pi * 2.0**2)  # S = 2 pi r^2, clause 7.2.3
    assert surface_term == pytest.approx(14.0024, abs=5e-5)
    le = lj_true - surface_term
    res = emission.sound_energy_pressure(np.full((10, 1), le), "hemisphere", radius=2.0)
    assert res.sound_energy_level[0] == pytest.approx(lj_true, abs=1e-9)
    assert res.sound_energy_level_a == pytest.approx(lj_true, abs=1e-9)
    assert res.surface_area == pytest.approx(2.0 * np.pi * 4.0)


def test_surface_and_free_field_defaults_match_the_power_path() -> None:
    """No background and no room: K1 = K2 = 0 and L_J = mean + 10 lg(S/S0)."""
    levels = np.full((10, 2), 80.0)
    res = emission.sound_energy_pressure(levels, "hemisphere", radius=1.0)
    np.testing.assert_allclose(res.background_correction, 0.0)
    np.testing.assert_allclose(res.environmental_correction, 0.0)
    np.testing.assert_allclose(
        res.sound_energy_level, 80.0 + 10.0 * np.log10(2.0 * np.pi)
    )
    np.testing.assert_allclose(res.directivity_index, 0.0)
    assert np.isnan(res.sound_energy_level_a)  # two bands, no frequencies


# --------------------------------------------------------------------------
# ISO 3744 8.3.2: Eq. (19) and Eq. (20)
# --------------------------------------------------------------------------
def test_one_measurement_of_n_events_subtracts_10lg_ne_eq20() -> None:
    """Eq. (20): one level encompassing Ne = 5 events, less 10 lg 5 = 6.9897 dB."""
    levels = np.full((10, 1), 90.0)
    res = emission.sound_energy_pressure(levels, "hemisphere", radius=2.0, events=5)
    assert res.mean_event_level[0] == pytest.approx(90.0 - 6.989700, abs=1e-5)
    assert res.events == 5


def test_per_event_axis_is_energy_averaged_eq19() -> None:
    """Eq. (19): five events on the first axis, energy-averaged by hand."""
    rng = np.random.default_rng(19)
    events = 88.0 + rng.normal(0.0, 2.0, (5, 10, 2))
    res = emission.sound_energy_pressure(events, "hemisphere", radius=2.0)
    expected = _energy_mean(_energy_mean(events, axis=0), axis=0)
    np.testing.assert_allclose(res.mean_event_level, expected)
    assert res.events == 5


def test_two_acquisition_modes_agree_for_identical_events() -> None:
    """Ne equal events one at a time (Eq. 19) and one measurement of the Ne
    events (Eq. 20) describe the same burst: the encompassing level is the
    energy sum L + 10 lg Ne of the single ones, so both give L back.
    """
    single = np.full((10, 3), 84.0)
    one_at_a_time = np.repeat(single[np.newaxis], 5, axis=0)
    encompassing = single + 10.0 * np.log10(5.0)
    a = emission.sound_energy_pressure(one_at_a_time, "hemisphere", radius=1.5)
    b = emission.sound_energy_pressure(encompassing, "hemisphere", radius=1.5, events=5)
    np.testing.assert_allclose(a.sound_energy_level, b.sound_energy_level)
    np.testing.assert_allclose(a.mean_event_level, 84.0)


def test_mean_single_event_level_standalone() -> None:
    """The public Eq. (19)/(20) helper on its own, in both modes."""
    per_event = np.array([80.0, 86.0, 83.0, 80.0, 86.0])
    got = emission.mean_single_event_level(per_event)
    assert float(got) == pytest.approx(float(_energy_mean(per_event)))
    got2 = emission.mean_single_event_level(np.array([90.0, 92.0]), events=5)
    np.testing.assert_allclose(got2, [90.0 - 6.989700, 92.0 - 6.989700], atol=1e-5)


def test_fewer_than_five_events_warns() -> None:
    """Ne is at least five (8.3.1); three events are accepted with a warning."""
    with pytest.warns(emission.SoundPowerWarning, match="at least 5"):
        emission.mean_single_event_level(np.full((3, 2), 80.0))
    with pytest.warns(emission.SoundPowerWarning, match="at least 5"):
        emission.sound_energy_pressure(
            np.full((10, 1), 80.0), "hemisphere", radius=1.0, events=2
        )


def test_per_event_array_with_events_raises() -> None:
    with pytest.raises(ValueError, match="'levels_positions' already carries"):
        emission.sound_energy_pressure(
            np.full((5, 10, 1), 80.0), "hemisphere", radius=1.0, events=5
        )


def test_bad_rank_and_bad_events_raise() -> None:
    with pytest.raises(ValueError, match="'levels_positions' must be a 2D"):
        emission.sound_energy_pressure(np.zeros((2, 2, 2, 2)), "hemisphere", radius=1.0)
    with pytest.raises(ValueError, match="'events' must be a positive integer"):
        emission.mean_single_event_level(np.zeros(3), events=0)
    with pytest.raises(ValueError, match="'events' must be an integer"):
        emission.mean_single_event_level(np.zeros(3), events=2.5)  # type: ignore[arg-type]
    with pytest.raises(
        ValueError, match="'levels' must carry one entry per single event"
    ):
        emission.mean_single_event_level(np.zeros((0, 3)))
    with pytest.raises(ValueError, match="'levels' must contain only finite"):
        emission.mean_single_event_level(np.array([80.0, np.nan]))


# --------------------------------------------------------------------------
# ISO 3744 8.3.4: background over the same integration time
# --------------------------------------------------------------------------
def test_k1_compares_the_background_as_its_exposure_over_t() -> None:
    """A 70 dB time-averaged background over T = 10 s has the exposure
    70 + 10 lg 10 = 80 dB (3.4 NOTE 1). Against an 86 dB event the margin is
    6 dB, the engineering criterion, so K1 is exactly its 1.2563 dB and no
    warning is raised. Read literally, 86 - 70 = 16 dB would give K1 = 0 and
    a 1.26 dB overestimate of the burst's energy.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error", emission.SoundPowerWarning)
        res = emission.sound_energy_pressure(
            np.full((10, 1), 86.0),
            "hemisphere",
            radius=1.0,
            background_levels=np.full((10, 1), 70.0),
            integration_time=10.0,
        )
    assert res.background_correction[0] == pytest.approx(_k1(6.0), abs=1e-9)
    assert res.background_correction[0] == pytest.approx(1.2563, abs=5e-5)


def test_k1_below_criterion_clamps_and_warns() -> None:
    """A margin of 4 dB after the exposure is clamped to K1(6 dB) with a warning."""
    with pytest.warns(emission.SoundPowerWarning, match="Background margin below 6 dB"):
        res = emission.sound_energy_pressure(
            np.full((10, 1), 84.0),
            "hemisphere",
            radius=1.0,
            background_levels=np.full(1, 70.0),
            integration_time=10.0,
        )
    assert res.background_correction[0] == pytest.approx(_k1(6.0), abs=1e-9)


def test_k1_survey_grade_uses_iso3746_criteria() -> None:
    """ISO 3746 8.4.2: 3 dB <= dL_EA <= 10 dB applies Eq. (15); at 3 dB the
    correction is K1(3) = 3.0206 dB, and above 10 dB it is zero.
    """
    at_criterion = emission.sound_energy_pressure(
        np.full((4, 1), 83.0), "hemisphere", radius=1.0, grade="survey",
        background_levels=np.full(1, 80.0), integration_time=1.0,
    )  # fmt: skip
    assert at_criterion.background_correction[0] == pytest.approx(3.0206, abs=5e-5)
    negligible = emission.sound_energy_pressure(
        np.full((4, 1), 91.0), "hemisphere", radius=1.0, grade="survey",
        background_levels=np.full(1, 80.0), integration_time=1.0,
    )  # fmt: skip
    assert negligible.background_correction[0] == 0.0
    assert negligible.grade == "survey"


def test_background_requires_integration_time() -> None:
    with pytest.raises(ValueError, match="'integration_time' .* is required"):
        emission.sound_energy_pressure(
            np.full((10, 1), 80.0),
            "hemisphere",
            radius=1.0,
            background_levels=np.full(1, 60.0),
        )


@pytest.mark.parametrize("bad", [0.0, -2.0, np.nan, np.inf])
def test_integration_time_must_be_positive(bad: float) -> None:
    with pytest.raises(ValueError, match="'integration_time' must be positive"):
        emission.sound_energy_pressure(
            np.full((10, 1), 80.0),
            "hemisphere",
            radius=1.0,
            background_levels=np.full(1, 60.0),
            integration_time=bad,
        )
    with pytest.raises(ValueError, match="'integration_time' must be positive"):
        emission.sound_energy_pressure(
            np.full((10, 1), 80.0), "hemisphere", radius=1.0, integration_time=bad
        )


def test_non_finite_background_raises() -> None:
    with pytest.raises(
        ValueError, match="'background_levels' must contain only finite"
    ):
        emission.sound_energy_pressure(
            np.full((10, 1), 80.0),
            "hemisphere",
            radius=1.0,
            background_levels=np.array([np.nan]),
            integration_time=1.0,
        )


def test_surface_geometry_and_position_count_as_for_lw() -> None:
    """The surface, its minimum positions and the refusals are the LW ones."""
    with pytest.raises(ValueError, match="requires at least 10 microphone positions"):
        emission.sound_energy_pressure(np.full((9, 1), 80.0), "hemisphere", radius=1.0)
    with pytest.raises(ValueError, match="A positive 'radius' is required"):
        emission.sound_energy_pressure(np.full((10, 1), 80.0), "hemisphere")
    with pytest.raises(ValueError, match="'reflecting_planes' must be 1, 2 or 3"):
        emission.sound_energy_pressure(
            np.full((10, 1), 80.0), "hemisphere", radius=1.0, reflecting_planes=4
        )
    with pytest.raises(ValueError, match="'levels_positions' must contain only finite"):
        emission.sound_energy_pressure(
            np.full((10, 1), np.inf), "hemisphere", radius=1.0
        )
    box = emission.sound_energy_pressure(
        np.full((9, 1), 80.0), "box", dimensions=(1.4, 0.9, 1.1), distance=1.0
    )
    a, b, c = 0.5 * 1.4 + 1.0, 0.5 * 0.9 + 1.0, 1.1 + 1.0
    assert box.surface_area == pytest.approx(4.0 * (a * b + b * c + c * a))


def test_a_weighted_total_eq_e2_from_table_e1() -> None:
    """Eq. (E.2): L_JA = 10 lg sum 10^(0.1(L_Jk + C_k)), C_k from Table E.1."""
    lj_target = np.array([90.0, 92.0, 95.0, 93.0])
    le = np.tile(lj_target - 10.0 * np.log10(2.0 * np.pi * 4.0), (10, 1))
    res = emission.sound_energy_pressure(
        le, "hemisphere", radius=2.0, frequencies=_FREQS
    )
    expected = 10.0 * np.log10(np.sum(10.0 ** (0.1 * (lj_target + _CK))))
    assert res.sound_energy_level_a == pytest.approx(expected, abs=1e-9)


def test_uncertainty_is_the_sound_power_one_eq24() -> None:
    """u(L_J) = u(L_W) = sigma_tot (Eq. 24/25), U = 2 sqrt(1.5^2 + 2^2) = 5.0 dB
    (the ISO 3744 9.5 EXAMPLE, sigma_R0 = 1.5 dB, sigma_omc = 2.0 dB, k = 2).
    """
    res = emission.sound_energy_pressure(
        np.full((10, 1), 80.0), "hemisphere", radius=1.0, omc_uncertainty=2.0
    )
    assert res.uncertainty == pytest.approx(5.0)


def test_result_rejects_inconsistent_fields() -> None:
    base = emission.sound_energy_pressure(
        np.full((10, 2), 80.0), "hemisphere", radius=1.0, frequencies=[500.0, 1000.0]
    )
    import dataclasses

    with pytest.raises(ValueError, match="one value per band"):
        dataclasses.replace(base, background_correction=np.zeros(3))
    with pytest.raises(ValueError, match="'grade' must be one of"):
        dataclasses.replace(base, grade="precision")
    with pytest.raises(ValueError, match="'events' must be at least 1"):
        dataclasses.replace(base, events=0)
    with pytest.raises(ValueError, match="'integration_time' must be positive"):
        dataclasses.replace(base, integration_time=0.0)
    with pytest.raises(ValueError, match="'sound_energy_level' must be finite"):
        dataclasses.replace(base, sound_energy_level=np.array([80.0, np.nan]))
    with pytest.raises(ValueError, match="'surface_area' must be finite"):
        dataclasses.replace(base, surface_area=float("nan"))


# --------------------------------------------------------------------------
# ISO 3744 Annex G: reference meteorological conditions
# --------------------------------------------------------------------------
def test_annex_g_corrections_at_the_reference_atmosphere() -> None:
    """At 23 C and 101,325 kPa: C1 = 5 lg(296.15/314) = -0.1271 dB and
    C2 = 15 lg(296.15/296) = +0.0033 dB, both by hand.
    """
    corr = emission.reference_atmosphere_correction(23.0, 101.325)
    assert corr.c1 == pytest.approx(5.0 * np.log10(296.15 / 314.0), abs=1e-12)
    assert corr.c2 == pytest.approx(15.0 * np.log10(296.15 / 296.0), abs=1e-12)
    assert corr.c1 == pytest.approx(-0.1271, abs=5e-5)
    assert corr.c2 == pytest.approx(0.0033, abs=5e-5)
    assert corr.total == pytest.approx(corr.c1 + corr.c2)
    assert corr.static_pressure == pytest.approx(101.325)
    assert corr.temperature == pytest.approx(23.0)


def test_annex_g_static_pressure_from_altitude_eq_g2() -> None:
    """Eq. (G.2) at 500 m: 101.325 (1 - 2.2560e-5 * 500)^5.2553 = 95.46 kPa,
    and both terms carry -10 lg(p_s/p_s0) = +0.259 dB from it.
    """
    corr = emission.reference_atmosphere_correction(23.0, altitude=500.0)
    ps = 101.325 * (1.0 - 2.2560e-5 * 500.0) ** 5.2553
    assert corr.static_pressure == pytest.approx(ps, rel=1e-12)
    assert corr.static_pressure == pytest.approx(95.46, abs=5e-3)
    p_term = -10.0 * np.log10(ps / 101.325)
    assert corr.c1 == pytest.approx(p_term + 5.0 * np.log10(296.15 / 314.0), abs=1e-12)
    assert corr.c2 == pytest.approx(p_term + 15.0 * np.log10(296.15 / 296.0), abs=1e-12)


def test_annex_g_correction_is_zero_at_120m_and_23c() -> None:
    """ISO 3744 H.4.2.7: 'At 120 m altitude and 23 C the correction is zero'.

    Eq. (G.2) gives p_s(120 m) = 99.89 kPa, so -10 lg(p_s/p_s0) = 0.0621 dB
    in each term (0.124 dB in all), which the temperature terms
    5 lg(296.15/314) + 15 lg(296.15/296) = -0.124 dB cancel to under 1e-4 dB.
    """
    corr = emission.reference_atmosphere_correction(23.0, altitude=120.0)
    assert abs(corr.total) < 1e-4


def test_annex_g_applies_alike_to_lw_and_lj() -> None:
    """Eq. (G.1) and (G.3) add the same C1 + C2 to either level."""
    corr = emission.reference_atmosphere_correction(5.0, altitude=800.0)
    lj = emission.sound_energy_pressure(
        np.full((10, 1), 80.0), "hemisphere", radius=1.0
    )
    lw = emission.sound_power_pressure(np.full((10, 1), 80.0), "hemisphere", radius=1.0)
    lj_ref = lj.sound_energy_level + corr.total
    lw_ref = lw.sound_power_level + corr.total
    np.testing.assert_allclose(lj_ref, lw_ref)
    assert corr.total > 0.0  # thinner and colder air than the reference


def test_annex_g_refusals() -> None:
    with pytest.raises(ValueError, match="Give one of 'static_pressure'"):
        emission.reference_atmosphere_correction(23.0)
    with pytest.raises(ValueError, match="not both"):
        emission.reference_atmosphere_correction(23.0, 101.0, altitude=100.0)
    with pytest.raises(ValueError, match="'static_pressure' must be positive"):
        emission.reference_atmosphere_correction(23.0, 0.0)
    with pytest.raises(ValueError, match="'altitude' must be finite and below"):
        emission.reference_atmosphere_correction(23.0, altitude=50000.0)
    with pytest.raises(ValueError, match="'temperature' must be finite and above"):
        emission.reference_atmosphere_correction(-273.15, 101.325)


# --------------------------------------------------------------------------
# ISO 3741 9.2: reverberation room
# --------------------------------------------------------------------------
_ROOM_FREQS = np.array([100.0, 500.0, 1000.0, 5000.0, 10000.0])


def _c1(theta: float, ps: float) -> float:
    return -10.0 * np.log10(ps / 101.325) + 5.0 * np.log10((273.15 + theta) / 314.0)


def _c2(theta: float, ps: float) -> float:
    return -10.0 * np.log10(ps / 101.325) + 15.0 * np.log10((273.15 + theta) / 296.0)


def _bracket(
    t60: np.ndarray,
    volume: float,
    surface: float,
    freq: np.ndarray,
    theta: float,
    ps: float,
) -> np.ndarray:
    """The bracket of Eq. (30) written out (same as Eq. 20), for inversion."""
    c = 20.05 * np.sqrt(273.0 + theta)
    a = (55.26 / c) * (volume / np.asarray(t60, dtype=float))
    waterhouse = 10.0 * np.log10(1.0 + surface * c / (8.0 * volume * np.asarray(freq)))
    return (
        10.0 * np.log10(a / 1.0)
        + 4.34 * (a / surface)
        + waterhouse
        + _c1(theta, ps)
        + _c2(theta, ps)
        - 6.0
    )


def test_direct_method_exact_inversion_eq30() -> None:
    """Generate L_E(ST) from a known L_J with the bracket of Eq. (30), recover it."""
    volume, surface = 200.0, 210.0
    t60 = np.array([2.0, 1.8, 1.5, 1.0, 0.6])
    theta, ps = 23.0, 101.325
    lj_target = np.array([90.0, 95.0, 100.0, 92.0, 85.0])
    le = lj_target - _bracket(t60, volume, surface, _ROOM_FREQS, theta, ps)
    res = emission.sound_energy_reverberation(
        le, t60, volume, surface, _ROOM_FREQS, temperature=theta, static_pressure=ps
    )
    assert isinstance(res, emission.ReverberationSoundEnergyResult)
    assert res.method == "direct"
    np.testing.assert_allclose(res.sound_energy_level, lj_target, atol=1e-9, rtol=0.0)
    assert res.c1 == pytest.approx(_c1(theta, ps))
    assert res.c2 == pytest.approx(_c2(theta, ps))


def test_direct_method_identity_lj_equals_lw_plus_10lg_t() -> None:
    """A steady source for T = 5 s in the room: L_J = L_W + 10 lg 5 exactly,
    with the per-position K1i of Eq. (25) equal to Eq. (14)'s because the
    background enters as its exposure over the same T.
    """
    rng = np.random.default_rng(3741)
    lp = np.array([80.0, 82.0, 84.0, 81.0, 78.0]) + rng.normal(0.0, 0.4, (6, 5))
    bg = lp - rng.uniform(10.5, 14.0, (6, 5))  # inside the 9.1.2 criteria
    t60 = np.array([2.0, 1.8, 1.5, 1.0, 0.6])
    shift = 10.0 * np.log10(5.0)
    lw = emission.sound_power_reverberation(
        lp, t60, 200.0, 210.0, _ROOM_FREQS, background_levels=bg,
        temperature=20.0, static_pressure=100.0,
    )  # fmt: skip
    lj = emission.sound_energy_reverberation(
        lp + shift, t60, 200.0, 210.0, _ROOM_FREQS, background_levels=bg,
        integration_time=5.0, temperature=20.0, static_pressure=100.0,
    )  # fmt: skip
    np.testing.assert_allclose(lj.sound_energy_level, lw.sound_power_level + shift)
    np.testing.assert_allclose(lj.background_correction, lw.background_correction)
    np.testing.assert_allclose(lj.mean_event_level, lw.mean_pressure_level + shift)
    np.testing.assert_allclose(lj.absorption_area, lw.absorption_area)
    np.testing.assert_allclose(lj.waterhouse_correction, lw.waterhouse_correction)
    assert lj.sound_energy_level_a == pytest.approx(lw.sound_power_level_a + shift)
    assert lj.speed_of_sound == pytest.approx(lw.speed_of_sound)
    assert lj.integration_time == pytest.approx(5.0)


def test_room_k1i_uses_the_frequency_dependent_criterion_on_the_exposure() -> None:
    """9.2.2 'in a similar manner to that of 9.1.2': a 10 dB margin at 1 kHz
    is the mid-band criterion, K1 = 0.4576 dB; a 6 dB margin at 100 Hz is the
    edge-band one, K1 = 1.2563 dB. The margins are formed after the background
    is raised by 10 lg(T/T0) = 10 dB for T = 10 s.
    """
    freqs = np.array([100.0, 1000.0])
    levels = np.full((6, 2), 80.0)
    bg = np.tile([80.0 - 6.0 - 10.0, 80.0 - 10.0 - 10.0], (6, 1))
    with warnings.catch_warnings():
        warnings.simplefilter("error", emission.SoundPowerWarning)
        res = emission.sound_energy_reverberation(
            levels,
            1.5,
            200.0,
            210.0,
            freqs,
            background_levels=bg,
            integration_time=10.0,
        )
    np.testing.assert_allclose(
        res.background_correction, [_k1(6.0), _k1(10.0)], atol=1e-9
    )
    np.testing.assert_allclose(res.background_correction, [1.2563, 0.4576], atol=5e-5)


def test_room_per_event_axis_and_encompassing_measurement_agree() -> None:
    """Eq. (22) over five equal events and Eq. (23) on their energy sum coincide."""
    single = np.full((6, 5), 85.0)
    per_event = np.repeat(single[np.newaxis], 5, axis=0)
    a = emission.sound_energy_reverberation(per_event, 1.5, 200.0, 210.0, _ROOM_FREQS)
    b = emission.sound_energy_reverberation(
        single + 10.0 * np.log10(5.0), 1.5, 200.0, 210.0, _ROOM_FREQS, events=5
    )
    np.testing.assert_allclose(a.sound_energy_level, b.sound_energy_level)
    np.testing.assert_allclose(a.mean_event_level, 85.0)
    assert a.events == 5
    assert b.events == 5


def test_room_refusals() -> None:
    with pytest.raises(ValueError, match="'levels' already carries"):
        emission.sound_energy_reverberation(
            np.full((5, 6, 5), 80.0), 1.5, 200.0, 210.0, _ROOM_FREQS, events=5
        )
    with pytest.raises(ValueError, match="'levels' must be a 1D spectrum"):
        emission.sound_energy_reverberation(
            np.zeros((2, 2, 2, 2)), 1.5, 200.0, 210.0, _ROOM_FREQS
        )
    with pytest.raises(ValueError, match="'levels' must contain only finite"):
        emission.sound_energy_reverberation(
            np.array([np.nan] * 5), 1.5, 200.0, 210.0, _ROOM_FREQS
        )
    with pytest.raises(ValueError, match="'integration_time' .* is required"):
        emission.sound_energy_reverberation(
            np.full(5, 80.0), 1.5, 200.0, 210.0, _ROOM_FREQS,
            background_levels=np.full(5, 60.0),
        )  # fmt: skip
    with pytest.raises(
        ValueError, match="'volume' and 'surface_area' must be positive"
    ):
        emission.sound_energy_reverberation(
            np.full(5, 80.0), 1.5, 0.0, 210.0, _ROOM_FREQS
        )
    with pytest.raises(ValueError, match="'frequencies' length must match"):
        emission.sound_energy_reverberation(
            np.full(5, 80.0), 1.5, 200.0, 210.0, [1000.0]
        )


def test_comparison_method_eq31_exact_by_construction() -> None:
    """L_J = L_W(RSS) + (L_E(ST) - L_p(RSS)) + C2, digit-exact."""
    theta, ps = 20.0, 100.0
    lw_ref = np.array([90.0, 92.0, 88.0])
    le = np.array([84.0, 85.5, 83.0])
    lp_rss = np.array([78.0, 79.0, 76.0])
    res = emission.sound_energy_comparison(
        le, lp_rss, lw_ref, temperature=theta, static_pressure=ps
    )
    expected = lw_ref + (le - lp_rss) + _c2(theta, ps)
    np.testing.assert_allclose(res.sound_energy_level, expected, atol=1e-12)
    assert res.method == "comparison"
    assert np.isnan(res.c1)
    assert np.all(np.isnan(res.absorption_area))
    assert np.all(np.isnan(res.waterhouse_correction))


def test_comparison_identity_with_the_sound_power_comparison() -> None:
    """The same room and reference source: L_J = L_W + 10 lg(T/T0), T = 8 s,
    with the test source's background as an exposure and the steady reference
    source's as a time average.
    """
    rng = np.random.default_rng(31)
    lp = np.array([80.0, 82.0, 84.0]) + rng.normal(0.0, 0.3, (6, 3))
    bg = lp - 9.0
    lp_rss = lp - 2.0
    bg_rss = lp_rss - 12.0
    lw_ref = np.array([85.0, 86.0, 84.0])
    freqs = np.array([500.0, 1000.0, 2000.0])
    shift = 10.0 * np.log10(8.0)
    lw = emission.sound_power_comparison(
        lp, lp_rss, lw_ref, frequencies=freqs, background_levels=bg,
        background_levels_ref=bg_rss,
    )  # fmt: skip
    lj = emission.sound_energy_comparison(
        lp + shift, lp_rss, lw_ref, frequencies=freqs, background_levels=bg,
        integration_time=8.0, background_levels_ref=bg_rss,
    )  # fmt: skip
    np.testing.assert_allclose(lj.sound_energy_level, lw.sound_power_level + shift)
    np.testing.assert_allclose(lj.background_correction, lw.background_correction)
    assert lj.sound_energy_level_a == pytest.approx(lw.sound_power_level_a + shift)


def test_comparison_refusals() -> None:
    le = np.full(3, 80.0)
    with pytest.raises(ValueError, match="span the same bands"):
        emission.sound_energy_comparison(le, np.full(2, 70.0), np.full(3, 85.0))
    with pytest.raises(
        ValueError, match="'frequencies' are required to apply 'background_levels'"
    ):
        emission.sound_energy_comparison(
            le,
            le - 2.0,
            np.full(3, 85.0),
            background_levels=le - 10.0,
            integration_time=1.0,
        )
    with pytest.raises(ValueError, match="'integration_time' .* is required"):
        emission.sound_energy_comparison(
            le, le - 2.0, np.full(3, 85.0), frequencies=[500.0, 1000.0, 2000.0],
            background_levels=le - 10.0,
        )  # fmt: skip
    with pytest.raises(
        ValueError, match="'frequencies' are required to apply 'background_levels_ref'"
    ):
        emission.sound_energy_comparison(
            le, le - 2.0, np.full(3, 85.0), background_levels_ref=le - 12.0
        )


def test_room_result_rejects_inconsistent_fields() -> None:
    import dataclasses

    base = emission.sound_energy_reverberation(
        np.full(5, 80.0), 1.5, 200.0, 210.0, _ROOM_FREQS
    )
    with pytest.raises(ValueError, match="'method' must be one of"):
        dataclasses.replace(base, method="survey")
    with pytest.raises(ValueError, match="'events' must be at least 1"):
        dataclasses.replace(base, events=0)
    with pytest.raises(ValueError, match="'integration_time' must be positive"):
        dataclasses.replace(base, integration_time=-1.0)
    with pytest.raises(ValueError, match="one value per band"):
        dataclasses.replace(base, absorption_area=np.zeros(2))


# --------------------------------------------------------------------------
# ISO 3741 Annex F: octave bands from one-third-octave bands
# --------------------------------------------------------------------------
def test_octave_band_levels_three_equal_thirds_add_10lg3() -> None:
    """Eq. (F.4): three equal thirds sum to L + 10 lg 3 = L + 4.7712 dB."""
    thirds = np.array([800.0, 1000.0, 1250.0])
    freqs, levels = emission.octave_band_levels(np.full(3, 90.0), thirds)
    np.testing.assert_allclose(freqs, [1000.0])
    assert levels[0] == pytest.approx(90.0 + 4.771213, abs=1e-5)


def test_octave_band_levels_group_by_table_f1() -> None:
    """Table F.1: k = 1 is 50 Hz, so 50/63/80 Hz make the 63 Hz octave and
    6,3/8/10 kHz the 8 kHz one; a full 24-band input gives all eight octaves.
    """
    thirds = np.array(
        [50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250,
         1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000], dtype=float,
    )  # fmt: skip
    levels = np.arange(24, dtype=float)
    freqs, out = emission.octave_band_levels(levels, thirds)
    np.testing.assert_allclose(freqs, [63, 125, 250, 500, 1000, 2000, 4000, 8000])
    expected = [
        10.0 * np.log10(np.sum(10.0 ** (0.1 * levels[3 * i : 3 * i + 3])))
        for i in range(8)
    ]
    np.testing.assert_allclose(out, expected)
    # Order of the input does not matter, only membership.
    rng = np.random.default_rng(1)
    perm = rng.permutation(24)
    _, shuffled = emission.octave_band_levels(levels[perm], thirds[perm])
    np.testing.assert_allclose(shuffled, expected)


def test_octave_band_levels_several_spectra_on_the_last_axis() -> None:
    thirds = np.array([400.0, 500.0, 630.0, 800.0, 1000.0, 1250.0])
    levels = np.array([[80.0] * 6, [70.0] * 6])
    freqs, out = emission.octave_band_levels(levels, thirds)
    assert out.shape == (2, 2)
    np.testing.assert_allclose(freqs, [500.0, 1000.0])
    np.testing.assert_allclose(out[1], 70.0 + 10.0 * np.log10(3.0))


def test_octave_band_levels_refusals() -> None:
    with pytest.raises(
        ValueError, match="three one-third-octave bands of every octave"
    ):
        emission.octave_band_levels(np.zeros(2), [800.0, 1000.0])
    with pytest.raises(ValueError, match="'frequencies' must be nominal"):
        emission.octave_band_levels(np.zeros(3), [900.0, 1000.0, 1250.0])
    with pytest.raises(ValueError, match="'frequencies' must not repeat"):
        emission.octave_band_levels(np.zeros(3), [1000.0, 1000.0, 1250.0])
    with pytest.raises(ValueError, match="'levels' must carry one value per band"):
        emission.octave_band_levels(np.zeros(2), [800.0, 1000.0, 1250.0])
    with pytest.raises(ValueError, match="'levels' must contain only finite"):
        emission.octave_band_levels([np.nan, 0.0, 0.0], [800.0, 1000.0, 1250.0])
    with pytest.raises(ValueError, match="'frequencies' must be a non-empty 1-D"):
        emission.octave_band_levels(np.zeros((1, 0)), [])


# --------------------------------------------------------------------------
# .plot() on both results
# --------------------------------------------------------------------------
def test_plot_bars_match_lj_and_title_carries_lja() -> None:
    res = emission.sound_energy_pressure(
        np.full((10, 4), 80.0), "hemisphere", radius=2.0, frequencies=_FREQS
    )
    ax = res.plot()
    heights = [p.get_height() for p in ax.patches]
    np.testing.assert_allclose(heights, res.sound_energy_level)
    assert f"{res.sound_energy_level_a:.1f}" in ax.get_title()
    assert "ISO 3744/3746" in ax.get_title()
    assert "$L_J$" in ax.get_ylabel()
    plt.close("all")

    room = emission.sound_energy_reverberation(
        np.full((6, 5), 80.0), 1.5, 200.0, 210.0, _ROOM_FREQS
    )
    ax = room.plot()
    assert "ISO 3741" in ax.get_title()
    np.testing.assert_allclose(
        [p.get_height() for p in ax.patches], room.sound_energy_level
    )
    plt.close("all")


def test_plot_without_frequencies_labels_the_bands() -> None:
    res = emission.sound_energy_pressure(
        np.full((10, 2), 80.0), "hemisphere", radius=2.0
    )
    ax = res.plot()
    assert "Band" in ax.get_xlabel()
    assert "dB(A)" not in ax.get_title()  # LJA is NaN for two bands, no frequencies
    plt.close("all")


def test_plot_spanish_labels() -> None:
    res = emission.sound_energy_pressure(
        np.full((10, 4), 80.0), "hemisphere", radius=2.0, frequencies=_FREQS
    )
    ax = res.plot(language="es")
    assert "espectro de energía acústica" in ax.get_title()
    assert "Nivel de energía acústica" in ax.get_ylabel()
    plt.close("all")
    with pytest.raises(ValueError, match="Unknown language"):
        res.plot(language="xx")
