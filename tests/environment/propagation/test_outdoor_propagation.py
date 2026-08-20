#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Tests for ISO 9613-2:1996 outdoor sound propagation (general method).

Normative anchors (ISO 9613-2:1996):
- Eq. (3): LfT(DW) = Lw + Dc - A.
- Eq. (4): A = Adiv + Aatm + Agr + Abar + Amisc.
- Eq. (7): Adiv = 20 lg(d/d0) + 11, d0 = 1 m  -> 51 dB at 100 m.
- Eq. (8): Aatm = alpha * d (alpha = ISO 9613-1 coefficient).
- Eq. (9) + Table 3: Agr = As + Ar + Am with the a'/b'/c'/d' functions and the
  overlap factor q of note 2.
- Eq. (10): Agr = 4,8 - (2 hm/d)[17 + 300/d] >= 0 (alternative method, 7.3.2).
- Eq. (11): DOmega solid-angle index.
- Eq. (12)/(13): Abar = Dz - Agr >= 0 (top edge) / Abar = Dz >= 0 (lateral).
- Eq. (14): Dz = 10 lg[3 + (C2/lambda) C3 z Kmet], C2 = 20/40, lambda = 340/f.
- Eq. (15): C3 double-diffraction factor (e=0 -> 1, e >> lambda -> 3).
- Eq. (16)/(17): pathlength difference z (single / double).
- Eq. (18): Kmet meteorological factor.
- Eq. (21)/(22): Cmet meteorological correction.
- Clause 7.4: Dz limits 20 dB (single) / 25 dB (double).

Primary oracles: the exact closed forms above (Adiv digit-exact points, the
Table 3 function limits, the barrier caps and C3 transition) and consistency of
Aatm with ``phonometry.environment.propagation.air_absorption``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from phonometry import environment

BANDS = np.array(environment.DEFAULT_FREQUENCIES)


# --------------------------------------------------------------------------- #
# Geometrical divergence (Eq. 7)
# --------------------------------------------------------------------------- #
class TestGeometricDivergence:
    @pytest.mark.parametrize(
        ("distance", "expected"),
        [(1.0, 11.0), (10.0, 31.0), (100.0, 51.0), (1000.0, 71.0)],
    )
    def test_exact_points(self, distance: float, expected: float) -> None:
        assert environment.geometric_divergence(distance) == pytest.approx(expected)

    def test_doubling_adds_6db(self) -> None:
        delta = environment.geometric_divergence(
            200.0
        ) - environment.geometric_divergence(100.0)
        assert delta == pytest.approx(20.0 * math.log10(2.0), abs=1e-9)
        assert delta == pytest.approx(6.0206, abs=1e-3)

    def test_non_positive_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            environment.geometric_divergence(0.0)


# --------------------------------------------------------------------------- #
# Atmospheric absorption (Eq. 8)
# --------------------------------------------------------------------------- #
class TestAtmosphericAbsorption:
    def test_consistency_with_air_absorption(self) -> None:
        # Aatm evaluates alpha at the exact base-10 midbands behind the
        # nominal labels (the ISO 9613-2 Table 2 convention).
        d = 500.0
        alpha = environment.air_attenuation(
            BANDS, 15.0, 70.0, 101.325, exact_midband=True
        )
        got = environment.atmospheric_absorption(d, BANDS, 15.0, 70.0, 101.325)
        assert np.allclose(got, alpha * d)

    def test_exact_midband_convention_at_8khz(self) -> None:
        # At 8 kHz / 20 C / 70 % the nominal-frequency alpha runs ~1.3 % high
        # (77.6 vs 76.6 dB/km); Aatm must follow the exact-midband value.
        got = environment.atmospheric_absorption(1000.0, [8000.0], 20.0, 70.0, 101.325)
        exact = environment.air_attenuation(
            [8000.0], 20.0, 70.0, 101.325, exact_midband=True
        )
        nominal = environment.air_attenuation([8000.0], 20.0, 70.0, 101.325)
        assert got[0] == pytest.approx(float(exact[0]) * 1000.0)
        assert got[0] < float(nominal[0]) * 1000.0

    def test_scales_linearly_with_distance(self) -> None:
        a1 = environment.atmospheric_absorption(100.0, BANDS)
        a2 = environment.atmospheric_absorption(300.0, BANDS)
        assert np.allclose(a2, 3.0 * a1)

    def test_grows_with_frequency(self) -> None:
        a = environment.atmospheric_absorption(1000.0, BANDS)
        assert np.all(np.diff(a) > 0.0)


# --------------------------------------------------------------------------- #
# Ground effect, general method (Eq. 9 + Table 3)
# --------------------------------------------------------------------------- #
class TestGroundGeneral:
    def test_hard_ground_no_middle_region(self) -> None:
        # G = 0 everywhere and dp <= 30(hs+hr) (q=0): As=Ar=-1,5, Am=0 => -3 dB.
        agr = environment.ground_attenuation(50.0, 1.0, 1.0, BANDS, 0.0, 0.0, 0.0)
        assert np.allclose(agr, -3.0)

    def test_hard_ground_far_still_minus_three_low_band(self) -> None:
        # 63 Hz: As=Ar=-1,5 regardless of G; Am=-3q with Gm irrelevant at 63 Hz.
        # For hard ground at 2k-8k the term is -1,5(1-0) = -1,5 each => -3.
        agr = environment.ground_attenuation(
            1000.0,
            1.0,
            1.0,
            [63.0, 8000.0],
            0.0,
            0.0,
            0.0,
            projected_distance=1000.0,
        )
        # dp=1000 > 30*2=60 => q = 1 - 60/1000 = 0,94; Am(63) = -3*0,94 = -2,82.
        q = 1.0 - 60.0 / 1000.0
        assert agr[0] == pytest.approx(-3.0 - 3.0 * q)  # -1,5-1,5-2,82
        assert agr[1] == pytest.approx(-3.0 - 3.0 * q * (1.0 - 0.0))

    def test_table3_prime_limits_porous(self) -> None:
        # Large dp => (1 - e^{-dp/50}) -> 1 and (1 - e^{-2,8e-6 dp^2}) -> 1.
        # Source on ground (h=0): b'(0)=10,1; c'(0)=15,5; d'(0)=6,5.
        big = 1.0e7
        # Isolate one region by putting receiver far above (its prime ~ small)
        # and reading the porous source contribution via symmetry hs=hr=0.
        agr = environment.ground_attenuation(
            big,
            0.0,
            0.0,
            [250.0, 500.0, 1000.0],
            1.0,
            1.0,
            1.0,
            projected_distance=big,
        )
        # hs=hr=0 => threshold=0, q=1; Gm=1 => Am(>=125)=0.
        # As=Ar=-1,5 + G*prime(0). b'(0)=10,1 => As=8,6; Agr=17,2.
        assert agr[0] == pytest.approx(2.0 * (-1.5 + 10.1), abs=1e-6)
        assert agr[1] == pytest.approx(2.0 * (-1.5 + 15.5), abs=1e-6)
        assert agr[2] == pytest.approx(2.0 * (-1.5 + 6.5), abs=1e-6)

    def test_a_prime_125_value(self) -> None:
        # a'(h) at h=5, huge dp: 3,0*e^0 + 5,7*e^{-0,09*25} => 1,5+3,0+5,7*e^-2,25.
        big = 1.0e7
        agr = environment.ground_attenuation(
            big, 5.0, 5.0, [125.0], 1.0, 1.0, 1.0, projected_distance=big
        )
        a_prime = 1.5 + 3.0 + 5.7 * math.exp(-2.25)
        assert agr[0] == pytest.approx(2.0 * (-1.5 + a_prime), abs=1e-6)

    def test_porous_more_attenuation_than_hard(self) -> None:
        hard = environment.ground_attenuation(500.0, 2.0, 2.0, BANDS, 0.0, 0.0, 0.0)
        porous = environment.ground_attenuation(500.0, 2.0, 2.0, BANDS, 1.0, 1.0, 1.0)
        # Porous yields larger (more positive) Agr at mid/high bands.
        assert np.all(porous[1:] >= hard[1:] - 1e-9)
        assert porous[3] > hard[3]

    def test_invalid_ground_factor_raises(self) -> None:
        with pytest.raises(ValueError, match=r"ground_source"):
            environment.ground_attenuation(100.0, 1.0, 1.0, BANDS, 1.5, 0.0, 0.0)
        with pytest.raises(ValueError, match=r"ground_middle"):
            environment.ground_attenuation(100.0, 1.0, 1.0, BANDS, 0.0, -0.1, 0.0)

    def test_default_projected_distance(self) -> None:
        # dp = sqrt(d^2 - (hs-hr)^2); with hs=hr it equals d.
        a_default = environment.ground_attenuation(
            100.0, 3.0, 3.0, BANDS, 1.0, 1.0, 1.0
        )
        a_explicit = environment.ground_attenuation(
            100.0, 3.0, 3.0, BANDS, 1.0, 1.0, 1.0, projected_distance=100.0
        )
        assert np.allclose(a_default, a_explicit)


# --------------------------------------------------------------------------- #
# Ground effect, alternative method (Eq. 10) + DOmega (Eq. 11)
# --------------------------------------------------------------------------- #
class TestGroundAlternative:
    def test_eq10_closed_form(self) -> None:
        # 4,8 - (2*2/100)(17 + 300/100) = 4,8 - 0,04*20 = 4,0.
        assert environment.ground_attenuation_alternative(100.0, 2.0) == pytest.approx(
            4.0
        )

    def test_negative_clamped_to_zero(self) -> None:
        # Large hm, short d -> strongly negative -> replaced by 0 (7.3.2).
        assert environment.ground_attenuation_alternative(50.0, 20.0) == 0.0

    def test_grows_with_distance_towards_4_8(self) -> None:
        near = environment.ground_attenuation_alternative(200.0, 2.0)
        far = environment.ground_attenuation_alternative(2000.0, 2.0)
        assert far > near
        assert far < 4.8

    def test_directivity_omega_symmetric_upper_bound(self) -> None:
        # hs=hr, dp >> h: ratio -> 1 => DOmega -> 10 lg 2 ~ 3,01 dB.
        assert environment.directivity_omega(1.0, 1.0, 1.0e6) == pytest.approx(
            10.0 * math.log10(2.0), abs=1e-6
        )

    def test_directivity_omega_range(self) -> None:
        dw = environment.directivity_omega(2.0, 5.0, 100.0)
        assert 0.0 <= dw <= 3.02


# --------------------------------------------------------------------------- #
# Screening / barriers (Eq. 12-18)
# --------------------------------------------------------------------------- #
class TestBarrier:
    def test_single_diffraction_grows_with_frequency(self) -> None:
        b = environment.Barrier(source_to_edge=10.0, edge_to_receiver=10.0)
        dz = environment.barrier_attenuation(b, 19.0, BANDS)
        # Barrier helps more at high frequency (until the 20 dB cap).
        assert dz[0] < dz[3]
        assert np.all(dz >= 0.0)

    def test_single_cap_20db(self) -> None:
        b = environment.Barrier(source_to_edge=50.0, edge_to_receiver=50.0)
        dz = environment.barrier_attenuation(b, 60.0, BANDS)
        assert np.max(dz) == pytest.approx(20.0)
        assert np.all(dz <= 20.0 + 1e-9)

    def test_double_cap_25db(self) -> None:
        b = environment.Barrier(
            source_to_edge=50.0, edge_to_receiver=50.0, edge_separation=5.0
        )
        dz = environment.barrier_attenuation(b, 60.0, BANDS)
        assert np.max(dz) == pytest.approx(25.0)
        assert np.all(dz <= 25.0 + 1e-9)

    def test_double_exceeds_single(self) -> None:
        single = environment.Barrier(source_to_edge=10.0, edge_to_receiver=10.0)
        double = environment.Barrier(
            source_to_edge=10.0, edge_to_receiver=10.0, edge_separation=2.0
        )
        dz_s = environment.barrier_attenuation(single, 19.0, BANDS)
        dz_d = environment.barrier_attenuation(double, 19.0, BANDS)
        assert np.all(dz_d >= dz_s - 1e-9)

    def test_grazing_incidence_gives_10lg3(self) -> None:
        # dss+dsr == d (edge exactly on the sight line): z = 0 and Eq. (14)
        # gives Dz = 10 lg 3 = 4.77 dB -- continuous across grazing, not 0.
        b = environment.Barrier(source_to_edge=10.0, edge_to_receiver=10.0)
        dz = environment.barrier_attenuation(b, 20.0, BANDS)
        assert np.allclose(dz, 10.0 * math.log10(3.0))

    def test_negative_z_line_of_sight_clear(self) -> None:
        # Top edge 0.5 m below the sight line, midway at d = 100 m:
        # z = -(2*hypot(50, 0.5) - 100) = -0.005 m, Kmet = 1 (Eq. (18)).
        # Hand-derived Dz = 10 lg(3 - (20/lambda)*0.005): 4.74 dB at 63 Hz,
        # 4.55 dB at 500 Hz, 2.61 dB at 4 kHz -- decaying with frequency
        # instead of accruing screening credit.
        edge = math.hypot(50.0, 0.5)
        b = environment.Barrier(
            source_to_edge=edge, edge_to_receiver=edge, line_of_sight_clear=True
        )
        dz = environment.barrier_attenuation(b, 100.0, [63.0, 500.0, 4000.0])
        z = 2.0 * edge - 100.0
        expected = [
            10.0 * math.log10(3.0 - (20.0 * f / 340.0) * z)
            for f in (63.0, 500.0, 4000.0)
        ]
        assert dz == pytest.approx(expected, abs=1e-9)
        assert dz[0] == pytest.approx(4.744, abs=0.005)
        assert dz[1] == pytest.approx(4.553, abs=0.005)
        assert dz[2] == pytest.approx(2.610, abs=0.005)
        # Monotone decay with frequency below the sight line.
        assert dz[0] > dz[1] > dz[2] >= 0.0

    def test_deep_clearance_tends_to_zero(self) -> None:
        # Top edge 3 m below the sight line: no screening effect remains at
        # mid/high frequency (Dz clamped at 0; z ~ -lambda/10 boundary).
        edge = math.hypot(50.0, 3.0)
        b = environment.Barrier(
            source_to_edge=edge, edge_to_receiver=edge, line_of_sight_clear=True
        )
        dz = environment.barrier_attenuation(b, 100.0, BANDS)
        assert np.all(dz >= 0.0)
        assert np.all(np.diff(dz) <= 1e-12)
        assert dz[-1] == pytest.approx(0.0, abs=1e-9)

    def test_c3_transition_single_to_triple(self) -> None:
        # Eq. (15): e -> 0 gives C3 = 1 (matches single); e >> lambda gives C3 -> 3.
        from phonometry.environment.propagation.outdoor_propagation import _c3_double

        lam = 340.0 / 500.0
        assert _c3_double(lam, 1e-6) == pytest.approx(1.0, abs=1e-3)
        assert _c3_double(lam, 1e6) == pytest.approx(3.0, rel=1e-3)

    def test_c2_image_source_variant(self) -> None:
        # C2 = 40 (image sources) gives more attenuation than C2 = 20.
        normal = environment.Barrier(source_to_edge=10.0, edge_to_receiver=10.0)
        image = environment.Barrier(
            source_to_edge=10.0, edge_to_receiver=10.0, ground_reflections_by_image=True
        )
        dz_n = environment.barrier_attenuation(normal, 19.0, BANDS)
        dz_i = environment.barrier_attenuation(image, 19.0, BANDS)
        assert np.all(dz_i[dz_n < 20.0 - 1e-6] > dz_n[dz_n < 20.0 - 1e-6])

    def test_kmet_reduces_dz_at_long_range(self) -> None:
        # For z>0, Kmet<1 (long range) => Dz below the Kmet=1 (lateral) value.
        top = environment.Barrier(source_to_edge=100.0, edge_to_receiver=100.0)
        lateral = environment.Barrier(
            source_to_edge=100.0, edge_to_receiver=100.0, lateral=True
        )
        dz_top = environment.barrier_attenuation(top, 195.0, [63.0])
        dz_lat = environment.barrier_attenuation(lateral, 195.0, [63.0])
        assert dz_top[0] < dz_lat[0] < 20.0

    def test_dz_floor_is_10lg3(self) -> None:
        # z>0 but tiny lambda-scaled term: Dz -> 10 lg 3 ~ 4,77 dB floor.
        b = environment.Barrier(source_to_edge=1.0, edge_to_receiver=1.0)
        dz = environment.barrier_attenuation(b, 1.999, [63.0])
        assert dz[0] >= 10.0 * math.log10(3.0) - 1e-6


# --------------------------------------------------------------------------- #
# Meteorological correction (Eq. 21/22)
# --------------------------------------------------------------------------- #
class TestMeteorologicalCorrection:
    def test_zero_below_threshold(self) -> None:
        # dp <= 10(hs+hr) => Cmet = 0.
        assert environment.meteorological_correction(20.0, 1.0, 1.0, 2.0) == 0.0

    def test_eq22_closed_form(self) -> None:
        # dp=200, hs+hr=2 => 2*(1 - 20/200) = 1,8.
        assert environment.meteorological_correction(
            200.0, 1.0, 1.0, 2.0
        ) == pytest.approx(1.8)

    def test_grows_towards_c0(self) -> None:
        c0 = 3.0
        near = environment.meteorological_correction(50.0, 1.0, 1.0, c0)
        far = environment.meteorological_correction(5000.0, 1.0, 1.0, c0)
        assert far > near
        assert far < c0


# --------------------------------------------------------------------------- #
# Assembly (Eq. 3/4) and receiver level
# --------------------------------------------------------------------------- #
class TestOutdoorPropagation:
    def test_breakdown_sums_to_total(self) -> None:
        r = environment.outdoor_propagation_attenuation(
            200.0, 2.0, 2.0, BANDS, 1.0, 1.0, 1.0
        )
        assert isinstance(r, environment.OutdoorAttenuation)
        assert np.allclose(r.a_total, r.a_div + r.a_atm + r.a_gr + r.a_bar)

    def test_total_grows_with_distance(self) -> None:
        near = environment.outdoor_propagation_attenuation(100.0, 2.0, 2.0, BANDS)
        far = environment.outdoor_propagation_attenuation(400.0, 2.0, 2.0, BANDS)
        assert np.all(far.a_total > near.a_total)

    def test_barrier_increases_attenuation(self) -> None:
        no_bar = environment.outdoor_propagation_attenuation(
            100.0, 1.0, 1.0, BANDS, 0.0, 0.0, 0.0
        )
        bar = environment.Barrier(source_to_edge=52.0, edge_to_receiver=52.0)
        with_bar = environment.outdoor_propagation_attenuation(
            100.0, 1.0, 1.0, BANDS, 0.0, 0.0, 0.0, barrier=bar
        )
        assert np.all(with_bar.a_total >= no_bar.a_total - 1e-9)
        assert with_bar.a_total[-1] > no_bar.a_total[-1]

    def test_barrier_top_edge_folds_ground(self) -> None:
        # Note 13: with a top-edge barrier, Agr + Abar == Dz (ground cancels).
        bar = environment.Barrier(source_to_edge=52.0, edge_to_receiver=52.0)
        r = environment.outdoor_propagation_attenuation(
            100.0, 1.0, 1.0, BANDS, 0.0, 0.0, 0.0, barrier=bar
        )
        dz = environment.barrier_attenuation(bar, 100.0, BANDS)
        # Where the barrier is effective (Dz - Agr > 0) the sum equals Dz.
        eff = dz - r.a_gr > 0.0
        assert np.allclose((r.a_gr + r.a_bar)[eff], dz[eff])

    def test_barrier_helps_more_at_high_frequency(self) -> None:
        bar = environment.Barrier(source_to_edge=52.0, edge_to_receiver=52.0)
        with_bar = environment.outdoor_propagation_attenuation(
            100.0, 1.0, 1.0, BANDS, barrier=bar
        )
        gain = with_bar.a_bar
        # Screening gain is larger at 4 kHz than at 63 Hz.
        assert gain[-2] > gain[0]

    def test_invalid_distance_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            environment.outdoor_propagation_attenuation(-1.0, 1.0, 1.0)


class TestPredictedReceiverLevel:
    def test_composition_eq3(self) -> None:
        lw = np.full(len(BANDS), 100.0)
        r = environment.outdoor_propagation_attenuation(
            200.0, 2.0, 2.0, BANDS, 1.0, 1.0, 1.0
        )
        level = environment.predicted_receiver_level(
            lw,
            environment.PropagationGeometry(200.0, 2.0, 2.0),
            frequencies=BANDS,
            ground=environment.GroundFactors(1.0, 1.0, 1.0),
        )
        assert np.allclose(level, lw - r.a_total)

    def test_directivity_and_omega_add(self) -> None:
        lw = np.full(len(BANDS), 90.0)
        geometry = environment.PropagationGeometry(200.0, 2.0, 2.0)
        base = environment.predicted_receiver_level(lw, geometry, frequencies=BANDS)
        boosted = environment.predicted_receiver_level(
            lw,
            geometry,
            frequencies=BANDS,
            directivity=environment.DirectivityCorrection(index=2.0, d_omega=3.0),
        )
        assert np.allclose(boosted - base, 5.0)

    def test_cmet_subtracted(self) -> None:
        lw = np.full(len(BANDS), 90.0)
        geometry = environment.PropagationGeometry(400.0, 2.0, 2.0)
        porous = environment.GroundFactors(1.0, 1.0, 1.0)
        dw = environment.predicted_receiver_level(
            lw, geometry, frequencies=BANDS, ground=porous
        )
        lt = environment.predicted_receiver_level(
            lw, geometry, frequencies=BANDS, ground=porous, c0=2.0
        )
        cmet = environment.meteorological_correction(400.0, 2.0, 2.0, 2.0)
        assert cmet > 0.0
        assert np.allclose(dw - lt, cmet)

    def test_level_decreases_with_distance(self) -> None:
        lw = np.full(len(BANDS), 100.0)
        near = environment.predicted_receiver_level(
            lw,
            environment.PropagationGeometry(100.0, 2.0, 2.0),
            frequencies=BANDS,
        )
        far = environment.predicted_receiver_level(
            lw,
            environment.PropagationGeometry(1000.0, 2.0, 2.0),
            frequencies=BANDS,
        )
        assert np.all(far < near)


class TestInputValidation:
    def test_barrier_rejects_zero_edge_separation(self) -> None:
        with pytest.raises(ValueError, match="edge_separation"):
            environment.Barrier(
                source_to_edge=50.0, edge_to_receiver=50.0, edge_separation=0.0
            )

    def test_barrier_rejects_negative_edge_distance(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            environment.Barrier(source_to_edge=-1.0, edge_to_receiver=50.0)

    def test_ground_attenuation_rejects_nonpositive_frequency(self) -> None:
        with_zero = np.array([0.0, 500.0])
        with pytest.raises(ValueError, match="frequencies"):
            environment.ground_attenuation(200.0, 2.0, 2.0, with_zero, 1.0, 1.0, 1.0)

    def test_ground_attenuation_rejects_nonpositive_distance(self) -> None:
        with pytest.raises(ValueError, match="distance"):
            environment.ground_attenuation(0.0, 2.0, 2.0, BANDS, 1.0, 1.0, 1.0)

    def test_barrier_attenuation_rejects_nonpositive_frequency(self) -> None:
        barrier = environment.Barrier(source_to_edge=50.0, edge_to_receiver=50.0)
        with_negative = np.array([-1.0, 500.0])
        with pytest.raises(ValueError, match="frequencies"):
            environment.barrier_attenuation(barrier, 90.0, with_negative)


def test_public_exports() -> None:
    from phonometry import environment

    for name in (
        "Barrier",
        "OutdoorAttenuation",
        "outdoor_propagation_attenuation",
        "predicted_receiver_level",
        "geometric_divergence",
        "atmospheric_absorption",
        "ground_attenuation",
        "ground_attenuation_alternative",
        "barrier_attenuation",
        "meteorological_correction",
        "directivity_omega",
        "DEFAULT_FREQUENCIES",
        "PropagationGeometry",
        "GroundFactors",
        "AtmosphericConditions",
        "DirectivityCorrection",
    ):
        assert hasattr(environment, name), name
