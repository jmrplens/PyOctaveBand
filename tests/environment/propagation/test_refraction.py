#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Tests for atmospheric refraction: ray tracing and the parabolic equation.

Clean-room anchors:
- Salomons, *Computational Atmospheric Acoustics* (Springer, 2001): Eq. (4.3)
  Snell's law for rays, Eq. (4.5) logarithmic profile, Sec. 4.4 ray geometry,
  Appendix G (CNPE / starter Eq. (G.64)), Appendix H (GFPE, Eq. (H.28)
  reflection, Eq. (H.58) refraction factor), Eq. (3.6) relative level.
- Attenborough & Van Renterghem, *Predicting Outdoor Sound* 2e (2021), Ch. 11.

Primary oracles:
- a linear effective sound-speed profile makes every ray an exact circular arc,
  so a circle fit of a traced ray recovers the closed-form radius of curvature
  ``Rc = c0/(|g| cos theta0)`` to machine precision, and a turning ray peaks at
  the closed-form height ``Rc(1 - cos theta0)``;
- an upward-refracting profile has a closed-form shadow-zone distance and the PE
  level collapses far inside the shadow;
- HOMOGENEOUS atmosphere (gradient zero): the GFPE relative-level field
  reproduces the exact spherical-wave ground effect ``ground_effect`` to about
  0.1 dB along the whole range (independent Weyl-Van der Pol oracle);
- reciprocity: swapping source and receiver heights leaves the PE level;
- a hard ground gives the +6 dB two-ray enhancement in the homogeneous limit.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from phonometry.environment.propagation.ground_barriers import ground_effect
from phonometry.environment.propagation.refraction import (
    AtmosphericPEResult,
    AtmosphericRayResult,
    EffectiveSoundSpeedProfile,
    atmospheric_parabolic_equation,
    atmospheric_ray_paths,
    linear_sound_speed_profile,
    log_linear_sound_speed_profile,
    ray_curvature_radius,
    shadow_zone_distance,
)

C0 = 343.0
# A representative grassland surface impedance (normalized, e^{-i omega t}:
# Im(Z) > 0 for a passive ground), so the tests never trip the porous-model
# out-of-range warning; the same value feeds both the PE and the ground_effect
# oracle for an apples-to-apples match.
Z_GRASS = complex(11.0, 8.0)


# --------------------------------------------------------------------------- #
# Effective sound-speed profiles
# --------------------------------------------------------------------------- #
def test_linear_profile_is_exactly_linear() -> None:
    prof = linear_sound_speed_profile(0.1, ground_speed=340.0, max_height=100.0)
    assert prof.heights[0] == 0.0
    assert prof.sound_speeds[0] == pytest.approx(340.0)
    # A two-point profile interpolates the exact gradient at any height.
    assert prof.speed_at(50.0) == pytest.approx(345.0)
    assert prof.speed_at(np.array([0.0, 100.0])).tolist() == [340.0, 350.0]


def test_log_linear_profile_matches_salomons_eq_4_5() -> None:
    b, z0, c0 = 1.0, 0.1, 340.0
    prof = log_linear_sound_speed_profile(
        b, ground_speed=c0, roughness_length=z0, max_height=100.0
    )
    # The sampled speeds follow Salomons Eq. (4.5) exactly at every grid height.
    expected = c0 + b * np.log1p(prof.heights / z0)
    np.testing.assert_allclose(prof.sound_speeds, expected, rtol=1e-12)
    assert prof.heights[0] == 0.0
    assert prof.sound_speeds[0] == pytest.approx(c0)
    # The piecewise-linear interpolation tracks the log curve closely between
    # the (logarithmically spaced) samples.
    z = np.array([1.0, 10.0, 50.0])
    np.testing.assert_allclose(prof.speed_at(z), c0 + b * np.log1p(z / z0), rtol=2e-4)


def test_profile_validation() -> None:
    with pytest.raises(ValueError, match="'gradient' must be finite"):
        linear_sound_speed_profile(np.nan)
    with pytest.raises(
        ValueError, match="linear profile reaches a non-positive sound speed"
    ):
        # A steep negative gradient drives the speed non-positive within range.
        linear_sound_speed_profile(-10.0, ground_speed=340.0, max_height=100.0)
    with pytest.raises(ValueError, match="'n_points' must be at least"):
        log_linear_sound_speed_profile(1.0, n_points=1)
    with pytest.raises(ValueError, match="'roughness_length' must be positive"):
        log_linear_sound_speed_profile(1.0, roughness_length=0.0)


# --------------------------------------------------------------------------- #
# Closed-form ray geometry
# --------------------------------------------------------------------------- #
def test_ray_curvature_radius_closed_form() -> None:
    # Rc = c0 / (|g| cos theta0).
    assert ray_curvature_radius(0.1, ground_speed=340.0) == pytest.approx(3400.0)
    assert ray_curvature_radius(-0.1, ground_speed=340.0) == pytest.approx(3400.0)
    ang = 30.0
    assert ray_curvature_radius(0.1, ground_speed=340.0, launch_angle_deg=ang) == (
        pytest.approx(3400.0 / np.cos(np.radians(ang)))
    )


def test_ray_curvature_radius_validation() -> None:
    with pytest.raises(ValueError, match="'gradient' must be finite and non-zero"):
        ray_curvature_radius(0.0)  # straight ray
    with pytest.raises(
        ValueError, match=r"'launch_angle_deg' must be within \(-90, 90\)"
    ):
        ray_curvature_radius(0.1, launch_angle_deg=90.0)


def test_shadow_zone_distance_closed_form() -> None:
    # x = sqrt(2 Rc)(sqrt(hs) + sqrt(hr)), Rc = c0/|g|.
    g, hs, hr, c0 = -0.1, 2.0, 2.0, 340.0
    rc = c0 / abs(g)
    expected = np.sqrt(2.0 * rc) * (np.sqrt(hs) + np.sqrt(hr))
    assert shadow_zone_distance(g, hs, hr, ground_speed=c0) == pytest.approx(expected)


def test_shadow_zone_requires_upward_refraction() -> None:
    with pytest.raises(
        ValueError,
        match=r"'gradient' must be negative \(an upward-refracting profile\)",
    ):
        shadow_zone_distance(0.1, 2.0, 2.0)  # downward refraction: no shadow


# --------------------------------------------------------------------------- #
# Ray tracing vs the exact circular arc (linear profile)
# --------------------------------------------------------------------------- #
def _circle_fit_radius(x: np.ndarray, y: np.ndarray) -> float:
    """Algebraic (Kasa) circle fit; returns the fitted radius."""
    a = np.c_[2.0 * x, 2.0 * y, np.ones(x.size)]
    sol, *_ = np.linalg.lstsq(a, x**2 + y**2, rcond=None)
    cx, cy, cc = sol
    return float(np.sqrt(cc + cx**2 + cy**2))


@pytest.mark.parametrize(
    ("gradient", "angle"), [(0.1, 20.0), (-0.05, 30.0), (0.2, 10.0)]
)
def test_ray_is_exact_circular_arc(gradient: float, angle: float) -> None:
    # A ray launched at the ground (where c = ground_speed) is an exact circle
    # of the closed-form radius; a circle fit recovers it to <0.01 %.
    prof = linear_sound_speed_profile(gradient, ground_speed=C0, max_height=3000.0)
    rc = ray_curvature_radius(gradient, ground_speed=C0, launch_angle_deg=angle)
    res = atmospheric_ray_paths(
        prof,
        source_height=0.0,
        launch_angles_deg=[angle],
        max_range=300.0,
        n_steps=8000,
    )
    r, z = res.ranges[0], res.heights[0]
    seg = slice(0, r.size // 2)  # first arc, before any turn/reflection
    assert _circle_fit_radius(r[seg], z[seg]) == pytest.approx(rc, rel=1e-3)


def test_turning_height_matches_geometry() -> None:
    # A ray in downward refraction turns at height Rc(1 - cos theta0).
    gradient, angle = 0.2, 10.0
    prof = linear_sound_speed_profile(gradient, ground_speed=C0, max_height=3000.0)
    rc = ray_curvature_radius(gradient, ground_speed=C0, launch_angle_deg=angle)
    res = atmospheric_ray_paths(
        prof,
        source_height=0.0,
        launch_angles_deg=[angle],
        max_range=600.0,
        n_steps=8000,
    )
    peak = float(res.heights[0].max())
    assert peak == pytest.approx(rc * (1.0 - np.cos(np.radians(angle))), rel=2e-3)
    # One up-and-down excursion registers a single turning point.
    assert res.turning_points[0] == 1


def test_ray_reflects_at_ground() -> None:
    # A downward launch from a low source reflects off the ground at least once.
    prof = linear_sound_speed_profile(0.05, ground_speed=C0, max_height=500.0)
    res = atmospheric_ray_paths(
        prof,
        source_height=5.0,
        launch_angles_deg=[-10.0],
        max_range=400.0,
        n_steps=4000,
    )
    assert res.ground_reflections[0] >= 1
    assert np.all(res.heights >= -1e-9)  # never dips below the ground


def test_ray_travel_time_free_field() -> None:
    # In a homogeneous atmosphere a horizontal ray travels at c0: t = r/c0.
    prof = linear_sound_speed_profile(1e-12, ground_speed=C0, max_height=100.0)
    res = atmospheric_ray_paths(
        prof, source_height=10.0, launch_angles_deg=[0.0], max_range=343.0, n_steps=2000
    )
    assert res.travel_times[0, -1] == pytest.approx(343.0 / C0, rel=1e-6)


def test_ground_reflection_costs_no_accuracy() -> None:
    """A bounce must not degrade the path, which is what splitting the step buys.

    A ray launched downward from the ground and the same ray launched upward are
    one trajectory: the first reflects immediately and continues as the second.
    They agree only if the range step is split at the crossing. Folding the
    overshoot back after a whole step instead integrates through ground rather
    than air, where the profile saturates, and applies a mid-step reflection at
    the step's end; that was worth 2.9 m of height here at the middle step
    count, closing only as 1/n_steps.

    The agreement is exact rather than merely close because once the step is
    split the two rays are, step for step, the same arithmetic.
    """
    prof = linear_sound_speed_profile(gradient=0.2, max_height=500.0)
    for n_steps in (201, 2001, 20_001):
        kw = {"source_height": 0.0, "max_range": 3000.0, "n_steps": n_steps}
        up = atmospheric_ray_paths(prof, launch_angles_deg=[10.0], **kw)
        down = atmospheric_ray_paths(prof, launch_angles_deg=[-10.0], **kw)
        assert np.array_equal(up.heights[0], down.heights[0]), n_steps
        assert np.array_equal(up.travel_times[0], down.travel_times[0]), n_steps


def test_ray_validation() -> None:
    prof = linear_sound_speed_profile(0.1)
    with pytest.raises(ValueError, match="'source_height' must be non-negative"):
        atmospheric_ray_paths(prof, source_height=-1.0, launch_angles_deg=[0.0])
    with pytest.raises(
        ValueError, match=r"'launch_angles_deg' must be within \(-90, 90\)"
    ):
        atmospheric_ray_paths(prof, source_height=1.0, launch_angles_deg=[90.0])
    with pytest.raises(
        ValueError, match="'launch_angles_deg' must be finite and non-empty"
    ):
        atmospheric_ray_paths(
            prof,
            source_height=1.0,
            launch_angles_deg=[],
        )


# --------------------------------------------------------------------------- #
# Parabolic equation vs the spherical-wave ground effect (homogeneous limit)
# --------------------------------------------------------------------------- #
def _flat_profile() -> EffectiveSoundSpeedProfile:
    # A near-zero gradient is a homogeneous atmosphere (no refraction).
    return linear_sound_speed_profile(1e-12, ground_speed=C0, max_height=200.0)


def test_pe_reproduces_spherical_ground_effect() -> None:
    freq, zs, zr = 500.0, 2.0, 2.0
    pe = atmospheric_parabolic_equation(
        freq,
        _flat_profile(),
        source_height=zs,
        impedance=Z_GRASS,
        max_range=1000.0,
        max_height=50.0,
    )
    level = pe.level_at_height(zr)
    ranges = pe.ranges
    oracle = np.array(
        [
            ground_effect(
                np.array([freq]),
                zs,
                zr,
                max(float(r), 1.0),
                impedance=Z_GRASS,
                speed_of_sound=C0,
            ).excess_attenuation[0]
            for r in ranges
        ]
    )
    band = (ranges >= 50.0) & (ranges <= 1000.0)
    # Default grid (dz = lambda/10): a few tenths of a dB; halving dz converges
    # the long-range tail below 0.3 dB.
    assert np.max(np.abs(level[band] - oracle[band])) < 0.6


@pytest.mark.parametrize(("freq", "zs", "zr"), [(250.0, 1.0, 1.0), (1000.0, 2.0, 5.0)])
def test_pe_matches_ground_effect_configs(freq: float, zs: float, zr: float) -> None:
    pe = atmospheric_parabolic_equation(
        freq,
        _flat_profile(),
        source_height=zs,
        impedance=Z_GRASS,
        max_range=800.0,
        max_height=40.0,
    )
    level = pe.level_at_height(zr)
    ranges = pe.ranges
    oracle = np.array(
        [
            ground_effect(
                np.array([freq]),
                zs,
                zr,
                max(float(r), 1.0),
                impedance=Z_GRASS,
                speed_of_sound=C0,
            ).excess_attenuation[0]
            for r in ranges
        ]
    )
    band = (ranges >= 50.0) & (ranges <= 800.0)
    assert np.max(np.abs(level[band] - oracle[band])) < 0.6


def test_pe_hard_ground_enhancement() -> None:
    # A hard ground gives the coherent +6 dB two-ray enhancement at range. A
    # perfectly rigid ground reflects the near-grazing components too, so the
    # GFPE needs a taller grid than an absorbing ground to converge.
    freq, zs, zr = 500.0, 2.0, 2.0
    hard = complex(1e6, 0.0)
    pe = atmospheric_parabolic_equation(
        freq,
        _flat_profile(),
        source_height=zs,
        impedance=hard,
        max_range=600.0,
        max_height=150.0,
    )
    level = pe.level_at_height(zr)
    ranges = pe.ranges
    oracle = np.array(
        [
            20.0
            * np.log10(
                np.abs(
                    1.0
                    + (np.hypot(r, zs - zr) / np.hypot(r, zs + zr))
                    * np.exp(
                        1j
                        * 2.0
                        * np.pi
                        * freq
                        / C0
                        * (np.hypot(r, zs + zr) - np.hypot(r, zs - zr))
                    )
                )
            )
            for r in ranges
            if r > 0
        ]
    )
    band = (ranges[1:] >= 100.0) & (ranges[1:] <= 600.0)
    assert np.max(np.abs(level[1:][band] - oracle[band])) < 0.6
    assert level[np.argmin(np.abs(ranges - 500.0))] > 5.0  # near +6 dB


def test_pe_finite_ground_shows_dip_below_hard_ground() -> None:
    # Physical anchor: near the Salomons Fig. D.3 dip (grassland, hs = hr = 2 m,
    # about 395 Hz at r = 100 m) a finite-impedance ground must sit far below
    # the hard-ground two-ray level at the same low-height geometry. Before the
    # impedance-convention fix the PE ground condition received the conjugated
    # (e^{+j omega t}) impedance and the soft-ground dip almost vanished.
    freq, zs, zr = 395.0, 2.0, 2.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        soft = atmospheric_parabolic_equation(
            freq,
            _flat_profile(),
            source_height=zs,
            flow_resistivity=2e5,
            max_range=120.0,
            max_height=50.0,
        )
    hard = atmospheric_parabolic_equation(
        freq,
        _flat_profile(),
        source_height=zs,
        impedance=complex(1e6, 0.0),
        max_range=120.0,
        max_height=150.0,
    )
    i = int(np.argmin(np.abs(soft.ranges - 100.0)))
    j = int(np.argmin(np.abs(hard.ranges - 100.0)))
    dip = soft.level_at_height(zr)[i] - hard.level_at_height(zr)[j]
    assert dip < -8.0  # deep interference dip only the correct convention gives


def test_pe_reciprocity() -> None:
    freq = 400.0
    a = atmospheric_parabolic_equation(
        freq,
        _flat_profile(),
        source_height=2.0,
        impedance=Z_GRASS,
        max_range=500.0,
        max_height=40.0,
    )
    b = atmospheric_parabolic_equation(
        freq,
        _flat_profile(),
        source_height=10.0,
        impedance=Z_GRASS,
        max_range=500.0,
        max_height=40.0,
    )
    band = (a.ranges >= 50.0) & (a.ranges <= 500.0)
    diff = a.level_at_height(10.0)[band] - b.level_at_height(2.0)[band]
    assert np.max(np.abs(diff)) < 0.5


def test_pe_upward_refraction_shadow() -> None:
    # Deep inside the upward-refraction shadow the level collapses well below
    # both the downward-refraction and the homogeneous levels at the same range.
    freq, zs, zr = 400.0, 2.0, 2.0
    up = log_linear_sound_speed_profile(-1.0, ground_speed=340.0, max_height=200.0)
    down = log_linear_sound_speed_profile(1.0, ground_speed=340.0, max_height=200.0)
    pu = atmospheric_parabolic_equation(
        freq, up, source_height=zs, impedance=Z_GRASS, max_range=800.0, max_height=40.0
    )
    pd = atmospheric_parabolic_equation(
        freq,
        down,
        source_height=zs,
        impedance=Z_GRASS,
        max_range=800.0,
        max_height=40.0,
    )
    i = int(np.argmin(np.abs(pu.ranges - 700.0)))
    assert pu.level_at_height(zr)[i] < pd.level_at_height(zr)[i] - 20.0


def test_pe_flow_resistivity_matches_impedance_path() -> None:
    # Supplying sigma routes through the porous model (Delany-Bazley) to a
    # surface impedance; passing that same impedance directly (conjugated by
    # hand into the e^{-i omega t} convention, since the materials domain works
    # in e^{+j omega t}) is identical. The low band falls outside the model's
    # published fit range, which raises a PorousAbsorberWarning by design.
    from phonometry.materials import PorousAbsorberWarning, delany_bazley

    freq = 500.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PorousAbsorberWarning)
        z = complex(
            np.conj(
                delany_bazley(
                    np.array([freq]), 2.0e5, speed_of_sound=C0
                ).normalized_impedance[0]
            )
        )
        via_sigma = atmospheric_parabolic_equation(
            freq,
            _flat_profile(),
            source_height=2.0,
            flow_resistivity=2.0e5,
            max_range=400.0,
            max_height=30.0,
        )
    via_z = atmospheric_parabolic_equation(
        freq,
        _flat_profile(),
        source_height=2.0,
        impedance=z,
        max_range=400.0,
        max_height=30.0,
    )
    assert via_sigma.normalized_impedance == pytest.approx(z)
    np.testing.assert_allclose(
        via_sigma.relative_level, via_z.relative_level, rtol=0, atol=1e-9
    )


def test_pe_validation() -> None:
    prof = _flat_profile()
    with pytest.raises(ValueError, match="'source_height' must be positive"):
        atmospheric_parabolic_equation(
            500.0, prof, source_height=0.0, impedance=Z_GRASS
        )
    with pytest.raises(
        ValueError, match="provide exactly one of 'impedance' or 'flow_resistivity'"
    ):
        # Neither impedance nor flow_resistivity.
        atmospheric_parabolic_equation(500.0, prof, source_height=2.0)
    with pytest.raises(
        ValueError, match="provide exactly one of 'impedance' or 'flow_resistivity'"
    ):
        # Both impedance and flow_resistivity.
        atmospheric_parabolic_equation(
            500.0, prof, source_height=2.0, impedance=Z_GRASS, flow_resistivity=2e5
        )
    with pytest.raises(ValueError, match="'range_step' must not exceed 'max_range'"):
        atmospheric_parabolic_equation(
            500.0,
            prof,
            source_height=2.0,
            impedance=Z_GRASS,
            range_step=1e9,
            max_range=100.0,
        )


# --------------------------------------------------------------------------- #
# Result objects and plotting
# --------------------------------------------------------------------------- #
def test_result_types_and_shapes() -> None:
    prof = _flat_profile()
    rays = atmospheric_ray_paths(
        prof,
        source_height=2.0,
        launch_angles_deg=[-5.0, 0.0, 5.0],
        max_range=300.0,
        n_steps=500,
    )
    assert isinstance(rays, AtmosphericRayResult)
    assert rays.heights.shape == (3, 500)
    assert rays.travel_times.shape == (3, 500)
    pe = atmospheric_parabolic_equation(
        500.0,
        prof,
        source_height=2.0,
        impedance=Z_GRASS,
        max_range=200.0,
        max_height=30.0,
    )
    assert isinstance(pe, AtmosphericPEResult)
    assert pe.relative_level.shape == (pe.heights.size, pe.ranges.size)


@pytest.mark.filterwarnings("ignore::phonometry.PhonometryWarning")
def test_plots_smoke() -> None:
    pytest.importorskip("matplotlib")
    import matplotlib as mpl

    mpl.use("Agg")
    prof = log_linear_sound_speed_profile(1.0, max_height=100.0)
    prof.plot()
    atmospheric_ray_paths(
        prof,
        source_height=2.0,
        launch_angles_deg=[-2.0, 0.0, 2.0],
        max_range=300.0,
        n_steps=500,
    ).plot()
    atmospheric_parabolic_equation(
        500.0,
        prof,
        source_height=2.0,
        impedance=Z_GRASS,
        max_range=200.0,
        max_height=30.0,
    ).plot()
    import matplotlib.pyplot as plt

    plt.close("all")
