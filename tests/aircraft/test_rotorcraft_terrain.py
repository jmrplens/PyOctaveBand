#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Tests for terrain under the rotorcraft hemisphere method (ECAC Doc 32 / NORAH2).

Flat ground is an assumption, and guidance §A.4.4-A.4.5 is what happens when it
fails. The two clauses are one subject: a terrain section is first reduced to
its mean ground plane, which fixes the equivalent source and receiver heights
and the flow resistivity seen along the path, and only then is the line of sight
tested against the profile to decide whether the section screens. Ground effect
and screening are therefore alternatives on the same geometry, never a sum, and
the tests below hold both ends of that: a flat section must reproduce the
flat-ground model exactly, a uniformly sloped one must reproduce it under the
analytic rotation, and a hill above the line of sight must diffract with the
path difference SO + OR - SR computed by hand.

Oracles: closed-form least squares for the mean plane; the log-average of
Eq. 41 for mixed flow resistivity; the closed-form diffraction values
(10 lg 3 with the edge on the line of sight, 10 lg 43 one wavelength above it)
together with the low-edge factor Ch of Eq. 43 and the multiple-edge factor of
Eq. 44; and, for the event chain, stitched scalar runs, which a heterogeneous
run must reproduce cell by cell.
"""

from __future__ import annotations

import matplotlib as mpl

mpl.use("Agg")
import numpy as np
import pytest

from phonometry import aircraft

# --------------------------------------------------------------------------- #
# Topography and screening (guidance §A.4.4-A.4.5)
# --------------------------------------------------------------------------- #


def test_mean_ground_plane_linear_profile_is_exact() -> None:
    r = aircraft.mean_ground_plane([0.0, 50.0, 100.0], [10.0, 20.0, 30.0])
    assert r.slope == pytest.approx(0.2, abs=1e-12)
    assert r.intercept == pytest.approx(10.0, abs=1e-9)


def test_mean_ground_plane_roofline_closed_form() -> None:
    # Symmetric roof (0,0)-(100,20)-(200,0): zero slope; the continuous
    # least-squares intercept is the mean height, area/span = 10 m.
    r = aircraft.mean_ground_plane([0.0, 100.0, 200.0], [0.0, 20.0, 0.0])
    assert r.slope == pytest.approx(0.0, abs=1e-12)
    assert r.intercept == pytest.approx(10.0, abs=1e-9)
    # Non-uniform vertex spacing must not bias the continuous fit: adding a
    # collinear vertex changes nothing.
    r2 = aircraft.mean_ground_plane([0.0, 50.0, 100.0, 200.0], [0.0, 10.0, 20.0, 0.0])
    assert r2.slope == pytest.approx(0.0, abs=1e-12)
    assert r2.intercept == pytest.approx(10.0, abs=1e-9)


def test_mean_ground_plane_equivalent_height_is_orthogonal() -> None:
    # A 45 degree plane: a point 10 m vertically above it sits 10/sqrt(2)
    # orthogonally (the guidance's equivalent height).
    r = aircraft.mean_ground_plane([0.0, 100.0], [0.0, 100.0])
    assert r.equivalent_height(50.0, 60.0) == pytest.approx(10.0 / np.sqrt(2.0))
    assert r.height(50.0) == pytest.approx(50.0)


def test_mean_ground_plane_validation_and_plot() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        aircraft.mean_ground_plane([0.0, 0.0], [1.0, 2.0])
    with pytest.raises(ValueError, match="equal size"):
        aircraft.mean_ground_plane([0.0, 1.0, 2.0], [1.0, 2.0])
    assert (
        aircraft.mean_ground_plane([0.0, 30.0, 100.0], [2.0, 8.0, 3.0]).plot()
        is not None
    )


def test_mean_flow_resistivity_log_average() -> None:
    # Equal lengths of 1e4 and 1e6 average to the geometric mean 1e5 (Eq. 41).
    assert aircraft.mean_flow_resistivity([1.0, 1.0], [1.0e4, 1.0e6]) == pytest.approx(
        1.0e5
    )
    assert aircraft.mean_flow_resistivity([7.0], [2.0e5]) == pytest.approx(2.0e5)
    # Length weighting: 3:1 towards 1e6.
    got = aircraft.mean_flow_resistivity([1.0, 3.0], [1.0e4, 1.0e6])
    assert got == pytest.approx(10.0 ** (0.25 * 4.0 + 0.75 * 6.0))
    with pytest.raises(ValueError, match="equal shape"):
        aircraft.mean_flow_resistivity([1.0, 2.0], [1.0e5])


def test_diffraction_grazing_and_classical_values() -> None:
    # delta = 0 (edge on the line of sight): 10 lg 3 = 4.771 dB.
    assert aircraft.diffraction_attenuation([1000.0], 0.0, edge_height=10.0)[
        0
    ] == pytest.approx(10.0 * np.log10(3.0), abs=1e-9)
    # delta = lambda: 10 lg(3 + 40) = 16.335 dB.
    lam = 346.1 / 1000.0
    assert aircraft.diffraction_attenuation([1000.0], lam, edge_height=10.0)[
        0
    ] == pytest.approx(10.0 * np.log10(43.0), abs=1e-9)


def test_diffraction_low_edge_scales_with_ch() -> None:
    # Ch = min(f h0 / 250, 1) (Eq. 43): a 0.5 m edge at 100 Hz gives Ch = 0.2.
    full = aircraft.diffraction_attenuation([100.0], 1.0, edge_height=1e6)[0]
    low = aircraft.diffraction_attenuation([100.0], 1.0, edge_height=0.5)[0]
    assert low == pytest.approx(0.2 * full)


def test_diffraction_negative_path_difference_branch() -> None:
    # Slightly below the line of sight there is still some attenuation; at
    # (40/lambda) delta = -2 the argument reaches 1 and the attenuation is 0.
    lam = 346.1 / 1000.0
    small = aircraft.diffraction_attenuation([1000.0], -lam / 40.0, edge_height=10.0)[0]
    assert 0.0 < small < 10.0 * np.log10(3.0)
    assert (
        aircraft.diffraction_attenuation([1000.0], -3.0 * lam / 40.0, edge_height=10.0)[
            0
        ]
        == 0.0
    )


def test_diffraction_multiple_edges_and_cap() -> None:
    # e <= 0.3 m keeps C'' = 1; a wide edge span raises C'' towards 3 (Eq. 44).
    single = aircraft.diffraction_attenuation([1000.0], 0.5, edge_height=10.0)
    narrow = aircraft.diffraction_attenuation(
        [1000.0], 0.5, edge_height=10.0, edge_span=0.3
    )
    wide = aircraft.diffraction_attenuation(
        [1000.0], 0.5, edge_height=10.0, edge_span=1e9
    )
    assert narrow[0] == pytest.approx(single[0])
    lam = 346.1 / 1000.0
    assert wide[0] == pytest.approx(10.0 * np.log10(3.0 + 120.0 * 0.5 / lam), rel=1e-6)
    # The 25 dB bound applies to the direct term only when requested.
    big = aircraft.diffraction_attenuation([4000.0], 50.0, edge_height=10.0)
    unc = aircraft.diffraction_attenuation(
        [4000.0], 50.0, edge_height=10.0, capped=False
    )
    assert big[0] == 25.0
    assert unc[0] > 25.0


def test_screening_flat_section_matches_ground_effect() -> None:
    f = np.array([63.0, 250.0, 1000.0])
    res = aircraft.terrain_screening_adjustment(
        f, (0.0, 50.0), (500.0, 1.2), [0.0, 500.0], [0.0, 0.0], flow_resistivity="D"
    )
    ref = aircraft.ground_effect_adjustment(f, 50.0, 1.2, 500.0, flow_resistivity="D")
    assert not res.screened
    assert np.isnan(res.path_difference)
    assert np.allclose(res.adjustment, ref, atol=1e-12)


def test_screening_inclined_plane_uses_equivalent_geometry() -> None:
    # A uniform 10 % slope: the mean plane IS the terrain, so the result must
    # equal the flat-ground model with orthogonal heights and the separation
    # of the projection feet (an analytic rotation of the flat case).
    f = np.array([125.0, 500.0, 2000.0])
    slope = 0.1
    d = np.array([0.0, 400.0])
    z = slope * d
    src = (0.0, 60.0)
    rcv = (400.0, slope * 400.0 + 1.2)
    res = aircraft.terrain_screening_adjustment(f, src, rcv, d, z, flow_resistivity="E")
    scale = np.hypot(1.0, slope)
    hs = 60.0 / scale
    hr = 1.2 / scale
    s_src = (src[0] + slope * src[1]) / scale
    s_rcv = (rcv[0] + slope * rcv[1]) / scale
    ref = aircraft.ground_effect_adjustment(
        f, hs, hr, s_rcv - s_src, flow_resistivity="E"
    )
    assert not res.screened
    assert np.allclose(res.adjustment, ref, atol=1e-9)


def test_screening_blocked_hill_hand_checked_delta() -> None:
    # A triangular hill peaking above the line of sight: single edge, and the
    # path difference is SO + OR - SR by hand.
    f = np.array([250.0, 1000.0])
    src, rcv = (0.0, 20.0), (400.0, 1.2)
    d = np.array([0.0, 190.0, 200.0, 210.0, 400.0])
    z = np.array([0.0, 0.0, 40.0, 0.0, 0.0])
    res = aircraft.terrain_screening_adjustment(f, src, rcv, d, z, flow_resistivity="G")
    assert res.screened
    so = np.hypot(200.0, 40.0 - 20.0)
    orr = np.hypot(200.0, 40.0 - 1.2)
    sr = np.hypot(400.0, 20.0 - 1.2)
    assert res.path_difference == pytest.approx(so + orr - sr, abs=1e-9)
    assert res.diffraction_points.shape == (1, 2)
    assert tuple(res.diffraction_points[0]) == (200.0, 40.0)
    # Screening must lose level relative to the unblocked flat site at high f.
    flat = aircraft.terrain_screening_adjustment(
        f, src, rcv, [0.0, 400.0], [0.0, 0.0], flow_resistivity="G"
    )
    assert res.adjustment[-1] < flat.adjustment[-1] - 5.0


def test_screening_terrain_below_sight_never_diffracts() -> None:
    # Terrain points below the line of sight are not obstacles (the guidance's
    # topography rule): the bump changes the mean plane, not the regime.
    f = np.array([500.0])
    res = aircraft.terrain_screening_adjustment(
        f,
        (0.0, 50.0),
        (400.0, 1.2),
        [0.0, 200.0, 400.0],
        [0.0, 10.0, 0.0],
        flow_resistivity="D",
    )
    assert not res.screened


def test_screening_per_segment_resistivity_uses_log_mean() -> None:
    # Clear path over two equal halves of 1e4 and 1e6: identical to a scalar
    # run at the Eq. 41 log-mean 1e5.
    f = np.array([125.0, 1000.0])
    mixed = aircraft.terrain_screening_adjustment(
        f,
        (0.0, 30.0),
        (300.0, 1.2),
        [0.0, 150.0, 300.0],
        [0.0, 0.0, 0.0],
        flow_resistivity=[1.0e4, 1.0e6],
    )
    scalar = aircraft.terrain_screening_adjustment(
        f,
        (0.0, 30.0),
        (300.0, 1.2),
        [0.0, 150.0, 300.0],
        [0.0, 0.0, 0.0],
        flow_resistivity=1.0e5,
    )
    assert np.allclose(mixed.adjustment, scalar.adjustment, atol=1e-12)


def test_screening_validation_and_plot() -> None:
    f = [500.0]
    with pytest.raises(ValueError, match="cover"):
        aircraft.terrain_screening_adjustment(
            f, (-10.0, 5.0), (100.0, 1.2), [0.0, 100.0], [0.0, 0.0]
        )
    with pytest.raises(ValueError, match="smaller section distance"):
        aircraft.terrain_screening_adjustment(
            f, (100.0, 5.0), (0.0, 1.2), [0.0, 100.0], [0.0, 0.0]
        )
    with pytest.raises(ValueError, match="one value per"):
        aircraft.terrain_screening_adjustment(
            f,
            (0.0, 5.0),
            (100.0, 1.2),
            [0.0, 50.0, 100.0],
            [0.0, 0.0, 0.0],
            flow_resistivity=[1e5],
        )
    blocked = aircraft.terrain_screening_adjustment(
        [250.0, 1000.0],
        (0.0, 20.0),
        (400.0, 1.2),
        [0.0, 190.0, 200.0, 210.0, 400.0],
        [0.0, 0.0, 40.0, 0.0, 0.0],
    )
    assert blocked.plot() is not None
    clear = aircraft.terrain_screening_adjustment(
        [250.0], (0.0, 20.0), (400.0, 1.2), [0.0, 400.0], [0.0, 0.0]
    )
    assert clear.plot() is not None


# --------------------------------------------------------------------------- #
# Terrain and heterogeneous ground in the event chain
# --------------------------------------------------------------------------- #


def _terrain_case(tz: np.ndarray) -> tuple:
    freqs = np.array([63.0, 250.0, 1000.0])
    az = np.arange(-90.0, 91.0, 10.0)
    po = np.arange(0.0, 181.0, 10.0)
    h = aircraft.RotorcraftHemisphere(
        freqs, az, po, np.full((az.size, po.size, freqs.size), 95.0)
    )
    t = np.arange(0.0, 40.5, 0.5)
    pos = np.column_stack([np.zeros_like(t), 50.0 * t - 1000.0, np.full_like(t, 60.0)])
    tx = np.linspace(-2000.0, 2000.0, 41)
    ty = np.linspace(-2000.0, 2000.0, 41)
    return [h], [50.0], [0.0], t, pos, (tx, ty, tz)


def test_event_flat_terrain_matches_flat_ground() -> None:
    hems, spd, ang, t, pos, dem = _terrain_case(np.zeros((41, 41)))
    base = aircraft.rotorcraft_event_level(
        hems,
        spd,
        ang,
        t,
        pos,
        (300.0, 0.0),
        ground=aircraft.RotorcraftGround(flow_resistivity="D"),
    )
    flat = aircraft.rotorcraft_event_level(
        hems,
        spd,
        ang,
        t,
        pos,
        (300.0, 0.0),
        ground=aircraft.RotorcraftGround(
            flow_resistivity="D", terrain=dem, terrain_resolution=50.0
        ),
    )
    assert np.allclose(flat.a_levels, base.a_levels, atol=1e-9)
    assert flat.sel == pytest.approx(base.sel, abs=1e-9)


def test_event_blocking_ridge_attenuates() -> None:
    tz = np.zeros((41, 41))
    tx = np.linspace(-2000.0, 2000.0, 41)
    tz[:, (tx >= 100.0) & (tx <= 200.0)] = 80.0
    hems, spd, ang, t, pos, dem = _terrain_case(tz)
    flat = aircraft.rotorcraft_event_level(
        hems,
        spd,
        ang,
        t,
        pos,
        (300.0, 0.0),
        ground=aircraft.RotorcraftGround(
            flow_resistivity="D",
            terrain=_terrain_case(np.zeros((41, 41)))[5],
            terrain_resolution=25.0,
        ),
    )
    ridge = aircraft.rotorcraft_event_level(
        hems,
        spd,
        ang,
        t,
        pos,
        (300.0, 0.0),
        ground=aircraft.RotorcraftGround(
            flow_resistivity="D", terrain=dem, terrain_resolution=25.0
        ),
    )
    assert ridge.sel < flat.sel - 2.0
    assert ridge.la_max < flat.la_max


def test_event_terrain_sets_receiver_elevation() -> None:
    # A 30 m plateau under the receiver: the microphone rides on the model.
    tz = np.full((41, 41), 30.0)
    hems, spd, ang, t, pos, dem = _terrain_case(tz)
    res = aircraft.rotorcraft_event_level(
        hems,
        spd,
        ang,
        t,
        pos,
        (300.0, 0.0),
        ground=aircraft.RotorcraftGround(terrain=dem, terrain_resolution=100.0),
    )
    k = int(np.argmin(res.distance))
    # CPA slant: track (x=0, z=60) to mic (x=300, z=30+1.2).
    assert res.distance[k] == pytest.approx(np.hypot(300.0, 60.0 - 31.2), abs=0.5)


def test_event_and_contour_array_argument_validation() -> None:
    hems, spd, ang, t, pos, dem = _terrain_case(np.zeros((41, 41)))
    vector_sigma_ground = aircraft.RotorcraftGround(flow_resistivity=np.full(4, 2.0e5))
    mismatched_sigma_ground = aircraft.RotorcraftGround(
        flow_resistivity=np.full(3, 2.0e5)
    )
    mismatched_elevation_ground = aircraft.RotorcraftGround(
        ground_elevation=np.zeros(3)
    )
    incomplete_dem_ground = aircraft.RotorcraftGround(terrain=(dem[0], dem[1]))
    reversed_dem_ground = aircraft.RotorcraftGround(
        terrain=(dem[0][::-1], dem[1], dem[2])
    )
    short_sigma_map_ground = aircraft.RotorcraftGround(
        terrain=dem, flow_resistivity=np.full(1, 2.0e5)
    )
    elevation_with_dem_ground = aircraft.RotorcraftGround(
        terrain=dem, ground_elevation=np.zeros(4)
    )
    with pytest.raises(ValueError, match="scalar"):
        aircraft.rotorcraft_event_level(
            hems, spd, ang, t, pos, (0.0, 0.0), ground=vector_sigma_ground
        )
    with pytest.raises(ValueError, match="one value per grid"):
        aircraft.rotorcraft_noise_contour(
            hems,
            spd,
            ang,
            t,
            pos,
            x=[0.0, 100.0],
            y=[0.0, 100.0],
            ground=mismatched_sigma_ground,
        )
    with pytest.raises(ValueError, match="one value per grid"):
        aircraft.rotorcraft_noise_contour(
            hems,
            spd,
            ang,
            t,
            pos,
            x=[0.0, 100.0],
            y=[0.0, 100.0],
            ground=mismatched_elevation_ground,
        )
    with pytest.raises(ValueError, match="elevation model"):
        aircraft.rotorcraft_event_level(
            hems, spd, ang, t, pos, (0.0, 0.0), ground=incomplete_dem_ground
        )
    with pytest.raises(ValueError, match="strictly increasing"):
        aircraft.rotorcraft_event_level(
            hems, spd, ang, t, pos, (0.0, 0.0), ground=reversed_dem_ground
        )
    with pytest.raises(ValueError, match="per-path maps"):
        aircraft.rotorcraft_event_level(
            hems, spd, ang, t, pos, (0.0, 0.0), ground=short_sigma_map_ground
        )
    with pytest.raises(ValueError, match="left scalar"):
        aircraft.rotorcraft_noise_contour(
            hems,
            spd,
            ang,
            t,
            pos,
            x=[0.0, 100.0],
            y=[0.0, 100.0],
            ground=elevation_with_dem_ground,
        )


def test_event_numpy_scalar_flow_resistivity_is_scalar() -> None:
    # 0-d scalar-likes (np.float32, np.int64, 0-d arrays) mean a single
    # resistivity, not a per-receiver map, also when a terrain model is given.
    hems, spd, ang, t, pos, dem = _terrain_case(np.zeros((41, 41)))
    base = aircraft.rotorcraft_event_level(
        hems,
        spd,
        ang,
        t,
        pos,
        (300.0, 0.0),
        ground=aircraft.RotorcraftGround(flow_resistivity=2.0e5),
    )
    for sigma in (np.float32(2.0e5), np.int64(200_000), np.array(2.0e5)):
        res = aircraft.rotorcraft_event_level(
            hems,
            spd,
            ang,
            t,
            pos,
            (300.0, 0.0),
            ground=aircraft.RotorcraftGround(flow_resistivity=sigma),
        )
        assert np.allclose(res.a_levels, base.a_levels, atol=1e-9)
    with_dem = aircraft.rotorcraft_event_level(
        hems,
        spd,
        ang,
        t,
        pos,
        (300.0, 0.0),
        ground=aircraft.RotorcraftGround(
            flow_resistivity=np.float32(2.0e5),
            terrain=dem,
            terrain_resolution=100.0,
        ),
    )
    assert with_dem.sel == pytest.approx(base.sel, abs=1e-9)


def test_terrain_must_cover_track_and_receivers() -> None:
    # Points beyond the elevation model would otherwise use fabricated edge
    # elevations; both the track and the receivers must be covered.
    hems, spd, ang, t, pos, dem = _terrain_case(np.zeros((41, 41)))
    small = (
        np.linspace(-500.0, 500.0, 11),
        np.linspace(-500.0, 500.0, 11),
        np.zeros((11, 11)),
    )  # track spans y = -1000..1025: not covered
    undersized_dem_ground = aircraft.RotorcraftGround(terrain=small)
    track_only_dem_ground = aircraft.RotorcraftGround(terrain=dem)  # ends at x = 2000
    with pytest.raises(ValueError, match="cover the whole track"):
        aircraft.rotorcraft_event_level(
            hems, spd, ang, t, pos, (0.0, 0.0), ground=undersized_dem_ground
        )
    with pytest.raises(ValueError, match="cover the whole receiver"):
        aircraft.rotorcraft_event_level(
            hems, spd, ang, t, pos, (2500.0, 0.0), ground=track_only_dem_ground
        )
    with pytest.raises(ValueError, match="cover the whole receiver grid"):
        aircraft.rotorcraft_noise_contour(
            hems,
            spd,
            ang,
            t,
            pos,
            x=[-100.0, 2500.0],
            y=[0.0, 100.0],
            ground=track_only_dem_ground,
        )


def test_contour_mixed_sigma_matches_stitched_scalar_runs() -> None:
    hems, spd, ang, t, pos, _ = _terrain_case(np.zeros((41, 41)))
    x = np.array([-200.0, 0.0, 200.0])
    y = np.array([-100.0, 100.0])
    sig = np.array([[2.0e5, 8.0e5, 2.0e5], [8.0e5, 2.0e5, 8.0e5]])
    mixed = aircraft.rotorcraft_noise_contour(
        hems,
        spd,
        ang,
        t,
        pos,
        x=x,
        y=y,
        ground=aircraft.RotorcraftGround(flow_resistivity=sig),
    )
    lo = aircraft.rotorcraft_noise_contour(
        hems,
        spd,
        ang,
        t,
        pos,
        x=x,
        y=y,
        ground=aircraft.RotorcraftGround(flow_resistivity=2.0e5),
    )
    hi = aircraft.rotorcraft_noise_contour(
        hems,
        spd,
        ang,
        t,
        pos,
        x=x,
        y=y,
        ground=aircraft.RotorcraftGround(flow_resistivity=8.0e5),
    )
    assert np.array_equal(mixed.level, np.where(sig == 2.0e5, lo.level, hi.level))


def test_contour_mixed_elevation_matches_stitched_scalar_runs() -> None:
    hems, spd, ang, t, pos, _ = _terrain_case(np.zeros((41, 41)))
    x = np.array([-200.0, 0.0, 200.0])
    y = np.array([-100.0, 100.0])
    ge = np.array([[0.0, 5.0, 0.0], [5.0, 0.0, 5.0]])
    mixed = aircraft.rotorcraft_noise_contour(
        hems,
        spd,
        ang,
        t,
        pos,
        x=x,
        y=y,
        ground=aircraft.RotorcraftGround(ground_elevation=ge),
    )
    lo = aircraft.rotorcraft_noise_contour(
        hems,
        spd,
        ang,
        t,
        pos,
        x=x,
        y=y,
        ground=aircraft.RotorcraftGround(ground_elevation=0.0),
    )
    hi = aircraft.rotorcraft_noise_contour(
        hems,
        spd,
        ang,
        t,
        pos,
        x=x,
        y=y,
        ground=aircraft.RotorcraftGround(ground_elevation=5.0),
    )
    assert np.array_equal(mixed.level, np.where(ge == 0.0, lo.level, hi.level))


def test_contour_with_terrain_smoke() -> None:
    tz = np.zeros((41, 41))
    hems, spd, ang, t, pos, dem = _terrain_case(tz)
    res = aircraft.rotorcraft_noise_contour(
        hems,
        spd,
        ang,
        t,
        pos,
        x=[-100.0, 100.0],
        y=[-100.0, 100.0],
        ground=aircraft.RotorcraftGround(terrain=dem, terrain_resolution=200.0),
    )
    base = aircraft.rotorcraft_noise_contour(
        hems, spd, ang, t, pos, x=[-100.0, 100.0], y=[-100.0, 100.0]
    )
    assert np.allclose(res.level, base.level, atol=1e-9)
