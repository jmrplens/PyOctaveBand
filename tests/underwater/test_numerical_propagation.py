#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Tests for the numerical underwater-propagation solvers.

Oracles are analytic and independent of the implementation: the ideal
(pressure-release) waveguide's exact normal modes, the circular-arc ray paths of
a linear sound-speed gradient, and free-field spherical spreading for the
parabolic equation.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pytest

from phonometry.underwater.numerical_propagation import (
    NormalModeResult,
    ParabolicEquationResult,
    RayTraceResult,
    normal_modes,
    parabolic_equation,
    ray_trace,
)

_ISO = (np.array([0.0, 100.0]), np.array([1500.0, 1500.0]))  # isovelocity 100 m


# --- Normal modes -----------------------------------------------------------


def test_normal_modes_ideal_waveguide_wavenumbers() -> None:
    # Ideal waveguide krm = sqrt(k^2 - (m pi/D)^2); pressure-release bottom.
    D, c, f = 100.0, 1500.0, 20.0
    res = normal_modes(f, *_ISO, source_depth=36.0, receiver_depth=46.0,
                       bottom="pressure-release", n_depth_points=800)
    assert isinstance(res, NormalModeResult)
    k = 2 * np.pi * f / c
    m_max = int(np.floor(k * D / np.pi))
    kr_exact = np.sqrt(k**2 - (np.arange(1, m_max + 1) * np.pi / D) ** 2)
    assert res.wavenumbers.size == m_max
    assert np.allclose(res.wavenumbers[:m_max], kr_exact, atol=1e-4)


def test_normal_modes_tl_matches_analytic_modal_sum() -> None:
    D, c, f, rho = 100.0, 1500.0, 20.0, 1000.0
    zs, zr = 36.0, 46.0
    r = np.linspace(100.0, 10_000.0, 200)
    res = normal_modes(f, *_ISO, source_depth=zs, receiver_depth=zr, ranges_m=r,
                       density=rho, n_depth_points=800)
    k = 2 * np.pi * f / c
    m = np.arange(1, int(np.floor(k * D / np.pi)) + 1)
    kzm = m * np.pi / D
    kr = np.sqrt(k**2 - kzm**2)
    psis = np.sqrt(2 * rho / D) * np.sin(kzm * zs)
    psir = np.sqrt(2 * rho / D) * np.sin(kzm * zr)
    modal = (psis * psir / np.sqrt(kr))[:, None] * np.exp(1j * kr[:, None] * r[None, :])
    field = (1j / (rho * np.sqrt(8 * np.pi * r))) * np.exp(-1j * np.pi / 4) * modal.sum(0)
    tl_exact = -20 * np.log10(np.abs(field) / (1 / (4 * np.pi)))
    assert np.max(np.abs(res.transmission_loss - tl_exact)) < 0.05


def test_normal_modes_rigid_bottom_wavenumbers() -> None:
    # Rigid bottom (Neumann): analytic kzm = (2m-1)pi/(2D), so
    # krm = sqrt(k^2 - ((2m-1)pi/2D)^2).
    D, c, f = 100.0, 1500.0, 50.0
    res = normal_modes(f, [0.0, D], [c, c], source_depth=30.0, receiver_depth=60.0,
                       bottom="rigid", n_depth_points=2000)
    k = 2 * np.pi * f / c
    m_max = int(np.floor((k * 2 * D / np.pi + 1) / 2))
    kzm = (2 * np.arange(1, m_max + 1) - 1) * np.pi / (2 * D)
    kr_exact = np.sqrt(k**2 - kzm**2)
    assert res.wavenumbers.size == m_max
    assert np.allclose(res.wavenumbers[:m_max], kr_exact, atol=1e-3)


# --- Ray tracing ------------------------------------------------------------


def test_ray_trace_linear_gradient_turning_depth() -> None:
    # Linear gradient: a ray turns where c(z_t) = c(zs)/cos(theta0).
    c0, g, D = 1500.0, 0.05, 2000.0
    prof = (np.array([0.0, D]), np.array([c0, c0 + g * D]))
    theta0 = 10.0
    res = ray_trace(*prof, source_depth=0.0, launch_angles_deg=[theta0],
                    max_range=10_500.0, n_steps=20_000)
    xi = np.cos(np.radians(theta0)) / c0
    z_turn = (1.0 / xi - c0) / g
    assert isinstance(res, RayTraceResult)
    assert res.depths[0].max() == pytest.approx(z_turn, abs=1.0)


def test_ray_trace_linear_gradient_circular_arc() -> None:
    # The ray path is an arc of a circle of radius R = 1/(xi |g|).
    c0, g = 1500.0, 0.05
    prof = (np.array([0.0, 2000.0]), np.array([c0, c0 + g * 2000.0]))
    res = ray_trace(*prof, source_depth=0.0, launch_angles_deg=[10.0],
                    max_range=10_400.0, n_steps=20_000)
    r, z = res.ranges[0], res.depths[0]
    xi = np.cos(np.radians(10.0)) / c0
    r_oracle = 1.0 / (xi * g)
    a = np.c_[r, z, np.ones_like(r)]
    sol, *_ = np.linalg.lstsq(a, r**2 + z**2, rcond=None)
    r_fit = np.sqrt(sol[2] + (sol[0] / 2) ** 2 + (sol[1] / 2) ** 2)
    assert r_fit == pytest.approx(r_oracle, rel=1e-3)


def test_ray_trace_constant_speed_is_straight() -> None:
    res = ray_trace(*_ISO, source_depth=50.0, launch_angles_deg=[0.0],
                    max_range=1000.0, n_steps=2000)
    # A horizontal ray in isovelocity water stays at the source depth.
    assert np.allclose(res.depths[0], 50.0, atol=1e-6)


# --- Parabolic equation -----------------------------------------------------


def test_parabolic_equation_free_field_spherical_spreading() -> None:
    # In a deep homogeneous medium the PE reproduces TL = 20 log10(r) at the
    # source depth (spherical spreading) before any boundary interaction.
    D, zs = 20_000.0, 10_000.0
    res = parabolic_equation(50.0, np.array([0.0, D]), np.array([1500.0, 1500.0]),
                             source_depth=zs, max_range=5000.0, range_step=2.0,
                             n_depth_points=8192)
    assert isinstance(res, ParabolicEquationResult)
    zi = int(np.argmin(np.abs(res.depths - zs)))
    r = res.ranges
    mask = (r > 500.0) & (r < 3000.0)
    d = res.transmission_loss[zi][mask] - 20 * np.log10(r[mask])
    assert np.max(np.abs(d)) < 0.05


def test_parabolic_equation_tracks_normal_modes_trend() -> None:
    # Smoothed (incoherent) PE and normal-mode TL agree in trend within a few dB
    # for a range-independent, depth-VARYING waveguide (exercises the PE's
    # refraction term n²≠1, not just free propagation).
    D, f = 200.0, 50.0
    zs, zr = 30.0, 120.0
    prof = (np.array([0.0, D]), np.array([1500.0, 1530.0]))  # downward gradient
    r = np.linspace(500.0, 9000.0, 200)
    nm = normal_modes(f, *prof, source_depth=zs, receiver_depth=zr, ranges_m=r,
                      n_depth_points=1000)
    pe = parabolic_equation(f, *prof, source_depth=zs, max_range=9500.0,
                            range_step=5.0, n_depth_points=1024)
    zi = int(np.argmin(np.abs(pe.depths - zr)))
    pe_tl = np.interp(r, pe.ranges, pe.transmission_loss[zi])

    def smooth(tl: np.ndarray) -> np.ndarray:
        w = np.ones(21) / 21
        return -10 * np.log10(np.convolve(10 ** (-tl / 10), w, mode="same"))

    diff = smooth(pe_tl) - smooth(nm.transmission_loss)
    core = diff[20:-20]
    # constant offset from the Gaussian starter's angular spectrum; low scatter.
    assert core.std() < 2.0


# --- Validation & plots -----------------------------------------------------


def test_invalid_inputs_rejected() -> None:
    with pytest.raises(ValueError, match="depths"):
        normal_modes(20.0, [10.0, 20.0], [1500.0, 1500.0], source_depth=15.0,
                     receiver_depth=15.0)  # does not start at z = 0
    with pytest.raises(ValueError, match="bottom"):
        normal_modes(20.0, *_ISO, source_depth=50.0, receiver_depth=50.0, bottom="soft")
    with pytest.raises(ValueError, match="source_depth"):
        parabolic_equation(50.0, *_ISO, source_depth=150.0)
    with pytest.raises(ValueError, match="launch_angles_deg"):
        ray_trace(*_ISO, source_depth=50.0, launch_angles_deg=[90.0])  # not forward
    with pytest.raises(ValueError, match="range_step"):
        parabolic_equation(50.0, *_ISO, source_depth=50.0, max_range=1000.0,
                           range_step=2000.0)


def test_parabolic_equation_covers_max_range_when_step_not_divisor() -> None:
    # A range step that does not divide max_range must still cover it (the grid
    # extends to at least max_range, not silently short).
    res = parabolic_equation(50.0, *_ISO, source_depth=50.0, max_range=1000.0,
                             range_step=300.0, n_depth_points=64)
    assert res.ranges[-1] >= 1000.0


def test_ray_trace_steep_angle_reaches_max_range() -> None:
    # Range-marching guarantees every valid ray spans [0, max_range], even steep
    # ones (arc-length marching would fall short here).
    res = ray_trace(*_ISO, source_depth=100.0, launch_angles_deg=[60.0],
                    max_range=5000.0, n_steps=3000)
    assert res.ranges[0, -1] == pytest.approx(5000.0)
    # Depth stays within the water column at all times (reflections folded).
    assert np.all((res.depths >= 0.0) & (res.depths <= _ISO[0][-1]))


def test_plots_smoke() -> None:
    nm = normal_modes(20.0, *_ISO, source_depth=36.0, receiver_depth=46.0, n_depth_points=400)
    rt = ray_trace(*_ISO, source_depth=50.0, launch_angles_deg=[-5.0, 0.0, 5.0],
                   max_range=5000.0, n_steps=2000)
    pe = parabolic_equation(50.0, *_ISO, source_depth=50.0, max_range=5000.0,
                            range_step=10.0, n_depth_points=256)
    assert nm.plot() is not None
    assert rt.plot() is not None
    assert pe.plot() is not None


# Independent absolute TL anchors: converged image-source sum (|m| <= 4000)
# for the ideal pressure-release waveguide D = 100 m, c = 1500 m/s, f = 20 Hz,
# zs = 36 m, zr = 46 m, rho = 1000 kg/m3, TL = -20*lg(4*pi*|p|). Unlike the
# analytic modal-sum test, this oracle does not share the Eq. 5.14 prefactor
# with the implementation, so it pins the absolute calibration.
_IMAGE_SOURCE_TL = [
    (200.0, 39.145), (500.0, 42.297), (1000.0, 48.238),
    (3000.0, 52.137), (5000.0, 53.165),
]


def test_normal_modes_match_image_source_oracle() -> None:
    ranges = [r for r, _ in _IMAGE_SOURCE_TL]
    res = normal_modes(20.0, [0.0, 100.0], [1500.0, 1500.0],
                       source_depth=36.0, receiver_depth=46.0,
                       ranges_m=ranges, n_depth_points=3000)
    for (rng, tl_ref), tl_mod in zip(_IMAGE_SOURCE_TL, res.transmission_loss):
        assert tl_mod == pytest.approx(tl_ref, abs=0.02), rng


def test_parabolic_equation_long_range_absolute_anchor() -> None:
    # The paraxial PE converges to the exact field at long range: pin the
    # absolute calibration against the 5 km image-source anchor.
    res = parabolic_equation(20.0, [0.0, 100.0], [1500.0, 1500.0],
                             source_depth=36.0, max_range=5000.0,
                             range_step=5.0, n_depth_points=1024)
    iz = int(np.argmin(np.abs(res.depths - 46.0)))
    tl = float(res.transmission_loss[iz, -1])
    assert tl == pytest.approx(53.165, abs=0.05)
