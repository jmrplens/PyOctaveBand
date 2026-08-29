#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Near-to-far-field (NTFF) transformation: analytic and full-wave oracles.

The contour capture and the 2D Kirchhoff-Helmholtz integral are checked in
three tiers, each independent of the code under test:

* **Analytic line sources.** A CW monopole enclosed by a contour must
  transform to an omnidirectional far field whose absolute level follows
  from the 2D free-space Green function ``-(j/4) H0(2)(k r)`` calibrated
  at an independent probe point; a contour that does not enclose the
  source must transform to (nearly) nothing, which is the extinction
  property the scattered-field workflow relies on; and an antiphase source
  pair must produce the two-element array pattern with its nulls.

* **Meshed quadratic-residue diffuser.** A rigid N = 5 QRD (design
  frequency 500 Hz, wells up to 27.4 cm deep on the 7 cm pitch of the
  metadiffuser paper) is meshed with ``obstacle_mask`` cells, driven to
  steady state by a normally incident 2 kHz plane wave, and its NTFF polar
  response is compared against the library's Fraunhofer model
  (:func:`~phonometry.materials.predict_diffuser_polar_response`). The
  Kirchhoff obliquity factor is kept: against a full-wave reference it is
  the better Kirchhoff model (dropping it lowers the correlation from
  0.94 to 0.78 here). The remaining disagreement is physical, not
  numerical: the Fraunhofer model assumes locally reacting wells with no
  evanescent coupling between mouths, an approximation that is coarsest
  for deep open wells well above the design frequency (Cox & D'Antonio,
  ch. 9).

* **Meshed Table-1 metadiffuser.** The five-slit, 2 cm deep
  quadratic-residue metadiffuser of Jimenez, Cox, Romero-Garcia and Groby
  (Sci. Rep. 7, 5389, 2017) is meshed at ``dx = 0.5 mm`` (slits, necks and
  cavities all resolved by at least four cells) and its 2 kHz NTFF polar
  response is compared against the library's TMM + Fraunhofer chain
  (:func:`~phonometry.materials.diffusers.metadiffuser_polar_response`), the
  end-to-end counterpart of the paper's TMM-vs-FEM comparison (their
  Fig. 3g), which the paper itself reports agreeing up to "small
  discrepancies". Grid convergence of the pattern was verified against a
  0.25 mm run of the same scene: levels within 0.45 dB everywhere
  (0.24 dB RMS, pattern correlation 0.9994) and the diffusion
  coefficient within 0.007, so the committed test uses the 0.5 mm grid.

The steady-state protocol everywhere: ramped CW drive, transient run out,
then on-the-fly DFT over a whole number of periods. The scattered far
field is obtained from the *total* contour phasors directly: the enclosed
region is source-free for the incident wave, so the incident contribution
extinguishes in the exterior integral (verified both by the dedicated
extinction test and by a reference-run subtraction changing the meshed-QRD
levels by less than 0.01 dB).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from scipy.special import hankel2

from phonometry import materials
from phonometry.materials import (
    metadiffuser_polar_response,
    metadiffuser_reflection,
    predict_diffuser_polar_response,
)
from phonometry.materials.diffusers.scattering_diffusion import (
    directional_diffusion_coefficient,
)
from phonometry.simulation import (
    FDTD2D,
    ContourPhasors,
    CWSource,
    PlaneWaveSource,
    far_field_from_contour,
)

C0 = 343.0
RHO0 = 1.2
F0 = 2000.0
K0 = 2.0 * np.pi * F0 / C0

#: ISO 17497-2 style receiver arc, in degrees from the panel normal.
POLAR_ANGLES = np.arange(-90.0, 90.1, 5.0)

#: Metadiffuser paper pitch and the Table-1 rows (slit h, neck l_n, cavity
#: l_c, neck w_n, cavity w_c, in mm).
PITCH = 0.07
META_DEPTH, META_BACK = 0.02, 0.003
META_ROWS = (
    (14.7, 13.0, 16.4, 6.2, 9.0),
    (30.9, 9.1, 4.3, 3.5, 9.0),
    (30.9, 9.1, 4.3, 3.5, 9.0),
    (15.7, 13.3, 17.0, 6.3, 9.0),
    (20.3, 18.0, 20.7, 3.2, 9.0),
)

#: QRD twin of the metadiffuser: same residue order, 500 Hz design.
QRD_DEPTHS = np.array([1, 4, 4, 1, 0]) * (C0 / 500.0) / 10.0
QRD_FIN = 0.005


def _levels(pattern: np.ndarray) -> np.ndarray:
    """Polar levels in dB referenced to the peak, floored like the model."""
    magnitude = np.abs(pattern)
    ratio = magnitude / magnitude.max()
    return 20.0 * np.log10(np.maximum(ratio, 1e-10))


# ---------------------------------------------------------------------------
# Analytic line-source oracles.
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def monopole_far_field() -> dict[str, object]:
    """One CW monopole run: enclosing and non-enclosing contour phasors.

    Returns the far-field pattern of the enclosing contour, the pattern of
    an off-centre contour that excludes the source, the finite-distance
    reconstruction, and the reference amplitude ``A`` of the analytic
    field ``p = A H0(2)(k r)`` fitted at an independent probe cell.
    """
    dx, ny, nx = 0.005, 300, 300
    sim = FDTD2D(C0, dx, shape=(ny, nx), sponge_width=40)
    cx, cy = nx // 2, ny // 2
    sim.add_source(CWSource(ix=cx, iy=cy, frequency=F0, ramp_cycles=4.0))
    probe = sim.add_contour_probe(cx - 60, cx + 60, cy - 60, cy + 60, frequencies=[F0])
    probe_off = sim.add_contour_probe(48, 84, cy - 18, cy + 18, frequencies=[F0])
    dt = sim.dt
    sim.run(round(4.5e-3 / dt))
    probe.reset()
    probe_off.reset()
    # Whole periods of on-the-fly DFT plus a manual DFT at a probe cell,
    # 0.55 m from the source, that calibrates the analytic amplitude.
    window = round(10.0 / F0 / dt)
    acc = 0.0 + 0.0j
    for _ in range(window):
        sim.step()
        acc += sim.p[cy, cx + 110] * np.exp(-2j * np.pi * F0 * sim.n * dt)
    amplitude = (2.0 * acc / window) / hankel2(0, K0 * 110 * dx)
    origin = ((cx + 0.5) * dx, (cy + 0.5) * dx)
    angles = np.arange(0.0, 360.0, 5.0)
    phasors = probe.phasors(F0)
    return {
        "phasors": phasors,
        "pattern": far_field_from_contour(phasors, angles, origin=origin),
        "pattern_off": far_field_from_contour(
            probe_off.phasors(F0), angles, origin=origin
        ),
        "at_060": far_field_from_contour(phasors, angles, distance=0.60, origin=origin),
        "amplitude": complex(amplitude),
    }


@pytest.mark.xdist_group("fdtd-ntff-monopole")
def test_monopole_pattern_is_omnidirectional(
    monopole_far_field: dict[str, object],
) -> None:
    # A line source has no directivity: the reconstructed far-field
    # pattern must be flat all around the circle.
    levels = 20.0 * np.log10(np.abs(monopole_far_field["pattern"]))
    assert float(levels.max() - levels.min()) < 0.2


@pytest.mark.xdist_group("fdtd-ntff-monopole")
def test_monopole_level_matches_2d_green_function(
    monopole_far_field: dict[str, object],
) -> None:
    # p = A H0(2)(k r) has the far-field pattern A sqrt(2/(pi k))
    # exp(j pi/4): the NTFF absolute level must reproduce it from the
    # amplitude fitted at the independent probe cell.
    pattern = np.asarray(monopole_far_field["pattern"])
    expected = (
        complex(monopole_far_field["amplitude"])
        * np.sqrt(2.0 / (np.pi * K0))
        * np.exp(0.25j * np.pi)
    )
    level_error = 20.0 * np.log10(np.abs(pattern) / np.abs(expected))
    assert float(np.abs(level_error).max()) < 0.3
    phase_error = np.degrees(np.angle(pattern / expected))
    assert float(np.abs(phase_error).max()) < 5.0


@pytest.mark.xdist_group("fdtd-ntff-monopole")
def test_monopole_finite_distance_matches_hankel(
    monopole_far_field: dict[str, object],
) -> None:
    # The exact (finite-radius) evaluation must reproduce A H0(2)(k R) on
    # a circle outside the contour, in magnitude and phase.
    reconstructed = np.asarray(monopole_far_field["at_060"])
    exact = complex(monopole_far_field["amplitude"]) * hankel2(0, K0 * 0.60)
    assert float(np.abs(np.abs(reconstructed / exact) - 1.0).max()) < 0.03
    assert float(np.abs(np.degrees(np.angle(reconstructed / exact))).max()) < 5.0


@pytest.mark.xdist_group("fdtd-ntff-monopole")
def test_contour_not_enclosing_the_source_extinguishes(
    monopole_far_field: dict[str, object],
) -> None:
    # Fields whose sources lie outside the contour contribute nothing to
    # the exterior integral; the residual is grid-dispersion leakage.
    inside = float(np.abs(monopole_far_field["pattern"]).mean())
    outside = float(np.abs(monopole_far_field["pattern_off"]).max())
    assert outside < 0.02 * inside


@pytest.mark.xdist_group("fdtd-ntff-monopole")
def test_subtracting_a_run_from_itself_cancels(
    monopole_far_field: dict[str, object],
) -> None:
    phasors = monopole_far_field["phasors"]
    assert isinstance(phasors, ContourPhasors)
    null = phasors.subtract(phasors)
    assert np.allclose(null.pressure, 0.0)
    assert np.allclose(null.normal_velocity, 0.0)


def test_dipole_pattern_matches_two_source_oracle() -> None:
    # Two antiphase monopoles 3 cm apart: |F| ~ |sin(k s/2 cos a)| with
    # nulls broadside, the two-element array-factor oracle.
    dx, ny, nx = 0.005, 300, 300
    sim = FDTD2D(C0, dx, shape=(ny, nx), sponge_width=40)
    cx, cy = nx // 2, ny // 2
    sim.add_source(
        CWSource(ix=cx - 3, iy=cy, frequency=F0, amplitude=1.0, ramp_cycles=4.0)
    )
    sim.add_source(
        CWSource(ix=cx + 3, iy=cy, frequency=F0, amplitude=-1.0, ramp_cycles=4.0)
    )
    probe = sim.add_contour_probe(cx - 60, cx + 60, cy - 60, cy + 60, frequencies=[F0])
    dt = sim.dt
    sim.run(round(4.5e-3 / dt))
    probe.reset()
    sim.run(round(10.0 / F0 / dt))
    angles = np.arange(0.0, 360.0, 5.0)
    pattern = np.abs(
        far_field_from_contour(
            probe.phasors(F0), angles, origin=((cx + 0.5) * dx, (cy + 0.5) * dx)
        )
    )
    oracle = np.abs(np.sin(K0 * 3 * dx * np.cos(np.radians(angles))))
    error = pattern / pattern.max() - oracle / oracle.max()
    assert float(np.abs(error).max()) < 0.02
    broadside = np.isin(angles, (90.0, 270.0))
    assert float(pattern[broadside].max()) < 0.01 * float(pattern.max())


# ---------------------------------------------------------------------------
# Meshed Schroeder surfaces: full-wave far field vs the library models.
# ---------------------------------------------------------------------------
def _cw_panel_levels(
    mask: np.ndarray,
    dx: float,
    sponge: int,
    contour: tuple[int, int, int, int],
    origin: tuple[float, float],
    settle: float,
    cycles: float,
) -> np.ndarray:
    """Steady-state 2 kHz scattered polar levels of one meshed panel.

    A normally incident plane wave is injected just inside the top sponge,
    the transient is run out, the contour DFT integrates ``cycles`` whole
    periods and the total-field phasors are transformed straight to the
    far field (incident extinction; see the module docstring). The
    receiver arc of ``POLAR_ANGLES`` is measured from the upward panel
    normal, so a polar angle ``theta`` is the grid direction
    ``theta - 90``. The Courant number 0.9 trims the step count: at 340
    cells per wavelength the numerical dispersion is negligible at any
    stable ``cfl``.
    """
    ny, nx = mask.shape
    sim = FDTD2D(
        C0, dx, shape=(ny, nx), sponge_width=sponge, cfl=0.9, obstacle_mask=mask
    )
    sim.add_source(PlaneWaveSource("down", CWSource(0, 0, F0).value, offset=sponge))
    probe = sim.add_contour_probe(*contour, frequencies=[F0])
    sim.run(round(settle / sim.dt))
    probe.reset()
    sim.run(round(cycles / F0 / sim.dt))
    pattern = far_field_from_contour(
        probe.phasors(F0), POLAR_ANGLES - 90.0, origin=origin
    )
    return _levels(pattern)


def _panel_scene(
    dx: float, slab_depth: float, sponge: int
) -> tuple[np.ndarray, tuple[int, int, int, int], tuple[float, float], int, int]:
    """Empty scene around a 35 cm panel: mask, contour, origin, px0, r_face.

    Vertical budget: top sponge (with the injection line at its inner
    edge), a gap to the contour, the contour-to-face clearance, the slab,
    and a clearance-gap-sponge sandwich below.
    """
    gap = round(0.030 / dx)
    marg = round(0.010 / dx)
    front = round(0.020 / dx)
    face_cells = round(5 * PITCH / dx)
    lat = marg + gap + sponge
    nx = face_cells + 2 * lat
    r_face = sponge + gap + front
    slab = round(slab_depth / dx)
    ny = r_face + slab + marg + gap + sponge
    mask = np.zeros((ny, nx), dtype=bool)
    contour = (
        lat - marg,
        lat + face_cells + marg - 1,
        r_face - front,
        r_face + slab + marg - 1,
    )
    origin = ((lat + face_cells / 2.0) * dx, r_face * dx)
    return mask, contour, origin, lat, r_face


@pytest.fixture(scope="module")
def qrd_levels() -> np.ndarray:
    """NTFF polar levels of the meshed reference QRD at 2 kHz.

    The metadiffuser paper's N = 5 quadratic-residue diffuser (500 Hz
    design, wells to 27.4 cm on the 7 cm pitch), meshed at dx = 1.25 mm:
    the 5 mm fins span four cells and the wavelength 137.
    """
    dx, sponge = 0.00125, 60
    mask, contour, origin, px0, r_face = _panel_scene(
        dx, float(QRD_DEPTHS.max()) + QRD_FIN, sponge
    )
    mask[
        r_face : r_face + round((QRD_DEPTHS.max() + QRD_FIN) / dx),
        px0 : px0 + round(5 * PITCH / dx),
    ] = True
    for n, depth in enumerate(QRD_DEPTHS):
        if depth > 0.0:
            c0 = px0 + round((n * PITCH + QRD_FIN) / dx)
            c1 = px0 + round((n + 1) * PITCH / dx)
            mask[r_face : r_face + round(depth / dx), c0:c1] = False
    return _cw_panel_levels(mask, dx, sponge, contour, origin, settle=6e-3, cycles=10.0)


@pytest.fixture(scope="module")
def metadiffuser_levels() -> np.ndarray:
    """NTFF polar levels of the meshed Table-1 metadiffuser at 2 kHz.

    Meshed at dx = 0.5 mm: the 3.2 mm narrowest neck spans six cells, and
    the unit-cell reflection phases at this step sit within 2 degrees of a
    0.25 mm run. The 8 ms settle covers the source ramp, the flight and
    the resonator ring-up (14 ms moves the pattern correlation by under
    0.003).
    """
    dx, sponge = 0.0005, 60
    mask, contour, origin, px0, r_face = _panel_scene(
        dx, META_DEPTH + META_BACK, sponge
    )
    mask[
        r_face : r_face + round((META_DEPTH + META_BACK) / dx),
        px0 : px0 + round(5 * PITCH / dx),
    ] = True
    lattice = META_DEPTH / 2.0
    for n, (h, l_n, l_c, w_n, w_c) in enumerate(META_ROWS):
        x_slit = n * PITCH + 0.12 * PITCH
        c0 = px0 + round(x_slit / dx)
        c1 = px0 + round((x_slit + h * 1e-3) / dx)
        mask[r_face : r_face + round(META_DEPTH / dx), c0:c1] = False
        for m in range(2):
            y_m = (m + 0.5) * lattice
            x_neck = x_slit + h * 1e-3
            r0 = r_face + round((y_m - 0.5e-3 * w_n) / dx)
            r1 = r_face + round((y_m + 0.5e-3 * w_n) / dx)
            mask[r0:r1, c1 : px0 + round((x_neck + l_n * 1e-3) / dx)] = False
            r0 = r_face + round((y_m - 0.5e-3 * w_c) / dx)
            r1 = r_face + round((y_m + 0.5e-3 * w_c) / dx)
            mask[
                r0:r1,
                px0 + round((x_neck + l_n * 1e-3) / dx) : px0
                + round((x_neck + (l_n + l_c) * 1e-3) / dx),
            ] = False
    return _cw_panel_levels(mask, dx, sponge, contour, origin, settle=8e-3, cycles=10.0)


@pytest.mark.xdist_group("fdtd-ntff-mesh")
def test_meshed_qrd_far_field_matches_fraunhofer_model(qrd_levels: np.ndarray) -> None:
    model = predict_diffuser_polar_response(
        PITCH,
        F0,
        depths=QRD_DEPTHS,
        angles=POLAR_ANGLES,
        periods=1,
        include_obliquity=True,
    )
    # Full wave vs Fraunhofer: high pattern correlation, lobe-level
    # agreement away from grazing, and a diffusion coefficient within
    # 0.1. Measured: correlation 0.94, worst lobe error 5.1 dB inside
    # +/-60 degrees, d 0.57 vs 0.64 (the model without the obliquity
    # factor sits at 0.78 correlation, d 0.71: the near-grazing roll-off
    # the factor carries is real for a finite panel probed by a full-wave
    # solver).
    assert float(np.corrcoef(qrd_levels, model.levels)[0, 1]) > 0.9
    sel = np.abs(POLAR_ANGLES) <= 60.0
    assert float(np.abs(qrd_levels - model.levels)[sel].max()) < 6.0
    assert abs(directional_diffusion_coefficient(qrd_levels) - model.coefficient) < 0.1


@pytest.mark.xdist_group("fdtd-ntff-mesh")
def test_meshed_metadiffuser_mimics_the_meshed_qrd(
    qrd_levels: np.ndarray, metadiffuser_levels: np.ndarray
) -> None:
    # The paper's headline claim (their Fig. 3), reproduced end to end
    # with one independent full-wave solver on both panels: the 2 cm
    # metadiffuser scatters at 2 kHz like the 13.7x deeper QRD whose
    # phase profile it mimics. Measured: pattern correlation 0.97, worst
    # lobe difference 2.9 dB inside +/-60 degrees, and directional
    # diffusion coefficients 0.51 vs 0.57.
    assert float(np.corrcoef(metadiffuser_levels, qrd_levels)[0, 1]) > 0.9
    sel = np.abs(POLAR_ANGLES) <= 60.0
    assert float(np.abs(metadiffuser_levels - qrd_levels)[sel].max()) < 6.0
    assert (
        abs(
            directional_diffusion_coefficient(metadiffuser_levels)
            - directional_diffusion_coefficient(qrd_levels)
        )
        < 0.15
    )


@pytest.mark.xdist_group("fdtd-ntff-mesh")
def test_meshed_metadiffuser_far_field_matches_tmm_model(
    metadiffuser_levels: np.ndarray,
) -> None:
    # End to end against the TMM + Fraunhofer chain, the library-side
    # counterpart of the paper's TMM-vs-FEM comparison, which the paper
    # itself reports agreeing up to small discrepancies. The TMM
    # homogenises each 7 cm cell (0.41 wavelengths at 2 kHz) into one
    # locally reacting reflection coefficient with point-scatterer
    # resonators and analytic end corrections, so its per-well phases
    # differ from the meshed cells by up to about 30 degrees on the
    # near-critical slit 1; with the Kirchhoff obliquity factor restored
    # (the fair Kirchhoff model against a full-wave reference, as in the
    # QRD test) the measured agreement is: correlation 0.89, worst lobe
    # error 4.9 dB inside +/-60 degrees, diffusion coefficient 0.51 vs
    # 0.64. The module's own paper-faithful chain (no obliquity,
    # metadiffuser_polar_response) shares the lobe structure with a
    # weaker correlation, 0.64, and d = 0.71; both are asserted with
    # tolerances wide enough to cover those documented modelling gaps and
    # no more.
    wells = [
        materials.MetadiffuserWell(
            h * 1e-3,
            (
                materials.HelmholtzResonator(
                    l_n * 1e-3, w_n * 1e-3, l_c * 1e-3, w_c * 1e-3
                ),
            )
            * 2,
        )
        for h, l_n, l_c, w_n, w_c in META_ROWS
    ]
    module = metadiffuser_polar_response(
        F0, wells, depth=META_DEPTH, period=PITCH, angles=POLAR_ANGLES, periods=1
    )
    reflection = metadiffuser_reflection(
        np.asarray([F0]), wells, depth=META_DEPTH, period=PITCH
    ).reflection[:, 0]
    obliq = predict_diffuser_polar_response(
        PITCH,
        F0,
        reflection=reflection,
        angles=POLAR_ANGLES,
        periods=1,
        include_obliquity=True,
    )
    sel = np.abs(POLAR_ANGLES) <= 60.0
    d_fdtd = directional_diffusion_coefficient(metadiffuser_levels)
    assert float(np.corrcoef(metadiffuser_levels, obliq.levels)[0, 1]) > 0.8
    assert float(np.abs(metadiffuser_levels - obliq.levels)[sel].max()) < 7.0
    assert abs(d_fdtd - obliq.coefficient) < 0.2
    assert float(np.corrcoef(metadiffuser_levels, module.levels)[0, 1]) > 0.5
    assert abs(d_fdtd - module.coefficient) < 0.25


# ---------------------------------------------------------------------------
# Validation of the probe and transformation APIs.
# ---------------------------------------------------------------------------
def test_contour_probe_validation() -> None:
    sim = FDTD2D(C0, 0.01, shape=(40, 50))
    with pytest.raises(ValueError, match="open cell on both sides"):
        sim.add_contour_probe(0, 10, 5, 10, frequencies=[F0])
    with pytest.raises(ValueError, match="open cell on both sides"):
        sim.add_contour_probe(5, 49, 5, 10, frequencies=[F0])
    with pytest.raises(ValueError, match="open cell on both sides"):
        sim.add_contour_probe(10, 5, 5, 10, frequencies=[F0])
    with pytest.raises(
        ValueError, match=r"frequencies must be a non-empty 1D sequence of positive"
    ):
        sim.add_contour_probe(5, 10, 5, 10, frequencies=[])
    with pytest.raises(
        ValueError, match=r"frequencies must be a non-empty 1D sequence of positive"
    ):
        sim.add_contour_probe(5, 10, 5, 10, frequencies=[0.0])
    probe = sim.add_contour_probe(5, 10, 5, 10, frequencies=[F0])
    with pytest.raises(ValueError, match=r"frequency .* is not tracked by this probe"):
        probe.phasors(999.0)
    with pytest.raises(ValueError, match=r"no samples accumulated yet"):
        probe.phasors(F0)


def test_contour_probe_rejects_touching_obstacles() -> None:
    mask = np.zeros((40, 50), dtype=bool)
    mask[18:22, 20:30] = True
    sim = FDTD2D(C0, 0.01, shape=(40, 50), obstacle_mask=mask)
    with pytest.raises(ValueError, match=r"contour faces touch obstacle cells"):
        sim.add_contour_probe(20, 35, 10, 30, frequencies=[F0])
    sim.add_contour_probe(15, 35, 10, 30, frequencies=[F0])


def test_contour_probe_geometry_record() -> None:
    sim = FDTD2D(C0, 0.01, shape=(40, 50))
    probe = sim.add_contour_probe(5, 10, 20, 25, frequencies=[F0])
    n_points = 2 * (6 + 6)
    assert probe.positions.shape == (n_points, 2)
    assert probe.normals.shape == (n_points, 2)
    assert np.allclose(np.hypot(probe.normals[:, 0], probe.normals[:, 1]), 1.0)
    # The left side runs along x = ix0 dx with an outward -x normal.
    assert np.allclose(probe.positions[:6, 0], 0.05)
    assert np.allclose(probe.normals[:6], (-1.0, 0.0))


def test_far_field_input_validation() -> None:
    sim = FDTD2D(C0, 0.01, shape=(40, 50))
    probe = sim.add_contour_probe(5, 10, 20, 25, frequencies=[F0])
    sim.run(4)
    phasors = probe.phasors(F0)
    not_phasors = np.zeros(4)
    with pytest.raises(TypeError, match=r"contour must be a ContourPhasors"):
        far_field_from_contour(not_phasors, [0.0])  # type: ignore[arg-type]
    with pytest.raises(
        ValueError, match=r"angles must be a non-empty 1D sequence of finite"
    ):
        far_field_from_contour(phasors, [])
    with pytest.raises(ValueError, match=r"does not clear the contour"):
        far_field_from_contour(phasors, [0.0], distance=0.01)
    with pytest.raises(ValueError, match=r"distance must be positive and finite"):
        far_field_from_contour(phasors, [0.0], distance=-1.0)
    with pytest.raises(ValueError, match=r"speed_of_sound must be positive and finite"):
        far_field_from_contour(phasors, [0.0], speed_of_sound=0.0)


def test_contour_phasors_subtract_mismatch() -> None:
    sim = FDTD2D(C0, 0.01, shape=(40, 50))
    p1 = sim.add_contour_probe(5, 10, 20, 25, frequencies=[F0, 2.0 * F0])
    p2 = sim.add_contour_probe(5, 11, 20, 25, frequencies=[F0])
    p3 = sim.add_contour_probe(6, 11, 20, 25, frequencies=[F0])
    sim.run(4)
    base = p1.phasors(F0)
    harmonic = p1.phasors(2.0 * F0)
    longer = p2.phasors(F0)
    shifted = p3.phasors(F0)
    with pytest.raises(
        ValueError, match=r"cannot subtract phasors at different frequencies"
    ):
        base.subtract(harmonic)
    with pytest.raises(
        ValueError,
        match=r"ContourPhasors\.subtract: 'positions' .*'reference\.positions' "
        r".*same shape",
    ):
        base.subtract(longer)
    # Same sample count, one cell along: only the coordinates disagree.
    assert shifted.positions.shape == base.positions.shape
    with pytest.raises(
        ValueError, match=r"cannot subtract phasors sampled on different contours"
    ):
        base.subtract(shifted)


def _square_contour(n_side: int = 4) -> dict[str, Any]:
    """Hand-built phasors on a square contour, the form the docstring invites."""
    side = np.linspace(-0.5, 0.5, n_side)
    positions = np.concatenate(
        [
            np.stack((side, np.full(n_side, -0.5)), axis=1),
            np.stack((np.full(n_side, 0.5), side), axis=1),
            np.stack((side[::-1], np.full(n_side, 0.5)), axis=1),
            np.stack((np.full(n_side, -0.5), side[::-1]), axis=1),
        ]
    )
    normals = np.sign(positions) * (np.abs(positions) >= 0.5)
    n = positions.shape[0]
    return {
        "frequency": F0,
        "positions": positions,
        "normals": normals,
        "pressure": np.exp(1j * K0 * positions[:, 0]),
        "normal_velocity": np.full(n, 0.01 + 0j),
        "segment": 1.0 / n_side,
    }


def test_contour_phasors_shape_mismatch_refused() -> None:
    """Every sample-axis mismatch is refused at construction, both directions.

    Without this the mistake reaches numpy inside the integral, which names
    no field, or, when a quantity collapses to a single sample, reaches
    nobody: it broadcasts over the whole contour and hands back a far-field
    pattern of exactly the expected length.
    """
    fields = _square_contour()
    assert isinstance(ContourPhasors(**fields), ContourPhasors)
    n = fields["positions"].shape[0]
    per_sample = ("positions", "normals", "pressure", "normal_velocity")
    for name in per_sample:
        good = np.asarray(fields[name])
        for bad in (good[:-1], np.concatenate([good, good[:1]]), good[:1]):
            with pytest.raises(ValueError, match="for the same n >= 1"):
                ContourPhasors(**{**fields, name: bad})
    # An extra axis carries the right number of samples past any count.
    for name in ("pressure", "normal_velocity"):
        extra_axis = {**fields, name: np.asarray(fields[name])[np.newaxis]}
        with pytest.raises(ValueError, match="for the same n >= 1"):
            ContourPhasors(**extra_axis)
    with pytest.raises(ValueError, match="for the same n >= 1"):
        ContourPhasors(**{**fields, "positions": np.zeros((n, 3))})


def test_contour_phasors_accepts_hand_built_lists() -> None:
    """Lists coerce to the probe-built arrays, as the class docstring promises."""
    fields = _square_contour()
    listed = {
        name: (value.tolist() if isinstance(value, np.ndarray) else value)
        for name, value in fields.items()
    }
    built = ContourPhasors(**listed)
    assert built.positions.dtype == np.float64
    assert built.pressure.dtype == np.complex128
    assert np.allclose(built.normals, fields["normals"])
    pattern = far_field_from_contour(built, [0.0, 90.0])
    assert pattern.shape == (2,)
