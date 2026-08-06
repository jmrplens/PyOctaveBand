#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Tests for the public 2D FDTD simulation API.

Oracles are analytic and independent of the implementation: the exact
eigenfrequencies of a rigid rectangular box and of an effectively 1D tube,
free-field pulse arrival times and cylindrical ``1/sqrt(r)`` peak decay,
the image-source echo of a rigid wall, the normal-incidence reflection
coefficient ``R = (Z - rho c)/(Z + rho c)`` of a real-impedance edge
(Attenborough & Van Renterghem 2021, Eqs. 4.33-4.35), the leapfrog
scheme's discrete dispersion relation, and second-order convergence of the
mode frequency under grid refinement.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from phonometry.simulation.fdtd import (
    FDTD2D,
    CWSource,
    FDTDResult,
    GaussianPulse,
    SignalSource,
    fdtd_simulation,
)

C0 = 343.0
RHO0 = 1.2


def _spectrum_peak_near(
    pressure: np.ndarray, dt: float, f_expected: float,
) -> float:
    """Frequency of the highest zero-padded FFT peak within +/-7 %."""
    window = np.hanning(pressure.size)
    spec = np.abs(np.fft.rfft(pressure * window, n=8 * pressure.size))
    freqs = np.fft.rfftfreq(8 * pressure.size, dt)
    sel = (freqs > 0.93 * f_expected) & (freqs < 1.07 * f_expected)
    return float(freqs[sel][np.argmax(spec[sel])])


# --- Rigid-box modes (exact 2D eigenfrequencies) ---------------------------


def test_rigid_box_modes_match_analytic_eigenfrequencies() -> None:
    # f_mn = (c/2) sqrt((m/lx)^2 + (n/ly)^2) for a rigid rectangular box.
    lx, ly, dx = 1.0, 0.7, 0.02
    nx, ny = round(lx / dx), round(ly / dx)
    res = fdtd_simulation(
        C0, dx, 0.35, shape=(ny, nx),
        sources=[GaussianPulse(ix=7, iy=5, width=2.0e-4)],
        probes=[(nx - 4, ny - 3)],
    )
    for m, n in ((1, 0), (0, 1), (1, 1), (2, 0)):
        f_mn = 0.5 * C0 * float(np.hypot(m / lx, n / ly))
        f_meas = _spectrum_peak_near(res.pressures[0], res.dt, f_mn)
        # Measured: <= 0.06 % off; the bound covers FFT-bin quantisation
        # plus the O((k dx)^2) dispersion bias of the scheme.
        assert f_meas == pytest.approx(f_mn, rel=5e-3)


def test_tube_resonances_match_analytic_harmonics() -> None:
    # An ny = 3 rigid duct is effectively 1D: f_m = m c / (2 L).
    length, dx = 2.0, 0.01
    nx = round(length / dx)
    res = fdtd_simulation(
        C0, dx, 0.5, shape=(3, nx),
        sources=[GaussianPulse(ix=30, iy=1, width=1.5e-4)],
        probes=[(nx - 5, 1)],
    )
    for m in (1, 2, 3, 4):
        f_m = m * C0 / (2.0 * length)
        f_meas = _spectrum_peak_near(res.pressures[0], res.dt, f_m)
        assert f_meas == pytest.approx(f_m, rel=2e-3)


# --- Free-field pulse: arrival delay and cylindrical decay -----------------


@pytest.fixture(scope="module")
def free_field() -> FDTDResult:
    """A resolved Gaussian pulse radiating into absorbing boundaries."""
    return fdtd_simulation(
        C0, 0.01, 8.6e-3, shape=(320, 420),
        sources=[GaussianPulse(ix=60, iy=160, width=3.0e-4)],
        probes=[(180, 160), (300, 160)],
        boundaries="absorbing", absorbing_layer_cells=40,
    )


def test_free_field_pulse_arrival_delay(free_field: FDTDResult) -> None:
    # Probes 120 and 240 cells from the source: delay = (r2 - r1)/c.
    t1 = free_field.times[np.argmax(free_field.pressures[0])]
    t2 = free_field.times[np.argmax(free_field.pressures[1])]
    delay_exact = (240 - 120) * free_field.dx / C0
    assert t2 - t1 == pytest.approx(delay_exact, rel=0.02)


def test_free_field_pulse_decays_cylindrically(free_field: FDTDResult) -> None:
    # 2D spreading: peak ~ 1/sqrt(r) (a 2D point source is a line source),
    # clearly distinct from the 3D 1/r law.
    peak1, peak2 = free_field.pressures.max(axis=1)
    r1, r2 = 120 * free_field.dx, 240 * free_field.dx
    assert peak2 / peak1 == pytest.approx(np.sqrt(r1 / r2), rel=0.03)
    alpha = float(np.log(peak1 / peak2) / np.log(r2 / r1))
    assert 0.45 < alpha < 0.57            # 3D spreading would give ~1.0


def test_rigid_wall_reflection_matches_image_source() -> None:
    # Source 50 cells from the rigid left wall, probe 50 cells beyond it:
    # the echo is the image source at -50 cells, path 150 vs 50 cells,
    # arriving (d_image - d_direct)/c after the direct pulse, same sign.
    dx = 0.01
    res = fdtd_simulation(
        C0, dx, 7.5e-3, shape=(300, 300),
        sources=[GaussianPulse(ix=50, iy=150, width=1.0e-4)],
        probes=[(100, 150)],
        boundaries={"right": "absorbing", "top": "absorbing",
                    "bottom": "absorbing"},
        absorbing_layer_cells=30,
    )
    p = res.pressures[0]
    i_direct = int(np.argmax(p))
    masked = p.copy()
    masked[: i_direct + int(1.5e-3 / res.dt)] = 0.0
    i_echo = int(np.argmax(masked))
    delay_exact = (150 - 50) * dx / C0
    delay = res.times[i_echo] - res.times[i_direct]
    assert delay == pytest.approx(delay_exact, rel=0.03)
    assert p[i_echo] > 0.0                # rigid wall: R = +1, no sign flip


# --- Impedance boundary: normal-incidence reflection coefficient -----------


def _measure_edge_reflection(z_over_rhoc: float) -> float:
    """Plane-pulse reflection coefficient of the right impedance edge."""
    nx, dx = 1200, 0.01
    sim = FDTD2D(C0, dx, rho=RHO0, shape=(3, nx),
                 edge_impedance={"right": z_over_rhoc * RHO0 * C0})
    x = (np.arange(nx) + 0.5) * dx
    # A y-uniform initial pressure splits into two plane pulses; the
    # right-going half reflects off the impedance edge. The run stops
    # before the left-going half returns from the rigid left wall.
    sim.p[:] = np.exp(-(((x - 6.0) / 0.15) ** 2))[None, :]
    probe = 900
    n_steps = round(0.032 / sim.dt)
    trace = np.empty(n_steps)
    for i in range(n_steps):
        sim.step()
        trace[i] = sim.p[1, probe]
    times = (np.arange(n_steps) + 1) * sim.dt
    t_return = (12.0 - 6.0) / C0 + (12.0 - 9.0) / C0
    incident = trace[times < t_return - 1.5 * 0.15 / C0]
    reflected = trace[times > t_return]
    return float(reflected[np.abs(reflected).argmax()] / incident.max())


@pytest.mark.parametrize("z_over_rhoc", [3.0, 1.0 / 3.0, 0.05, 20.0])
def test_impedance_edge_reflection_coefficient(z_over_rhoc: float) -> None:
    # Normal incidence: R = (Z - rho c)/(Z + rho c), including the sign
    # flip for Z < rho c. Measured within 0.002 of theory.
    r_theory = (z_over_rhoc - 1.0) / (z_over_rhoc + 1.0)
    assert _measure_edge_reflection(z_over_rhoc) == pytest.approx(
        r_theory, abs=0.01)


def test_matched_impedance_edge_is_anechoic() -> None:
    # Z = rho c absorbs a normally incident plane pulse almost completely.
    assert abs(_measure_edge_reflection(1.0)) < 5e-3


def test_impedance_edge_dissipates_energy() -> None:
    sim = FDTD2D(C0, 0.02, rho=RHO0, shape=(60, 60),
                 edge_impedance={"left": RHO0 * C0, "top": 2.0 * RHO0 * C0})
    sim.add_source(GaussianPulse(ix=30, iy=30, width=10 * sim.dt))
    sim.run(200)
    e_ref = sim.energy()
    sim.run(3000)
    assert sim.energy() < 1e-6 * e_ref


# --- Numerical dispersion and convergence ----------------------------------


def _measured_mode_frequency(nx: int, length: float = 1.0) -> tuple[float,
                                                                    float]:
    """Fundamental frequency of a 1D rigid tube started on the exact mode.

    Returns the zero-crossing estimate and the analytic prediction of the
    scheme's discrete dispersion relation
    ``sin(w dt / 2) = S sin(k dx / 2)`` with ``S = c dt / dx``.
    """
    dx = length / nx
    sim = FDTD2D(C0, dx, shape=(3, nx))
    k = np.pi / length
    x = (np.arange(nx) + 0.5) * dx
    sim.p[:] = np.cos(k * x)[None, :]
    f_exact = C0 / (2.0 * length)
    steps = round(40.0 / f_exact / sim.dt)
    trace = np.empty(steps)
    for i in range(steps):
        sim.step()
        trace[i] = sim.p[1, 0]
    times = (np.arange(steps) + 1) * sim.dt
    sign_change = np.where(np.diff(np.sign(trace)) != 0)[0]
    t_zero = times[sign_change] - trace[sign_change] * (
        times[sign_change + 1] - times[sign_change]
    ) / (trace[sign_change + 1] - trace[sign_change])
    f_measured = (t_zero.size - 1) / (2.0 * (t_zero[-1] - t_zero[0]))
    courant = C0 * sim.dt / dx
    f_dispersion = float(
        2.0 * np.arcsin(courant * np.sin(k * dx / 2.0)) / sim.dt
        / (2.0 * np.pi))
    return float(f_measured), f_dispersion


def test_mode_frequency_follows_discrete_dispersion_relation() -> None:
    # The measured frequency reproduces the leapfrog dispersion relation
    # (the discrete counterpart of Attenborough & Van Renterghem Eq. 4.15)
    # far more tightly than it matches the continuum frequency.
    f_measured, f_dispersion = _measured_mode_frequency(nx=25)
    assert f_measured == pytest.approx(f_dispersion, rel=1e-4)


def test_grid_refinement_converges_at_second_order() -> None:
    f_exact = C0 / 2.0
    errors = [abs(_measured_mode_frequency(nx)[0] - f_exact) / f_exact
              for nx in (25, 50, 100)]
    orders = [float(np.log2(errors[i] / errors[i + 1])) for i in range(2)]
    for order in orders:
        assert 1.7 < order < 2.3


# --- Sources ----------------------------------------------------------------


def test_signal_source_reproduces_gaussian_pulse() -> None:
    # A SignalSource fed with the sampled Gaussian waveform at the
    # simulation rate reproduces the GaussianPulse run.
    sim_a = FDTD2D(C0, 0.05, shape=(40, 50))
    pulse = GaussianPulse(ix=20, iy=20, width=8 * sim_a.dt)
    sim_a.add_source(pulse)
    sim_a.run(200)

    sim_b = FDTD2D(C0, 0.05, shape=(40, 50))
    t = np.arange(1, 202) * sim_b.dt
    sim_b.add_source(SignalSource(
        ix=20, iy=20, samples=np.array([0.0, *(pulse.value(ti) for ti in t)]),
        sample_rate=1.0 / sim_b.dt))
    sim_b.run(200)
    np.testing.assert_allclose(sim_b.p, sim_a.p, atol=1e-12)


def test_signal_source_is_zero_outside_its_span() -> None:
    src = SignalSource(ix=0, iy=0, samples=np.array([1.0, 2.0, 3.0]),
                       sample_rate=10.0, amplitude=2.0)
    assert src.value(-0.01) == 0.0
    assert src.value(0.0) == 2.0
    assert src.value(0.05) == pytest.approx(3.0)   # linear interpolation
    assert src.value(0.2) == pytest.approx(6.0)    # last sample
    assert src.value(0.21) == 0.0


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"samples": np.zeros((2, 3)), "sample_rate": 8000.0},
         "samples must be a 1D array"),
        ({"samples": np.zeros(0), "sample_rate": 8000.0},
         "samples must not be empty"),
        ({"samples": np.array([np.nan]), "sample_rate": 8000.0},
         "samples must be finite"),
        ({"samples": np.zeros(4), "sample_rate": 0.0},
         "sample_rate must be positive"),
        ({"samples": np.zeros(4), "sample_rate": 8000.0,
          "amplitude": np.inf}, "amplitude must be finite"),
    ],
)
def test_signal_source_rejects_invalid_parameters(
    kwargs: dict[str, object], match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        SignalSource(ix=0, iy=0, **kwargs)


# --- Obstacles ---------------------------------------------------------------


def _slab_run(
    obstacle: np.ndarray | None,
) -> tuple[float, float, np.ndarray | None]:
    """Amplitude behind / beside the slab and the snapshots of one run."""
    res = fdtd_simulation(
        C0, 0.02, 7.0e-3, shape=(100, 150),
        sources=[GaussianPulse(ix=30, iy=50, width=5e-5)],
        probes=[(110, 50), (30, 90)],      # behind the slab / open path
        obstacle_mask=obstacle, boundaries="absorbing",
        snapshot_every=50,
    )
    return (float(np.abs(res.pressures[0]).max()),
            float(np.abs(res.pressures[1]).max()), res.snapshots)


def test_obstacle_casts_a_shadow_and_stays_silent_inside() -> None:
    # A rigid slab between source and probe: the probe behind it only
    # receives the weaker wave diffracted around the slab ends, and the
    # field inside the slab stays exactly zero. The 7 ms duration covers
    # both the 4.7 ms direct and the ~4.8 ms diffracted arrivals.
    mask = np.zeros((100, 150), dtype=bool)
    mask[40:60, 70:74] = True              # a vertical rigid slab
    shadowed, open_path, snapshots = _slab_run(mask)
    unblocked, _, _ = _slab_run(None)
    # Diffraction reaches the probe (nonzero) but the slab attenuates it
    # clearly (measured: 0.57x the open-path amplitude with the slab and
    # 1.31x without it, i.e. the slab removes ~56 % of the peak).
    assert 0.1 * open_path < shadowed < 0.75 * open_path
    assert shadowed < 0.6 * unblocked
    assert snapshots is not None
    assert float(np.abs(snapshots[-1][mask]).max()) == 0.0


def test_obstacle_mask_validation() -> None:
    mask = np.zeros((10, 10), dtype=bool)
    mask[5, 5] = True
    float_mask = mask.astype(float)
    full_mask = np.ones((10, 10), dtype=bool)
    masked_source = GaussianPulse(ix=5, iy=5, width=1e-4)
    open_source = GaussianPulse(ix=1, iy=1, width=1e-4)
    with pytest.raises(ValueError, match="match the grid shape"):
        FDTD2D(C0, 0.05, shape=(10, 12), obstacle_mask=mask)
    with pytest.raises(ValueError, match="boolean"):
        FDTD2D(C0, 0.05, shape=(10, 10), obstacle_mask=float_mask)
    with pytest.raises(ValueError, match="open cells"):
        FDTD2D(C0, 0.05, shape=(10, 10), obstacle_mask=full_mask)
    sim = FDTD2D(C0, 0.05, shape=(10, 10), obstacle_mask=mask)
    with pytest.raises(ValueError, match="inside an obstacle"):
        sim.add_source(masked_source)
    with pytest.raises(ValueError, match="inside an obstacle"):
        fdtd_simulation(C0, 0.05, 1e-3, shape=(10, 10), obstacle_mask=mask,
                        sources=[open_source], probes=[(5, 5)])


# --- Boundary specification and CFL stability ------------------------------


@pytest.mark.parametrize(
    ("boundaries", "match"),
    [
        ({"north": "rigid"}, "unknown boundary sides"),
        ({"left": "open"}, "must be one of"),
        ("anechoic", "must be one of"),
    ],
)
def test_boundary_spec_rejects_unknown_values(
    boundaries: object, match: str,
) -> None:
    sources = [GaussianPulse(ix=5, iy=5, width=1e-4)]
    with pytest.raises(ValueError, match=match):
        fdtd_simulation(C0, 0.05, 1e-3, shape=(30, 30),
                        sources=sources, boundaries=boundaries)


@pytest.mark.parametrize(
    ("impedance", "match"),
    [
        ({"left": 0.0}, "strictly positive"),
        ({"left": np.full(30, np.inf)}, "strictly positive"),
        ({"left": np.zeros(7)}, "length 30"),
        ({"up": 400.0}, "unknown impedance sides"),
    ],
)
def test_edge_impedance_validation(
    impedance: dict[str, object], match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        FDTD2D(C0, 0.05, shape=(30, 30),
               edge_impedance=impedance)


def test_side_cannot_be_absorbing_and_impedance() -> None:
    with pytest.raises(ValueError, match="both absorbing"):
        FDTD2D(C0, 0.05, shape=(30, 30), sponge_width=5,
               sponge_sides="left", edge_impedance={"left": 400.0})


def _duct_echo(
    boundaries: str | dict[str, float | np.ndarray],
) -> tuple[float, np.ndarray]:
    """Right-wall echo/incident peak ratio of a duct probe via the public API.

    Source at x = 9 m in a 12 m duct, probe at x = 6 m: the left-going pulse
    passes the probe at ~9.5 ms, the right-going one reflects off the right
    boundary and returns at ~27 ms, and the run stops before the left-wall
    echo (~44 ms) arrives.
    """
    res = fdtd_simulation(
        C0, 0.01, 0.030, rho=RHO0, shape=(3, 1200),
        sources=[GaussianPulse(ix=900, iy=1, width=2e-4)],
        probes=[(600, 1)], boundaries=boundaries)
    p, t = res.pressures[0], res.times
    t_inc = 8e-4 + 3.0 / C0
    t_echo = 8e-4 + 9.0 / C0
    incident = float(p[np.abs(t - t_inc) < 1.5e-3].max())
    window = p[np.abs(t - t_echo) < 1.5e-3]
    return float(window[np.abs(window).argmax()] / incident), res.pressures


def test_boundary_impedance_through_the_public_api() -> None:
    # The boundaries mapping accepts a numeric impedance per side: a matched
    # right edge swallows the pulse a rigid duct reflects. The point source
    # is not a perfect plane wave in the 3-cell duct, so the tolerances are
    # looser than the plane-pulse engine test above (measured: rigid +0.93,
    # matched -0.002, 3 rho c +0.47 vs the +0.50 plane-wave value).
    rigid_ratio, _ = _duct_echo("rigid")
    matched_ratio, _ = _duct_echo({"right": RHO0 * C0})
    partial_ratio, _ = _duct_echo({"right": 3.0 * RHO0 * C0})
    assert rigid_ratio > 0.85
    assert abs(matched_ratio) < 0.02
    assert partial_ratio == pytest.approx(0.5, abs=0.06)


def test_per_cell_impedance_array_matches_the_scalar() -> None:
    # A per-edge-cell 1D array with one value per cell is the scalar case.
    _, scalar = _duct_echo({"right": RHO0 * C0})
    _, array = _duct_echo({"right": np.full(3, RHO0 * C0)})
    assert np.array_equal(scalar, array)


def test_damping_through_the_public_api() -> None:
    def peak(damping: float) -> float:
        res = fdtd_simulation(
            C0, 0.05, 5e-3, shape=(40, 60), damping=damping,
            sources=[GaussianPulse(ix=10, iy=10, width=1e-4)],
            probes=[(40, 20)])
        return float(np.abs(res.pressures).max())

    undamped, damped = peak(0.0), peak(800.0)
    assert 0.0 < damped < 0.1 * undamped   # measured: 0.019 * undamped


@pytest.mark.parametrize("cfl", [0.0, -0.5, 1.0, 1.5, np.nan])
def test_unstable_or_invalid_cfl_is_rejected(cfl: float) -> None:
    # dt = cfl * dx / (c sqrt(2)); cfl is the Courant number CN of
    # Eq. (4.13) and the explicit scheme requires CN <= 1 (Eq. 4.14).
    sources = [GaussianPulse(ix=5, iy=5, width=1e-4)]
    with pytest.raises(ValueError, match="cfl"):
        fdtd_simulation(C0, 0.05, 1e-3, shape=(20, 20), cfl=cfl,
                        sources=sources)


def test_stable_cfl_values_are_accepted() -> None:
    for cfl in (0.1, 0.6, 0.99):
        res = fdtd_simulation(C0, 0.05, 2e-3, shape=(20, 20), cfl=cfl,
                              sources=[GaussianPulse(ix=5, iy=5, width=1e-4)],
                              probes=[(10, 10)])
        assert np.all(np.isfinite(res.pressures))


# --- Simulation driver: recording, determinism, validation -----------------


def test_result_records_probes_and_snapshots() -> None:
    res = fdtd_simulation(
        C0, 0.05, 5e-3, shape=(40, 60),
        sources=[CWSource(ix=10, iy=10, frequency=800.0)],
        probes=[(30, 20), (50, 35)], snapshot_every=25,
    )
    n_steps = round(5e-3 / res.dt)
    assert res.times.shape == (n_steps + 1,)
    assert res.pressures.shape == (2, n_steps + 1)
    assert res.times[0] == 0.0
    assert np.all(res.pressures[:, 0] == 0.0)
    assert res.shape == (40, 60)
    assert res.size == (60 * 0.05, 40 * 0.05)
    np.testing.assert_allclose(res.probe_positions[0],
                               [(30 + 0.5) * 0.05, (20 + 0.5) * 0.05])
    assert res.snapshots is not None
    assert res.snapshot_times is not None
    assert res.snapshots.shape == (n_steps // 25 + 1, 40, 60)
    np.testing.assert_allclose(
        res.snapshot_times, np.arange(n_steps // 25 + 1) * 25 * res.dt)
    assert float(np.abs(res.pressures).max()) > 0.0


def test_two_runs_are_bit_identical() -> None:
    def run() -> FDTDResult:
        return fdtd_simulation(
            C0, 0.02, 3e-3, shape=(80, 100),
            sources=[GaussianPulse(ix=50, iy=40, width=1e-4)],
            probes=[(70, 40)], boundaries="absorbing",
            absorbing_layer_cells=15, snapshot_every=40,
        )

    a, b = run(), run()
    assert np.array_equal(a.pressures, b.pressures)
    assert a.snapshots is not None
    assert b.snapshots is not None
    assert np.array_equal(a.snapshots, b.snapshots)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"sources": []}, "at least one source"),
        ({"duration": 0.0}, "duration must be positive"),
        ({"duration": 1e-9}, "at least one time step"),
        ({"snapshot_every": 0}, "snapshot_every"),
        ({"snapshot_every": 2.5}, "snapshot_every must be an integer"),
        ({"probes": [(99, 0)]}, "outside the grid"),
        ({"probes": [(2.5, 3)]}, "probe ix must be an integer"),
        ({"probes": [(2, 3.5)]}, "probe iy must be an integer"),
        ({"boundaries": "absorbing", "absorbing_layer_cells": 0},
         "absorbing_layer_cells"),
        ({"boundaries": "absorbing", "absorbing_layer_cells": 2.5},
         "absorbing_layer_cells must be an integer"),
    ],
)
def test_simulation_rejects_invalid_arguments(
    kwargs: dict[str, object], match: str,
) -> None:
    full: dict[str, object] = {
        "duration": 1e-3,
        "sources": [GaussianPulse(ix=5, iy=5, width=1e-4)],
        **kwargs,
    }
    duration = full.pop("duration")
    with pytest.raises(ValueError, match=match):
        fdtd_simulation(C0, 0.05, duration,
                        shape=(30, 30), **full)


def test_non_integral_counts_and_coordinates_are_rejected() -> None:
    # Fractional cell counts or indices raise instead of being truncated
    # (a float probe would silently record a different cell) or crashing
    # later inside numpy indexing, slicing or range().
    with pytest.raises(ValueError, match="sponge_width must be an integer"):
        FDTD2D(C0, 0.05, shape=(20, 20), sponge_width=2.5)  # type: ignore[arg-type]
    sim = FDTD2D(C0, 0.05, shape=(20, 20))
    float_source = GaussianPulse(ix=2.5, iy=3, width=1e-4)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="source ix must be an integer"):
        sim.add_source(float_source)
    with pytest.raises(ValueError, match="steps must be an integer"):
        sim.run(1.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="record_every must be an integer"):
        sim.run(10, record_every=1.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="decimate must be an integer"):
        sim.run(10, decimate=1.5)  # type: ignore[arg-type]
    assert sim.n == 0                  # nothing ran on the rejected calls


# --- Plotting ---------------------------------------------------------------


@pytest.fixture(scope="module")
def small_result() -> FDTDResult:
    mask = np.zeros((30, 40), dtype=bool)
    mask[12:18, 20:22] = True
    return fdtd_simulation(
        C0, 0.05, 3e-3, shape=(30, 40),
        sources=[GaussianPulse(ix=8, iy=15, width=1e-4)],
        probes=[(30, 15)], obstacle_mask=mask, snapshot_every=20,
    )


def test_plot_probes(small_result: FDTDResult) -> None:
    ax = small_result.plot()
    line = ax.get_lines()[0]
    np.testing.assert_allclose(line.get_xdata(), small_result.times * 1e3)
    np.testing.assert_allclose(line.get_ydata(), small_result.pressures[0])
    assert ax.get_xlabel() == "Time [ms]"
    plt.close(ax.figure)


def test_plot_snapshot(small_result: FDTDResult) -> None:
    ax = small_result.plot(kind="snapshot", frame=-1)
    assert small_result.snapshots is not None
    images = ax.get_images()
    assert len(images) == 2                 # field + obstacle overlay
    np.testing.assert_allclose(np.asarray(images[0].get_array()),
                               small_result.snapshots[-1])
    assert "ms" in ax.get_title()
    plt.close(ax.figure)


def test_plot_snapshot_cmap_follows_background(small_result: FDTDResult) -> None:
    """Diverging default: white-centred on light axes, black-centred on dark."""
    ax = small_result.plot(kind="snapshot")
    assert ax.get_images()[0].get_cmap().name == "RdBu_r"
    plt.close(ax.figure)

    _fig, dark_ax = plt.subplots()
    dark_ax.set_facecolor("black")
    ax = small_result.plot(kind="snapshot", ax=dark_ax)
    assert ax.get_images()[0].get_cmap().name == "phonometry_field_dark"
    plt.close(ax.figure)

    _fig, dark_ax = plt.subplots()
    dark_ax.set_facecolor("black")
    ax = small_result.plot(kind="snapshot", ax=dark_ax, cmap="viridis")
    assert ax.get_images()[0].get_cmap().name == "viridis"
    plt.close(ax.figure)


def test_plot_rejects_unknown_kind_and_missing_snapshots() -> None:
    res = fdtd_simulation(C0, 0.05, 1e-3, shape=(20, 20),
                          sources=[GaussianPulse(ix=5, iy=5, width=1e-4)],
                          probes=[(10, 10)])
    with pytest.raises(ValueError, match="kind"):
        res.plot(kind="field")
    with pytest.raises(ValueError, match="no snapshots"):
        res.plot(kind="snapshot")
