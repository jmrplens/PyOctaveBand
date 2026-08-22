#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Tests for the public 2D elastic (P-SV) FDTD simulation API.

Oracles are analytic and independent of the implementation: P- and S-wave
times of flight against ``c_P = sqrt((lambda + 2 mu)/rho)`` and
``c_S = sqrt(mu/rho)``, the Rayleigh-wave speed against the exact
characteristic equation (Cremer, Heckl & Petersson 2005, Eq. 3.149), the
Kirchhoff flexural dispersion ``c_B = (B'/m'')**0.25 sqrt(omega)`` of a
thin free-free strip (with the Timoshenko thickness correction of
Eq. 3.196b), the normal-incidence fluid-solid reflection and transmission
coefficients from the characteristic impedances, the normal-incidence mass
law of a thin immersed steel plate
(:func:`~phonometry.building.prediction.panel_transmission.mass_law_transmission_loss`),
and the exact (bit-for-bit) reduction to the acoustic solver when
``c_s = 0`` everywhere.
"""

from __future__ import annotations

from dataclasses import replace

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy.optimize import brentq

from phonometry.building.prediction.panel_transmission import mass_law_transmission_loss
from phonometry.simulation.elastic_fdtd import (
    ElasticBoundaries,
    ElasticFDTD2D,
    ElasticFDTDResult,
    ElasticRecording,
    ExplosionSource,
    ForceSource,
    elastic_fdtd_simulation,
)
from phonometry.simulation.fdtd import FDTD2D, GaussianPulse

# Aluminium (Rayleigh / body waves), steel (plate) and water (fluid).
CP_AL, CS_AL, RHO_AL = 6320.0, 3130.0, 2700.0
CP_ST, CS_ST, RHO_ST = 5900.0, 3200.0, 7850.0
CP_W, RHO_W = 1480.0, 1000.0


def _rayleigh_speed(c_p: float, c_s: float) -> float:
    """Root of the exact Rayleigh characteristic polynomial (C&H 3.149).

    ``16 kR^6 (kT^2 - kL^2) + 8 kR^4 (2 kL^2 kT^2 - 3 kT^4)
    + 8 kR^2 kT^6 - kT^8 = 0`` with ``k = omega / c``, solved for the real
    root just below ``c_T``.
    """
    kt2, kl2 = 1.0 / c_s**2, 1.0 / c_p**2

    def char(c_r: float) -> float:
        kr2 = 1.0 / c_r**2
        return (
            16.0 * kr2**3 * (kt2 - kl2)
            + 8.0 * kr2**2 * (2.0 * kl2 * kt2 - 3.0 * kt2**2)
            + 8.0 * kr2 * kt2**3
            - kt2**4
        )

    return float(brentq(char, 0.5 * c_s, 0.999 * c_s, xtol=1e-9))


# --- Body-wave speeds by time of flight ------------------------------------


@pytest.fixture(scope="module")
def body_waves() -> ElasticFDTDResult:
    """Explosion (pure P) and vertical force in one aluminium free field.

    The explosion radiates only P; the vertical force radiates P along its
    axis (y) and S perpendicular to it (x). Probes on the +y axis therefore
    see only P arrivals in ``p`` and probes on the +x axis only the S
    arrival in ``vy`` (the force's P has no amplitude at 90 degrees and the
    explosion has no ``vy`` on the x axis), so a single run measures both
    speeds by differential time of flight.
    """
    w = 8e-6
    return elastic_fdtd_simulation(
        CP_AL,
        CS_AL,
        0.002,
        2.2e-4,
        rho=RHO_AL,
        shape=(501, 501),
        sources=[
            ExplosionSource(
                ix=250, iy=250, waveform=GaussianPulse(0, 0, width=w).value
            ),
            ForceSource(
                ix=250,
                iy=250,
                direction="y",
                amplitude=1e6,
                waveform=GaussianPulse(0, 0, width=w).value,
            ),
        ],
        recording=ElasticRecording(
            probes=[(250, 350), (250, 470), (350, 250), (470, 250)],
            probe_fields=("p", "vy"),
        ),
    )


def _arrival(res: ElasticFDTDResult, probe: int, field: int) -> float:
    return float(res.times[1:][np.abs(res.signals[probe, field, 1:]).argmax()])


@pytest.mark.xdist_group("elastic-body-waves")
def test_p_wave_speed_by_time_of_flight(body_waves: ElasticFDTDResult) -> None:
    # P probes 100 and 220 cells below the source: c_P = dr / dt within 1 %
    # (measured: -0.06 %; the pulse spans ~19 cells, so grid dispersion of
    # the dominant wavelength stays well under the bound).
    dt_p = _arrival(body_waves, 1, 0) - _arrival(body_waves, 0, 0)
    c_meas = 120 * body_waves.dx / dt_p
    assert c_meas == pytest.approx(CP_AL, rel=0.01)


@pytest.mark.xdist_group("elastic-body-waves")
def test_s_wave_speed_by_time_of_flight(body_waves: ElasticFDTDResult) -> None:
    # S probes 100 and 220 cells beside the force: c_S = dr / dt within 1 %
    # (measured: +0.02 %).
    dt_s = _arrival(body_waves, 3, 1) - _arrival(body_waves, 2, 1)
    c_meas = 120 * body_waves.dx / dt_s
    assert c_meas == pytest.approx(CS_AL, rel=0.01)


# --- Rayleigh wave along a free surface ------------------------------------


def test_rayleigh_wave_speed_matches_characteristic_equation() -> None:
    # Lamb-style set-up: vertical force just below the free surface of an
    # aluminium half-space, two surface probes 300 cells apart. The
    # dominant surface train travels at c_R, the root of the exact
    # characteristic equation (~0.933 c_T at nu ~ 0.34, inside the C&H
    # Table 3.3 bracket 0.928-0.943). Sampled at ~35 cells per Rayleigh
    # wavelength, comfortably above the 15-20 the second-order
    # stress-imaging surface needs; measured: -0.2 %.
    c_r = _rayleigh_speed(CP_AL, CS_AL)
    assert 0.928 * CS_AL < c_r < 0.943 * CS_AL
    w = 8e-6
    res = elastic_fdtd_simulation(
        CP_AL,
        CS_AL,
        0.002,
        5.0e-4,
        rho=RHO_AL,
        shape=(240, 660),
        sources=[
            ForceSource(
                ix=60,
                iy=0,
                direction="y",
                amplitude=1e6,
                waveform=GaussianPulse(0, 0, width=w).value,
            )
        ],
        recording=ElasticRecording(probes=[(300, 0), (600, 0)], probe_fields=("vy",)),
        boundaries=ElasticBoundaries(
            {
                "top": "free",
                "left": "absorbing",
                "right": "absorbing",
                "bottom": "absorbing",
            },
            absorbing_layer_cells=40,
        ),
    )
    t1 = _arrival(res, 0, 0)
    t2 = _arrival(res, 1, 0)
    c_meas = 300 * res.dx / (t2 - t1)
    assert c_meas == pytest.approx(c_r, rel=0.02)
    assert res.free_sides == ("top",)


# --- Kirchhoff flexural dispersion of a thin strip -------------------------


def test_flexural_dispersion_of_thin_plate_strip() -> None:
    # A free-free steel strip is a plate in cylindrical bending (plane
    # strain): c_B = (B'/m'')**0.25 sqrt(omega) with the apparent modulus
    # E' = 4 mu (lambda + mu) / (lambda + 2 mu) (C&H Eqs. 3.83-3.85). The
    # cross-spectrum phase between two probes gives the per-frequency phase
    # speed over a whole band in one broadband run. Band restricted to
    # lambda_B >= 10 h, where the Timoshenko-corrected speed
    # c_C = c_B (1 - 4 (h/lambda_B)^2) (Eq. 3.196b) deviates from
    # Kirchhoff by <= 4 %; tolerance 3.5 % against c_C covers the grid
    # dispersion and the phase-estimation noise (measured: <= 2.5 %), and
    # the log-log slope check pins the sqrt(omega) scaling itself.
    dx = 0.005
    ny, nx = 10, 2400
    h = (ny - 1) * dx  # surfaces run through the
    mu = RHO_ST * CS_ST**2  # boundary rows of centres
    lam = RHO_ST * (CP_ST**2 - 2.0 * CS_ST**2)
    e_apparent = 4.0 * mu * (lam + mu) / (lam + 2.0 * mu)
    a_kirchhoff = (e_apparent * h**2 / (12.0 * RHO_ST)) ** 0.25

    # Graded damping ramps play the sponge role at both strip ends (the
    # strip is far thinner than any usable sponge_width).
    wd = 250
    sig = np.zeros((ny, nx))
    ramp = (np.arange(wd) + 1.0) / wd
    sig[:, :wd] = 4e4 * (ramp[::-1] ** 2)[np.newaxis, :]
    sig[:, -wd:] = 4e4 * (ramp**2)[np.newaxis, :]
    sim = ElasticFDTD2D(
        CP_ST,
        CS_ST,
        dx,
        rho=RHO_ST,
        shape=(ny, nx),
        free_sides=("top", "bottom"),
        damping=sig,
    )
    w = 2e-4
    src_ix = 400
    sim.add_source(
        ForceSource(
            ix=src_ix,
            iy=4,
            direction="y",
            amplitude=1e6,
            waveform=GaussianPulse(0, 0, width=w).value,
        )
    )
    n_steps = round(9.0e-3 / sim.dt)
    probe_ix = (1200, 1800)
    rec = np.zeros((2, n_steps))
    for i in range(n_steps):
        sim.step()
        rec[0, i] = sim.vy[4, probe_ix[0]]
        rec[1, i] = sim.vy[4, probe_ix[1]]
    times = (np.arange(n_steps) + 1) * sim.dt
    # Gate the direct A0 packet (group speeds 700-2600 m/s in the band) so
    # the weak S0 precursor and any ramp residue stay out of the spectra.
    t0 = 4.0 * w
    gated = np.zeros_like(rec)
    for k, ix in enumerate(probe_ix):
        d = (ix - src_ix) * dx
        gate = np.clip((times - (t0 + d / 2600.0)) / 3e-4, 0.0, 1.0) * np.clip(
            ((t0 + d / 700.0 + 2.0 * w) - times) / 3e-4, 0.0, 1.0
        )
        gated[k] = rec[k] * gate
    spectra = np.fft.rfft(gated, axis=1)
    freqs = np.fft.rfftfreq(n_steps, sim.dt)
    cross = spectra[1] * np.conj(spectra[0])
    distance = (probe_ix[1] - probe_ix[0]) * dx
    band = (freqs >= 550.0) & (freqs <= 2100.0)
    f_b = freqs[band]
    omega = 2.0 * np.pi * f_b
    phase = np.unwrap(np.angle(cross[band]))
    c_b = a_kirchhoff * np.sqrt(omega)
    lambda_b = c_b / f_b
    assert float(lambda_b.min()) >= 10.0 * h
    c_corrected = c_b * (1.0 - 4.0 * (h / lambda_b) ** 2)
    # Fix the absolute 2 pi n of the cross phase with the corrected model
    # at one mid-band bin (the model is within a few %, far below the
    # ~30 % of the total phase span that one wrap represents).
    k_ref = int(np.argmin(np.abs(f_b - 1200.0)))
    wraps = round(
        ((-omega[k_ref] * distance / c_corrected[k_ref]) - phase[k_ref]) / (2.0 * np.pi)
    )
    phase = phase + 2.0 * np.pi * wraps
    c_meas = -omega * distance / phase
    np.testing.assert_allclose(c_meas, c_corrected, rtol=0.035)
    slope = float(np.polyfit(np.log(omega), np.log(c_meas), 1)[0])
    assert slope == pytest.approx(0.5, abs=0.06)


# --- Fluid-solid interface: normal-incidence R and T -----------------------


@pytest.mark.parametrize(
    ("cp2", "cs2", "rho2"),
    [
        pytest.param(CP_ST, CS_ST, RHO_ST, id="water-steel"),
        pytest.param(CP_AL, CS_AL, RHO_AL, id="water-aluminium"),
    ],
)
def test_fluid_solid_normal_incidence_reflection_transmission(
    cp2: float,
    cs2: float,
    rho2: float,
) -> None:
    # A plane pulse in water over a solid half-space. With Z = rho c_P,
    # R = (Z2 - Z1)/(Z2 + Z1) and the transmitted normal-stress amplitude
    # is 2 Z2/(Z1 + Z2) times the incident pressure (no S conversion at
    # normal incidence; the solid behaves as a liquid of rho1, c_l1,
    # Brekhovskikh & Godin 1990 Eq. 4.2.27). R = 0.938069 for steel and
    # 0.840380 for aluminium; measured: R -0.002 % / +0.001 %,
    # T -0.03 % / -0.03 %.
    dx = 0.005
    ny, nx = 2400, 3
    c_p = np.full((ny, nx), CP_W)
    c_s = np.zeros((ny, nx))
    rho = np.full((ny, nx), RHO_W)
    c_p[1200:], c_s[1200:], rho[1200:] = cp2, cs2, rho2
    sim = ElasticFDTD2D(c_p, c_s, dx, rho=rho)
    y = (np.arange(ny) + 0.5) * dx
    p0 = np.exp(-(((y - 3.0) / 0.15) ** 2))
    sim.txx[:] = -p0[:, np.newaxis]
    sim.tyy[:] = -p0[:, np.newaxis]
    # Leapfrog-consistent downward velocity, half a time step back.
    y_face = np.arange(1, ny) * dx + 0.5 * CP_W * sim.dt
    v0 = np.exp(-(((y_face - 3.0) / 0.15) ** 2)) / (RHO_W * CP_W)
    sim.vy += v0[:, np.newaxis]
    n_steps = round(3.4e-3 / sim.dt)
    p_water = np.zeros(n_steps)
    s_solid = np.zeros(n_steps)
    for i in range(n_steps):
        sim.step()
        p_water[i] = -0.5 * (sim.txx[900, 1] + sim.tyy[900, 1])
        s_solid[i] = sim.tyy[1600, 1]
    times = (np.arange(n_steps) + 1) * sim.dt
    z1, z2 = RHO_W * CP_W, rho2 * cp2
    r_exact = (z2 - z1) / (z2 + z1)
    t_exact = 2.0 * z2 / (z1 + z2)
    t_inc = 1.5 / CP_W  # probe 1.5 m below the pulse
    t_ref = 4.5 / CP_W  # via the interface and back
    incident = float(p_water[times < t_inc + 8e-4].max())
    window = p_water[times > t_ref - 4e-4]
    reflected = float(window[np.abs(window).argmax()])
    transmitted = float(s_solid[np.abs(s_solid).argmax()])
    assert reflected / incident == pytest.approx(r_exact, rel=0.02)
    assert -transmitted / incident == pytest.approx(t_exact, rel=0.02)


# --- Mass-law crosscheck: thin steel plate in water ------------------------


def test_immersed_plate_reproduces_normal_incidence_mass_law() -> None:
    # A 3 mm steel plate in water at 50 kHz (far below its ~1 MHz thickness
    # resonance): the transmitted amplitude follows the normal-incidence
    # mass law TL = 10 lg(1 + (pi f m'' / rho0 c0)^2) of
    # mass_law_transmission_loss. Measured: within 0.02 dB of the 8.6 dB
    # prediction; the 0.3 dB bound covers the narrowband packet's finite
    # bandwidth.
    dx = 0.001
    ny, nx = 2600, 3
    f0 = 50e3

    def run(with_plate: bool) -> np.ndarray:
        c_p = np.full((ny, nx), CP_W)
        c_s = np.zeros((ny, nx))
        rho = np.full((ny, nx), RHO_W)
        if with_plate:
            c_p[1300:1303] = CP_ST
            c_s[1300:1303] = CS_ST
            rho[1300:1303] = RHO_ST
        sim = ElasticFDTD2D(c_p, c_s, dx, rho=rho)
        y = (np.arange(ny) + 0.5) * dx
        p0 = np.exp(-(((y - 0.6) / 0.06) ** 2)) * np.sin(
            2.0 * np.pi * f0 * (y - 0.6) / CP_W
        )
        sim.txx[:] = -p0[:, np.newaxis]
        sim.tyy[:] = -p0[:, np.newaxis]
        y_face = np.arange(1, ny) * dx + 0.5 * CP_W * sim.dt
        v0 = (
            np.exp(-(((y_face - 0.6) / 0.06) ** 2))
            * np.sin(2.0 * np.pi * f0 * (y_face - 0.6) / CP_W)
            / (RHO_W * CP_W)
        )
        sim.vy += v0[:, np.newaxis]
        n_steps = round(1.35e-3 / sim.dt)
        trace = np.zeros(n_steps)
        for i in range(n_steps):
            sim.step()
            trace[i] = -0.5 * (sim.txx[1800, 1] + sim.tyy[1800, 1])
        return trace

    reference = run(with_plate=False)
    through = run(with_plate=True)
    tl_meas = 20.0 * np.log10(np.abs(reference).max() / np.abs(through).max())
    tl_exact = float(
        mass_law_transmission_loss(
            f0,
            RHO_ST * 0.003,
            incidence="normal",
            speed_of_sound=CP_W,
            air_density=RHO_W,
        )
    )
    assert tl_meas == pytest.approx(tl_exact, abs=0.3)


# --- Acoustic limit: c_s = 0 reproduces FDTD2D bit for bit ------------------


def test_mu_zero_everywhere_reproduces_acoustic_fdtd() -> None:
    # With c_s = 0 the elastic system degenerates to the acoustic one with
    # txx = tyy = -p, and the elastic update is arranged so the shared
    # terms are evaluated in the acoustic operation order: the two solvers
    # agree bit for bit, heterogeneous maps, sponge and obstacle included.
    rng = np.random.default_rng(0)
    ny, nx = 60, 80
    c = 340.0 + 30.0 * rng.random((ny, nx))
    rho = 1.0 + 0.5 * rng.random((ny, nx))
    mask = np.zeros((ny, nx), dtype=bool)
    mask[25:32, 40:44] = True
    acoustic = FDTD2D(
        c,
        0.01,
        rho=rho,
        sponge_width=10,
        sponge_sides=("left", "top"),
        obstacle_mask=mask,
    )
    elastic = ElasticFDTD2D(
        c,
        0.0,
        0.01,
        rho=rho,
        sponge_width=10,
        sponge_sides=("left", "top"),
        obstacle_mask=mask,
    )
    assert elastic.dt == acoustic.dt
    width = 8 * acoustic.dt
    acoustic.add_source(GaussianPulse(ix=20, iy=20, width=width))
    elastic.add_source(
        ExplosionSource(ix=20, iy=20, waveform=GaussianPulse(0, 0, width=width).value)
    )
    for _ in range(400):
        acoustic.step()
        elastic.step()
    np.testing.assert_array_equal(elastic.p, acoustic.p)
    np.testing.assert_array_equal(elastic.vx, acoustic.vx)
    np.testing.assert_array_equal(elastic.vy, acoustic.vy)
    assert float(np.abs(acoustic.p).max()) > 0.0
    np.testing.assert_array_equal(elastic.txy, np.zeros_like(elastic.txy))


# --- Energy ----------------------------------------------------------------


def test_energy_is_conserved_in_a_closed_domain() -> None:
    # Shear-free rigid walls do no work: after the sources die out the
    # total energy stays constant (measured drift: 0.2 %, the usual
    # leapfrog energy oscillation).
    sim = ElasticFDTD2D(CP_AL, CS_AL, 0.002, rho=RHO_AL, shape=(120, 120))
    sim.add_source(
        ExplosionSource(ix=60, iy=60, waveform=GaussianPulse(0, 0, width=1e-6).value)
    )
    sim.add_source(
        ForceSource(
            ix=40,
            iy=40,
            direction="x",
            amplitude=100.0,
            waveform=GaussianPulse(0, 0, width=1e-6).value,
        )
    )
    sim.run(300)
    e_ref = sim.energy()
    assert e_ref > 0.0
    drifts = []
    for _ in range(5):
        sim.run(200)
        drifts.append(abs(sim.energy() / e_ref - 1.0))
    assert max(drifts) < 0.01


# --- Constructor and source validation -------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"c_p": 6000.0, "c_s": 3000.0}, "shape is required"),
        (
            {"c_p": 6000.0, "c_s": np.zeros((3, 4)), "shape": (4, 4)},
            "c_s map must match",
        ),
        (
            {"c_p": 3000.0, "c_s": 2500.0, "shape": (10, 10)},
            r"c_p\*\*2 must be at least",
        ),
        ({"c_p": 6000.0, "c_s": -1.0, "shape": (10, 10)}, "c_s must be non-negative"),
        (
            {
                "c_p": 6000.0,
                "c_s": 3000.0,
                "shape": (10, 10),
                "rho": np.ones((9, 10)) * 2700.0,
            },
            "rho map must match",
        ),
        (
            {"c_p": 6000.0, "c_s": 3000.0, "shape": (10, 10), "rho": 0.0},
            "rho must be strictly positive",
        ),
        (
            {"c_p": 6000.0, "c_s": 3000.0, "shape": (10, 10), "cfl": 1.0},
            "cfl must lie in",
        ),
        ({"c_p": 6000.0, "c_s": 3000.0, "shape": (1, 10)}, "at least 2 x 2"),
        (
            {"c_p": 6000.0, "c_s": 3000.0, "shape": (10, 10), "free_sides": "north"},
            "unknown free sides",
        ),
        (
            {
                "c_p": 6000.0,
                "c_s": 3000.0,
                "shape": (10, 10),
                "free_sides": "top",
                "sponge_width": 3,
            },
            "both free and absorbing",
        ),
        (
            {"c_p": 6000.0, "c_s": 3000.0, "shape": (10, 10), "damping": -1.0},
            "damping must be non-negative",
        ),
    ],
)
def test_engine_rejects_invalid_parameters(
    kwargs: dict[str, object],
    match: str,
) -> None:
    full: dict[str, object] = {"rho": 2700.0, **kwargs}
    c_p = full.pop("c_p")
    c_s = full.pop("c_s")
    with pytest.raises(ValueError, match=match):
        ElasticFDTD2D(c_p, c_s, 0.01, **full)  # type: ignore[arg-type]


def test_lambda_zero_medium_is_accepted() -> None:
    # c_p = sqrt(2) c_s (lambda = 0) is the validity edge and must pass,
    # including when the ratio is given in rounded floats.
    c_s = 3000.0
    sim = ElasticFDTD2D(
        float(np.sqrt(2.0)) * c_s, c_s, 0.01, rho=2700.0, shape=(10, 10)
    )
    assert float(np.abs(sim.lam).max()) == pytest.approx(0.0, abs=1e-3)


def test_source_validation() -> None:
    mask = np.zeros((20, 20), dtype=bool)
    mask[8:12, 8:12] = True
    sim = ElasticFDTD2D(
        6000.0, 3000.0, 0.01, rho=2700.0, shape=(20, 20), obstacle_mask=mask
    )
    with pytest.raises(ValueError, match="direction must be"):
        ForceSource(ix=1, iy=1, direction="z", waveform=lambda t: 0.0)
    # Built up front so each raises block holds a single throwing call.
    explosion_off_grid = ExplosionSource(ix=20, iy=1, waveform=lambda t: 0.0)
    force_off_grid = ForceSource(ix=19, iy=1, direction="x", waveform=lambda t: 0.0)
    explosion_in_obstacle = ExplosionSource(ix=9, iy=9, waveform=lambda t: 0.0)
    force_in_obstacle = ForceSource(ix=9, iy=9, direction="y", waveform=lambda t: 0.0)
    explosion_bad_ix = ExplosionSource(
        ix=1.5,
        iy=1,  # type: ignore[arg-type]
        waveform=lambda t: 0.0,
    )
    with pytest.raises(ValueError, match="outside the grid"):
        sim.add_source(explosion_off_grid)
    with pytest.raises(ValueError, match="outside the grid"):
        sim.add_source(force_off_grid)
    with pytest.raises(ValueError, match="inside an obstacle"):
        sim.add_source(explosion_in_obstacle)
    with pytest.raises(ValueError, match="inside an obstacle"):
        sim.add_source(force_in_obstacle)
    with pytest.raises(ValueError, match="source ix must be an integer"):
        sim.add_source(explosion_bad_ix)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"sources": []}, "at least one source"),
        ({"duration": 1e-9}, "at least one time step"),
        (
            {"recording": ElasticRecording(probe_fields=())},
            "probe_fields must not be empty",
        ),
        ({"recording": ElasticRecording(probe_fields=("vz",))}, "unknown probe fields"),
        ({"recording": ElasticRecording(probe_fields=("vy", "vy"))}, "must not repeat"),
        (
            {"recording": ElasticRecording(snapshot_field="txx")},
            "snapshot_field must be one of",
        ),
        ({"recording": ElasticRecording(snapshot_every=0)}, "snapshot_every"),
        ({"recording": ElasticRecording(probes=[(99, 0)])}, "outside the grid"),
        (
            {"boundaries": ElasticBoundaries({"north": "free"})},
            "unknown boundary sides",
        ),
        ({"boundaries": ElasticBoundaries("anechoic")}, "must be one of"),
        ({"boundaries": ElasticBoundaries({"left": "open"})}, "must be one of"),
        (
            {"boundaries": ElasticBoundaries("absorbing", absorbing_layer_cells=0)},
            "absorbing_layer_cells",
        ),
    ],
)
def test_simulation_rejects_invalid_arguments(
    kwargs: dict[str, object],
    match: str,
) -> None:
    full: dict[str, object] = {
        "duration": 1e-3,
        "sources": [
            ExplosionSource(ix=5, iy=5, waveform=GaussianPulse(0, 0, width=1e-4).value)
        ],
        **kwargs,
    }
    duration = full.pop("duration")
    with pytest.raises(ValueError, match=match):
        elastic_fdtd_simulation(
            6000.0, 3000.0, 0.05, duration, rho=2700.0, shape=(30, 30), **full
        )


# --- Simulation driver: recording, determinism -----------------------------

#: What the small runs record: two probes with all three fields, and a vy
#: snapshot every 25 steps.
_SMALL_RECORDING = ElasticRecording(
    probes=[(30, 15), (20, 25)],
    probe_fields=("p", "vx", "vy"),
    snapshot_every=25,
    snapshot_field="vy",
)


def _small_run(**kwargs: object) -> ElasticFDTDResult:
    defaults: dict[str, object] = {
        "sources": [
            ExplosionSource(
                ix=10, iy=15, waveform=GaussianPulse(0, 0, width=2e-5).value
            )
        ],
        "recording": _SMALL_RECORDING,
        "boundaries": ElasticBoundaries({"top": "free", "bottom": "absorbing"}),
    }
    defaults.update(kwargs)
    return elastic_fdtd_simulation(
        6000.0, 3000.0, 0.02, 2e-3, rho=2700.0, shape=(40, 60), **defaults
    )  # type: ignore[arg-type]


def test_result_records_signals_and_snapshots() -> None:
    res = _small_run()
    n_steps = round(2e-3 / res.dt)
    assert res.times.shape == (n_steps + 1,)
    assert res.signals.shape == (2, 3, n_steps + 1)
    assert res.probe_fields == ("p", "vx", "vy")
    assert res.times[0] == 0.0
    assert np.all(res.signals[:, :, 0] == 0.0)
    assert res.shape == (40, 60)
    assert res.size == (60 * 0.02, 40 * 0.02)
    assert res.free_sides == ("top",)
    assert res.snapshot_field == "vy"
    np.testing.assert_allclose(
        res.probe_positions[0], [(30 + 0.5) * 0.02, (15 + 0.5) * 0.02]
    )
    assert res.snapshots is not None
    assert res.snapshot_times is not None
    assert res.snapshots.shape == (n_steps // 25 + 1, 40, 60)
    np.testing.assert_allclose(
        res.snapshot_times, np.arange(n_steps // 25 + 1) * 25 * res.dt
    )
    assert float(np.abs(res.signals).max()) > 0.0


def test_two_runs_are_bit_identical() -> None:
    a, b = _small_run(), _small_run()
    np.testing.assert_array_equal(a.signals, b.signals)
    assert a.snapshots is not None
    assert b.snapshots is not None
    np.testing.assert_array_equal(a.snapshots, b.snapshots)


def test_positive_waveform_injects_positive_pressure() -> None:
    # The explosion convention is pressure-like: a positive waveform
    # raises the synthetic pressure at the source cell.
    sim = ElasticFDTD2D(6000.0, 3000.0, 0.01, rho=2700.0, shape=(30, 30))
    sim.add_source(
        ExplosionSource(
            ix=15, iy=15, waveform=GaussianPulse(0, 0, width=8 * sim.dt).value
        )
    )
    sim.run(20)
    assert sim.p[15, 15] > 0.0


def test_orthogonal_free_sides_pin_both_corner_stresses() -> None:
    # Where two free surfaces meet, the corner cell lies on both surface
    # planes: both normal stresses stay pinned to zero and the run stays
    # stable (a free-free-free-free block is a legitimate configuration).
    sim = ElasticFDTD2D(
        6000.0,
        3000.0,
        0.01,
        rho=2700.0,
        shape=(40, 40),
        free_sides=("top", "bottom", "left", "right"),
    )
    sim.add_source(
        ExplosionSource(
            ix=20, iy=20, waveform=GaussianPulse(0, 0, width=8 * sim.dt).value
        )
    )
    sim.run(400)
    assert np.all(np.isfinite(sim.txx))
    assert np.all(np.isfinite(sim.vy))
    for r in (0, -1):
        for c in (0, -1):
            assert sim.txx[r, c] == 0.0
            assert sim.tyy[r, c] == 0.0
    assert float(np.abs(sim.vy).max()) > 0.0


def test_collocated_rejects_unknown_field() -> None:
    sim = ElasticFDTD2D(6000.0, 3000.0, 0.01, rho=2700.0, shape=(10, 10))
    with pytest.raises(ValueError, match="field must be one of"):
        sim.collocated("txy")


# --- Plotting ---------------------------------------------------------------


@pytest.fixture(scope="module")
def small_result() -> ElasticFDTDResult:
    return _small_run()


def test_result_rejects_snapshots_recorded_on_another_grid(
    small_result: ElasticFDTDResult,
) -> None:
    # The grid check is shared with the acoustic result, so it is asserted at
    # both call sites: were this result to stop making it, the acoustic test
    # would still pass. Both results reach the same renderer, which takes the
    # extent of the picture from `shape` alone; measured without the guard,
    # a frame cropped to (30, 60) is drawn across the full 1.2 x 0.8 m
    # domain, a plausible field over cells that were never simulated.
    assert small_result.snapshots is not None
    fewer_rows = small_result.snapshots[:, 5:-5, :]
    with pytest.raises(ValueError, match=r"'snapshots \(axis 1\)'"):
        replace(small_result, snapshots=fewer_rows)
    fewer_columns = small_result.snapshots[:, :, 5:-5]
    with pytest.raises(ValueError, match=r"'snapshots \(axis 2\)'"):
        replace(small_result, snapshots=fewer_columns)


def test_plot_probes(small_result: ElasticFDTDResult) -> None:
    ax = small_result.plot()
    lines = ax.get_lines()
    assert len(lines) == 6  # 2 probes x 3 fields
    np.testing.assert_allclose(lines[0].get_xdata(), small_result.times * 1e3)
    np.testing.assert_allclose(lines[0].get_ydata(), small_result.signals[0, 0])
    assert lines[0].get_label().startswith("$p$, probe")
    assert ax.get_xlabel() == "Time [ms]"
    plt.close(ax.figure)


def test_plot_snapshot(small_result: ElasticFDTDResult) -> None:
    ax = small_result.plot(kind="snapshot", frame=-1)
    assert small_result.snapshots is not None
    images = ax.get_images()
    np.testing.assert_allclose(
        np.asarray(images[0].get_array()), small_result.snapshots[-1]
    )
    assert "$v_y$" in ax.get_title()
    plt.close(ax.figure)


def test_plot_localizes_labels(small_result: ElasticFDTDResult) -> None:
    ax = small_result.plot(language="es")
    assert ax.get_xlabel() == "Tiempo [ms]"
    assert ax.get_title() == "Señales en las sondas del FDTD elástico"
    plt.close(ax.figure)

    ax = small_result.plot(kind="snapshot", language="es")
    assert ax.get_title().startswith("Campo de $v_y$ del FDTD elástico")
    plt.close(ax.figure)


def test_plot_rejects_unknown_kind_and_missing_snapshots() -> None:
    res = _small_run(recording=replace(_SMALL_RECORDING, snapshot_every=None))
    with pytest.raises(ValueError, match="kind"):
        res.plot(kind="field")
    with pytest.raises(ValueError, match="no snapshots"):
        res.plot(kind="snapshot")
