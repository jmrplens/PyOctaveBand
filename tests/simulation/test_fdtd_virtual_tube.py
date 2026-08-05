#  Copyright (c) 2026. Jose Manuel Requena Plens
"""
Virtual impedance-tube experiments: FDTD end to end through the standards.

A narrow rigid-walled FDTD domain is a plane-wave tube. A one-way packet is
launched at a sample modelled as an equivalent fluid (density, speed and a
per-cell damping map), the pressure histories at the normative microphone
positions are reduced through the library's own ISO 10534-2 / ASTM E2611
chains, and the recovered absorption and transmission loss are compared
against the exact analytic answer for that same lossy fluid layer.

The analytic oracle is independent of the solver: with the solver's damping
model (equal decay on pressure and velocity), a uniform lossy region carries
plane waves with ``k = (omega - j sigma) / c`` and the *real* characteristic
impedance ``rho c``; a rigid-backed layer then has the surface impedance
``Zs = -j Z2 / tan(k2 d)`` and the free layer has the classical fluid-layer
transfer matrix, evaluated here through ``air_layer_transfer_matrix`` with
the complex ``k2``.
"""

from __future__ import annotations

import numpy as np
import pytest

from phonometry.materials.absorbers.four_microphone import (
    air_layer_transfer_matrix,
    transfer_matrix_one_load,
)
from phonometry.materials.absorbers.impedance_tube import (
    two_microphone_impedance,
)
from phonometry.simulation import FDTD2D

C0 = 343.0
RHO0 = 1.2
DX = 0.005
NY = 3

# Equivalent-fluid sample: slower, denser and lossy.
C2 = 0.6 * C0
RHO2 = 3.0 * RHO0
SIGMA = 600.0
THICKNESS = 0.10

# Analysis band: comfortably inside the packet spectrum and the tube's
# plane-wave range for the virtual geometry.
BAND = np.arange(300.0, 1201.0, 25.0)


def _run_tube(
    nx: int,
    *,
    sample_cells: slice | None,
    probe_cells: tuple[int, ...],
    steps: int,
    right_impedance: float | None = None,
) -> np.ndarray:
    """Launch a packet down the tube and record the probe histories."""
    c_map = np.full((NY, nx), C0)
    rho_map = np.full((NY, nx), RHO0)
    damping = np.zeros((NY, nx))
    if sample_cells is not None:
        c_map[:, sample_cells] = C2
        rho_map[:, sample_cells] = RHO2
        damping[:, sample_cells] = SIGMA
    # A rho*c impedance edge absorbs the normally incident return wave
    # exactly, which is all a plane-wave tube needs (no sponge fits in a
    # three-row domain).
    boundaries: dict[str, float] = {"left": RHO0 * C0}
    if right_impedance is not None:
        boundaries["right"] = right_impedance
    sim = FDTD2D(
        c_map, DX, rho=rho_map, damping=damping, edge_impedance=boundaries,
    )
    sim.add_plane_wave("right", center=0.35, width=0.05)
    records = np.empty((len(probe_cells), steps))
    for n in range(steps):
        sim.step()
        for k, ix in enumerate(probe_cells):
            records[k, n] = sim.p[1, ix]
    return records


def _spectra(records: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """One-sided spectra of the probe histories and their frequencies."""
    spec = np.fft.rfft(records, axis=1)
    freqs = np.fft.rfftfreq(records.shape[1], dt)
    return freqs, spec


def _band_ratio(
    freqs: np.ndarray, num: np.ndarray, den: np.ndarray
) -> np.ndarray:
    """Transfer function sampled on the analysis band."""
    floor = 1e-12 * float(np.abs(den).max())
    safe = np.abs(den) > floor
    ratio = np.zeros_like(num)
    ratio[safe] = num[safe] / den[safe]
    return np.interp(BAND, freqs, ratio.real) + 1j * np.interp(
        BAND, freqs, ratio.imag
    )


def _analytic_layer_wavenumber() -> np.ndarray:
    """Complex wavenumber of the lossy sample on the analysis band."""
    omega = 2.0 * np.pi * BAND
    return (omega - 1j * SIGMA) / C2


def test_virtual_iso10534_recovers_analytic_absorption() -> None:
    # Rigid-backed sample at the right end; two microphones at the
    # normative positions; the full ISO 10534-2 reduction runs on the
    # simulated histories.
    nx = 280
    face_cell = nx - round(THICKNESS / DX)       # sample front face
    face_x = face_cell * DX
    mic1, mic2 = 219, 229                              # farther, nearer
    x1 = face_x - (mic1 + 0.5) * DX
    spacing = (mic2 - mic1) * DX
    records = _run_tube(
        nx, sample_cells=slice(face_cell, nx),
        probe_cells=(mic1, mic2), steps=9000,
    )
    sim_dt = FDTD2D(C0, DX, shape=(NY, 8)).dt
    freqs, spec = _spectra(records, sim_dt)
    h12 = _band_ratio(freqs, spec[1], spec[0])
    result = two_microphone_impedance(
        h12, frequency=BAND, spacing=spacing, x1=x1, speed_of_sound=C0,
        characteristic_impedance=RHO0 * C0,
    )
    k2 = _analytic_layer_wavenumber()
    z2 = RHO2 * C2
    zs = -1j * z2 / np.tan(k2 * THICKNESS)
    r = (zs - RHO0 * C0) / (zs + RHO0 * C0)
    alpha_exact = 1.0 - np.abs(r) ** 2
    assert np.max(np.abs(result.absorption - alpha_exact)) < 0.035
    # And the absorption is substantial somewhere in band (a real test).
    assert float(alpha_exact.max()) > 0.5


def test_virtual_iso10534_empty_tube_is_rigid() -> None:
    # Without a sample the rigid end reflects everything: |r| ~ 1, alpha ~ 0.
    nx = 280
    mic1, mic2 = 219, 229
    x1 = nx * DX - (mic1 + 0.5) * DX
    spacing = (mic2 - mic1) * DX
    records = _run_tube(
        nx, sample_cells=None, probe_cells=(mic1, mic2), steps=2600,
    )
    sim_dt = FDTD2D(C0, DX, shape=(NY, 8)).dt
    freqs, spec = _spectra(records, sim_dt)
    h12 = _band_ratio(freqs, spec[1], spec[0])
    result = two_microphone_impedance(
        h12, frequency=BAND, spacing=spacing, x1=x1, speed_of_sound=C0,
        characteristic_impedance=RHO0 * C0,
    )
    assert np.max(result.absorption) < 0.03
    assert np.min(np.abs(result.reflection)) > 0.97


def test_virtual_astm_one_load_recovers_layer_tl() -> None:
    # Symmetric free layer mid-tube, anechoic termination (rho c edge), four
    # microphones around it, one-load ASTM E2611 reduction.
    nx = 420
    face_cell = 260
    back_cell = face_cell + round(THICKNESS / DX)
    face_x = face_cell * DX
    up1, up2 = 199, 209          # H1 farther, H2 nearer (upstream)
    dn3, dn4 = 309, 319          # H3 nearer, H4 farther (downstream)
    l1 = face_x - (up2 + 0.5) * DX
    s1 = (up2 - up1) * DX
    l2 = (dn3 + 0.5) * DX - face_x
    s2 = (dn4 - dn3) * DX
    records = _run_tube(
        nx, sample_cells=slice(face_cell, back_cell),
        probe_cells=(up1, up2, dn3, dn4), steps=7000,
        right_impedance=RHO0 * C0,
    )
    sim_dt = FDTD2D(C0, DX, shape=(NY, 8)).dt
    freqs, spec = _spectra(records, sim_dt)
    reference = spec[1]
    loads = tuple(
        _band_ratio(freqs, spec[k], reference) for k in range(4)
    )
    k_air = 2.0 * np.pi * BAND / C0
    matrix = transfer_matrix_one_load(
        loads, l1=l1, s1=s1, l2=l2, s2=s2, thickness=THICKNESS,
        wavenumber=k_air, characteristic_impedance=RHO0 * C0,
    )
    tl_fdtd = matrix.transmission_loss(RHO0 * C0)
    analytic = air_layer_transfer_matrix(
        _analytic_layer_wavenumber(), THICKNESS, RHO2 * C2
    )
    tl_exact = analytic.transmission_loss(RHO0 * C0)
    assert np.max(np.abs(tl_fdtd - tl_exact)) < 0.3
    assert float(np.max(tl_exact)) > 3.0


def test_virtual_astm_empty_tube_tl_is_zero() -> None:
    nx = 420
    up1, up2, dn3, dn4 = 199, 209, 309, 319
    face_x = 260 * DX
    l1 = face_x - (up2 + 0.5) * DX
    s1 = (up2 - up1) * DX
    l2 = (dn3 + 0.5) * DX - face_x
    s2 = (dn4 - dn3) * DX
    records = _run_tube(
        nx, sample_cells=None, probe_cells=(up1, up2, dn3, dn4),
        steps=3600, right_impedance=RHO0 * C0,
    )
    sim_dt = FDTD2D(C0, DX, shape=(NY, 8)).dt
    freqs, spec = _spectra(records, sim_dt)
    loads = tuple(
        _band_ratio(freqs, spec[k], spec[1]) for k in range(4)
    )
    k_air = 2.0 * np.pi * BAND / C0
    matrix = transfer_matrix_one_load(
        loads, l1=l1, s1=s1, l2=l2, s2=s2, thickness=THICKNESS,
        wavenumber=k_air, characteristic_impedance=RHO0 * C0,
    )
    assert np.max(np.abs(matrix.transmission_loss(RHO0 * C0))) < 0.5

def test_damping_map_validation() -> None:
    # The per-cell damping map must be 2-D and match the grid.
    with pytest.raises(ValueError, match="scalar or an"):
        FDTD2D(C0, DX, shape=(NY, 8), damping=np.zeros(8))
    with pytest.raises(ValueError, match="does not match the"):
        FDTD2D(C0, DX, shape=(NY, 8), damping=np.zeros((NY, 9)))
