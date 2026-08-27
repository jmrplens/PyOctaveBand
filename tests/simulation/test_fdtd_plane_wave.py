#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Plane-wave machinery of the 2-D FDTD solver.

Anchors on exact plane-wave physics: the one-way initial packet must stay
plane, travel at the phase speed toward its direction and leave (almost) no
energy behind; the sustained injection line must produce a transversely
plane, unit-amplitude field on the far side and (almost) nothing behind the
line; a rigid wall must double the pressure of an incident packet.
"""

from __future__ import annotations

import numpy as np
import pytest

from phonometry.simulation import (
    FDTD2D,
    CWSource,
    PlaneWaveSource,
)

C0 = 343.0
DX = 0.01


def _centroid(sim: FDTD2D, axis: int) -> float:
    """Energy centroid of ``p**2`` along the given axis (cells)."""
    weight = sim.p**2
    idx = np.arange(sim.p.shape[axis]) + 0.5
    marginal = weight.sum(axis=1 - axis)
    return float((marginal * idx).sum() / marginal.sum())


@pytest.mark.parametrize(
    ("direction", "axis", "sign"),
    [("down", 0, 1.0), ("up", 0, -1.0), ("right", 1, 1.0), ("left", 1, -1.0)],
)
def test_packet_travels_at_phase_speed_one_way(
    direction: str, axis: int, sign: float
) -> None:
    sim = FDTD2D(C0, DX, shape=(120, 120))
    sim.add_plane_wave(direction, center=0.6, width=0.08)
    start = _centroid(sim, axis)
    steps = 30
    for _ in range(steps):
        sim.step()
    moved_cells = (_centroid(sim, axis) - start) * sign
    expected = C0 * steps * sim.dt / DX
    assert moved_cells == pytest.approx(expected, rel=0.02)
    # One-way: energy left in the half-domain behind the launch region.
    total = float((sim.p**2).sum())
    if direction in ("down", "right"):
        behind = sim.p[:40, :] if axis == 0 else sim.p[:, :40]
    else:
        behind = sim.p[80:, :] if axis == 0 else sim.p[:, 80:]
    assert float((behind**2).sum()) / total < 1e-8


def test_packet_stays_plane_and_supports_carrier() -> None:
    sim = FDTD2D(C0, DX, shape=(100, 60))
    sim.add_plane_wave("down", center=0.35, width=0.10, wavelength=0.20)
    for _ in range(25):
        sim.step()
    # Transverse flatness: every row is constant across x.
    assert float(np.abs(np.diff(sim.p, axis=1)).max()) < 1e-12
    # The carrier is still there: sign changes along the travel axis.
    column = sim.p[:, 0]
    assert np.any(column > 0.02)
    assert np.any(column < -0.02)


def test_packet_validation() -> None:
    sim = FDTD2D(C0, DX, shape=(20, 20))
    with pytest.raises(ValueError, match=r"'direction' must be"):
        sim.add_plane_wave("north", center=0.1, width=0.02)
    with pytest.raises(ValueError, match="'width' must be positive"):
        sim.add_plane_wave("down", center=0.1, width=0.0)
    with pytest.raises(ValueError, match="'wavelength' must be positive"):
        sim.add_plane_wave("down", center=0.1, width=0.02, wavelength=-1.0)


def test_packet_rejects_non_finite_parameters() -> None:
    # A NaN width or centre used to be accepted and silently turn the whole
    # pressure field into NaN; an infinite width painted a uniform amplitude
    # over the entire domain.
    sim = FDTD2D(C0, DX, shape=(20, 20))
    with pytest.raises(ValueError, match="'width' must be positive"):
        sim.add_plane_wave("down", center=0.1, width=np.nan)
    with pytest.raises(ValueError, match="'width' must be positive"):
        sim.add_plane_wave("down", center=0.1, width=np.inf)
    with pytest.raises(ValueError, match="'wavelength' must be positive"):
        sim.add_plane_wave("down", center=0.1, width=0.02, wavelength=np.nan)
    with pytest.raises(ValueError, match="center must be finite"):
        sim.add_plane_wave("down", center=np.nan, width=0.02)
    with pytest.raises(ValueError, match="amplitude must be finite"):
        sim.add_plane_wave("down", center=0.1, width=0.02, amplitude=np.nan)
    assert float(np.abs(sim.p).max()) == 0.0  # nothing landed on the field


def test_plane_wave_source_rejects_bad_waveform_and_amplitude() -> None:
    # A non-callable waveform used to surface from inside _inject_plane as
    # the interpreter's "'float' object is not callable", and a NaN
    # amplitude silently NaNed the field step by step.
    with pytest.raises(ValueError, match="waveform must be callable"):
        PlaneWaveSource("down", 1.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="amplitude must be finite"):
        PlaneWaveSource("down", lambda t: 0.0, amplitude=np.nan)


def test_rigid_wall_doubles_incident_pressure() -> None:
    # A packet hitting the rigid bottom edge doubles while reflecting.
    sim = FDTD2D(C0, DX, shape=(160, 40))
    sim.add_plane_wave("down", center=0.6, width=0.06)
    peak_free = float(np.abs(sim.p).max())
    best = 0.0
    for _ in range(260):
        sim.step()
        best = max(best, float(np.abs(sim.p[-6:, :]).max()))
    assert best == pytest.approx(2.0 * peak_free, rel=0.05)


def test_plane_source_steady_state_is_plane_and_one_way() -> None:
    sim = FDTD2D(
        C0,
        DX,
        shape=(160, 80),
        sponge_width=20,
        sponge_sides=("top", "bottom"),
    )
    waveform = CWSource(0, 0, frequency=1000.0)
    sim.add_source(PlaneWaveSource("down", waveform.value, offset=22))
    for _ in range(700):
        sim.step()
    body = sim.p[60:140, :]
    # Transversely plane to machine precision.
    assert float(np.abs(np.diff(body, axis=1)).max()) < 1e-12
    # Unit amplitude within a percent, nothing behind the line.
    assert float(np.abs(body).max()) == pytest.approx(1.0, rel=0.02)
    total = float((sim.p**2).sum())
    assert float((sim.p[:20, :] ** 2).sum()) / total < 1e-3


def test_plane_source_validation_and_geometry_preview() -> None:
    sim = FDTD2D(C0, DX, shape=(40, 40))
    bad_direction = PlaneWaveSource("sideways", lambda t: 0.0)
    with pytest.raises(ValueError, match=r"'direction' must be"):
        sim.add_source(bad_direction)
    bad_offset = PlaneWaveSource("down", lambda t: 0.0, offset=40)
    with pytest.raises(ValueError, match=r"plane-wave offset lies outside the grid"):
        sim.add_source(bad_offset)
    sim.add_source(PlaneWaveSource("left", lambda t: 0.0, offset=2))
    assert len(sim._plane_sources) == 1
    # An injection line may not cross an obstacle.
    mask = np.zeros((40, 40), dtype=bool)
    mask[5, 10:20] = True
    blocked = FDTD2D(C0, DX, shape=(40, 40), obstacle_mask=mask)
    on_obstacle = PlaneWaveSource("down", lambda t: 0.0, offset=5)
    with pytest.raises(
        ValueError, match=r"plane-wave injection line crosses an obstacle"
    ):
        blocked.add_source(on_obstacle)
