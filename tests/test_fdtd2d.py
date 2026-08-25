#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Basic physics and determinism tests for the ``fdtd2d`` engine shim.

``scripts/fdtd2d.py`` re-exports :mod:`phonometry.simulation.fdtd`, and the
engine renders the committed FDTD documentation animations, so what
matters here is that (a) the shim import path used by the animation
scripts keeps working, (b) the physics is not obviously wrong (a rigid box
conserves energy, a sponge absorbs it, a centred pulse stays symmetric),
and (c) the output is bit-reproducible run to run. The analytic-oracle
validation of the solver lives in ``tests/simulation/test_fdtd.py``.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

_SCRIPTS = str(pathlib.Path(__file__).resolve().parent.parent / "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import fdtd2d


def _pulse_box(sponge_width: int) -> fdtd2d.FDTD2D:
    """A homogeneous box with a centred Gaussian pulse (odd grid)."""
    sim = fdtd2d.FDTD2D(343.0, 0.05, shape=(81, 101), sponge_width=sponge_width)
    sim.add_source(fdtd2d.GaussianPulse(ix=50, iy=40, width=6 * sim.dt))
    return sim


def test_rigid_box_conserves_energy() -> None:
    sim = _pulse_box(sponge_width=0)
    sim.run(200)
    e_ref = sim.energy()
    sim.run(600)
    # Leapfrog in a rigid box is lossless up to the O(dt) oscillation of the
    # staggered-in-time energy functional: no systematic gain or loss.
    assert sim.energy() == pytest.approx(e_ref, rel=1e-2)


def test_sponge_absorbs_energy() -> None:
    sim = _pulse_box(sponge_width=20)
    sim.run(60)  # pulse radiated, front just reaching the layer
    e_ref = sim.energy()
    sim.run(740)  # wavefront crosses the absorbing layer
    assert sim.energy() < 1e-2 * e_ref


def test_centered_pulse_stays_symmetric() -> None:
    sim = _pulse_box(sponge_width=0)
    sim.run(500)
    # A centred source in a homogeneous rigid box has both mirror symmetries.
    np.testing.assert_allclose(sim.p, sim.p[::-1, :], atol=1e-12)
    np.testing.assert_allclose(sim.p, sim.p[:, ::-1], atol=1e-12)


def test_two_runs_are_bit_identical() -> None:
    frames = [_pulse_box(sponge_width=10).run(300, record_every=50) for _ in range(2)]
    assert frames[0].shape == (7, 81, 101)
    assert np.array_equal(frames[0], frames[1])


def test_cw_source_and_decimation() -> None:
    sim = fdtd2d.FDTD2D(343.0, 0.05, shape=(80, 100), damping=5.0)
    sim.add_source(fdtd2d.CWSource(ix=10, iy=10, frequency=200.0))
    frames = sim.run(400, record_every=100, decimate=2)
    assert frames.shape == (5, 40, 50)
    assert float(np.abs(frames[-1]).max()) > 0.0


def test_heterogeneous_speed_refracts_faster() -> None:
    # A faster right half-space must carry the front further in x than the
    # slow left half does in the same time.
    ny, nx = 81, 161
    c = np.full((ny, nx), 200.0)
    c[:, nx // 2 :] = 400.0
    sim = fdtd2d.FDTD2D(c, 0.05)
    sim.add_source(fdtd2d.GaussianPulse(ix=nx // 2, iy=ny // 2, width=6 * sim.dt))
    sim.run(150)
    row = np.abs(sim.p[ny // 2])
    threshold = 1e-3 * row.max()
    reach_right = np.max(np.nonzero(row > threshold)[0]) - nx // 2
    reach_left = nx // 2 - np.min(np.nonzero(row > threshold)[0])
    assert reach_right > 1.5 * reach_left


def test_scalar_c_requires_shape() -> None:
    with pytest.raises(ValueError, match="shape is required"):
        fdtd2d.FDTD2D(343.0, 0.05)


def test_source_outside_the_grid_is_rejected() -> None:
    sim = fdtd2d.FDTD2D(343.0, 0.05, shape=(10, 10))
    pulse = fdtd2d.GaussianPulse(ix=99, iy=0, width=1e-4)
    with pytest.raises(ValueError, match="outside the grid"):
        sim.add_source(pulse)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"cfl": 1.2}, "cfl"),
        ({"c": 0.0}, "c must be strictly positive"),
        ({"c": np.full((10, 10), np.nan)}, "c must be strictly positive"),
        ({"rho": -1.2}, "rho must be strictly positive"),
        ({"rho": np.full((10, 10), np.inf)}, "rho must be strictly positive"),
        ({"dx": 0.0}, "dx must be positive"),
        ({"dx": np.nan}, "dx must be positive"),
        ({"damping": -1.0}, "damping must be non-negative"),
        ({"damping": np.inf}, "damping must be non-negative"),
        ({"sponge_width": -1}, "sponge_width must be non-negative"),
        ({"sponge_width": 10}, "sponge_width must be narrower"),
        ({"sponge_width": 4, "sponge_reflection": 0.0}, "sponge_reflection"),
        ({"sponge_width": 4, "sponge_reflection": 1.0}, "sponge_reflection"),
        ({"sponge_width": 4, "sponge_sides": ("north",)}, "sponge sides"),
        ({"sponge_width": 4, "sponge_sides": "north"}, "sponge sides"),
        # A malformed grid shape used to be reported by somebody else: numpy,
        # or the sponge_width guard for a zero-length side, or a message
        # blaming c. A complex c or rho ran to completion on its real part.
        ({"shape": (0, 10)}, "shape must be a pair of positive integers"),
        ({"shape": (-1, 10)}, "shape must be a pair of positive integers"),
        ({"shape": (10, 10.5)}, "shape must be a pair of positive integers"),
        ({"shape": 10}, "shape must be a pair of positive integers"),
        ({"shape": (10, 10, 10)}, "shape must be a pair of positive integers"),
        ({"c": 343.0 + 50.0j}, "c must be real"),
        ({"c": np.full((10, 10), 343.0 + 50.0j)}, "c must be real"),
        ({"rho": 1.2 + 3.0j}, "rho must be real"),
    ],
)
def test_constructor_rejects_invalid_arguments(
    kwargs: dict[str, object],
    match: str,
) -> None:
    full: dict[str, object] = {"c": 343.0, "dx": 0.05, "shape": (10, 10), **kwargs}
    c = full.pop("c")
    dx = full.pop("dx")
    with pytest.raises(ValueError, match=match):
        fdtd2d.FDTD2D(c, dx, **full)  # type: ignore[arg-type]


def test_sponge_sides_accepts_a_bare_string() -> None:
    # A single side name must mean that side, not its individual characters.
    sim = fdtd2d.FDTD2D(
        343.0, 0.05, shape=(20, 20), sponge_width=5, sponge_sides="left"
    )
    assert float(sim._decay_p[10, 0]) < 1.0  # left edge absorbs
    assert float(sim._decay_p[10, -1]) == 1.0  # right edge stays rigid


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"width": 0.0}, "width must be positive"),
        ({"width": np.inf}, "width must be positive"),
        ({"width": 1e-4, "amplitude": np.nan}, "amplitude must be finite"),
        ({"width": 1e-4, "t0": np.inf}, "t0 must be finite"),
    ],
)
def test_gaussian_pulse_rejects_invalid_parameters(
    kwargs: dict[str, float],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        fdtd2d.GaussianPulse(ix=0, iy=0, **kwargs)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"frequency": 0.0}, "frequency must be positive"),
        ({"frequency": np.nan}, "frequency must be positive"),
        ({"frequency": 100.0, "ramp_cycles": -1.0}, "ramp_cycles"),
        ({"frequency": 100.0, "amplitude": np.inf}, "amplitude must be finite"),
    ],
)
def test_cw_source_rejects_invalid_parameters(
    kwargs: dict[str, float],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        fdtd2d.CWSource(ix=0, iy=0, **kwargs)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"steps": -1}, "steps must be non-negative"),
        ({"steps": 1, "record_every": 0}, "record_every"),
        ({"steps": 1, "record_every": 1, "decimate": 0}, "decimate"),
    ],
)
def test_run_rejects_invalid_recording_controls(
    kwargs: dict[str, int],
    match: str,
) -> None:
    sim = fdtd2d.FDTD2D(343.0, 0.05, shape=(10, 10))
    with pytest.raises(ValueError, match=match):
        sim.run(**kwargs)
