#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Parity tests: ``scripts/fdtd_gpu.py`` versus the library FDTD engine.

``GpuFDTD2D`` is a backend-injectable replica of
:class:`phonometry.simulation.fdtd.FDTD2D` for the feature subset the
documentation animations use. These tests lock the replica to the library
step by step on small grids: with ``xp=numpy`` every recorded pressure
field must match the reference within 1e-12 of the field peak (in practice
the NumPy path is bit-identical because the update expressions share the
exact operation order). When CuPy and a CUDA device are available the same
scenarios run with ``xp=cupy`` under a looser (ULP-scale reassociation)
tolerance; on machines without a GPU those cases skip.

The job-packaging round trip of ``fdtd_gpu_remote.build_job`` through
``job_runner.run_job`` is covered too, since the remote runner feeds the
engine only through that path.
"""

from __future__ import annotations

import os
import pathlib
import sys
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from phonometry.simulation.fdtd import FDTD2D

if TYPE_CHECKING:
    from types import ModuleType

_SCRIPTS = str(pathlib.Path(__file__).resolve().parent.parent / "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import fdtd_gpu
import fdtd_gpu_remote
import job_runner


def _cupy_or_none() -> ModuleType | None:
    """CuPy with a working CUDA device, else None."""
    try:
        import cupy

        cupy.cuda.runtime.getDeviceCount()
        cupy.arange(1).sum()
    except Exception:  # noqa: BLE001 - any import/CUDA failure means no GPU
        return None
    return cupy


_CUPY = _cupy_or_none()


def _remote_or_none() -> fdtd_gpu_remote.RemoteConfig | None:
    """The configured GPU host when it answers, else None.

    A CUDA card is not the only way to reach one. ``fdtd_gpu_remote`` already
    ships a job to a machine that has one, and that is the path the animation
    renders take, so the same parity can be proved on a workstation with no
    GPU of its own: what is under test is the engine, not the local hardware.
    """
    fdtd_gpu_remote.load_env()
    config = fdtd_gpu_remote.RemoteConfig.from_env()
    if not config.host or not fdtd_gpu_remote.remote_available(config):
        return None
    return config


_REMOTE = _remote_or_none()

#: Skip mark for the cases that need the remote GPU.
_needs_remote = pytest.mark.skipif(
    _REMOTE is None,
    reason="no PHONO_GPU_HOST answers (set it in .env to run the GPU parity here)",
)

BACKENDS = [
    pytest.param(np, 1e-12, id="numpy"),
    pytest.param(
        _CUPY,
        1e-10,
        id="cupy",
        marks=pytest.mark.skipif(
            _CUPY is None, reason="CuPy with a CUDA device is not available"
        ),
    ),
]

_NY, _NX = 60, 80
_DX = 0.01
_STEPS = 200

#: Relative to the run peak. The GPU reassociates the update sums, so the two
#: engines agree to ULP scale rather than bit for bit, as the local `cupy`
#: parameter above already allows.
_GPU_TOL = 1e-10

#: One remote job is an SSH round trip plus a Docker start, measured at about
#: six seconds for these grids. The ceiling is generous enough that a cold
#: image pull does not turn a slow run into a red one.
_REMOTE_TIMEOUT_S = 180.0


def _scenarios() -> dict[str, dict[str, Any]]:
    """Constructor kwargs of every parity scenario (library names).

    An optional ``"c"`` entry replaces the default scalar sound speed.
    """
    rho_obstacle = np.full((_NY, _NX), 1.2)
    rho_obstacle[24:36, 30:44] = 6000.0  # dense block scatterer
    damping_map = np.zeros((_NY, _NX))
    damping_map[:, _NX - 16 :] = 35.0  # lossy slab at the right
    mask = np.zeros((_NY, _NX), dtype=np.bool_)
    mask[40:46, 10:70] = True  # rigid slat obstacle
    c_hetero = np.full((_NY, _NX), 343.0)  # air over a water layer
    c_hetero[_NY // 2 :, :] = 1500.0
    rho_hetero = np.full((_NY, _NX), 1.2)
    rho_hetero[_NY // 2 :, :] = 1000.0
    return {
        "rigid_box": {},
        "dense_rho_obstacle": {"rho": rho_obstacle},
        "heterogeneous_c": {"c": c_hetero, "rho": rho_hetero},
        "sponge": {
            "sponge_width": 12,
            "sponge_sides": ("left", "right", "bottom"),
            "sponge_reflection": 1e-3,
        },
        "sponge_side_string": {"sponge_width": 12, "sponge_sides": "top"},
        "impedance_edges": {
            "edge_impedance": {
                "top": 413.0,
                "left": np.linspace(200.0, 800.0, _NY),
            }
        },
        "impedance_right_bottom": {
            "edge_impedance": {
                "right": 620.0,
                "bottom": np.linspace(300.0, 900.0, _NX),
            }
        },
        "damping_scalar": {"damping": 25.0},
        "damping_map": {"damping": damping_map},
        "obstacle_mask": {"obstacle_mask": mask},
        "combined": {
            "rho": rho_obstacle,
            "sponge_width": 10,
            "sponge_sides": ("left", "right"),
            "damping": damping_map,
            "edge_impedance": {"top": 413.0},
        },
    }


_WAVES: dict[str, dict[str, Any]] = {
    "down": {"center": 0.12, "width": 0.05},
    "up": {"center": 0.45, "width": 0.05, "wavelength": 0.08},
    "left": {"center": 0.60, "width": 0.06, "amplitude": 0.7},
    "right": {"center": 0.20, "width": 0.05, "wavelength": 0.10},
}


def _split_c(kwargs: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    """Pop the optional ``"c"`` entry (default scalar 343) from *kwargs*."""
    rest = dict(kwargs)
    return rest.pop("c", 343.0), rest


def _reference(
    kwargs: dict[str, Any], wave: dict[str, Any], direction: str
) -> list[np.ndarray]:
    """Library-engine pressure fields every 50 steps."""
    c, rest = _split_c(kwargs)
    ref = FDTD2D(c, _DX, shape=(_NY, _NX), **rest)
    ref.add_plane_wave(direction, **wave)
    fields = []
    for i in range(_STEPS):
        ref.step()
        if (i + 1) % 50 == 0:
            fields.append(ref.p.copy())
    return fields


def _assert_parity(
    fields: list[np.ndarray], got: list[np.ndarray], rtol_peak: float
) -> None:
    """Every recorded field matches within *rtol_peak* of the run peak."""
    peak = max(float(np.max(np.abs(f))) for f in fields)
    assert peak > 0.0
    for ref_f, got_f in zip(fields, got, strict=True):
        np.testing.assert_allclose(got_f, ref_f, rtol=0.0, atol=rtol_peak * peak)


@pytest.mark.parametrize(("xp", "tol"), BACKENDS)
@pytest.mark.parametrize("direction", list(_WAVES))
def test_plane_wave_directions_match_library(
    xp: ModuleType, tol: float, direction: str
) -> None:
    """A plane packet in each travel direction reproduces the library."""
    wave = _WAVES[direction]
    fields = _reference({}, wave, direction)
    sim = fdtd_gpu.GpuFDTD2D(343.0, _DX, shape=(_NY, _NX), xp=xp)
    sim.add_plane_wave(direction, **wave)
    got = []
    for i in range(_STEPS):
        sim.step()
        if (i + 1) % 50 == 0:
            got.append(sim.pressure())
    _assert_parity(fields, got, tol)
    assert sim.n == _STEPS
    assert sim.dt == pytest.approx(0.6 * _DX / (343.0 * np.sqrt(2.0)))
    assert sim.time == pytest.approx(_STEPS * sim.dt)


@pytest.mark.parametrize(("xp", "tol"), BACKENDS)
@pytest.mark.parametrize("scenario", list(_scenarios()))
def test_scenarios_match_library(xp: ModuleType, tol: float, scenario: str) -> None:
    """Each engine feature (and their combination) reproduces the library."""
    kwargs = _scenarios()[scenario]
    wave = {"center": 0.15, "width": 0.06, "wavelength": 0.09}
    fields = _reference(kwargs, wave, "down")
    c, rest = _split_c(kwargs)
    sim = fdtd_gpu.GpuFDTD2D(c, _DX, shape=(_NY, _NX), xp=xp, **rest)
    sim.add_plane_wave("down", **wave)
    got = []
    for i in range(_STEPS):
        sim.step()
        if (i + 1) % 50 == 0:
            got.append(sim.pressure())
    _assert_parity(fields, got, tol)


def test_numpy_path_is_bit_identical() -> None:
    """With xp=numpy the replica is exactly the library, bit for bit."""
    kwargs = _scenarios()["combined"]
    wave = {"center": 0.15, "width": 0.06, "wavelength": 0.09}
    ref = FDTD2D(343.0, _DX, shape=(_NY, _NX), **kwargs)
    ref.add_plane_wave("down", **wave)
    sim = fdtd_gpu.GpuFDTD2D(343.0, _DX, shape=(_NY, _NX), xp=np, **kwargs)
    sim.add_plane_wave("down", **wave)
    for _ in range(_STEPS):
        ref.step()
        sim.step()
    assert np.array_equal(sim.p, ref.p)


def test_job_round_trip_matches_library() -> None:
    """build_job -> job_runner.run_job reproduces the library fields."""
    kwargs = _scenarios()["combined"]
    wave = {"center": 0.15, "width": 0.06, "wavelength": 0.09}
    fields = _reference(kwargs, wave, "down")
    job = fdtd_gpu_remote.build_job(
        343.0,
        _DX,
        shape=(_NY, _NX),
        steps=_STEPS,
        sample_steps=[50, 100, 150, 200],
        plane_waves=[{"direction": "down", **wave}],
        **kwargs,
    )
    result = job_runner.run_job(job)
    assert result["backend"] in ("numpy", "cupy")
    assert int(result["cells"]) == _NY * _NX
    tol = 1e-12 if result["backend"] == "numpy" else 1e-10
    _assert_parity(fields, list(np.asarray(result["frames"])), tol)


def test_job_round_trip_edge_array_and_obstacle() -> None:
    """Per-cell 1D edge impedance and a non-trivial obstacle survive the npz."""
    mask = np.zeros((_NY, _NX), dtype=np.bool_)
    mask[18:22, 8:36] = True  # horizontal slat
    mask[30:52, 44:48] = True  # vertical baffle
    kwargs = {
        "edge_impedance": {"left": np.linspace(220.0, 760.0, _NY)},
        "obstacle_mask": mask,
    }
    wave = {"center": 0.15, "width": 0.06, "wavelength": 0.09}
    fields = _reference(kwargs, wave, "down")
    job = fdtd_gpu_remote.build_job(
        343.0,
        _DX,
        shape=(_NY, _NX),
        steps=_STEPS,
        sample_steps=[50, 100, 150, 200],
        plane_waves=[{"direction": "down", **wave}],
        **kwargs,
    )
    result = job_runner.run_job(job)
    tol = 1e-12 if result["backend"] == "numpy" else 1e-10
    _assert_parity(fields, list(np.asarray(result["frames"])), tol)


def test_job_round_trip_empty_sponge_sides() -> None:
    """An explicit sponge_sides=() with sponge_width > 0 stays empty.

    The library treats the empty tuple as "no sponge anywhere" even when a
    width is given; the job archive must not turn it back into the
    all-four-sides default.
    """
    kwargs: dict[str, Any] = {"sponge_width": 12, "sponge_sides": ()}
    wave = {"center": 0.15, "width": 0.06, "wavelength": 0.09}
    fields = _reference(kwargs, wave, "down")
    job = fdtd_gpu_remote.build_job(
        343.0,
        _DX,
        shape=(_NY, _NX),
        steps=_STEPS,
        sample_steps=[50, 100, 150, 200],
        plane_waves=[{"direction": "down", **wave}],
        **kwargs,
    )
    assert job["sponge_sides"].size == 0
    result = job_runner.run_job(job)
    tol = 1e-12 if result["backend"] == "numpy" else 1e-10
    _assert_parity(fields, list(np.asarray(result["frames"])), tol)


def test_job_round_trip_init_scale_x() -> None:
    """A cosine taper on the initial front matches the manual scaling."""
    window = 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(_NX) / (_NX - 1)))
    wave = {"center": 0.15, "width": 0.06, "wavelength": 0.09}
    sim = fdtd_gpu.GpuFDTD2D(343.0, _DX, shape=(_NY, _NX), xp=np)
    sim.add_plane_wave("down", **wave)
    sim.p *= window[np.newaxis, :]
    sim.vy *= window[np.newaxis, :]
    fields = []
    for i in range(_STEPS):
        sim.step()
        if (i + 1) % 50 == 0:
            fields.append(sim.pressure())
    job = fdtd_gpu_remote.build_job(
        343.0,
        _DX,
        shape=(_NY, _NX),
        steps=_STEPS,
        sample_steps=[50, 100, 150, 200],
        plane_waves=[{"direction": "down", **wave}],
        init_scale_x=window,
    )
    result = job_runner.run_job(job)
    tol = 1e-12 if result["backend"] == "numpy" else 1e-10
    _assert_parity(fields, list(np.asarray(result["frames"])), tol)


def test_job_round_trip_subsampled_frames() -> None:
    """sample_stride=4 + float32 equals the full frames subsampled locally."""
    kwargs = _scenarios()["combined"]
    wave = {"center": 0.15, "width": 0.06, "wavelength": 0.09}
    common: dict[str, Any] = dict(
        shape=(_NY, _NX),
        steps=_STEPS,
        sample_steps=[0, 100, 200],
        plane_waves=[{"direction": "down", **wave}],
        **kwargs,
    )
    full = job_runner.run_job(fdtd_gpu_remote.build_job(343.0, _DX, **common))
    sub = job_runner.run_job(
        fdtd_gpu_remote.build_job(
            343.0, _DX, sample_stride=4, sample_dtype="float32", **common
        )
    )
    full_frames = np.asarray(full["frames"])
    sub_frames = np.asarray(sub["frames"])
    assert full_frames.dtype == np.float64
    assert sub_frames.dtype == np.float32
    assert sub_frames.shape == (3, (_NY + 3) // 4, (_NX + 3) // 4)
    assert sub_frames.nbytes * 32 == full_frames.nbytes
    np.testing.assert_array_equal(
        sub_frames, full_frames[:, ::4, ::4].astype(np.float32)
    )


def test_build_job_validation() -> None:
    """build_job rejects inconsistent sampling and non-integer widths."""
    ok: dict[str, Any] = {"shape": (_NY, _NX), "steps": 100, "sample_steps": [50]}
    with pytest.raises(ValueError, match=r"cfl must lie in \(0, 1\)"):
        fdtd_gpu_remote.build_job(343.0, _DX, cfl=1.5, **ok)
    with pytest.raises(ValueError, match=r"damping must be non-negative and finite"):
        fdtd_gpu_remote.build_job(343.0, _DX, damping=-1.0, **ok)
    bad_damping = np.zeros((_NY + 1, _NX))
    with pytest.raises(
        ValueError, match=r"damping map shape .* does not match the grid"
    ):
        fdtd_gpu_remote.build_job(343.0, _DX, damping=bad_damping, **ok)
    with pytest.raises(ValueError, match=r"unknown impedance sides"):
        fdtd_gpu_remote.build_job(343.0, _DX, edge_impedance={"front": 413.0}, **ok)
    with pytest.raises(
        ValueError,
        match=r"side 'top' cannot be both absorbing and an impedance boundary",
    ):
        fdtd_gpu_remote.build_job(
            343.0, _DX, sponge_width=10, edge_impedance={"top": 413.0}, **ok
        )
    bad_profile = {"top": np.ones(_NX + 2)}
    with pytest.raises(
        ValueError, match=r"impedance for side 'top' must be a scalar or a 1D array"
    ):
        fdtd_gpu_remote.build_job(343.0, _DX, edge_impedance=bad_profile, **ok)
    bad_mask_shape = np.zeros((_NY + 1, _NX), np.bool_)
    with pytest.raises(ValueError, match=r"obstacle_mask must match the grid shape"):
        fdtd_gpu_remote.build_job(343.0, _DX, obstacle_mask=bad_mask_shape, **ok)
    bad_mask_dtype = np.zeros((_NY, _NX), np.int64)
    with pytest.raises(ValueError, match=r"obstacle_mask must be a boolean array"):
        fdtd_gpu_remote.build_job(343.0, _DX, obstacle_mask=bad_mask_dtype, **ok)
    with pytest.raises(ValueError, match=r"sample_steps must lie within \[0, steps\]"):
        fdtd_gpu_remote.build_job(
            343.0, _DX, shape=(_NY, _NX), steps=100, sample_steps=[50, 150]
        )
    with pytest.raises(ValueError, match=r"sample_steps must lie within \[0, steps\]"):
        fdtd_gpu_remote.build_job(
            343.0, _DX, shape=(_NY, _NX), steps=100, sample_steps=[-1, 50]
        )
    with pytest.raises(ValueError, match=r"sample_stride must be >= 1"):
        fdtd_gpu_remote.build_job(343.0, _DX, sample_stride=0, **ok)
    with pytest.raises(ValueError, match=r"sample_stride must be an integer"):
        fdtd_gpu_remote.build_job(343.0, _DX, sample_stride=2.0, **ok)
    with pytest.raises(ValueError, match=r"sample_dtype must be"):
        fdtd_gpu_remote.build_job(343.0, _DX, sample_dtype="int16", **ok)
    with pytest.raises(ValueError, match=r"sponge_width must be an integer"):
        fdtd_gpu_remote.build_job(343.0, _DX, sponge_width=True, **ok)
    with pytest.raises(ValueError, match=r"sponge_width must be an integer"):
        fdtd_gpu_remote.build_job(343.0, _DX, sponge_width=3.5, **ok)
    with pytest.raises(ValueError, match=r"sponge_width must be an integer"):
        fdtd_gpu.GpuFDTD2D(343.0, _DX, shape=(_NY, _NX), sponge_width=2.5)
    too_wide = min(_NY, _NX)
    with pytest.raises(ValueError, match="smallest grid side"):
        fdtd_gpu_remote.build_job(343.0, _DX, sponge_width=too_wide, **ok)
    with pytest.raises(
        ValueError, match=r"sponge_reflection must lie strictly between"
    ):
        fdtd_gpu_remote.build_job(
            343.0, _DX, sponge_width=4, sponge_reflection=1.5, **ok
        )
    bad_scale_length = np.ones(_NX + 1)
    with pytest.raises(ValueError, match=r"init_scale_x must be a 1D array of length"):
        fdtd_gpu_remote.build_job(343.0, _DX, init_scale_x=bad_scale_length, **ok)
    bad_scale_nan = np.full(_NX, np.nan)
    with pytest.raises(ValueError, match=r"init_scale_x must be finite everywhere"):
        fdtd_gpu_remote.build_job(343.0, _DX, init_scale_x=bad_scale_nan, **ok)


def test_sample_steps_deduplicated_local_and_remote_contract() -> None:
    """Repeated sample_steps collapse to one frame per unique step.

    build_job stores the unique ascending schedule and run_job dedups on
    its own too, so a hand-built archive with duplicates yields the same
    frames/sample_steps contract as the packed job.
    """
    job = fdtd_gpu_remote.build_job(
        343.0, _DX, shape=(_NY, _NX), steps=100, sample_steps=[100, 0, 50, 50, 0, 100]
    )
    np.testing.assert_array_equal(job["sample_steps"], [0, 50, 100])
    job["sample_steps"] = np.asarray([100, 0, 50, 50, 0, 100], dtype=np.int64)
    result = job_runner.run_job(job)
    np.testing.assert_array_equal(result["sample_steps"], [0, 50, 100])
    assert np.asarray(result["frames"]).shape[0] == 3


def test_submit_falls_back_to_local_numpy_run(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A reachable remote whose run fails degrades to a local NumPy run."""
    config = fdtd_gpu_remote.RemoteConfig(
        host="203.0.113.9",
        user="gpuuser",
        name="test GPU",
        image="cupy/cupy:v13.6.0",
        workdir="/tmp/phonometry-gpu",
    )
    monkeypatch.setattr(fdtd_gpu_remote, "remote_available", lambda config: True)

    def _boom(
        job: dict[str, Any],
        config: fdtd_gpu_remote.RemoteConfig,
        timeout: float = 0.0,
    ) -> dict[str, Any]:
        msg = "docker run failed"
        raise fdtd_gpu_remote.RemoteRunError(msg)

    monkeypatch.setattr(fdtd_gpu_remote, "run_remote", _boom)
    monkeypatch.setattr(job_runner, "_pick_backend", lambda: (np, "numpy"))
    job = fdtd_gpu_remote.build_job(
        343.0, _DX, shape=(20, 24), steps=10, sample_steps=[0, 10]
    )
    result = fdtd_gpu_remote.submit(job, config)
    assert result["backend"] == "numpy"
    assert int(result["steps"]) == 10
    assert np.asarray(result["frames"]).shape == (2, 20, 24)
    err = capsys.readouterr().err
    assert "docker run failed" in err
    assert "falling back to a local NumPy run" in err


def test_load_env_real_environment_wins(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """An exported variable beats the .env file; missing ones are loaded."""
    env_file = tmp_path / ".env"
    env_file.write_text(
        '# comment line\nPHONO_GPU_HOST=file-host\nPHONO_GPU_NAME="lab GPU"\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("PHONO_GPU_HOST", "env-host")
    # Guarantee the variable is absent and restored either way.
    monkeypatch.setenv("PHONO_GPU_NAME", "sentinel")
    monkeypatch.delenv("PHONO_GPU_NAME")
    values = fdtd_gpu_remote.load_env(env_file)
    assert values == {"PHONO_GPU_HOST": "file-host", "PHONO_GPU_NAME": "lab GPU"}
    assert os.environ["PHONO_GPU_HOST"] == "env-host"  # real env wins
    assert os.environ["PHONO_GPU_NAME"] == "lab GPU"  # quotes stripped


def _remote_frames(
    kwargs: dict[str, Any], wave: dict[str, Any], direction: str
) -> list[np.ndarray]:
    """The same scenario, stepped on the remote GPU, frame by frame.

    ``run_remote`` rather than ``submit``: submitting falls back to a local
    NumPy run when anything goes wrong, which for a render is the right call
    and for a test is the wrong one. A silent fallback here would compare
    NumPy against NumPy and pass while proving nothing about the GPU.
    """
    c, rest = _split_c(kwargs)
    job = fdtd_gpu_remote.build_job(
        c,
        _DX,
        shape=(_NY, _NX),
        steps=_STEPS,
        sample_steps=[50, 100, 150, 200],
        plane_waves=[{"direction": direction, **wave}],
        **rest,
    )
    result = fdtd_gpu_remote.run_remote(job, _REMOTE, timeout=_REMOTE_TIMEOUT_S)
    # The backend the job actually ran on is the whole point of the exercise:
    # the runner picks NumPy when CuPy is missing on the far side, and that
    # result would satisfy every tolerance below without touching a GPU.
    assert result["backend"] == "cupy"
    assert int(result["cells"]) == _NY * _NX
    return list(np.asarray(result["frames"]))


@_needs_remote
@pytest.mark.parametrize("direction", list(_WAVES))
def test_plane_wave_directions_match_library_on_the_remote_gpu(direction: str) -> None:
    """A plane packet in each direction reproduces the library on the GPU."""
    wave = _WAVES[direction]
    fields = _reference({}, wave, direction)
    _assert_parity(fields, _remote_frames({}, wave, direction), _GPU_TOL)


@_needs_remote
@pytest.mark.parametrize("scenario", list(_scenarios()))
def test_scenarios_match_library_on_the_remote_gpu(scenario: str) -> None:
    """Each engine feature reproduces the library on the GPU."""
    kwargs = _scenarios()[scenario]
    wave = {"center": 0.15, "width": 0.06, "wavelength": 0.09}
    fields = _reference(kwargs, wave, "down")
    _assert_parity(fields, _remote_frames(kwargs, wave, "down"), _GPU_TOL)
