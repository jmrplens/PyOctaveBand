#  Copyright (c) 2026. Jose M. Requena-Plens
"""Executable payload of the remote GPU FDTD runner.

Runs inside the CuPy container (or on any plain Python with NumPy): loads a
job archive produced by ``fdtd_gpu_remote.py``, builds the backend-agnostic
:class:`fdtd_gpu.GpuFDTD2D` engine on CuPy when a GPU is reachable (NumPy
otherwise), steps the simulation and writes the requested pressure frames
back to a result archive.

Usage: ``python3 job_runner.py <job.npz> <frames.npz>``.

Job archive keys (all produced by ``fdtd_gpu_remote.build_job``):

* ``c``, ``rho``: float64 ``(ny, nx)`` maps.
* ``dx``, ``cfl``, ``sponge_reflection``: scalars.
* ``sponge_width``: int scalar; ``sponge_sides``: array of side names.
* ``damping``: scalar or ``(ny, nx)`` map.
* ``edge_sides``: array of side names; per side an ``edge_z_<side>`` array.
* ``obstacle``: boolean ``(ny, nx)`` map (all-False when unused).
* ``plane_waves``: JSON string, list of ``add_plane_wave`` keyword dicts.
* ``init_scale_x`` (optional): 1D window of length ``nx``, multiplied into
  ``p`` and ``vy`` column-wise after the plane waves (``vx`` untouched).
* ``steps``: total leapfrog steps; ``sample_steps``: sorted step counts at
  which the pressure field is recorded (0 records the initial state).
* ``sample_stride``: spatial subsampling of the recorded frames
  (``frame[::stride, ::stride]``); ``sample_dtype``: dtype of the returned
  frames (``"float64"`` or ``"float32"``). Both are applied on the compute
  device before the host transfer, so they shrink the shipped archive.

Result archive keys: ``frames`` ``(n_frames, ceil(ny / stride),
ceil(nx / stride))`` in ``sample_dtype``, ``sample_steps``,
``sample_stride``, ``sample_dtype``, ``backend`` (``"cupy"`` or
``"numpy"``), ``elapsed`` (stepping wall time in seconds,
device-synchronised), ``steps`` and ``cells``.

The file is self-contained next to ``fdtd_gpu.py``; both are copied into
the container work directory, so no package installation is needed.
"""

from __future__ import annotations

import json
import sys
import time
from typing import Any

import fdtd_gpu
import numpy as np


def _pick_backend() -> tuple[Any, str]:
    """Return (array module, name): CuPy with a usable GPU, else NumPy."""
    try:
        import cupy

        cupy.cuda.runtime.getDeviceCount()       # raises without a GPU
        cupy.arange(1).sum()                     # force a real kernel launch
    except Exception:                            # noqa: BLE001 - any CUDA/import failure means CPU
        return np, "numpy"
    return cupy, "cupy"


def _build_engine(job: Any, xp: Any) -> fdtd_gpu.GpuFDTD2D:
    """Instantiate the engine from the job archive on backend *xp*."""
    damping = job["damping"]
    edge_impedance: dict[str, Any] = {
        str(side): job[f"edge_z_{side}"] for side in job["edge_sides"]
    }
    obstacle = np.asarray(job["obstacle"], dtype=np.bool_)
    sim = fdtd_gpu.GpuFDTD2D(
        np.asarray(job["c"], dtype=np.float64),
        float(job["dx"]),
        xp=xp,
        rho=np.asarray(job["rho"], dtype=np.float64),
        cfl=float(job["cfl"]),
        sponge_width=int(job["sponge_width"]),
        # build_job stores the fully resolved sponge sides, so an empty
        # array is the library's explicit "()" (no sponge sides), never
        # "not given": pass the tuple through and keep None strictly as
        # the never-stored "not given" sentinel.
        sponge_sides=tuple(str(s) for s in job["sponge_sides"]),
        sponge_reflection=float(job["sponge_reflection"]),
        damping=(float(damping) if damping.ndim == 0
                 else np.asarray(damping, dtype=np.float64)),
        edge_impedance=edge_impedance or None,
        obstacle_mask=obstacle if bool(obstacle.any()) else None,
    )
    for spec in json.loads(str(job["plane_waves"])):
        sim.add_plane_wave(
            spec["direction"],
            center=float(spec["center"]),
            width=float(spec["width"]),
            amplitude=float(spec.get("amplitude", 1.0)),
            wavelength=(None if spec.get("wavelength") is None
                        else float(spec["wavelength"])),
        )
    if "init_scale_x" in job:
        # Lateral taper of the initial front: column-wise window on the
        # pressure and the y-velocity (vx keeps its (ny, nx - 1) columns).
        w = xp.asarray(np.asarray(job["init_scale_x"],
                                  dtype=np.float64)[np.newaxis, :])
        sim.p *= w
        sim.vy *= w
    return sim


def _sample_frame(sim: fdtd_gpu.GpuFDTD2D, stride: int,
                  dtype: np.dtype[Any]) -> np.ndarray:
    """Subsample and cast the pressure on the device, then transfer it."""
    frame = sim.p[::stride, ::stride].astype(dtype, copy=True)
    if hasattr(frame, "get"):                    # CuPy device array
        return np.asarray(frame.get())
    return np.asarray(frame)


def run_job(job: Any) -> dict[str, Any]:
    """Run one job dict/archive and return the result payload."""
    xp, backend = _pick_backend()
    sim = _build_engine(job, xp)
    steps = int(job["steps"])
    stride = int(job["sample_stride"]) if "sample_stride" in job else 1
    if stride < 1:
        raise ValueError("sample_stride must be >= 1")
    dtype = np.dtype(str(job["sample_dtype"]) if "sample_dtype" in job
                     else "float64")
    sample_steps = sorted(int(s) for s in np.asarray(job["sample_steps"]))
    wanted = set(sample_steps)
    frames: list[np.ndarray] = []
    if 0 in wanted:
        frames.append(_sample_frame(sim, stride, dtype))
    if backend == "cupy":
        xp.cuda.Stream.null.synchronize()
    t0 = time.perf_counter()
    for i in range(steps):
        sim.step()
        if (i + 1) in wanted:
            frames.append(_sample_frame(sim, stride, dtype))
    if backend == "cupy":
        xp.cuda.Stream.null.synchronize()
    elapsed = time.perf_counter() - t0
    ny, nx = sim.p.shape
    return {
        "frames": (np.stack(frames) if frames
                   else np.zeros((0, 0, 0), dtype=dtype)),
        "sample_steps": np.asarray(sample_steps, dtype=np.int64),
        "sample_stride": stride,
        "sample_dtype": str(dtype),
        "backend": backend,
        "elapsed": elapsed,
        "steps": steps,
        "cells": ny * nx,
    }


def main(argv: list[str]) -> int:
    """CLI entry point: ``job_runner.py <job.npz> <frames.npz>``."""
    if len(argv) != 3:
        print("usage: job_runner.py <job.npz> <frames.npz>", file=sys.stderr)
        return 2
    with np.load(argv[1], allow_pickle=False) as job:
        result = run_job(job)
    np.savez_compressed(argv[2], **result)
    print(f"backend={result['backend']} steps={result['steps']} "
          f"cells={result['cells']} elapsed={result['elapsed']:.3f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
