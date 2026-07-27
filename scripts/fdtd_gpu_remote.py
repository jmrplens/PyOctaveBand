#  Copyright (c) 2026. Jose M. Requena-Plens
"""Remote GPU runner for the animation FDTD fields.

Offloads a 2D acoustic FDTD run (the ``fdtd_gpu.GpuFDTD2D`` engine) to a
CUDA machine over SSH and Docker, falling back to a local NumPy run when no
remote is configured or reachable. The flow is file-based and stateless:

1. :func:`build_job` packs the domain (c/rho maps, sponges, impedance
   edges, damping, obstacles, plane-wave initial conditions) plus the step
   count and the frame sampling schedule into a single ``.npz`` archive.
2. :func:`submit` copies ``fdtd_gpu.py``, ``job_runner.py`` and the job
   archive into a fresh per-job directory under the remote work directory,
   executes ``docker run --rm --gpus all -v <jobdir>:/work <image>
   python3 /work/job_runner.py`` over SSH, copies the resulting frames
   archive back and removes the remote job directory.
3. The result dict carries the sampled pressure frames, the backend that
   actually ran (``"cupy"`` or ``"numpy"``) and the device-synchronised
   stepping time, so callers can report throughput honestly.

Configuration comes from environment variables, optionally loaded from the
repository ``.env`` (never committed; see ``.env.example``): PHONO_GPU_HOST,
PHONO_GPU_USER, PHONO_GPU_NAME, PHONO_GPU_DOCKER_IMAGE, PHONO_GPU_WORKDIR.
Real environment variables take precedence over the file, dotenv-style; the
parser is self-contained so python-dotenv is not a dependency.

If the remote host does not answer (or the transfer/run fails), the runner
prints a clear notice and finishes the job locally instead of raising, so
an unplugged GPU box never breaks a render.

``--benchmark`` runs a medium representative case (default 1200 x 1600
cells, 500 steps: heterogeneous rho with a dense block, three sponge sides,
one impedance edge, a damping map and a sine-carrier plane wave) once
locally and once remotely, and reports cell updates per second for both.
Verified end to end against a Docker + CuPy host: the committed default
image ``cupy/cupy:v13.6.0`` runs as-is on an NVIDIA driver 570 machine.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import fdtd_gpu
import job_runner

_REPO_ROOT = _SCRIPTS.parent
_ENV_FILE = _REPO_ROOT / ".env"

_CONNECT_TIMEOUT_S = 8
_DEFAULT_JOB_TIMEOUT_S = 900.0


def load_env(path: Path = _ENV_FILE) -> dict[str, str]:
    """Read ``KEY=VALUE`` lines from *path* into the process environment.

    Comments (``#``) and blank lines are skipped, surrounding quotes are
    stripped, and variables already present in ``os.environ`` are left
    untouched (real environment wins, as python-dotenv does). Returns the
    mapping that was read (before precedence), for inspection.
    """
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'\"")
        if key:
            values[key] = value
            os.environ.setdefault(key, value)
    return values


@dataclass(frozen=True)
class RemoteConfig:
    """Connection settings of the remote GPU box.

    :ivar host: SSH host name or address; empty disables the remote path.
    :ivar user: SSH user.
    :ivar name: Human label of the machine/GPU, used only in messages.
    :ivar image: Docker image with CUDA-enabled CuPy and Python 3.
    :ivar workdir: Remote directory where per-job directories are created.
    """

    host: str
    user: str
    name: str
    image: str
    workdir: str

    @classmethod
    def from_env(cls) -> RemoteConfig:
        """Build the config from PHONO_GPU_* (after :func:`load_env`)."""
        return cls(
            host=os.environ.get("PHONO_GPU_HOST", ""),
            user=os.environ.get("PHONO_GPU_USER", "root"),
            name=os.environ.get("PHONO_GPU_NAME", "remote GPU"),
            image=os.environ.get("PHONO_GPU_DOCKER_IMAGE",
                                 "cupy/cupy:v13.6.0"),
            workdir=os.environ.get("PHONO_GPU_WORKDIR",
                                   "/tmp/phonometry-gpu"),
        )

    @property
    def target(self) -> str:
        """The ``user@host`` SSH target."""
        return f"{self.user}@{self.host}"


class RemoteRunError(RuntimeError):
    """A remote stage (ssh, scp or docker) failed or timed out."""


def build_job(
    c: float | NDArray[np.float64],
    dx: float,
    *,
    steps: int,
    sample_steps: list[int] | NDArray[np.int_],
    shape: tuple[int, int] | None = None,
    rho: float | NDArray[np.float64] = 1.2,
    cfl: float = 0.6,
    sponge_width: int = 0,
    sponge_sides: Any = None,
    sponge_reflection: float = 1e-4,
    damping: float | NDArray[np.float64] = 0.0,
    edge_impedance: dict[str, float | NDArray[np.float64]] | None = None,
    obstacle_mask: NDArray[np.bool_] | None = None,
    plane_waves: list[dict[str, Any]] | None = None,
    init_scale_x: NDArray[np.float64] | None = None,
    sample_stride: int = 1,
    sample_dtype: str = "float64",
) -> dict[str, Any]:
    """Pack one simulation into the flat job payload of ``job_runner``.

    Parameters mirror :class:`fdtd_gpu.GpuFDTD2D` (same names, same
    semantics); ``plane_waves`` is a list of keyword dicts for
    ``add_plane_wave`` (``direction``, ``center``, ``width`` and optional
    ``amplitude``/``wavelength``). ``sample_steps`` lists the step counts
    at which a pressure frame is recorded (0 = the initial state); every
    entry must lie in ``[0, steps]``. ``init_scale_x`` is an optional 1D
    window of length ``nx`` applied after the plane waves are laid down
    (``p *= w`` and ``vy *= w`` column-wise, ``vx`` untouched), giving the
    initial front a lateral taper that leaves the sponges alone.
    ``sample_stride`` subsamples each
    recorded frame spatially (``frame[::stride, ::stride]``, applied on
    the compute device before the transfer) and ``sample_dtype``
    (``"float64"`` or ``"float32"``) is the dtype the frames are cast to,
    also on the device, so a coarse animation grid ships fewer bytes.
    """
    c_map = fdtd_gpu._resolve_c_map(c, shape)
    ny, nx = c_map.shape
    rho_map = fdtd_gpu._resolve_rho_map(rho, ny, nx)
    steps = fdtd_gpu._integer("steps", steps)
    sponge_width = fdtd_gpu._integer("sponge_width", sponge_width)
    sample_stride = fdtd_gpu._integer("sample_stride", sample_stride)
    if sample_stride < 1:
        raise ValueError("sample_stride must be >= 1")
    if sample_dtype not in ("float64", "float32"):
        raise ValueError("sample_dtype must be 'float64' or 'float32'")
    sample_arr = np.asarray(sample_steps, dtype=np.int64)
    if sample_arr.size and (int(sample_arr.min()) < 0
                            or int(sample_arr.max()) > steps):
        raise ValueError("sample_steps must lie within [0, steps]")
    # None is the "not given" sentinel (all four sides when a sponge is
    # active, as the library); an explicit iterable, including the empty
    # tuple, is resolved and stored literally.
    if sponge_sides is None:
        sides: tuple[str, ...] = (fdtd_gpu._resolve_sponge_sides(None)
                                  if sponge_width > 0 else ())
    else:
        sides = fdtd_gpu._resolve_sponge_sides(sponge_sides)
    edge_impedance = edge_impedance or {}
    obstacle = (np.zeros((ny, nx), dtype=np.bool_) if obstacle_mask is None
                else np.asarray(obstacle_mask, dtype=np.bool_))
    job: dict[str, Any] = {
        "c": c_map,
        "rho": rho_map,
        "dx": np.float64(dx),
        "cfl": np.float64(cfl),
        "sponge_width": np.int64(sponge_width),
        "sponge_sides": np.asarray(sides, dtype=np.str_),
        "sponge_reflection": np.float64(sponge_reflection),
        "damping": np.asarray(damping, dtype=np.float64),
        "edge_sides": np.asarray(sorted(edge_impedance), dtype=np.str_),
        "obstacle": obstacle,
        "plane_waves": np.str_(json.dumps(plane_waves or [])),
        "steps": np.int64(steps),
        "sample_steps": sample_arr,
        "sample_stride": np.int64(sample_stride),
        "sample_dtype": np.str_(sample_dtype),
    }
    for side, z in edge_impedance.items():
        job[f"edge_z_{side}"] = np.asarray(z, dtype=np.float64)
    if init_scale_x is not None:
        w = np.asarray(init_scale_x, dtype=np.float64)
        if w.ndim != 1 or w.shape[0] != nx:
            raise ValueError(
                f"init_scale_x must be a 1D array of length nx = {nx}")
        if not np.all(np.isfinite(w)):
            raise ValueError("init_scale_x must be finite everywhere")
        job["init_scale_x"] = w
    return job


def _ssh(config: RemoteConfig, command: str,
         timeout: float) -> subprocess.CompletedProcess[str]:
    """Run one remote command over SSH (BatchMode, no prompts)."""
    return subprocess.run(
        ["ssh", "-o", "BatchMode=yes",
         "-o", f"ConnectTimeout={_CONNECT_TIMEOUT_S}",
         "--", config.target, command],
        capture_output=True, text=True, timeout=timeout, check=False,
    )


def _scp(sources: list[str], dest: str, timeout: float) -> None:
    """Copy files with scp; raise :class:`RemoteRunError` on failure."""
    proc = subprocess.run(
        ["scp", "-q", "-o", "BatchMode=yes",
         "-o", f"ConnectTimeout={_CONNECT_TIMEOUT_S}", "--", *sources, dest],
        capture_output=True, text=True, timeout=timeout, check=False,
    )
    if proc.returncode != 0:
        raise RemoteRunError(f"scp failed: {proc.stderr.strip()}")


def remote_available(config: RemoteConfig) -> bool:
    """True when the configured host answers an SSH no-op."""
    if not config.host:
        return False
    try:
        proc = _ssh(config, "true", timeout=_CONNECT_TIMEOUT_S + 4)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return proc.returncode == 0


def run_remote(job: dict[str, Any], config: RemoteConfig,
               timeout: float = _DEFAULT_JOB_TIMEOUT_S) -> dict[str, Any]:
    """Execute one job on the remote GPU and return the result payload.

    Creates a unique job directory under ``config.workdir``, ships the
    engine, the runner and the job archive, runs the container and fetches
    the frames back. The remote directory is removed afterwards, also on
    failure. Raises :class:`RemoteRunError` on any remote-stage failure
    (transport-level ``OSError``/``subprocess.TimeoutExpired`` from the
    scp/ssh steps propagate as-is); use :func:`submit` for the version
    with the local fallback.
    """
    job_dir = f"{config.workdir.rstrip('/')}/job-{int(time.time())}-{os.getpid()}"
    q_dir = shlex.quote(job_dir)
    with tempfile.TemporaryDirectory(prefix="phono-gpu-") as tmp:
        job_path = Path(tmp) / "job.npz"
        np.savez_compressed(job_path, **job)
        mkdir = _ssh(config, f"mkdir -p {q_dir}", timeout=30.0)
        if mkdir.returncode != 0:
            raise RemoteRunError(
                f"cannot create {job_dir} on {config.target}: "
                f"{mkdir.stderr.strip()}")
        try:
            _scp([str(_SCRIPTS / "fdtd_gpu.py"),
                  str(_SCRIPTS / "job_runner.py"),
                  str(job_path)],
                 f"{config.target}:{q_dir}/", timeout=120.0)
            docker_cmd = (
                f"docker run --rm --gpus all -v {q_dir}:/work "
                f"{shlex.quote(config.image)} "
                "python3 /work/job_runner.py /work/job.npz /work/frames.npz"
            )
            try:
                run = _ssh(config, docker_cmd, timeout=timeout)
            except subprocess.TimeoutExpired as exc:
                raise RemoteRunError(
                    f"remote job exceeded {timeout:.0f} s") from exc
            if run.returncode != 0:
                raise RemoteRunError(
                    f"remote docker run failed: {run.stderr.strip()[-800:]}")
            frames_path = Path(tmp) / "frames.npz"
            _scp([f"{config.target}:{q_dir}/frames.npz"],
                 str(frames_path), timeout=300.0)
            with np.load(frames_path, allow_pickle=False) as data:
                return {key: (data[key].copy() if data[key].ndim
                              else data[key][()])
                        for key in data.files}
        finally:
            try:
                _ssh(config, f"rm -rf {q_dir}", timeout=30.0)
            except (OSError, subprocess.TimeoutExpired):
                pass                             # best-effort cleanup


def submit(job: dict[str, Any], config: RemoteConfig | None = None,
           timeout: float = _DEFAULT_JOB_TIMEOUT_S) -> dict[str, Any]:
    """Run one job remotely when possible, locally (NumPy) otherwise.

    The local fallback is a message, not an exception: a missing .env, an
    unreachable host or any remote failure degrades to the same result
    computed on the CPU.
    """
    if config is None:
        load_env()
        config = RemoteConfig.from_env()
    if config.host:
        if remote_available(config):
            try:
                return run_remote(job, config, timeout=timeout)
            except (RemoteRunError, subprocess.TimeoutExpired,
                    OSError) as exc:
                print(f"[fdtd-gpu] remote run on {config.name} "
                      f"({config.target}) failed: {exc}", file=sys.stderr)
                print("[fdtd-gpu] falling back to a local NumPy run",
                      file=sys.stderr)
        else:
            print(f"[fdtd-gpu] {config.name} ({config.target}) does not "
                  "answer; running locally on NumPy", file=sys.stderr)
    else:
        print("[fdtd-gpu] no PHONO_GPU_HOST configured; running locally "
              "on NumPy", file=sys.stderr)
    return job_runner.run_job(job)


def _benchmark_job(ny: int, nx: int, steps: int) -> dict[str, Any]:
    """A representative medium case exercising every engine feature."""
    dx = 0.01
    c_map = np.full((ny, nx), 343.0)
    rho_map = np.full((ny, nx), 1.2)
    # Dense rigid-like block (a 1000:1 density contrast scatterer).
    rho_map[ny // 2:ny // 2 + ny // 10, nx // 3:nx // 3 + nx // 5] = 1200.0
    damping = np.zeros((ny, nx))
    damping[:, -nx // 8:] = 40.0                 # lossy sample at the right
    return build_job(
        c_map, dx,
        rho=rho_map,
        sponge_width=40,
        sponge_sides=("left", "right", "bottom"),
        damping=damping,
        edge_impedance={"top": 413.0},
        plane_waves=[{"direction": "down", "center": (ny // 4) * dx,
                      "width": (ny // 12) * dx, "wavelength": 30 * dx}],
        steps=steps,
        sample_steps=[0, steps // 2, steps],
    )


def _report(label: str, result: dict[str, Any], wall: float) -> float:
    """Print one benchmark line; return the stepping cells/s."""
    cells = int(result["cells"])
    steps = int(result["steps"])
    elapsed = float(result["elapsed"])
    rate = cells * steps / elapsed
    print(f"  {label:24s} backend={result['backend']!s:6s} "
          f"stepping={elapsed:7.2f} s  wall={wall:7.2f} s  "
          f"rate={rate / 1e6:8.1f} Mcells/s")
    return rate


def _run_benchmark(args: argparse.Namespace) -> int:
    """Local-vs-remote throughput comparison on the medium case."""
    load_env()
    config = RemoteConfig.from_env()
    ny, nx, steps = args.ny, args.nx, args.steps
    print(f"FDTD GPU benchmark: {ny} x {nx} cells, {steps} steps "
          f"({ny * nx * steps / 1e9:.1f} Gcell-updates)")
    job = _benchmark_job(ny, nx, steps)

    t0 = time.perf_counter()
    local = job_runner.run_job(job)
    local_rate = _report("local", local, time.perf_counter() - t0)

    if args.local_only:
        return 0
    if not config.host or not remote_available(config):
        print(f"  remote {config.name} ({config.target or 'unset'}) is not "
              "reachable; no GPU numbers")
        return 1
    t0 = time.perf_counter()
    try:
        remote = run_remote(job, config, timeout=args.timeout)
    except RemoteRunError as exc:
        print(f"  remote run failed: {exc}")
        return 1
    remote_rate = _report(f"remote ({config.name})", remote,
                          time.perf_counter() - t0)
    print(f"  speedup (stepping): {remote_rate / local_rate:.1f}x")
    ref = np.asarray(local["frames"][-1])
    got = np.asarray(remote["frames"][-1])
    peak = float(np.max(np.abs(ref))) or 1.0
    print(f"  max frame deviation: {np.max(np.abs(got - ref)) / peak:.2e} "
          "(relative to peak)")
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run animation FDTD fields on a remote GPU over "
                    "SSH + Docker, with a local NumPy fallback.")
    parser.add_argument("--benchmark", action="store_true",
                        help="time a medium case locally and remotely and "
                             "report cell updates per second")
    parser.add_argument("--ny", type=int, default=1200,
                        help="benchmark grid rows (default 1200)")
    parser.add_argument("--nx", type=int, default=1600,
                        help="benchmark grid columns (default 1600)")
    parser.add_argument("--steps", type=int, default=500,
                        help="benchmark time steps (default 500)")
    parser.add_argument("--local-only", action="store_true",
                        help="benchmark only the local NumPy path")
    parser.add_argument("--timeout", type=float,
                        default=_DEFAULT_JOB_TIMEOUT_S,
                        help="remote job timeout in seconds")
    args = parser.parse_args(argv)
    if args.benchmark:
        return _run_benchmark(args)
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
