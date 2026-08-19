#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Micro-benchmarks and golden-value capture for the performance hotspots.

Phase 0 of the 2026-07 overhaul (quality/performance plan). Three modes:

* default        -- time each hotspot case on fixed synthetic inputs and print
                    a table (best of N ``timeit`` repeats). Not wired into CI;
                    the tables are pasted into performance-PR descriptions.
* ``--golden``   -- recompute every case and rewrite ``tests/golden_data.py``
                    with the exact current outputs. Run ONLY when a numeric
                    change is intended and reviewed; the golden test suite
                    (``tests/test_golden_baseline.py``) guards refactors
                    (modularization, vectorization) against numeric drift.
* ``--figures``  -- run every ``generate_*`` figure function once (EN, light
                    theme) with per-figure wall-time logging and print the
                    slowest figures; the top-10 table is the Phase 0 baseline.

The input builders are the single source of truth for both the benchmarks and
the golden data; the golden tests import them from this module.
"""

from __future__ import annotations

import argparse
import sys
import time
import timeit
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:  # allow `python scripts/bench.py` from anywhere
    sys.path.insert(0, str(_ROOT))

_GOLDEN_PATH = _ROOT / "tests" / "golden_data.py"


# ---------------------------------------------------------------------------
# Fixed synthetic inputs (deterministic: no RNG, no clock)
# ---------------------------------------------------------------------------


def airport_inputs() -> dict[str, Any]:
    """A small departure scenario for the ECAC Doc 29 contour chain."""
    xs = np.linspace(0.0, 12000.0, 13)
    path = np.column_stack(
        [xs, np.zeros_like(xs), np.clip((xs - 1500.0) * 0.10, 0.0, 1800.0),
         np.where(xs < 3000.0, 12000.0, 10000.0), np.full_like(xs, 82.311)])
    return {
        "path": path,
        "powers": [8000.0, 12000.0],
        "distances": [60.0, 120.0, 240.0, 480.0, 960.0, 1920.0, 3840.0],
        "sel": [[98.0, 92.0, 86.0, 80.0, 74.0, 68.0, 62.0],
                [104.0, 98.0, 92.0, 86.0, 80.0, 74.0, 68.0]],
        "lmax": [[94.0, 88.0, 82.0, 76.0, 70.0, 64.0, 58.0],
                 [100.0, 94.0, 88.0, 82.0, 76.0, 70.0, 64.0]],
        "grid_x": np.linspace(-2000.0, 14000.0, 10),
        "grid_y": np.linspace(-4000.0, 4000.0, 8),
        "observer": [2000.0, 500.0, 0.0],
    }


def hemisphere_inputs() -> dict[str, Any]:
    """A full 19x19x3 rotorcraft hemisphere with deterministic gaps.

    The gaps exercise the nearest-bin fill (Eq. 14/15) including the exact
    four-way tie at (0, 90) on the symmetric grid.
    """
    az = np.arange(-90.0, 91.0, 10.0)
    po = np.arange(0.0, 181.0, 10.0)
    fr = np.array([500.0, 1000.0, 2000.0])
    a_r = np.radians(az)[:, None, None]
    p_r = np.radians(po)[None, :, None]
    band = np.arange(fr.size)[None, None, :]
    levels = (80.0 + 6.0 * np.sin(p_r) * np.cos(a_r) - 3.0 * band
              + 0.5 * np.cos(3.0 * p_r))
    # Out-of-coverage wedge (as in real measured hemispheres) and a tie hole.
    levels[az < -55.0, :, :] = np.nan          # port wedge unmeasured
    levels[:, po < 25.0, :] = np.nan           # forward cap unmeasured
    levels[az == 0.0, po == 90.0, 0] = np.nan  # 4-way-tie hole, 500 Hz band
    queries = [(0.0, 90.0), (5.0, 85.0), (-45.0, 120.0), (88.0, 178.0), (0.0, 25.0)]
    return {"frequencies": fr, "azimuth": az, "polar": po, "levels": levels,
            "queries": queries}


def pe_inputs() -> dict[str, Any]:
    """Isovelocity duct for the parabolic-equation marching solver."""
    return {"frequency_hz": 100.0, "depths": [0.0, 1000.0],
            "sound_speeds": [1500.0, 1500.0], "source_depth": 50.0,
            "max_range": 3000.0, "range_step": 20.0, "n_depth_points": 256}


def ecma_signal() -> tuple[np.ndarray, float]:
    """A 0.5 s two-tone signal for the ECMA-418-2 analysers."""
    fs = 48000.0
    t = np.arange(int(0.5 * fs)) / fs
    sig = 0.08 * np.sin(2.0 * np.pi * 1000.0 * t) * (
        1.0 + 0.6 * np.sin(2.0 * np.pi * 70.0 * t))
    return sig, fs


def vibration_signal() -> tuple[np.ndarray, float]:
    """A 10 s two-component acceleration signal for Wd time-domain weighting."""
    fs = 1600.0
    t = np.arange(int(10.0 * fs)) / fs
    sig = 0.5 * np.sin(2.0 * np.pi * 8.0 * t) + 0.2 * np.sin(2.0 * np.pi * 63.0 * t)
    return sig, fs


# ---------------------------------------------------------------------------
# Hotspot cases: name -> (callable, golden extractor)
# ---------------------------------------------------------------------------


def run_airport_contour() -> np.ndarray:
    import phonometry as ph

    a = airport_inputs()
    res = ph.noise_contour(a["path"], a["powers"], a["distances"], a["sel"],
                           a["lmax"], x=a["grid_x"], y=a["grid_y"])
    return np.asarray(res.level)


def run_event_level() -> np.ndarray:
    import phonometry as ph

    a = airport_inputs()
    res = ph.event_level(a["path"], a["observer"], a["powers"], a["distances"],
                         a["sel"], a["lmax"])
    return np.concatenate([[res.level], np.asarray(res.segment_levels)])


def run_npd_batch() -> np.ndarray:
    import phonometry as ph

    a = airport_inputs()
    q = np.array([75.0, 150.0, 300.0, 600.0, 1200.0, 2400.0, 5000.0])
    return np.asarray(ph.npd_level(a["powers"], a["distances"], a["sel"], 10000.0, q))


def run_hemisphere_queries() -> np.ndarray:
    import phonometry as ph

    h = hemisphere_inputs()
    hemi = ph.RotorcraftHemisphere(h["frequencies"], h["azimuth"], h["polar"],
                                   h["levels"])
    out = [ph.hemisphere_source_level(hemi, phi, theta)
           for phi, theta in h["queries"]]
    return np.concatenate(out)


def run_parabolic_equation() -> np.ndarray:
    import phonometry as ph

    p = pe_inputs()
    res = ph.parabolic_equation(p["frequency_hz"], p["depths"], p["sound_speeds"],
                                source_depth=p["source_depth"],
                                max_range=p["max_range"],
                                range_step=p["range_step"],
                                n_depth_points=p["n_depth_points"])
    pl = np.asarray(res.propagation_loss)
    return pl[::16, 1::10].ravel()  # skip the r = 0 column (PL is inf there)


def run_ecma_loudness() -> np.ndarray:
    import phonometry as ph

    sig, fs = ecma_signal()
    res = ph.loudness_ecma(sig, fs)
    return np.concatenate([[float(res.loudness)],
                           np.asarray(res.specific_loudness).ravel()])


def run_ecma_roughness() -> np.ndarray:
    import phonometry as ph

    sig, fs = ecma_signal()
    res = ph.roughness_ecma(sig, fs)
    return np.concatenate([[float(res.roughness)],
                           np.asarray(res.specific_roughness).ravel()])


def run_ecma_tonality() -> np.ndarray:
    import phonometry as ph

    sig, fs = ecma_signal()
    res = ph.tonality_ecma(sig, fs)
    return np.concatenate([[float(res.tonality)],
                           np.asarray(res.specific_tonality).ravel()])


def run_vibration_weighting() -> np.ndarray:
    from phonometry.vibration.human.exposure import apply_weighting

    sig, fs = vibration_signal()
    out = np.asarray(apply_weighting(sig, fs, name="Wd"), dtype=np.float64)
    rms = float(np.sqrt(np.mean(out**2)))
    return np.concatenate([[rms], out[::2000]])


CASES: dict[str, Callable[[], np.ndarray]] = {
    "airport_contour_10x8": run_airport_contour,
    "airport_event_level": run_event_level,
    "airport_npd_batch": run_npd_batch,
    "rotorcraft_hemisphere_5q": run_hemisphere_queries,
    "underwater_pe_3km": run_parabolic_equation,
    "ecma_loudness_0.5s": run_ecma_loudness,
    "ecma_roughness_0.5s": run_ecma_roughness,
    "ecma_tonality_0.5s": run_ecma_tonality,
    "vibration_wd_weighting_10s": run_vibration_weighting,
}


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------


def _bench() -> None:
    print(f"{'case':<30} {'best [ms]':>12} {'runs':>6}   output")
    print("-" * 72)
    for name, fn in CASES.items():
        fn()  # warm-up (imports, caches)
        t0 = time.perf_counter()
        out = fn()
        once = time.perf_counter() - t0
        # Aim for ~1 s of total measurement, at least 3 repeats of >=1 run.
        number = max(1, int(0.35 / max(once, 1e-4)))
        best = min(timeit.repeat(fn, number=number, repeat=3)) / number
        print(f"{name:<30} {best * 1e3:>12.2f} {3 * number:>6}   "
              f"shape={np.asarray(out).shape}")


def _write_golden() -> None:
    lines = [
        "#  Copyright (c) 2026. Jose Manuel Requena Plens",
        '"""Golden regression arrays captured by ``scripts/bench.py --golden``.',
        "",
        "Auto-generated; do not edit by hand. These freeze the numeric outputs of",
        "the hotspot cases before the modularization/vectorization refactors so",
        "``tests/test_golden_baseline.py`` can assert equivalence. Regenerate only",
        "when a numeric change is intended and has been reviewed.",
        '"""',
        "",
        "import numpy as np",
        "",
        "GOLDEN = {",
    ]
    def _literal(v: float) -> str:
        if np.isnan(v):
            return "np.nan"
        if np.isinf(v):
            return "np.inf" if v > 0 else "-np.inf"
        return repr(v)

    for name, fn in CASES.items():
        arr = np.asarray(fn(), dtype=np.float64).ravel()
        chunks = [
            ", ".join(_literal(float(v)) for v in arr[i:i + 4])
            for i in range(0, arr.size, 4)
        ]
        vals = ",\n        ".join(chunks)
        lines.append(f'    "{name}": np.array([\n        {vals},\n    ]),')
        print(f"captured {name}: {arr.size} values")
    lines.append("}")
    lines.append("")
    _GOLDEN_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {_GOLDEN_PATH}")


def _time_figures() -> None:
    sys.path.insert(0, str(_ROOT / "scripts"))
    import generate_graphs as gg

    gg.set_lang("en")
    gg.set_theme(False)
    out_dir = str(_ROOT / ".github" / "images")
    rows: list[tuple[float, str]] = []
    import inspect

    for name in sorted(dir(gg)):
        if not name.startswith("generate_") or name in ("generate_all",
                                                        "generate_animations"):
            continue
        fn = getattr(gg, name)
        params = list(inspect.signature(fn).parameters)
        if params[:1] != ["output_dir"]:  # a generator taking something else
            continue
        t0 = time.perf_counter()
        fn(out_dir)
        rows.append((time.perf_counter() - t0, name))
    rows.sort(reverse=True)
    print(f"\n{'figure function':<45} {'wall [s]':>10}")
    print("-" * 57)
    for dt, name in rows:
        print(f"{name:<45} {dt:>10.2f}")
    print(f"\ntotal: {sum(dt for dt, _ in rows):.1f} s "
          f"(single EN/light variant, {len(rows)} functions)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden", action="store_true",
                        help="rewrite tests/golden_data.py with current outputs")
    parser.add_argument("--figures", action="store_true",
                        help="time every figure function (EN/light) and rank them")
    args = parser.parse_args()
    if args.golden:
        _write_golden()
    elif args.figures:
        _time_figures()
    else:
        _bench()


if __name__ == "__main__":
    main()
