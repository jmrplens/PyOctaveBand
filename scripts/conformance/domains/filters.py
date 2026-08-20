#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Domain 1 - Filters & weightings.

The fractional-octave bank against the IEC 61260-1 class masks, and the
frequency weightings (A and C of IEC 61672-1, G of ISO 7196, B of
ANSI S1.4-1983, AU of IEC 61012, D of IEC 537) against the design-goal tables
and tolerance masks they are defined by.

The class verdicts and the weighting deviations are not computed here: they
come from :mod:`conformance.shared`, which the showcase table at the head of
the report also calls, so a row below and the table above can never disagree
about the same filter.
"""

from __future__ import annotations

import numpy as np
import reference_data as ref

import phonometry as ph
from phonometry import filters
from phonometry.filters.weighting import _runtime_frequency_response

from ..registry import Outcome, numeric, register
from ..render import _snap
from ..shared import _filter_class, _weighting_deviation


def _filter_class_check(arch: str, fraction: float, label: str) -> Outcome:
    res = _filter_class(arch, fraction)
    margin = res.min_margin1
    ok = res.overall_class == 1
    return Outcome(
        expected="class 1",
        computed=(f"class {res.overall_class}" if res.overall_class else "none")
        + f" (margin {margin:+.3f} dB)",
        delta=f"{margin:+.3f} dB",
        passed=ok,
    )


@register(
    "Filters & weightings",
    "IEC 61260-1:2014 Table 1",
    "Octave-band filter class (butterworth, fs=48 kHz)",
)
def _chk_butter_octave() -> Outcome:
    return _filter_class_check("butter", 1, "octave")


@register(
    "Filters & weightings",
    "IEC 61260-1:2014 Table 1",
    "One-third-octave filter class (butterworth, fs=48 kHz)",
)
def _chk_butter_third() -> Outcome:
    return _filter_class_check("butter", 3, "third")


@register(
    "Filters & weightings",
    "IEC 61260:1995 / ANSI S1.11-2004 Table 1",
    "Class 0 (strictest) octave-band filter (butterworth, fs=48 kHz)",
)
def _chk_butter_class0_1995() -> Outcome:
    bank = filters.OctaveFilterBank(
        48000,
        fraction=1,
        order=6,
        limits=[100, 10000],
        design=filters.FilterDesign(filter_type="butter"),
    )
    result = ph.filters.verify_filter_class(bank, edition="1995")
    margin = min(b["margin_class0_db"] for b in result["bands"])
    ok = result["overall_class"] == 0
    return Outcome(
        expected="class 0",
        computed=(
            f"class {result['overall_class']}"
            if result["overall_class"] is not None
            else "none"
        )
        + f" (margin {margin:+.3f} dB)",
        delta=f"{margin:+.3f} dB",
        passed=ok,
    )


@register(
    "Filters & weightings",
    "IEC 61260-1:2014 Table F.1",
    "Formula (9) breakpoint mapping, b=3, Omega at G**(1/2)",
)
def _chk_map_breakpoint_table_f1() -> Outcome:
    from phonometry.filters.compliance import _map_breakpoint

    return numeric(
        ref.IEC61260_TABLE_F1[0.5][0], _map_breakpoint(0.5, 3), 5e-6, places=5
    )


def _weighting_check(curve: str, fs: int) -> Outcome:
    res = _weighting_deviation(curve, fs)
    headroom = res.min_headroom
    band = f"[{res.bind_lower:+.2f}, {res.bind_upper:+.2f}] dB"
    return Outcome(
        expected=f"deviation within limits @ {res.bind_freq:.0f} Hz",
        computed=f"{_snap(res.bind_dev):+.3f} dB in {band}",
        delta=f"headroom {headroom:+.3f} dB",
        passed=headroom >= 0.0,
    )


@register(
    "Filters & weightings",
    "IEC 61672-1:2013 Table 3",
    "A-weighting deviation vs class-1 limits (fs=48 kHz)",
)
def _chk_a_weighting() -> Outcome:
    return _weighting_check("A", 48000)


@register(
    "Filters & weightings",
    "IEC 61672-1:2013 Table 3",
    "C-weighting deviation vs class-1 limits (fs=48 kHz)",
)
def _chk_c_weighting() -> Outcome:
    return _weighting_check("C", 48000)


@register(
    "Filters & weightings",
    "ISO 7196:1995 Table 2 / A.3",
    "G-weighting deviation vs +/-1 dB tolerance (fs=48 kHz)",
)
def _chk_g_weighting() -> Outcome:
    return _weighting_check("G", 48000)


@register(
    "Filters & weightings",
    "ANSI S1.4-1983 Tables IV/V",
    "B-weighting (historical) deviation vs Type 0 limits (fs=48 kHz)",
)
def _chk_b_weighting() -> Outcome:
    return _weighting_check("B", 48000)


@register(
    "Filters & weightings",
    "IEC 61012:1990 Table 1 / 2.2",
    "AU-weighting deviation vs separate-unit tolerances (fs=96 kHz)",
)
def _chk_au_weighting() -> Outcome:
    # 96 kHz so the 25/31.5/40 kHz rows (exact base-10 frequencies up to
    # 39 811 Hz) fall below Nyquist and the full Table 1 range is checked.
    return _weighting_check("AU", 96000)


@register(
    "Filters & weightings",
    "IEC 537:1976 (withdrawn) via NASA CR-3406 Table SLD-I",
    "D-weighting response vs the published tabulated curve (fs=48 kHz)",
)
def _chk_d_weighting() -> Outcome:
    # IEC 537 is withdrawn and published no surviving tolerance table, so the
    # D response is pinned against the tabulated curve republished in the
    # NASA Handbook of Aircraft Noise Metrics (Table SLD-I, printed at the
    # integer nominal frequencies to 0.1 dB). The rational transfer function
    # reproduces every row within 0.1 dB except 1600/2500 Hz, which appear to
    # round a different source curve; the realized filter adds bilinear
    # residuals below 0.1 dB, so the acceptance bound is 0.2 dB (0.45 dB at
    # the two outlier cells).
    wf = filters.WeightingFilter(48000, "D")
    freqs = np.array([r[0] for r in ref.IEC537_NASA_TABLE_SLD1], dtype=float)
    table = np.array([r[1] for r in ref.IEC537_NASA_TABLE_SLD1], dtype=float)
    h = _runtime_frequency_response(wf, np.concatenate([freqs, [1000.0]]))
    gain = 20.0 * np.log10(np.abs(h))
    dev = (gain[:-1] - gain[-1]) - table
    bound = np.where(np.isin(freqs, (1600.0, 2500.0)), 0.45, 0.2)
    worst = int(np.argmax(np.abs(dev) / bound))
    ok = bool(np.all(np.abs(dev) <= bound))
    return Outcome(
        expected="abs(response - table) <= 0.2 dB (0.45 dB at 1600/2500 Hz)",
        computed=(
            f"{dev[worst]:+.3f} dB @ {freqs[worst]:.0f} Hz "
            f"(bound {bound[worst]:.2f} dB)"
        ),
        delta=f"headroom {bound[worst] - abs(dev[worst]):+.3f} dB",
        passed=ok,
    )
