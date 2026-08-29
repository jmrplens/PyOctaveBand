#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Psophometric quasi-peak meter (ITU-R BS.468-4 clause 2).

The detector half of the CCIR/ITU noise meter. Clause 2 prints no time
constant, no rise time and no transfer function: it says the dynamic
performance "may be realized in a variety of ways" and then defines it
entirely through eleven acceptance windows. Table 2 reads a single 5 kHz
tone burst at eight durations, Table 3 a train of 5 ms bursts at three
repetition rates, and every cell is a percentage of the reading the same
tone gives steadily.

Two things follow for a conformance report, and both are unusual enough to
say out loud. First, these rows attest an **acceptance region**, not
agreement with a value: every reading inside a window conforms, and the
Recommendation tolerates about 2.5 dB of disagreement between two conforming
instruments on a single burst. Second, the time constants behind the
readings are a fit this library made to the reference column, not a quoted
value, so the row that would be circular - "does the fit reproduce what it
was fitted to" - is deliberately not here. What is here is the fit measured
against the windows, which is the only claim BS.468-4 authorises.

The twelfth row is clause 2.6, the one absolute statement in clause 2 and
the only cell of the detector specification that is an equality rather than
a range.
"""

from __future__ import annotations

import functools
import math
from typing import Any

import numpy as np
import reference_data as ref

import phonometry as ph

from ..registry import Outcome, numeric, register

_QUASI_PEAK = "Quasi-peak meter (ITU-R BS.468-4)"

#: The rate every row is measured at. The readings move by at most 0.080 dB
#: over 32 to 192 kHz, or 4.2 % of the narrowest window in the document.
_FS = 48000.0


@functools.cache
def _dynamics() -> dict[str, dict[str, Any]]:
    """The eleven readings, keyed by the stimulus label, computed once."""
    report = ph.broadcast.verify_quasi_peak_dynamics(_FS)
    return {row["stimulus"]: row for row in report["stimuli"]}


def _window_outcome(stimulus: str) -> Outcome:
    """One acceptance window as a report row: the band, the reading, the margin."""
    row = _dynamics()[stimulus]
    lower, upper = row["lower_percent"], row["upper_percent"]
    reading = row["reading_percent"]
    return Outcome(
        expected=f"{lower:g} to {upper:g} %",
        computed=f"{reading:.2f} %",
        delta=f"{row['margin_db']:+.3f} dB",
        passed=lower <= reading <= upper,
    )


def _register_single_bursts() -> None:
    """Register the eight Table 2 rows, one per burst duration."""
    for duration_ms, cycles, *_limits in ref.BS468_TABLE2_SINGLE_BURSTS:
        register(
            _QUASI_PEAK,
            "ITU-R BS.468-4 Table 2",
            f"Single {duration_ms:g} ms 5 kHz burst ({cycles} periods), "
            "% of the steady reading",
        )(functools.partial(_window_outcome, f"{duration_ms:g} ms"))


def _register_burst_trains() -> None:
    """Register the three Table 3 rows, one per repetition rate."""
    for rate, *_limits in ref.BS468_TABLE3_BURST_TRAINS:
        register(
            _QUASI_PEAK,
            "ITU-R BS.468-4 Table 3",
            f"5 ms 5 kHz bursts at {rate:g} per second, % of the steady reading",
        )(functools.partial(_window_outcome, f"{rate:g} bursts/s"))


_register_single_bursts()
_register_burst_trains()


@register(
    _QUASI_PEAK,
    "ITU-R BS.468-4 clause 2.6",
    "Steady 1 kHz sine at 0.775 V r.m.s., dBqps",
)
def _chk_calibration() -> Outcome:
    """Clause 2.6 fixes the scale of the whole instrument, as an equality."""
    t = np.arange(round(3.0 * _FS)) / _FS
    tone = math.sqrt(2.0) * ref.BS468_CALIBRATION_V * np.sin(2.0 * math.pi * 1000.0 * t)
    computed = ph.broadcast.quasi_peak_meter(tone, _FS).level_db
    return numeric(0.0, computed, 1e-6, unit="dBqps", places=6)
