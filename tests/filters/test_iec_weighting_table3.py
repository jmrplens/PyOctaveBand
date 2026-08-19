#  Copyright (c) 2026. Jose Manuel Requena Plens
"""
IEC 61672-1:2013 frequency weighting compliance (Table 3).

Nominal A/C/Z weightings and class 1 acceptance limits transcribed from the
official text: BS EN 61672-1:2013, **Table 3** ("Frequency weightings and
acceptance limits", standard page 22). Lower limits of ``-inf`` in the
standard are represented as negative infinity (only the upper limit applies).

The measured gain is the steady-state RMS ratio of a pure tone through the
real filter path (default ``high_accuracy``), at 16, 32, 48 and 96 kHz. The
two low rates are here because that path is not the designed second-order
sections alone: it reaches them through an interpolation and a decimation
stage whose anti-alias filter has its transition band on the input Nyquist
frequency, and at 16 and 32 kHz the highest Table 3 row below Nyquist sits
inside that transition band.
"""

import numpy as np
import pytest
from reference_data import IEC61672_TABLE3 as TABLE3

from phonometry import filters

# (nominal_freq_Hz, A_dB, C_dB, class1_upper_dB, class1_lower_dB)
# BS EN 61672-1:2013 Table 3 is imported from reference_data (shared with the
# CI conformance report). Z weighting is 0.0 dB at every frequency.


def _measured_gain_db(
    wf: filters.WeightingFilter, fs: int, f0: float
) -> float:
    """Steady-state RMS gain of the weighting filter at a single frequency."""
    # Longer windows at low frequencies keep the partial-cycle RMS error tiny.
    duration = max(0.5, 12 / f0)
    t = np.arange(int(fs * duration)) / fs
    x = np.sin(2 * np.pi * f0 * t)
    y = wf.filter(x)
    n0 = int(0.2 * fs)  # skip the filter transient
    return float(20 * np.log10(np.std(y[n0:]) / np.std(x[n0:])))


# Nominal labels of the Table 3 rows whose *measured* deviation leaves the
# class 1 mask, per curve and sample rate. Above 40 kHz the whole in-range
# table is met. Below it the highest row under Nyquist falls inside the
# resampler's anti-alias transition band (7 943.3 Hz is 0.993 of Nyquist at
# 16 kHz, 15 848.9 Hz is 0.991 of it at 32 kHz) and no internal design rate
# lifts it back onto the mask. C survives at 32 kHz because its class 1
# lower limit there is -16.0 dB and it lands at -15.3 dB; A lands at
# -16.2 dB and does not.
_CLASS1_MISSES: dict[tuple[str, int], set[float]] = {
    ("A", 16000): {8000.0},
    ("A", 32000): {16000.0},
    ("A", 48000): set(),
    ("A", 96000): set(),
    ("C", 16000): {8000.0},
    ("C", 32000): set(),
    ("C", 48000): set(),
    ("C", 96000): set(),
}


@pytest.mark.parametrize("fs", [16000, 32000, 48000, 96000])
@pytest.mark.parametrize("curve,column", [("A", 1), ("C", 2)])
def test_weighting_within_class1_limits_table3(fs: int, curve: str, column: int) -> None:
    wf = filters.WeightingFilter(fs, curve)
    missed: set[float] = set()
    detail = []
    for row in TABLE3:
        # Table 3 NOTE: the design goals are the analytic curve at the exact
        # base-10 frequency behind the nominal label (15 848.9 Hz for "16 k"),
        # so the tone probes that frequency.
        f0 = float(10.0 ** (np.round(10.0 * np.log10(row[0])) / 10.0))
        if f0 >= fs / 2:
            continue
        nominal, upper, lower = row[column], row[3], row[4]
        deviation = _measured_gain_db(wf, fs, f0) - nominal
        if not (lower <= deviation <= upper):
            missed.add(row[0])
            detail.append(f"{f0} Hz: deviation {deviation:+.2f} dB (limits {upper:+}/{lower:+})")
    assert missed == _CLASS1_MISSES[(curve, fs)], (
        f"{curve} @ fs={fs}: measured " + ("; ".join(detail) or "no miss")
    )


def test_z_weighting_is_flat() -> None:
    """Z weighting is 0.0 dB at every Table 3 frequency (bypass)."""
    wf = filters.WeightingFilter(48000, "Z")
    x = np.random.default_rng(0).standard_normal(4800)
    np.testing.assert_array_equal(wf.filter(x), x)
