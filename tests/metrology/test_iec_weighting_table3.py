#  Copyright (c) 2026. Jose Manuel Requena Plens
"""
IEC 61672-1:2013 frequency weighting compliance (Table 3).

Nominal A/C/Z weightings and class 1 acceptance limits transcribed from the
official text: BS EN 61672-1:2013, **Table 3** ("Frequency weightings and
acceptance limits", standard page 22). Lower limits of ``-inf`` in the
standard are represented as negative infinity (only the upper limit applies).

The measured gain is the steady-state RMS ratio of a pure tone through the
real filter path (default ``high_accuracy``), at 48 kHz and 96 kHz.
"""

import numpy as np
import pytest
from reference_data import IEC61672_TABLE3 as TABLE3

from phonometry import WeightingFilter

# (nominal_freq_Hz, A_dB, C_dB, class1_upper_dB, class1_lower_dB)
# BS EN 61672-1:2013 Table 3 is imported from reference_data (shared with the
# CI conformance report). Z weighting is 0.0 dB at every frequency.


def _measured_gain_db(wf: WeightingFilter, fs: int, f0: float) -> float:
    """Steady-state RMS gain of the weighting filter at a single frequency."""
    # Longer windows at low frequencies keep the partial-cycle RMS error tiny.
    duration = max(0.5, 12 / f0)
    t = np.arange(int(fs * duration)) / fs
    x = np.sin(2 * np.pi * f0 * t)
    y = wf.filter(x)
    n0 = int(0.2 * fs)  # skip the filter transient
    return float(20 * np.log10(np.std(y[n0:]) / np.std(x[n0:])))


@pytest.mark.parametrize("fs", [48000, 96000])
@pytest.mark.parametrize("curve,column", [("A", 1), ("C", 2)])
def test_weighting_within_class1_limits_table3(fs: int, curve: str, column: int) -> None:
    wf = WeightingFilter(fs, curve)
    failures = []
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
            failures.append(f"{f0} Hz: deviation {deviation:+.2f} dB (limits {upper:+}/{lower:+})")
    assert not failures, f"{curve} @ fs={fs}: " + "; ".join(failures)


def test_z_weighting_is_flat() -> None:
    """Z weighting is 0.0 dB at every Table 3 frequency (bypass)."""
    wf = WeightingFilter(48000, "Z")
    x = np.random.default_rng(0).standard_normal(4800)
    np.testing.assert_array_equal(wf.filter(x), x)
