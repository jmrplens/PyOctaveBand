#  Copyright (c) 2026. Jose Manuel Requena Plens
"""IEC 61672-1:2013 frequency weighting compliance (Table 3).

Nominal A/C/Z weightings and class 1 acceptance limits transcribed from the
official text: BS EN 61672-1:2013, **Table 3** ("Frequency weightings and
acceptance limits", standard page 22). Lower limits of ``-inf`` in the
standard are represented as negative infinity (only the upper limit applies).

The measured gain is the steady-state RMS ratio of a pure tone through the
real filter path (default ``high_accuracy``), at 16, 32, 48 and 96 kHz. The
two low rates are here because they used to be the hard ones: the path reached
its sections through an interpolation and a decimation stage whose anti-alias
filter had its transition band on the input Nyquist frequency, and at 16 and
32 kHz the highest Table 3 row below Nyquist sits inside that band -- 7 943.3
and 15 848.9 Hz are 0.993 and 0.991 of the respective Nyquist frequencies. The
A weighting missed class 1 at both, by 12.0 and 0.245 dB. The design is now
fitted at the input rate over a band that reaches 0.995 of Nyquist, so those
rows are inside what the design controls and no rate in this file misses a
single row.
"""

import numpy as np
import pytest
from reference_data import IEC61672_TABLE3 as TABLE3

from phonometry import filters

# (nominal_freq_Hz, A_dB, C_dB, class1_upper_dB, class1_lower_dB)
# BS EN 61672-1:2013 Table 3 is imported from reference_data (shared with the
# CI conformance report). Z weighting is 0.0 dB at every frequency.


def _measured_gain_db(wf: filters.WeightingFilter, fs: int, f0: float) -> float:
    """Steady-state RMS gain of the weighting filter at a single frequency."""
    # Longer windows at low frequencies keep the partial-cycle RMS error tiny.
    duration = max(0.5, 12 / f0)
    t = np.arange(int(fs * duration)) / fs
    x = np.sin(2 * np.pi * f0 * t)
    y = wf.filter(x)
    n0 = int(0.2 * fs)  # skip the filter transient
    # Close the window on a whole number of periods. Input and output differ in
    # phase, so a part cycle biases the two RMS values by different amounts:
    # 0.02 dB at the 16 and 20 Hz rows, which is the size of the quantity the
    # rounding test below is measuring.
    period = fs / f0
    whole = int((t.size - n0) / period)
    n1 = n0 + max(int(round(whole * period)), 1)
    return float(20 * np.log10(np.std(y[n0:n1]) / np.std(x[n0:n1])))


#: Widest deviation from a Table 3 design goal this file will accept at any
#: row, at any of its rates. It is not a tolerance from the standard: Table 3
#: prints its goals to 0.1 dB, so a filter that reproduced the analytic Annex E
#: curve exactly would still read up to 0.05 dB off the printed number, and
#: that rounding is what this bound is. Holding the filter to it says the
#: realisation adds nothing measurable to the table's own quantum -- a far
#: stronger statement than the class 1 mask, whose narrowest row is +/-0.7 dB.
_TABLE3_ROUNDING_DB = 0.055


@pytest.mark.parametrize("fs", [16000, 32000, 48000, 96000])
@pytest.mark.parametrize(("curve", "column"), [("A", 1), ("C", 2)])
def test_weighting_within_class1_limits_table3(
    fs: int, curve: str, column: int
) -> None:
    """Every Table 3 row below Nyquist, at every rate, inside the class 1 mask."""
    wf = filters.WeightingFilter(fs, curve)
    missed = []
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
            missed.append(
                f"{f0} Hz: deviation {deviation:+.2f} dB (limits {upper:+}/{lower:+})"
            )
    assert not missed, f"{curve} @ fs={fs}: " + "; ".join(missed)


@pytest.mark.parametrize("fs", [16000, 32000, 48000, 96000])
@pytest.mark.parametrize(("curve", "column"), [("A", 1), ("C", 2)])
def test_weighting_sits_inside_the_rounding_of_table_3(
    fs: int, curve: str, column: int
) -> None:
    """The realised filter reads the printed goal back, to the table's quantum.

    The class 1 mask above is what conformance asks for. This is what the
    library holds itself to: a deviation no larger than the 0.05 dB rounding
    Table 3 is printed to, so the realisation is invisible against the table
    at every row it can reach, including the ones that sit within 1 % of the
    Nyquist frequency at 16 and 32 kHz.
    """
    wf = filters.WeightingFilter(fs, curve)
    for row in TABLE3:
        f0 = float(10.0 ** (np.round(10.0 * np.log10(row[0])) / 10.0))
        if f0 >= fs / 2:
            continue
        deviation = _measured_gain_db(wf, fs, f0) - row[column]
        assert abs(deviation) <= _TABLE3_ROUNDING_DB, (
            f"{curve} @ fs={fs}, {f0} Hz: {deviation:+.4f} dB"
        )


def test_z_weighting_is_flat() -> None:
    """Z weighting is 0.0 dB at every Table 3 frequency (bypass)."""
    wf = filters.WeightingFilter(48000, "Z")
    x = np.random.default_rng(0).standard_normal(4800)
    np.testing.assert_array_equal(wf.filter(x), x)
