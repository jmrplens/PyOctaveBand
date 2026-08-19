#  Copyright (c) 2026. Jose Manuel Requena Plens
"""
IEC 61672-1:2013 tone-burst response compliance.

Reference values and acceptance limits transcribed from the official text:
BS EN 61672-1:2013, **Table 4** ("Reference 4 kHz toneburst responses and
acceptance limits", standard page 25). The table defines reference responses
delta_ref for maximum time-weighted sound levels (Equation 7,
``10*lg(1 - exp(-Tb/tau))``) for the F and S time weightings; the standard
does not specify toneburst responses for an impulse weighting.

Tolerances below are the **class 1** acceptance limits from the same table
(asymmetric where the standard specifies them). FAST values cross-checked
against the maintainer's analysis in
https://github.com/jmrplens/phonometry/issues/38.
"""

import numpy as np
import pytest

from phonometry import filters

FS = 48000
F0 = 4000


def _burst_response_db(duration: float, mode: str, total: float) -> float:
    """Max time-weighted level of a 4 kHz toneburst relative to steady state."""
    t = np.arange(int(FS * total)) / FS
    x = np.sin(2 * np.pi * F0 * t)

    steady = filters.time_weighting(x, FS, mode=mode)
    ref = steady[int((total - 0.5) * FS):].mean()

    burst = np.zeros_like(t)
    start = int(1.0 * FS)
    burst[start:start + round(duration * FS)] = x[start:start + round(duration * FS)]
    env = filters.time_weighting(burst, FS, mode=mode)
    return float(10 * np.log10(env.max() / ref))


# (duration_s, delta_ref_dB, class1_upper_dB, class1_lower_dB)
# BS EN 61672-1:2013 Table 4, column "LAFmax - LA" (F weighting), class 1 limits.
FAST_CASES = [
    (1.000, 0.0, 0.5, -0.5),
    (0.500, -0.1, 0.5, -0.5),
    (0.200, -1.0, 0.5, -0.5),
    (0.100, -2.6, 1.0, -1.0),
    (0.050, -4.8, 1.0, -1.0),
    (0.020, -8.3, 1.0, -1.0),
    (0.010, -11.1, 1.0, -1.0),
    (0.005, -14.1, 1.0, -1.0),
    (0.002, -18.0, 1.0, -1.5),
    (0.001, -21.0, 1.0, -2.0),
]

# BS EN 61672-1:2013 Table 4, column "LASmax - LA" (S weighting), class 1 limits.
SLOW_CASES = [
    (1.000, -2.0, 0.5, -0.5),
    (0.500, -4.1, 0.5, -0.5),
    (0.200, -7.4, 0.5, -0.5),
    (0.100, -10.2, 1.0, -1.0),
    (0.050, -13.1, 1.0, -1.0),
    (0.020, -17.0, 1.0, -1.5),
    (0.010, -20.0, 1.0, -2.0),
    (0.005, -23.0, 1.0, -2.5),
    (0.002, -27.0, 1.0, -3.0),
]


@pytest.mark.parametrize("duration,ref,upper,lower", FAST_CASES)
def test_fast_tone_burst_iec_table4(duration: float, ref: float, upper: float, lower: float) -> None:
    measured = _burst_response_db(duration, "fast", total=3.0)
    assert lower <= measured - ref <= upper, (
        f"FAST {duration * 1000:g} ms burst: {measured:.2f} dB vs delta_ref {ref} dB "
        f"(class 1 limits {upper:+}/{lower:+})"
    )


@pytest.mark.parametrize("duration,ref,upper,lower", SLOW_CASES)
def test_slow_tone_burst_iec_table4(duration: float, ref: float, upper: float, lower: float) -> None:
    measured = _burst_response_db(duration, "slow", total=8.0)
    assert lower <= measured - ref <= upper, (
        f"SLOW {duration * 1000:g} ms burst: {measured:.2f} dB vs delta_ref {ref} dB "
        f"(class 1 limits {upper:+}/{lower:+})"
    )


def test_delta_ref_equation7_consistency() -> None:
    """
    Sanity-check the transcription: Table 4 values follow Equation (7),
    ``delta_ref = 10*lg(1 - exp(-Tb/tau))`` (tau_F = 0.125 s, tau_S = 1 s),
    within the standard's own rounding (0.1 dB, plus its -0.1-quirk rows).
    """
    for cases, tau in ((FAST_CASES, 0.125), (SLOW_CASES, 1.0)):
        for duration, ref, _, _ in cases:
            eq7 = 10 * np.log10(1 - np.exp(-duration / tau))
            assert eq7 == pytest.approx(ref, abs=0.15), (
                f"tau={tau}: Tb={duration}: table {ref} vs Eq.(7) {eq7:.2f}"
            )


def _burst_sel_response_db(duration: float) -> float:
    """LAE of a 4 kHz toneburst relative to the steady A-weighted level."""
    from phonometry import signals
    from phonometry.filters.weighting import weighting_filter

    total = 3.0
    t = np.arange(int(FS * total)) / FS
    x = np.sin(2 * np.pi * F0 * t)
    la_steady = signals.leq(weighting_filter(x, FS, "A")[int(0.5 * FS):])

    burst = np.zeros_like(t)
    start = int(1.0 * FS)
    burst[start:start + round(duration * FS)] = x[start:start + round(duration * FS)]
    return float(signals.sel(burst, FS, weighting="A")) - float(la_steady)


# BS EN 61672-1:2013 Table 4, column "LAE - LA" (Equation 8:
# delta_ref = 10*lg(Tb/T0), T0 = 1 s) with the per-row class 1 limits.
SEL_CASES = [
    (1.000, 0.0, 0.5, -0.5),
    (0.500, -3.0, 0.5, -0.5),
    (0.200, -7.0, 0.5, -0.5),
    (0.100, -10.0, 1.0, -1.0),
    (0.050, -13.0, 1.0, -1.0),
    (0.020, -17.0, 1.0, -1.0),
    (0.010, -20.0, 1.0, -1.0),
    (0.005, -23.0, 1.0, -1.0),
    (0.002, -27.0, 1.0, -1.5),
    (0.001, -30.0, 1.0, -2.0),
]


@pytest.mark.parametrize("duration,ref,upper,lower", SEL_CASES)
def test_sel_tone_burst_iec_table4(duration: float, ref: float, upper: float, lower: float) -> None:
    measured = _burst_sel_response_db(duration)
    assert lower <= measured - ref <= upper, (
        f"LAE {duration * 1000:g} ms burst: {measured:.2f} dB vs delta_ref {ref} dB "
        f"(class 1 limits {upper:+}/{lower:+})"
    )


def test_delta_ref_equation8_consistency() -> None:
    """
    Sanity-check the transcription: the ``LAE - LA`` column of Table 4 follows
    Equation (8), ``delta_ref = 10*lg(Tb/T0)`` with ``T0 = 1 s``.

    The counterpart of :func:`test_delta_ref_equation7_consistency` for the
    third column. Without it, ``SEL_CASES`` is ten numbers copied out of the
    standard that nothing recomputes: a digit slipped while transcribing them
    would move the target the burst response is measured against, and the only
    thing standing between that and a green run is the class 1 tolerance the
    same row carries.

    Equation (8) is exact where Equation (7) is not: every row is the table's
    own 0.1 dB rounding of ``10*lg(Tb)``, with no rounding quirks to allow for,
    so this compares the rounded value outright rather than within a band.
    """
    for duration, ref, _, _ in SEL_CASES:
        eq8 = 10 * np.log10(duration / 1.0)
        assert round(eq8, 1) == pytest.approx(ref), (
            f"Tb={duration}: table {ref} vs Eq.(8) {eq8:.4f} (rounds to {round(eq8, 1)})"
        )
