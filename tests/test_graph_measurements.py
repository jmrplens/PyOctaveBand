#  Copyright (c) 2026. Jose M. Requena-Plens
"""
Guards for the measurement methods used by generate_graphs.py.

The published weighting-curves figure once showed A/C ~2.4 dB low (never
crossing 0 dB) because the impulse sat at sample 0 and the high-accuracy
resampling path truncated the interpolation kernel at the array edge.
These tests call the ACTUAL graph measurement helper, so CI breaks if the
figures would ship distorted curves again.
"""

import math
import pathlib
import sys

import numpy as np
import pytest
from reference_data import (
    ANSIS14_TABLE4_B,
    ANSIS14_TABLE5,
    IEC537_NASA_TABLE_SLD1,
    IEC61012_TABLE1,
)

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "scripts"))
import generate_graphs

FS = 48000
#: Sample rate of the special-curves figure (B, D, AU). It runs at 96 kHz, not
#: the 48 kHz of the IEC 61672-1 figure, because the IEC 61012 U low-pass that
#: makes AU what it is is specified up to 40 kHz.
SPECIAL_FS = 96000


def _exact(freq: float) -> float:
    """Exact base-10 frequency behind a nominal one-third-octave label."""
    return float(10.0 ** (round(10.0 * math.log10(freq)) / 10.0))


def _analytic_a_db(f: np.ndarray) -> np.ndarray:
    """IEC 61672-1 analytic A-weighting curve."""
    f2 = np.asarray(f, dtype=float) ** 2
    ra = (12194**2 * f2**2) / (
        (f2 + 20.6**2)
        * np.sqrt((f2 + 107.7**2) * (f2 + 737.9**2))
        * (f2 + 12194**2)
    )
    return 20 * np.log10(ra) + 2.0


def _analytic_c_db(f: np.ndarray) -> np.ndarray:
    """IEC 61672-1 analytic C-weighting curve."""
    f2 = np.asarray(f, dtype=float) ** 2
    rc = (12194**2 * f2) / ((f2 + 20.6**2) * (f2 + 12194**2))
    return 20 * np.log10(rc) + 0.06


def test_graph_a_curve_matches_analytic() -> None:
    """The figure's A curve must match the IEC analytic curve, LF to HF."""
    freqs = np.array([16.0, 31.5, 100.0, 1000.0, 2500.0, 4000.0, 8000.0])
    _, mag = generate_graphs.measure_weighting_response(FS, "A", freqs)
    np.testing.assert_allclose(mag, _analytic_a_db(freqs), atol=0.3)


def test_graph_c_curve_matches_analytic() -> None:
    """The figure's C curve must match the IEC analytic curve, LF to HF."""
    freqs = np.array([16.0, 31.5, 100.0, 1000.0, 4000.0, 8000.0])
    _, mag = generate_graphs.measure_weighting_response(FS, "C", freqs)
    np.testing.assert_allclose(mag, _analytic_c_db(freqs), atol=0.3)


def test_graph_a_curve_shows_positive_bump() -> None:
    """
    The exact failure mode that shipped: an A curve that never crosses 0 dB.

    The A-weighting is positive between ~1.1 and ~6.2 kHz with a maximum of
    +1.27 dB at ~2.5 kHz (IEC 61672-1 Table 2). The plotted curve must show it.
    """
    w, mag = generate_graphs.measure_weighting_response(FS, "A")
    assert mag.max() == pytest.approx(1.27, abs=0.15), (
        f"A-curve max is {mag.max():+.2f} dB; the +1.27 dB bump is missing"
    )
    f_max = w[mag.argmax()]
    assert 2000 < f_max < 3200, f"A-curve peak at {f_max:.0f} Hz, expected ~2.5 kHz"


def test_graph_curves_anchor_at_1khz() -> None:
    """Both weightings are normalized to 0 dB at 1 kHz; the figure must agree."""
    for curve in ("A", "C"):
        _, mag = generate_graphs.measure_weighting_response(FS, curve, np.array([1000.0]))
        assert mag[0] == pytest.approx(0.0, abs=0.1), f"{curve} at 1 kHz: {mag[0]:+.2f} dB"


def test_graph_b_curve_matches_published_table() -> None:
    """
    The B trace of the special-curves figure, against its published table.

    Oracle: ANSI S1.4-1983 Table IV design goals (B column) with the Table V
    Type 0 tolerance, evaluated at the exact base-10 frequency behind each
    nominal label (the standard's rows are labels for 1000*10**(n/10) Hz).
    """
    freqs = np.array([_exact(row[0]) for row in ANSIS14_TABLE4_B])
    _, mag = generate_graphs.measure_weighting_response(SPECIAL_FS, "B", freqs)
    for (freq, goal), limits, got in zip(
        ANSIS14_TABLE4_B, ANSIS14_TABLE5, mag, strict=True
    ):
        upper0, lower0 = limits[1], limits[2]
        assert lower0 <= got - goal <= upper0, (
            f"B at {freq:.0f} Hz deviates {got - goal:+.2f} dB from the "
            f"Table IV goal, outside the Type 0 mask"
        )


def test_graph_d_curve_matches_published_table() -> None:
    """
    The D trace of the special-curves figure, including the annotated hump.

    Oracle: the IEC 537:1976 one-third-octave table republished in NASA
    CR-3406 Table SLD-I, whose maximum is the +11.5 dB at 3.15 kHz the
    figure annotates. The 1600 Hz and 2500 Hz cells round a different source
    curve (see ``reference_data``), hence the wider bound there.
    """
    freqs = np.array([f for f, _ in IEC537_NASA_TABLE_SLD1])
    _, mag = generate_graphs.measure_weighting_response(SPECIAL_FS, "D", freqs)
    for (freq, value), got in zip(IEC537_NASA_TABLE_SLD1, mag, strict=True):
        bound = 0.45 if freq in (1600, 2500) else 0.25
        assert got == pytest.approx(value, abs=bound), f"D at {freq:.0f} Hz"
    peak = max(value for _, value in IEC537_NASA_TABLE_SLD1)
    assert peak == 11.5, "the annotated +11.5 dB hump left the table"


def test_graph_au_trace_is_a_plus_the_u_lowpass() -> None:
    """
    AU must sit on the A reference through the audible range, then cut.

    Oracle: IEC 61012:1990 Table 1, the nominal relative response of the U
    weighting as a separate unit with its per-row tolerance. AU is A
    cascaded with U (subclause 2.2), so the difference between the figure's
    own AU and A traces is the U response. The extra 0,2 dB of slack covers
    the 1 kHz row, whose published tolerance is exactly zero.
    """
    rows = [row for row in IEC61012_TABLE1
            if row[0] in (1000.0, 8000.0, 16000.0, 20000.0, 40000.0)]
    freqs = np.array([row[0] for row in rows])
    _, mag_au = generate_graphs.measure_weighting_response(SPECIAL_FS, "AU", freqs)
    _, mag_a = generate_graphs.measure_weighting_response(SPECIAL_FS, "A", freqs)
    for (freq, nominal, upper, lower), u_db in zip(rows, mag_au - mag_a, strict=True):
        assert nominal + lower - 0.2 <= u_db <= nominal + upper + 0.2, (
            f"U response at {freq:.0f} Hz is {u_db:+.2f} dB, outside the "
            f"IEC 61012 Table 1 band {nominal + lower:+.1f}/{nominal + upper:+.1f} dB"
        )
