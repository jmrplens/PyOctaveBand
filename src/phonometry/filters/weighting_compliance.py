#  Copyright (c) 2026. Jose Manuel Requena Plens
r"""IEC 61672-1:2013 frequency-weighting class verification.

A/C/Z frequency-weighting acceptance limits transcribed from
BS EN 61672-1:2013, **Table 3** (standard page 22): the design-goal responses
and the class 1 and class 2 upper/lower limits at the 34 nominal frequencies
from 10 Hz to 20 kHz. A lower limit of ``-inf`` means only the upper limit
applies (subclause 5.5.6 checks measured deviations at the nominal frequencies).

IEC 61672-1:2013 defines only classes 1 and 2. **Type 0**, the tightest of the
four instrument types, lives in the superseded **IEC 651:1979 Table V**, held
here as the identical British adoption BS 5969:1981. Its four masks differ
numerically from the 2013 edition (e.g. Type 0 is +2/-3 dB at both 16 and
20 kHz where class 1 is +2.5/-16 and +3/-inf), so the two editions are kept as
separate mask tables selected by the ``edition`` argument (``"2013"`` default
-> classes 1/2; ``"1979"`` -> Types 0/1/2/3, offered as classes 0-3).

The historical **B weighting** is verified against ANSI S1.4-1983: design
goals from the B column of **Table IV** (whose A and C columns equal IEC
61672-1:2013 Table 3 digit for digit) and tolerance limits from **Table V**,
whose instrument Types 1 and 2 fill the class 1 / class 2 verdict slots. The
ANSI Type 0 column is a *different* mask from the IEC 651 one - two-sided and
stricter at 10/12.5/16 Hz where IEC 651 is upper-only - so the two are carried
as the two editions they are and never merged.
The **AU weighting** is verified against IEC 61012:1990: design goals are the
sum of the nominal A response and the **Table 1** nominal U response (with the
subclause 2.2 explicit AU values at 25/31.5/40 kHz), checked against the
Table 1 tolerances for the filter as a separate unit, the tighter of the two
tolerance readings the standard offers. IEC 61012 publishes a single
tolerance set, so both verdict slots carry the same margin for AU.

One subject: the weighting network a sound level meter applies to the whole
signal, whose acceptance limits qualify the deviation of its measured relative
response from a design goal at the nominal frequencies. The band-filter class
limits of IEC 61260-1, which qualify a relative attenuation against a mask
around each mid-band frequency, live in :mod:`phonometry.filters.compliance`.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .weighting import WeightingFilter, _runtime_frequency_response

__all__ = [
    "verify_weighting_class",
    "weighting_class_limits",
]

_INF = float("inf")

# Floor on the `sweep_points` parameter of `verify_weighting_class` (default
# 4096): fewer grid frequencies would make the geometric grid of the
# between-nominals sweep too coarse for its verdict to mean anything. The
# floor is the function's own guard, not a value from the standard.
_MIN_SWEEP_POINTS = 64

# BS EN 61672-1:2013 Table 3 (standard page 22): design-goal frequency
# weightings and class 1 / class 2 acceptance limits at the 34 nominal
# frequencies. Columns: (nominal Hz, A dB, C dB, class1 upper, class1 lower,
# class2 upper, class2 lower); Z is 0.0 dB at every frequency. A lower limit
# of -inf means only the upper limit applies.
_WEIGHTING_TABLE3: list[tuple[float, float, float, float, float, float, float]] = [
    (10.0, -70.4, -14.3, 3.0, -_INF, 5.0, -_INF),
    (12.5, -63.4, -11.2, 2.5, -_INF, 5.0, -_INF),
    (16.0, -56.7, -8.5, 2.0, -4.0, 5.0, -_INF),
    (20.0, -50.5, -6.2, 2.0, -2.0, 3.0, -3.0),
    (25.0, -44.7, -4.4, 2.0, -1.5, 3.0, -3.0),
    (31.5, -39.4, -3.0, 1.5, -1.5, 3.0, -3.0),
    (40.0, -34.6, -2.0, 1.0, -1.0, 2.0, -2.0),
    (50.0, -30.2, -1.3, 1.0, -1.0, 2.0, -2.0),
    (63.0, -26.2, -0.8, 1.0, -1.0, 2.0, -2.0),
    (80.0, -22.5, -0.5, 1.0, -1.0, 2.0, -2.0),
    (100.0, -19.1, -0.3, 1.0, -1.0, 1.5, -1.5),
    (125.0, -16.1, -0.2, 1.0, -1.0, 1.5, -1.5),
    (160.0, -13.4, -0.1, 1.0, -1.0, 1.5, -1.5),
    (200.0, -10.9, 0.0, 1.0, -1.0, 1.5, -1.5),
    (250.0, -8.6, 0.0, 1.0, -1.0, 1.5, -1.5),
    (315.0, -6.6, 0.0, 1.0, -1.0, 1.5, -1.5),
    (400.0, -4.8, 0.0, 1.0, -1.0, 1.5, -1.5),
    (500.0, -3.2, 0.0, 1.0, -1.0, 1.5, -1.5),
    (630.0, -1.9, 0.0, 1.0, -1.0, 1.5, -1.5),
    (800.0, -0.8, 0.0, 1.0, -1.0, 1.5, -1.5),
    (1000.0, 0.0, 0.0, 0.7, -0.7, 1.0, -1.0),
    (1250.0, 0.6, 0.0, 1.0, -1.0, 1.5, -1.5),
    (1600.0, 1.0, -0.1, 1.0, -1.0, 2.0, -2.0),
    (2000.0, 1.2, -0.2, 1.0, -1.0, 2.0, -2.0),
    (2500.0, 1.3, -0.3, 1.0, -1.0, 2.5, -2.5),
    (3150.0, 1.2, -0.5, 1.0, -1.0, 2.5, -2.5),
    (4000.0, 1.0, -0.8, 1.0, -1.0, 3.0, -3.0),
    (5000.0, 0.5, -1.3, 1.5, -1.5, 3.5, -3.5),
    (6300.0, -0.1, -2.0, 1.5, -2.0, 4.5, -4.5),
    (8000.0, -1.1, -3.0, 1.5, -2.5, 5.0, -5.0),
    (10000.0, -2.5, -4.4, 2.0, -3.0, 5.0, -_INF),
    (12500.0, -4.3, -6.2, 2.0, -5.0, 5.0, -_INF),
    (16000.0, -6.6, -8.5, 2.5, -16.0, 5.0, -_INF),
    (20000.0, -9.3, -11.2, 3.0, -_INF, 5.0, -_INF),
]

_WEIGHTING_COL = {"A": 1, "C": 2, "Z": None}

# ---------------------------------------------------------------------------
# IEC 651:1979 - the superseded edition, and the only one that publishes a
# Type 0 mask for the frequency weightings.
# ---------------------------------------------------------------------------

# IEC 651:1979 Table V, read from the identical British adoption
# BS 5969:1981 (Table V, standard page 8): tolerances on the Table IV
# frequency weightings for each of the four instrument types of subclause 1.2,
# in decibels, at the same 34 nominal frequencies as IEC 61672-1:2013 Table 3.
# The scan carries no text layer, so the table was read from the printed page
# rather than from extracted text.
#
# Type 0 is the tightest grade: subclause 1.3 says the four types share the
# same centre values and differ only in the tolerances allowed, which broaden
# as the type number rises. The table's footnote makes it govern *every*
# weighting, not one of them - "Tolerances are the same for all weighting
# characteristics" - which is why B is checked against these limits under this
# edition rather than against the ANSI table. Subclause 6.1 says the same of
# the D weighting when provided, but D is not offered here: IEC 651 leaves its
# design goals to IEC 537, and that table is not in this module.
#
# Transcription note, the 1 kHz row: the printed cell is +/-0.7 dB for Type 0,
# while the footnote adds "The tolerance shall be zero at the reference
# frequency (see Sub-clause 3.7)" - and 3.7 puts the reference frequency
# anywhere from 200 Hz to 1 kHz, at the manufacturer's choice, with 1 kHz only
# "preferred". The printed cell is transcribed as it stands. The two readings
# cannot disagree here: the response is normalized to its own 1 kHz gain, so
# the deviation at 1 kHz is identically zero and sits inside a +/-0.7 dB cell
# and on a zero-width one alike.
#
# Row = (nominal Hz, T0 upper, T0 lower, T1 upper, T1 lower, T2 upper,
# T2 lower, T3 upper, T3 lower); a lower limit of -inf is the "+n; -inf" cell
# of the print, meaning only the upper limit applies.
_IEC651_TABLE5: list[
    tuple[float, float, float, float, float, float, float, float, float]
] = [
    (10.0, 2.0, -_INF, 3.0, -_INF, 5.0, -_INF, 5.0, -_INF),
    (12.5, 2.0, -_INF, 3.0, -_INF, 5.0, -_INF, 5.0, -_INF),
    (16.0, 2.0, -_INF, 3.0, -_INF, 5.0, -_INF, 5.0, -_INF),
    (20.0, 2.0, -2.0, 3.0, -3.0, 3.0, -3.0, 5.0, -_INF),
    (25.0, 1.5, -1.5, 2.0, -2.0, 3.0, -3.0, 5.0, -_INF),
    (31.5, 1.0, -1.0, 1.5, -1.5, 3.0, -3.0, 4.0, -4.0),
    (40.0, 1.0, -1.0, 1.5, -1.5, 2.0, -2.0, 4.0, -4.0),
    (50.0, 1.0, -1.0, 1.5, -1.5, 2.0, -2.0, 3.0, -3.0),
    (63.0, 1.0, -1.0, 1.5, -1.5, 2.0, -2.0, 3.0, -3.0),
    (80.0, 1.0, -1.0, 1.5, -1.5, 2.0, -2.0, 3.0, -3.0),
    (100.0, 0.7, -0.7, 1.0, -1.0, 1.5, -1.5, 3.0, -3.0),
    (125.0, 0.7, -0.7, 1.0, -1.0, 1.5, -1.5, 2.0, -2.0),
    (160.0, 0.7, -0.7, 1.0, -1.0, 1.5, -1.5, 2.0, -2.0),
    (200.0, 0.7, -0.7, 1.0, -1.0, 1.5, -1.5, 2.0, -2.0),
    (250.0, 0.7, -0.7, 1.0, -1.0, 1.5, -1.5, 2.0, -2.0),
    (315.0, 0.7, -0.7, 1.0, -1.0, 1.5, -1.5, 2.0, -2.0),
    (400.0, 0.7, -0.7, 1.0, -1.0, 1.5, -1.5, 2.0, -2.0),
    (500.0, 0.7, -0.7, 1.0, -1.0, 1.5, -1.5, 2.0, -2.0),
    (630.0, 0.7, -0.7, 1.0, -1.0, 1.5, -1.5, 2.0, -2.0),
    (800.0, 0.7, -0.7, 1.0, -1.0, 1.5, -1.5, 2.0, -2.0),
    (1000.0, 0.7, -0.7, 1.0, -1.0, 1.5, -1.5, 2.0, -2.0),
    (1250.0, 0.7, -0.7, 1.0, -1.0, 1.5, -1.5, 2.5, -2.5),
    (1600.0, 0.7, -0.7, 1.0, -1.0, 2.0, -2.0, 3.0, -3.0),
    (2000.0, 0.7, -0.7, 1.0, -1.0, 2.0, -2.0, 3.0, -3.0),
    (2500.0, 0.7, -0.7, 1.0, -1.0, 2.5, -2.5, 4.0, -4.0),
    (3150.0, 0.7, -0.7, 1.0, -1.0, 2.5, -2.5, 4.5, -4.5),
    (4000.0, 0.7, -0.7, 1.0, -1.0, 3.0, -3.0, 5.0, -5.0),
    (5000.0, 1.0, -1.0, 1.5, -1.5, 3.5, -3.5, 6.0, -6.0),
    (6300.0, 1.0, -1.5, 1.5, -2.0, 4.5, -4.5, 6.0, -6.0),
    (8000.0, 1.0, -2.0, 1.5, -3.0, 5.0, -5.0, 6.0, -6.0),
    (10000.0, 2.0, -3.0, 2.0, -4.0, 5.0, -_INF, 6.0, -_INF),
    (12500.0, 2.0, -3.0, 3.0, -6.0, 5.0, -_INF, 6.0, -_INF),
    (16000.0, 2.0, -3.0, 3.0, -_INF, 5.0, -_INF, 6.0, -_INF),
    (20000.0, 2.0, -3.0, 3.0, -_INF, 5.0, -_INF, 6.0, -_INF),
]

# Per-edition mask spec, in the shape `filters.compliance` uses for the two
# band-filter editions: the ordered classes (strictest -> loosest), the curves
# the edition defines, the tolerance table and the (upper, lower) column index
# of each class within its rows. ``b_from_ansi`` says whether the B weighting
# has to borrow its mask from ANSI S1.4-1983 Table V: IEC 61672-1 dropped B
# and publishes no limits for it, while IEC 651:1979 Table V governs B along
# with every other weighting characteristic.
_WEIGHTING_EDITIONS: dict[str, dict[str, Any]] = {
    "2013": {
        "classes": (1, 2),
        "curves": ("A", "B", "C", "AU", "Z"),
        "table": _WEIGHTING_TABLE3,
        "col": {1: (3, 4), 2: (5, 6)},
        "b_from_ansi": True,
    },
    "1979": {
        "classes": (0, 1, 2, 3),
        "curves": ("A", "B", "C"),
        "table": _IEC651_TABLE5,
        "col": {0: (1, 2), 1: (3, 4), 2: (5, 6), 3: (7, 8)},
        "b_from_ansi": False,
    },
}

# ---------------------------------------------------------------------------
# ANSI S1.4-1983 - historical B weighting (dropped when IEC 61672-1 replaced
# the older sound-level-meter standards).
# ---------------------------------------------------------------------------

# ANSI S1.4-1983 Table IV (standard page 6): random-incidence relative
# response level of the B weighting at the 34 nominal frequencies. The A and
# C columns of Table IV equal IEC 61672-1:2013 Table 3 digit for digit, so
# only the B column is transcribed here. Row = (nominal Hz, B dB).
_ANSI_S14_TABLE4_B: list[tuple[float, float]] = [
    (10.0, -38.2),
    (12.5, -33.2),
    (16.0, -28.5),
    (20.0, -24.2),
    (25.0, -20.4),
    (31.5, -17.1),
    (40.0, -14.2),
    (50.0, -11.6),
    (63.0, -9.3),
    (80.0, -7.4),
    (100.0, -5.6),
    (125.0, -4.2),
    (160.0, -3.0),
    (200.0, -2.0),
    (250.0, -1.3),
    (315.0, -0.8),
    (400.0, -0.5),
    (500.0, -0.3),
    (630.0, -0.1),
    (800.0, 0.0),
    (1000.0, 0.0),
    (1250.0, 0.0),
    (1600.0, 0.0),
    (2000.0, -0.1),
    (2500.0, -0.2),
    (3150.0, -0.4),
    (4000.0, -0.7),
    (5000.0, -1.2),
    (6300.0, -1.9),
    (8000.0, -2.9),
    (10000.0, -4.3),
    (12500.0, -6.1),
    (16000.0, -8.4),
    (20000.0, -11.1),
]

# ANSI S1.4-1983 Table V (standard page 6): tolerance limits on relative
# response levels for Type 1 and Type 2 instruments; the verifier maps them
# to the class 1 / class 2 verdict slots. (The stricter laboratory Type 0
# column lives in tests/reference_data/ and is pinned by the CI
# conformance report.) Row = (nominal Hz, type1 upper, type1 lower,
# type2 upper, type2 lower); a -inf lower limit means upper-only.
# Transcription note, 20 Hz Type 2: the standard prints a bare "+3" there,
# where every one-sided cell of that same column prints "+5, -inf". The cell
# is read as +/-3, because IEC 651:1979 Table V - which agrees with this
# column at all 33 other rows - prints "+/-3" at exactly that cell, so the
# missing bar under the plus sign is a defect of this print and not a national
# deviation. The reading is also the strict one, and it cannot change the
# verdict of the realized B filter: its response at 20 Hz sits 0.05 dB below
# nominal.
_ANSI_S14_TABLE5_12: list[tuple[float, float, float, float, float]] = [
    (10.0, 4.0, -4.0, 5.0, -_INF),
    (12.5, 3.5, -3.5, 5.0, -_INF),
    (16.0, 3.0, -3.0, 5.0, -_INF),
    (20.0, 2.5, -2.5, 3.0, -3.0),
    (25.0, 2.0, -2.0, 3.0, -3.0),
    (31.5, 1.5, -1.5, 3.0, -3.0),
    (40.0, 1.5, -1.5, 2.0, -2.0),
    (50.0, 1.0, -1.0, 2.0, -2.0),
    (63.0, 1.0, -1.0, 2.0, -2.0),
    (80.0, 1.0, -1.0, 2.0, -2.0),
    (100.0, 1.0, -1.0, 1.5, -1.5),
    (125.0, 1.0, -1.0, 1.5, -1.5),
    (160.0, 1.0, -1.0, 1.5, -1.5),
    (200.0, 1.0, -1.0, 1.5, -1.5),
    (250.0, 1.0, -1.0, 1.5, -1.5),
    (315.0, 1.0, -1.0, 1.5, -1.5),
    (400.0, 1.0, -1.0, 1.5, -1.5),
    (500.0, 1.0, -1.0, 1.5, -1.5),
    (630.0, 1.0, -1.0, 1.5, -1.5),
    (800.0, 1.0, -1.0, 1.5, -1.5),
    (1000.0, 1.0, -1.0, 1.5, -1.5),
    (1250.0, 1.0, -1.0, 1.5, -1.5),
    (1600.0, 1.0, -1.0, 2.0, -2.0),
    (2000.0, 1.0, -1.0, 2.0, -2.0),
    (2500.0, 1.0, -1.0, 2.5, -2.5),
    (3150.0, 1.0, -1.0, 2.5, -2.5),
    (4000.0, 1.0, -1.0, 3.0, -3.0),
    (5000.0, 1.5, -1.5, 3.5, -3.5),
    (6300.0, 1.5, -2.0, 4.5, -4.5),
    (8000.0, 1.5, -3.0, 5.0, -5.0),
    (10000.0, 2.0, -4.0, 5.0, -_INF),
    (12500.0, 3.0, -6.0, 5.0, -_INF),
    (16000.0, 3.0, -_INF, 5.0, -_INF),
    (20000.0, 3.0, -_INF, 5.0, -_INF),
]

# ANSI S1.4-1983 Appendix C: the B weighting is the C weighting with one
# extra zero at the origin and one extra real pole at f5 (Formula C2).
_F5 = 158.48932

# ---------------------------------------------------------------------------
# IEC 61012:1990 - AU weighting (audible sound in the presence of ultrasound)
# ---------------------------------------------------------------------------

# IEC 61012:1990 Table 1 (standard page 11): nominal relative response and
# tolerances of the U weighting as a separate filter unit, at the 37 nominal
# frequencies from 10 Hz to 40 kHz. The tolerance is zero at the 1 kHz
# reference frequency (Table 1 note; IEC 651 subclause 3.7) and the -inf
# lower limit at 40 kHz means upper-only. Row = (nominal Hz, U dB, upper
# tolerance, lower tolerance).
_IEC61012_TABLE1: list[tuple[float, float, float, float]] = [
    (10.0, 0.0, 3.0, -3.0),
    (12.5, 0.0, 3.0, -3.0),
    (16.0, 0.0, 3.0, -3.0),
    (20.0, 0.0, 3.0, -3.0),
    (25.0, 0.0, 2.0, -2.0),
    (31.5, 0.0, 1.0, -1.0),
    (40.0, 0.0, 1.0, -1.0),
    (50.0, 0.0, 1.0, -1.0),
    (63.0, 0.0, 1.0, -1.0),
    (80.0, 0.0, 1.0, -1.0),
    (100.0, 0.0, 1.0, -1.0),
    (125.0, 0.0, 1.0, -1.0),
    (160.0, 0.0, 1.0, -1.0),
    (200.0, 0.0, 1.0, -1.0),
    (250.0, 0.0, 1.0, -1.0),
    (315.0, 0.0, 1.0, -1.0),
    (400.0, 0.0, 1.0, -1.0),
    (500.0, 0.0, 1.0, -1.0),
    (630.0, 0.0, 1.0, -1.0),
    (800.0, 0.0, 1.0, -1.0),
    (1000.0, 0.0, 0.0, 0.0),
    (1250.0, 0.0, 1.0, -1.0),
    (1600.0, 0.0, 1.0, -1.0),
    (2000.0, 0.0, 1.0, -1.0),
    (2500.0, 0.0, 1.0, -1.0),
    (3150.0, 0.0, 1.0, -1.0),
    (4000.0, 0.0, 1.0, -1.0),
    (5000.0, 0.0, 1.0, -1.0),
    (6300.0, 0.0, 1.0, -1.0),
    (8000.0, 0.0, 1.0, -1.0),
    (10000.0, 0.0, 1.0, -1.0),
    (12500.0, -2.8, 2.0, -2.0),
    (16000.0, -13.0, 3.0, -3.0),
    (20000.0, -25.3, 3.0, -6.0),
    (25000.0, -37.6, 3.0, -6.0),
    (31500.0, -49.7, 3.0, -10.0),
    (40000.0, -61.8, 3.0, -_INF),
]

# IEC 61012:1990 subclause 2.2: explicit nominal AU values at the three
# frequencies above the last IEC 651 A-weighting row (20 kHz), prescribed
# directly because A + U cannot be summed from tabulated columns there.
_IEC61012_AU_HF = {25000.0: -50.0, 31500.0: -65.4, 40000.0: -81.1}

# IEC 61012:1990 Table 2: pole locations of the U weighting (Hz).
_U_POLES_HZ = np.array(
    [
        -12200.0,
        -12200.0,
        -7850.0 + 8800.0j,
        -7850.0 - 8800.0j,
        -2900.0 + 12150.0j,
        -2900.0 - 12150.0j,
    ]
)

# IEC 61672-1:2013 Annex E pole frequencies of the analytic A/C design goals
# (E.4.1-E.4.8); identical to the constants the WeightingFilter design uses.
_F1 = 20.598997
_F2 = 107.65265
_F3 = 737.86223
_F4 = 12194.217


def _exact_base10(frequencies: np.ndarray) -> np.ndarray:
    r"""Exact base-10 frequencies :math:`1000 \cdot 10^{n/10}`.

    IEC 61672-1:2013 Table 3 NOTE: the tabulated weightings are computed
    at the exact frequencies
    :math:`f = 1000 \cdot 10^{0.1 (n - 30)}`, not at the nominal
    labels (e.g. 15 848.9 Hz behind "16 kHz").
    """
    return np.asarray(
        10.0 ** (np.round(10.0 * np.log10(frequencies)) / 10.0), dtype=np.float64
    )


def _analytic_weighting_db(curve: str, frequencies: np.ndarray) -> np.ndarray:
    """Analytic design-goal weighting, re 1 kHz.

    For A/C(/Z) this evaluates the exact transfer-function magnitudes of IEC
    61672-1:2013 Annex E (E.4.1/E.4.2) at ``frequencies`` and normalizes to
    the 1 kHz value, reproducing every Table 3 design goal after 0.1 dB
    rounding. ``B`` adds the ANSI S1.4-1983 Appendix C Formula (C2) factor to
    the C response, and ``AU`` cascades the A response with the U low-pass
    built from the IEC 61012:1990 Table 2 poles (which reproduces every
    Table 1 nominal value within 0.05 dB).
    """
    f = np.asarray(frequencies, dtype=np.float64)
    if curve == "Z":
        return np.zeros_like(f)

    def _c_gain(x: np.ndarray) -> np.ndarray:
        x2 = x**2
        return np.asarray(
            (_F4**2 * x2) / ((x2 + _F1**2) * (x2 + _F4**2)), dtype=np.float64
        )

    def _a_gain(x: np.ndarray) -> np.ndarray:
        x2 = x**2
        return np.asarray(
            _c_gain(x) * x2 / np.sqrt((x2 + _F2**2) * (x2 + _F3**2)),
            dtype=np.float64,
        )

    def _b_gain(x: np.ndarray) -> np.ndarray:
        # ANSI S1.4-1983 Formula (C2): W_B = 10 lg(K2 f^2/(f^2 + f5^2)) + W_C
        # (the constant K2 cancels in the 1 kHz normalization below).
        return np.asarray(_c_gain(x) * x / np.sqrt(x**2 + _F5**2), dtype=np.float64)

    def _u_gain(x: np.ndarray) -> np.ndarray:
        # Magnitude of the all-pole U weighting from the Table 2 pole
        # coordinates in Hz (the 2*pi scale cancels in the normalization).
        den = np.prod(1j * x[:, None] - _U_POLES_HZ[None, :], axis=1)
        return np.asarray(1.0 / np.abs(den), dtype=np.float64)

    def _au_gain(x: np.ndarray) -> np.ndarray:
        return np.asarray(_a_gain(x) * _u_gain(x), dtype=np.float64)

    gain = {"A": _a_gain, "B": _b_gain, "C": _c_gain, "AU": _au_gain}[curve]
    ref = gain(np.asarray([1000.0]))[0]
    return np.asarray(20.0 * np.log10(gain(f) / ref), dtype=np.float64)


def _edition_spec(edition: str) -> dict[str, Any]:
    """The mask spec of *edition*, refusing an unknown one.

    :raises ValueError: if ``edition`` names no published edition.
    """
    spec = _WEIGHTING_EDITIONS.get(edition)
    if spec is None:
        msg = "edition must be '2013' or '1979'."
        raise ValueError(msg)
    return spec


def _quoted_list(items: tuple[str, ...]) -> str:
    """Render ``("A", "B", "C")`` as ``"'A', 'B' or 'C'"``."""
    quoted = [f"'{item}'" for item in items]
    return f"{', '.join(quoted[:-1])} or {quoted[-1]}"


def _edition_masks(spec: dict[str, Any]) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """One ``(lower, upper)`` acceptance mask per class of the edition.

    Keyed strictest class first, so the verdict functions can take the first
    class that passes and be taking the tightest one.
    """
    table = spec["table"]
    return {
        cls: (
            np.array([row[lo_col] for row in table], dtype=np.float64),
            np.array([row[up_col] for row in table], dtype=np.float64),
        )
        for cls, (up_col, lo_col) in spec["col"].items()
    }


def weighting_class_limits(
    weighting_class: int, *, edition: str = "2013"
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Acceptance limits of one performance class of a weighting standard.

    The limits apply to every weighting the edition defines; they qualify the
    deviation of the measured relative response from the design goal at each
    nominal frequency, not the response itself. Under ``edition="2013"`` they
    come from IEC 61672-1:2013 Table 3 and govern A, C and Z (the B and AU
    masks that ``verify_weighting_class`` uses come from ANSI S1.4-1983
    Table V and IEC 61012:1990 Table 1 instead and are not returned here).
    Under ``edition="1979"`` they come from IEC 651:1979 Table V, whose
    footnote makes one mask govern every weighting characteristic, B included.

    :param weighting_class: Performance class: 1 or 2 for ``edition="2013"``;
        0, 1, 2 or 3 for ``edition="1979"``, where class N is the standard's
        instrument Type N.
    :param edition: ``"2013"`` (IEC 61672-1:2013, classes 1/2) or ``"1979"``
        (IEC 651:1979, which adds the stricter Type 0 and a Type 3).
    :return: Tuple ``(frequencies, lower, upper)`` of the 34 nominal
        frequencies (Hz) and the lower/upper deviation limits in dB. A lower
        limit of ``-inf`` means only the upper limit applies.
    :raises ValueError: if the edition is unknown or does not define the
        requested class.
    """
    spec = _edition_spec(edition)
    if weighting_class not in spec["classes"]:
        msg = (
            f"weighting_class must be one of {spec['classes']} for edition '{edition}'."
        )
        raise ValueError(msg)
    up_col, lo_col = spec["col"][weighting_class]
    table = spec["table"]
    freqs = np.array([row[0] for row in table], dtype=np.float64)
    upper = np.array([row[up_col] for row in table], dtype=np.float64)
    lower = np.array([row[lo_col] for row in table], dtype=np.float64)
    return freqs, lower, upper


def _design_goal_db(curve: str) -> np.ndarray:
    """Design-goal relative response at the 34 nominal frequencies, in dB.

    A and C come from IEC 61672-1:2013 Table 3, B from ANSI S1.4-1983
    Table IV, Z is flat. The same three columns serve the IEC 651:1979
    edition: the A, B and C columns of BS 5969:1981 Table IV (standard
    page 7, read rendered) equal these digit for digit at all 34 rows.
    """
    if curve == "B":
        return np.array([row[1] for row in _ANSI_S14_TABLE4_B], dtype=np.float64)
    col = _WEIGHTING_COL[curve]
    if col is None:
        return np.zeros(len(_WEIGHTING_TABLE3), dtype=np.float64)
    return np.array([row[col] for row in _WEIGHTING_TABLE3], dtype=np.float64)


def _au_design_and_limits() -> tuple[
    np.ndarray, np.ndarray, dict[int, tuple[np.ndarray, np.ndarray]]
]:
    """IEC 61012:1990 Table 1 nominal frequencies, AU design goals and mask.

    The design goal is nominal A + nominal U (with the subclause 2.2 explicit
    values above 20 kHz) and the single Table 1 tolerance set for the filter
    as a separate unit fills both class slots.
    """
    a_design = {row[0]: row[1] for row in _WEIGHTING_TABLE3}
    nominal = np.array([row[0] for row in _IEC61012_TABLE1], dtype=np.float64)
    design = np.array(
        [
            _IEC61012_AU_HF.get(row[0], a_design.get(row[0], 0.0) + row[1])
            for row in _IEC61012_TABLE1
        ],
        dtype=np.float64,
    )
    upper = np.array([row[2] for row in _IEC61012_TABLE1], dtype=np.float64)
    lower = np.array([row[3] for row in _IEC61012_TABLE1], dtype=np.float64)
    return nominal, design, {1: (lower, upper), 2: (lower, upper)}


def _curve_design_and_limits(
    curve: str, spec: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, dict[int, tuple[np.ndarray, np.ndarray]]]:
    """Nominal frequencies, design goals and one acceptance mask per class.

    The mask is the edition's own tolerance table, with one exception: under
    the 2013 edition B has no mask of its own there (IEC 61672-1 dropped the
    weighting), so it borrows ANSI S1.4-1983 Table V, whose Types 1 and 2 fill
    the class 1 / class 2 slots. Under the 1979 edition B needs no exception -
    the Table V footnote makes that one mask govern every weighting
    characteristic.
    """
    if curve == "AU":
        return _au_design_and_limits()
    nominal = np.array([row[0] for row in _WEIGHTING_TABLE3], dtype=np.float64)
    design = _design_goal_db(curve)
    if curve == "B" and spec["b_from_ansi"]:
        upper1 = np.array([row[1] for row in _ANSI_S14_TABLE5_12], dtype=np.float64)
        lower1 = np.array([row[2] for row in _ANSI_S14_TABLE5_12], dtype=np.float64)
        upper2 = np.array([row[3] for row in _ANSI_S14_TABLE5_12], dtype=np.float64)
        lower2 = np.array([row[4] for row in _ANSI_S14_TABLE5_12], dtype=np.float64)
        return nominal, design, {1: (lower1, upper1), 2: (lower2, upper2)}
    return nominal, design, _edition_masks(spec)


def _weighting_response_db(wf: WeightingFilter, frequencies: np.ndarray) -> np.ndarray:
    """Relative steady-state response of *wf* in dB, normalized to 1 kHz.

    Measured over the whole path the signal takes (see
    :func:`_runtime_frequency_response`), which is now the second-order
    sections and nothing else, so that a verdict describes the filter the
    caller runs.
    """
    if wf.curve == "Z" or wf.sos.size == 0:
        return np.zeros_like(frequencies)
    worn = np.concatenate([frequencies, [1000.0]])
    h = _runtime_frequency_response(wf, worn)
    gain_db = 20.0 * np.log10(np.abs(h) + np.finfo(float).eps)
    return np.asarray(gain_db[:-1] - gain_db[-1], dtype=np.float64)  # relative to 1 kHz


def _band_class(margins: dict[int, float]) -> int | None:
    """Narrowest class the band still meets, or ``None`` if it meets none.

    :param margins: Distance to the nearer limit of each class, in decibels,
        keyed strictest class first.
    :return: The first class whose margin is not negative, or ``None``.
    """
    return next((cls for cls, margin in margins.items() if margin >= 0), None)


def _weighting_band_verdicts(
    freqs_nom: np.ndarray,
    deviation: np.ndarray,
    masks: dict[int, tuple[np.ndarray, np.ndarray]],
) -> list[dict[str, Any]]:
    """Per-band class verdicts against the edition's acceptance limits.

    Margin = distance to the nearer limit; a -inf lower limit makes that side
    non-binding (its term is +inf), i.e. an upper-only limit.
    """
    bands: list[dict[str, Any]] = []
    for i, fm in enumerate(freqs_nom):
        margins = {
            cls: float(min(upper[i] - deviation[i], deviation[i] - lower[i]))
            for cls, (lower, upper) in masks.items()
        }
        band: dict[str, Any] = {
            "freq": float(fm),
            "class": _band_class(margins),
            "deviation_db": float(deviation[i]),
        }
        band.update({f"margin_class{cls}_db": m for cls, m in margins.items()})
        bands.append(band)
    return bands


def _between_nominals_sweep(
    wf: WeightingFilter,
    freqs_exact: np.ndarray,
    masks: dict[int, tuple[np.ndarray, np.ndarray]],
    sweep_points: int,
) -> dict[str, float]:
    """Subclause 5.5.7 sweep between adjacent exact nominal frequencies.

    The acceptance limits between two adjacent nominal frequencies are the
    larger of the two adjacent tabulated limits; the design goal there is the
    analytic Annex E response.
    """
    grid = np.geomspace(freqs_exact[0], freqs_exact[-1], sweep_points)
    sweep_dev = _weighting_response_db(wf, grid) - _analytic_weighting_db(
        wf.curve, grid
    )
    seg = np.clip(
        np.searchsorted(freqs_exact, grid, side="right") - 1, 0, freqs_exact.size - 2
    )
    sweep_margins: dict[int, np.ndarray] = {}
    for cls, (lower, upper) in masks.items():
        up = np.maximum(upper[seg], upper[seg + 1])
        lo = np.minimum(lower[seg], lower[seg + 1])
        sweep_margins[cls] = np.minimum(up - sweep_dev, sweep_dev - lo)
    # The worst frequency is read off the strictest class, the first key.
    worst = int(np.argmin(next(iter(sweep_margins.values()))))
    between: dict[str, float] = {"worst_freq": float(grid[worst])}
    between.update(
        {f"margin_class{cls}_db": float(np.min(m)) for cls, m in sweep_margins.items()}
    )
    return between


def _overall_class(
    bands: list[dict[str, Any]],
    between: dict[str, float],
    classes_ordered: tuple[int, ...],
) -> int | None:
    """Strictest class met at every nominal frequency *and* across the sweep.

    Both readings have to hold: a filter that clears every tabulated row but
    dips outside the mask between two of them has not met the class.
    """
    for cls in classes_ordered:
        key = f"margin_class{cls}_db"
        if all(band[key] >= 0.0 for band in bands) and between[key] >= 0.0:
            return cls
    return None


def verify_weighting_class(
    wf: WeightingFilter, *, sweep_points: int = 4096, edition: str = "2013"
) -> dict[str, Any]:
    r"""Verify a frequency-weighting filter against its standard's tolerances.

    ``A``/``C``/``Z`` are checked against IEC 61672-1:2013 Table 3 (classes 1
    and 2). The historical ``B`` weighting is checked against ANSI S1.4-1983:
    Table IV design goals with the Table V tolerance limits, whose instrument
    Types 1 and 2 fill the class 1 / class 2 verdict slots (an
    ``overall_class`` of 1 then reads "ANSI S1.4-1983 Type 1"). ``AU`` is
    checked against IEC 61012:1990: design goals are nominal A + nominal U
    (Table 1, plus the subclause 2.2 explicit AU values at 25/31.5/40 kHz)
    with the single Table 1 tolerance set for the filter as a separate unit,
    so both class slots carry the same margin and ``overall_class`` is 1
    (complies) or ``None``. ``G`` is not supported here (ISO 7196 defines one
    +/-1 dB instrumentation tolerance, no class structure; the CI conformance
    report pins it), nor is ``D`` (the tolerance tables of the withdrawn
    IEC 537 did not survive it; the conformance report pins the D response
    against its published transfer function and tabulated curve).

    ``edition="1979"`` swaps in the tolerance table of the superseded
    IEC 651:1979 instead, whose **Table V** publishes the laboratory-grade
    **Type 0** mask that IEC 61672-1 has no equivalent for, and three further
    types. Class N is then the standard's instrument Type N, so an
    ``overall_class`` of 0 reads "IEC 651:1979 Type 0". That edition covers
    ``A``, ``B`` and ``C`` (the weightings of subclause 3.2, whose Table IV
    design goals equal the ones used above digit for digit): its Table V
    footnote makes one mask govern every weighting characteristic, so ``B``
    does not borrow the ANSI limits there. It is a genuinely different mask,
    not a rename - Type 0 is +2/-3 dB at 16 kHz and at 20 kHz, where class 1
    is +2.5/-16 and +3/-inf, so an error class 1 cannot see is visible under
    Type 0.

    The filter's relative response (normalized to its 1 kHz gain) is evaluated
    at the *exact* base-10 frequency behind each nominal label below
    the Nyquist frequency (IEC 61672-1 Table 3 NOTE: the design goals are
    computed at :math:`f = 1000 \cdot 10^{0.1 (n - 30)}`, e.g.
    15 848.9 Hz for
    "16 kHz"; IEC 61672-3:2013 subclause 13.3 tests the deviation at the same
    exact frequencies, and IEC 61012 Table 1 lists the same exact
    frequencies). The deviation from the design-goal weighting is checked
    against the two acceptance masks.

    A dense logarithmic sweep between the checked frequencies additionally
    enforces IEC 61672-1 subclause 5.5.7: at any frequency between two
    adjacent nominal frequencies, the deviation of the response from the
    analytic design goal (Annex E for A/C/Z, the ANSI S1.4-1983 Appendix C
    formulas for B, the A response cascaded with the IEC 61012 Table 2 poles
    for AU) must stay within the *larger* of the two adjacent limits. Without
    it a resonance or notch between the nominal frequencies would go
    unnoticed (for B, for AU and under the 1979 edition, whose tables are
    tabulated at the nominal frequencies and nowhere else, the sweep is
    applied as the analogous engineering check). Both the per-frequency
    verdicts and the sweep must pass for
    ``overall_class``. The sweep samples ``sweep_points`` grid frequencies; a
    violation narrower than the grid spacing could in principle fall between
    samples, so raise ``sweep_points`` for higher-Q suspects (the verdict
    attests the sampled grid, not a continuous proof).

    The response is taken over the whole path a signal travels through
    :meth:`~phonometry.WeightingFilter.filter`, which is one cascade of
    second-order sections at the input rate for every curve and in both
    stateful and single-shot use. It used to be more than that: the sections
    were reached through an interpolation and a decimation stage whose
    anti-alias filter had its transition band on the input Nyquist frequency
    and dominated the response above roughly ``0.9 * fs / 2``, so a verdict
    read from the sections alone attested a filter the user never ran. That is
    why the verdict is measured through :func:`_runtime_frequency_response`
    rather than through ``sosfreqz`` here, and it stays that way so the next
    stage added to the path cannot go unmodelled. The ``Z`` weighting is a
    flat bypass and always complies.

    When rows that carry a *finite lower* acceptance limit fall at or
    above the Nyquist frequency (e.g. the 8-16 kHz class 1 rows of a 16 kHz
    sampled system, or the 25-40 kHz AU rows of a 48 kHz one), they cannot be
    checked and ``range_limited`` is ``True``: the returned class then
    attests conformance over the checked frequencies only, not conformance
    over the standard's full frequency range.

    :param wf: The weighting filter to verify (``A``, ``B``, ``C``, ``AU``
        or ``Z``; ``A``, ``B`` or ``C`` for ``edition="1979"``).
    :param sweep_points: Number of points of the 5.5.7 between-nominals sweep
        (>= 64).
    :param edition: ``"2013"`` (IEC 61672-1:2013, classes 1/2) or ``"1979"``
        (IEC 651:1979, Types 0/1/2/3 offered as classes 0-3).
    :return: Dict with ``overall_class`` (the strictest class of the edition
        that every checked frequency and the sweep meet, or ``None``),
        ``range_limited`` (see above), ``bands``: a list of ``{"freq",
        "class", "deviation_db", "margin_class<c>_db"}`` for each class ``c``
        of the edition, where ``freq`` is the nominal label and a positive
        margin means the limits are met with that much room, and
        ``between_nominals``: ``{"worst_freq", "margin_class<c>_db"}`` for the
        sweep.
    :raises ValueError: if the edition is unknown, the edition does not define
        the filter's curve, or ``sweep_points`` is below 64.
    """
    spec = _edition_spec(edition)
    if wf.curve not in spec["curves"]:
        msg = (
            f"Weighting curve must be {_quoted_list(spec['curves'])} "
            f"for edition '{edition}'."
        )
        raise ValueError(msg)
    if sweep_points < _MIN_SWEEP_POINTS:
        msg = "'sweep_points' must be at least 64."
        raise ValueError(msg)

    nyquist = wf.fs / 2.0
    nominal, design_all, masks_all = _curve_design_and_limits(wf.curve, spec)
    exact = _exact_base10(nominal)
    in_range = exact < nyquist

    # Any row beyond Nyquist cannot be demonstrated (its acceptance
    # limits and the adjacent between-nominal interval go unchecked), so the
    # verdict is then range-limited, not full-range conformance.
    dropped = ~in_range
    range_limited = bool(np.any(dropped))

    freqs_nom = nominal[in_range]
    freqs_exact = exact[in_range]
    design = design_all[in_range]
    response = _weighting_response_db(wf, freqs_exact)
    deviation = response - design

    masks = {
        cls: (lower[in_range], upper[in_range])
        for cls, (lower, upper) in masks_all.items()
    }

    bands = _weighting_band_verdicts(freqs_nom, deviation, masks)

    if not bands:
        return {
            "overall_class": None,
            "range_limited": range_limited,
            "bands": [],
            "between_nominals": None,
        }

    between = _between_nominals_sweep(wf, freqs_exact, masks, sweep_points)

    return {
        "overall_class": _overall_class(bands, between, tuple(masks)),
        "range_limited": range_limited,
        "bands": bands,
        "between_nominals": between,
    }
