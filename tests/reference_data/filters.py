#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Frequency weightings and band filters: nominal responses and their masks.

One subject: what a filter is required to do to a pure tone, and how far a
given class of instrument may stray from it. The weightings live together
because they are specified the same way and read from the same kind of
table - A, C and Z (IEC 61672-1 Table 3), the historical B (ANSI S1.4
Table IV), the infrasound G (ISO 7196 Table 2), the ultrasound U
(IEC 61012 Tables 1 and 2) and the aircraft D of the withdrawn IEC 537 -
and so are the octave and one-third-octave band filters of IEC 61260.

The tolerance columns travel with the nominal ones. A transcription that
separates them cannot be checked: the nominal response is only meaningful
against the acceptance limits its own edition prints beside it.
"""

from __future__ import annotations

import math

INF = math.inf

# ---------------------------------------------------------------------------
# IEC 61672-1:2013 Table 3 - frequency weightings and class-1 acceptance
# limits (standard page 22). Z weighting is 0.0 dB at every frequency.
# Row = (nominal_freq_Hz, A_dB, C_dB, class1_upper_dB, class1_lower_dB).
# ---------------------------------------------------------------------------
IEC61672_TABLE3: list[tuple[float, float, float, float, float]] = [
    (10, -70.4, -14.3, 3.0, -INF),
    (12.5, -63.4, -11.2, 2.5, -INF),
    (16, -56.7, -8.5, 2.0, -4.0),
    (20, -50.5, -6.2, 2.0, -2.0),
    (25, -44.7, -4.4, 2.0, -1.5),
    (31.5, -39.4, -3.0, 1.5, -1.5),
    (40, -34.6, -2.0, 1.0, -1.0),
    (50, -30.2, -1.3, 1.0, -1.0),
    (63, -26.2, -0.8, 1.0, -1.0),
    (80, -22.5, -0.5, 1.0, -1.0),
    (100, -19.1, -0.3, 1.0, -1.0),
    (125, -16.1, -0.2, 1.0, -1.0),
    (160, -13.4, -0.1, 1.0, -1.0),
    (200, -10.9, 0.0, 1.0, -1.0),
    (250, -8.6, 0.0, 1.0, -1.0),
    (315, -6.6, 0.0, 1.0, -1.0),
    (400, -4.8, 0.0, 1.0, -1.0),
    (500, -3.2, 0.0, 1.0, -1.0),
    (630, -1.9, 0.0, 1.0, -1.0),
    (800, -0.8, 0.0, 1.0, -1.0),
    (1000, 0.0, 0.0, 0.7, -0.7),
    (1250, 0.6, 0.0, 1.0, -1.0),
    (1600, 1.0, -0.1, 1.0, -1.0),
    (2000, 1.2, -0.2, 1.0, -1.0),
    (2500, 1.3, -0.3, 1.0, -1.0),
    (3150, 1.2, -0.5, 1.0, -1.0),
    (4000, 1.0, -0.8, 1.0, -1.0),
    (5000, 0.5, -1.3, 1.5, -1.5),
    (6300, -0.1, -2.0, 1.5, -2.0),
    (8000, -1.1, -3.0, 1.5, -2.5),
    (10000, -2.5, -4.4, 2.0, -3.0),
    (12500, -4.3, -6.2, 2.0, -5.0),
    (16000, -6.6, -8.5, 2.5, -16.0),
    (20000, -9.3, -11.2, 3.0, -INF),
]

# ---------------------------------------------------------------------------
# IEC 61260:1995 / EN 61260:1995 Table 1 == ANSI S1.11-2004 Table 1 (octave-band
# limits on relative attenuation, dB). Independently transcribed and verified
# digit-for-digit between the two standards, which agree exactly. This edition
# adds the stricter class 0 (dropped by IEC 61260-1:2014). Rows give the octave
# breakpoint exponent x of Omega = G**x (G = 10**0.3) and the limits per class.
# The pass-band minimum is a constant per class; the max is interpolated across
# the pass-band breakpoints and the min across the stop-band breakpoints.
# ---------------------------------------------------------------------------
IEC61260_1995_PASSBAND_MIN = {0: -0.15, 1: -0.3, 2: -0.5}
# (exponent, class 0 max, class 1 max, class 2 max)
IEC61260_1995_PASSBAND_MAX: list[tuple[float, float, float, float]] = [
    (0.0, 0.15, 0.3, 0.5),
    (0.125, 0.2, 0.4, 0.6),
    (0.25, 0.4, 0.6, 0.8),
    (0.375, 1.1, 1.3, 1.6),
    (0.5, 4.5, 5.0, 5.5),
]
# (exponent, class 0 min, class 1 min, class 2 min)
IEC61260_1995_STOPBAND_MIN: list[tuple[float, float, float, float]] = [
    (0.5, 2.3, 2.0, 1.6),
    (1.0, 18.0, 17.5, 16.5),
    (2.0, 42.5, 42.0, 41.0),
    (3.0, 62.0, 61.0, 55.0),
    (4.0, 75.0, 70.0, 60.0),
]

# IEC 61260-1:2014 Table F.1 (informative annex F): normalized frequency
# breakpoints of the one-third-octave-band (b = 3) acceptance masks, i.e. the
# Formula (9) mapping of the octave-band breakpoints G**x, printed to five
# decimals with their reciprocals. The best published oracle for the
# Formula (9)/(10) breakpoint mapping. Rows: exponent x -> (Omega, 1/Omega).
IEC61260_TABLE_F1: dict[float, tuple[float, float]] = {
    1 / 8: (1.02667, 0.97402),
    1 / 4: (1.05575, 0.94719),
    3 / 8: (1.08746, 0.91958),
    1 / 2: (1.12202, 0.89125),
    1.0: (1.29437, 0.77257),
    2.0: (1.88173, 0.53143),
    3.0: (3.05365, 0.32748),
    4.0: (5.39195, 0.18546),
}
# IEC 61260-1:2014 E.3.4 worked rounding examples (nominal frequencies for
# b = 24): 41,567 Hz -> 41,6 Hz (MSD 4: three significant figures) and
# 8 785,2 Hz -> 8 800 Hz (MSD 8: two significant figures).
IEC61260_E34_EXAMPLES = [(41.567, 41.6), (8785.2, 8800.0)]

# ---------------------------------------------------------------------------
# ISO 7196:1995 Table 2 - nominal G-weighting response at one-third-octave
# frequencies (standard page 2). Row = (freq_Hz, dB). Annex A.3 gives the
# instrumentation tolerance of +/- 1 dB from 1 Hz to 20 Hz.
# ---------------------------------------------------------------------------
ISO7196_TABLE2: list[tuple[float, float]] = [
    (0.25, -88.0),
    (0.315, -80.0),
    (0.4, -72.1),
    (0.5, -64.3),
    (0.63, -56.6),
    (0.8, -49.5),
    (1.00, -43.0),
    (1.25, -37.5),
    (1.6, -32.6),
    (2.0, -28.3),
    (2.5, -24.1),
    (3.15, -20.0),
    (4.0, -16.0),
    (5.0, -12.0),
    (6.3, -8.0),
    (8.0, -4.0),
    (10.0, 0.0),
    (12.5, 4.0),
    (16.0, 7.7),
    (20.0, 9.0),
    (25.0, 3.7),
    (31.5, -4.0),
    (40.0, -12.0),
    (50.0, -20.0),
    (63.0, -28.0),
    (80.0, -36.0),
    (100.0, -44.0),
    (125.0, -52.0),
    (160.0, -60.0),
    (200.0, -68.0),
    (250.0, -76.0),
    (315.0, -84.0),
]
ISO7196_G_TOLERANCE_DB = 1.0

# ---------------------------------------------------------------------------
# ANSI S1.4-1983 Table IV (standard page 6) - random-incidence relative
# response level of the historical B weighting at the 34 nominal
# frequencies. The A and C columns of Table IV equal IEC 61672-1:2013
# Table 3 digit for digit, so only the B column is transcribed.
# Row = (freq_Hz, B_dB).
# ---------------------------------------------------------------------------
ANSIS14_TABLE4_B: list[tuple[float, float]] = [
    (10, -38.2),
    (12.5, -33.2),
    (16, -28.5),
    (20, -24.2),
    (25, -20.4),
    (31.5, -17.1),
    (40, -14.2),
    (50, -11.6),
    (63, -9.3),
    (80, -7.4),
    (100, -5.6),
    (125, -4.2),
    (160, -3.0),
    (200, -2.0),
    (250, -1.3),
    (315, -0.8),
    (400, -0.5),
    (500, -0.3),
    (630, -0.1),
    (800, 0.0),
    (1000, 0.0),
    (1250, 0.0),
    (1600, 0.0),
    (2000, -0.1),
    (2500, -0.2),
    (3150, -0.4),
    (4000, -0.7),
    (5000, -1.2),
    (6300, -1.9),
    (8000, -2.9),
    (10000, -4.3),
    (12500, -6.1),
    (16000, -8.4),
    (20000, -11.1),
]

# ANSI S1.4-1983 Table V (standard page 6) - tolerance limits on relative
# response levels for Type 0 (laboratory), Type 1 (precision) and Type 2
# (general purpose) instruments; they apply to every weighting. A lower
# limit of -inf means only the upper limit applies.
# Transcription note, 20 Hz Type 2: the standard prints a bare "+3"; read
# as +3/upper-only (like the surrounding upper-only rows), with +/-3 a
# plausible alternative. The realized B response there is only 0.05 dB
# below nominal, so the reading cannot change any verdict.
# Row = (freq_Hz, t0_up, t0_lo, t1_up, t1_lo, t2_up, t2_lo).
ANSIS14_TABLE5: list[tuple[float, float, float, float, float, float, float]] = [
    (10, 2.0, -5.0, 4.0, -4.0, 5.0, -INF),
    (12.5, 2.0, -4.0, 3.5, -3.5, 5.0, -INF),
    (16, 2.0, -3.0, 3.0, -3.0, 5.0, -INF),
    (20, 2.0, -2.0, 2.5, -2.5, 3.0, -INF),
    (25, 1.5, -1.5, 2.0, -2.0, 3.0, -3.0),
    (31.5, 1.0, -1.0, 1.5, -1.5, 3.0, -3.0),
    (40, 1.0, -1.0, 1.5, -1.5, 2.0, -2.0),
    (50, 1.0, -1.0, 1.0, -1.0, 2.0, -2.0),
    (63, 1.0, -1.0, 1.0, -1.0, 2.0, -2.0),
    (80, 1.0, -1.0, 1.0, -1.0, 2.0, -2.0),
    (100, 0.7, -0.7, 1.0, -1.0, 1.5, -1.5),
    (125, 0.7, -0.7, 1.0, -1.0, 1.5, -1.5),
    (160, 0.7, -0.7, 1.0, -1.0, 1.5, -1.5),
    (200, 0.7, -0.7, 1.0, -1.0, 1.5, -1.5),
    (250, 0.7, -0.7, 1.0, -1.0, 1.5, -1.5),
    (315, 0.7, -0.7, 1.0, -1.0, 1.5, -1.5),
    (400, 0.7, -0.7, 1.0, -1.0, 1.5, -1.5),
    (500, 0.7, -0.7, 1.0, -1.0, 1.5, -1.5),
    (630, 0.7, -0.7, 1.0, -1.0, 1.5, -1.5),
    (800, 0.7, -0.7, 1.0, -1.0, 1.5, -1.5),
    (1000, 0.7, -0.7, 1.0, -1.0, 1.5, -1.5),
    (1250, 0.7, -0.7, 1.0, -1.0, 1.5, -1.5),
    (1600, 0.7, -0.7, 1.0, -1.0, 2.0, -2.0),
    (2000, 0.7, -0.7, 1.0, -1.0, 2.0, -2.0),
    (2500, 0.7, -0.7, 1.0, -1.0, 2.5, -2.5),
    (3150, 0.7, -0.7, 1.0, -1.0, 2.5, -2.5),
    (4000, 0.7, -0.7, 1.0, -1.0, 3.0, -3.0),
    (5000, 1.0, -1.0, 1.5, -1.5, 3.5, -3.5),
    (6300, 1.0, -1.5, 1.5, -2.0, 4.5, -4.5),
    (8000, 1.0, -2.0, 1.5, -3.0, 5.0, -5.0),
    (10000, 2.0, -3.0, 2.0, -4.0, 5.0, -INF),
    (12500, 2.0, -3.0, 3.0, -6.0, 5.0, -INF),
    (16000, 2.0, -3.0, 3.0, -INF, 5.0, -INF),
    (20000, 2.0, -3.0, 3.0, -INF, 5.0, -INF),
]

# ANSI S1.4-1983 Appendix C: analytic constants of the B weighting,
# W_B = 10 lg(K2 f^2 / (f^2 + f5^2)) + W_C (Formula C2).
ANSIS14_F5 = 158.48932
ANSIS14_K2 = 1.025119

# ---------------------------------------------------------------------------
# IEC 61012:1990 Table 1 (standard page 11) - nominal relative response and
# tolerances of the U weighting as a separate filter unit (10 Hz - 40 kHz).
# The tolerance is zero at the 1 kHz reference frequency (Table 1 note;
# IEC 651 subclause 3.7); the -inf lower limit at 40 kHz means upper-only.
# Row = (freq_Hz, U_dB, upper_dB, lower_dB).
# ---------------------------------------------------------------------------
IEC61012_TABLE1: list[tuple[float, float, float, float]] = [
    (10, 0.0, 3.0, -3.0),
    (12.5, 0.0, 3.0, -3.0),
    (16, 0.0, 3.0, -3.0),
    (20, 0.0, 3.0, -3.0),
    (25, 0.0, 2.0, -2.0),
    (31.5, 0.0, 1.0, -1.0),
    (40, 0.0, 1.0, -1.0),
    (50, 0.0, 1.0, -1.0),
    (63, 0.0, 1.0, -1.0),
    (80, 0.0, 1.0, -1.0),
    (100, 0.0, 1.0, -1.0),
    (125, 0.0, 1.0, -1.0),
    (160, 0.0, 1.0, -1.0),
    (200, 0.0, 1.0, -1.0),
    (250, 0.0, 1.0, -1.0),
    (315, 0.0, 1.0, -1.0),
    (400, 0.0, 1.0, -1.0),
    (500, 0.0, 1.0, -1.0),
    (630, 0.0, 1.0, -1.0),
    (800, 0.0, 1.0, -1.0),
    (1000, 0.0, 0.0, 0.0),
    (1250, 0.0, 1.0, -1.0),
    (1600, 0.0, 1.0, -1.0),
    (2000, 0.0, 1.0, -1.0),
    (2500, 0.0, 1.0, -1.0),
    (3150, 0.0, 1.0, -1.0),
    (4000, 0.0, 1.0, -1.0),
    (5000, 0.0, 1.0, -1.0),
    (6300, 0.0, 1.0, -1.0),
    (8000, 0.0, 1.0, -1.0),
    (10000, 0.0, 1.0, -1.0),
    (12500, -2.8, 2.0, -2.0),
    (16000, -13.0, 3.0, -3.0),
    (20000, -25.3, 3.0, -6.0),
    (25000, -37.6, 3.0, -6.0),
    (31500, -49.7, 3.0, -10.0),
    (40000, -61.8, 3.0, -INF),
]

# IEC 61012:1990 Table 2 - pole locations of the U weighting, in Hz.
IEC61012_TABLE2_POLES_HZ: list[tuple[float, float]] = [
    (-12200.0, 0.0),
    (-12200.0, 0.0),
    (-7850.0, 8800.0),
    (-7850.0, -8800.0),
    (-2900.0, 12150.0),
    (-2900.0, -12150.0),
]

# IEC 61012:1990 subclause 2.2 - explicit nominal AU values at the three
# frequencies above the last IEC 651 A-weighting row; elsewhere the nominal
# AU response is the sum of the nominal A and U responses.
IEC61012_AU_HF: dict[float, float] = {25000: -50.0, 31500: -65.4, 40000: -81.1}

# ---------------------------------------------------------------------------
# IEC 537:1976 (withdrawn) D weighting - published one-third-octave curve as
# republished in the NASA Handbook of Aircraft Noise Metrics (NASA CR-3406,
# 1981, Table SLD-I, which cites IEC 537:1976). Row = (freq_Hz, D_dB).
# The rational transfer function reproduces every row within 0.1 dB except
# 1600 Hz (0.15 dB) and 2500 Hz (0.28 dB); those two cells appear to round a
# different source curve, so the pinning test carries a wider tolerance
# there.
# ---------------------------------------------------------------------------
IEC537_NASA_TABLE_SLD1: list[tuple[float, float]] = [
    (50, -12.8),
    (63, -10.9),
    (80, -9.0),
    (100, -7.2),
    (125, -5.5),
    (160, -4.0),
    (200, -2.6),
    (250, -1.6),
    (315, -0.8),
    (400, -0.4),
    (500, -0.3),
    (630, -0.5),
    (800, -0.6),
    (1000, 0.0),
    (1250, 2.0),
    (1600, 4.9),
    (2000, 7.9),
    (2500, 10.6),
    (3150, 11.5),
    (4000, 11.1),
    (5000, 9.6),
    (6300, 7.6),
    (8000, 5.5),
    (10000, 3.4),
]

# librosa's D_weighting closed-form constants (librosa/core/convert.py,
# ISC license): an independent frequency-domain implementation of the same
# IEC 537 curve, used as a cross-check oracle. The magnitude in dB is
# 20*(lg f - lg c0 + 0.5*(lg((c1^2 - f^2)^2 + c2^2 f^2)
#     - lg((c3^2 - f^2)^2 + c4^2 f^2) - lg(c5^2 + f^2) - lg(c6^2 + f^2))).
LIBROSA_D_WEIGHTING_CONSTS: list[float] = [
    8.3046305e-3,
    1018.7,
    1039.6,
    3136.5,
    3424.0,
    282.7,
    1160.0,
]
