#  Copyright (c) 2026. Jose M. Requena-Plens
"""Shared normative reference data (single source of truth).

Tables transcribed verbatim from the published standards. Both the test
suite (``tests/test_*.py``) and the CI conformance report
(``scripts/conformance_report.py``) import these constants, so the report's
expected values can never drift from what the tests assert. Test modules
either import a constant directly where they assert it, or - where the
oracle lives inside a larger inline table (e.g. the ISO 1999 Annex D or
ANSI S12.2 Table D.1 transcriptions) - pin the inline copy to the shared
constant with an explicit consistency assertion. The PR-B building-acoustics
and PR-F human-vibration oracles are additionally pinned to their published
values by dedicated tests in ``tests/test_conformance_report.py``
(``test_building_reference_data_matches_published_oracles`` and
``test_human_vibration_reference_data_matches_oracles``).

This module is deliberately dependency-free (stdlib only) so it can be
imported in the ``pr-comment`` CI job, which installs the runtime
requirements but not ``pytest``.
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
    (0.25, -88.0), (0.315, -80.0), (0.4, -72.1),
    (0.5, -64.3), (0.63, -56.6), (0.8, -49.5),
    (1.00, -43.0), (1.25, -37.5), (1.6, -32.6),
    (2.0, -28.3), (2.5, -24.1), (3.15, -20.0),
    (4.0, -16.0), (5.0, -12.0), (6.3, -8.0),
    (8.0, -4.0), (10.0, 0.0), (12.5, 4.0),
    (16.0, 7.7), (20.0, 9.0), (25.0, 3.7),
    (31.5, -4.0), (40.0, -12.0), (50.0, -20.0),
    (63.0, -28.0), (80.0, -36.0), (100.0, -44.0),
    (125.0, -52.0), (160.0, -60.0), (200.0, -68.0),
    (250.0, -76.0), (315.0, -84.0),
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
    (10, -38.2), (12.5, -33.2), (16, -28.5), (20, -24.2),
    (25, -20.4), (31.5, -17.1), (40, -14.2), (50, -11.6),
    (63, -9.3), (80, -7.4), (100, -5.6), (125, -4.2),
    (160, -3.0), (200, -2.0), (250, -1.3), (315, -0.8),
    (400, -0.5), (500, -0.3), (630, -0.1), (800, 0.0),
    (1000, 0.0), (1250, 0.0), (1600, 0.0), (2000, -0.1),
    (2500, -0.2), (3150, -0.4), (4000, -0.7), (5000, -1.2),
    (6300, -1.9), (8000, -2.9), (10000, -4.3), (12500, -6.1),
    (16000, -8.4), (20000, -11.1),
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
    (10, 0.0, 3.0, -3.0), (12.5, 0.0, 3.0, -3.0), (16, 0.0, 3.0, -3.0),
    (20, 0.0, 3.0, -3.0), (25, 0.0, 2.0, -2.0), (31.5, 0.0, 1.0, -1.0),
    (40, 0.0, 1.0, -1.0), (50, 0.0, 1.0, -1.0), (63, 0.0, 1.0, -1.0),
    (80, 0.0, 1.0, -1.0), (100, 0.0, 1.0, -1.0), (125, 0.0, 1.0, -1.0),
    (160, 0.0, 1.0, -1.0), (200, 0.0, 1.0, -1.0), (250, 0.0, 1.0, -1.0),
    (315, 0.0, 1.0, -1.0), (400, 0.0, 1.0, -1.0), (500, 0.0, 1.0, -1.0),
    (630, 0.0, 1.0, -1.0), (800, 0.0, 1.0, -1.0), (1000, 0.0, 0.0, 0.0),
    (1250, 0.0, 1.0, -1.0), (1600, 0.0, 1.0, -1.0), (2000, 0.0, 1.0, -1.0),
    (2500, 0.0, 1.0, -1.0), (3150, 0.0, 1.0, -1.0), (4000, 0.0, 1.0, -1.0),
    (5000, 0.0, 1.0, -1.0), (6300, 0.0, 1.0, -1.0), (8000, 0.0, 1.0, -1.0),
    (10000, 0.0, 1.0, -1.0), (12500, -2.8, 2.0, -2.0),
    (16000, -13.0, 3.0, -3.0), (20000, -25.3, 3.0, -6.0),
    (25000, -37.6, 3.0, -6.0), (31500, -49.7, 3.0, -10.0),
    (40000, -61.8, 3.0, -INF),
]

# IEC 61012:1990 Table 2 - pole locations of the U weighting, in Hz.
IEC61012_TABLE2_POLES_HZ: list[tuple[float, float]] = [
    (-12200.0, 0.0), (-12200.0, 0.0),
    (-7850.0, 8800.0), (-7850.0, -8800.0),
    (-2900.0, 12150.0), (-2900.0, -12150.0),
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
    (50, -12.8), (63, -10.9), (80, -9.0), (100, -7.2),
    (125, -5.5), (160, -4.0), (200, -2.6), (250, -1.6),
    (315, -0.8), (400, -0.4), (500, -0.3), (630, -0.5),
    (800, -0.6), (1000, 0.0), (1250, 2.0), (1600, 4.9),
    (2000, 7.9), (2500, 10.6), (3150, 11.5), (4000, 11.1),
    (5000, 9.6), (6300, 7.6), (8000, 5.5), (10000, 3.4),
]

# librosa's D_weighting closed-form constants (librosa/core/convert.py,
# ISC license): an independent frequency-domain implementation of the same
# IEC 537 curve, used as a cross-check oracle. The magnitude in dB is
# 20*(lg f - lg c0 + 0.5*(lg((c1^2 - f^2)^2 + c2^2 f^2)
#     - lg((c3^2 - f^2)^2 + c4^2 f^2) - lg(c5^2 + f^2) - lg(c6^2 + f^2))).
LIBROSA_D_WEIGHTING_CONSTS: list[float] = [
    8.3046305e-3, 1018.7, 1039.6, 3136.5, 3424.0, 282.7, 1160.0,
]

# ---------------------------------------------------------------------------
# ISO 717-1 Annex C, Table C.1 - measured airborne sound reduction index R
# (100-3150 Hz, one-third-octave). The worked example gives
# Rw(C;Ctr) = 30(-2;-3) dB with an unfavourable-deviation sum of 31,8 dB.
# ---------------------------------------------------------------------------
ISO717_1_ANNEX_C_R: list[float] = [
    20.4, 16.3, 17.7, 22.6, 22.4, 22.7, 24.8, 26.6,
    28.0, 30.5, 31.8, 32.5, 33.4, 33.0, 31.0, 25.5,
]
ISO717_1_ANNEX_C_EXPECTED = {
    "rw": 30,
    "c": -2,
    "ctr": -3,
    "unfavourable_sum": 31.8,
}
# ISO 717-1:2020 Annex C, Table C.2 - the same element measured over the
# enlarged range 50-5000 Hz (21 bands). The worked example states
# Rw(C;Ctr;C50-5000;Ctr,50-5000) = 30 (-2; -3; -2; -4) dB, with the printed
# intermediate sums -10 lg = 28,212 (spectrum No. 1) / 26,355 (No. 2).
ISO717_1_ANNEX_C2_R_50_5000: list[float] = [
    18.7, 19.2, 20.0, *ISO717_1_ANNEX_C_R, 26.8, 29.2,
]
ISO717_1_ANNEX_C2_EXPECTED = {
    "rw": 30,
    "c": -2,
    "ctr": -3,
    "c_50_5000": -2,
    "ctr_50_5000": -4,
}

# ---------------------------------------------------------------------------
# ISO 226:2023 Table B.1 - normal equal-loudness-level contours. Row =
# (loudness_level_phon, frequency_Hz, sound_pressure_level_dB). Annex B is
# rounded to 0.1 dB. The definitional identity is at 1 kHz (SPL == phon); we
# anchor the conformance check at a NON-1 kHz point so it exercises the
# contour formula (Table 1 alpha_f/L_U/T_f) rather than the trivial identity.
# ---------------------------------------------------------------------------
ISO226_2023_TABLE_B1_ANCHOR: tuple[float, float, float] = (60.0, 100.0, 78.5)

# ---------------------------------------------------------------------------
# Psychoacoustics "block-A" calibration anchors. Each is the single reference
# value its standard tabulates for the stated calibration signal.
# ---------------------------------------------------------------------------
# ECMA-418-2:2025 (Sottek Hearing Model). Loudness: Clause 5.1.8 defines the
# calibration constant c_N = 0.0211964 so a 1 kHz / 40 dB SPL tone yields
# 1 sone_HMS via the Clause 8 method (c_N adjustable within 0.25 %).
ECMA418_2_LOUDNESS_1KHZ_40DB_SONE = 1.0
ECMA418_2_LOUDNESS_C_N = 0.0211964
# Tonality: Clause 6.2.8 defines c_T = 2.8758615 so a 1 kHz / 40 dB tone
# yields 1 tu_HMS (c_T adjustable within 0.25 %).
ECMA418_2_TONALITY_1KHZ_40DB_TU = 1.0
ECMA418_2_TONALITY_C_T = 2.8758615
# Decision thresholds the standard states as verbatim constants: a signal is
# audible when its total basis loudness exceeds 0.01 sone_HMS (Clause 5.1.9,
# dz = 0.5 sum of Formula 25); a tonality is prominent when the single value
# T exceeds 0.4 tu_HMS (Clause 6.3); a roughness is prominent when the single
# value R exceeds 0.2 asper (Clause 7.2). Annexes A/B/C are graphical only,
# so beyond the three calibration points these thresholds are the standard's
# only further numeric anchors.
ECMA418_2_AUDIBILITY_THRESHOLD_SONE = 0.01
ECMA418_2_PROMINENT_TONALITY_TU = 0.4
ECMA418_2_PROMINENT_ROUGHNESS_ASPER = 0.2
# Roughness: Clause 7 defines the reference as a 1 kHz carrier, 100 % AM at
# 70 Hz, with "a sound pressure level of 60 dB" -- the OVERALL RMS level of
# the modulated signal (not the carrier-alone level) -> 1.0 asper. With the
# tabulated c_R = 0.0180685 (not reverse-fit) and the Clause 5.1.2 fade-in
# applied, the chain reproduces this anchor to 0.9999 asper.
ECMA418_2_ROUGHNESS_1KHZ_70HZ_60DB_ASPER = 1.0
ECMA418_2_ROUGHNESS_C_R = 0.0180685
# Fluctuation strength: Clause 9 defines the reference as a 1 kHz carrier,
# 100 % AM at 4 Hz, with "a sound pressure level of 60 dB" -- the OVERALL RMS
# level, same convention as roughness -> 1.0 vacil_HMS. With the tabulated
# c_F = 0.003840572 (not reverse-fit) the chain reproduces this anchor to
# 0.9931 vacil_HMS for a 5 s signal (0.9957 at 8 s, converged 0.9958 by 12 s).
# A fluctuation strength is prominent when the single value F exceeds
# 0.2 vacil_HMS (Clause 9.2).
ECMA418_2_FLUCTUATION_1KHZ_4HZ_60DB_VACIL = 1.0
ECMA418_2_FLUCTUATION_C_F = 0.003840572
ECMA418_2_PROMINENT_FLUCTUATION_VACIL = 0.2
# ISO 532-2:2017 (Moore-Glasberg, stationary). Clause 3.17 / Annex B.1: the
# sone is defined so a 1 kHz / 40 dB SPL tone (binaural, free field) is
# 1.000 sone / 40 phon, following from the tabulated C = 0.0617 sone/Cam.
ISO532_2_ANCHOR_1KHZ_40DB_SONE = 1.0
ISO532_2_C = 0.0617
# ISO 532-3:2023 (Moore-Glasberg-Schlittenlacher, time-varying). Annex C.1:
# a steady 1 kHz / 40 dB SPL tone reaches a peak long-term loudness of
# 1.0 sone / 40 phon (the spectral calibration is fixed to this anchor).
ISO532_3_ANCHOR_1KHZ_40DB_SONE = 1.0

# ---------------------------------------------------------------------------
# ISO 16283-3:2016 field facade sound insulation. Clause 3.12 defines the
# apparent sound reduction index of the element (loudspeaker) method as
# R'45deg = L1,s - L2 + 10 lg(S/A) - 1,5. Choosing the specimen area S equal to
# the equivalent absorption area A (A = 0,16 V/T = 0,16 * 62,5 / 1,0 = 10 m2)
# collapses the 10 lg(S/A) coupling term, isolating the -1,5 dB oblique-
# incidence correction exactly: R' = 60 - 20 - 1,5 = 38,5 dB. (Road-traffic
# method R'tr,s uses -3 dB instead; Clause 3.13.)
# ---------------------------------------------------------------------------
ISO16283_3_R45_LOUDSPEAKER_CORRECTION_DB = 1.5
ISO16283_3_R45_SURFACE_LEVEL_DB = 60.0
ISO16283_3_R45_RECEIVE_LEVEL_DB = 20.0
ISO16283_3_R45_AREA_M2 = 10.0
ISO16283_3_R45_VOLUME_M3 = 62.5
ISO16283_3_R45_REVERB_TIME_S = 1.0
ISO16283_3_R45_EXPECTED_DB = 38.5

# ---------------------------------------------------------------------------
# ISO 10140-2:2010 laboratory airborne sound reduction index R (Formula (2)):
# R = L1 - L2 + 10 lg(S/A), A = 0,16 V/T. The reference-curve construction lays
# R exactly on the ISO 717-1 Table 3 shape (100-3150 Hz) by choosing S = A
# (S = 10 m2, A = 0,16 * 50 / 0,8 = 10 m2), so R = L1 - L2 = the reference. The
# 32 dB unfavourable-deviation allowance then permits a 2 dB upward shift of the
# reference (32 dB / 16 bands), giving Rw = curve@500 Hz (52) + 2 = 54 dB - the
# analytic +2-shift anchor (mirrors tests/test_lab_insulation.py).
# ---------------------------------------------------------------------------
ISO10140_2_REF_AIRBORNE_R: list[float] = [
    33, 36, 39, 42, 45, 48, 51, 52, 53, 54, 55, 56, 56, 56, 56, 56,
]
ISO10140_2_REF_AIRBORNE_RW = 54

# ---------------------------------------------------------------------------
# EN 12354-1:2000 Annex H.3 airborne prediction worked example. A separating
# element of Rw = 57 dB and area S = 11,5 m2 is flanked by four elements; each
# contributes an Ff/Fd/Df triplet (12 flanking paths), which with the direct
# Dd path make 13 transmission paths. Energy summation (Formula (26)) gives
# R'w = 52,2 dB -> 52 dB. Row = (label, Rw_flanking, KFf, KFd=KDf, coupling
# length lf). Mirrors tests/test_building_prediction.py (_annex_h_paths).
# ---------------------------------------------------------------------------
EN12354_1_ANNEX_H3_R_DIRECT = 57.0
EN12354_1_ANNEX_H3_SEPARATING_AREA = 11.5
EN12354_1_ANNEX_H3_ELEMENTS: list[tuple[str, float, float, float, float]] = [
    ("floor", 49.0, 12.4, 8.9, 4.5),
    ("ceiling", 46.0, 14.4, 9.2, 4.5),
    ("facade", 42.0, 12.6, 6.7, 2.55),
    ("intwall", 33.0, 33.5, 15.7, 2.55),
]
EN12354_1_ANNEX_H3_NUM_PATHS = 13
EN12354_1_ANNEX_H3_RPRIME_W = 52  # 52,2 dB rounds to 52
# All twelve printed flanking-path values of the H.3 results table (dB): for
# each element the standard prints RFd = RDf (the "R_1d"/"R_D1" rows are the
# same value) and RFf. Keyed by element label -> (R_Ff, R_Fd = R_Df).
EN12354_1_ANNEX_H3_PATH_RW: dict[str, tuple[float, float]] = {
    "floor": (65.5, 66.0),
    "ceiling": (64.5, 64.8),
    "facade": (61.1, 62.7),
    "intwall": (73.0, 67.2),
}
# Formula (5b) closure of both H.3 examples: V = 50 m3, Ss = 11,5 m2. The
# standard prints DnT,w = 52,2 + 10 lg[50/(3 x 11,5)] = 53,8 ~ 54 dB and (with
# the floating floor) 52,7 + 1,6 = 54,3 ~ 54 dB; the exact Formula (5b) factor
# 0,32 V/Ss gives 53,6 / 54,1 dB - same integer ratings.
EN12354_1_ANNEX_H3_VOLUME = 50.0
EN12354_1_ANNEX_H3_DNT_W = 54
EN12354_1_ANNEX_H3_DNT_W_PRINTED = 53.8   # with the standard's V/(3 S) rounding
EN12354_1_ANNEX_H3_DNT_W_SECOND = 54      # second example: 54,3 ~ 54 dB

# ---------------------------------------------------------------------------
# EN 12354-2:2000 Annex E.3 impact prediction worked example. A concrete floor
# of mass per area m' = 322 kg/m2 has an equivalent normalized impact level
# Ln,w,eq = 164 - 35 lg(m') ~ 76 dB (Formula for heavy floors). With a floating-
# floor improvement ΔLw = 33 dB and a flanking correction K = 2 dB (Table 1;
# separating 322 -> row 300, flanking mean 145 -> col 150), the predicted
# apparent normalized impact level is L'n,w = 76 - 33 + 2 = 45 dB (Formula 21).
# ---------------------------------------------------------------------------
EN12354_2_ANNEX_E3_MASS = 322.0
EN12354_2_ANNEX_E3_FLANKING_MEAN_MASS = 145.0
EN12354_2_ANNEX_E3_DELTA_LW = 33.0
EN12354_2_ANNEX_E3_K = 2
EN12354_2_ANNEX_E3_LPRIME_N_W = 45

# ---------------------------------------------------------------------------
# EN 12354-3:2000 Annex F worked example - facade airborne insulation. A 11,3 m2
# facade (V = 50 m3, flat reflecting so ΔLfs = 0) of four elements, octave bands
# 125-2000 Hz. Rows: (name, area_m2, R_or_Dne_dB[5]); the air inlet is a small
# element entered as Dn,e (already length-corrected to the installed 3 m). The
# apparent R' = -10 lg Σ τe (Formula 10) and D2m,nT = R' + 10 lg(V/(6 T0 S))
# (Formula 13, T0 = 0,5 s) give the single numbers R'tr,s,w = 31 (Ctr = -3) and
# D2m,nT,w = 33 dB (Table F.1.3). NOTE: the standard's own printed per-element
# partial indices sum to R' = 35,8 / 38,0 dB at 1 k / 2 k, whereas its R' row
# prints 35,4 / 37,5 - an internal rounding inconsistency in the 2000 example;
# the low bands (125-500 Hz) and every single-number rating are exact.
# NOTE 2: the printed D2m,nT row of Annex F equals R' + 1,5 dB, whereas
# Formula (13) with V = 50 m3, S = 11,3 m2, T0 = 0,5 s gives
# 10 lg(50/(6*0,5*11,3)) = +1,69 dB - another internal inconsistency of the
# 2000 example. The module implements the formula; the single-number oracle
# D2m,nT,w = 33 reproduces either way.
# ---------------------------------------------------------------------------
EN12354_3_ANNEX_F_BANDS = (125.0, 250.0, 500.0, 1000.0, 2000.0)
EN12354_3_ANNEX_F_AREA = 11.3
EN12354_3_ANNEX_F_VOLUME = 50.0
EN12354_3_ANNEX_F_ELEMENTS: list[tuple[str, float, list[float]]] = [
    ("wall", 6.0, [41.0, 46.0, 52.0, 58.0, 64.0]),
    ("window2", 4.5, [23.0, 22.0, 30.0, 36.0, 37.0]),
    ("window3", 0.5, [24.0, 27.0, 30.0, 33.0, 30.0]),
]
EN12354_3_ANNEX_F_INLET_DNE = [28.0, 23.0, 25.0, 38.0, 44.0]  # small element, Dn,e
EN12354_3_ANNEX_F_RPRIME_LOW = [24.4, 21.5, 24.9]  # 125/250/500 Hz, digit-exact
EN12354_3_ANNEX_F_RTRS_W = 31
EN12354_3_ANNEX_F_CTR = -3
EN12354_3_ANNEX_F_D2MNT_W = 33

# ---------------------------------------------------------------------------
# EN 12354-4:2000 Annex G worked example - sound radiated to the outside. An
# industrial building (Lp,in in Table G.1, octave 63-8000 Hz, Cd = -5 dB from
# Annex B, apparent R' limited to 40 dB per the Table G.3 footnote). Side 1 is a
# 10x60 m wall (100 mm light concrete, R in Table G.2) with a 6x4 m industrial
# door, segmented into 10x20 = 200 m2 panels; LW = Lp,in + Cd - R' + 10 lg(S/S0)
# (Formula 2). The exterior level uses the simplified Annex E attenuation of a
# finite radiating side: Table G.9 gives A'tot and Lp at reception points in
# front of side 1 (60x10 m) and side 4 (100x10 m). NOTE: the standard's own R'
# rows above 500 Hz are internally inconsistent with its Table G.2 inputs (e.g.
# the wall-only R' prints 36 dB at 1 k while the concrete input is 39 dB, which
# no 40 dB cap can produce); the low bands, the LW relation and the whole Annex E
# propagation reproduce exactly.
# ---------------------------------------------------------------------------
EN12354_4_ANNEX_G_BANDS = (63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0)
EN12354_4_ANNEX_G_LP_IN = [70.0, 74.0, 76.0, 72.0, 70.0, 67.0, 62.0, 57.0]
EN12354_4_ANNEX_G_CD = -5.0
EN12354_4_ANNEX_G_RPRIME_CAP = 40.0
EN12354_4_ANNEX_G_CONCRETE_R = [32.0, 36.0, 36.0, 33.0, 39.0, 49.0, 57.0, 63.0]
EN12354_4_ANNEX_G_DOOR_R = [21.0, 23.0, 28.0, 30.0, 30.0, 30.0, 30.0, 30.0]
EN12354_4_ANNEX_G_SEGMENT_AREA = 200.0
EN12354_4_ANNEX_G_DOOR_AREA = 24.0  # 6 x 4 m
EN12354_4_ANNEX_G_SIDE1_RPRIME_LOW = [28.2, 30.8, 33.9]  # 63/125/250 Hz
EN12354_4_ANNEX_G_SIDE1_LW_LOW = [59.8, 61.2]  # 63/125 Hz, digit-exact
# Table G.9 attenuation and exterior level (side W x H, distance d) -> A'tot, Lp.
# Lp uses the side's A-weighted power level (Table G.8): side1 62,9 / side4 72,9.
EN12354_4_ANNEX_G_SIDE1_LWA = 62.9
EN12354_4_ANNEX_G_SIDE4_LWA = 72.9
EN12354_4_ANNEX_G_ATTENUATION: list[tuple[float, float, float, float]] = [
    # (width, height, distance, A'tot_dB)
    (60.0, 10.0, 5.0, 26.3),
    (60.0, 10.0, 25.0, 34.4),
    (100.0, 10.0, 5.0, 28.3),
    (100.0, 10.0, 25.0, 35.6),
]
EN12354_4_ANNEX_G_LP_SIDE1_D5 = 36.6
EN12354_4_ANNEX_G_LP_SIDE4_D25 = 37.3
# Remaining Table G.9 cells: Lp = LWA - A'tot for the other two reception
# points (side 1 at 25 m; side 4 at 5 m).
EN12354_4_ANNEX_G_LP_SIDE1_D25 = 28.5  # 62,9 - 34,4
EN12354_4_ANNEX_G_LP_SIDE4_D5 = 44.6   # 72,9 - 28,3

# ---------------------------------------------------------------------------
# EN 12354-3:2000 Annex C, Figure C.2 - facade-shape level difference ΔLfs
# (dB). Sample cells transcribed from the figure (verified against the page
# render; the 2017 DIN EN ISO 12354-3 Tabelle C.1 tabulates identical
# values). Row = (shape, line_of_sight_m, alpha_w, dLfs_dB).
# ---------------------------------------------------------------------------
EN12354_3_ANNEX_C_DLFS: list[tuple[str, float, float, float]] = [
    ("plane_facade", 1.0, 0.3, 0.0),
    ("plane_facade", 3.0, 0.9, 0.0),
    ("gallery_2", 1.0, 0.3, -1.0),
    ("gallery_3", 2.0, 0.9, 2.0),
    ("gallery_4", 2.0, 0.6, 1.0),
    ("gallery_5", 3.0, 0.6, 4.0),
    ("balcony_6", 1.0, 0.3, -1.0),
    ("balcony_6", 2.0, 0.6, 1.0),
    ("balcony_6", 3.0, 0.9, 3.0),
    ("balcony_7", 2.0, 0.9, 4.0),
    ("balcony_8", 3.0, 0.3, 1.0),
    ("terrace_open", 2.0, 0.6, 4.0),
    ("terrace_closed", 1.0, 0.3, 3.0),
    ("terrace_closed", 2.0, 0.6, 6.0),
    ("terrace_closed", 3.0, 0.9, 7.0),
]

# ---------------------------------------------------------------------------
# ISO 12999-1:2020 measurement uncertainty. Table 2 (Clause 7.2) tabulates the
# airborne one-third-octave standard uncertainty; situation A at 1000 Hz is
# 1,8 dB (digit-exact oracle). Table 8 (Clause 8) gives the two-sided 95 %
# coverage factor k = 1,96, so the expanded uncertainty is U = k u = 1,96 u
# exactly; for Rw in situation A (u = 1,2 dB, Table 3) this is U = 2,352 dB.
# ---------------------------------------------------------------------------
ISO12999_1_TABLE2_AIRBORNE_A_1000HZ = 1.8
ISO12999_1_COVERAGE_K_95 = 1.96
ISO12999_1_RW_A_STANDARD_UNCERTAINTY = 1.2

# ISO 12999-1:2020 Annex B worked example. Table B.1 gives a measured Ri
# spectrum (21 one-third-octave bands 50-5000 Hz) with the Table 2
# situation-A uncertainties ui; Table B.2 the resulting one-decimal single
# numbers (0,1 dB reference-curve shift per B.2) and their uncertainties:
# correlated per Formulae (B.3)-(B.6), uncorrelated per Formula (B.2).
ISO12999_1_ANNEX_B_FREQ: list[float] = [
    50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500,
    630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000,
]
ISO12999_1_ANNEX_B_RI: list[float] = [
    39.5, 40.3, 41.6, 43.1, 43.3, 43.1, 42.5, 44.7, 48.0, 50.5, 53.2,
    55.9, 58.1, 60.0, 62.2, 63.7, 65.4, 66.8, 68.4, 68.8, 65.1,
]
ISO12999_1_ANNEX_B_UI: list[float] = [
    6.8, 4.6, 3.8, 3.0, 2.7, 2.4, 2.1, 1.8, 1.8, 1.8, 1.8,
    1.8, 1.8, 1.8, 1.8, 1.8, 1.8, 1.9, 2.0, 2.4, 2.8,
]
ISO12999_1_ANNEX_B_RW = 57.4               # one-decimal Rw (B.2)
ISO12999_1_ANNEX_B_RW_C50_5000 = 56.4      # one-decimal Rw + C50-5000
ISO12999_1_ANNEX_B_RW_CTR50_5000 = 51.1    # one-decimal Rw + Ctr,50-5000
ISO12999_1_ANNEX_B_U_CORR_RW = 1.9         # u(Rw), correlated (B.6)
ISO12999_1_ANNEX_B_U_CORR_C = 2.1          # u(Rw+C50-5000), correlated (B.5)
ISO12999_1_ANNEX_B_U_CORR_CTR = 2.6        # u(Rw+Ctr,50-5000), correlated
ISO12999_1_ANNEX_B_U_UNCORR_C = 0.6        # u(Rw+C50-5000), uncorrelated (B.2)
ISO12999_1_ANNEX_B_U_UNCORR_CTR = 0.8      # u(Rw+Ctr,50-5000), uncorrelated

# ---------------------------------------------------------------------------
# ISO 9613-1:1993 Table 1 - pure-tone atmospheric-absorption attenuation
# coefficient (dB/km) at one standard atmosphere (101,325 kPa). Rows are the
# ISO 266 preferred one-third-octave frequencies but the values are computed at
# the EXACT midband frequencies fm = 1000*10^(k/10) (Note 5). Each entry is
# (temperature_degC, relative_humidity_percent, preferred_freq_Hz, alpha_dB_km).
# Digit-exact transcription against the FULL standard text (37 pp): sub-tables
# 1(a)-1(p) span -20 degC to +50 degC in 5 degC steps. The Eq. (3)-(5)
# implementation reproduces every point to < 0,4 % (limited only by the
# 3-significant-figure printed values), well inside the standard's own +/- 10 %
# claimed accuracy (clause 7.1). The second block (20-50 degC) was added once
# the full Table 1 became available, extending the earlier -20..+15 degC oracle.
# ---------------------------------------------------------------------------
ISO9613_1_TABLE1: list[tuple[float, float, float, float]] = [
    (-20.0, 10.0, 50.0, 0.589),
    (-20.0, 50.0, 1000.0, 9.14),
    (-20.0, 70.0, 8000.0, 27.8),
    (-20.0, 100.0, 10000.0, 47.0),
    (0.0, 10.0, 50.0, 0.302),
    (0.0, 50.0, 1000.0, 6.83),
    (0.0, 20.0, 2000.0, 34.6),
    (0.0, 100.0, 6300.0, 88.0),
    (5.0, 50.0, 1000.0, 5.08),
    (10.0, 70.0, 1000.0, 3.66),
    (10.0, 50.0, 4000.0, 46.7),
    (15.0, 50.0, 1000.0, 4.16),
    (15.0, 100.0, 10000.0, 105.0),
    # Extension from the full-text sub-tables 1(i)-1(p) (20 degC .. 50 degC).
    (20.0, 50.0, 1000.0, 4.66),
    (20.0, 50.0, 2000.0, 9.86),
    (25.0, 50.0, 1000.0, 5.68),
    (25.0, 100.0, 2000.0, 11.4),
    (30.0, 50.0, 1000.0, 7.03),
    (30.0, 50.0, 4000.0, 24.5),
    (35.0, 50.0, 1000.0, 8.43),
    (40.0, 50.0, 1000.0, 9.52),
    (45.0, 50.0, 800.0, 7.16),
    (50.0, 50.0, 2000.0, 24.8),
]
# Two representative grid points for the conformance registry (middle + corner).
ISO9613_1_TABLE1_MID = (10.0, 70.0, 1000.0, 3.66)  # dB/km
ISO9613_1_TABLE1_CORNER = (0.0, 20.0, 2000.0, 34.6)  # dB/km

# ---------------------------------------------------------------------------
# ISO 9613-2:1996 outdoor sound propagation — closed-form oracles. The general
# method (clause 7) sums independent physical terms, each with an exact limit:
#   * Eq. (7) geometrical divergence Adiv = 20 lg(d/d0) + 11 = 51,0 dB at 100 m.
#   * Table 3 ground functions have exact on-ground (h = 0), fully-developed
#     (dp -> inf) limits: b'(0) = 1,5 + 8,6 = 10,1 (250 Hz). With porous ground
#     both sides (Gs = Gr = 1) and hs = hr = 0 the source and receiver regions
#     each add (-1,5 + b'(0)), so Agr(250 Hz) = 2*(-1,5 + 10,1) = 17,2 dB.
#   * Clause 7.4 caps the diffraction term Dz at 20 dB (single edge) and 25 dB
#     (double edge); a deep-shadow geometry saturates to those caps exactly.
# ---------------------------------------------------------------------------
ISO9613_2_ADIV_100M = 51.0  # Eq. (7): 20 lg(100/1) + 11
ISO9613_2_GROUND_BPRIME_ZERO = 10.1  # Table 3 b'(h=0, dp->inf) = 1,5 + 8,6
ISO9613_2_GROUND_AGR_250_POROUS = 17.2  # 2*(-1,5 + 10,1), hs=hr=0, Gs=Gr=1
ISO9613_2_BARRIER_CAP_SINGLE = 20.0  # clause 7.4 single-diffraction limit, dB
ISO9613_2_BARRIER_CAP_DOUBLE = 25.0  # clause 7.4 double-diffraction limit, dB

# ISO 9613-2:1996 Table 2: atmospheric attenuation coefficient alpha (dB/km)
# for the eight nominal octave bands, six atmospheric conditions, transcribed
# from the printed table (page 5). The values come from ISO 9613-1 at the
# EXACT base-10 midband frequencies; the recomputation agrees with every cell
# to half a unit of the last printed digit, except 15 degC / 80 % / 1 kHz
# where the print gives 4,1 while the exact-midband recomputation yields
# 4,151 (rounds to 4,2) -- a print-side rounding artifact of ~0.05 dB/km.
ISO9613_2_TABLE2_BANDS = (63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0)
ISO9613_2_TABLE2 = {
    # (temperature degC, relative humidity %): alpha per band, dB/km
    (10.0, 70.0): (0.1, 0.4, 1.0, 1.9, 3.7, 9.7, 32.8, 117.0),
    (20.0, 70.0): (0.1, 0.3, 1.1, 2.8, 5.0, 9.0, 22.9, 76.6),
    (30.0, 70.0): (0.1, 0.3, 1.0, 3.1, 7.4, 12.7, 23.1, 59.3),
    (15.0, 20.0): (0.3, 0.6, 1.2, 2.7, 8.2, 28.2, 88.8, 202.0),
    (15.0, 50.0): (0.1, 0.5, 1.2, 2.2, 4.2, 10.8, 36.2, 129.0),
    (15.0, 80.0): (0.1, 0.3, 1.1, 2.4, 4.1, 8.3, 23.7, 82.8),
}

# ---------------------------------------------------------------------------
# ISO 9612:2009 occupational noise exposure — the three normative worked
# examples (Annexes D/E/F), reproduced digit-for-digit by the test suite. Each
# stores the raw measured levels/durations and the standard's reported LEX,8h
# and expanded uncertainty U (k = 1,65, one-sided 95 %). Annex D is the
# task-based welder day; its case (a) omits the task-duration uncertainty
# (U = 2,7 dB), case (b) includes it (U = 3,2 dB). Annexes E (job-based, 18
# workers) and F (full-day forklift drivers) use the Table C.4 sampling budget.
# Task tuples are (samples, duration_hours, duration_range) so the conformance
# report can rebuild the Task objects (Task is not importable here — this module
# is stdlib-only). Mirrors tests/test_occupational_exposure.py.
# ---------------------------------------------------------------------------
ISO9612_ANNEX_D_TASKS: tuple[tuple, ...] = (
    ((70.0,), 1.5, None),
    ((80.1, 82.2, 79.6), 5.0, (4.0, 6.0)),
    ((86.5, 92.4, 89.3, 93.2, 87.8, 86.2), 1.5, (1.0, 2.0)),
)
ISO9612_ANNEX_D_LEX_8H = 84.3
ISO9612_ANNEX_D_U = 2.7  # case (a): task-duration uncertainty omitted
ISO9612_ANNEX_E_SAMPLES: tuple[float, ...] = (88.1, 86.1, 89.7, 86.5, 91.1, 86.7)
ISO9612_ANNEX_E_TE_HOURS = 7.5
ISO9612_ANNEX_E_LEX_8H = 88.1
ISO9612_ANNEX_E_U = 3.8
ISO9612_ANNEX_F_SAMPLES: tuple[float, ...] = (88.0, 91.9, 87.6, 90.4, 89.0, 88.4)
ISO9612_ANNEX_F_TE_HOURS = 9.25
ISO9612_ANNEX_F_LEX_8H = 90.1
ISO9612_ANNEX_F_U = 3.4

# ---------------------------------------------------------------------------
# ISO 11654:1997 rating of sound absorption — the two normative worked examples
# of Annex A. Both use the same practical-coefficient spectrum except at 500 Hz;
# A.1 gives alpha_w = 0,60 with no shape indicator, A.2 (500 Hz raised to 1,00)
# gives alpha_w = 0,60(M). Bands are 250/500/1000/2000/4000 Hz. Mirrors
# tests/test_absorption_rating.py.
# ---------------------------------------------------------------------------
ISO11654_ANNEX_A1_ALPHA_P: tuple[float, ...] = (0.35, 0.70, 0.65, 0.60, 0.55)
ISO11654_ANNEX_A1_ALPHA_W = 0.60
ISO11654_ANNEX_A1_CLASS = "C"
ISO11654_ANNEX_A1_INDICATOR = ""
ISO11654_ANNEX_A2_ALPHA_P: tuple[float, ...] = (0.35, 1.00, 0.65, 0.60, 0.55)
ISO11654_ANNEX_A2_ALPHA_W = 0.60
ISO11654_ANNEX_A2_INDICATOR = "M"

# ---------------------------------------------------------------------------
# ISO 9053-2:2020 alternating-method airflow resistance — the Annex A.3 worked
# example of the effective ratio of specific heats. A closed cylinder 100 mm x
# 100 mm gives V = 7,854e-4 m3 and S = 0,0471 m2; with the IEC 61094-2:2009 air
# properties at 23 C and f = 2 Hz the standard prints b = 1,83e-3 m and the
# heat-conduction-corrected kappa' = kappa*0,978 = 1,370. Mirrors
# tests/test_airflow_resistance.py.
# ---------------------------------------------------------------------------
ISO9053_2_ANNEX_A_SURFACE = 0.0471  # S (m2)
ISO9053_2_ANNEX_A_VOLUME = 7.854e-4  # V (m3)
ISO9053_2_ANNEX_A_FREQUENCY = 2.0  # f (Hz)
ISO9053_2_ANNEX_A_BOUNDARY_LAYER = 1.83e-3  # b (m)
ISO9053_2_ANNEX_A_KAPPA_PRIME = 1.370  # kappa' = kappa*0,978

# ---------------------------------------------------------------------------
# ISO 10534-1:1996 standing-wave-ratio method — closed-form physics oracle from
# Eqs (13)/(14)/(9): a standing-wave ratio s = 3 gives |r| = (s-1)/(s+1) = 0,5
# and absorption alpha = 1 - |r|^2 = 0,75.
# ---------------------------------------------------------------------------
ISO10534_1_SWR = 3.0
ISO10534_1_REFLECTION_MAGNITUDE = 0.5
ISO10534_1_ABSORPTION = 0.75

# ---------------------------------------------------------------------------
# ISO 17497-1:2004 random-incidence scattering coefficient. Eq (2) fixes the
# reference speed of sound c = 343,2 m/s at 20 C. The synthetic worked chain
# (T1..T4 = 8,0/6,0/7,5/5,0 s, V/S from V = 200 m3, S = 10 m2) exercises the
# Sabine absorptions Eq (1)/(4) and the scattering Eq (5). Mirrors
# tests/test_scattering_diffusion.py.
# ---------------------------------------------------------------------------
ISO17497_1_SPEED_OF_SOUND_20C = 343.2  # Eq (2) reference condition (m/s)
ISO17497_1_CHAIN_V = 200.0  # chamber volume V (m3)
ISO17497_1_CHAIN_S = 10.0  # sample area S (m2)
ISO17497_1_CHAIN_C = 343.2  # speed of sound used throughout (m/s)
ISO17497_1_CHAIN_T: tuple[float, float, float, float] = (8.0, 6.0, 7.5, 5.0)
ISO17497_1_CHAIN_ALPHA_S = 0.1342754467754468  # random-incidence absorption
ISO17497_1_CHAIN_ALPHA_SPEC = 0.21484071484071485  # specular absorption
ISO17497_1_CHAIN_SCATTERING = 0.09306108711505018  # s = (a_spec-a_s)/(1-a_s)
# Annex A.5 combined uncertainty of the scattering coefficient. For
# a_spec = 0,6, a_s = 0,3 with u(a_spec) = 0,02 and u(a_s) = 0,01 the
# error-propagation form gives u(s) = 0,0297.
ISO17497_1_A5_ALPHA_SPEC = 0.6
ISO17497_1_A5_ALPHA_S = 0.3
ISO17497_1_A5_U_ALPHA_SPEC = 0.02
ISO17497_1_A5_U_ALPHA_S = 0.01
ISO17497_1_A5_U_SCATTERING = 0.0297147342419613  # combined u(s)

# ---------------------------------------------------------------------------
# ISO 17497-2:2012 directional diffusion coefficient d_theta, Formula (5).
# Arithmetic oracle on the standard single-plane semicircular receiver arc
# (37 receivers at 5 deg spacing, -90..+90 deg): the committed levels are
# GENERATED BY THIS LIBRARY'S OWN Fraunhofer far-field phase-grating model
# (materials/diffuser_design.py, Cox & D'Antonio Eq. (5.8)/(9.32)) for a
# published diffuser geometry; they are model output, not third-party data.
# The independent external anchor against published third-party BEM values is
# the Cox & D'Antonio Appendix B comparison further below.
#
# Geometry (published): an N = 7 quadratic-residue diffuser, 6 periods,
# 3.6 m total width, 0.2 m maximum well depth - the "N = 7 QRD, 6 periods,
# 0.2 m deep" row of Cox & D'Antonio, "Acoustic Absorbers and Diffusers",
# 3rd ed. (2017), Appendix B section 7 (Schroeder diffusers, 3.6 m wide).
# The commercial single-plane QRD measured by Hargreaves, Cox, Lam & D'Antonio,
# J. Acoust. Soc. Am. 108(4), 1710-1720 (2000), Table I is the same diffuser
# family (N = 7, 0.2 m maximum well depth). Period 3.6/6 = 0.6 m, well width
# 3.6/42 m; well depths d_n = s_n lambda0 / (2 N) (Eq. (10.3), s_n = n^2 mod 7)
# with design frequency f0 = 490 Hz (c = 343 m/s), so the deepest well
# (s_max = 4) is exactly 0.2 m: depths = (0, 0.05, 0.2, 0.1, 0.1, 0.2, 0.05) m.
# Prediction at 1000 Hz, normal incidence. The flat reference is the model's
# own normalisation pathway: the same 3.6 m footprint with all wells at zero
# depth (Hargreaves et al. used a 0.57 m plane panel; the Fraunhofer model
# normalises against the equal-footprint flat panel instead, as
# predicted_diffusion_spectrum does).
#
# Levels are peak-referenced (0 dB at the maximum; Formula (5) is invariant
# to a constant level shift), rounded to 1e-3 dB; the coefficients below are
# recomputed from the rounded committed levels, so the conformance/test
# tolerance is a tight 1e-6 (exact arithmetic on the committed levels). Six
# periods of a periodic QRD concentrate the reflected energy into grating
# lobes, so d_theta is modest - consistent with the low published Appendix B
# values for periodic arrays (Cox & D'Antonio section 5.2.5).
ISO17497_2_QRD_N = 7  # quadratic-residue prime N
ISO17497_2_QRD_PERIODS = 6  # periods across the 3.6 m array
ISO17497_2_QRD_TOTAL_WIDTH = 3.6  # m, published total array width
ISO17497_2_QRD_WELL_WIDTH = 3.6 / 42  # m, period 0.6 m / N = 7 wells
ISO17497_2_QRD_MAX_DEPTH = 0.2  # m, published maximum well depth
ISO17497_2_QRD_DESIGN_FREQUENCY = 490.0  # Hz, gives the 0.2 m deepest well
ISO17497_2_SPEED_OF_SOUND = 343.0  # m/s, used throughout the prediction
ISO17497_2_PREDICTION_FREQUENCY = 1000.0  # Hz, single-frequency arc below
ISO17497_2_QRD_LEVELS: tuple[float, ...] = (
    -19.337, -18.867, -19.798, -26.178, -26.347, -18.811, -29.685, -18.212,
    -34.309, -13.850, -11.212, -1.191, -12.143, -16.334, -21.342, -25.559,
    -24.896, -22.567, 0.000, -19.414, -18.483, -17.966, -17.789, -16.949,
    -13.506, -1.146, -9.559, -10.912, -30.481, -13.875, -25.163, -14.346,
    -22.085, -22.176, -16.038, -15.275, -15.804,
)
ISO17497_2_FLAT_LEVELS: tuple[float, ...] = (
    -36.385, -35.709, -36.065, -41.617, -40.861, -32.474, -42.750, -31.127,
    -47.771, -28.995, -30.520, -50.381, -28.015, -23.478, -21.660, -20.958,
    -20.753, -20.733, 0.000, -20.733, -20.753, -20.958, -21.660, -23.478,
    -28.015, -50.381, -30.520, -28.995, -47.771, -31.127, -42.750, -32.474,
    -40.861, -41.617, -36.065, -35.709, -36.385,
)
ISO17497_2_QRD_DIFFUSION = 0.10985146785866741  # d_theta of the QRD arc, Formula (5)
ISO17497_2_FLAT_DIFFUSION = 0.004871959138901796  # d_theta of the flat reference arc
ISO17497_2_NORMALIZED_DIFFUSION = 0.10549346858814809  # d_theta_n, Formula (7)
# ---------------------------------------------------------------------------
# External anchor: Cox & D'Antonio, "Acoustic Absorbers and Diffusers",
# 3rd ed. (2017), Appendix B "Normalized diffusion coefficient table"
# (pp. 481-485), section 7, row "N = 7 QRD, 6 periods, 0.2 m deep", normal
# incidence. Third-party published data: 2D BEM predictions (thin-panel
# extrusions, source at 100 m, receiver arc at 50 m; each one-third-octave
# polar response is the average of seven single-frequency responses -
# section 5.2.5). Our Fraunhofer model reproduces the published normalised
# diffusion coefficient d_n in the 200-400 Hz one-third-octave bands within
# 0.01 (asserted at +/-0.015). CAVEAT: this is a low-band anchor, not
# full-band conformance - across the full published 100-5000 Hz range at
# normal incidence the model-vs-BEM mean absolute deviation is ~0.09 (the
# far-field phase-grating model ignores the edge diffraction and near-grazing
# effects the BEM resolves).
# ---------------------------------------------------------------------------
COX3E_APPENDIX_B_QRD_BANDS: tuple[float, ...] = (200.0, 250.0, 315.0, 400.0)
COX3E_APPENDIX_B_QRD_DN: tuple[float, ...] = (0.00, 0.01, 0.01, 0.01)
COX3E_APPENDIX_B_TOLERANCE = 0.015  # |model d_n - published BEM d_n| bound
# Formula (8) area factors use RADIANS internally, so the zenith weight is
# N0 = (4*pi/dphi)*sin^2(dtheta/4) / A_min with dtheta = dphi = 5 deg.
ISO17497_2_AREA_FACTOR_ZENITH = 1.571045588794762  # N0, radians convention

# ---------------------------------------------------------------------------
# Diffuser-design far-field prediction (Cox & D'Antonio, Fraunhofer model).
# Analytic anchors for the QRD-vs-flat behaviour of the design predictor.
#
# Geometry: an N = 7 quadratic residue diffuser (Eq. (10.2), s_n = n^2 mod N,
# so s = {0,1,4,2,2,4,1}) with design frequency f0 = 500 Hz and c = 343 m/s
# has design wavelength lambda0 = 0,686 m. The deepest well (s_max = 4) has,
# by Eq. (10.3) d_n = s_n*lambda0/(2N), depth 4*0,686/14 = 0,196 m exactly.
DIFFUSER_QRD7_MAX_DEPTH = 0.196  # m, closed form d_max = s_max*c/(2 N f0)
# A flat panel (all wells zero depth) normalises against itself, so Formula (7)
# gives (d - d_ref)/(1 - d_ref) = 0 identically: the exact zero anchor.
DIFFUSER_FLAT_NORMALIZED_DIFFUSION = 0.0
# The same N = 7 QRD (10 cm wells, five periods) predicted at 2 kHz: the
# normalised diffusion is well above the flat-panel zero, as a diffuser must be.
# The value is the far-field model prediction, committed as a regression guard.
DIFFUSER_QRD7_NORMALIZED_DIFFUSION_2K = 0.20802829817091092

# ---------------------------------------------------------------------------
# ISO 13472-1:2002 in-situ road-surface absorption. The mandatory geometry
# ds = 1,25 m, dm = 0,25 m gives the geometrical-spreading factor Kr = 2/3
# (Clause 4.2). The Annex A worked example (c = 340 m/s, 5 ms flat window)
# gives a maximum-sampled-area radius r ~ 1,34 m. Mirrors
# tests/test_road_absorption.py.
# ---------------------------------------------------------------------------
ISO13472_1_KR = 2.0 / 3.0  # geometrical-spreading factor
ISO13472_1_MSA_WINDOW = 5.0e-3  # reflected-wave window width Tw (s)
ISO13472_1_MSA_RADIUS = 1.3425466996067585  # Annex A worked example (m)

# ---------------------------------------------------------------------------
# ISO 13472-2:2010 spot method. The upper usable (plane-wave) frequency of a
# circular tube is f_u = 0,58 c0/d (Clause 5.4.1); a 100 mm tube at
# c0 = 343 m/s gives f_u = 1989,4 Hz.
# ---------------------------------------------------------------------------
ISO13472_2_SPOT_DIAMETER = 0.100  # tube diameter d (m)
ISO13472_2_SPOT_SPEED = 343.0  # speed of sound c0 (m/s)
ISO13472_2_SPOT_FU = 1989.4  # upper usable frequency (Hz)

# ---------------------------------------------------------------------------
# ISO 3745:2012 precision sound power (anechoic/hemi-anechoic). The Clause 10.5
# EXAMPLE combines sigma_omc = 2,0 dB and sigma_R0 = 0,5 dB at k = 2 to the
# expanded uncertainty U = 4,1 dB. The K1 background correction floors at
# 1,26 dB (>= 6 dB signal-to-noise edge bands). The meteorological correction
# C1 at the 23 C, ps0 reference is 5*lg(296/314) = -0,128 dB. Mirrors
# tests/test_sound_power_precision.py.
# ---------------------------------------------------------------------------
ISO3745_U_SIGMA_R0 = 0.5  # reproducibility standard deviation (dB)
ISO3745_U_SIGMA_OMC = 2.0  # operating/mounting/... std. deviation (dB)
ISO3745_U_COVERAGE = 2.0  # coverage factor k
ISO3745_U_EXPANDED = 4.123105625617661  # U = k*sqrt(sR0^2+somc^2) (dB)
ISO3745_K1_EDGE_LEVEL = 56.0  # measured Lp in the edge band (dB)
ISO3745_K1_EDGE_BACKGROUND = 50.0  # background Lp -> dLp = 6 dB (dB)
ISO3745_K1_EDGE_FREQUENCY = 200.0  # <= 200 Hz band uses the 6 dB floor (Hz)
ISO3745_K1_EDGE_FLOOR = 1.25628  # K1 floor, 6 dB S/N edge band (dB)
ISO3745_C1_REFERENCE = -0.12819  # C1 at 23 C, ps = ps0 (dB)

# ---------------------------------------------------------------------------
# ISO 9614-3:2002 precision intensity scanning. A fully enclosing surface with
# a uniform normal intensity In = W/S recovers the source power exactly, so
# LW = 10*lg(W/P0). For W = 100 uW this is LW = 80 dB (P0 = 1 pW).
# ---------------------------------------------------------------------------
ISO9614_3_UNIFORM_POWER = 1.0e-4  # radiated power W (W)
ISO9614_3_UNIFORM_AREAS: tuple[float, ...] = (0.5, 1.0, 0.25, 2.0)
ISO9614_3_UNIFORM_LW = 80.0  # 10*lg(W/1e-12) (dB)

# ---------------------------------------------------------------------------
# IEC 61043:1993 (EN 61043:1994) Table 2, standard page 14: minimum
# pressure-residual intensity index delta_pI0 requirements, in decibels, for
# probes, processors and instruments at the 25 mm nominal microphone
# separation. Transcribed digit for digit and cross-checked against the same
# table as reproduced in Fahy, "Sound Intensity" 2nd ed., Table 6.1 (printed
# page 136), which agrees exactly. Note 1 of the table: for a microphone
# separation x in millimetres, add 10 lg(x/25) dB to every figure.
# Row = (nominal_third_octave_Hz, probe_class1, probe_class2,
#        processor_class1, processor_class2, instrument_class1,
#        instrument_class2).
# ---------------------------------------------------------------------------
IEC61043_TABLE2: list[tuple[float, float, float, float, float, float, float]] = [
    (50, 13, 7, 19, 13, 12, 6),
    (63, 14, 8, 20, 14, 13, 7),
    (80, 15, 9, 21, 15, 14, 8),
    (100, 16, 10, 22, 16, 15, 9),
    (125, 17, 11, 23, 17, 16, 10),
    (160, 18, 12, 24, 18, 17, 11),
    (200, 19, 13, 25, 19, 18, 12),
    (250, 20, 14, 26, 20, 19, 13),
    (315, 20, 15, 26, 20, 19, 14),
    (400, 20, 16, 26, 20, 19, 14.5),
    (500, 20, 17, 26, 20, 19, 15),
    (630, 20, 18, 26, 20, 19, 16),
    (800, 20, 18, 26, 20, 19, 16),
    (1000, 20, 18, 26, 20, 19, 16),
    (1250, 20, 18, 26, 20, 19, 16),
    (1600, 20, 18, 26, 20, 19, 16),
    (2000, 20, 18, 26, 20, 19, 16),
    (2500, 20, 18, 26, 20, 19, 16),
    (3150, 20, 18, 26, 20, 19, 16),
    (4000, 20, 18, 26, 20, 19, 16),
    (5000, 20, 18, 26, 20, 19, 16),
    (6300, 20, 18, 26, 20, 19, 16),
]

# Fahy, "Sound Intensity" 2nd ed., section 6.8 (printed page 135), explaining
# the effect of the Table 2 requirement on the allowable channel phase
# mismatch: "a specified value of delta_pI0 of 20 dB corresponds to a phase
# mismatch of one-hundredth of the phase difference kd ...: at 1000 Hz, this
# corresponds to a phase mismatch of about 0.26 degrees" (25 mm separation).
IEC61043_PHASE_INDEX_DB = 20.0
IEC61043_PHASE_FREQUENCY_HZ = 1000.0
IEC61043_PHASE_SPACING_M = 0.025
IEC61043_PHASE_MISMATCH_DEG = 0.26

# ---------------------------------------------------------------------------
# PR-F human vibration (ISO 8041-1 / ISO 2631 / ISO 5349 / Directive 2002/44/EC).
# The true IEC 61260 one-third-octave centre is 10^(n/10) Hz; the reference
# frequencies of ISO 8041-1 Table 1 are exact (rad/s -> Hz). Design-goal
# factors are from ISO 8041-1:2017 Annex B (Tables B.1-B.9, 4 sig. figs).
# ---------------------------------------------------------------------------
# ISO 8041-1 Annex B design-goal weighting factors at the true band centre.
ISO8041_1_WK_FACTOR_6P31HZ = 1.054  # Table B.8, n = 8 (6,31 Hz) - Wk peak
ISO8041_1_WM_FACTOR_1P585HZ = 0.9342  # Table B.9, n = 2 (1,585 Hz) - Wm
ISO8041_1_WB_FACTOR_6P31HZ = 1.054  # Table B.1, n = 8 (6,31 Hz) - Wb peak
ISO8041_1_WB_FACTOR_1HZ = 0.3853  # Table B.1, n = 0 (1 Hz)
ISO8041_1_WB_FACTOR_100HZ = 0.1154  # Table B.1, n = 20 (100 Hz)
ISO8041_1_WD_FACTOR_1HZ = 1.011  # Table B.3, n = 0 (1 Hz)
ISO8041_1_WE_FACTOR_8HZ = 0.1263  # Table B.4, n = 9 (7,943 Hz)
ISO8041_1_WF_FACTOR_0P1585HZ = 1.004  # Table B.5, n = -8 (0,1585 Hz)
ISO8041_1_WF_FACTOR_0P1HZ = 0.6951  # Table B.5, n = -10 (0,1 Hz)
ISO8041_1_WJ_FACTOR_6P31HZ = 0.947  # Table B.7, n = 8 (6,31 Hz)
ISO8041_1_WJ_FACTOR_8HZ = 1.016  # Table B.7, n = 9 (7,943 Hz)
# ISO 8041-1 Table 1 weighting factors at the reference frequencies.
ISO8041_1_WH_REF_FREQ_HZ = 500.0 / (2.0 * math.pi)  # 79,577 Hz (500 rad/s)
ISO8041_1_WH_REF_FACTOR = 0.2020  # Table 1, Wh @ 500 rad/s
ISO8041_1_WBV_REF_FREQ_HZ = 100.0 / (2.0 * math.pi)  # 15,915 Hz (100 rad/s)
ISO8041_1_WC_REF_FACTOR = 0.5145  # Table 1, Wc @ 100 rad/s
ISO8041_1_WD_REF_FACTOR = 0.1261  # Table 1, Wd @ 100 rad/s

# ISO 8041-1:2017 Annex B, Tables B.1-B.9: the printed design-goal weighting
# factor per one-third-octave band (band number n -> true centre 10^(n/10) Hz;
# factors to 4 significant figures). Transcribed from the standard and
# cross-validated against the printed dB column (20 lg factor within the
# 0,01 dB print rounding) row by row.
ISO8041_1_ANNEX_B_FACTORS: dict[str, tuple[tuple[int, float], ...]] = {
    "Wb": (
        (-10, 0.02494), (-9, 0.03941), (-8, 0.06198), (-7, 0.09645), (-6, 0.1464),
        (-5, 0.2113), (-4, 0.28), (-3, 0.3347), (-2, 0.3666), (-1, 0.3808),
        (0, 0.3853), (1, 0.3864), (2, 0.3916), (3, 0.4168), (4, 0.496), (5, 0.6653),
        (6, 0.885), (7, 1.026), (8, 1.054), (9, 1.026), (10, 0.9745), (11, 0.9042),
        (12, 0.8144), (13, 0.7088), (14, 0.5973), (15, 0.4906), (16, 0.395),
        (17, 0.3118), (18, 0.2389), (19, 0.1734), (20, 0.1154), (21, 0.06929),
        (22, 0.03818), (23, 0.01999), (24, 0.0102), (25, 0.005154), (26, 0.002591),
    ),
    "Wc": (
        (-10, 0.06238), (-9, 0.09858), (-8, 0.1551), (-7, 0.2415), (-6, 0.3669),
        (-5, 0.5302), (-4, 0.7042), (-3, 0.8442), (-2, 0.9292), (-1, 0.9716),
        (0, 0.991), (1, 1.0), (2, 1.006), (3, 1.012), (4, 1.017), (5, 1.023),
        (6, 1.024), (7, 1.013), (8, 0.9739), (9, 0.8941), (10, 0.7762), (11, 0.6425),
        (12, 0.5166), (13, 0.4098), (14, 0.3236), (15, 0.2549), (16, 0.2002),
        (17, 0.1557), (18, 0.1182), (19, 0.08538), (20, 0.05665), (21, 0.03394),
        (22, 0.01868), (23, 0.009772), (24, 0.004987), (25, 0.002518), (26, 0.001266),
    ),
    "Wd": (
        (-10, 0.06242), (-9, 0.09867), (-8, 0.1553), (-7, 0.242), (-6, 0.3682),
        (-5, 0.533), (-4, 0.7097), (-3, 0.854), (-2, 0.9443), (-1, 0.9914), (0, 1.011),
        (1, 1.007), (2, 0.9707), (3, 0.8913), (4, 0.7733), (5, 0.6398), (6, 0.5143),
        (7, 0.4081), (8, 0.3226), (9, 0.255), (10, 0.2017), (11, 0.1597), (12, 0.1266),
        (13, 0.1004), (14, 0.07958), (15, 0.06299), (16, 0.04965), (17, 0.03872),
        (18, 0.02946), (19, 0.0213), (20, 0.01414), (21, 0.008478), (22, 0.004668),
        (23, 0.002442), (24, 0.001246), (25, 0.0006293), (26, 0.0003164),
    ),
    "We": (
        (-10, 0.06252), (-9, 0.09893), (-8, 0.156), (-7, 0.2435), (-6, 0.3715),
        (-5, 0.5394), (-4, 0.7198), (-3, 0.8635), (-2, 0.9389), (-1, 0.9423),
        (0, 0.8798), (1, 0.7683), (2, 0.6372), (3, 0.5127), (4, 0.407), (5, 0.3218),
        (6, 0.2543), (7, 0.2012), (8, 0.1594), (9, 0.1263), (10, 0.1002),
        (11, 0.07954), (12, 0.06314), (13, 0.05011), (14, 0.03975), (15, 0.03147),
        (16, 0.02481), (17, 0.01935), (18, 0.01473), (19, 0.01065), (20, 0.007071),
        (21, 0.004239), (22, 0.002334), (23, 0.001221), (24, 0.0006232),
        (25, 0.0003147), (26, 0.0001582),
    ),
    "Wf": (
        (-17, 0.02407), (-16, 0.03803), (-15, 0.06021), (-14, 0.09619), (-13, 0.1575),
        (-12, 0.2675), (-11, 0.4537), (-10, 0.6951), (-9, 0.9), (-8, 1.004),
        (-7, 0.9928), (-6, 0.8501), (-5, 0.6149), (-4, 0.3884), (-3, 0.2225),
        (-2, 0.1157), (-1, 0.05434), (0, 0.02352), (1, 0.009705), (2, 0.003916),
        (3, 0.001566),
    ),
    "Wh": (
        (-1, 0.01586), (0, 0.02514), (1, 0.03985), (2, 0.06314), (3, 0.09992),
        (4, 0.1576), (5, 0.2461), (6, 0.3754), (7, 0.545), (8, 0.7272), (9, 0.8731),
        (10, 0.9514), (11, 0.9576), (12, 0.8958), (13, 0.782), (14, 0.6471),
        (15, 0.5192), (16, 0.4111), (17, 0.3244), (18, 0.256), (19, 0.2024),
        (20, 0.1602), (21, 0.127), (22, 0.1007), (23, 0.07988), (24, 0.06338),
        (25, 0.05026), (26, 0.0398), (27, 0.03137), (28, 0.02447), (29, 0.01862),
        (30, 0.01346), (31, 0.00894), (32, 0.005359), (33, 0.00295), (34, 0.001544),
        (35, 0.0007878), (36, 0.0003978),
    ),
    "Wj": (
        (-10, 0.03099), (-9, 0.04897), (-8, 0.07703), (-7, 0.1199), (-6, 0.1821),
        (-5, 0.263), (-4, 0.3489), (-3, 0.4176), (-2, 0.4585), (-1, 0.4776),
        (0, 0.4844), (1, 0.4851), (2, 0.4832), (3, 0.4819), (4, 0.4889), (5, 0.5246),
        (6, 0.6251), (7, 0.7948), (8, 0.947), (9, 1.016), (10, 1.03), (11, 1.026),
        (12, 1.019), (13, 1.012), (14, 1.006), (15, 1.0), (16, 0.9911), (17, 0.972),
        (18, 0.9304), (19, 0.8465), (20, 0.7075), (21, 0.5338), (22, 0.37),
        (23, 0.2437), (24, 0.1565), (25, 0.09951), (26, 0.06297),
    ),
    "Wk": (
        (-10, 0.03121), (-9, 0.04931), (-8, 0.07756), (-7, 0.1207), (-6, 0.1832),
        (-5, 0.2644), (-4, 0.3504), (-3, 0.4188), (-2, 0.4588), (-1, 0.4767),
        (0, 0.4825), (1, 0.4846), (2, 0.4935), (3, 0.5308), (4, 0.6335), (5, 0.8071),
        (6, 0.9648), (7, 1.039), (8, 1.054), (9, 1.037), (10, 0.9884), (11, 0.8989),
        (12, 0.7743), (13, 0.6373), (14, 0.5103), (15, 0.4031), (16, 0.316),
        (17, 0.2451), (18, 0.1857), (19, 0.1339), (20, 0.08873), (21, 0.05311),
        (22, 0.02922), (23, 0.01528), (24, 0.007795), (25, 0.003935), (26, 0.001978),
    ),
    "Wm": (
        (-10, 0.01584), (-9, 0.0251), (-8, 0.03976), (-7, 0.06293), (-6, 0.09941),
        (-5, 0.1563), (-4, 0.243), (-3, 0.3684), (-2, 0.5304), (-1, 0.7003),
        (0, 0.8329), (1, 0.9071), (2, 0.9342), (3, 0.9319), (4, 0.9101), (5, 0.8721),
        (6, 0.8184), (7, 0.7498), (8, 0.6692), (9, 0.5819), (10, 0.4941), (11, 0.4114),
        (12, 0.3375), (13, 0.2738), (14, 0.2203), (15, 0.176), (16, 0.1396),
        (17, 0.1093), (18, 0.08336), (19, 0.06036), (20, 0.04013), (21, 0.02407),
        (22, 0.01326), (23, 0.006937), (24, 0.003541), (25, 0.001788), (26, 0.000899),
    ),
}

# ISO 8041-1:2017 Table 4: tolerance transition frequencies ft1..ft4 per
# weighting, as exact powers 10^(k/10) Hz. The Table 5 magnitude tolerances
# per region: f <= ft1 and f >= ft4: +26 %/-100 %; ft1 < f < ft2 and
# ft3 < f < ft4: +26 %/-21 %; ft2 <= f <= ft3: +12 %/-11 %.
_WBV_TRANSITIONS = (10.0**-0.6, 10.0**-0.2, 10.0**1.8, 10.0**2.2)
ISO8041_1_TABLE4_TRANSITIONS: dict[str, tuple[float, float, float, float]] = {
    "Wb": _WBV_TRANSITIONS,
    "Wc": _WBV_TRANSITIONS,
    "Wd": _WBV_TRANSITIONS,
    "We": _WBV_TRANSITIONS,
    "Wf": (10.0**-1.3, 10.0**-0.9, 10.0**-0.4, 10.0**0.0),
    "Wh": (10.0**0.6, 10.0**1.0, 10.0**2.9, 10.0**3.3),
    "Wj": _WBV_TRANSITIONS,
    "Wk": _WBV_TRANSITIONS,
    "Wm": (10.0**-0.3, 10.0**0.1, 10.0**1.8, 10.0**2.2),
}
# Table 5 magnitude tolerances (upper, lower), as fractions, by region index
# 0: f <= ft1, 1: ft1 < f < ft2, 2: ft2 <= f <= ft3, 3: ft3 < f < ft4,
# 4: f >= ft4.
ISO8041_1_TABLE5_TOLERANCES: tuple[tuple[float, float], ...] = (
    (0.26, 1.00), (0.26, 0.21), (0.12, 0.11), (0.26, 0.21), (0.26, 1.00),
)
# ISO 5349-2:2001 Annex E worked-example daily exposures A(8), m/s^2.
ISO5349_2_E21_A8 = 4.1  # E.2.1 single tool: 7,4*sqrt(2,5/8)
ISO5349_2_E3_A8 = 3.6  # E.3 forestry three-task combination
# ISO 5349-1:2001 Annex C: Dy = 31,8*A(8)^-1,06; Table C.1 A(8)=7 -> Dy=4 yr.
ISO5349_1_VWF_A8 = 7.0
ISO5349_1_VWF_DY_YEARS = 4.0
# Directive 2002/44/EC Article 3 daily exposure action/limit values.
DIRECTIVE_2002_44_HAV_EAV = 2.5  # A(8) m/s^2, Art. 3(1)(a)
DIRECTIVE_2002_44_HAV_ELV = 5.0  # A(8) m/s^2, Art. 3(1)(b)
DIRECTIVE_2002_44_WBV_EAV = 0.5  # A(8) m/s^2, Art. 3(2)(a)
DIRECTIVE_2002_44_WBV_ELV = 1.15  # A(8) m/s^2, Art. 3(2)(b)

# ---------------------------------------------------------------------------
# ANSI S3.5-1997 Speech Intelligibility Index (one-third-octave method).
# Primary oracle: the reference implementation of ASA Working Group S3-79
# (the committee that maintains ANSI S3.5), published on its support site
# sii.to as SII.C together with official test-input files (*.TST) and their
# published results (DevelopmentKit readme, three decimals). "SII.C run"
# below is the value printed by that C program, compiled unmodified with gcc
# and run on the stated input: committee code, independent of this library.
# Secondary cross-checks retained where they agree: the Hornsby SII
# worksheet (ANSIS3_51997SII.xlsx) and the R CRAN package "SII", both
# independent implementations.
# ---------------------------------------------------------------------------
ANSIS3_5_BAND_IMPORTANCE_SUM = 1.0  # Table 3, sum of Ii (identical digits in SII.C i_avg[])
# Equivalent masking spectrum level Zi at 200 Hz, standard normal-effort
# spectrum in quiet: SII.C run prints z[1] = -1.664717 (the Hornsby
# worksheet agrees).
ANSIS3_5_MASKING_Z_200HZ = -1.665
# SII for the standard normal-effort spectrum in quiet (-80 dB noise) with
# normal hearing: SII.C run = 0.9958251667; the Hornsby SII worksheet gives
# the same digits at full precision (its column M is the clause 5.6 maximum
# Di = max(Zi, Xi'); an energy sum instead reads 5e-6 low here but up to
# 0.042 low in noise-plus-hearing-loss conditions).
ANSIS3_5_STANDARD_QUIET = 0.99582516666667
# Equivalent disturbance spectrum level Di at 5000 Hz for the same condition:
# the quiet field leaves Di = Xi' = -23.6 dB (Table 3 reference internal
# noise, also SII.C x[15]), which the clause 5.6 maximum preserves exactly.
ANSIS3_5_DISTURBANCE_5000HZ = -23.6
# Discriminating adverse-condition oracle (would catch an energy-sum Di):
# normal speech, flat 30 dB noise spectrum, flat 40 dB hearing loss.
# SII.C run = 0.2184539329; the Hornsby worksheet rounds it to 0.2185 (the
# energy-sum variant reads 0.1841).
ANSIS3_5_NOISE_PLUS_LOSS = 0.2184539329
# ANSI S3.5-1997 Annex C.2 worked example (one-third-octave method): speech
# 54 dB in all bands, noise 40/30/20 dB in the first three bands, normal
# hearing. SII.C run = 0.8513748619; the R CRAN package "SII" prints
# 0.8513749 for the same input (its vignette reproduces the standard's
# Table C.2). Table C.2 carries an official WG S3-79 erratum in its first
# row (self-masking slope Ci printed -45.59, corrected -46.59, sii.to
# errata list); the corrected chain is what both implementations compute
# (C1 = -46.587) and the printed Zi column below is only consistent with it.
ANSIS3_5_ANNEX_C2 = 0.8513748619
# Table C.2, printed equivalent masking spectrum level Zi of the first three
# rows (two decimals, errata-consistent): the misprinted slope would give
# 34.76 dB at 200 Hz instead of the printed 34.66 dB. The 250 Hz cell is
# printed 25.04 while the exact chain gives 25.0468 (SII.C z[2] agrees), so
# that print truncates rather than rounds; tests use a 0.01 dB tolerance.
ANSIS3_5_ANNEX_C2_MASKING = (40.00, 34.66, 25.04)
# Table 3, loud-effort standard speech spectrum at 1 kHz. Surrogate-anchored
# (R CRAN "SII" and Google implementation transcriptions of Table 3): the WG
# kit's SII.C carries only the normal-effort spectrum.
ANSIS3_5_LOUD_1KHZ = 42.16
# Official WG S3-79 test cases for the one-third-octave procedure
# (DevelopmentKit SOURCES/TO.TST and TO_1.TST): lines are the equivalent
# speech spectrum level, equivalent noise spectrum level and equivalent
# hearing threshold level over the 18 bands; TO_1 adds an alternative
# band-importance function (18 values consumed; the file's spurious 19th
# entry is ignored by SII.C). Published results (readme, 3 decimals):
# TO.TST -> 0.445, TO_1.TST -> 0.438; SII.C run at full precision:
# 0.4453910059 and 0.4382176540.
ANSIS3_5_WG_TO_SPEECH = (
    90.0, 5.0, 40.0, 40.0, 40.0, 40.0, 40.0, 40.0, 40.0, 40.0, 40.0, 40.0,
    40.0, 40.0, -10.0, -10.0, -10.0, -10.0,
)
ANSIS3_5_WG_TO_NOISE = (
    10.0, -10.0, -10.0, 75.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0,
    -10.0, -10.0, -10.0, -10.0, 10.0, 10.0, 10.0, 10.0,
)
ANSIS3_5_WG_TO_THRESHOLD = (
    90.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
)
ANSIS3_5_WG_TO_SII = 0.445  # published in the WG DevelopmentKit readme
ANSIS3_5_WG_TO_SII_EXACT = 0.4453910059  # SII.C run on TO.TST
ANSIS3_5_WG_TO1_IMPORTANCE = (
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.1, 0.1, 0.1, 0.1,
    0.1, 0.1, 0.3, 0.0,
)
ANSIS3_5_WG_TO1_SII = 0.438  # published in the WG DevelopmentKit readme
ANSIS3_5_WG_TO1_SII_EXACT = 0.4382176540  # SII.C run on TO_1.TST

# ---------------------------------------------------------------------------
# Prominence of impulsive sounds - NT ACOU 112:2002.
# P = 3*lg(1000) + 2*lg(30) = 11.9542 (Formula 1); the adjustment at P = 10 is
# KI = 1.8*(10 - 5) = 9.0 dB (Formula 2).
# ---------------------------------------------------------------------------
NTACOU112_PROMINENCE = 11.9542  # P for onset rate 1000 dB/s, level difference 30 dB
NTACOU112_ADJUSTMENT_P10 = 9.0  # KI at P = 10

# ---------------------------------------------------------------------------
# Objective prominence of impulsive sounds - ISO/PAS 1996-3:2022.
# The standard gives no worked numeric example, so the oracle is derived by
# hand from its own definitions. A linear LpAF ramp of 30 dB over 0.30 s has,
# by construction, level difference LD = Le - Ls = 30 dB (3.4) and onset rate
# OR = 30/0.30 = 100 dB/s (the least-squares slope of a straight ramp, 3.5).
# The prominence is P = 3*lg(100) + 2*lg(30) = 6 + 2*1.4771213 = 8.9542426
# (Clause 5, Formula 2) and the adjustment KI = 1.8*(P - 5) = 7.1176366 dB
# (Clause 6, Formula 3).
# ---------------------------------------------------------------------------
ISO1996_3_RAMP_ONSET_RATE = 100.0  # OR, dB/s (30 dB over 0.30 s)
ISO1996_3_RAMP_LEVEL_DIFFERENCE = 30.0  # LD, dB
ISO1996_3_RAMP_PROMINENCE = 3.0 * math.log10(100.0) + 2.0 * math.log10(30.0)  # 8.9542426
ISO1996_3_RAMP_ADJUSTMENT = 1.8 * (ISO1996_3_RAMP_PROMINENCE - 5.0)  # 7.1176366 dB

# ---------------------------------------------------------------------------
# Room-noise criteria - ANSI/ASA S12.2-2019.
# Feeding an NC curve of Table 1 back through the tangency method returns its
# NC value; the RC Mark II curves reproduce Table D.1 (the 63 Hz level of the
# RC-31 curve is 51 dB); the mid-frequency average of the RC-35 curve is 35 dB.
# ---------------------------------------------------------------------------
ANSIS12_2_NC40_SELF = 40.0  # NC-40 curve -> tangency rating (Table 1)
ANSIS12_2_RC31_63HZ = 51.0  # RC-31 curve, 63 Hz octave-band level (Table D.1)
ANSIS12_2_RC35_LMF = 35.0  # RC-35 curve, mid-frequency average LMF (clause D.4)

# ---------------------------------------------------------------------------
# Hearing thresholds - ISO 7029:2017 (age) and ISO 389-7:2005 (reference).
# The median deviation follows a*(age-18)**b (Table 1); at 4 kHz for a 60-year
# male it is 20.21 dB. The upper spread su is a degree-5 polynomial (Table 2);
# at 1 kHz age 60 male it is 10.15 dB. The free-field reference threshold at
# 1 kHz is 2.4 dB (ISO 389-7 Table 1).
# ---------------------------------------------------------------------------
ISO7029_MEDIAN_MALE_60_4KHZ = 20.2085  # dB, ISO 7029 Table 1 median formula
ISO7029_SU_MALE_60_1KHZ = 10.1533  # dB, ISO 7029 Table 2 upper spread
ISO389_7_REF_FREE_1KHZ = 2.4  # dB, ISO 389-7 Table 1 free-field

# ---------------------------------------------------------------------------
# Measurement uncertainty - ISO/IEC Guide 98-3 (GUM) and Supplement 1.
# The additive model y = x1+x2+x3+x4 with u(xi)=1 has uc = 2.0 (Suppl 1, 9.2);
# the coverage factor at p=0.99 with 16 degrees of freedom is 2.92 (GUM Annex
# H.1 / Table G.2); equal contributions each with 10 degrees of freedom give a
# Welch-Satterthwaite effective dof of 40 (Annex G.4).
# ---------------------------------------------------------------------------
GUM_ADDITIVE_UC = 2.0  # combined standard uncertainty, additive model
GUM_COVERAGE_K99_16 = 2.92  # coverage factor t at p=0.99, v=16
GUM_WELCH_VEFF = 40.0  # Welch-Satterthwaite effective degrees of freedom

# GUM Annex H.1 end-gauge calibration, end to end: model
# l = lS + d - lS*(dalpha*theta + alphaS*dtheta) with the H.1.3 inputs
# (value, u, dof): lS = 50 000 623 nm (25, 18); d = 215 nm (9.7, 25.6);
# alphaS = 11.5e-6 /degC (1.2e-6, inf); theta = -0.1 degC (0.41, inf);
# dalpha = 0 (0.58e-6, 50); dtheta = 0 (0.029, 2). Published results
# (H.1.4-H.1.6): l = 50 000 838 nm; uc = 32 nm (unrounded 31.71);
# contributions (25, 9.7, 0, 0, 2.9, 16.7) nm -- alphaS and theta are
# genuinely flat directions at the estimates; veff = 16 (truncated from
# 16.66, G.4.2); U99 = 93 nm at k(0.99, 16) = 2.92 (interpolation at the
# untruncated veff, permitted by G.4.2 NOTE 1, gives 92.1 nm).
GUM_H1_INPUTS = [
    # (value, standard uncertainty, dof)
    (50_000_623.0, 25.0, 18.0),        # lS, nm
    (215.0, 9.7, 25.6),                # d, nm
    (11.5e-6, 1.2e-6, math.inf),       # alphaS, 1/degC
    (-0.1, 0.41, math.inf),            # theta, degC
    (0.0, 0.58e-6, 50.0),              # dalpha, 1/degC
    (0.0, 0.029, 2.0),                 # dtheta, degC
]
GUM_H1_VALUE = 50_000_838.0            # nm
GUM_H1_UC = 31.71                      # nm (printed 32)
GUM_H1_CONTRIBUTIONS = [25.0, 9.7, 0.0, 0.0, 2.9, 16.7]
GUM_H1_VEFF = 16.66                    # (printed truncated to 16)
GUM_H1_U99 = 92.1                      # nm at the untruncated veff (printed 93)

# GUM Annex H.2 simultaneous resistance/reactance measurement: the only
# published numeric oracle of the correlated Equation (16) path. Five
# simultaneous observation sets of (V / V, I / mA, phi / rad) from Table H.2;
# their means, standard deviations of the means and sample correlation
# coefficients (r(V,I) = -0.36, r(V,phi) = 0.86, r(I,phi) = -0.65 after
# 2-decimal print rounding) feed R = (V/I) cos phi, X = (V/I) sin phi,
# Z = V/I. Published results (Table H.3): R = 127.732 ohm, uc = 0.071;
# X = 219.847 ohm, uc = 0.295; Z = 254.260 ohm, uc = 0.236. The uc reproduce
# with the correlations computed from the observations; the 2-decimal printed
# r values give uc(R) = 0.070 (their rounding).
GUM_H2_OBSERVATIONS = [
    (5.007, 19.663, 1.0456),
    (4.994, 19.639, 1.0438),
    (5.005, 19.640, 1.0468),
    (4.990, 19.685, 1.0428),
    (4.999, 19.678, 1.0433),
]
GUM_H2_RESULTS = {                     # measurand: (value / ohm, uc / ohm)
    "R": (127.732, 0.071),
    "X": (219.847, 0.295),
    "Z": (254.260, 0.236),
}

# GUM Supplement 1 clause 9.2 additive model Y = X1+X2+X3+X4: the 95 %
# probabilistically symmetric coverage intervals. Table 2 (standard Gaussian
# inputs): +/-3.92 (analytic; GUF identical). Table 3 (rectangular inputs of
# unit standard deviation): u(y) = 2.00 and +/-3.88, analytically
# 2*sqrt(3)*(2 - (3/5)^(1/4)) = 3.8807 (Annex E).
GUMS1_TABLE2_INTERVAL_95 = 3.92
GUMS1_TABLE3_INTERVAL_95 = 3.88
GUMS1_TABLE3_U = 2.00

# ---------------------------------------------------------------------------
# Noise-induced hearing loss - ISO 1999:2013, Annex D worked examples (dB).
# Table D.2 (L_EX,8h = 90 dB, 20 years) at 4 kHz: median NIPTS = 13 dB and the
# most-susceptible tenth (fractile 0.9) = 18 dB. Table D.4 (100 dB, 40 years)
# at 3 kHz, fractile 0.9 = 60 dB.
# ---------------------------------------------------------------------------
ISO1999_N50_4K_90_20 = 13.0  # median NIPTS, 4 kHz, 90 dB, 20 yr
ISO1999_N10_4K_90_20 = 18.0  # worst-10 % NIPTS, 4 kHz, 90 dB, 20 yr
ISO1999_N10_3K_100_40 = 60.0  # worst-10 % NIPTS, 3 kHz, 100 dB, 40 yr

# ---------------------------------------------------------------------------
# Noise-induced hearing loss - ISO 1999:2013, Annex C worked example (risk of
# noise-induced hearing loss and disability). A highly screened male
# population aged 50 exposed to L_EX,8h = 90 dB for 30 years, assessed on the
# 1/2/4 kHz frequency combination, at the percentage Q = 10 % (the
# most-susceptible tenth; the library fractile 0.9).
#
# The annex's own printed inputs are the Table A.3 age-associated thresholds
# H = 14, 21 and 36 dB and the Table D.2 shifts N = 0, 9 and 19 dB. The
# quantities pinned here are the results the annex derives from them:
#   C.5   the 4 kHz shift after the Formula (1) compression,
#         19 - 36 x 19 / 120 = 13,3 dB;
#   C.8   the 1/2/4 kHz mean shift, (0 + 9 + 13,3) / 3 = 7,4 dB;
#   C.3   the 1/2/4 kHz mean age threshold, (14 + 21 + 36) / 3 = 23,7 dB;
#   C.11  the combined threshold, 23,7 + 7,4 = 31,1 dB.
# The annex applies the compression only where it matters: "when (H + N) <
# 40 dB, the NIPTS can be taken directly from Table D.2", so of these three
# bands only 4 kHz (36 + 19 dB) is compressed.
# ---------------------------------------------------------------------------
ISO1999_ANNEX_C_H = (14.0, 21.0, 36.0)  # Table A.3 H, male 50 yr, Q = 10 %
ISO1999_ANNEX_C_N = (0.0, 9.0, 19.0)  # Table D.2 NIPTS, 90 dB, 30 yr, Q = 10 %
ISO1999_ANNEX_C_N_4K_COMPRESSED = 13.3  # C.5, dB
ISO1999_ANNEX_C_COMPRESSION_FENCE = 40.0  # H + N above which the annex compresses
ISO1999_ANNEX_C_N_MEAN = 7.4  # C.8, dB
ISO1999_ANNEX_C_H_MEAN = 23.7  # C.3, dB
ISO1999_ANNEX_C_HTLAN = 31.1  # C.11, dB

# ---------------------------------------------------------------------------
# Multiple-shock whole-body vibration - ISO 2631-5:2018, Annex C worked
# example: five 40 m/s2 response peaks per day, 82 kg male, exposure from age
# b = 20 for n = 20 years at N = 120 days/year. Daily acceleration dose
# Dzd = 55.97 m/s2, stress variable R = 1.22 and injury probability Pi = 0.37.
# ---------------------------------------------------------------------------
ISO2631_5_DZD_MALE = 55.97  # daily acceleration dose, m/s2 (Formula 3)
ISO2631_5_R_MALE = 1.22  # cumulative stress variable R (Formula C.3)
ISO2631_5_PI_MALE = 0.37  # probability of lumbar injury (Formula C.5)
# Annex C NOTE 5: the same exposure for a 64 kg female (mz = 0,025 MPa/(m/s2)).
ISO2631_5_SD_FEMALE = 1.40  # daily compressive stress Sd, MPa (Formula C.1)
ISO2631_5_R_FEMALE = 0.97  # cumulative stress variable R (Formula C.3).
# Exact recomputation of NOTE 5 (mz = 0,025, Sage = 0,039, b = 20, n = 20)
# gives R = 0.9621, which rounds to 0.96; the printed 0.97 is a last-digit
# inconsistency of the standard's own note (the male path reproduces the
# printed 1.22 exactly with the identical code). Tolerance 0.01 covers it.

# ISO 2631-5:2018 Annex D, Table D.1: digital-filter realization of the
# clause 5.2 seat-to-spine transfer function (Formula 1) at fs = 256 Hz,
# 12 taps. An independent cross-check of the Formula 1 coefficients: the
# analog magnitude must match the filter within the clause 5.2 tolerance
# (+/- 0,04 up to 40 Hz, +/- 0,08 up to 80 Hz).
ISO2631_5_ANNEX_D_FS = 256.0  # sampling frequency of the Table D.1 filter, Hz
ISO2631_5_ANNEX_D_B: tuple[float, ...] = (
    -0.000005710, 0.000020010, 0.001373900, 0.014541920, 0.025152310,
    -0.014242050, -0.044262840, -0.008888510, 0.017715720, 0.010216420,
    0.002030740, 0.000055980,
)
ISO2631_5_ANNEX_D_A: tuple[float, ...] = (
    1.000000000, -3.323217600, 4.256126150, -1.980417270, -1.488735470,
    3.329511290, -2.949072140, 1.653403410, -0.635677800, 0.167519420,
    -0.028076980, 0.002348730,
)

# ---------------------------------------------------------------------------
# Mechanical mobility and transfer stiffness - ISO 7626 / ISO 10846 anchors.
# ISO 7626-2:2015, 7.5.2: the FRF of a freely suspended rigid block of mass
# m = 10 kg is mag(A) = 1/m = 0,100 1/kg at every frequency, and
# mag(Y) = 1/(2*pi*f*m) = 1,59155e-4 m/(N.s) at 100 Hz, within +/-5 %.
# Annex A: coherence 0,8 with n = 75 averages -> normalized random error
# eps = sqrt((1-g2)/(2*n*g2)) = 4,08 % (< 5 %, the 8.1.3 criterion).
# The omega = 1000 rad/s decade identity across the FRF family: a rigid 1 kg
# mass has accelerance 1 1/kg, mobility 1e-3 m/(N.s), compliance 1e-6 m/N.
# ISO 10846-3:2002, 6.1: valid where DeltaL1,2 >= 20 dB (mag(T) <= 0,1); the
# T << 1 approximation then holds within 1 dB (12 %) - at the limit the
# undamped mass-spring model gives k_indirect/k = 1,1 (0,83 dB, 10 %).
# ISO 10846-1:2008, Eq. (6): at mag(k2,2/kt) = 0,1 the delivered force is
# F2/F2,b = 1/1,1 = 0,9091 - the Eq. (7) "within 10 %" claim.
# ISO 10846-2/-3, 7.6: two input spectra 10 dB apart must give transfer-
# stiffness levels within 1,5 dB for the data to count as linear.
# ---------------------------------------------------------------------------
ISO7626_2_CAL_MASS_KG = 10.0  # rigid calibration block mass (7.5.2)
ISO7626_2_CAL_ACCELERANCE = 0.100  # mag(A) = 1/m, 1/kg
ISO7626_2_CAL_MOBILITY_100HZ = 1.59155e-4  # mag(Y) at 100 Hz, m/(N.s)
ISO7626_2_RANDOM_ERROR_PCT = 4.08  # Annex A example: g2 = 0,8, n = 75
ISO7626_1_DECADE_FREQ_HZ = 1000.0 / (2.0 * math.pi)  # 159,155 Hz (1000 rad/s)
ISO7626_1_DECADE_MOBILITY = 1.0e-3  # mag(Y) of a rigid 1 kg mass, m/(N.s)
ISO7626_1_DECADE_COMPLIANCE = 1.0e-6  # mag(H) of a rigid 1 kg mass, m/N
ISO10846_3_LIMIT_DELTA_L_DB = 20.0  # Inequality (2): DeltaL1,2 >= 20 dB
ISO10846_3_LIMIT_BIAS_RATIO = 1.1  # k_indirect/k of the model at mag(T) = 0,1
ISO10846_3_ACCURACY_DB = 1.0  # 6.1: Formula (1) accurate within 1 dB
ISO10846_3_ACCURACY_FRACTION = 0.12  # i.e. within 12 %
ISO10846_1_EQ6_FORCE_RATIO = 1.0 / 1.1  # F2/F2,b at mag(k2,2/kt) = 0,1
ISO10846_LINEARITY_STEP_DB = 10.0  # 7.6: input spectra A/B, 10 dB apart
ISO10846_LINEARITY_TOL_DB = 1.5  # 7.6 c): levels equal within 1,5 dB

# ---------------------------------------------------------------------------
# Sound absorption in enclosed spaces - EN 12354-6:2003, Annex E worked
# example (4,54 x 2,73 x 2,40 = 29,75 m3 room, 1000 Hz octave band). Case 1
# (bare) A = 2,26 m2, T = 2,1 s; Case 2 (with hard objects) A = 5,03 m2.
# The six bare-room surfaces are (area_m2, alpha at 1000 Hz): floor, ceiling,
# long wall, side wall, side wall, glass facade.
# ---------------------------------------------------------------------------
EN12354_6_ANNEX_E_VOLUME = 29.75  # room volume (m3)
EN12354_6_ANNEX_E_BARE_SURFACES: list[tuple[float, float]] = [
    (12.39, 0.05),  # floor
    (12.39, 0.02),  # ceiling
    (10.90, 0.04),  # long wall
    (10.90, 0.04),  # side wall
    (6.55, 0.04),   # side wall
    (6.55, 0.04),   # glass facade
]
EN12354_6_A_BARE = 2.26  # equivalent absorption area, bare room (m2)
EN12354_6_T_BARE = 2.1  # reverberation time, bare room (s)
EN12354_6_A_OBJECTS = 5.03  # equivalent absorption area, with objects (m2)

# ---------------------------------------------------------------------------
# Prominent discrete tones - ECMA-418-1:2024, clause-EXAMPLE anchors
# (transcribed from the official PDF, printed to one decimal).
# Clause 10 Formula (2): critical band around ft = 1 kHz is
# f1,c = 922,2 Hz .. f2,c = 1084,4 Hz, width dfc = 162,2 Hz (117,3 Hz at
# 500 Hz). Clause 11.6 Formula (14): proximity spacing dfprox = 23 Hz at
# 150 Hz and 63,8 Hz at 850 Hz.
# ---------------------------------------------------------------------------
ECMA418_1_DFC_1KHZ = 162.2  # critical bandwidth at 1 kHz (Hz)
ECMA418_1_DFC_500HZ = 117.3  # critical bandwidth at 500 Hz (Hz)
ECMA418_1_F1_1KHZ = 922.2  # lower critical-band edge at 1 kHz (Hz)
ECMA418_1_F2_1KHZ = 1084.4  # upper critical-band edge at 1 kHz (Hz)
ECMA418_1_PROX_150HZ = 23.0  # proximity spacing at 150 Hz (Hz)
ECMA418_1_PROX_850HZ = 63.8  # proximity spacing at 850 Hz (Hz)

# ---------------------------------------------------------------------------
# ISO 717-2 Annex C, Table C.1 - measured normalized impact sound pressure
# level Ln (100-3150 Hz, one-third-octave, laboratory). The worked example
# gives Ln,w = 79 dB, CI = -11 dB with an unfavourable-deviation sum of
# 28,0 dB. CI = -11 pins the ISO 717-2:2013 Annex C print: the 2020 reprint
# of this example says CI = -10 because its Ln,sum (83,5238 -> 84) erroneously
# includes the 3 150 Hz band, contradicting its own A.2.1 (100-2500 Hz);
# summing 100-2500 Hz gives 83,2613 -> 83 and CI = -11.
# ---------------------------------------------------------------------------
ISO717_2_ANNEX_C1_LN: list[float] = [
    62.1, 63.2, 63.5, 66.2, 68.5, 70.0, 71.7, 73.1,
    73.8, 73.5, 73.8, 73.3, 73.1, 73.0, 72.4, 71.2,
]
ISO717_2_ANNEX_C1_EXPECTED = {
    "ln_w": 79,
    "ci": -11,
    "unfavourable_sum": 28.0,
}
# Same Table C.1, right-hand columns: the floor WITH the floor covering.
# Ln,w = 64 dB, CI = -3 dB, unfavourable-deviation sum 30,0 dB.
ISO717_2_ANNEX_C1_COVERED_LN: list[float] = [
    59.1, 59.5, 61.6, 63.2, 65.3, 66.5, 67.7, 67.0,
    67.1, 66.5, 66.1, 62.5, 57.9, 52.7, 47.0, 48.0,
]
ISO717_2_ANNEX_C1_COVERED_EXPECTED = {
    "ln_w": 64,
    "ci": -3,
    "unfavourable_sum": 30.0,
}

# ---------------------------------------------------------------------------
# ISO 717-2 Annex C, Table C.2 - reduction of impact sound pressure level ΔL
# of a floor covering on the standard reference floor. The worked example
# gives ΔLw = 15 dB (Ln,r,w = 63 dB). The module additionally derives
# CI,Δ = CI,r,0 - CI,r = -11 - (-2) = -9 dB from the normative Table 4
# reference floor. The printed C.2 chain reaches CI,r = -3 because its
# "Ln,sum = 75,2527" is the energy sum of the wrong column over the wrong
# range (the measured floor with covering, 16 bands 100-3150 Hz); the A.2.1
# sum of the covered reference floor over 100-2500 Hz rounds to 76 dB with
# either the misprinted 71,0 or Table 4's 71,5 at 800 Hz (see docs/ERRATA.md).
# ---------------------------------------------------------------------------
ISO717_2_ANNEX_C2_DELTA_L: list[float] = [
    3.0, 3.7, 1.9, 3.0, 3.2, 3.5, 4.0, 6.1,
    6.7, 7.0, 7.7, 10.8, 15.2, 20.3, 25.4, 23.2,
]
ISO717_2_ANNEX_C2_DELTA_LW = 15
ISO717_2_ANNEX_C2_CI_DELTA = -9

# ---------------------------------------------------------------------------
# ISO 15186-1:2000 - sound insulation measured with sound intensity.
# The standard gives no fully worked numeric example, so the intensity sound
# reduction index RI = Lp1 - 6 - [LIn + 10 lg(Sm/S)] (Formula (7)) is anchored
# on the identity that, when the receiving-side intensity levels are chosen so
# that RI reproduces the ISO 717-1 Annex C airborne curve above, the ISO 717-1
# engine returns the same Rw = 30 dB through the intensity path. The
# adaptation term Kc (Annex B) oracle is the standard's own printed
# Table B.1 (21 one-third-octave rows, 50-5000 Hz, one decimal): the
# Formula (B.2) approximation Kc = 10 lg(1 + 61,4/f) reproduces every row at
# 1 dp, and Formula (B.1) with the reference room (Sb2 = 117 m², V2 = 81 m³,
# c = 340 m/s) reduces to (B.2) within 0,001 dB.
# ---------------------------------------------------------------------------
ISO15186_1_REF_LP1 = 85.0  # flat source-room level (dB)
ISO15186_1_REF_SM = 12.0  # measurement-surface area (m²)
ISO15186_1_REF_S = 10.0  # specimen area (m²)
ISO15186_1_REF_RI = ISO717_1_ANNEX_C_R  # target intensity SRI (16 bands)
ISO15186_1_REF_RIW = 30  # RI,w through the ISO 717-1 engine
# Printed Table B.1: (frequency_Hz, Kc_dB) as published (one decimal).
ISO15186_1_KC_TABLE_B1: list[tuple[float, float]] = [
    (50.0, 3.5), (63.0, 3.0), (80.0, 2.5), (100.0, 2.1), (125.0, 1.7),
    (160.0, 1.4), (200.0, 1.2), (250.0, 1.0), (315.0, 0.8), (400.0, 0.6),
    (500.0, 0.5), (630.0, 0.4), (800.0, 0.3), (1000.0, 0.3), (1250.0, 0.2),
    (1600.0, 0.2), (2000.0, 0.1), (2500.0, 0.1), (3150.0, 0.1), (4000.0, 0.1),
    (5000.0, 0.1),
]
ISO15186_1_KC_BANDS = tuple(f for f, _ in ISO15186_1_KC_TABLE_B1)
ISO15186_1_KC_B1_PRINTED = [kc for _, kc in ISO15186_1_KC_TABLE_B1]

# ---------------------------------------------------------------------------
# ISO 12999-2:2020 - measurement uncertainty for sound absorption.
# The standard's own worked examples are the oracle: Table 4 (sound absorption
# coefficient alpha_s and expanded uncertainty +/-U at k=2, reproducibility,
# one-third-octave 63-5000 Hz) and Table 5 (practical coefficient alpha_p,
# octave 250-4000 Hz). Example 1: alpha_w = 0,70 (MH) +/- 0,07 (k=2);
# Example 2: DLalpha,NRD = (8,1 +/- 1,6) dB (k=2).
# ---------------------------------------------------------------------------
ISO12999_2_TABLE4_FREQ = [
    63, 80, 100, 125, 160, 200, 250, 315, 400, 500,
    630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000,
]
ISO12999_2_TABLE4_ALPHA_S = [
    0.33, 0.35, 0.39, 0.38, 0.37, 0.36, 0.36, 0.36, 0.43, 0.49,
    0.58, 0.63, 0.68, 0.71, 0.73, 0.75, 0.77, 0.79, 0.81, 0.81,
]
ISO12999_2_TABLE4_U_K2 = [
    0.33, 0.26, 0.22, 0.17, 0.13, 0.11, 0.09, 0.08, 0.08, 0.08,
    0.08, 0.08, 0.08, 0.09, 0.09, 0.09, 0.10, 0.11, 0.13, 0.16,
]
ISO12999_2_TABLE5_FREQ = [250, 500, 1000, 2000, 4000]
ISO12999_2_TABLE5_ALPHA_P = [0.50, 0.65, 0.70, 0.85, 0.80]
ISO12999_2_TABLE5_U_K2 = [0.09, 0.08, 0.08, 0.08, 0.10]
ISO12999_2_ALPHA_W_EXAMPLE = 0.70
ISO12999_2_ALPHA_W_U_K2 = 0.07
ISO12999_2_DLALPHA_EXAMPLE = 8.1
ISO12999_2_DLALPHA_U_K2 = 1.6

# ---------------------------------------------------------------------------
# ISO 16251-1:2014 - impact sound improvement of floor coverings (mock-up).
# The standard's Annex B "Table B.1" is a blank report form (no numeric worked
# example), so the conformance anchor is the ISO 717-2:2020 reference floor:
# weighted_impact_rating(Ln,r,0) must return exactly 78 dB (CI = -11), and a
# zero improvement must give Delta-Lw = 0 (Formula 2: Delta-Lw = 78 - Ln,r,w).
# ---------------------------------------------------------------------------
ISO717_2_REFERENCE_FLOOR_FREQ = [
    100, 125, 160, 200, 250, 315, 400, 500,
    630, 800, 1000, 1250, 1600, 2000, 2500, 3150,
]
ISO717_2_REFERENCE_FLOOR_LN_R0 = [
    67.0, 67.5, 68.0, 68.5, 69.0, 69.5, 70.0, 70.5,
    71.0, 71.5, 72.0, 72.0, 72.0, 72.0, 72.0, 72.0,
]
ISO717_2_REFERENCE_FLOOR_LN_R0_W = 78
ISO717_2_REFERENCE_FLOOR_CI = -11

# ---------------------------------------------------------------------------
# ISO 16251-1 real measured oracle - textile carpet on the heavyweight mock-up.
# Source: R. Foret, J.-B. Chene, C. Guigou-Carter, "A comparison of the
# reduction of transmitted impact noise by floor coverings measured using
# ISO 140-8 and ISO/CD 16251-1", Forum Acusticum 2011, Aalborg (CSTB), Figure
# 4 (textile floor covering, not bonded). The standard itself carries no
# numeric worked example, so this is the anchoring real measurement.
#
# The improvement of impact sound insulation Delta-L is only PLOTTED, but the
# chart is a Microsoft Office vector object (the two data series are vector
# paths, not a raster; only the test-setup photographs in the paper are
# raster), so the per-band spectrum below was digitized from the vector
# markers. Axis calibration: the nine major horizontal gridlines span 0 to
# 80 dB over 1549 px at 600 dpi (0,0517 dB/px); each data point is the colour
# segmented centroid of the red ISO 16251-1 square markers. Digitization
# tolerance +/- 0,5 dB per band. The ISO 717-2 single-number rating is
# invariant to that band error and to rounding the spectrum to integers or to
# half-decibels. End-check: the paper's published Delta-Lw = 29 dB
# (ISO 16251-1) is reproduced exactly; the companion ISO 140-8 series, read the
# same way, reproduces its published Delta-Lw = 30 dB, independently
# confirming the calibration.
# ---------------------------------------------------------------------------
FORET2011_CARPET_FREQ = [
    100, 125, 160, 200, 250, 315, 400, 500, 630, 800,
    1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000,
]
FORET2011_CARPET_ISO16251_DELTA_L: list[float] = [
    5.0, 8.0, 10.0, 14.0, 18.0, 23.0, 30.0, 31.0, 39.0, 49.0,
    53.0, 57.0, 60.0, 67.0, 68.0, 71.0, 74.0, 72.0,
]
FORET2011_CARPET_ISO16251_DELTA_LW = 29

# ---------------------------------------------------------------------------
# ISO 10848-1:2006 — flanking transmission (vibration reduction index Kij).
# The standard contains NO worked numeric example anywhere in Parts 1-4, so
# these are closed-form identities synthesized to hand-checkable results (the
# clean-room oracle strategy). Each expected value is a literal computed by
# hand from the transcribed formula so it is independent of the library code.
# ---------------------------------------------------------------------------
# Simplified Kij, Formula (14): Kij = D̄v,ij + 10 lg(lij / sqrt(Si·Sj)).
# D̄v,ij = 5 dB, lij = 2 m, Si = Sj = 4 m² -> 10 lg(2/4) = -3.0102999566 dB.
ISO10848_KIJ_DBAR = 5.0
ISO10848_KIJ_LIJ = 2.0
ISO10848_KIJ_AREA = 4.0
ISO10848_KIJ_SIMPLIFIED = 1.9897000434  # 5 + 10*log10(0.5)
# Equivalent absorption length, Formula (12), at f = f_ref = 1000 Hz (so the
# sqrt(f_ref/f) factor is 1): aj = 2.2·π²·S/(Ts·c0).
# S = 10 m², Ts = 0.5 s, c0 = 343 m/s -> 2.2·π²·10/(0.5·343).
ISO10848_ABS_AREA = 10.0
ISO10848_ABS_TS = 0.5
ISO10848_ABS_C0 = 343.0
ISO10848_ABS_LENGTH_AT_FREF = 1.2660717015974685  # 2.2*pi**2*10/(0.5*343)
# Total loss factor, η = 2.2/(f·Ts). f = 1000 Hz, Ts = 0.5 s -> 0.0044 exactly.
ISO10848_LOSS_FACTOR = 0.0044

# ---------------------------------------------------------------------------
# ISO 1996-2 tonal audibility -- Annex C.5 worked examples (2007/2009 edition).
# Each row: (tone level Lpt, masking noise Lpn, band centre fc, printed ΔLta,
# printed Kt). Examples 1/2/4 reproduce Formula (C.3) to < 0.05 dB; example 3
# is printed 10.6 but (C.3) gives ~11.2 (rounding in the printed figure), so it
# is used only as a loose check.
ISO1996_2_TONAL_EXAMPLES = [
    (46.7, 37.3, 4000.0, 13.7, 6.0),   # Example 1 (Fig C.3)
    (54.1, 45.2, 430.0, 11.1, 6.0),    # Example 2 (Fig C.4)
    (53.6, 45.5, 755.0, 10.7, 6.0),    # Example 4 (Fig C.6)
]
ISO1996_2_TONAL_EXAMPLE3 = (54.6, 45.5, 308.0, 10.6, 6.0)  # loose (rounding)
# Annex G.2 -- single 1 h measurement uncertainty budget. The tabulated
# per-component products cj*uj (dB) combine to u = 2.18 dB; expanded (k = 2) 4.36.
ISO1996_2_G2_CONTRIBUTIONS = [0.59, 0.3, 2.0, 0.40, 0.38]
ISO1996_2_G2_COMBINED = 2.18
ISO1996_2_G2_EXPANDED = 4.36

# ---------------------------------------------------------------------------
# Reverberation-time prediction -- real worked oracle.
# F. A. Everest & K. C. Pohlmann, *Master Handbook of Acoustics*, 4th ed.,
# Fig. 7-22 "Reverberation Calculation: Example 1": an untreated 23.3 x 16 x 10 ft
# room (concrete floor + 1/2" gypsum-board walls/ceiling) solved with the Sabine
# equation RT60 = 0.049 V / Sa (imperial). Converting the areas and volume to SI
# and evaluating the module's SI Sabine (k = 24 ln10 / 343) reproduces the six
# printed reverberation times to <= 0.012 s (the residual is the book rounding
# the imperial 0.049 constant). This anchors the whole family on measured
# material data, not only on the closed-form identities.
_FT2 = 0.3048 ** 2   # square foot -> m2
_FT3 = 0.3048 ** 3   # cubic foot -> m3
EVEREST_EX1_VOLUME = 3728.0 * _FT3            # 105.565 m3
EVEREST_EX1_FLOOR_AREA = 373.0 * _FT2         # concrete floor, 34.653 m2
EVEREST_EX1_SHELL_AREA = 1159.0 * _FT2        # gypsum walls+ceiling, 107.675 m2
EVEREST_EX1_BANDS = [125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0]
EVEREST_EX1_FLOOR_ALPHA = [0.01, 0.01, 0.015, 0.02, 0.02, 0.02]   # concrete
EVEREST_EX1_SHELL_ALPHA = [0.29, 0.10, 0.05, 0.04, 0.07, 0.09]    # gypsum board
EVEREST_EX1_RT = [0.54, 1.53, 2.87, 3.39, 2.06, 1.63]             # printed RT60, s

# ---------------------------------------------------------------------------
# ISO/PAS 20065:2016 tonal audibility -- Annex E combustion-engine example.
# Spectrum 1 (the one with the greatest decisive audibility), Tables E.2/E.3;
# line spacing Δf = 2.7 Hz (Table E.1). The engineering-method chain
# (LG Formula (12), av Formula (13), ΔL Formula (14)) reproduces the printed
# per-tone ΔL to <= 0.03 dB (the residual is the standard's 2-decimal
# rounding). Each tone: (fT, LS, LT, printed ΔL).
ISO20065_LINE_SPACING = 2.7
ISO20065_ANNEX_E_TONES = [
    (118.4, 48.91, 64.56, 1.92),
    (137.3, 49.22, 67.96, 4.99),
    (158.8, 50.50, 68.63, 4.37),
    (314.9, 52.85, 68.50, 1.78),
    (433.4, 58.29, 73.17, 0.87),
    (592.2, 59.53, 78.31, 4.55),
    (629.8, 59.71, 75.00, 1.01),
    (643.3, 61.98, 79.75, 3.47),
    (1582.7, 54.16, 71.07, 0.73),
]
# Table E.2: masking index av at 137.3 Hz and 592.2 Hz (Formula (13)).
ISO20065_AV_137 = -2.02
ISO20065_AV_592 = -2.40
# Table E.2: critical-band level LG at 137.3 Hz (Formula (12)), LS = 49.22 dB.
ISO20065_LG_137 = 64.98
# Decisive audibilities ΔLj of the five staggered spectra (Table E.3/E.4, bold)
# and their energy-mean mean audibility ΔL (Formula (20)); printed ΔL = 6.96 dB.
ISO20065_DECISIVE_AUDIBILITIES = [9.18, 6.04, 7.46, 2.67, 7.17]
ISO20065_MEAN_AUDIBILITY = 6.96

# Table E.1 -- the 38 narrow-band lines (fi, Li) of the critical band about the
# 137.3 Hz tone of spectrum 1 (Δf = 2.7 Hz). The from-spectrum front-end must
# reproduce the tabulated mean narrow-band level LS2 = 49.22 dB (Formula (6),
# iterative Annex D, with the −1.76 dB Hanning correction confirmed by
# DIN 45681:2005-03 5.3.2) and tone level LT2 = 67.96 dB (Formula (8)).
ISO20065_E1_FREQUENCIES = [
    96.9, 99.6, 102.3, 105.0, 107.7, 110.4, 113.0, 115.7, 118.4, 121.1,
    123.8, 126.5, 129.2, 131.9, 134.6, 137.3, 140.0, 142.7, 145.3, 148.0,
    150.7, 153.4, 156.1, 158.8, 161.5, 164.2, 166.9, 169.6, 172.3, 175.0,
    177.6, 180.3, 183.0, 185.7, 188.4, 191.1, 193.8, 196.5,
]
ISO20065_E1_LEVELS = [
    49.40, 50.68, 50.09, 53.37, 44.47, 50.91, 51.41, 59.40, 64.54, 57.57,
    51.02, 50.76, 59.93, 62.94, 58.49, 65.87, 62.66, 50.25, 51.32, 52.30,
    52.58, 53.15, 67.04, 67.27, 57.40, 57.17, 52.56, 51.39, 52.49, 47.68,
    51.26, 49.03, 61.42, 59.52, 48.43, 50.84, 48.20, 55.95,
]
ISO20065_E1_LS = 49.22   # mean narrow-band level of the 137.3 Hz tone (Table E.1)
ISO20065_E1_LT = 67.96   # tone level of the 137.3 Hz tone (single, Table E.2)
# Full-spectrum front-end on Table E.1: peak detection finds the three tones,
# and the multi-tone "FG" combination (Formula 17) sums their tonal lines to
# LT = 72.15 dB (Table E.2 row "2 FG"). Their mean narrow-band levels (Table E.2)
# are LS = 48.91 / 49.22 / 50.50 dB. (The FG decisive audibility ΔL = 9.18 dB
# needs the *complete* spectrum: Table E.1 is truncated to the 137.3 Hz critical
# band, so the 158.8 Hz tone's LS is underestimated from it.)
ISO20065_E1_TONE_FREQUENCIES = [118.4, 137.3, 158.8]
ISO20065_E1_TONE_LS = [48.91, 49.22, 50.50]
ISO20065_E1_LT_FG = 72.15

# Table E.2 full columns for spectrum 1, k = 1..9 (same tone order as
# ISO20065_ANNEX_E_TONES): critical-band level LG (Formula (12)), masking
# index av (Formula (13)), extended uncertainty U of the individual spectrum
# (Clause 6) and the printed band limits (f1, f2). The printed f1/f2 are the
# first/last FFT lines *inside* the analytic Formula (4)/(5) band, not the
# analytic corners themselves (line-snapped).
ISO20065_E2_LG = [64.66, 64.98, 66.28, 68.84, 74.52, 76.16, 76.44, 78.74, 73.60]
ISO20065_E2_AV = [-2.01, -2.02, -2.02, -2.12, -2.23, -2.40, -2.44, -2.46, -3.27]
ISO20065_E2_U = [3.66, 2.79, 3.51, 2.46, 3.09, 2.82, 2.67, 3.56, 2.27]
ISO20065_E2_BAND_LIMITS = [
    (80.70, 177.60), (96.90, 196.50), (118.40, 215.30), (266.50, 371.40),
    (382.20, 492.60), (535.60, 656.80), (570.60, 694.40), (584.10, 707.90),
    (1469.60, 1703.80),
]
# Table E.2 "2 FG" row (the decisive FG group at 137.3 Hz): U = 3.21 dB. The
# Clause 6 note for summated tones reads "the sum of all tone-containing
# narrow-band levels ... is to be used for K": the reading that reproduces
# the printed value uses the N summated TONE levels (E.2 rows 1-3) as the K
# summands (3.215 dB); the union of the individual tonal lines gives 2.18 dB.
ISO20065_E2_FG_U = 3.21
ISO20065_E2_FG_TONE_LEVELS = [64.56, 67.96, 68.63]  # LT of tones 1..3
# Table E.1 NOTE 1 prints LS1 = 49.91 / LS3 = 49.90 dB for the flanking tones
# computed from the truncated E.1 band. These do NOT reproduce from the E.1
# lines with the Annex D iteration (measured 49.51 / 49.36 dB) under any
# tested exclusion-rule variant; the note's exact recipe is unstated, so the
# values are recorded here without an assertion. (The full-spectrum Table E.2
# values, 48.91 / 50.50 dB, are the pinned oracles.)
ISO20065_E1_NOTE1_LS = {118.4: 49.91, 158.8: 49.90}

# Table E.4 - decisive-tone parameters of the five staggered spectra:
# (fT, printed dL, LS, LT, LG, av, U). FG rows carry the combined LT of
# Formula (17). The audibility chain dL = LT - LG - av reproduces the
# printed dL to <= 0.03 dB (2-decimal rounding of the intermediates).
ISO20065_E4_DECISIVE_ROWS = [
    (137.3, 9.18, 49.22, 72.15, 64.98, -2.02, 3.21),
    (430.7, 6.04, 55.06, 75.11, 71.29, -2.23, 2.95),
    (137.3, 7.46, 50.40, 71.61, 66.16, -2.02, 2.44),
    (433.4, 2.67, 55.12, 71.79, 71.35, -2.23, 2.52),
    (137.3, 7.17, 50.70, 71.61, 66.46, -2.02, 2.14),
]
# Annex E Step 4: extended uncertainty of the mean audibility over the five
# spectra (Formulae (28)/(29)), printed U = +/-1.38 dB, checked against the
# 1.4 dB margin for < 12 spectra.
ISO20065_E4_MEAN_UNCERTAINTY = 1.38

# Table E.3 - all tonal components of the five spectra: per spectrum j, the
# printed (fT, dL) pairs, plus the FG audibilities where several tones share
# a critical band. The decisive audibility of each spectrum (bold in the
# print) is the maximum over both lists and reproduces
# ISO20065_DECISIVE_AUDIBILITIES; the narrow-band lines of spectra 2-5 are
# not printed, so these rows serve as a data-consistency record rather than
# a from-levels chain.
ISO20065_E3_TONES = {
    1: [(118.4, 1.92), (137.3, 4.99), (158.8, 4.37), (314.9, 1.78),
        (433.4, 0.87), (592.2, 4.55), (629.8, 1.01), (643.3, 3.47),
        (1582.7, 0.73)],
    2: [(156.1, 0.52), (430.7, 6.04), (465.7, 0.60), (963.6, 4.11),
        (1512.7, 0.27), (1590.8, 3.42)],
    3: [(118.4, 1.77), (137.3, 2.99), (158.8, 2.71), (433.4, 1.78),
        (589.5, 2.56), (643.3, 1.40), (963.6, 0.34), (1512.7, 2.44),
        (1580.0, 3.48)],
    4: [(156.1, 0.65), (433.4, 2.67), (465.7, 0.25), (643.3, 0.40),
        (707.9, 0.35), (963.6, 1.61), (1580.0, 2.14)],
    5: [(118.4, 1.48), (137.3, 2.95), (156.1, 1.50), (433.4, 0.93),
        (640.6, 2.63), (699.8, 1.73), (942.1, 0.00), (960.9, 1.88),
        (1512.7, 0.37), (1590.8, 2.52)],
}
ISO20065_E3_FG = {
    1: [(137.3, 9.18), (592.2, 9.12)],
    2: [(1590.8, 3.52)],
    3: [(137.3, 7.46), (1580.0, 4.48)],
    4: [],
    5: [(137.3, 7.17), (960.9, 2.32), (1590.8, 2.82)],
}

# Two-tone separation frequency fD (Formulae (18)/(19), Clause 5.3.8): two tones
# in one critical band, both below 1000 Hz, are rated separately (not FG-combined)
# when |fT1 − fT2| exceeds fD = 21·10^(1.2·|lg(fT/212)|^1.8) Hz, evaluated at the
# more prominent tone. No ISO/PAS 20065 worked example exercises this branch (the
# Annex E band groups three tones); the values below reproduce the DIN 45681:2005-03
# Annex J reference program (fD = 21 * 10 ^ (1.2 * Abs(Log(fT / 212) / Log(10)) ^ 1.8)).
# fD bottoms out at 21 Hz at the reference fT = 212 Hz (the |lg| minimum).
ISO20065_FD_212 = 21.00
ISO20065_FD_137 = 24.09

# ---------------------------------------------------------------------------
# DIN 45681:2005-03 Anhang I, Beispiel I.3 -- wind-energy-plant example (the
# parent standard's second end-to-end oracle, independent of the ISO/PAS 20065
# Annex E combustion-engine example above). Line spacing 2.6917 Hz.
# ---------------------------------------------------------------------------
# Tabelle I.9: the 39 narrow-band lines (fi, Li) of the critical band about
# the 298.8 Hz decisive tone of spectrum j = 24 (of 53).
DIN45681_I9_FREQUENCIES = [
    253.0, 255.7, 258.4, 261.1, 263.8, 266.5, 269.2, 271.9, 274.5, 277.2,
    279.9, 282.6, 285.3, 288.0, 290.7, 293.4, 296.1, 298.8, 301.5, 304.2,
    306.8, 309.5, 312.2, 314.9, 317.6, 320.3, 323.0, 325.7, 328.4, 331.1,
    333.8, 336.5, 339.1, 341.8, 344.5, 347.2, 349.9, 352.6, 355.3,
]
DIN45681_I9_LEVELS = [
    37.83, 38.45, 38.23, 38.74, 40.37, 47.09, 45.59, 43.27, 46.73, 46.54,
    51.81, 49.74, 50.95, 52.50, 55.45, 55.32, 61.55, 66.68, 64.77, 57.80,
    55.75, 43.29, 46.26, 47.47, 46.22, 40.64, 38.10, 42.13, 46.34, 39.03,
    37.42, 42.59, 41.93, 41.81, 50.24, 52.30, 48.38, 36.27, 40.63,
]
DIN45681_LINE_SPACING = 2.6917
# Tabelle I.10 decisive-tone row (j = 24, k = 2): fT, dL, LS, LT, LG, av, u.
DIN45681_I10_DECISIVE = (298.8, 12.52, 41.71, 68.10, 57.68, -2.10, 3.18)
# Tabelle I.10 rows k = 4 and k = 5 (705.2 / 732.1 Hz) share one critical
# band; both lie below 1000 Hz and their spacing (26.9 Hz) is below the
# Formula (19) separation frequency, so the print combines them into the
# "5 FG" row: (fT, dL, LS, LT_FG, LG, av, u).
DIN45681_I10_K4 = (705.2, 1.35, 39.35, 55.12)     # (fT, dL, LS, LT)
DIN45681_I10_K5 = (732.1, 1.51, 38.26, 54.23)
DIN45681_I10_5FG = (732.1, 3.22, 38.26, 55.95, 55.28, -2.55, 3.67)
# Tabelle I.11 (parameters of the 53 spectra), rows j = 45 and j = 48:
# (fT, dL, LS, LT, LG, av, u). The from-levels chain (Formulae (12)-(14))
# reproduces the printed dL/LG/av columns.
DIN45681_I11_J45 = (258.4, 3.04, 42.24, 59.11, 58.14, -2.08, 2.69)
DIN45681_I11_J48 = (228.8, 6.11, 38.32, 58.24, 54.19, -2.06, 3.00)
# Tabelle I.6 row "6 FG" (combustion-engine spectrum 1, the ISO Annex E
# example): tones k = 6/7/8 (LT = 78.31 / 75.00 / 79.75 dB) share the 592.2 Hz
# critical band. The printed dL = 9.12 dB reproduces from the *plain*
# Formula (17) energy sum of the three tone levels (82.87 dB) through the
# audibility chain at 592.2 Hz (LS = 59.53); the printed LT column (81.11 dB)
# is consistent with the Anmerkung-2 shared-line dedupe instead and does NOT
# reproduce the printed dL -- the two printed cells contradict each other, so
# only the dL chain is pinned.
DIN45681_I6_6FG_TONE_LEVELS = [78.31, 75.00, 79.75]
DIN45681_I6_6FG = (592.2, 9.12, 59.53, -2.40)     # (fT, dL, LS, av)
# Anhang I.3 Step 3/5: mean audibility over the 53 spectra and the resulting
# tone adjustment (DIN Abschnitt 6 Tabelle 1 == ISO 1996-2 Table J.1).
DIN45681_I3_MEAN_AUDIBILITY = 6.38
DIN45681_I3_KT = 4
# Tabelle A.1 (informative): printed critical bandwidths dfc (Hz, integer)
# of the frequency groups at the tabulated tone frequencies fT. Every row
# matches Formula (2) to <= 0.5 Hz except 250 Hz, where the print gives
# 105 Hz while Formula (2) yields 104.47 Hz (integer-rounds to 104); the
# table cites 5.2 for the computation, so the 250 Hz cell is a print quirk
# (possibly carried over from Zwicker's literature table).
DIN45681_A1_BANDWIDTHS = [
    (100.0, 101.0), (150.0, 102.0), (250.0, 105.0), (350.0, 109.0),
    (450.0, 114.0), (570.0, 122.0), (700.0, 133.0), (840.0, 145.0),
    (1000.0, 162.0), (1170.0, 182.0), (1370.0, 207.0), (1600.0, 239.0),
    (1850.0, 277.0), (2150.0, 325.0), (2500.0, 386.0), (2900.0, 460.0),
    (3400.0, 559.0), (4000.0, 685.0), (4800.0, 867.0), (5800.0, 1111.0),
    (7000.0, 1426.0), (8500.0, 1851.0), (10500.0, 2463.0), (13500.0, 3469.0),
]

# ---------------------------------------------------------------------------
# Psychoacoustic annoyance (Fastl & Zwicker Eq. 16.2-16.4; Widmann 1992) and
# fluctuation strength (Fastl & Zwicker Ch. 10; Osses et al. 2016).
# ---------------------------------------------------------------------------
# PA is exact. Worked tuple (N5, S, F, R) = (30 sone, 2.0 acum, 0.5 vacil,
# 0.3 asper) -> the terms and PA computed by hand from Eqs 16.2-16.4 (the
# "1 +" of Eq. (16.2) sits OUTSIDE the radical; F&Z 2006, p. 328):
#   wS  = (2.0 - 1.75) * 0.25 * lg(30 + 10)          = 0.100129
#   wFR = (2.18 / 30**0.4) * (0.4*0.5 + 0.6*0.3)     = 0.212516
#   PA  = 30 * (1 + sqrt(wS**2 + wFR**2))            = 37.0478
PA_WORKED_INPUT = (30.0, 2.0, 0.5, 0.3)  # (N5, S, F, R)
PA_WORKED_WS = 0.100129
PA_WORKED_WFR = 0.212516
PA_WORKED_VALUE = 37.0478

# Fluctuation strength closed form for AM broadband noise (Fastl & Zwicker
# Eq. 10.2), exact. F(L=60 dB, m=1, fmod=4 Hz):
#   5.8*(1.25-0.25)*(0.05*60-1) / ((4/5)**2 + (4/4) + 1.5) = 11.6/3.14 = 3.6943
FS_BBN_60_1_4 = 3.6943
# Fluctuation strength calibration definition: 1 kHz tone, 60 dB, m=1, 4 Hz AM
# is 1 vacil (Fastl & Zwicker Ch. 10; the signal model is anchored to it).
FS_CALIBRATION_VACIL = 1.00

# Fluctuation-strength signal-model cross-check (Osses 2016 Table 1): literature
# values for a 1 kHz AM tone at 70 dB, m=1, over fmod = {1,2,4,8,16,32} Hz. No
# numeric standard exists; the Osses model reproduces these TRENDS (Pearson
# r >= 0.9, band-pass peak at 4 Hz, within ~2.1x), not the exact figures.
FS_AM_TONE_FMOD_HZ = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
FS_AM_TONE_70DB_LITERATURE = [0.39, 0.84, 1.25, 1.30, 0.36, 0.06]

# Carrier-frequency sweep of an AM tone (70 dB, m=1, fmod=4 Hz) at
# fc = {125, 250, 500, 1000, 4000, 8000} Hz. Values measured through this
# implementation with the corrected Zwicker-Terhardt Bark constant (0.76e-3;
# Osses 2016 Eq. 3 misprints 0.76e-4, see docs/ERRATA.md), reproducing the
# Fastl & Zwicker Fig. 10.5 trend: a low-mid carrier plateau and a roll-off
# at 8 kHz. Measured-by-reviewer values, hence the generous tolerances.
FS_CARRIER_SWEEP_HZ = [125.0, 250.0, 500.0, 1000.0, 4000.0, 8000.0]
FS_CARRIER_SWEEP_VACIL = [0.86, 1.25, 1.03, 1.09, 0.92, 0.58]

# Osses 2016 Table 1, AM broadband noise (BW 16 kHz, 60 dB, m=1) over
# fmod = {1,2,4,8,16,32} Hz: literature values in vacil. The excitation
# front-end spreads the modulated energy across bands and overshoots the
# absolute pass-band level by up to ~3x, so this row is a TREND cross-check
# (band-pass shape, Pearson correlation, high-fmod tail) only. The FM-tone
# row of Table 1 is not pinned at all: the model documentedly does not
# pursue FM accuracy (the reference method itself overestimates it > 4 Hz).
FS_AM_BBN_60DB_LITERATURE = [1.12, 1.58, 1.80, 1.57, 0.48, 0.14]

# ---------------------------------------------------------------------------
# Electroacoustic distortion (IEC 60268-3:2013) and frequency response
# (Bendat & Piersol, Random Data 4e). All quantities are exact analytic
# oracles evaluated on synthetic signals with known harmonic / intermodulation
# amplitudes, or on a known LTI path.
# ---------------------------------------------------------------------------
# A 1 kHz fundamental (a1 = 1) with harmonics a2 = 0.1, a3 = 0.05, a4 = 0.02.
#   THD_F = sqrt(a2^2 + a3^2 + a4^2) / a1            = 0.1135782
#   THD_R = sqrt(a2^2 + a3^2 + a4^2) / sqrt(sum a^2) = 0.1128526
#   d2    = a2 / sqrt(sum a^2)                       = 0.0993612
DISTORTION_HARMONICS = (1.0, 0.1, 0.05, 0.02)  # a1..a4
DISTORTION_THD_F = 0.11357816691600547
DISTORTION_THD_R = 0.11285260010027609
DISTORTION_D2 = 0.09936117403949127

# Clipped-sine THD oracle: a unit sine symmetrically clipped at 0.7, sampled
# at 48 samples per period, has these odd-harmonic Fourier amplitudes and
# THD_F over n <= 10 (independent single-period Fourier series of the sampled
# waveform). The continuous-time fundamental is b1 = (2/pi)(arcsin 0.7 +
# 0.7 sqrt(0.51)) = 0.8118795956258127; the sampled value differs by the
# 6.5e-4 aliasing of the clipped wave's high harmonics, so the sampled value
# is the one pinned here.
CLIPPED_SINE_THD_F = 0.13794482640558078
CLIPPED_SINE_B1 = 0.8124127489373637
CLIPPED_SINE_B3 = 0.1087038092372312
CLIPPED_SINE_B5 = 0.0205013791213361
CLIPPED_SINE_B7 = 0.0165310026995253
CLIPPED_SINE_B9 = 0.0070120099075438

# Ordinary coherence of a signal-plus-independent-noise output with a flat
# (frequency-independent) SNR: gamma^2 = SNR / (1 + SNR). At SNR = 10 -> 0.90909.
COHERENCE_SNR = 10.0
COHERENCE_EXPECTED = COHERENCE_SNR / (1.0 + COHERENCE_SNR)

# ---------------------------------------------------------------------------
# Underwater acoustics (ISO 18405 / 17208 / 18406). Reference pressure 1 µPa,
# reference exposure 1 µPa²·s. Level offset between the in-air (20 µPa) and
# underwater (1 µPa) references: 20·lg(20) = 26.0206 dB.
# ---------------------------------------------------------------------------
UW_REFERENCE_OFFSET_DB = 26.020599913279624

# ---------------------------------------------------------------------------
# EN 12354-5:2009 Annex I - installed structure-borne sound worked examples
# (octave bands 63-2000 Hz). The printed tables carry one-decimal
# intermediates, so chained values reproduce within +/-0,15 dB.
# ---------------------------------------------------------------------------
EN12354_5_ANNEX_I_BANDS: tuple[float, ...] = (63, 125, 250, 500, 1000, 2000)

# I.2 whirlpool bath (Tables I.6a/I.7). Floor power component: the laboratory
# characteristic reception-plate levels L_Ws,n,1 (re Y_inf,rec = 5e-6 m/Ns)
# are corrected to the installed floor (Y_inf,1 = 1.25e-6 m/Ns -> -6,0 dB);
# path 11 then follows Formula (18a) with the -4 dB area/absorption terms of
# the example (S_i = S0 = 10 m2). Table I.7 totals the whirlpool at 26 dB(A).
EN12354_5_I6A_LWSN_FLOOR = [67.6, 67.3, 64.4, 48.4, 42.5, 41.3]
EN12354_5_I6A_LWSN_INST_FLOOR = [61.6, 61.3, 58.4, 42.4, 36.5, 35.3]
EN12354_5_I6A_Y_FLOOR = 1.25e-6  # m/(N.s)
EN12354_5_I6A_DSA_FLOOR = [-26.1, -24.8, -30.3, -36.6, -40.8, -46.6]
EN12354_5_I6A_R11 = [48.4, 48.9, 57.3, 66.2, 72.9, 81.2]
EN12354_5_I6A_LNS_11 = [35.4, 33.3, 27.4, 8.8, 0.4, -3.3]

# I.3 flushing cistern (Tables I.8/I.9). Source measured on a reception plate
# of Y_plate = 5.34e-6 m/Ns; characteristic level L_Ws,c via
# +10 lg(Y_source/Y_plate) (Y_source = 1.0e-3 m/Ns); D_C per Formula (19c);
# Dsa per (20b); four paths per Formula (18a); total per Formula (17).
EN12354_5_I8_PLATE_MOBILITY = 5.34e-6  # m/(N.s)
EN12354_5_I8_Y_SOURCE = 1.0e-3  # m/(N.s)
EN12354_5_I8_Y_WALL = 24.1e-6  # m/(N.s)
EN12354_5_I8_Y_FLOOR = 1.65e-6  # m/(N.s)
EN12354_5_I8_WALL_LWS = [61.7, 59.8, 47.2, 44.9, 38.8, 27.2]  # measured
EN12354_5_I8_WALL_INSTALLED = [68.2, 66.3, 53.7, 51.5, 45.4, 33.7]
EN12354_5_I8_WALL_LWSC = [84.4, 82.5, 69.9, 67.6, 61.6, 49.9]
EN12354_5_I8_FLOOR_LWS = [57.4, 56.2, 44.0, 42.4, 34.9, 28.9]  # measured
EN12354_5_I8_FLOOR_INSTALLED = [52.3, 51.1, 38.9, 37.3, 29.8, 23.8]
EN12354_5_I8_FLOOR_LWSC = [80.1, 78.9, 66.7, 65.1, 57.6, 51.6]
EN12354_5_I9_DC_WALL = 16.2  # dB, all bands
EN12354_5_I9_DC_FLOOR = 27.8  # dB, all bands
EN12354_5_I9_DSA_WALL = [-13.6, -17.3, -17.4, -20.0, -26.9, -32.9]
EN12354_5_I9_DSA_FLOOR = [-15.5, -19.4, -26.7, -33.2, -39.1, -44.8]
EN12354_5_I9_S_WALL = 12.8  # m2
EN12354_5_I9_S_FLOOR = 15.4  # m2
EN12354_5_I9_R_WALL_FLOOR = [43.0, 46.0, 50.2, 54.7, 64.6, 73.0]
EN12354_5_I9_R_WALL_WALL = [37.0, 41.2, 35.9, 37.7, 49.0, 57.8]
EN12354_5_I9_R_FLOOR_FLOOR = [42.4, 45.9, 50.1, 54.7, 64.6, 73.0]
EN12354_5_I9_R_FLOOR_WALL = [29.1, 32.3, 43.7, 53.5, 62.1, 70.1]
EN12354_5_I9_LNS_WALL_FLOOR = [33.8, 32.6, 15.9, 11.7, 2.6, -11.4]
EN12354_5_I9_LNS_WALL_WALL = [39.8, 37.4, 30.1, 28.7, 18.3, 3.8]
EN12354_5_I9_LNS_FLOOR_FLOOR = [19.5, 18.7, 9.7, 9.9, -1.5, -10.3]
EN12354_5_I9_LNS_FLOOR_WALL = [32.8, 32.3, 16.1, 11.1, 1.0, -7.4]
EN12354_5_I9_LNS_TOTAL = [41.4, 39.6, 30.5, 28.9, 18.5, 4.4]
EN12354_5_I9_LNS_TOTAL_A = 29  # dB(A)
EN12354_5_ANNEX_I_TOL = 0.15  # dB - one-decimal table intermediates

# ---------------------------------------------------------------------------
# ISO 9611:1996 - characterization of structure-borne sound sources by the
# free velocity of the contact points. Equation (9) mean velocity level over
# N positions (energy mean), v0 = 5e-8 m/s (clause 7). No numeric example in
# the standard; the anchor is the closed form recomputed by hand:
# levels 70/72/74 dB -> 10 lg((10^7 + 10^7.2 + 10^7.4)/3) = 72.3017 dB.
# ---------------------------------------------------------------------------
ISO9611_MEAN_LEVELS = (70.0, 72.0, 74.0)
ISO9611_MEAN_EXPECTED = 72.30174601124772
ISO9611_FREE_VELOCITY_REFERENCE = 5.0e-8  # m/s

# ---------------------------------------------------------------------------
# ISO 10140-5:2010+A1 - reference building elements (real printed end-to-end
# anchors). Annex B Table B.1 gives the sound reduction index R of three
# airborne reference elements (16 one-third-octave bands 100-3150 Hz) with
# their printed weighted ratings; Annex C Table C.1 gives the normalized
# impact sound pressure levels of two lightweight reference floors with
# their printed Ln,t,r,0,w (CI).
# ---------------------------------------------------------------------------
ISO10140_5_B1_HEAVY_WALL_R: list[float] = [
    40, 40, 40, 40, 41, 43.5, 46.1, 48.5,
    51, 53.6, 56, 58.4, 61.1, 63.6, 65, 65,
]
ISO10140_5_B1_HEAVY_WALL_RATING = (53, -1, -5)  # Rw (C; Ctr)
ISO10140_5_B1_HEAVY_FLOOR_R: list[float] = [
    40, 40, 40, 40, 40, 41.8, 44.4, 46.8,
    49.3, 51.9, 54.4, 56.8, 59.5, 61.9, 64.3, 65,
]
ISO10140_5_B1_HEAVY_FLOOR_RATING = (52, -1, -5)
ISO10140_5_B1_LIGHT_WALL_R: list[float] = [
    27, 27, 27, 27, 27, 27, 27, 27,
    28, 30.5, 32.8, 35.1, 37.6, 40, 42.3, 44.6,
]
ISO10140_5_B1_LIGHT_WALL_RATING = (33, -1, -2)
ISO10140_5_C1_FLOOR_C1C2_LN: list[float] = [
    78, 78, 78, 78, 78, 78, 76, 74, 72, 69, 66, 63, 60, 57, 54, 51,
]
ISO10140_5_C1_FLOOR_C1C2_RATING = (72, 0)  # Ln,t,r,0,w (CI)
ISO10140_5_C1_FLOOR_C3_LN: list[float] = [
    69, 72, 75, 78, 78, 78, 78, 78, 78, 76, 74, 72, 69, 66, 63, 60,
]
ISO10140_5_C1_FLOOR_C3_RATING = (75, -3)

# ---------------------------------------------------------------------------
# ITU-R BS.1770-5 / EBU R 128 - programme loudness and true peak.
# BS.1770-5 Annex 1 anchors a 0 dB FS 997 Hz sine on one front channel at
# -3.01 LKFS. EBU Tech 3341 Table 1 gives the 'minimum requirements' test
# signals (stereo 1 kHz sines unless noted, per-channel peak levels in dBFS)
# with +/-0.1 LU tolerance for loudness and +0.2/-0.4 dB for true peak;
# EBU Tech 3342 Table 1 gives the loudness-range signals (20 s stereo 1 kHz
# tone steps) with +/-1 LU tolerance.
# ---------------------------------------------------------------------------
BS1770_ANCHOR_997_LKFS = -3.01  # LKFS

EBU_TECH3341_TOL_LU = 0.1
#: Tech 3341 Table 1 integrated-loudness cases 1-5:
#: (case, ((per-channel peak level dBFS, duration s), ...), expected I LUFS).
EBU_TECH3341_INTEGRATED_CASES: list[
    tuple[int, tuple[tuple[float, float], ...], float]
] = [
    (1, ((-23.0, 20.0),), -23.0),
    (2, ((-33.0, 20.0),), -33.0),
    (3, ((-36.0, 10.0), (-23.0, 60.0), (-36.0, 10.0)), -23.0),
    (
        4,
        ((-72.0, 10.0), (-36.0, 10.0), (-23.0, 60.0), (-36.0, 10.0), (-72.0, 10.0)),
        -23.0,
    ),
    (5, ((-26.0, 20.0), (-20.0, 20.1), (-26.0, 20.0)), -23.0),
]
#: Tech 3341 case 6: 5.0-channel sine, per-channel peak levels (L, R, C,
#: Ls, Rs), 20 s, expected I = -23.0 LUFS.
EBU_TECH3341_CASE6_LEVELS = (-28.0, -28.0, -24.0, -30.0, -30.0)
EBU_TECH3341_CASE6_EXPECTED = -23.0

#: Tech 3341 Table 1 true-peak cases 15-19:
#: (case, frequency as a fraction of fs, amplitude FFS, phase deg,
#: expected max true-peak level dBTP). Tolerance +0.2/-0.4 dB.
EBU_TECH3341_TRUE_PEAK_CASES: list[tuple[int, float, float, float, float]] = [
    (15, 1.0 / 4.0, 0.50, 0.0, -6.0),
    (16, 1.0 / 4.0, 0.50, 45.0, -6.0),
    (17, 1.0 / 6.0, 0.50, 60.0, -6.0),
    (18, 1.0 / 8.0, 0.50, 67.5, -6.0),
    (19, 1.0 / 4.0, 1.41, 45.0, 3.0),
]
EBU_TECH3341_TP_TOL_UP = 0.2  # dB
EBU_TECH3341_TP_TOL_DOWN = 0.4  # dB
#: Cases 20-23 (a single fs/4 period inside an fs/6 tone, synthesized at
#: 4 fs and downsampled with a 0-3 sample offset) all expect 0.0 dBTP.
EBU_TECH3341_TP_OFFSET_EXPECTED = 0.0

EBU_TECH3342_TOL_LU = 1.0
#: Tech 3342 Table 1 LRA cases 1-4: (case, per-channel peak level dBFS of
#: each 20 s tone segment, expected LRA LU).
EBU_TECH3342_LRA_CASES: list[tuple[int, tuple[float, ...], float]] = [
    (1, (-20.0, -30.0), 10.0),
    (2, (-20.0, -15.0), 5.0),
    (3, (-40.0, -20.0), 20.0),
    (4, (-50.0, -35.0, -20.0, -35.0, -50.0), 15.0),
]

# Porous materials & multilayer absorbers - published anchors.
#
# Delany-Bazley power law (Bies 5e Appendix D Table D.1 first row = Mechel 2e
# Sect. G.11 Eqs. (1)-(2) = Hopkins Eqs. (1.171)-(1.172)), evaluated by hand
# at the digitization point X = rho f / sigma = 0.1 (mid fit range):
#   Zc/(rho c) = 1 + 0.0571*0.1^-0.754 - j 0.087*0.1^-0.732
#   k/k0       = 1 + 0.0978*0.1^-0.700 - j 0.189*0.1^-0.595
# Miki (1990) Eqs. (30)-(34) evaluated by hand at f/sigma = 0.1:
#   Zc/(rho c) = 1 + 0.070*0.1^-0.632 - j 0.107*0.1^-0.632
#   k/k0       = 1 + 0.109*0.1^-0.618 - j 0.160*0.1^-0.618
# Mechel 2e Sect. D.5: the maximum possible statistical absorption
# coefficient of a locally reacting plane is the published 0.951.
# ---------------------------------------------------------------------------
POROUS_DB_X_POINT = 0.1
POROUS_DB_ZC_EXPECTED = 1.3240679696882804 - 0.46937424158816093j
POROUS_DB_K_EXPECTED = 1.4901611144874722 - 0.7438096426114194j
POROUS_MIKI_Y_POINT = 0.1
POROUS_MIKI_ZC_EXPECTED = 1.2999839642782076 - 0.4585469168252603j
POROUS_MIKI_K_EXPECTED = 1.4522999064714557 - 0.6639264682149806j
POROUS_STATISTICAL_ALPHA_MAX = 0.951

# ---------------------------------------------------------------------------
# Maa (1998), "Potential of microperforated panel absorber", JASA 104(5).
# Table I (printed): maximum absorption alpha0 = 4r/(1+r)^2 (Eq. (10)) and
# absorption-band frequency interval B = f2/f1 = pi/arccot(1+r) - 1
# (Eq. (21)) for r = 1..5 at k = 0.
# Fig. 5 design example (also Cox & D'Antonio 3e Fig. 7.28): d = t = 0.2 mm,
# hole separation b = 2.5 mm on a square lattice (sigma = (pi/4)(d/b)^2,
# Eq. (25)), cavity D = 6 cm; theory vs standing-wave-tube measurement.
# ---------------------------------------------------------------------------
MAA_TABLE_I_R = (1.0, 2.0, 3.0, 4.0, 5.0)
MAA_TABLE_I_ALPHA0 = (1.0, 0.89, 0.75, 0.64, 0.56)
MAA_TABLE_I_BANDWIDTH = (5.78, 8.76, 11.82, 14.91, 18.02)
MAA_FIG5_DIAMETER = 0.2e-3
MAA_FIG5_THICKNESS = 0.2e-3
MAA_FIG5_SEPARATION = 2.5e-3
MAA_FIG5_CAVITY = 0.06

# ---------------------------------------------------------------------------
# Hopkins, Sound Insulation (2007), Table A2 (printed p. 608 / pdf p. 635):
# material properties of 25 building-material rows. Each row pairs the
# quasi-longitudinal thin-plate phase speed cL (m/s) with the printed product
# h.fc (m.Hz), stated for c0 = 343 m/s. The product follows from
# h fc = c0^2 sqrt(12) / (2 pi cL) and is independent of density and Poisson
# ratio, so every row is an independent check on the coincidence frequency.
# ---------------------------------------------------------------------------
HOPKINS_TABLE_A2_H_FC: tuple[tuple[float, float], ...] = (
    (1900.0, 34.1),   # aircrete / AAC blocks (solid)
    (5100.0, 12.7),   # aluminium
    (2700.0, 24.0),   # bricks (solid)
    (2500.0, 25.9),   # calcium-silicate blocks (solid)
    (2200.0, 29.5),   # chipboard
    (1850.0, 35.1),   # clinker concrete blocks, 1030 kg/m3
    (2200.0, 29.5),   # clinker concrete blocks, 1720 kg/m3
    (1910.0, 34.0),   # clinker concrete slabs
    (3800.0, 17.1),   # concrete, cast in situ
    (3200.0, 20.3),   # dense aggregate blocks (solid)
    (2300.0, 28.2),   # expanded clay blocks (solid)
    (5200.0, 12.5),   # glass
    (2200.0, 29.5),   # lightweight aggregate blocks (solid)
    (2560.0, 25.3),   # medium density fibreboard
    (2450.0, 26.5),   # mortar
    (2570.0, 25.2),   # oriented strand board
    (2350.0, 27.6),   # perspex, plexiglass
    (1610.0, 40.3),   # plaster, gypsum based
    (1490.0, 43.5),   # plasterboard, natural gypsum
    (1810.0, 35.8),   # plasterboard, flue gas plus natural gypsum
    (2010.0, 32.3),   # plasterboard, gypsum with glass fibre
    (3850.0, 16.8),   # plywood (birch)
    (3250.0, 20.0),   # sand-cement screed
    (5270.0, 12.3),   # steel
    (5000.0, 13.0),   # timber (soft wood)
)

# ---------------------------------------------------------------------------
# Speech transmission index - IEC 60268-16 Annex M worked example (the "full
# STI" calculation of a public-address system in a reverberant space). The
# annex prints the adjusted MTF matrix without noise, masking and threshold
# (step 2; rows are the 14 modulation frequencies 0,63 Hz to 12,5 Hz, columns
# the 7 octave bands 125 Hz to 8 kHz), the operational speech and ambient
# noise spectra it is combined with, and the resulting per-band modulation
# transfer indices (step 4c) and STI. Reproducing the printed MTI row and STI
# from the printed MTF exercises the whole clause A.5.3 to A.5.6 chain
# (auditory masking, reception threshold, the SNR clamp, the MTI average and
# the alpha/beta weighting).
# ---------------------------------------------------------------------------
IEC60268_16_ANNEX_M_MTF = (
    (0.983, 0.960, 0.978, 0.990, 0.990, 0.986, 0.997),
    (0.968, 0.936, 0.959, 0.974, 0.980, 0.979, 0.995),
    (0.947, 0.904, 0.931, 0.953, 0.966, 0.968, 0.992),
    (0.920, 0.869, 0.898, 0.927, 0.949, 0.955, 0.987),
    (0.886, 0.826, 0.852, 0.892, 0.925, 0.935, 0.981),
    (0.851, 0.791, 0.808, 0.856, 0.900, 0.914, 0.974),
    (0.816, 0.756, 0.764, 0.816, 0.871, 0.891, 0.964),
    (0.773, 0.721, 0.730, 0.776, 0.841, 0.866, 0.953),
    (0.741, 0.684, 0.705, 0.745, 0.809, 0.838, 0.941),
    (0.726, 0.628, 0.678, 0.736, 0.780, 0.812, 0.929),
    (0.714, 0.557, 0.656, 0.723, 0.753, 0.786, 0.916),
    (0.670, 0.520, 0.623, 0.678, 0.728, 0.765, 0.904),
    (0.591, 0.483, 0.556, 0.615, 0.701, 0.749, 0.893),
    (0.554, 0.446, 0.523, 0.614, 0.685, 0.737, 0.884),
)
IEC60268_16_ANNEX_M_LEVEL = (82.9, 82.9, 79.2, 73.2, 67.2, 61.2, 55.2)
IEC60268_16_ANNEX_M_AMBIENT = (55.5, 47.5, 41.5, 37.5, 34.5, 32.5, 30.5)
IEC60268_16_ANNEX_M_MTI = (0.73, 0.66, 0.67, 0.71, 0.77, 0.80, 0.92)
IEC60268_16_ANNEX_M_STI = 0.76

# ---------------------------------------------------------------------------
# ISO 12354-1:2017 Annex L and ISO 12354-2:2017 Annex G - detailed model.
#
# Both annexes drive the SAME building (two dwellings one above the other,
# 55 m3 rooms, a 220 mm concrete separating floor with a floating floor, two
# 365 mm AAC external walls and two 200 mm calcium-silicate internal walls), so
# one fixture feeds the airborne and the impact chain. Every table below is
# transcribed from the printed annexes; the defects found in them are recorded
# in docs/ERRATA.md.
# ---------------------------------------------------------------------------
ISO12354_ANNEX_L_BANDS = (
    50.0, 63.0, 80.0, 100.0, 125.0, 160.0, 200.0, 250.0, 315.0, 400.0, 500.0,
    630.0, 800.0, 1000.0, 1250.0, 1600.0, 2000.0, 2500.0, 3150.0, 4000.0,
    5000.0,
)

# Shared input data (ISO 12354-1 printed p. 78 / ISO 12354-2 printed p. 35):
# label -> (area, length1, length2, mass, critical frequency, internal loss
# factor, density, longitudinal velocity, junction coupling length).
ISO12354_ANNEX_L_ELEMENTS: dict[str, tuple[float, ...]] = {
    "floor": (20.00, 5.00, 4.00, 484.0, 76.8, 0.0050, 2200.0, 3800.0, 0.0),
    "ext1": (11.00, 4.00, 2.75, 219.0, 92.6, 0.0125, 600.0, 1900.0, 4.0),
    "ext2": (13.75, 5.00, 2.75, 219.0, 92.6, 0.0125, 600.0, 1900.0, 5.0),
    "int1": (11.00, 4.00, 2.75, 360.0, 128.4, 0.0100, 1800.0, 2500.0, 4.0),
    "int2": (13.75, 5.00, 2.75, 360.0, 128.4, 0.0100, 1800.0, 2500.0, 5.0),
}
# The rows above carry the *element specifications*: 0,005 for the concrete
# floor and 0,012 5 for the AAC external walls (Annex B Table B.3). The
# Table L.3 / G.3 input block prints 0,013 for the external walls, which does
# not reproduce their own column; see docs/ERRATA.md.
ISO12354_ANNEX_L_FLOOR_ETA_INT = 0.005
ISO12354_ANNEX_L3_PRINTED_EXT_ETA_INT = 0.013

# Floating floor on the separating element: 35 mm screed on mineral wool.
ISO12354_ANNEX_L_FLOATING_MASS = 73.5
ISO12354_ANNEX_L_FLOATING_STIFFNESS = 8.0
ISO12354_ANNEX_L_FLOATING_F0 = 52.8

# Table L.2 / G.2 - radiation factor for free bending waves.
ISO12354_ANNEX_L2_SIGMA: dict[str, tuple[float, ...]] = {
    "floor": (0.7209, 0.8092, 0.9119, 1.0196, 1.1399, 1.2896, 1.2742, 1.2015,
              1.1500, 1.1125, 1.0870, 1.0672, 1.0518, 1.0408, 1.0322, 1.0249,
              1.0198, 1.0157, 1.0124, 1.0097, 1.0078),
    "ext1": (0.6243, 0.7008, 0.7897, 0.8830, 0.9872, 1.1169, 1.2487, 1.2603,
             1.1901, 1.1407, 1.1078, 1.0827, 1.0634, 1.0498, 1.0392, 1.0303,
             1.0240, 1.0191, 1.0150, 1.0118, 1.0094),
    "ext2": (0.6690, 0.7510, 0.8462, 0.9461, 1.0578, 1.1967, 1.3380, 1.2603,
             1.1901, 1.1407, 1.1078, 1.0827, 1.0634, 1.0498, 1.0392, 1.0303,
             1.0240, 1.0191, 1.0150, 1.0118, 1.0094),
    "int1": (0.3929, 0.5190, 0.8473, 1.8783, 2.0000, 2.0000, 1.6718, 1.4341,
             1.2994, 1.2137, 1.1600, 1.1208, 1.0915, 1.0712, 1.0557, 1.0427,
             1.0337, 1.0267, 1.0210, 1.0165, 1.0131),
    "int2": (0.3581, 0.4765, 0.7783, 1.7252, 2.0000, 2.0000, 1.6718, 1.4341,
             1.2994, 1.2137, 1.1600, 1.1208, 1.0915, 1.0712, 1.0557, 1.0427,
             1.0337, 1.0267, 1.0210, 1.0165, 1.0131),
}
# Table L.2 / G.2 - radiation factor for forced waves (depends only on the
# element dimensions, so the 4,00 x 2,75 and 5,00 x 2,75 walls pair up).
ISO12354_ANNEX_L2_SIGMA_F: dict[str, tuple[float, ...]] = {
    "floor": (0.7912, 0.9059, 1.0248, 1.1361, 1.2474, 1.3707, 1.4822, 1.5937,
              1.7092, 1.8287, 1.9402, 2.0000, 2.0000, 2.0000, 2.0000, 2.0000,
              2.0000, 2.0000, 2.0000, 2.0000, 2.0000),
    "ext1": (0.6380, 0.7520, 0.8704, 0.9814, 1.0926, 1.2157, 1.3271, 1.4386,
             1.5541, 1.6735, 1.7851, 1.9006, 2.0000, 2.0000, 2.0000, 2.0000,
             2.0000, 2.0000, 2.0000, 2.0000, 2.0000),
    "ext2": (0.6805, 0.7948, 0.9134, 1.0245, 1.1358, 1.2590, 1.3705, 1.4820,
             1.5975, 1.7169, 1.8284, 1.9440, 2.0000, 2.0000, 2.0000, 2.0000,
             2.0000, 2.0000, 2.0000, 2.0000, 2.0000),
}

# Table L.3 / G.3 - in-situ total loss factor.
ISO12354_ANNEX_L3_ETA: dict[str, tuple[float, ...]] = {
    "floor": (0.0831, 0.0746, 0.0667, 0.0602, 0.0544, 0.0486, 0.0438, 0.0394,
              0.0355, 0.0319, 0.0290, 0.0263, 0.0239, 0.0218, 0.0200, 0.0183,
              0.0168, 0.0156, 0.0144, 0.0133, 0.0124),
    "ext1": (0.1298, 0.1170, 0.1052, 0.0954, 0.0867, 0.0781, 0.0711, 0.0646,
             0.0585, 0.0530, 0.0485, 0.0444, 0.0407, 0.0376, 0.0349, 0.0322,
             0.0301, 0.0282, 0.0265, 0.0249, 0.0236),
    "ext2": (0.1149, 0.1037, 0.0934, 0.0849, 0.0772, 0.0697, 0.0637, 0.0577,
             0.0523, 0.0475, 0.0436, 0.0400, 0.0368, 0.0342, 0.0318, 0.0295,
             0.0277, 0.0260, 0.0245, 0.0232, 0.0220),
    "int1": (0.0770, 0.0702, 0.0647, 0.0625, 0.0566, 0.0506, 0.0452, 0.0408,
             0.0371, 0.0338, 0.0311, 0.0287, 0.0265, 0.0247, 0.0231, 0.0216,
             0.0203, 0.0192, 0.0182, 0.0172, 0.0165),
    "int2": (0.0703, 0.0642, 0.0592, 0.0574, 0.0526, 0.0470, 0.0420, 0.0379,
             0.0345, 0.0315, 0.0291, 0.0269, 0.0249, 0.0233, 0.0218, 0.0204,
             0.0193, 0.0183, 0.0174, 0.0165, 0.0158),
}
# Table L.3 / G.3 - in-situ sound reduction index.
ISO12354_ANNEX_L3_R_SITU: dict[str, tuple[float, ...]] = {
    "floor": (31.8, 31.5, 35.9, 37.6, 39.1, 40.8, 43.3, 46.3, 49.2, 52.2, 54.9,
              57.6, 60.4, 63.0, 65.6, 68.5, 71.1, 73.3, 73.0, 72.6, 72.3),
    "ext1": (27.4, 28.2, 26.2, 32.8, 34.7, 36.4, 37.9, 40.3, 43.4, 46.4, 49.2,
             52.0, 54.9, 57.6, 59.5, 59.2, 58.9, 58.6, 58.3, 58.0, 57.8),
    "ext2": (26.4, 27.3, 25.7, 31.7, 33.6, 35.3, 36.8, 39.8, 42.9, 46.0, 48.8,
             51.6, 54.5, 57.2, 59.1, 58.8, 58.5, 58.2, 58.0, 57.7, 57.5),
    "int1": (32.3, 32.5, 31.2, 27.2, 29.7, 32.3, 36.3, 40.1, 43.5, 46.8, 49.8,
             52.7, 55.7, 58.5, 61.3, 64.3, 67.0, 68.8, 68.6, 68.4, 68.2),
    "int2": (32.5, 32.7, 31.4, 27.5, 29.4, 32.0, 36.0, 39.8, 43.2, 46.5, 49.5,
             52.5, 55.5, 58.3, 61.0, 64.0, 66.8, 68.6, 68.4, 68.2, 68.0),
}
# Table G.3 - in-situ normalized impact level of the bare separating floor.
ISO12354_ANNEX_G3_LN_SITU = (
    57.3, 58.2, 59.2, 60.2, 61.1, 62.1, 62.5, 62.7, 63.0, 63.3, 63.6, 64.0,
    64.3, 64.7, 65.0, 65.4, 65.7, 66.0, 66.3, 66.7, 67.0,
)

# Table L.4 / G.4 - in-situ equivalent absorption lengths (the D1 block gives
# the floor and external wall 1; the second block gives internal wall 2).
ISO12354_ANNEX_L4_ABSORPTION: dict[str, tuple[float, ...]] = {
    "floor": (10.8, 10.9, 11.0, 11.1, 11.2, 11.3, 11.4, 11.4, 11.6, 11.7, 11.9,
              12.1, 12.4, 12.7, 13.0, 13.4, 13.8, 14.3, 14.8, 15.5, 16.2),
    "ext1": (9.3, 9.4, 9.5, 9.6, 9.8, 10.0, 10.2, 10.3, 10.5, 10.7, 10.9, 11.3,
             11.6, 12.0, 12.5, 13.0, 13.6, 14.2, 15.0, 15.9, 16.8),
    "int2": (6.3, 6.4, 6.7, 7.2, 7.4, 7.5, 7.5, 7.6, 7.7, 8.0, 8.2, 8.5, 8.9,
             9.3, 9.7, 10.3, 10.9, 11.5, 12.3, 13.2, 14.1),
}
# Table L.4 / G.4 - in-situ velocity level differences of the two printed paths.
ISO12354_ANNEX_L4_DV: dict[str, tuple[float, ...]] = {
    "D1": (10.4, 10.4, 10.4, 10.5, 10.5, 10.6, 10.7, 10.7, 10.8, 10.8, 10.9,
           11.0, 11.1, 11.3, 11.4, 11.6, 11.7, 11.9, 12.1, 12.3, 12.5),
    "4d": (11.0, 11.0, 11.1, 11.3, 11.4, 11.4, 11.5, 11.5, 11.6, 11.7, 11.8,
           11.9, 12.0, 12.2, 12.3, 12.5, 12.7, 12.9, 13.1, 13.4, 13.6),
}
# Table L.4 / G.4 - improvement of the floating floor, 30 lg(f/f0).
ISO12354_ANNEX_L4_DELTA = (
    0.0, 2.3, 5.4, 8.3, 11.2, 14.4, 17.4, 20.3, 23.3, 26.4, 29.3, 32.3, 35.4,
    38.3, 41.2, 44.4, 47.4, 50.3, 53.3, 56.4, 59.3,
)

# Table L.1 - direct and flanking sound reduction indices, per path.
ISO12354_ANNEX_L1_PATHS: dict[str, tuple[float, ...]] = {
    "Dd": (31.8, 33.8, 41.4, 45.9, 50.3, 55.2, 60.7, 66.5, 72.5, 78.5, 84.1,
           89.9, 95.8, 101.3, 106.9, 112.9, 118.5, 123.6, 126.2, 129.0, 131.6),
    "1d": (41.2, 41.5, 42.8, 47.0, 48.7, 50.5, 52.6, 55.3, 58.4, 61.4, 64.3,
           67.2, 70.1, 72.9, 75.3, 76.7, 78.0, 79.1, 79.0, 78.9, 78.9),
    "2d": (39.5, 39.9, 41.3, 45.2, 47.0, 48.7, 50.8, 53.9, 56.9, 60.0, 62.8,
           65.7, 68.7, 71.5, 73.9, 75.3, 76.7, 77.8, 77.7, 77.7, 77.6),
    "3d": (45.0, 45.0, 46.6, 45.7, 47.8, 49.9, 53.2, 56.6, 59.9, 63.1, 66.0,
           69.0, 72.0, 74.8, 77.7, 80.8, 83.6, 85.8, 85.8, 85.7, 85.7),
    "4d": (43.9, 44.0, 45.6, 44.7, 46.5, 48.6, 51.9, 55.3, 58.6, 61.8, 64.7,
           67.7, 70.8, 73.6, 76.4, 79.6, 82.4, 84.7, 84.6, 84.6, 84.6),
    "D1": (41.2, 43.8, 48.2, 55.3, 60.0, 64.9, 69.9, 75.6, 81.6, 87.8, 93.6,
           99.5, 105.5, 111.2, 116.5, 121.1, 125.4, 129.4, 132.3, 135.3, 138.2),
    "11": (44.8, 45.7, 43.8, 50.5, 52.4, 54.2, 55.8, 58.3, 61.4, 64.5, 67.4,
           70.4, 73.4, 76.2, 78.3, 78.1, 78.0, 77.9, 77.9, 77.9, 77.9),
    "D2": (39.5, 42.2, 46.8, 53.6, 58.2, 63.2, 68.2, 74.1, 80.2, 86.4, 92.1,
           98.0, 104.1, 109.8, 115.1, 119.8, 124.0, 128.1, 131.0, 134.0, 136.9),
    "22": (42.4, 43.4, 41.8, 47.9, 49.8, 51.6, 53.3, 56.3, 59.5, 62.6, 65.5,
           68.5, 71.6, 74.4, 76.5, 76.4, 76.3, 76.3, 76.2, 76.3, 76.3),
    "D3": (45.0, 47.3, 52.1, 54.0, 59.0, 64.4, 70.6, 76.9, 83.2, 89.5, 95.3,
           101.3, 107.4, 113.2, 118.9, 125.2, 131.0, 136.1, 139.0, 142.1, 145.0),
    "33": (47.2, 47.5, 46.4, 42.8, 45.3, 48.0, 52.0, 55.8, 59.3, 62.8, 65.8,
           68.9, 72.1, 75.1, 78.0, 81.2, 84.2, 86.3, 86.3, 86.4, 86.4),
    "D4": (43.9, 46.3, 51.0, 53.0, 57.7, 63.1, 69.3, 75.6, 81.9, 88.2, 94.0,
           100.0, 106.2, 111.9, 117.7, 124.0, 129.8, 134.9, 137.9, 141.0, 143.9),
    "44": (46.1, 46.4, 45.3, 41.7, 43.7, 46.4, 50.4, 54.2, 57.7, 61.2, 64.3,
           67.4, 70.6, 73.6, 76.5, 79.8, 82.8, 84.9, 84.9, 85.0, 85.1),
}
ISO12354_ANNEX_L1_R_PRIME = (
    28.8, 30.4, 33.4, 35.3, 37.6, 40.0, 43.1, 46.4, 49.7, 52.9, 55.9, 58.9,
    61.9, 64.8, 67.3, 69.0, 70.2, 71.0, 70.9, 70.9, 70.9,
)
# ISO 717-1 rating of that spectrum. Table L.1 prints a non-integer 57,8 /
# 57,9 (see docs/ERRATA.md); the ISO 717-1 rating in 1 dB steps is 57 dB.
ISO12354_ANNEX_L1_R_PRIME_W = 57

# Table G.4 - the printed direct and Df (external wall 1) impact levels.
ISO12354_ANNEX_G4_LN_DD = (
    57.3, 55.9, 53.8, 51.8, 49.9, 47.7, 45.2, 42.5, 39.7, 36.9, 34.3, 31.7,
    28.9, 26.3, 23.8, 20.9, 18.4, 15.8, 13.1, 10.3, 7.7,
)
ISO12354_ANNEX_G4_LN_DF = (
    47.8, 45.9, 47.0, 42.4, 40.2, 38.0, 35.9, 33.4, 30.6, 27.6, 24.9, 22.1,
    19.2, 16.5, 14.1, 12.7, 11.4, 9.9, 7.0, 4.0, 1.1,
)
# Table G.1 - direct and flanking impact levels per element, and the total.
ISO12354_ANNEX_G1_PATHS: dict[str, tuple[float, ...]] = {
    "Dd": ISO12354_ANNEX_G4_LN_DD,
    "Df1": (47.3, 44.9, 46.2, 42.4, 40.2, 38.0, 35.9, 33.4, 30.6, 27.6, 24.9,
            22.1, 19.2, 16.5, 14.1, 12.7, 11.4, 9.9, 7.0, 4.0, 1.1),
    "Df2": (49.0, 46.6, 47.9, 44.2, 42.0, 39.7, 37.7, 34.9, 32.0, 29.1, 26.3,
            23.5, 20.6, 17.9, 15.5, 14.1, 12.8, 11.2, 8.3, 5.3, 2.4),
    "Df3": (43.9, 41.9, 43.2, 43.8, 41.2, 38.5, 35.3, 32.1, 29.1, 26.0, 23.2,
            20.3, 17.3, 14.5, 11.7, 8.6, 5.8, 3.2, 0.3, -2.8, -5.7),
    "Df4": (45.0, 43.0, 44.3, 44.9, 42.5, 39.8, 36.6, 33.4, 30.3, 27.3, 24.4,
            21.5, 18.5, 15.8, 13.0, 9.8, 7.0, 4.4, 1.4, -1.7, -4.6),
}
ISO12354_ANNEX_G1_L_PRIME_N = (
    58.6, 57.0, 55.9, 54.0, 51.9, 49.6, 47.1, 44.3, 41.4, 38.6, 35.9, 33.3,
    30.4, 27.8, 25.3, 22.7, 20.4, 18.2, 15.4, 12.5, 9.8,
)
ISO12354_ANNEX_G1_L_PRIME_N_W = 41
ISO12354_ANNEX_G1_CI = 2

# Tables L.5 to L.9 / G.5 to G.9 - the junction vibration reduction indices, as
# printed (see docs/ERRATA.md for the Table G.8 value that disagrees with L.8).
ISO12354_ANNEX_L_KIJ: dict[str, float] = {
    "floor-ext": 6.4,
    "ext-ext": 11.2,
    "floor-floor": 6.6,
    "int-int": 11.0,
    "floor-int": 8.8,
    "ext1-ext2": -2.0,
    "int-ext": 6.0,
    "extT-ext": 9.0,
    "int1-int2": 8.7,
}
# Table L.3 / G.3 input block: the printed perimeter sums. Only the external
# wall 1 and internal wall 2 values reproduce their tabulated columns; see
# docs/ERRATA.md and the Formula (C.4) derivation used by the fixture.
ISO12354_ANNEX_L3_PRINTED_PERIMETER = {"floor": 2.364, "ext1": 2.375, "int": 1.840}

# Table L.10 / G.10 - the simplified model applied to the same building.
ISO12354_ANNEX_L10_RW = {"floor": 58.7, "ext": 45.8, "int": 53.9}
ISO12354_ANNEX_L10_DELTA_RW = 10.6
ISO12354_ANNEX_L10_PATH_RW: dict[str, float] = {
    "Dd": 69.3, "D1": 76.3, "D2": 75.3, "D3": 82.7, "D4": 81.7,
    "1d": 65.7, "11": 64.0, "2d": 64.7, "22": 63.0,
    "3d": 72.1, "33": 71.9, "4d": 71.1, "44": 70.9,
}
ISO12354_ANNEX_L10_R_PRIME_W = 57.0
ISO12354_ANNEX_G10_LN_EQ_0_W = 70.0
ISO12354_ANNEX_G10_DELTA_LW = 32.2
ISO12354_ANNEX_G10_PATH_LN_W: dict[str, float] = {
    "Dd": 37.8, "Df1": 30.9, "Df2": 31.9, "Df3": 24.4, "Df4": 25.4,
}
ISO12354_ANNEX_G10_L_PRIME_N_W = 39.7

# ---------------------------------------------------------------------------
# ISO 12354-1:2017 L.2.1 / ISO 12354-2:2017 G.2 - wood frame lightweight
# building. A room above another, only the flanking transmission through the
# junction between the floor and a double frame separating wall is considered;
# the floor is 20 m2 with a 4 m junction length. Type B elements, so the paths
# use Formula (17) / Part 2 Formula (14) with Dv,ij,n.
# ---------------------------------------------------------------------------
ISO12354_LIGHTWEIGHT_AREA = 20.0
ISO12354_LIGHTWEIGHT_COUPLING = 4.0
# Table L.12 - the direct floor index, the separating wall inner leaf index,
# its resonant-only correction R* and the normalized velocity level difference.
ISO12354_TABLE_L12_RD_FLOOR = (
    24.6, 23.8, 33.2, 36.5, 42.2, 48.2, 53.8, 55.3, 58.8, 63.2, 67.0, 69.8,
    72.1, 73.8, 76.2, 77.3, 76.4, 77.6, 80.1, 84.4, 86.6,
)
ISO12354_TABLE_L12_R_WALL = (
    15.0, 18.7, 20.1, 21.3, 20.5, 22.1, 24.1, 24.9, 25.3, 27.6, 28.9, 31.5,
    32.2, 33.5, 34.8, 35.2, 34.8, 32.5, 34.3, 38.1, 40.3,
)
ISO12354_TABLE_L12_R_STAR_WALL = (
    23.0, 26.7, 28.1, 29.3, 28.5, 30.1, 32.1, 32.9, 33.3, 35.6, 36.9, 39.5,
    40.2, 41.5, 42.8, 43.2, 42.8, 32.5, 34.3, 38.1, 40.3,
)
ISO12354_TABLE_L12_DV_FF = (
    18.7, 19.0, 19.4, 19.7, 20.0, 20.4, 20.7, 21.0, 21.3, 21.7, 22.0, 22.3,
    22.7, 23.0, 23.3, 23.7, 24.0, 24.3, 24.6, 25.0, 25.3,
)
# Table L.13 - the bare floor index, the floating floor improvement, the
# resonant-only bare floor index and the Df/Fd normalized level difference.
ISO12354_TABLE_L13_R_BARE = (
    10.0, 12.7, 14.0, 15.0, 13.0, 14.0, 17.0, 20.0, 23.0, 25.0, 22.0, 26.0,
    28.0, 29.0, 30.0, 28.5, 27.0, 29.0, 36.0, 38.0, 41.0,
)
ISO12354_TABLE_L13_DELTA_R = (
    0.0, 0.0, 2.5, 5.0, 6.5, 10.0, 11.0, 11.5, 11.5, 10.5, 11.0, 11.5, 12.0,
    12.0, 13.5, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0,
)
ISO12354_TABLE_L13_R_STAR_BARE = (
    18.0, 20.7, 22.0, 23.0, 21.0, 22.0, 25.0, 28.0, 31.0, 33.0, 30.0, 34.0,
    36.0, 37.0, 38.0, 36.5, 27.0, 29.0, 36.0, 38.0, 41.0,
)
ISO12354_TABLE_L13_DV_DF = (
    14.7, 15.0, 15.4, 15.7, 16.0, 16.4, 16.7, 17.0, 17.3, 17.7, 18.0, 18.3,
    18.7, 19.0, 19.3, 19.7, 20.0, 20.3, 20.6, 21.0, 21.3,
)
# Table L.11 - the resulting Ff and Df paths and the total R'.
ISO12354_TABLE_L11_R_FF = (
    48.7, 52.7, 54.5, 56.0, 55.5, 57.5, 59.8, 60.9, 61.6, 64.3, 65.9, 68.8,
    69.9, 71.5, 73.1, 73.9, 73.8, 63.8, 65.9, 70.1, 72.6,
)
ISO12354_TABLE_L11_R_DF = (
    42.2, 45.7, 49.9, 53.8, 54.3, 59.4, 63.2, 65.9, 68.0, 69.5, 69.4, 73.6,
    75.8, 77.2, 80.2, 81.5, 76.9, 73.0, 77.8, 81.0, 83.9,
)
ISO12354_TABLE_L11_R_PRIME = (
    24.5, 23.8, 33.1, 36.4, 41.8, 47.4, 52.4, 54.0, 56.6, 60.2, 62.4, 65.5,
    67.2, 68.8, 70.8, 71.7, 70.7, 63.1, 65.5, 69.6, 72.1,
)
ISO12354_TABLE_L11_RATINGS = {"Rd": 65, "RFf": 69, "RDf": 74, "R_prime": 63}

# ISO 12354-1 L.2.2, Tables L.14 to L.16 - a laboratory-measured junction
# between two timber frame gypsum board walls (Reference [36]). Formula (16)
# turns the measured Dn,f,13 into R13; Formula (17) predicts the same path from
# the wall index, its resonant correction and Dv,13,n.
ISO12354_TABLE_L14_SEPARATING_AREA = 10.44
ISO12354_TABLE_L14_COUPLING = 2.41
ISO12354_TABLE_L14_LAB_COUPLING = 2.5
ISO12354_TABLE_L15_DNF = (
    51.8, 53.8, 55.8, 57.8, 59.8, 61.4, 63.0, 65.2, 65.4, 68.3, 68.7, 72.3,
    76.7, 76.9, 78.8, 78.2, 71.9, 72.0, 77.1, 81.3, 84.8,
)
ISO12354_TABLE_L15_R13 = (
    52.1, 54.1, 56.1, 58.1, 60.1, 61.7, 63.3, 65.5, 65.7, 68.6, 69.0, 72.6,
    77.0, 77.2, 79.1, 78.5, 72.2, 72.3, 77.4, 81.6, 85.1,
)
ISO12354_TABLE_L16_R_SITU = (
    27.6, 22.3, 27.4, 28.9, 33.6, 36.9, 37.6, 42.0, 47.3, 53.1, 56.7, 59.3,
    61.7, 63.3, 64.4, 62.7, 54.7, 50.8, 55.3, 59.1, 61.8,
)
ISO12354_TABLE_L16_DV = (
    18.3, 18.0, 17.6, 17.3, 17.0, 16.6, 16.3, 16.0, 15.7, 15.3, 15.0, 14.7,
    14.3, 14.0, 13.7, 13.3, 13.0, 12.7, 12.4, 12.0, 11.7,
)
ISO12354_TABLE_L16_R13_PRED = (
    52.3, 46.6, 51.4, 52.5, 56.9, 59.9, 60.3, 64.3, 69.3, 74.8, 78.0, 80.4,
    82.4, 83.7, 84.5, 82.4, 74.1, 69.9, 74.0, 77.5, 79.8,
)

# ISO 12354-2 G.2, Tables G.11 to G.13 - the impact side of the same
# lightweight building.
ISO12354_TABLE_G12_LN_BARE = (
    78.2, 73.5, 80.0, 82.0, 87.0, 89.0, 93.0, 93.0, 91.0, 92.0, 97.0, 94.0,
    93.0, 93.0, 90.0, 87.0, 83.0, 79.0, 74.0, 69.0, 64.0,
)
ISO12354_TABLE_G12_DELTA_LI = (
    0.0, 1.0, 3.0, 5.0, 7.0, 8.0, 10.0, 13.0, 16.0, 18.0, 19.0, 19.0, 20.0,
    20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0,
)
ISO12354_TABLE_G12_DELTA_LDI = (
    15.5, 6.3, 13.7, 15.6, 19.3, 19.7, 22.7, 22.3, 19.6, 19.7, 25.2, 23.5,
    22.3, 23.9, 21.7, 21.5, 20.8, 17.3, 15.1, 18.1, 21.1,
)
ISO12354_TABLE_G13_R_BARE = ISO12354_TABLE_L13_R_BARE
ISO12354_TABLE_G13_R_WALL = ISO12354_TABLE_L12_R_WALL
ISO12354_TABLE_G13_DV = ISO12354_TABLE_L13_DV_DF
ISO12354_TABLE_G11_LN_DD = (
    62.7, 66.2, 63.3, 61.4, 60.7, 61.3, 60.3, 57.7, 55.4, 54.3, 52.8, 51.5,
    50.7, 49.1, 48.3, 45.5, 42.2, 41.7, 38.9, 30.9, 22.9,
)
ISO12354_TABLE_G11_LN_DF = (
    54.0, 47.5, 51.6, 51.2, 53.2, 53.6, 55.8, 53.6, 49.5, 48.0, 49.6, 46.9,
    45.2, 44.8, 41.3, 37.0, 32.1, 30.0, 27.2, 21.0, 16.1,
)
ISO12354_TABLE_G11_LN_TOTAL = (
    63.3, 66.3, 63.6, 61.8, 61.4, 62.0, 61.6, 59.1, 56.4, 55.2, 54.5, 52.8,
    51.8, 50.5, 49.1, 46.1, 42.6, 42.0, 39.2, 31.3, 23.7,
)
ISO12354_TABLE_G11_RATINGS = {"LnDd": 54, "LnDf": 47, "total": 55}

# ISO 12354-2:2017 Table B.2 - calculated octave-band normalized impact level
# of monolithic floors in a laboratory situation (Annex C.3 loss factor), with
# the Table B.1 material properties. The standard states the values were
# computed at one-third-octave spacing and averaged over an octave and does not
# print the radiation factor or the structural reverberation time used, so this
# is a plausibility oracle rather than a bit-exact one.
ISO12354_2_TABLE_B1 = {
    "concrete": (2300.0, 3500.0, 0.006),
    "lightweight": (1300.0, 1700.0, 0.015),
}
ISO12354_2_TABLE_B2_BANDS = (63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0)
ISO12354_2_TABLE_B2: tuple[tuple[str, float, tuple[float, ...], int], ...] = (
    ("concrete", 268.0, (65.0, 73.0, 78.0, 78.0, 78.0, 78.0, 76.0), 80),
    ("concrete", 509.0, (64.0, 60.0, 65.0, 66.0, 67.0, 68.0, 66.0), 69),
    ("lightweight", 260.0, (65.0, 72.0, 78.0, 77.0, 77.0, 76.0, 70.0), 77),
    ("lightweight", 390.0, (64.0, 68.0, 70.0, 70.0, 70.0, 70.0, 64.0), 71),
)
