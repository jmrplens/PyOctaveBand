#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Hearing: thresholds, noise-induced loss and occupational exposure.

The listener rather than the sound. ISO 389-7 fixes the reference
threshold of hearing, ISO 7029 the statistical distribution of thresholds
with age, ISO 1999 the noise-induced permanent threshold shift those
thresholds acquire under exposure, and ISO 9612 the exposure itself -
the task-based, job-based and full-day strategies of Annexes D, E and F,
each with the LEX,8h and the uncertainty its worked example prints.
"""

from __future__ import annotations

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
# is stdlib-only). Mirrors tests/hearing/test_occupational_exposure.py.
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
# ISO 4869-2:2018 - the effective A-weighted level behind a hearing protector.
# Four informative annexes carry one worked example between them, all built on
# the same 16-subject attenuation grid of Table A.1, and every one of them
# reproduces from the printed inputs:
#   A   m_f and s_f per band, exactly. The annex's own APV row is the
#       difference of the ROUNDED m_f and s_f it displays, which differs from
#       Formula (1) applied to the data by 0,1 dB at 250 Hz, 500 Hz and
#       4 kHz; both readings are pinned below.
#   B   the octave-band method: the per-band net levels and L'p,A84 = 81,4 dB,
#       81 dB after rounding.
#   C   the HML method: all 16 x 8 PNR values, all 16 H/M/L triples, the three
#       means, the three standard deviations, H84/M84/L84 = 24/18/13 dB, and
#       the application PNR84 = 22,5 dB -> L'p,A84 = 81,5 dB -> 82 dB.
#   D   the SNR method: all 16 SNRj, SNRm, SNRs, SNR84 = 21 dB and both of its
#       applications, which land on 82 dB.
#
# Table C.1 reprints Table 2 and disagrees with it in two cells; Table 2 is
# the one that reproduces Annex C. See docs/ERRATA.md.
# ---------------------------------------------------------------------------
#: Table A.1: sound attenuation in dB, one row per test subject, over the
#: eight octave bands from 63 Hz to 8 kHz.
ISO4869_2_ATTENUATION: list[list[float]] = [
    [4.0, 8.0, 13.0, 18.0, 20.0, 30.0, 35.0, 30.0],
    [6.0, 12.0, 16.0, 21.0, 29.0, 35.0, 47.0, 35.0],
    [10.0, 16.0, 17.0, 23.0, 25.0, 32.0, 48.0, 37.0],
    [3.0, 7.0, 12.0, 18.0, 20.0, 25.0, 33.0, 30.0],
    [8.0, 10.0, 16.0, 16.0, 25.0, 27.0, 43.0, 32.0],
    [4.0, 7.0, 10.0, 15.0, 19.0, 32.0, 35.0, 31.0],
    [5.0, 5.0, 9.0, 16.0, 20.0, 25.0, 30.0, 28.0],
    [15.0, 15.0, 21.0, 26.0, 25.0, 38.0, 46.0, 38.0],
    [5.0, 6.0, 10.0, 13.0, 19.0, 22.0, 29.0, 28.0],
    [9.0, 9.0, 10.0, 19.0, 20.0, 27.0, 37.0, 31.0],
    [9.0, 16.0, 18.0, 24.0, 25.0, 35.0, 44.0, 39.0],
    [5.0, 6.0, 11.0, 12.0, 17.0, 20.0, 28.0, 28.0],
    [7.0, 10.0, 17.0, 22.0, 25.0, 35.0, 41.0, 44.0],
    [6.0, 8.0, 16.0, 18.0, 19.0, 19.0, 30.0, 33.0],
    [10.0, 12.0, 17.0, 25.0, 28.0, 33.0, 45.0, 40.0],
    [12.0, 13.0, 17.0, 27.0, 29.0, 38.0, 49.0, 41.0],
]
#: Table A.1, the printed ``m_f`` and ``s_f`` rows, in dB.
ISO4869_2_MEAN = [7.4, 10.0, 14.4, 19.6, 22.8, 29.6, 38.8, 34.1]
ISO4869_2_STANDARD_DEVIATION = [3.3, 3.6, 3.6, 4.6, 4.0, 6.2, 7.4, 5.2]
#: Table A.1, the printed ``APV_f84`` row, which subtracts the two rows above
#: as displayed rather than as computed.
ISO4869_2_APV84_PRINTED = [4.1, 6.4, 10.8, 15.0, 18.8, 23.4, 31.4, 28.9]

#: Annex B: the octave-band levels of the example noise, in dB, and the
#: frequency weighting A the annex prints beside them.
ISO4869_2_ANNEX_B_NOISE = [75.0, 84.0, 86.0, 88.0, 97.0, 99.0, 97.0, 96.0]
ISO4869_2_ANNEX_B_A_WEIGHTING = [-26.2, -16.1, -8.6, -3.2, 0.0, 1.2, 1.0, -1.1]
#: Annex B: the last row of Table B.1, ``Lp + A - APV`` per band, in dB.
ISO4869_2_ANNEX_B_NET = [44.7, 61.5, 66.6, 69.8, 78.2, 76.8, 66.6, 66.0]
ISO4869_2_ANNEX_B_EFFECTIVE = 81.4  # dB, before rounding
ISO4869_2_ANNEX_B_REPORTED = 81  # dB, after rounding
ISO4869_2_ANNEX_B_LPA = 104.0  # dB, the A-weighted level of the same noise
ISO4869_2_ANNEX_B_LPC = 103.0  # dB, its C-weighted level (Annex C.2, D.2)

#: Annex C, Table C.2: the printed ``Hj``, ``Mj`` and ``Lj`` per subject, dB.
ISO4869_2_ANNEX_C_H = [
    27.8,
    34.5,
    32.1,
    26.0,
    28.7,
    27.2,
    25.6,
    33.8,
    23.5,
    27.1,
    33.1,
    21.6,
    33.0,
    21.6,
    34.2,
    36.7,
]
ISO4869_2_ANNEX_C_M = [
    20.1,
    24.8,
    24.9,
    19.6,
    21.2,
    18.0,
    18.0,
    26.5,
    16.9,
    19.4,
    25.5,
    15.9,
    24.3,
    19.1,
    26.2,
    27.1,
]
ISO4869_2_ANNEX_C_L = [
    14.7,
    18.2,
    20.2,
    13.9,
    16.4,
    12.5,
    11.4,
    22.2,
    11.9,
    13.5,
    20.9,
    12.0,
    17.7,
    15.6,
    19.0,
    19.5,
]
#: Annex C, the sixth row of Table C.2: ``PNRj6`` per subject, in dB. The two
#: cells Table C.1 misprints are the ones this row is sensitive to.
ISO4869_2_ANNEX_C_PNR_NOISE6 = [
    18.5,
    22.7,
    23.5,
    17.9,
    19.7,
    16.3,
    15.9,
    25.4,
    15.3,
    17.6,
    24.2,
    14.7,
    22.3,
    18.2,
    23.9,
    24.7,
]
ISO4869_2_ANNEX_C_MEANS = (29.2, 21.7, 16.2)  # Hm, Mm, Lm in dB
ISO4869_2_ANNEX_C_DEVIATIONS = (4.8, 3.8, 3.5)  # Hs, Ms, Ls in dB
ISO4869_2_ANNEX_C_HML84 = (24, 18, 13)  # H84, M84, L84 in dB
ISO4869_2_ANNEX_C_PNR84 = 22.5  # dB
ISO4869_2_ANNEX_C_EFFECTIVE = 81.5  # dB, before rounding
ISO4869_2_ANNEX_C_REPORTED = 82  # dB, after rounding

#: Annex D, Table D.2: the printed ``SNRj`` per subject, in dB.
ISO4869_2_ANNEX_D_SNR = [
    23.1,
    27.7,
    28.0,
    22.3,
    24.1,
    21.2,
    20.7,
    29.8,
    19.7,
    22.4,
    28.7,
    18.8,
    27.2,
    21.4,
    28.9,
    29.9,
]
ISO4869_2_ANNEX_D_MEAN = 24.6  # SNRm, dB
ISO4869_2_ANNEX_D_DEVIATION = 3.9  # SNRs, dB
ISO4869_2_ANNEX_D_SNR84 = 21  # dB
ISO4869_2_ANNEX_D_REPORTED = 82  # dB, both applications

#: Table C.1's two misprinted cells, at 250 Hz and 500 Hz of the sixth
#: reference noise. Kept so the test that tells the two tables apart names
#: what it is refusing.
ISO4869_2_TABLE_C1_NOISE6 = [82.0, 89.4, 93.5, 95.6, 93.0, 90.1, 83.0]
