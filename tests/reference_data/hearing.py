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
