#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Outdoor sound: propagation, source emission and the assessment of tones.

The path from a source to an assessed rating level. ISO 9613-1 tabulates
the atmospheric attenuation and ISO 9613-2 the propagation terms; the
CNOSSOS-EU road and rail source models of Annex II to Directive 2002/49/EC
supply the emission, including the committed extracts of the Commission's
own test workbooks; ISO 1996-2, ISO 20065, DIN 45681 and NT ACOU 112 turn
the received spectrum into the adjustment for audible tones and impulses.

The oracle sets that are too large to inline are read from the committed
CSVs under ``tests/data/cnossos/`` by the small accessors at the end of
the module, so the tables and the code that reads them stay together.
"""

from __future__ import annotations

import csv
import math
import pathlib

#: Versioned root of the committed oracle data.
_DATA = pathlib.Path(__file__).parent.parent / "data"

#: Committed oracle sets that are too large to inline (see tests/data/README.md).
_DATA = pathlib.Path(__file__).parent.parent / "data"

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
# is that same sum less exactly 1.763 dB = 10 lg 1.5, the standard's own
# Hanning correction, and does NOT reproduce the printed dL -- the two printed
# cells contradict each other, so only the dL chain is pinned. The same 1.76 dB
# offset carries the "5 FG" row of Tabelle I.10 (55.12 + 54.23 -> 57.71 dB
# summed, 55.95 dB printed), where the printed dL = 3.22 dB does follow the
# printed LT, so that row is self-consistent and this one is not. See
# docs/ERRATA.md.
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
# CNOSSOS-EU railway source, Appendix G of Annex II to Directive 2002/49/EC.
# Transcribed from the Official Journal text of the instrument named in each
# comment: Commission Delegated Directive (EU) 2021/1226 (OJ L 269, 28.7.2021)
# where it replaced the table, Commission Directive (EU) 2015/996 (OJ L 168,
# 1.7.2015) where it did not. These constants pin the tables shipped in
# ``phonometry.environment.sources.cnossos_rail``.
# ---------------------------------------------------------------------------

#: Wavelength grid of Tables G-1b, G-2 and G-4 as replaced by (EU) 2021/1226, mm.
CNOSSOS_RAIL_WAVELENGTHS: tuple[float, ...] = (
    2000, 1600, 1250, 1000, 800, 630, 500, 400, 315, 250, 200, 160, 125, 100,
    80, 63, 50, 40, 31.5, 25, 20, 16, 12.5, 10, 8, 6.3, 5, 4, 3.15, 2.5, 2,
    1.6, 1.25, 1, 0.8,
)
#: Wavelength grid of Table G-1a, which (EU) 2021/1226 left untouched, mm. It
#: keeps the non-standard steps 120, 12, 3,2 and 1,2 mm.
CNOSSOS_RAIL_WHEEL_WAVELENGTHS: tuple[float, ...] = (
    1000, 800, 630, 500, 400, 315, 250, 200, 160, 120, 100, 80, 63, 50, 40,
    31.5, 25, 20, 16, 12, 10, 8, 6.3, 5, 4, 3.2, 2.5, 2, 1.6, 1.2, 1, 0.8,
)

#: Table G-1a wheel roughness (2015/996, unchanged by (EU) 2021/1226).
CNOSSOS_RAIL_G1A_CAST_IRON: tuple[float, ...] = (
    2.2, 2.2, 2.2, 2.2, 2.2, 2.2, 2.2, 2.2, 2.4, 0.6, 2.6, 5.8, 8.8, 11.1,
    11, 9.8, 7.5, 5.1, 3, 1.3, 0.2, -0.7, -1.2, -1, 0.3, 0.2, 1.3, 3.1,
    3.1, 3.1, 3.1, 3.1,
)

CNOSSOS_RAIL_G1A_COMPOSITE: tuple[float, ...] = (
    -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4.3, -4.6, -4.9, -5.2,
    -6.3, -6.8, -7.2, -7.3, -7.3, -7.1, -6.9, -6.7, -6, -3.7, -2.4, -2.6,
    -2.5, -2.5, -2.5, -2.5, -2.5,
)

CNOSSOS_RAIL_G1A_NON_TREAD: tuple[float, ...] = (
    -5.9, -5.9, -5.9, -5.9, -5.9, -5.9, 2.3, 2.8, 2.6, 1.2, 2.1, 0.9, -0.3,
    -1.6, -2.9, -4.9, -7, -8.6, -9.3, -9.5, -10.1, -10.3, -10.3, -10.8,
    -10.9, -9.5, -9.5, -9.5, -9.5, -9.5, -9.5, -9.5,
)

#: Table G-1b rail roughness, as replaced by (EU) 2021/1226 point (20)(a).
CNOSSOS_RAIL_G1B_E: tuple[float, ...] = (
    17.1, 17.1, 17.1, 17.1, 17.1, 17.1, 17.1, 17.1, 15, 13, 11, 9, 7, 4.9,
    2.9, 0.9, -1.1, -3.2, -5, -5.6, -6.2, -6.8, -7.4, -8, -8.6, -9.2, -9.8,
    -10.4, -11, -11.6, -12.2, -12.8, -13.4, -14, -14,
)

CNOSSOS_RAIL_G1B_M: tuple[float, ...] = (
    35, 31, 28, 25, 23, 20, 17, 13.5, 10.5, 9, 6.5, 5.5, 5, 3.5, 2, 0.1,
    -0.2, -0.3, -0.8, -3, -5, -7, -8, -9, -10, -12, -13, -14, -15, -16,
    -17, -18, -19, -19, -19,
)

#: Table G-2 contact filter, as replaced by (EU) 2021/1226 point (20)(b).
#: Keys are (wheel load in kN, wheel diameter in mm).
CNOSSOS_RAIL_G2: dict[tuple[float, float], tuple[float, ...]] = {
    (50.0, 360.0): (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.1, -0.2, -0.3, -0.6, -1, -1.8, -3.2, -5.4, -8.7, -12.2, -16.7, -17.7, -17.8, -20.7, -22.1, -22.8, -24, -24.5, -24.7, -27, -27.8),
    (50.0, 680.0): (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.1, -0.2, -0.3, -0.7, -1.2, -2, -4.1, -6, -9.2, -13.8, -17.2, -17.7, -18.6, -21.5, -22.3, -23.1, -24.4, -24.5, -25, -28, -28.8, -29.6),
    (50.0, 920.0): (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.1, -0.1, -0.3, -0.6, -1.1, -1.3, -3.5, -5.3, -8, -12, -16.8, -17.7, -18, -21.5, -21.8, -22.8, -24, -24.5, -25, -27.3, -28.1, -28.9, -29.7),
    (25.0, 920.0): (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.1, -0.3, -0.5, -1.1, -1.8, -3.3, -5.3, -7.9, -12.8, -16.8, -17.7, -18.2, -20.5, -22, -22.8, -24.2, -24.5, -25, -27.4, -28.2, -29),
    (100.0, 920.0): (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.1, -0.2, -0.3, -0.6, -1, -1.8, -3.2, -5.4, -8.7, -12.2, -16.7, -17.7, -17.8, -20.7, -22.1, -22.8, -24, -24.5, -24.7, -27, -27.8, -28.6, -29.4, -30.2),
}

#: Table G-3a track transfer, as replaced by (EU) 2021/1226 point (20)(c).
CNOSSOS_RAIL_G3A: dict[str, tuple[float, ...]] = {
    "M/S": (53.3, 59.3, 67.2, 75.9, 79.2, 81.8, 84.2, 88.6, 91, 94.5, 97, 99.2, 104, 107.1, 108.3, 108.5, 109.7, 110, 110, 110, 110.3, 110, 110.1, 110.6),
    "M/M": (50.9, 57.8, 66.5, 76.8, 80.9, 83.3, 85.8, 90, 91.6, 93.9, 95.6, 97.4, 101.7, 104.4, 106, 106.8, 108.3, 108.9, 109.1, 109.4, 109.9, 109.9, 110.3, 111),
    "M/H": (50.1, 57.2, 66.3, 77.2, 81.6, 84, 86.5, 90.7, 92.1, 94.3, 95.8, 97, 100.3, 102.5, 104.2, 105.4, 107.1, 107.9, 108.2, 108.7, 109.4, 109.7, 110.4, 111.4),
    "B/S": (50.9, 56.6, 64.3, 72.3, 75.4, 78.5, 81.8, 86.6, 89.1, 91.9, 94.5, 97.5, 104, 107.9, 108.9, 108.8, 109.8, 110.2, 110.1, 110.1, 110.3, 109.9, 110, 110.4),
    "B/M": (50, 56.1, 64.1, 72.5, 75.8, 79.1, 83.6, 88.7, 89.6, 89.7, 90.6, 93.8, 100.6, 104.7, 106.3, 107.1, 108.8, 109.3, 109.4, 109.7, 110, 109.8, 110, 110.5),
    "B/H": (49.8, 55.9, 64, 72.5, 75.9, 79.4, 84.4, 89.7, 90.2, 90.2, 90.8, 93.1, 97.9, 101.1, 103.4, 105.4, 107.7, 108.5, 108.7, 109.1, 109.6, 109.6, 109.9, 110.6),
    "W": (44, 51, 59.9, 70.8, 75.1, 76.9, 77.2, 80.9, 85.3, 92.5, 97, 98.7, 102.8, 105.4, 106.5, 106.4, 107.5, 108.1, 108.4, 108.7, 109.1, 109.1, 109.5, 110.2),
    "D": (75.4, 77.4, 81.4, 87.1, 88, 89.7, 83.4, 87.7, 89.8, 97.5, 99, 100.8, 104.9, 111.8, 113.9, 115.5, 114.9, 118.2, 118.3, 118.4, 118.9, 117.5, 117.9, 118.6),
}

#: Table G-3b wheel transfer (2015/996 values; band labels corrected
#: by (EU) 2021/1226 point (20)(d)). Keys are the wheel diameter in mm.
CNOSSOS_RAIL_G3B: dict[float, tuple[float, ...]] = {
    920.0: (75.4, 77.3, 81.1, 84.1, 83.3, 84.3, 86, 90.1, 89.8, 89, 88.8, 90.4, 92.4, 94.9, 100.4, 104.6, 109.6, 114.9, 115, 115, 115.5, 115.6, 116, 116.7),
    840.0: (75.4, 77.3, 81.1, 84.1, 82.8, 83.3, 84.1, 86.9, 87.9, 89.9, 90.9, 91.5, 91.5, 93, 98.7, 101.6, 107.6, 111.9, 114.5, 114.5, 115, 115.1, 115.5, 116.2),
    680.0: (75.4, 77.3, 81.1, 84.1, 82.8, 83.3, 83.9, 86.3, 88, 92.2, 93.9, 92.5, 90.9, 90.4, 93.2, 93.5, 99.6, 104.9, 108, 111, 111.5, 111.6, 112, 112.7),
    1200.0: (75.4, 77.3, 81.1, 84.1, 82.8, 83.3, 84.5, 90.4, 90.4, 89.9, 90.1, 91.3, 91.5, 93.6, 100.5, 104.6, 115.6, 115.9, 116, 116, 116.5, 116.6, 117, 117.7),
}

CNOSSOS_RAIL_G3C: tuple[float, ...] = (
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
)

#: Table G-4 impact roughness, as replaced by (EU) 2021/1226 point (20)(e).
CNOSSOS_RAIL_G4: tuple[float, ...] = (
    22, 22, 22, 22, 22, 20, 16, 15, 14, 15, 14, 12, 11, 10, 9, 8, 6, 3, 2,
    -3, -8, -13, -17, -19, -22, -25, -26, -32, -35, -40, -43, -45, -47,
    -49, -50,
)

#: Table G-5 traction sound power (2015/996 values). (EU) 2021/1226
#: point (20)(f) replaced the 6 300 Hz pair of the 2 200 kW diesel
#: locomotive, 31,4 / 30,7, by 81,4 / 80,7; the replacement is applied
#: below at the two marked positions.
CNOSSOS_RAIL_G5: dict[str, tuple[tuple[float, ...], tuple[float, ...]]] = {
    "diesel locomotive, c. 800 kW": (
        (98.9, 94.8, 92.6, 94.6, 92.8, 92.8, 93, 94.8, 94.6, 95.7, 95.6, 98.6, 95.2, 95.1, 95.1, 94.1, 94.1, 99.4, 92.5, 89.5, 87, 84.1, 81.5, 79.2),
        (103.2, 100, 95.5, 94, 93.3, 93.6, 92.9, 92.7, 92.4, 92.8, 92.8, 96.8, 92.7, 93, 92.9, 93.1, 93.2, 98.3, 91.5, 88.7, 86, 83.4, 80.9, 78.7),
    ),
    "diesel locomotive, c. 2 200 kW": (
        (99.4, 107.3, 103.1, 102.1, 99.3, 99.3, 99.5, 101.3, 101.1, 102.2, 102.1, 101.1, 101.7, 101.6, 99.3, 96, 93.7, 101.9, 89.5, 87.1, 90.5, 81.4, 81.2, 79.6),
        (103.7, 112.5, 106, 101.5, 99.8, 100.1, 99.4, 99.2, 98.9, 99.3, 99.3, 99.3, 99.2, 99.5, 97.1, 95, 92.8, 100.8, 88.5, 86.3, 89.5, 80.7, 80.6, 79.1),
    ),
    "diesel multiple unit": (
        (82.6, 82.5, 89.3, 90.3, 93.5, 99.5, 98.7, 95.5, 90.3, 91.4, 91.3, 90.3, 90.9, 91.8, 92.8, 92.8, 90.8, 88.1, 85.2, 83.2, 81.7, 78.8, 76.2, 73.9),
        (86.9, 87.7, 92.2, 89.7, 94, 100.3, 98.6, 93.4, 88.1, 88.5, 88.5, 88.5, 88.4, 89.7, 90.6, 91.8, 89.9, 87, 84.2, 82.4, 80.7, 78.1, 75.6, 73.4),
    ),
    "electric locomotive": (
        (87.9, 90.8, 91.6, 94.6, 94.8, 96.8, 104, 100.8, 99.6, 101.7, 98.6, 95.6, 95.2, 96.1, 92.1, 89.1, 87.1, 85.4, 83.5, 81.5, 80, 78.1, 76.5, 75.2),
        (92.2, 96, 94.5, 94, 95.3, 97.6, 103.9, 98.7, 97.4, 98.8, 95.8, 93.8, 92.7, 94, 89.9, 88.1, 86.2, 84.3, 82.5, 80.7, 79, 77.4, 75.9, 74.7),
    ),
    "electric multiple unit": (
        (80.5, 81.4, 80.5, 82.2, 80, 79.7, 79.6, 96.4, 80.5, 81.3, 97.2, 79.5, 79.8, 86.7, 81.7, 82.7, 80.7, 78, 75.1, 72.1, 69.6, 66.7, 64.1, 61.8),
        (84.8, 86.6, 83.4, 81.6, 80.5, 80.5, 79.5, 94.3, 78.3, 78.4, 94.4, 77.7, 77.3, 84.6, 79.5, 81.7, 79.8, 76.9, 74.1, 71.3, 68.6, 66, 63.5, 61.3),
    ),
}

#: Table G-6 aerodynamic sound power at v_0 = 300 km/h (2015/996).
CNOSSOS_RAIL_G6_A: tuple[float, ...] = (
    112.6, 113.2, 115.7, 117.4, 115.3, 115, 114.9, 116.4, 115.9, 116.3,
    116.2, 115.2, 115.8, 115.7, 115.7, 114.7, 114.7, 115, 114.5, 113.1,
    112.1, 110.6, 109.6, 108.8,
)

CNOSSOS_RAIL_G6_B: tuple[float, ...] = (
    36.7, 38.5, 39, 37.5, 36.8, 37.1, 36.4, 36.2, 35.9, 36.3, 36.3, 36.3,
    36.2, 36.5, 36.4, 105.2, 110.3, 110.4, 105.6, 37.2, 37.5, 37.9, 38.4,
    39.2,
)

#: Speed exponents alpha_1 = alpha_2 of Table G-6, every band.
CNOSSOS_RAIL_G6_ALPHA = 50.0

#: Table G-7 bridge transfer, as replaced by (EU) 2021/1226 point (20)(h).
CNOSSOS_RAIL_G7: dict[str, tuple[float, ...]] = {
    "+10 dB(A)": (85.2, 87.1, 91, 94, 94.4, 96, 92.5, 96.7, 97.4, 99.4, 100.7, 102.5, 107.1, 109.8, 112, 107.2, 106.8, 107.3, 99.3, 91.4, 86.9, 79.7, 75.1, 70.8),
    "+15 dB(A)": (90.1, 92.1, 96, 99.5, 99.9, 101.5, 99.6, 103.8, 104.5, 106.5, 107.8, 109.6, 116.1, 118.8, 120.9, 109.5, 109.1, 109.6, 102, 94.1, 89.6, 83.6, 79, 74.7),
}


# ---------------------------------------------------------------------------
# CNOSSOS-EU railway emission oracle: the 2015 coefficient database the
# Commission's reference source module was run with, and the extract of its
# emission test workbook. See ``tests/data/cnossos/README.md`` for provenance.
# ---------------------------------------------------------------------------

#: 1/3-octave band centres of the railway source, as CSV column names.
CNOSSOS_RAIL_BANDS: tuple[str, ...] = (
    "50", "63", "80", "100", "125", "160", "200", "250", "315", "400", "500",
    "630", "800", "1000", "1250", "1600", "2000", "2500", "3150", "4000",
    "5000", "6300", "8000", "10000",
)
#: Wavelength grid of the 2015 catalogue tables, in mm, as CSV column names.
CNOSSOS_RAIL_2015_WAVELENGTHS: tuple[str, ...] = (
    "1000", "800", "630", "500", "400", "315", "250", "200", "160", "125",
    "100", "80", "63", "50", "40", "31.5", "25", "20", "16", "12.5", "10",
    "8", "6.3", "5", "4", "3.15", "2.5", "2", "1.6", "1.25", "1", "0.8",
)


def _cnossos_rail_rows(name: str) -> list[dict[str, str]]:
    """Rows of one committed CSV under ``tests/data/cnossos/``."""
    with (_DATA / "cnossos" / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def cnossos_rail_workbook_cases() -> list[dict[str, str]]:
    """The committed cases of the CIRCABC railway emission test workbook.

    Each row carries the vehicle, the track parameters, the running condition,
    the two receiver angles and the workbook's own per-octave-band and total
    line-power levels.
    """
    return _cnossos_rail_rows("rail_emission_cases.csv")


def cnossos_rail_2015_wavelength_tables() -> dict[tuple[str, str], tuple[float, ...]]:
    """The 2015 catalogue spectra given against wavelength, keyed by
    ``(table, id)``: wheel and rail roughness, contact filter, impact."""
    return {
        (row["table"], row["id"]): tuple(
            float(row[w]) for w in CNOSSOS_RAIL_2015_WAVELENGTHS
        )
        for row in _cnossos_rail_rows("rail_wavelength_tables_2015.csv")
    }


def cnossos_rail_2015_frequency_tables() -> (
    dict[tuple[str, str, str], tuple[float, ...]]
):
    """The 2015 catalogue spectra given against frequency, keyed by
    ``(table, id, source)``: transfer functions, traction, aerodynamic."""
    return {
        (row["table"], row["id"], row["source"]): tuple(
            float(row[b]) for b in CNOSSOS_RAIL_BANDS
        )
        for row in _cnossos_rail_rows("rail_frequency_tables_2015.csv")
    }


def cnossos_rail_2015_vehicles() -> dict[str, dict[str, str]]:
    """The 2015 catalogue vehicle definitions, keyed by their reference id."""
    return {row["id"]: row for row in _cnossos_rail_rows("rail_vehicles_2015.csv")}
# CNOSSOS-EU road traffic source (Directive 2002/49/EC Annex II, Appendix F).
# Machine-transcribed from the Official Journal text of the amending acts:
#   * Tables F-1 and F-4 from Commission Delegated Directive (EU) 2021/1226,
#     Annex points (19)(a) and (19)(b) (OJ L 269, 28.7.2021, pp. 96-99), which
#     replaced the versions published in (EU) 2015/996;
#   * Tables F-2 and F-3 from Commission Directive (EU) 2015/996 Appendix F
#     (OJ L 168, 1.7.2015, p. 125), never amended;
#   * the octave-band A-weighting of 2.5.5 as amended by (EU) 2021/1226 Annex
#     point (8)(b) (OJ L 269, 28.7.2021, p. 68).
# Table F-4 keeps the amended layout, in which categories 4a and 4b share a
# single "4a/4b" row.
# ---------------------------------------------------------------------------
CNOSSOS_ROAD_TABLE_F1: dict[str, dict[str, tuple[float, ...]]] = {
    "1": {
        "AR": (83.1, 89.2, 87.7, 93.1, 100.1, 96.7, 86.8, 76.2),
        "BR": (30.0, 41.5, 38.9, 25.7, 32.5, 37.2, 39.0, 40.0),
        "AP": (97.9, 92.5, 90.7, 87.2, 84.7, 88.0, 84.4, 77.1),
        "BP": (-1.3, 7.2, 7.7, 8.0, 8.0, 8.0, 8.0, 8.0),
    },
    "2": {
        "AR": (88.7, 93.2, 95.7, 100.9, 101.7, 95.1, 87.8, 83.6),
        "BR": (30.0, 35.8, 32.6, 23.8, 30.1, 36.2, 38.3, 40.1),
        "AP": (105.5, 100.2, 100.5, 98.7, 101.0, 97.8, 91.2, 85.0),
        "BP": (-1.9, 4.7, 6.4, 6.5, 6.5, 6.5, 6.5, 6.5),
    },
    "3": {
        "AR": (91.7, 96.2, 98.2, 104.9, 105.1, 98.5, 91.1, 85.6),
        "BR": (30.0, 33.5, 31.3, 25.4, 31.8, 37.1, 38.6, 40.6),
        "AP": (108.8, 104.2, 103.5, 102.9, 102.6, 98.5, 93.8, 87.5),
        "BP": (0.0, 3.0, 4.6, 5.0, 5.0, 5.0, 5.0, 5.0),
    },
    "4a": {
        "AR": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        "BR": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        "AP": (93.0, 93.0, 93.5, 95.3, 97.2, 100.4, 95.8, 90.9),
        "BP": (4.2, 7.4, 9.8, 11.6, 15.7, 18.9, 20.3, 20.6),
    },
    "4b": {
        "AR": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        "BR": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        "AP": (99.9, 101.9, 96.7, 94.4, 95.2, 94.7, 92.1, 88.6),
        "BP": (3.2, 5.9, 11.9, 11.6, 11.5, 12.6, 11.1, 12.0),
    },
}

CNOSSOS_ROAD_TABLE_F2: dict[str, tuple[float, ...]] = {
    "ai": (0.0, 0.0, 0.0, 2.6, 2.9, 1.5, 2.3, 9.2),
    "bi": (0.0, 0.0, 0.0, -3.1, -6.4, -14.0, -22.4, -11.4),
}

CNOSSOS_ROAD_TABLE_F3: dict[str, dict[int, tuple[float, float]]] = {
    "1": {
        1: (-4.5, 5.5),
        2: (-4.4, 3.1),
    },
    "2": {
        1: (-4.0, 9.0),
        2: (-2.3, 6.7),
    },
    "3": {
        1: (-4.0, 9.0),
        2: (-2.3, 6.7),
    },
    "4a": {
        1: (0.0, 0.0),
        2: (0.0, 0.0),
    },
    "4b": {
        1: (0.0, 0.0),
        2: (0.0, 0.0),
    },
}

CNOSSOS_ROAD_TABLE_F4: dict[str, dict[str, tuple[tuple[float, ...], float]]] = {
    "reference road surface": {
        "1": ((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), 0.0),
        "2": ((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), 0.0),
        "3": ((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), 0.0),
        "4a/4b": ((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), 0.0),
    },
    "1-layer ZOAB": {
        "1": ((0.0, 5.4, 4.3, 4.2, -1.0, -3.2, -2.6, 0.8), -6.5),
        "2": ((7.9, 4.3, 5.3, -0.4, -5.2, -4.6, -3.0, -1.4), 0.2),
        "3": ((9.3, 5.0, 5.5, -0.4, -5.2, -4.6, -3.0, -1.4), 0.2),
        "4a/4b": ((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), 0.0),
    },
    "2-layer ZOAB": {
        "1": ((1.6, 4.0, 0.3, -3.0, -4.0, -6.2, -4.8, -2.0), -3.0),
        "2": ((7.3, 2.0, -0.3, -5.2, -6.1, -6.0, -4.4, -3.5), 4.7),
        "3": ((8.3, 2.2, -0.4, -5.2, -6.2, -6.1, -4.5, -3.5), 4.7),
        "4a/4b": ((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), 0.0),
    },
    "2-layer ZOAB (fine)": {
        "1": ((-1.0, 3.0, -1.5, -5.3, -6.3, -8.5, -5.3, -2.4), -0.1),
        "2": ((7.9, 0.1, -1.9, -5.9, -6.1, -6.8, -4.9, -3.8), -0.8),
        "3": ((9.4, 0.2, -1.9, -5.9, -6.1, -6.7, -4.8, -3.8), -0.9),
        "4a/4b": ((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), 0.0),
    },
    "SMA-NL5": {
        "1": ((10.3, -0.9, 0.9, 1.8, -1.8, -2.7, -2.0, -1.3), -1.6),
        "2": ((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), 0.0),
        "3": ((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), 0.0),
        "4a/4b": ((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), 0.0),
    },
    "SMA-NL8": {
        "1": ((6.0, 0.3, 0.3, 0.0, -0.6, -1.2, -0.7, -0.7), -1.4),
        "2": ((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), 0.0),
        "3": ((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), 0.0),
        "4a/4b": ((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), 0.0),
    },
    "brushed down concrete": {
        "1": ((8.2, -0.4, 2.8, 2.7, 2.5, 0.8, -0.3, -0.1), 1.4),
        "2": ((0.3, 4.5, 2.5, -0.2, -0.1, -0.5, -0.9, -0.8), 5.0),
        "3": ((0.2, 5.3, 2.5, -0.2, -0.1, -0.6, -1.0, -0.9), 5.5),
        "4a/4b": ((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), 0.0),
    },
    "optimised brushed down concrete": {
        "1": ((-0.2, -0.7, 1.4, 1.2, 1.1, -1.6, -2.0, -1.8), 1.0),
        "2": ((-0.7, 3.0, -2.0, -1.4, -1.8, -2.7, -2.0, -1.9), -6.6),
        "3": ((-0.5, 4.2, -1.9, -1.3, -1.7, -2.5, -1.8, -1.8), -6.6),
        "4a/4b": ((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), 0.0),
    },
    "fine broomed concrete": {
        "1": ((8.0, -0.7, 4.8, 2.2, 1.2, 2.6, 1.5, -0.6), 7.6),
        "2": ((0.2, 8.6, 7.1, 3.2, 3.6, 3.1, 0.7, 0.1), 3.2),
        "3": ((0.1, 9.8, 7.4, 3.2, 3.1, 2.4, 0.4, 0.0), 2.0),
        "4a/4b": ((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), 0.0),
    },
    "worked surface": {
        "1": ((8.3, 2.3, 5.1, 4.8, 4.1, 0.1, -1.0, -0.8), -0.3),
        "2": ((0.1, 6.3, 5.8, 1.8, -0.6, -2.0, -1.8, -1.6), 1.7),
        "3": ((0.0, 7.4, 6.2, 1.8, -0.7, -2.1, -1.9, -1.7), 1.4),
        "4a/4b": ((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), 0.0),
    },
    "hard elements in herringbone": {
        "1": ((27.0, 16.2, 14.7, 6.1, 3.0, -1.0, 1.2, 4.5), 2.5),
        "2": ((29.5, 20.0, 17.6, 8.0, 6.2, -1.0, 3.1, 5.2), 2.5),
        "3": ((29.4, 21.2, 18.2, 8.4, 5.6, -1.0, 3.0, 5.8), 2.5),
        "4a/4b": ((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), 0.0),
    },
    "hard elements not in herringbone": {
        "1": ((31.4, 19.7, 16.8, 8.4, 7.2, 3.3, 7.8, 9.1), 2.9),
        "2": ((34.0, 23.6, 19.8, 10.5, 11.7, 8.2, 12.2, 10.0), 2.9),
        "3": ((33.8, 24.7, 20.4, 10.9, 10.9, 6.8, 12.0, 10.8), 2.9),
        "4a/4b": ((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), 0.0),
    },
    "quiet hard elements": {
        "1": ((26.8, 13.7, 11.9, 3.9, -1.8, -5.8, -2.7, 0.2), -1.7),
        "2": ((9.2, 5.7, 4.8, 2.3, 4.4, 5.1, 5.4, 0.9), 0.0),
        "3": ((9.1, 6.6, 5.2, 2.6, 3.9, 3.9, 5.2, 1.1), 0.0),
        "4a/4b": ((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), 0.0),
    },
    "thin layer A": {
        "1": ((10.4, 0.7, -0.6, -1.2, -3.0, -4.8, -3.4, -1.4), -2.9),
        "2": ((13.8, 5.4, 3.9, -0.4, -1.8, -2.1, -0.7, -0.2), 0.5),
        "3": ((14.1, 6.1, 4.1, -0.4, -1.8, -2.1, -0.7, -0.2), 0.3),
        "4a/4b": ((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), 0.0),
    },
    "thin layer B": {
        "1": ((6.8, -1.2, -1.2, -0.3, -4.9, -7.0, -4.8, -3.2), -1.8),
        "2": ((13.8, 5.4, 3.9, -0.4, -1.8, -2.1, -0.7, -0.2), 0.5),
        "3": ((14.1, 6.1, 4.1, -0.4, -1.8, -2.1, -0.7, -0.2), 0.3),
        "4a/4b": ((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), 0.0),
    },
}

CNOSSOS_A_WEIGHTING_TABLE: tuple[float, ...] = (-26.2, -16.1, -8.6, -3.2, 0.0, 1.2, 1.0, -1.1)

# Validity speed ranges printed in the two leading columns of Table F-4, in
# km/h; the reference road surface carries none. Same instrument as
# CNOSSOS_ROAD_TABLE_F4.
CNOSSOS_ROAD_TABLE_F4_SPEED_RANGE: dict[str, tuple[float, float] | None] = {
    "reference road surface": None,
    "1-layer ZOAB": (50.0, 130.0),
    "2-layer ZOAB": (50.0, 130.0),
    "2-layer ZOAB (fine)": (80.0, 130.0),
    "SMA-NL5": (40.0, 80.0),
    "SMA-NL8": (40.0, 80.0),
    "brushed down concrete": (70.0, 120.0),
    "optimised brushed down concrete": (70.0, 80.0),
    "fine broomed concrete": (70.0, 120.0),
    "worked surface": (50.0, 130.0),
    "hard elements in herringbone": (30.0, 60.0),
    "hard elements not in herringbone": (30.0, 60.0),
    "quiet hard elements": (30.0, 60.0),
    "thin layer A": (40.0, 130.0),
    "thin layer B": (40.0, 130.0),
}

# Air-temperature coefficients K_m of formula (2.2.10), in dB per degree
# Celsius: "a generic coefficient Km=1 = 0,08 dB/degC for light vehicles
# (category 1) and Km=2 = Km=3 = 0,04 dB/degC for heavy vehicles (categories 2
# and 3)" (Directive (EU) 2015/996, Annex II 2.2.3, unamended). The powered
# two-wheelers of category 4 have no rolling noise, so the Directive prints no
# K_m for them and the correction is identically zero.
CNOSSOS_ROAD_TEMPERATURE_K: dict[str, float] = {
    "1": 0.08, "2": 0.04, "3": 0.04, "4a": 0.0, "4b": 0.0,
}

#: The octave bands of the road source, as column names of the committed CSVs.
CNOSSOS_ROAD_BANDS: tuple[str, ...] = (
    "63", "125", "250", "500", "1000", "2000", "4000", "8000",
)


def _cnossos_road_rows(name: str) -> list[dict[str, str]]:
    """Rows of one committed CSV under ``tests/data/cnossos/``."""
    with (_DATA / "cnossos" / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def cnossos_road_workbook_cases() -> list[dict[str, str]]:
    """The 60 committed cases of the CIRCABC road emission test workbook.

    Each row carries the segment description (surface, temperature, studded
    season, gradient, junction and the flow and speed of the five vehicle
    categories) and the workbook's own per-band and total line-power levels.
    See ``tests/data/cnossos/README.md`` for the provenance.
    """
    return _cnossos_road_rows("road_emission_cases.csv")


def cnossos_road_2015_coefficients() -> dict[str, dict[str, tuple[float, ...]]]:
    """Table F-1 as published in (EU) 2015/996, keyed by category then ``AR``,
    ``BR``, ``AP``, ``BP``. This is the superseded database the workbook was
    computed with, not the one the library ships."""
    table: dict[str, dict[str, tuple[float, ...]]] = {}
    for row in _cnossos_road_rows("road_coefficients_2015.csv"):
        table.setdefault(row["category"], {})[row["coefficient"]] = tuple(
            float(row[band]) for band in CNOSSOS_ROAD_BANDS
        )
    return table


def cnossos_road_2015_surfaces() -> dict[
    str, tuple[str, dict[str, tuple[float, ...]], dict[str, float]]
]:
    """Table F-4 as published in (EU) 2015/996, keyed by the ``NLxx`` surface
    identifier the workbook uses: ``(description, alpha, beta)``."""
    names: dict[str, str] = {}
    alpha: dict[str, dict[str, tuple[float, ...]]] = {}
    beta: dict[str, dict[str, float]] = {}
    for row in _cnossos_road_rows("road_surfaces_2015.csv"):
        surface = row["surface"]
        names[surface] = row["description"]
        alpha.setdefault(surface, {})[row["category"]] = tuple(
            float(row[band]) for band in CNOSSOS_ROAD_BANDS
        )
        beta.setdefault(surface, {})[row["category"]] = float(row["beta"])
    return {s: (names[s], alpha[s], beta[s]) for s in names}
