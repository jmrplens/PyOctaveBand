#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Speech intelligibility: the SII of ANSI S3.5 and the STI of IEC 60268-16.

Two ways of asking the same question - how much of the speech survives the
room and the noise. ANSI S3.5-1997 answers it band by band, so its four
procedures (critical band, equally contributing critical band, one-third
octave, octave) each need their band importance function, their worked
example from the working group and the exact index the example returns;
IEC 60268-16 answers it through the modulation transfer function, so its
Annex M example is a full 7x14 MTF matrix with the MTI and STI it yields.
"""

from __future__ import annotations

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
ANSIS3_5_BAND_IMPORTANCE_SUM = (
    1.0  # Table 3, sum of Ii (identical digits in SII.C i_avg[])
)
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
    90.0,
    5.0,
    40.0,
    40.0,
    40.0,
    40.0,
    40.0,
    40.0,
    40.0,
    40.0,
    40.0,
    40.0,
    40.0,
    40.0,
    -10.0,
    -10.0,
    -10.0,
    -10.0,
)
ANSIS3_5_WG_TO_NOISE = (
    10.0,
    -10.0,
    -10.0,
    75.0,
    -10.0,
    -10.0,
    -10.0,
    -10.0,
    -10.0,
    -10.0,
    -10.0,
    -10.0,
    -10.0,
    -10.0,
    10.0,
    10.0,
    10.0,
    10.0,
)
ANSIS3_5_WG_TO_THRESHOLD = (
    90.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
)
ANSIS3_5_WG_TO_SII = 0.445  # published in the WG DevelopmentKit readme
ANSIS3_5_WG_TO_SII_EXACT = 0.4453910059  # SII.C run on TO.TST
ANSIS3_5_WG_TO1_IMPORTANCE = (
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.1,
    0.1,
    0.1,
    0.1,
    0.1,
    0.1,
    0.1,
    0.3,
    0.0,
)
ANSIS3_5_WG_TO1_SII = 0.438  # published in the WG DevelopmentKit readme
ANSIS3_5_WG_TO1_SII_EXACT = 0.4382176540  # SII.C run on TO_1.TST

# Official WG S3-79 test cases for the other three band procedures
# (DevelopmentKit SOURCES/CB.TST, CB_1.TST, ECB.TST, ECB_1.TST, OCTAVE.TST and
# OCTAVE_1.TST). Same file layout as the one-third-octave pair above: the
# equivalent speech spectrum level, the equivalent noise spectrum level and
# the equivalent hearing threshold level over the procedure's bands, and for
# the "_1" variants a fourth line with an alternative band-importance
# function. Published results (readme, 3 decimals) and the value SII.C prints
# when compiled unmodified with gcc and run on the file:
#
#   CB.TST      0.273   0.2729353808      critical band, 21 bands
#   CB_1.TST    0.410   0.4104741231      critical band, alternative Ii
#   ECB.TST     0.278   0.2781386550      equally contributing, 17 bands
#   ECB_1.TST   0.410   0.4104741231      equally contributing, alternative Ii
#   OCTAVE.TST  0.491   0.4909625062      octave, 6 bands
#   OCTAVE_1.TST 0.323  0.3229375000      octave, alternative Ii
#
# CB_1.TST and ECB_1.TST are NOT two independent oracles. They coincide to
# every printed digit because the equally-contributing bands are critical
# bands 3 to 19 and the two alternative importance functions select the same
# physical bands. The two extra critical bands below 300 Hz that only the
# critical-band procedure has do not perturb the weighted bands: their masker
# is the input's 10 dB noise line, and its upward spread arrives at the first
# weighted band at roughly 1e-19 of the local masking energy, under double
# precision. (In the lowest shared band the same contribution is 5e-2 and does
# matter, but that band carries zero weight in both alternative functions.)
# The eight official cases therefore carry seven independent confirmations,
# not eight; the pair is still worth running, because a wrong band mapping in
# either procedure would break the coincidence.
ANSIS3_5_WG_CB_SPEECH = (
    -10.0,
    -10.0,
    -10.0,
    90.0,
    5.0,
    40.0,
    40.0,
    40.0,
    40.0,
    40.0,
    40.0,
    40.0,
    40.0,
    40.0,
    40.0,
    40.0,
    40.0,
    -10.0,
    -10.0,
    -10.0,
    -10.0,
)
ANSIS3_5_WG_CB_NOISE = (
    10.0,
    10.0,
    10.0,
    10.0,
    -10.0,
    -10.0,
    70.0,
    -10.0,
    -10.0,
    -10.0,
    -10.0,
    -10.0,
    -10.0,
    -10.0,
    -10.0,
    -10.0,
    -10.0,
    10.0,
    10.0,
    10.0,
    10.0,
)
ANSIS3_5_WG_CB_THRESHOLD = (
    0.0,
    0.0,
    0.0,
    90.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
)
ANSIS3_5_WG_CB_SII = 0.273  # published in the WG DevelopmentKit readme
ANSIS3_5_WG_CB_SII_EXACT = 0.2729353808  # SII.C run on CB.TST
ANSIS3_5_WG_CB1_IMPORTANCE = (
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.1,
    0.1,
    0.1,
    0.1,
    0.1,
    0.1,
    0.1,
    0.3,
    0.0,
    0.0,
    0.0,
)
ANSIS3_5_WG_CB1_SII = 0.410  # published in the WG DevelopmentKit readme
ANSIS3_5_WG_CB1_SII_EXACT = 0.4104741231  # SII.C run on CB_1.TST
ANSIS3_5_WG_ECB_SPEECH = (
    -10.0,
    90.0,
    5.0,
    40.0,
    40.0,
    40.0,
    40.0,
    40.0,
    40.0,
    40.0,
    40.0,
    40.0,
    40.0,
    40.0,
    40.0,
    -10.0,
    -10.0,
)
ANSIS3_5_WG_ECB_NOISE = (
    10.0,
    10.0,
    -10.0,
    -10.0,
    70.0,
    -10.0,
    -10.0,
    -10.0,
    -10.0,
    -10.0,
    -10.0,
    -10.0,
    -10.0,
    -10.0,
    -10.0,
    10.0,
    10.0,
)
ANSIS3_5_WG_ECB_THRESHOLD = (
    0.0,
    90.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
)
ANSIS3_5_WG_ECB_SII = 0.278  # published in the WG DevelopmentKit readme
ANSIS3_5_WG_ECB_SII_EXACT = 0.2781386550  # SII.C run on ECB.TST
ANSIS3_5_WG_ECB1_IMPORTANCE = (
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.1,
    0.1,
    0.1,
    0.1,
    0.1,
    0.1,
    0.1,
    0.3,
    0.0,
)
ANSIS3_5_WG_ECB1_SII = 0.410  # published in the WG DevelopmentKit readme
ANSIS3_5_WG_ECB1_SII_EXACT = 0.4104741231  # SII.C run on ECB_1.TST
# The octave case is the input printed in the DevelopmentKit readme as "the
# octave band procedure of the example in Section C.1", except that OCTAVE.TST
# raises the 8000 Hz equivalent hearing threshold to 7.1 dB HL where Annex C.1
# has 0; ANSIS3_5_ANNEX_C1_* below carries the Annex C.1 input itself.
ANSIS3_5_WG_OCTAVE_SPEECH = (50.0, 40.0, 40.0, 30.0, 20.0, 0.0)
ANSIS3_5_WG_OCTAVE_NOISE = (70.0, 65.0, 45.0, 25.0, 1.0, -15.0)
ANSIS3_5_WG_OCTAVE_THRESHOLD = (0.0, 0.0, 0.0, 0.0, 0.0, 7.1)
ANSIS3_5_WG_OCTAVE_SII = 0.491  # published in the WG DevelopmentKit readme
ANSIS3_5_WG_OCTAVE_SII_EXACT = 0.4909625062  # SII.C run on OCTAVE.TST
ANSIS3_5_WG_OCTAVE1_IMPORTANCE = (0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
ANSIS3_5_WG_OCTAVE1_SII = 0.323  # published in the WG DevelopmentKit readme
ANSIS3_5_WG_OCTAVE1_SII_EXACT = 0.3229375000  # SII.C run on OCTAVE_1.TST
# Sums of the tabulated band-importance functions. Tables 1 and 4 sum to one
# like Table 3; Table 2 prints 0.0588 in each of its 17 bands, so it sums to
# 0.9996 (1/17 = 0.058823...) rather than to exactly one.
ANSIS3_5_CRITICAL_IMPORTANCE_SUM = 1.0
ANSIS3_5_OCTAVE_IMPORTANCE_SUM = 1.0
ANSIS3_5_EQUAL_IMPORTANCE_SUM = 0.9996
# Table 4 tabulates the standard speech spectrum level Ui and the reference
# internal noise spectrum level Xi at the six octave centres as the same
# figures Table 3 gives at those centres: both are spectrum (per-hertz) levels,
# so they do not depend on the analysis bandwidth. SII.C carries the two tables
# independently and its u[]/x[] entries agree at all six. Rows: centre
# frequency in hertz, Ui and Xi in dB SPL.
ANSIS3_5_OCTAVE_TABLE4_SHARED = (
    (250.0, 34.75, -3.9),
    (500.0, 34.27, -9.7),
    (1000.0, 25.01, -12.5),
    (2000.0, 17.32, -17.7),
    (4000.0, 9.33, -25.9),
    (8000.0, 1.13, -7.1),
)
# ANSI S3.5-1997 Annex C.1 worked example (octave-band procedure). Its input is
# printed in the WG DevelopmentKit readme: equivalent speech spectrum level
# 50/40/40/30/20/0 dB, equivalent noise spectrum level 70/65/45/25/1/-15 dB,
# normal hearing. SII.C run on that input = 0.5039555062.
ANSIS3_5_ANNEX_C1_SPEECH = (50.0, 40.0, 40.0, 30.0, 20.0, 0.0)
ANSIS3_5_ANNEX_C1_NOISE = (70.0, 65.0, 45.0, 25.0, 1.0, -15.0)
ANSIS3_5_ANNEX_C1 = 0.5039555062
# Table C.1, row i = 5 (the 4000 Hz octave band), column Li under Step 6, with
# the official WG S3-79 erratum applied: the printed 0.10 should be 1.00. The
# level-distortion factor of clause 5.7 with the example's own inputs
# (Ei' = 20 dB, Ui = 9.33 dB) is 1 - (20 - 9.33 - 10)/160 = 0.99581, which is
# what prints as 1.00 to two decimals.
ANSIS3_5_ANNEX_C1_LEVEL_DISTORTION_I5 = 1.00
# ANSI S3.5-1997 Table 1 in full, 21 rows: nominal centre frequency, lower and
# upper band limit, band importance Ii, normal-effort standard speech spectrum
# level Ui and reference internal noise spectrum level Xi.
#
# Provenance differs by column and is worth stating, because this table is a
# *pinning* fixture rather than an independent oracle for most of its cells:
#   - the band limits, Ii, Ui and Xi digits are the WG S3-79 reference
#     implementation SII.C (its cb() limit[], i_avg[], u[] and x[] arrays);
#   - the nominal centre frequencies are NOT in SII.C, which works from the
#     geometric centre of each band's limits. They are the classical
#     critical-band (Bark) centres, and the R CRAN package "SII" transcribes
#     the same values as Table 1's fi column;
#   - rows 1 to 6 are independently corroborated: the CRAN vignette prints
#     that head verbatim and it agrees cell for cell.
#
# The eight official .TST cases do not exercise most of these cells (a gross
# corruption of, say, Xi row 11 leaves every published result unchanged,
# because a 70 dB masker drowns it), so the table is asserted directly.
ANSIS3_5_CRITICAL_TABLE1 = (
    (150.0, 100.0, 200.0, 0.0103, 31.44, 1.5),
    (250.0, 200.0, 300.0, 0.0261, 34.75, -3.9),
    (350.0, 300.0, 400.0, 0.0419, 34.14, -7.2),
    (450.0, 400.0, 510.0, 0.0577, 34.58, -8.9),
    (570.0, 510.0, 630.0, 0.0577, 33.17, -10.3),
    (700.0, 630.0, 770.0, 0.0577, 30.64, -11.4),
    (840.0, 770.0, 920.0, 0.0577, 27.59, -12.0),
    (1000.0, 920.0, 1080.0, 0.0577, 25.01, -12.5),
    (1170.0, 1080.0, 1270.0, 0.0577, 23.52, -13.2),
    (1370.0, 1270.0, 1480.0, 0.0577, 22.28, -14.0),
    (1600.0, 1480.0, 1720.0, 0.0577, 20.15, -15.4),
    (1850.0, 1720.0, 2000.0, 0.0577, 18.29, -16.9),
    (2150.0, 2000.0, 2320.0, 0.0577, 16.37, -18.8),
    (2500.0, 2320.0, 2700.0, 0.0577, 13.80, -21.2),
    (2900.0, 2700.0, 3150.0, 0.0577, 12.21, -23.2),
    (3400.0, 3150.0, 3700.0, 0.0577, 11.09, -24.9),
    (4000.0, 3700.0, 4400.0, 0.0577, 9.33, -25.9),
    (4800.0, 4400.0, 5300.0, 0.0460, 5.84, -24.2),
    (5800.0, 5300.0, 6400.0, 0.0343, 3.47, -19.0),
    (7000.0, 6400.0, 7700.0, 0.0226, 1.78, -11.7),
    (8500.0, 7700.0, 9500.0, 0.0110, -0.14, -6.0),
)
# ANSI S3.5-1997 Table 3 in full, 18 rows: centre frequency in hertz, band
# importance Ii, normal-effort standard speech spectrum level Ui and reference
# internal noise spectrum level Xi. Unlike Tables 1 and 4 this one has an
# independent digit source: the Hornsby SII worksheet (ANSIS3_51997SII.xlsx)
# transcribes it directly from the standard, and SII.C's to()/u[]/x[] arrays
# agree with it cell for cell. The band limits are omitted because Table 3
# prints centre frequencies only; the procedure's exact 2**(-+1/6) fi limits
# are computed, not tabulated.
#
# Pinned directly for the same reason as Tables 1 and 4: the mutation campaign
# showed Xi at 2500 Hz and 3150 Hz surviving a whole-decibel corruption,
# because the band audibility there is clipped at 1 in every case the suite
# runs, so nothing else notices.
ANSIS3_5_THIRD_OCTAVE_TABLE3 = (
    (160.0, 0.0083, 32.41, 0.6),
    (200.0, 0.0095, 34.48, -1.7),
    (250.0, 0.0150, 34.75, -3.9),
    (315.0, 0.0289, 33.98, -6.1),
    (400.0, 0.0440, 34.59, -8.2),
    (500.0, 0.0578, 34.27, -9.7),
    (630.0, 0.0653, 32.06, -10.8),
    (800.0, 0.0711, 28.30, -11.9),
    (1000.0, 0.0818, 25.01, -12.5),
    (1250.0, 0.0844, 23.00, -13.5),
    (1600.0, 0.0882, 20.15, -15.4),
    (2000.0, 0.0898, 17.32, -17.7),
    (2500.0, 0.0868, 13.18, -21.2),
    (3150.0, 0.0844, 11.55, -24.2),
    (4000.0, 0.0771, 9.33, -25.9),
    (5000.0, 0.0527, 5.31, -23.6),
    (6300.0, 0.0364, 2.59, -15.8),
    (8000.0, 0.0185, 1.13, -7.1),
)
# ANSI S3.5-1997 Table 4 in full, 6 rows, same column order as Table 1 above.
# Same provenance: the limits, Ii, Ui and Xi are SII.C's oct() arrays; the
# nominal centres are the octave centres those limits bracket.
ANSIS3_5_OCTAVE_TABLE4 = (
    (250.0, 177.0, 354.0, 0.0617, 34.75, -3.9),
    (500.0, 354.0, 707.0, 0.1671, 34.27, -9.7),
    (1000.0, 707.0, 1414.0, 0.2373, 25.01, -12.5),
    (2000.0, 1414.0, 2828.0, 0.2648, 17.32, -17.7),
    (4000.0, 2828.0, 5657.0, 0.2142, 9.33, -25.9),
    (8000.0, 5657.0, 11314.0, 0.0549, 1.13, -7.1),
)
# Flat-spectrum cases run through SII.C to bring every band's Ui and Xi into
# the chain, which the eight official .TST cases do not do (their strong
# maskers and their -10 dB bands leave many table cells inert). Per row: the
# method, the flat equivalent speech spectrum level, the flat equivalent noise
# spectrum level, normal hearing throughout, and the value SII.C prints.
#   "quiet" (0 / -80 dB): the disturbance is Xi in every band, so every Xi
#       cell moves the answer.
#   "loud"  (80 / 20 dB): the clause 5.7 level-distortion factor is below
#       unity in every band, so every Ui cell moves the answer.
#   "mid"   (40 / -50 dB): both mechanisms partly active.
ANSIS3_5_WG_FLAT_CASES = (
    ("critical-band", "quiet", 0.0, -80.0, 0.9243383333),
    ("critical-band", "loud", 80.0, 20.0, 0.6892747063),
    ("critical-band", "mid", 40.0, -50.0, 0.9342973562),
    ("equally-contributing", "quiet", 0.0, -80.0, 0.9398200000),
    ("equally-contributing", "loud", 80.0, 20.0, 0.6877689000),
    ("equally-contributing", "mid", 40.0, -50.0, 0.9330641250),
    ("one-third-octave", "quiet", 0.0, -80.0, 0.9247666667),
    ("one-third-octave", "loud", 80.0, 20.0, 0.6891314875),
    ("one-third-octave", "mid", 40.0, -50.0, 0.9339307437),
    ("octave", "quiet", 0.0, -80.0, 0.9134180000),
    ("octave", "loud", 80.0, 20.0, 0.6903270250),
    ("octave", "mid", 40.0, -50.0, 0.9340358250),
)

# ---------------------------------------------------------------------------
# Speech transmission index - IEC 60268-16 Annex M, Table M.1 "Example
# calculation" (Ed.4, 2011, printed pp. 64-66): a measured STI adjusted to
# simulate occupancy noise and a different speech level. The annex walks four
# numbered steps and prints every intermediate, so each row below is one
# printed row of that table, transcribed verbatim.
#
#   step 1  the signal and background-noise octave-band levels present during
#           the measurement, and the MTF matrix measured with the noise,
#           masking and threshold in it (rows are the 14 modulation
#           frequencies 0,63 Hz to 12,5 Hz, columns the 7 octave bands 125 Hz
#           to 8 kHz);
#   step 2  the same correction the forward chain applies, divided out, which
#           leaves the transmission channel alone;
#   step 3  the correction of the operational speech and occupancy-noise
#           levels, applied;
#   step 4  the A.5.4 to A.5.6 processing into effective SNRs, band MTIs and
#           the STI.
#
# Two scale conventions of the printed table matter when these rows are read
# against a computed quantity, and both are recorded in docs/ERRATA.md. The
# combined squared sound pressure I_k is tabulated in units of
# IEC60268_16_ANNEX_M_INTENSITY_SCALE (a row labelled "MPa2"), while
# I_am,k and I_rt,k in the same sum are tabulated unscaled; and the auditory
# masking factor has no value in the 125 Hz band, which has no band below it
# to be masked by, where the table prints "not applicable" and these rows
# carry None.
# ---------------------------------------------------------------------------
#: Units the "Combined squared sound pressure I_k" row is printed in, as a
#: ratio to p0^2 = (20 uPa)^2. The rows it is added to are printed unscaled.
IEC60268_16_ANNEX_M_INTENSITY_SCALE = 1e6

# Step 1: acquire measurement data.
IEC60268_16_ANNEX_M_MEASURED_LEVEL = (77.9, 77.9, 74.2, 68.2, 62.2, 56.2, 50.2)
IEC60268_16_ANNEX_M_MEASURED_AMBIENT = (48.0, 40.0, 34.0, 30.0, 27.0, 25.0, 23.0)
IEC60268_16_ANNEX_M_MEASURED_MTF = (
    (0.982, 0.952, 0.960, 0.969, 0.979, 0.983, 0.994),
    (0.966, 0.928, 0.941, 0.954, 0.969, 0.976, 0.992),
    (0.945, 0.897, 0.914, 0.933, 0.955, 0.965, 0.989),
    (0.919, 0.862, 0.881, 0.908, 0.939, 0.952, 0.984),
    (0.884, 0.819, 0.836, 0.873, 0.915, 0.932, 0.978),
    (0.850, 0.784, 0.793, 0.838, 0.890, 0.911, 0.971),
    (0.815, 0.750, 0.749, 0.799, 0.862, 0.888, 0.961),
    (0.772, 0.715, 0.716, 0.760, 0.832, 0.863, 0.950),
    (0.740, 0.678, 0.691, 0.730, 0.800, 0.836, 0.938),
    (0.724, 0.623, 0.665, 0.721, 0.772, 0.811, 0.926),
    (0.713, 0.553, 0.643, 0.708, 0.745, 0.785, 0.913),
    (0.669, 0.515, 0.611, 0.664, 0.720, 0.764, 0.901),
    (0.590, 0.479, 0.545, 0.603, 0.693, 0.748, 0.890),
    (0.553, 0.442, 0.513, 0.602, 0.678, 0.736, 0.881),
)

# Step 2: remove background noise, masking and threshold factors. The two
# "adjustment" rows are reciprocals of the corresponding transfer factors,
# which is how the annex writes a correction it is undoing.
IEC60268_16_ANNEX_M_MEASURED_SNR = (29.90, 37.90, 40.20, 38.20, 35.20, 31.20, 27.20)
IEC60268_16_ANNEX_M_MEASURED_NOISE_TRANSFER = (
    0.999,
    1.000,
    1.000,
    1.000,
    1.000,
    0.999,
    0.998,
)
IEC60268_16_ANNEX_M_MEASURED_NOISE_ADJUSTMENT = (
    1.001,
    1.000,
    1.000,
    1.000,
    1.000,
    1.001,
    1.002,
)
IEC60268_16_ANNEX_M_MEASURED_COMBINED_LEVEL = (
    77.90,
    77.90,
    74.20,
    68.20,
    62.20,
    56.20,
    50.21,
)
IEC60268_16_ANNEX_M_MEASURED_MASKING_DB = (
    None,
    -20.8,
    -20.8,
    -22.7,
    -25.7,
    -33.9,
    -36.9,
)
IEC60268_16_ANNEX_M_MEASURED_INTENSITY = (61.7, 61.7, 26.3, 6.61, 1.66, 0.417, 0.105)
IEC60268_16_ANNEX_M_MEASURED_MASKING_MILLI = (
    None,
    8.22,
    8.22,
    5.37,
    2.69,
    0.407,
    0.204,
)
IEC60268_16_ANNEX_M_MEASURED_INTENSITY_MASKING = (
    0.0,
    508_000.0,
    507_000.0,
    141_000.0,
    17_800.0,
    676.0,
    85.2,
)
IEC60268_16_ANNEX_M_MEASURED_MASKING_THRESHOLD_ADJUSTMENT = (
    1.001,
    1.008,
    1.019,
    1.021,
    1.011,
    1.002,
    1.001,
)
IEC60268_16_ANNEX_M_MEASURED_COMBINED_ADJUSTMENT = (
    1.002,
    1.008,
    1.019,
    1.022,
    1.011,
    1.002,
    1.003,
)
#: Adjusted MTF matrix without noise, masking and threshold (step 2 output).
IEC60268_16_ANNEX_M_SOURCE_MTF = (
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

#: Absolute speech reception threshold ART_k, printed once per step with the
#: same values (Table A.2 of the same edition), and the intensity it becomes.
#: The step 3 print of the 250 Hz cell reads 500 where step 2 reads 501; both
#: are roundings of 10^2,7 = 501,19, at three and at two significant figures.
IEC60268_16_ANNEX_M_ART_DB = (46.0, 27.0, 12.0, 6.5, 7.5, 8.0, 12.0)
IEC60268_16_ANNEX_M_INTENSITY_THRESHOLD = (
    40_000.0,
    501.0,
    15.8,
    4.5,
    5.6,
    6.3,
    15.8,
)

# Step 3: adjust for operational speech and occupancy-noise levels. The annex
# prints these two as transfer factors, not as reciprocals, because here the
# correction is being applied rather than undone.
IEC60268_16_ANNEX_M_OPERATIONAL_LEVEL = (82.9, 82.9, 79.2, 73.2, 67.2, 61.2, 55.2)
IEC60268_16_ANNEX_M_OPERATIONAL_AMBIENT = (55.5, 47.5, 41.5, 37.5, 34.5, 32.5, 30.5)
IEC60268_16_ANNEX_M_OPERATIONAL_SNR = (27.40, 35.40, 37.70, 35.70, 32.70, 28.70, 24.70)
IEC60268_16_ANNEX_M_OPERATIONAL_NOISE_TRANSFER = (
    0.998,
    1.000,
    1.000,
    1.000,
    0.999,
    0.999,
    0.997,
)
IEC60268_16_ANNEX_M_OPERATIONAL_COMBINED_LEVEL = (
    82.9,
    82.9,
    79.2,
    73.2,
    67.2,
    61.2,
    55.2,
)
IEC60268_16_ANNEX_M_OPERATIONAL_MASKING_DB = (
    None,
    -18.3,
    -18.3,
    -20.2,
    -23.2,
    -26.2,
    -34.4,
)
IEC60268_16_ANNEX_M_OPERATIONAL_INTENSITY = (
    195.0,
    195.0,
    83.2,
    20.9,
    5.25,
    1.32,
    0.332,
)
IEC60268_16_ANNEX_M_OPERATIONAL_MASKING_MILLI = (
    None,
    14.6,
    14.6,
    9.55,
    4.79,
    2.40,
    0.363,
)
#: I_am,k of step 3. The 250 Hz cell is printed 2 850 000 where the quantity
#: it names is 2 858 700; see docs/ERRATA.md, "IEC 60268-16:2011, Table M.1,
#: step 3 I_am,k at 250 Hz".
IEC60268_16_ANNEX_M_OPERATIONAL_INTENSITY_MASKING = (
    0.0,
    2_850_000.0,
    2_850_000.0,
    795_000.0,
    100_000.0,
    12_600.0,
    480.0,
)
IEC60268_16_ANNEX_M_OPERATIONAL_MASKING_THRESHOLD_TRANSFER = (
    1.000,
    0.986,
    0.967,
    0.963,
    0.981,
    0.991,
    0.999,
)
IEC60268_16_ANNEX_M_OPERATIONAL_COMBINED_ADJUSTMENT = (
    0.998,
    0.985,
    0.967,
    0.963,
    0.981,
    0.989,
    0.995,
)
#: Adjusted MTF matrix for operational levels, masking and threshold.
IEC60268_16_ANNEX_M_OPERATIONAL_MTF = (
    (0.981, 0.946, 0.946, 0.953, 0.971, 0.975, 0.992),
    (0.966, 0.922, 0.927, 0.938, 0.961, 0.968, 0.990),
    (0.945, 0.891, 0.900, 0.918, 0.947, 0.957, 0.987),
    (0.919, 0.856, 0.868, 0.893, 0.931, 0.944, 0.982),
    (0.884, 0.814, 0.823, 0.859, 0.907, 0.925, 0.976),
    (0.850, 0.779, 0.781, 0.824, 0.882, 0.904, 0.969),
    (0.814, 0.745, 0.738, 0.786, 0.855, 0.881, 0.959),
    (0.772, 0.710, 0.706, 0.747, 0.825, 0.856, 0.948),
    (0.739, 0.674, 0.681, 0.718, 0.793, 0.829, 0.936),
    (0.724, 0.619, 0.656, 0.709, 0.765, 0.804, 0.924),
    (0.713, 0.549, 0.634, 0.696, 0.739, 0.778, 0.911),
    (0.668, 0.512, 0.602, 0.653, 0.714, 0.757, 0.900),
    (0.589, 0.476, 0.537, 0.593, 0.687, 0.741, 0.889),
    (0.553, 0.439, 0.505, 0.592, 0.672, 0.729, 0.880),
)

# Step 4: process the MTF matrix to yield the STI (4a effective SNRs, 4b the
# +/-15 dB clamp, 4c the transmission indices and the band MTIs).
IEC60268_16_ANNEX_M_EFFECTIVE_SNR = (
    (17.21, 12.44, 12.42, 13.09, 15.21, 15.93, 21.01),
    (14.55, 10.73, 11.04, 11.83, 13.90, 14.83, 20.02),
    (12.34, 9.13, 9.56, 10.47, 12.52, 13.50, 18.86),
    (10.52, 7.74, 8.17, 9.22, 11.31, 12.30, 17.41),
    (8.82, 6.41, 6.69, 7.84, 9.91, 10.88, 16.13),
    (7.52, 5.47, 5.52, 6.71, 8.76, 9.73, 14.98),
    (6.42, 4.66, 4.51, 5.64, 7.70, 8.69, 13.72),
    (5.29, 3.89, 3.80, 4.71, 6.73, 7.75, 12.64),
    (4.53, 3.16, 3.30, 4.06, 5.84, 6.87, 11.68),
    (4.19, 2.11, 2.79, 3.87, 5.14, 6.12, 10.87),
    (3.95, 0.85, 2.38, 3.60, 4.51, 5.44, 10.13),
    (3.04, 0.21, 1.80, 2.74, 3.97, 4.94, 9.52),
    (1.57, -0.42, 0.65, 1.63, 3.42, 4.57, 9.02),
    (0.92, -1.06, 0.10, 1.61, 3.12, 4.31, 8.64),
)
IEC60268_16_ANNEX_M_MTI = (0.73, 0.66, 0.67, 0.71, 0.77, 0.80, 0.92)
IEC60268_16_ANNEX_M_STI = 0.76
