#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Hearing-model metrics: loudness, tonality, roughness, fluctuation strength.

The calibration anchors of the perceptual metrics. Each standard in this
family defines its unit by one signal - a 1 kHz tone at 40 dB is one sone,
one tone unit; a 1 kHz carrier fully modulated at 70 Hz and 60 dB is one
asper; the same carrier at 4 Hz is one vacil - and then fixes a constant
so the published chain returns it. ECMA-418-2, ISO 532-2 and ISO 532-3
supply those constants and the thresholds above which a metric counts as
prominent; ECMA-418-1 supplies the critical-band and proximity anchors of
the discrete-tone method; ISO 226 supplies the equal-loudness contour.

The psychoacoustic annoyance and fluctuation strength values are anchored
to the literature instead, because no standard tabulates them; each one
names its source in the comment above it.
"""

from __future__ import annotations

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
