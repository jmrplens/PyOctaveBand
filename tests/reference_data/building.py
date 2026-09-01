#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Sound insulation of buildings: laboratory, field and prediction oracles.

The worked examples of the standards that measure a building element, rate
it with a single number and predict what it does once built into a
structure: ISO 10140 and ISO 16283 for the measurement, ISO 717-1 and
ISO 717-2 for the airborne and impact ratings, EN 12354 parts 1 to 5 and
ISO 12354 Annexes G and L for the flanking prediction, ISO 12999-1 for the
uncertainty, ISO 15186-1 for the intensity method, ISO 10848 for the
junctions, ISO 9611 for the structure-borne source and ISO 16251-1 for the
floor covering.

They are one subject because they are one chain: every prediction annex
ends on the same rating curve, and every rating is read off a spectrum the
measurement standards define. Splitting them would put the two halves of a
single worked example in two files.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# ISO 717-1 Annex C, Table C.1 - measured airborne sound reduction index R
# (100-3150 Hz, one-third-octave). The worked example gives
# Rw(C;Ctr) = 30(-2;-3) dB with an unfavourable-deviation sum of 31,8 dB.
# ---------------------------------------------------------------------------
ISO717_1_ANNEX_C_R: list[float] = [
    20.4,
    16.3,
    17.7,
    22.6,
    22.4,
    22.7,
    24.8,
    26.6,
    28.0,
    30.5,
    31.8,
    32.5,
    33.4,
    33.0,
    31.0,
    25.5,
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
    18.7,
    19.2,
    20.0,
    *ISO717_1_ANNEX_C_R,
    26.8,
    29.2,
]
ISO717_1_ANNEX_C2_EXPECTED = {
    "rw": 30,
    "c": -2,
    "ctr": -3,
    "c_50_5000": -2,
    "ctr_50_5000": -4,
}

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
# ISO 16283 low-frequency procedure. Neither part publishes a worked example
# of it (Annex B of Part 1 and Annex C of Part 2 are blank recording forms and
# the "Examples" of Annexes D and E are loudspeaker-position drawings), so what
# is tabulated here is the printed trigger, the printed band set, and one
# corner sheet whose combination is recomputed from the printed Formula (13)
# rather than read out of a table.
#
# The trigger is "smaller than 25 m3 (calculated to the nearest cubic metre)",
# printed in ISO 16283-1:2014 Clause 8.1 and 10.4, ISO 16283-2:2020 Clause 6,
# 8.1 and 10.4, and ISO 16283-3:2016 Clause 6, 7.3.1 and 8.4. Rounded half away
# from zero, the last volume that triggers is 24,49 m3 and the first that does
# not is 24,5 m3.
# ---------------------------------------------------------------------------
ISO16283_LF_BANDS_HZ: list[float] = [50.0, 63.0, 80.0]
ISO16283_LF_VOLUME_LIMIT_M3 = 25.0
#: (volume in m3, whether the low-frequency procedure is required).
ISO16283_LF_TRIGGER_CASES: list[tuple[float, bool]] = [
    (8.0, True),
    (24.0, True),
    (24.49, True),
    (24.5, False),
    # "smaller than", strictly: the printed limit itself never triggers.
    (ISO16283_LF_VOLUME_LIMIT_M3, False),
    (25.4, False),
    (40.0, False),
]
#: Four corners by the three low-frequency bands, in dB. The highest corner is
#: a different one in each band, which is what the NOTE under Formula (12)
#: allows ("the values for LCorner may be associated with different corners").
ISO16283_LF_CORNER_LEVELS: list[list[float]] = [
    [56.0, 58.0, 54.0],
    [55.0, 60.0, 53.0],
    [54.0, 57.0, 56.0],
    [53.0, 56.0, 55.0],
]
ISO16283_LF_CORNER_MAXIMA: list[float] = [56.0, 60.0, 56.0]
#: Energy-average levels from the default procedure at the same three bands.
ISO16283_LF_DEFAULT_LEVELS: list[float] = [50.0, 52.0, 49.0]
#: Reverberation time measured in the 63 Hz octave band, in s (Clause 10.4).
ISO16283_LF_T63_OCTAVE_S = 0.72

# ---------------------------------------------------------------------------
# ISO 10140-2:2010 laboratory airborne sound reduction index R (Formula (2)):
# R = L1 - L2 + 10 lg(S/A), A = 0,16 V/T. The reference-curve construction lays
# R exactly on the ISO 717-1 Table 3 shape (100-3150 Hz) by choosing S = A
# (S = 10 m2, A = 0,16 * 50 / 0,8 = 10 m2), so R = L1 - L2 = the reference. The
# 32 dB unfavourable-deviation allowance then permits a 2 dB upward shift of the
# reference (32 dB / 16 bands), giving Rw = curve@500 Hz (52) + 2 = 54 dB - the
# analytic +2-shift anchor (mirrors tests/building/measurement/test_lab_insulation.py).
# ---------------------------------------------------------------------------
ISO10140_2_REF_AIRBORNE_R: list[float] = [
    33,
    36,
    39,
    42,
    45,
    48,
    51,
    52,
    53,
    54,
    55,
    56,
    56,
    56,
    56,
    56,
]
ISO10140_2_REF_AIRBORNE_RW = 54

# ---------------------------------------------------------------------------
# EN 12354-1:2000 Annex H.3 airborne prediction worked example. A separating
# element of Rw = 57 dB and area S = 11,5 m2 is flanked by four elements; each
# contributes an Ff/Fd/Df triplet (12 flanking paths), which with the direct
# Dd path make 13 transmission paths. Energy summation (Formula (26)) gives
# R'w = 52,2 dB -> 52 dB. Row = (label, Rw_flanking, KFf, KFd=KDf, coupling
# length lf). Mirrors tests/building/prediction/test_simplified_model.py (_annex_h_paths).
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
EN12354_1_ANNEX_H3_DNT_W_PRINTED = 53.8  # with the standard's V/(3 S) rounding
EN12354_1_ANNEX_H3_DNT_W_SECOND = 54  # second example: 54,3 ~ 54 dB

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
# 10 lg(50/(6*0,5*11,3)) = +1,69 dB. The example is the self-consistent one
# here: 10 lg(0,16*50/(0,5*11,3)) = 1,5116 dB is the printed row, and
# Formula (13)'s "6" is a rounded 1/0,16 = 6,25, worth 10 lg(6,25/6) = 0,18 dB.
# ISO 12354-3:2017 Formula (4) replaces the 6 with an explicit Csab = 0,16 s/m,
# i.e. it adopts the example's constant. The module implements Formula (13) as
# printed; the single-number oracle D2m,nT,w = 33 reproduces either way.
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
EN12354_4_ANNEX_G_LP_SIDE4_D5 = 44.6  # 72,9 - 28,3

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
    50,
    63,
    80,
    100,
    125,
    160,
    200,
    250,
    315,
    400,
    500,
    630,
    800,
    1000,
    1250,
    1600,
    2000,
    2500,
    3150,
    4000,
    5000,
]
ISO12999_1_ANNEX_B_RI: list[float] = [
    39.5,
    40.3,
    41.6,
    43.1,
    43.3,
    43.1,
    42.5,
    44.7,
    48.0,
    50.5,
    53.2,
    55.9,
    58.1,
    60.0,
    62.2,
    63.7,
    65.4,
    66.8,
    68.4,
    68.8,
    65.1,
]
ISO12999_1_ANNEX_B_UI: list[float] = [
    6.8,
    4.6,
    3.8,
    3.0,
    2.7,
    2.4,
    2.1,
    1.8,
    1.8,
    1.8,
    1.8,
    1.8,
    1.8,
    1.8,
    1.8,
    1.8,
    1.8,
    1.9,
    2.0,
    2.4,
    2.8,
]
ISO12999_1_ANNEX_B_RW = 57.4  # one-decimal Rw (B.2)
ISO12999_1_ANNEX_B_RW_C50_5000 = 56.4  # one-decimal Rw + C50-5000
ISO12999_1_ANNEX_B_RW_CTR50_5000 = 51.1  # one-decimal Rw + Ctr,50-5000
ISO12999_1_ANNEX_B_U_CORR_RW = 1.9  # u(Rw), correlated (B.6)
ISO12999_1_ANNEX_B_U_CORR_C = 2.1  # u(Rw+C50-5000), correlated (B.5)
ISO12999_1_ANNEX_B_U_CORR_CTR = 2.6  # u(Rw+Ctr,50-5000), correlated
ISO12999_1_ANNEX_B_U_UNCORR_C = 0.6  # u(Rw+C50-5000), uncorrelated (B.2)
ISO12999_1_ANNEX_B_U_UNCORR_CTR = 0.8  # u(Rw+Ctr,50-5000), uncorrelated

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
    62.1,
    63.2,
    63.5,
    66.2,
    68.5,
    70.0,
    71.7,
    73.1,
    73.8,
    73.5,
    73.8,
    73.3,
    73.1,
    73.0,
    72.4,
    71.2,
]
ISO717_2_ANNEX_C1_EXPECTED = {
    "ln_w": 79,
    "ci": -11,
    "unfavourable_sum": 28.0,
}
# Same Table C.1, right-hand columns: the floor WITH the floor covering.
# Ln,w = 64 dB, CI = -3 dB, unfavourable-deviation sum 30,0 dB.
ISO717_2_ANNEX_C1_COVERED_LN: list[float] = [
    59.1,
    59.5,
    61.6,
    63.2,
    65.3,
    66.5,
    67.7,
    67.0,
    67.1,
    66.5,
    66.1,
    62.5,
    57.9,
    52.7,
    47.0,
    48.0,
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
    3.0,
    3.7,
    1.9,
    3.0,
    3.2,
    3.5,
    4.0,
    6.1,
    6.7,
    7.0,
    7.7,
    10.8,
    15.2,
    20.3,
    25.4,
    23.2,
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
    (50.0, 3.5),
    (63.0, 3.0),
    (80.0, 2.5),
    (100.0, 2.1),
    (125.0, 1.7),
    (160.0, 1.4),
    (200.0, 1.2),
    (250.0, 1.0),
    (315.0, 0.8),
    (400.0, 0.6),
    (500.0, 0.5),
    (630.0, 0.4),
    (800.0, 0.3),
    (1000.0, 0.3),
    (1250.0, 0.2),
    (1600.0, 0.2),
    (2000.0, 0.1),
    (2500.0, 0.1),
    (3150.0, 0.1),
    (4000.0, 0.1),
    (5000.0, 0.1),
]
ISO15186_1_KC_BANDS = tuple(f for f, _ in ISO15186_1_KC_TABLE_B1)
ISO15186_1_KC_B1_PRINTED = [kc for _, kc in ISO15186_1_KC_TABLE_B1]

# ---------------------------------------------------------------------------
# ISO 16251-1:2014 - impact sound improvement of floor coverings (mock-up).
# The standard's Annex B "Table B.1" is a blank report form (no numeric worked
# example), so the conformance anchor is the ISO 717-2:2020 reference floor:
# weighted_impact_rating(Ln,r,0) must return exactly 78 dB (CI = -11), and a
# zero improvement must give Delta-Lw = 0 (Formula 2: Delta-Lw = 78 - Ln,r,w).
# ---------------------------------------------------------------------------
ISO717_2_REFERENCE_FLOOR_FREQ = [
    100,
    125,
    160,
    200,
    250,
    315,
    400,
    500,
    630,
    800,
    1000,
    1250,
    1600,
    2000,
    2500,
    3150,
]
ISO717_2_REFERENCE_FLOOR_LN_R0 = [
    67.0,
    67.5,
    68.0,
    68.5,
    69.0,
    69.5,
    70.0,
    70.5,
    71.0,
    71.5,
    72.0,
    72.0,
    72.0,
    72.0,
    72.0,
    72.0,
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
    100,
    125,
    160,
    200,
    250,
    315,
    400,
    500,
    630,
    800,
    1000,
    1250,
    1600,
    2000,
    2500,
    3150,
    4000,
    5000,
]
FORET2011_CARPET_ISO16251_DELTA_L: list[float] = [
    5.0,
    8.0,
    10.0,
    14.0,
    18.0,
    23.0,
    30.0,
    31.0,
    39.0,
    49.0,
    53.0,
    57.0,
    60.0,
    67.0,
    68.0,
    71.0,
    74.0,
    72.0,
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
    40,
    40,
    40,
    40,
    41,
    43.5,
    46.1,
    48.5,
    51,
    53.6,
    56,
    58.4,
    61.1,
    63.6,
    65,
    65,
]
ISO10140_5_B1_HEAVY_WALL_RATING = (53, -1, -5)  # Rw (C; Ctr)
ISO10140_5_B1_HEAVY_FLOOR_R: list[float] = [
    40,
    40,
    40,
    40,
    40,
    41.8,
    44.4,
    46.8,
    49.3,
    51.9,
    54.4,
    56.8,
    59.5,
    61.9,
    64.3,
    65,
]
ISO10140_5_B1_HEAVY_FLOOR_RATING = (52, -1, -5)
ISO10140_5_B1_LIGHT_WALL_R: list[float] = [
    27,
    27,
    27,
    27,
    27,
    27,
    27,
    27,
    28,
    30.5,
    32.8,
    35.1,
    37.6,
    40,
    42.3,
    44.6,
]
ISO10140_5_B1_LIGHT_WALL_RATING = (33, -1, -2)
ISO10140_5_C1_FLOOR_C1C2_LN: list[float] = [
    78,
    78,
    78,
    78,
    78,
    78,
    76,
    74,
    72,
    69,
    66,
    63,
    60,
    57,
    54,
    51,
]
ISO10140_5_C1_FLOOR_C1C2_RATING = (72, 0)  # Ln,t,r,0,w (CI)
ISO10140_5_C1_FLOOR_C3_LN: list[float] = [
    69,
    72,
    75,
    78,
    78,
    78,
    78,
    78,
    78,
    76,
    74,
    72,
    69,
    66,
    63,
    60,
]
ISO10140_5_C1_FLOOR_C3_RATING = (75, -3)

# ---------------------------------------------------------------------------
# Hopkins, Sound Insulation (2007), Table A2 (printed p. 608 / pdf p. 635):
# material properties of 25 building-material rows. Each row pairs the
# quasi-longitudinal thin-plate phase speed cL (m/s) with the printed product
# h.fc (m.Hz), stated for c0 = 343 m/s. The product follows from
# h fc = c0^2 sqrt(12) / (2 pi cL) and is independent of density and Poisson
# ratio, so every row is an independent check on the coincidence frequency.
# ---------------------------------------------------------------------------
HOPKINS_TABLE_A2_H_FC: tuple[tuple[float, float], ...] = (
    (1900.0, 34.1),  # aircrete / AAC blocks (solid)
    (5100.0, 12.7),  # aluminium
    (2700.0, 24.0),  # bricks (solid)
    (2500.0, 25.9),  # calcium-silicate blocks (solid)
    (2200.0, 29.5),  # chipboard
    (1850.0, 35.1),  # clinker concrete blocks, 1030 kg/m3
    (2200.0, 29.5),  # clinker concrete blocks, 1720 kg/m3
    (1910.0, 34.0),  # clinker concrete slabs
    (3800.0, 17.1),  # concrete, cast in situ
    (3200.0, 20.3),  # dense aggregate blocks (solid)
    (2300.0, 28.2),  # expanded clay blocks (solid)
    (5200.0, 12.5),  # glass
    (2200.0, 29.5),  # lightweight aggregate blocks (solid)
    (2560.0, 25.3),  # medium density fibreboard
    (2450.0, 26.5),  # mortar
    (2570.0, 25.2),  # oriented strand board
    (2350.0, 27.6),  # perspex, plexiglass
    (1610.0, 40.3),  # plaster, gypsum based
    (1490.0, 43.5),  # plasterboard, natural gypsum
    (1810.0, 35.8),  # plasterboard, flue gas plus natural gypsum
    (2010.0, 32.3),  # plasterboard, gypsum with glass fibre
    (3850.0, 16.8),  # plywood (birch)
    (3250.0, 20.0),  # sand-cement screed
    (5270.0, 12.3),  # steel
    (5000.0, 13.0),  # timber (soft wood)
)

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
    50.0,
    63.0,
    80.0,
    100.0,
    125.0,
    160.0,
    200.0,
    250.0,
    315.0,
    400.0,
    500.0,
    630.0,
    800.0,
    1000.0,
    1250.0,
    1600.0,
    2000.0,
    2500.0,
    3150.0,
    4000.0,
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
    "floor": (
        0.7209,
        0.8092,
        0.9119,
        1.0196,
        1.1399,
        1.2896,
        1.2742,
        1.2015,
        1.1500,
        1.1125,
        1.0870,
        1.0672,
        1.0518,
        1.0408,
        1.0322,
        1.0249,
        1.0198,
        1.0157,
        1.0124,
        1.0097,
        1.0078,
    ),
    "ext1": (
        0.6243,
        0.7008,
        0.7897,
        0.8830,
        0.9872,
        1.1169,
        1.2487,
        1.2603,
        1.1901,
        1.1407,
        1.1078,
        1.0827,
        1.0634,
        1.0498,
        1.0392,
        1.0303,
        1.0240,
        1.0191,
        1.0150,
        1.0118,
        1.0094,
    ),
    "ext2": (
        0.6690,
        0.7510,
        0.8462,
        0.9461,
        1.0578,
        1.1967,
        1.3380,
        1.2603,
        1.1901,
        1.1407,
        1.1078,
        1.0827,
        1.0634,
        1.0498,
        1.0392,
        1.0303,
        1.0240,
        1.0191,
        1.0150,
        1.0118,
        1.0094,
    ),
    "int1": (
        0.3929,
        0.5190,
        0.8473,
        1.8783,
        2.0000,
        2.0000,
        1.6718,
        1.4341,
        1.2994,
        1.2137,
        1.1600,
        1.1208,
        1.0915,
        1.0712,
        1.0557,
        1.0427,
        1.0337,
        1.0267,
        1.0210,
        1.0165,
        1.0131,
    ),
    "int2": (
        0.3581,
        0.4765,
        0.7783,
        1.7252,
        2.0000,
        2.0000,
        1.6718,
        1.4341,
        1.2994,
        1.2137,
        1.1600,
        1.1208,
        1.0915,
        1.0712,
        1.0557,
        1.0427,
        1.0337,
        1.0267,
        1.0210,
        1.0165,
        1.0131,
    ),
}
# Table L.2 / G.2 - radiation factor for forced waves (depends only on the
# element dimensions, so the 4,00 x 2,75 and 5,00 x 2,75 walls pair up).
ISO12354_ANNEX_L2_SIGMA_F: dict[str, tuple[float, ...]] = {
    "floor": (
        0.7912,
        0.9059,
        1.0248,
        1.1361,
        1.2474,
        1.3707,
        1.4822,
        1.5937,
        1.7092,
        1.8287,
        1.9402,
        2.0000,
        2.0000,
        2.0000,
        2.0000,
        2.0000,
        2.0000,
        2.0000,
        2.0000,
        2.0000,
        2.0000,
    ),
    "ext1": (
        0.6380,
        0.7520,
        0.8704,
        0.9814,
        1.0926,
        1.2157,
        1.3271,
        1.4386,
        1.5541,
        1.6735,
        1.7851,
        1.9006,
        2.0000,
        2.0000,
        2.0000,
        2.0000,
        2.0000,
        2.0000,
        2.0000,
        2.0000,
        2.0000,
    ),
    "ext2": (
        0.6805,
        0.7948,
        0.9134,
        1.0245,
        1.1358,
        1.2590,
        1.3705,
        1.4820,
        1.5975,
        1.7169,
        1.8284,
        1.9440,
        2.0000,
        2.0000,
        2.0000,
        2.0000,
        2.0000,
        2.0000,
        2.0000,
        2.0000,
        2.0000,
    ),
}

# Table L.3 / G.3 - in-situ total loss factor.
ISO12354_ANNEX_L3_ETA: dict[str, tuple[float, ...]] = {
    "floor": (
        0.0831,
        0.0746,
        0.0667,
        0.0602,
        0.0544,
        0.0486,
        0.0438,
        0.0394,
        0.0355,
        0.0319,
        0.0290,
        0.0263,
        0.0239,
        0.0218,
        0.0200,
        0.0183,
        0.0168,
        0.0156,
        0.0144,
        0.0133,
        0.0124,
    ),
    "ext1": (
        0.1298,
        0.1170,
        0.1052,
        0.0954,
        0.0867,
        0.0781,
        0.0711,
        0.0646,
        0.0585,
        0.0530,
        0.0485,
        0.0444,
        0.0407,
        0.0376,
        0.0349,
        0.0322,
        0.0301,
        0.0282,
        0.0265,
        0.0249,
        0.0236,
    ),
    "ext2": (
        0.1149,
        0.1037,
        0.0934,
        0.0849,
        0.0772,
        0.0697,
        0.0637,
        0.0577,
        0.0523,
        0.0475,
        0.0436,
        0.0400,
        0.0368,
        0.0342,
        0.0318,
        0.0295,
        0.0277,
        0.0260,
        0.0245,
        0.0232,
        0.0220,
    ),
    "int1": (
        0.0770,
        0.0702,
        0.0647,
        0.0625,
        0.0566,
        0.0506,
        0.0452,
        0.0408,
        0.0371,
        0.0338,
        0.0311,
        0.0287,
        0.0265,
        0.0247,
        0.0231,
        0.0216,
        0.0203,
        0.0192,
        0.0182,
        0.0172,
        0.0165,
    ),
    "int2": (
        0.0703,
        0.0642,
        0.0592,
        0.0574,
        0.0526,
        0.0470,
        0.0420,
        0.0379,
        0.0345,
        0.0315,
        0.0291,
        0.0269,
        0.0249,
        0.0233,
        0.0218,
        0.0204,
        0.0193,
        0.0183,
        0.0174,
        0.0165,
        0.0158,
    ),
}
# Table L.3 / G.3 - in-situ sound reduction index.
ISO12354_ANNEX_L3_R_SITU: dict[str, tuple[float, ...]] = {
    "floor": (
        31.8,
        31.5,
        35.9,
        37.6,
        39.1,
        40.8,
        43.3,
        46.3,
        49.2,
        52.2,
        54.9,
        57.6,
        60.4,
        63.0,
        65.6,
        68.5,
        71.1,
        73.3,
        73.0,
        72.6,
        72.3,
    ),
    "ext1": (
        27.4,
        28.2,
        26.2,
        32.8,
        34.7,
        36.4,
        37.9,
        40.3,
        43.4,
        46.4,
        49.2,
        52.0,
        54.9,
        57.6,
        59.5,
        59.2,
        58.9,
        58.6,
        58.3,
        58.0,
        57.8,
    ),
    "ext2": (
        26.4,
        27.3,
        25.7,
        31.7,
        33.6,
        35.3,
        36.8,
        39.8,
        42.9,
        46.0,
        48.8,
        51.6,
        54.5,
        57.2,
        59.1,
        58.8,
        58.5,
        58.2,
        58.0,
        57.7,
        57.5,
    ),
    "int1": (
        32.3,
        32.5,
        31.2,
        27.2,
        29.7,
        32.3,
        36.3,
        40.1,
        43.5,
        46.8,
        49.8,
        52.7,
        55.7,
        58.5,
        61.3,
        64.3,
        67.0,
        68.8,
        68.6,
        68.4,
        68.2,
    ),
    "int2": (
        32.5,
        32.7,
        31.4,
        27.5,
        29.4,
        32.0,
        36.0,
        39.8,
        43.2,
        46.5,
        49.5,
        52.5,
        55.5,
        58.3,
        61.0,
        64.0,
        66.8,
        68.6,
        68.4,
        68.2,
        68.0,
    ),
}
# Table G.3 - in-situ normalized impact level of the bare separating floor.
ISO12354_ANNEX_G3_LN_SITU = (
    57.3,
    58.2,
    59.2,
    60.2,
    61.1,
    62.1,
    62.5,
    62.7,
    63.0,
    63.3,
    63.6,
    64.0,
    64.3,
    64.7,
    65.0,
    65.4,
    65.7,
    66.0,
    66.3,
    66.7,
    67.0,
)

# Table L.4 / G.4 - in-situ equivalent absorption lengths (the D1 block gives
# the floor and external wall 1; the second block gives internal wall 2).
ISO12354_ANNEX_L4_ABSORPTION: dict[str, tuple[float, ...]] = {
    "floor": (
        10.8,
        10.9,
        11.0,
        11.1,
        11.2,
        11.3,
        11.4,
        11.4,
        11.6,
        11.7,
        11.9,
        12.1,
        12.4,
        12.7,
        13.0,
        13.4,
        13.8,
        14.3,
        14.8,
        15.5,
        16.2,
    ),
    "ext1": (
        9.3,
        9.4,
        9.5,
        9.6,
        9.8,
        10.0,
        10.2,
        10.3,
        10.5,
        10.7,
        10.9,
        11.3,
        11.6,
        12.0,
        12.5,
        13.0,
        13.6,
        14.2,
        15.0,
        15.9,
        16.8,
    ),
    "int2": (
        6.3,
        6.4,
        6.7,
        7.2,
        7.4,
        7.5,
        7.5,
        7.6,
        7.7,
        8.0,
        8.2,
        8.5,
        8.9,
        9.3,
        9.7,
        10.3,
        10.9,
        11.5,
        12.3,
        13.2,
        14.1,
    ),
}
# Table L.4 / G.4 - in-situ velocity level differences of the two printed paths.
ISO12354_ANNEX_L4_DV: dict[str, tuple[float, ...]] = {
    "D1": (
        10.4,
        10.4,
        10.4,
        10.5,
        10.5,
        10.6,
        10.7,
        10.7,
        10.8,
        10.8,
        10.9,
        11.0,
        11.1,
        11.3,
        11.4,
        11.6,
        11.7,
        11.9,
        12.1,
        12.3,
        12.5,
    ),
    "4d": (
        11.0,
        11.0,
        11.1,
        11.3,
        11.4,
        11.4,
        11.5,
        11.5,
        11.6,
        11.7,
        11.8,
        11.9,
        12.0,
        12.2,
        12.3,
        12.5,
        12.7,
        12.9,
        13.1,
        13.4,
        13.6,
    ),
}
# Table L.4 / G.4 - improvement of the floating floor, 30 lg(f/f0).
ISO12354_ANNEX_L4_DELTA = (
    0.0,
    2.3,
    5.4,
    8.3,
    11.2,
    14.4,
    17.4,
    20.3,
    23.3,
    26.4,
    29.3,
    32.3,
    35.4,
    38.3,
    41.2,
    44.4,
    47.4,
    50.3,
    53.3,
    56.4,
    59.3,
)

# Table L.1 - direct and flanking sound reduction indices, per path.
ISO12354_ANNEX_L1_PATHS: dict[str, tuple[float, ...]] = {
    "Dd": (
        31.8,
        33.8,
        41.4,
        45.9,
        50.3,
        55.2,
        60.7,
        66.5,
        72.5,
        78.5,
        84.1,
        89.9,
        95.8,
        101.3,
        106.9,
        112.9,
        118.5,
        123.6,
        126.2,
        129.0,
        131.6,
    ),
    "1d": (
        41.2,
        41.5,
        42.8,
        47.0,
        48.7,
        50.5,
        52.6,
        55.3,
        58.4,
        61.4,
        64.3,
        67.2,
        70.1,
        72.9,
        75.3,
        76.7,
        78.0,
        79.1,
        79.0,
        78.9,
        78.9,
    ),
    "2d": (
        39.5,
        39.9,
        41.3,
        45.2,
        47.0,
        48.7,
        50.8,
        53.9,
        56.9,
        60.0,
        62.8,
        65.7,
        68.7,
        71.5,
        73.9,
        75.3,
        76.7,
        77.8,
        77.7,
        77.7,
        77.6,
    ),
    "3d": (
        45.0,
        45.0,
        46.6,
        45.7,
        47.8,
        49.9,
        53.2,
        56.6,
        59.9,
        63.1,
        66.0,
        69.0,
        72.0,
        74.8,
        77.7,
        80.8,
        83.6,
        85.8,
        85.8,
        85.7,
        85.7,
    ),
    "4d": (
        43.9,
        44.0,
        45.6,
        44.7,
        46.5,
        48.6,
        51.9,
        55.3,
        58.6,
        61.8,
        64.7,
        67.7,
        70.8,
        73.6,
        76.4,
        79.6,
        82.4,
        84.7,
        84.6,
        84.6,
        84.6,
    ),
    "D1": (
        41.2,
        43.8,
        48.2,
        55.3,
        60.0,
        64.9,
        69.9,
        75.6,
        81.6,
        87.8,
        93.6,
        99.5,
        105.5,
        111.2,
        116.5,
        121.1,
        125.4,
        129.4,
        132.3,
        135.3,
        138.2,
    ),
    "11": (
        44.8,
        45.7,
        43.8,
        50.5,
        52.4,
        54.2,
        55.8,
        58.3,
        61.4,
        64.5,
        67.4,
        70.4,
        73.4,
        76.2,
        78.3,
        78.1,
        78.0,
        77.9,
        77.9,
        77.9,
        77.9,
    ),
    "D2": (
        39.5,
        42.2,
        46.8,
        53.6,
        58.2,
        63.2,
        68.2,
        74.1,
        80.2,
        86.4,
        92.1,
        98.0,
        104.1,
        109.8,
        115.1,
        119.8,
        124.0,
        128.1,
        131.0,
        134.0,
        136.9,
    ),
    "22": (
        42.4,
        43.4,
        41.8,
        47.9,
        49.8,
        51.6,
        53.3,
        56.3,
        59.5,
        62.6,
        65.5,
        68.5,
        71.6,
        74.4,
        76.5,
        76.4,
        76.3,
        76.3,
        76.2,
        76.3,
        76.3,
    ),
    "D3": (
        45.0,
        47.3,
        52.1,
        54.0,
        59.0,
        64.4,
        70.6,
        76.9,
        83.2,
        89.5,
        95.3,
        101.3,
        107.4,
        113.2,
        118.9,
        125.2,
        131.0,
        136.1,
        139.0,
        142.1,
        145.0,
    ),
    "33": (
        47.2,
        47.5,
        46.4,
        42.8,
        45.3,
        48.0,
        52.0,
        55.8,
        59.3,
        62.8,
        65.8,
        68.9,
        72.1,
        75.1,
        78.0,
        81.2,
        84.2,
        86.3,
        86.3,
        86.4,
        86.4,
    ),
    "D4": (
        43.9,
        46.3,
        51.0,
        53.0,
        57.7,
        63.1,
        69.3,
        75.6,
        81.9,
        88.2,
        94.0,
        100.0,
        106.2,
        111.9,
        117.7,
        124.0,
        129.8,
        134.9,
        137.9,
        141.0,
        143.9,
    ),
    "44": (
        46.1,
        46.4,
        45.3,
        41.7,
        43.7,
        46.4,
        50.4,
        54.2,
        57.7,
        61.2,
        64.3,
        67.4,
        70.6,
        73.6,
        76.5,
        79.8,
        82.8,
        84.9,
        84.9,
        85.0,
        85.1,
    ),
}
ISO12354_ANNEX_L1_R_PRIME = (
    28.8,
    30.4,
    33.4,
    35.3,
    37.6,
    40.0,
    43.1,
    46.4,
    49.7,
    52.9,
    55.9,
    58.9,
    61.9,
    64.8,
    67.3,
    69.0,
    70.2,
    71.0,
    70.9,
    70.9,
    70.9,
)
# ISO 717-1 rating of that spectrum. Table L.1 prints a non-integer 57,8 /
# 57,9 (see docs/ERRATA.md); the ISO 717-1 rating in 1 dB steps is 57 dB.
ISO12354_ANNEX_L1_R_PRIME_W = 57

# Table G.4 - the printed per-band improvement of the floating floor,
# "Reduction of impact sound pressure level DeltaL,situ: Formula (C.1)" with
# "Floating floor, m' = 73,5 kg/m2, s' = 8 MN/m3, f0 = 52,8 Hz; for f > f0:
# DeltaL = 30 lg(f/f0)" (ISO 12354-2:2017 printed p. 39 / pdf p. 45). This is
# the printed per-band realisation of Formula (C.1); ISO 12354-1:2017 Table L.4
# prints the identical column as DeltaRd,situ for the same floor.
ISO12354_ANNEX_G4_DELTA_L = (
    0.0,
    2.3,
    5.4,
    8.3,
    11.2,
    14.4,
    17.4,
    20.3,
    23.3,
    26.4,
    29.3,
    32.3,
    35.4,
    38.3,
    41.2,
    44.4,
    47.4,
    50.3,
    53.3,
    56.4,
    59.3,
)
# Table G.4 - the printed direct and Df (external wall 1) impact levels.
ISO12354_ANNEX_G4_LN_DD = (
    57.3,
    55.9,
    53.8,
    51.8,
    49.9,
    47.7,
    45.2,
    42.5,
    39.7,
    36.9,
    34.3,
    31.7,
    28.9,
    26.3,
    23.8,
    20.9,
    18.4,
    15.8,
    13.1,
    10.3,
    7.7,
)
ISO12354_ANNEX_G4_LN_DF = (
    47.8,
    45.9,
    47.0,
    42.4,
    40.2,
    38.0,
    35.9,
    33.4,
    30.6,
    27.6,
    24.9,
    22.1,
    19.2,
    16.5,
    14.1,
    12.7,
    11.4,
    9.9,
    7.0,
    4.0,
    1.1,
)
# Table G.1 - direct and flanking impact levels per element, and the total.
ISO12354_ANNEX_G1_PATHS: dict[str, tuple[float, ...]] = {
    "Dd": ISO12354_ANNEX_G4_LN_DD,
    "Df1": (
        47.3,
        44.9,
        46.2,
        42.4,
        40.2,
        38.0,
        35.9,
        33.4,
        30.6,
        27.6,
        24.9,
        22.1,
        19.2,
        16.5,
        14.1,
        12.7,
        11.4,
        9.9,
        7.0,
        4.0,
        1.1,
    ),
    "Df2": (
        49.0,
        46.6,
        47.9,
        44.2,
        42.0,
        39.7,
        37.7,
        34.9,
        32.0,
        29.1,
        26.3,
        23.5,
        20.6,
        17.9,
        15.5,
        14.1,
        12.8,
        11.2,
        8.3,
        5.3,
        2.4,
    ),
    "Df3": (
        43.9,
        41.9,
        43.2,
        43.8,
        41.2,
        38.5,
        35.3,
        32.1,
        29.1,
        26.0,
        23.2,
        20.3,
        17.3,
        14.5,
        11.7,
        8.6,
        5.8,
        3.2,
        0.3,
        -2.8,
        -5.7,
    ),
    "Df4": (
        45.0,
        43.0,
        44.3,
        44.9,
        42.5,
        39.8,
        36.6,
        33.4,
        30.3,
        27.3,
        24.4,
        21.5,
        18.5,
        15.8,
        13.0,
        9.8,
        7.0,
        4.4,
        1.4,
        -1.7,
        -4.6,
    ),
}
ISO12354_ANNEX_G1_L_PRIME_N = (
    58.6,
    57.0,
    55.9,
    54.0,
    51.9,
    49.6,
    47.1,
    44.3,
    41.4,
    38.6,
    35.9,
    33.3,
    30.4,
    27.8,
    25.3,
    22.7,
    20.4,
    18.2,
    15.4,
    12.5,
    9.8,
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
    "Dd": 69.3,
    "D1": 76.3,
    "D2": 75.3,
    "D3": 82.7,
    "D4": 81.7,
    "1d": 65.7,
    "11": 64.0,
    "2d": 64.7,
    "22": 63.0,
    "3d": 72.1,
    "33": 71.9,
    "4d": 71.1,
    "44": 70.9,
}
ISO12354_ANNEX_L10_R_PRIME_W = 57.0
ISO12354_ANNEX_G10_LN_EQ_0_W = 70.0
ISO12354_ANNEX_G10_DELTA_LW = 32.2
ISO12354_ANNEX_G10_PATH_LN_W: dict[str, float] = {
    "Dd": 37.8,
    "Df1": 30.9,
    "Df2": 31.9,
    "Df3": 24.4,
    "Df4": 25.4,
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
    24.6,
    23.8,
    33.2,
    36.5,
    42.2,
    48.2,
    53.8,
    55.3,
    58.8,
    63.2,
    67.0,
    69.8,
    72.1,
    73.8,
    76.2,
    77.3,
    76.4,
    77.6,
    80.1,
    84.4,
    86.6,
)
ISO12354_TABLE_L12_R_WALL = (
    15.0,
    18.7,
    20.1,
    21.3,
    20.5,
    22.1,
    24.1,
    24.9,
    25.3,
    27.6,
    28.9,
    31.5,
    32.2,
    33.5,
    34.8,
    35.2,
    34.8,
    32.5,
    34.3,
    38.1,
    40.3,
)
ISO12354_TABLE_L12_R_STAR_WALL = (
    23.0,
    26.7,
    28.1,
    29.3,
    28.5,
    30.1,
    32.1,
    32.9,
    33.3,
    35.6,
    36.9,
    39.5,
    40.2,
    41.5,
    42.8,
    43.2,
    42.8,
    32.5,
    34.3,
    38.1,
    40.3,
)
ISO12354_TABLE_L12_DV_FF = (
    18.7,
    19.0,
    19.4,
    19.7,
    20.0,
    20.4,
    20.7,
    21.0,
    21.3,
    21.7,
    22.0,
    22.3,
    22.7,
    23.0,
    23.3,
    23.7,
    24.0,
    24.3,
    24.6,
    25.0,
    25.3,
)
# Table L.13 - the bare floor index, the floating floor improvement, the
# resonant-only bare floor index and the Df/Fd normalized level difference.
ISO12354_TABLE_L13_R_BARE = (
    10.0,
    12.7,
    14.0,
    15.0,
    13.0,
    14.0,
    17.0,
    20.0,
    23.0,
    25.0,
    22.0,
    26.0,
    28.0,
    29.0,
    30.0,
    28.5,
    27.0,
    29.0,
    36.0,
    38.0,
    41.0,
)
ISO12354_TABLE_L13_DELTA_R = (
    0.0,
    0.0,
    2.5,
    5.0,
    6.5,
    10.0,
    11.0,
    11.5,
    11.5,
    10.5,
    11.0,
    11.5,
    12.0,
    12.0,
    13.5,
    15.0,
    15.0,
    15.0,
    15.0,
    15.0,
    15.0,
)
ISO12354_TABLE_L13_R_STAR_BARE = (
    18.0,
    20.7,
    22.0,
    23.0,
    21.0,
    22.0,
    25.0,
    28.0,
    31.0,
    33.0,
    30.0,
    34.0,
    36.0,
    37.0,
    38.0,
    36.5,
    27.0,
    29.0,
    36.0,
    38.0,
    41.0,
)
ISO12354_TABLE_L13_DV_DF = (
    14.7,
    15.0,
    15.4,
    15.7,
    16.0,
    16.4,
    16.7,
    17.0,
    17.3,
    17.7,
    18.0,
    18.3,
    18.7,
    19.0,
    19.3,
    19.7,
    20.0,
    20.3,
    20.6,
    21.0,
    21.3,
)
# Table L.11 - the resulting Ff and Df paths and the total R'.
ISO12354_TABLE_L11_R_FF = (
    48.7,
    52.7,
    54.5,
    56.0,
    55.5,
    57.5,
    59.8,
    60.9,
    61.6,
    64.3,
    65.9,
    68.8,
    69.9,
    71.5,
    73.1,
    73.9,
    73.8,
    63.8,
    65.9,
    70.1,
    72.6,
)
ISO12354_TABLE_L11_R_DF = (
    42.2,
    45.7,
    49.9,
    53.8,
    54.3,
    59.4,
    63.2,
    65.9,
    68.0,
    69.5,
    69.4,
    73.6,
    75.8,
    77.2,
    80.2,
    81.5,
    76.9,
    73.0,
    77.8,
    81.0,
    83.9,
)
ISO12354_TABLE_L11_R_PRIME = (
    24.5,
    23.8,
    33.1,
    36.4,
    41.8,
    47.4,
    52.4,
    54.0,
    56.6,
    60.2,
    62.4,
    65.5,
    67.2,
    68.8,
    70.8,
    71.7,
    70.7,
    63.1,
    65.5,
    69.6,
    72.1,
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
    51.8,
    53.8,
    55.8,
    57.8,
    59.8,
    61.4,
    63.0,
    65.2,
    65.4,
    68.3,
    68.7,
    72.3,
    76.7,
    76.9,
    78.8,
    78.2,
    71.9,
    72.0,
    77.1,
    81.3,
    84.8,
)
ISO12354_TABLE_L15_R13 = (
    52.1,
    54.1,
    56.1,
    58.1,
    60.1,
    61.7,
    63.3,
    65.5,
    65.7,
    68.6,
    69.0,
    72.6,
    77.0,
    77.2,
    79.1,
    78.5,
    72.2,
    72.3,
    77.4,
    81.6,
    85.1,
)
ISO12354_TABLE_L16_R_SITU = (
    27.6,
    22.3,
    27.4,
    28.9,
    33.6,
    36.9,
    37.6,
    42.0,
    47.3,
    53.1,
    56.7,
    59.3,
    61.7,
    63.3,
    64.4,
    62.7,
    54.7,
    50.8,
    55.3,
    59.1,
    61.8,
)
ISO12354_TABLE_L16_DV = (
    18.3,
    18.0,
    17.6,
    17.3,
    17.0,
    16.6,
    16.3,
    16.0,
    15.7,
    15.3,
    15.0,
    14.7,
    14.3,
    14.0,
    13.7,
    13.3,
    13.0,
    12.7,
    12.4,
    12.0,
    11.7,
)
ISO12354_TABLE_L16_R13_PRED = (
    52.3,
    46.6,
    51.4,
    52.5,
    56.9,
    59.9,
    60.3,
    64.3,
    69.3,
    74.8,
    78.0,
    80.4,
    82.4,
    83.7,
    84.5,
    82.4,
    74.1,
    69.9,
    74.0,
    77.5,
    79.8,
)

# ISO 12354-2 G.2, Tables G.11 to G.13 - the impact side of the same
# lightweight building.
ISO12354_TABLE_G12_LN_BARE = (
    78.2,
    73.5,
    80.0,
    82.0,
    87.0,
    89.0,
    93.0,
    93.0,
    91.0,
    92.0,
    97.0,
    94.0,
    93.0,
    93.0,
    90.0,
    87.0,
    83.0,
    79.0,
    74.0,
    69.0,
    64.0,
)
ISO12354_TABLE_G12_DELTA_LI = (
    0.0,
    1.0,
    3.0,
    5.0,
    7.0,
    8.0,
    10.0,
    13.0,
    16.0,
    18.0,
    19.0,
    19.0,
    20.0,
    20.0,
    20.0,
    20.0,
    20.0,
    20.0,
    20.0,
    20.0,
    20.0,
)
ISO12354_TABLE_G12_DELTA_LDI = (
    15.5,
    6.3,
    13.7,
    15.6,
    19.3,
    19.7,
    22.7,
    22.3,
    19.6,
    19.7,
    25.2,
    23.5,
    22.3,
    23.9,
    21.7,
    21.5,
    20.8,
    17.3,
    15.1,
    18.1,
    21.1,
)
ISO12354_TABLE_G13_R_BARE = ISO12354_TABLE_L13_R_BARE
ISO12354_TABLE_G13_R_WALL = ISO12354_TABLE_L12_R_WALL
ISO12354_TABLE_G13_DV = ISO12354_TABLE_L13_DV_DF
ISO12354_TABLE_G11_LN_DD = (
    62.7,
    66.2,
    63.3,
    61.4,
    60.7,
    61.3,
    60.3,
    57.7,
    55.4,
    54.3,
    52.8,
    51.5,
    50.7,
    49.1,
    48.3,
    45.5,
    42.2,
    41.7,
    38.9,
    30.9,
    22.9,
)
ISO12354_TABLE_G11_LN_DF = (
    54.0,
    47.5,
    51.6,
    51.2,
    53.2,
    53.6,
    55.8,
    53.6,
    49.5,
    48.0,
    49.6,
    46.9,
    45.2,
    44.8,
    41.3,
    37.0,
    32.1,
    30.0,
    27.2,
    21.0,
    16.1,
)
ISO12354_TABLE_G11_LN_TOTAL = (
    63.3,
    66.3,
    63.6,
    61.8,
    61.4,
    62.0,
    61.6,
    59.1,
    56.4,
    55.2,
    54.5,
    52.8,
    51.8,
    50.5,
    49.1,
    46.1,
    42.6,
    42.0,
    39.2,
    31.3,
    23.7,
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
