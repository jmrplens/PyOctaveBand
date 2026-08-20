#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Human vibration and the vibration transmission of mechanical elements.

One subject in two halves that share a weighting chain. The human half is
the ISO 8041-1 frequency weightings (Wk, Wd, Wb, Wc, We, Wf, Wh, Wj, Wm)
with their Annex B factors, Table 4 transitions and Table 5 tolerances,
the ISO 5349 hand-arm exposure examples, the ISO 2631-5 multiple-shock
method and the exposure action and limit values of Directive 2002/44/EC.

The mechanical half is what the structure between source and person does:
the ISO 7626 mobility and accelerance calibration checks and the ISO 10846
limits on the dynamic transfer stiffness of a resilient element.
"""

from __future__ import annotations

import math

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
        (-10, 0.02494),
        (-9, 0.03941),
        (-8, 0.06198),
        (-7, 0.09645),
        (-6, 0.1464),
        (-5, 0.2113),
        (-4, 0.28),
        (-3, 0.3347),
        (-2, 0.3666),
        (-1, 0.3808),
        (0, 0.3853),
        (1, 0.3864),
        (2, 0.3916),
        (3, 0.4168),
        (4, 0.496),
        (5, 0.6653),
        (6, 0.885),
        (7, 1.026),
        (8, 1.054),
        (9, 1.026),
        (10, 0.9745),
        (11, 0.9042),
        (12, 0.8144),
        (13, 0.7088),
        (14, 0.5973),
        (15, 0.4906),
        (16, 0.395),
        (17, 0.3118),
        (18, 0.2389),
        (19, 0.1734),
        (20, 0.1154),
        (21, 0.06929),
        (22, 0.03818),
        (23, 0.01999),
        (24, 0.0102),
        (25, 0.005154),
        (26, 0.002591),
    ),
    "Wc": (
        (-10, 0.06238),
        (-9, 0.09858),
        (-8, 0.1551),
        (-7, 0.2415),
        (-6, 0.3669),
        (-5, 0.5302),
        (-4, 0.7042),
        (-3, 0.8442),
        (-2, 0.9292),
        (-1, 0.9716),
        (0, 0.991),
        (1, 1.0),
        (2, 1.006),
        (3, 1.012),
        (4, 1.017),
        (5, 1.023),
        (6, 1.024),
        (7, 1.013),
        (8, 0.9739),
        (9, 0.8941),
        (10, 0.7762),
        (11, 0.6425),
        (12, 0.5166),
        (13, 0.4098),
        (14, 0.3236),
        (15, 0.2549),
        (16, 0.2002),
        (17, 0.1557),
        (18, 0.1182),
        (19, 0.08538),
        (20, 0.05665),
        (21, 0.03394),
        (22, 0.01868),
        (23, 0.009772),
        (24, 0.004987),
        (25, 0.002518),
        (26, 0.001266),
    ),
    "Wd": (
        (-10, 0.06242),
        (-9, 0.09867),
        (-8, 0.1553),
        (-7, 0.242),
        (-6, 0.3682),
        (-5, 0.533),
        (-4, 0.7097),
        (-3, 0.854),
        (-2, 0.9443),
        (-1, 0.9914),
        (0, 1.011),
        (1, 1.007),
        (2, 0.9707),
        (3, 0.8913),
        (4, 0.7733),
        (5, 0.6398),
        (6, 0.5143),
        (7, 0.4081),
        (8, 0.3226),
        (9, 0.255),
        (10, 0.2017),
        (11, 0.1597),
        (12, 0.1266),
        (13, 0.1004),
        (14, 0.07958),
        (15, 0.06299),
        (16, 0.04965),
        (17, 0.03872),
        (18, 0.02946),
        (19, 0.0213),
        (20, 0.01414),
        (21, 0.008478),
        (22, 0.004668),
        (23, 0.002442),
        (24, 0.001246),
        (25, 0.0006293),
        (26, 0.0003164),
    ),
    "We": (
        (-10, 0.06252),
        (-9, 0.09893),
        (-8, 0.156),
        (-7, 0.2435),
        (-6, 0.3715),
        (-5, 0.5394),
        (-4, 0.7198),
        (-3, 0.8635),
        (-2, 0.9389),
        (-1, 0.9423),
        (0, 0.8798),
        (1, 0.7683),
        (2, 0.6372),
        (3, 0.5127),
        (4, 0.407),
        (5, 0.3218),
        (6, 0.2543),
        (7, 0.2012),
        (8, 0.1594),
        (9, 0.1263),
        (10, 0.1002),
        (11, 0.07954),
        (12, 0.06314),
        (13, 0.05011),
        (14, 0.03975),
        (15, 0.03147),
        (16, 0.02481),
        (17, 0.01935),
        (18, 0.01473),
        (19, 0.01065),
        (20, 0.007071),
        (21, 0.004239),
        (22, 0.002334),
        (23, 0.001221),
        (24, 0.0006232),
        (25, 0.0003147),
        (26, 0.0001582),
    ),
    "Wf": (
        (-17, 0.02407),
        (-16, 0.03803),
        (-15, 0.06021),
        (-14, 0.09619),
        (-13, 0.1575),
        (-12, 0.2675),
        (-11, 0.4537),
        (-10, 0.6951),
        (-9, 0.9),
        (-8, 1.004),
        (-7, 0.9928),
        (-6, 0.8501),
        (-5, 0.6149),
        (-4, 0.3884),
        (-3, 0.2225),
        (-2, 0.1157),
        (-1, 0.05434),
        (0, 0.02352),
        (1, 0.009705),
        (2, 0.003916),
        (3, 0.001566),
    ),
    "Wh": (
        (-1, 0.01586),
        (0, 0.02514),
        (1, 0.03985),
        (2, 0.06314),
        (3, 0.09992),
        (4, 0.1576),
        (5, 0.2461),
        (6, 0.3754),
        (7, 0.545),
        (8, 0.7272),
        (9, 0.8731),
        (10, 0.9514),
        (11, 0.9576),
        (12, 0.8958),
        (13, 0.782),
        (14, 0.6471),
        (15, 0.5192),
        (16, 0.4111),
        (17, 0.3244),
        (18, 0.256),
        (19, 0.2024),
        (20, 0.1602),
        (21, 0.127),
        (22, 0.1007),
        (23, 0.07988),
        (24, 0.06338),
        (25, 0.05026),
        (26, 0.0398),
        (27, 0.03137),
        (28, 0.02447),
        (29, 0.01862),
        (30, 0.01346),
        (31, 0.00894),
        (32, 0.005359),
        (33, 0.00295),
        (34, 0.001544),
        (35, 0.0007878),
        (36, 0.0003978),
    ),
    "Wj": (
        (-10, 0.03099),
        (-9, 0.04897),
        (-8, 0.07703),
        (-7, 0.1199),
        (-6, 0.1821),
        (-5, 0.263),
        (-4, 0.3489),
        (-3, 0.4176),
        (-2, 0.4585),
        (-1, 0.4776),
        (0, 0.4844),
        (1, 0.4851),
        (2, 0.4832),
        (3, 0.4819),
        (4, 0.4889),
        (5, 0.5246),
        (6, 0.6251),
        (7, 0.7948),
        (8, 0.947),
        (9, 1.016),
        (10, 1.03),
        (11, 1.026),
        (12, 1.019),
        (13, 1.012),
        (14, 1.006),
        (15, 1.0),
        (16, 0.9911),
        (17, 0.972),
        (18, 0.9304),
        (19, 0.8465),
        (20, 0.7075),
        (21, 0.5338),
        (22, 0.37),
        (23, 0.2437),
        (24, 0.1565),
        (25, 0.09951),
        (26, 0.06297),
    ),
    "Wk": (
        (-10, 0.03121),
        (-9, 0.04931),
        (-8, 0.07756),
        (-7, 0.1207),
        (-6, 0.1832),
        (-5, 0.2644),
        (-4, 0.3504),
        (-3, 0.4188),
        (-2, 0.4588),
        (-1, 0.4767),
        (0, 0.4825),
        (1, 0.4846),
        (2, 0.4935),
        (3, 0.5308),
        (4, 0.6335),
        (5, 0.8071),
        (6, 0.9648),
        (7, 1.039),
        (8, 1.054),
        (9, 1.037),
        (10, 0.9884),
        (11, 0.8989),
        (12, 0.7743),
        (13, 0.6373),
        (14, 0.5103),
        (15, 0.4031),
        (16, 0.316),
        (17, 0.2451),
        (18, 0.1857),
        (19, 0.1339),
        (20, 0.08873),
        (21, 0.05311),
        (22, 0.02922),
        (23, 0.01528),
        (24, 0.007795),
        (25, 0.003935),
        (26, 0.001978),
    ),
    "Wm": (
        (-10, 0.01584),
        (-9, 0.0251),
        (-8, 0.03976),
        (-7, 0.06293),
        (-6, 0.09941),
        (-5, 0.1563),
        (-4, 0.243),
        (-3, 0.3684),
        (-2, 0.5304),
        (-1, 0.7003),
        (0, 0.8329),
        (1, 0.9071),
        (2, 0.9342),
        (3, 0.9319),
        (4, 0.9101),
        (5, 0.8721),
        (6, 0.8184),
        (7, 0.7498),
        (8, 0.6692),
        (9, 0.5819),
        (10, 0.4941),
        (11, 0.4114),
        (12, 0.3375),
        (13, 0.2738),
        (14, 0.2203),
        (15, 0.176),
        (16, 0.1396),
        (17, 0.1093),
        (18, 0.08336),
        (19, 0.06036),
        (20, 0.04013),
        (21, 0.02407),
        (22, 0.01326),
        (23, 0.006937),
        (24, 0.003541),
        (25, 0.001788),
        (26, 0.000899),
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
    (0.26, 1.00),
    (0.26, 0.21),
    (0.12, 0.11),
    (0.26, 0.21),
    (0.26, 1.00),
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
    -0.000005710,
    0.000020010,
    0.001373900,
    0.014541920,
    0.025152310,
    -0.014242050,
    -0.044262840,
    -0.008888510,
    0.017715720,
    0.010216420,
    0.002030740,
    0.000055980,
)
ISO2631_5_ANNEX_D_A: tuple[float, ...] = (
    1.000000000,
    -3.323217600,
    4.256126150,
    -1.980417270,
    -1.488735470,
    3.329511290,
    -2.949072140,
    1.653403410,
    -0.635677800,
    0.167519420,
    -0.028076980,
    0.002348730,
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
