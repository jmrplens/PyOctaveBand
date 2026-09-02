#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Sound power and sound intensity: what a source emits and how well we know it.

The determination standards and their uncertainty budgets: the precision
free-field method of ISO 3745 with its reproducibility and environmental
corrections, the field-indicator scheme of ISO 9614-3 for scanning
intensity, and the IEC 61043 instrument specification - Table 2 residual
pressure-intensity index and the phase-mismatch error the index implies.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# ISO 3745:2012 precision sound power (anechoic/hemi-anechoic). The Clause 10.5
# EXAMPLE combines sigma_omc = 2,0 dB and sigma_R0 = 0,5 dB at k = 2 to the
# expanded uncertainty U = 4,1 dB. The K1 background correction floors at
# 1,26 dB (>= 6 dB signal-to-noise edge bands). The meteorological correction
# C1 at the 23 C, ps0 reference is 5*lg(296/314) = -0,128 dB. Mirrors
# tests/emission/test_sound_power_precision.py.
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
# ISO 9614-1:1993 (UNE-EN ISO 9614-1:2010, the Spanish official version of
# EN ISO 9614-1:2009, which adopts the 1993 ISO text unchanged). Four printed
# tables, transcribed cell for cell from the rendered pages; the page offset of
# this edition is zero, so PDF page N carries printed folio N.
#
# The three graded tables share one shape and one asymmetry: grades 1 and 2 are
# tabulated band by band with the A-weighted cell blank, and grade 3 carries
# only an A-weighted value with every band cell blank. A blank below is None,
# and it is blank in the print, not zero and not "not applicable".
# ---------------------------------------------------------------------------

# Table 1, "Factor de error de desviación, K" (printed p. 11), in dB, by grade.
ISO9614_1_TABLE_1_K: dict[str, float] = {
    "precision": 10.0,
    "engineering": 10.0,
    "survey": 7.0,
}

# Rows of Table 2 and Table B.2, which share their two frequency columns:
# (octave centre range in Hz, one-third-octave centre range in Hz). The 6 300 Hz
# row has no octave counterpart, and the last row of each table is A-weighted
# and has no frequency column at all.
ISO9614_1_BAND_ROWS: list[tuple[tuple[float, float] | None, tuple[float, float]]] = [
    ((63.0, 125.0), (50.0, 160.0)),
    ((250.0, 500.0), (200.0, 630.0)),
    ((1000.0, 4000.0), (800.0, 5000.0)),
    (None, (6300.0, 6300.0)),
]

# Table 2, "Incertidumbre en la determinación de los niveles de potencia
# sonora" (printed p. 13): the standard deviation s in dB, as
# (grade 1, grade 2, grade 3) per row of ISO9614_1_BAND_ROWS. Footnote 1: the
# true level lies within +/- 2s of the measured one with 95 % confidence.
# Footnote 2 fixes the A-weighted range at 63 Hz to 4 kHz or 50 Hz to 6,3 kHz;
# footnote 3 calls the grade-3 value tentative.
ISO9614_1_TABLE_2_S: list[tuple[float | None, float | None, float | None]] = [
    (2.0, 3.0, None),
    (1.5, 2.0, None),
    (1.0, 1.5, None),
    (2.0, 2.5, None),
]
ISO9614_1_TABLE_2_S_A_WEIGHTED: tuple[float | None, float | None, float | None] = (
    None,
    None,
    4.0,
)

# Table B.1, "Factor de error Delta" (printed p. 25): one row for all bands and
# one A-weighted row, each as (grade 1, grade 2, grade 3).
ISO9614_1_TABLE_B1_ALL_BANDS: tuple[float | None, float | None, float | None] = (
    0.20,
    0.29,
    None,
)
ISO9614_1_TABLE_B1_A_WEIGHTED: tuple[float | None, float | None, float | None] = (
    None,
    None,
    0.60,
)

# Table B.2, "Valores para el factor C" (printed p. 25): the criterion-2 factor
# C, as (grade 1, grade 2, grade 3) per row of ISO9614_1_BAND_ROWS, then the
# A-weighted row whose footnote reads "63 Hz a 4 kHz ó 50 Hz a 6,3 kHz".
ISO9614_1_TABLE_B2_C: list[tuple[float | None, float | None, float | None]] = [
    (19.0, 11.0, None),
    (29.0, 19.0, None),
    (57.0, 29.0, None),
    (19.0, 14.0, None),
]
ISO9614_1_TABLE_B2_C_A_WEIGHTED: tuple[float | None, float | None, float | None] = (
    None,
    None,
    8.0,
)

# Table B.3, "Acciones a tomar para incrementar el grado de precisión de la
# determinación" (printed p. 26): the criterion of each row and the action code
# or codes Figure B.1 routes it to. The second row prints "a o b", two
# alternatives under one criterion.
ISO9614_1_TABLE_B3: list[tuple[str, tuple[str, ...]]] = [
    ("F1 > 0,6", ("e",)),
    ("F2 > Ld o (F3 - F2) > 3 dB", ("a", "b")),
    ("No se satisface el criterio 2 y 1 dB <= (F3 - F2) <= 3 dB", ("c",)),
    (
        "No se satisface el criterio 2, (F3 - F2) <= 1 dB y el procedimiento "
        "del apartado 8.3.2 o bien falla, o bien no se selecciona",
        ("d",),
    ),
]

# ---------------------------------------------------------------------------
# ISO 5136:2003 (as reproduced in BS EN ISO 5136:2009), sound power radiated
# into a duct by fans, in-duct method. Every value below was read from the
# rendered page of the PDF (the printed folio is the PDF page index minus 10);
# the decimal comma of the print is written as a point.
# ---------------------------------------------------------------------------
#: The 27 nominal one-third-octave bands of Annex C Table C.1 (PDF page 44,
#: printed p. 34), j = 1 (50 Hz) to j_max = 27 (20 kHz), in hertz.
ISO5136_BANDS: tuple[int, ...] = (
    50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000,
    1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000, 12500,
    16000, 20000,
)  # fmt: skip
#: Table C.1, the A-weighting C_j "according to IEC 60651", dB, one per band
#: of ISO5136_BANDS.
ISO5136_TABLE_C1: tuple[float, ...] = (
    -30.2, -26.2, -22.5, -19.1, -16.1, -13.4, -10.9, -8.6, -6.6, -4.8, -3.2,
    -1.9, -0.8, 0.0, 0.6, 1.0, 1.2, 1.3, 1.2, 1.0, 0.5, -0.1, -1.1, -2.5,
    -4.3, -6.6, -9.3,
)  # fmt: skip
#: Table 2 (PDF page 17, printed p. 7), "Values of the standard deviation of
#: reproducibility for the sampling tube": the printed rows, two of which are
#: ranges ("80 to 100", "125 to 4 000"), as (lowest band, highest band,
#: sigma_R in dB).
ISO5136_TABLE_2_SIGMA_R: tuple[tuple[int, int, float], ...] = (
    (50, 50, 3.5),
    (63, 63, 3.0),
    (80, 100, 2.5),
    (125, 4000, 2.0),
    (5000, 5000, 2.5),
    (6300, 6300, 3.0),
    (8000, 8000, 3.5),
    (10000, 10000, 4.0),
)
#: Table 3 (PDF page 18, printed p. 8), "Extrapolated values" of sigma_R above
#: 10 kHz, dB, which clause 4 suggests for bands that are "not considered part
#: of this International Standard".
ISO5136_TABLE_3_SIGMA_R: tuple[tuple[int, float], ...] = (
    (12500, 4.5),
    (16000, 5.0),
    (20000, 5.5),
)
#: Annex D, Eqs (D.1) to (D.3) (PDF page 45, printed p. 35): for d = 0,5 m
#: at 1 000 Hz, C3,4 = (1,85 + 0,038 U) dB, which the print evaluates to
#: "≈ 2,4 dB" at U = 15 m/s (outlet duct) and "≈ 1,3 dB" at U = -15 m/s
#: (inlet duct); the two products are exactly 2,42 and 1,28.
ISO5136_ANNEX_D_A0 = 1.85  # dB
ISO5136_ANNEX_D_A1 = 0.038  # dB s/m
ISO5136_ANNEX_D_DIAMETER = 0.5  # m
ISO5136_ANNEX_D_FREQUENCY = 1000.0  # Hz
ISO5136_ANNEX_D_OUTLET: tuple[float, float] = (15.0, 2.4)  # (U, printed C3,4)
ISO5136_ANNEX_D_INLET: tuple[float, float] = (-15.0, 1.3)  # (U, printed C3,4)
#: Table D.1 (PDF pages 45 and 46, printed pp. 35 and 36), "Value of
#: correction C3,4 in decibels for d = 0,5 m and different flow velocities U":
#: the six columns are U = 5, -5, 15, -15, 30 and -30 m/s, the 27 rows are the
#: bands of ISO5136_BANDS. Transcribed as printed, to the printed 0,1 dB; the
#: page prints "2", "0" and "16" for cells the text layer carries as "2,0",
#: "0,0" and "16,0". The two cells of the worked example, 1 000 Hz at
#: +/-15 m/s, are the ones the print frames in bold.
ISO5136_TABLE_D1_VELOCITIES: tuple[float, ...] = (5.0, -5.0, 15.0, -15.0, 30.0, -30.0)
ISO5136_TABLE_D1: tuple[tuple[float, ...], ...] = (
    (0.1, -0.2, 0.4, -0.5, 0.8, -0.9),  # 50 Hz
    (0.1, -0.2, 0.4, -0.5, 0.8, -0.9),  # 63 Hz
    (0.1, -0.2, 0.4, -0.5, 0.8, -0.9),  # 80 Hz
    (0.1, -0.2, 0.4, -0.5, 0.8, -0.9),  # 100 Hz
    (0.1, -0.2, 0.4, -0.5, 0.8, -0.9),  # 125 Hz
    (0.1, -0.2, 0.4, -0.5, 0.8, -0.9),  # 160 Hz
    (0.1, -0.2, 0.4, -0.5, 0.8, -0.9),  # 200 Hz
    (0.1, -0.2, 0.4, -0.5, 0.8, -0.9),  # 250 Hz
    (-0.5, -0.8, -0.2, -1.1, 0.2, -1.5),  # 315 Hz
    (-0.3, -0.6, 0.0, -0.9, 0.5, -1.3),  # 400 Hz
    (-0.2, -0.5, 0.2, -0.8, 0.6, -1.2),  # 500 Hz
    (0.2, -0.1, 0.6, -0.4, 1.1, -0.9),  # 630 Hz
    (1.2, 0.9, 1.6, 0.5, 2.1, 0.0),  # 800 Hz
    (2.0, 1.7, 2.4, 1.3, 3.0, 0.7),  # 1 000 Hz
    (2.8, 2.4, 3.3, 2.0, 4.0, 1.4),  # 1 250 Hz
    (3.4, 2.9, 4.0, 2.4, 4.9, 1.7),  # 1 600 Hz
    (4.0, 3.3, 4.7, 2.7, 5.8, 1.8),  # 2 000 Hz
    (4.5, 3.7, 5.4, 2.9, 6.9, 1.9),  # 2 500 Hz
    (5.2, 4.1, 6.5, 3.1, 8.4, 2.1),  # 3 150 Hz
    (6.2, 4.9, 7.7, 3.8, 10.2, 2.8),  # 4 000 Hz
    (6.8, 5.3, 8.7, 4.2, 11.8, 3.2),  # 5 000 Hz
    (7.9, 6.1, 10.1, 4.7, 13.7, 3.8),  # 6 300 Hz
    (9.3, 7.0, 12.2, 5.6, 16.2, 4.9),  # 8 000 Hz
    (10.5, 7.6, 13.9, 5.9, 17.9, 5.6),  # 10 000 Hz
    (11.6, 8.0, 16.0, 6.7, 19.5, 6.4),  # 12 500 Hz
    (13.1, 8.7, 18.2, 7.5, 21.2, 7.3),  # 16 000 Hz
    (14.8, 9.4, 20.2, 8.4, 23.6, 7.9),  # 20 000 Hz
)
#: Half of the last printed place of Table D.1: a cell reproduces when the
#: polynomial lands within it of the printed value. The comparison has to be a
#: tolerance rather than a rounding rule, because the table rounds to 0,1 dB
#: and the polynomial lands wherever it lands: the widest gap over the 162
#: cells is the 4 000 Hz row at U = +5 m/s, where 6,151 dB is printed as 6,2,
#: which is 0,049 dB away and only just inside the budget.
ISO5136_TABLE_D1_TOLERANCE_DB = 0.05
#: Clause 1.1 (PDF page 11, printed p. 1): the maximum mean flow velocity at
#: the microphone head per shield, m/s, and the test-duct diameter range, m.
ISO5136_MAX_VELOCITY: dict[str, float] = {
    "foam-ball": 15.0,
    "nose-cone": 20.0,
    "sampling-tube": 40.0,
}
ISO5136_DUCT_DIAMETER_RANGE: tuple[float, float] = (0.15, 2.0)
#: Clause 5.3.3.4 NOTE and the footnote of Tables A.1 to A.6 (PDF pages 28
#: and 35 to 40): the sampling-tube coefficients extend to |U| <= 60 m/s for
#: information only.
ISO5136_SAMPLING_TUBE_INFORMATIVE_VELOCITY = 60.0
#: Clause 5.3.4.3 (PDF page 29, printed p. 19): Eq. (8) and its "under normal
#: conditions, c = 340 m/s".
ISO5136_C_NORMAL = 340.0
#: Clause 8.2 (PDF pages 33 and 34, printed pp. 23 and 24): S0 = 1 m^2 and
#: (rho c)_0 = 400 N s/m^3 of Eq. (12).
ISO5136_S0 = 1.0
ISO5136_RHO_C_0 = 400.0
#: Clause 9.2 (PDF page 34, printed p. 24): the expanded uncertainty at 95 %
#: coverage is twice the standard deviation of reproducibility of clause 4.
ISO5136_COVERAGE_FACTOR = 2.0
