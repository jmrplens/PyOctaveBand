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
