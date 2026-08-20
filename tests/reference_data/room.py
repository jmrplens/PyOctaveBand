#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Rooms: reverberation, the absorption they contain and the noise they allow.

What a finished space does to sound already inside it. Everest's worked
example gives a full Sabine reverberation calculation from surfaces and
absorption coefficients; EN 12354-6 Annex E gives the equivalent
absorption area of a furnished room, bare and with objects; ANSI S12.2
gives the NC and RC criterion curves a room's background noise is rated
against. Three standards, one room: its surfaces, its contents, its
acceptable background.
"""

from __future__ import annotations

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
    (6.55, 0.04),  # side wall
    (6.55, 0.04),  # glass facade
]
EN12354_6_A_BARE = 2.26  # equivalent absorption area, bare room (m2)
EN12354_6_T_BARE = 2.1  # reverberation time, bare room (s)
EN12354_6_A_OBJECTS = 5.03  # equivalent absorption area, with objects (m2)

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
_FT2 = 0.3048**2  # square foot -> m2
_FT3 = 0.3048**3  # cubic foot -> m3
EVEREST_EX1_VOLUME = 3728.0 * _FT3  # 105.565 m3
EVEREST_EX1_FLOOR_AREA = 373.0 * _FT2  # concrete floor, 34.653 m2
EVEREST_EX1_SHELL_AREA = 1159.0 * _FT2  # gypsum walls+ceiling, 107.675 m2
EVEREST_EX1_BANDS = [125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0]
EVEREST_EX1_FLOOR_ALPHA = [0.01, 0.01, 0.015, 0.02, 0.02, 0.02]  # concrete
EVEREST_EX1_SHELL_ALPHA = [0.29, 0.10, 0.05, 0.04, 0.07, 0.09]  # gypsum board
EVEREST_EX1_RT = [0.54, 1.53, 2.87, 3.39, 2.06, 1.63]  # printed RT60, s
