#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Doc 29 5th ed. Vol 3 Part 2 reference-workbook rows (ECAC).

Flight-performance oracle for Volume 2 Appendix B: the procedural steps and
aircraft data of the two hypothetical reference aircraft (JETW turbofan, PROP
turboprop), the 9 arrival and 17 departure reference cases built from them, and
the fixed-point profiles those cases must produce. Factual reference values
transcribed with attribution; the workbook itself is not redistributed.

Units are the standard's throughout: feet, knots, pounds, degrees Celsius,
inches of mercury. ``None`` marks a cell the workbook leaves blank (printed as
``-`` or ``n/a``), which is a quantity that step or coefficient row does not
define, never a zero.

Identifiers are transcribed verbatim apart from stripped padding whitespace, so
the workbook's own inconsistencies survive: thrust rating ``MaxTakeoff`` in
C-2/C-6.1 against ``MaxTakeOff`` in C-3 and in the JETW ICAO_A_ESF steps, and
flap identifiers stored as numbers in some sheets and as text in others (all
normalised here to text, so ``5`` becomes ``'5'``).

Two step columns of C-6.2 carry something other than their heading, verified
against the D-1 point spacing: a Level-Idle step keeps its ground distance in
feet in the ``Descent Angle (deg)`` column (16500 ft and 5000 ft for
Level_Idle, reproduced exactly by the case 2C point spacing), and a Decelerate
step's ``Start Thrust (%)`` is a percentage of maximum sea-level static thrust
(40 % of 25000 lb = the 10000 lb of D-1 point 11).
"""

PRINTED_TOLERANCE = 0.05
"""Half of the last printed digit of every D-1/D-2 column, in that column's unit.

All four result columns are stored and printed to one decimal (Excel format
``0.0``), so a value read back sits at most 0.05 ft, 0.05 kt or 0.05 lb from the
number the reference implementation computed. Nothing in the workbook justifies
a looser figure, and no exact comparison can be met.

The bound is closed: a true value ending in exactly .x5 is 0.05 from whichever
neighbour the workbook rounded it to, so compare with ``<=``. Compare absolutely
and per column, never relatively: 104 of the 1256 numbers are exactly zero
(altitudes on the ground, distances at touchdown, airspeeds at rest), and at the
other end 0.05 ft of the 400814.3 ft of cases 56 and 68 is already 1.2e-7.
"""

AIRCRAFT: dict[str, tuple[str, int, float, float, float, float]] = {
    # C-1: engine type, engines, MTOW (lb), max landing weight (lb),
    # max landing distance (ft), max sea-level static thrust (lb).
    "JETW": ("Jet", 2, 165347.0, 159222.0, 4921.0, 25000.0),
    "PROP": ("Turboprop", 2, 165347.0, 159222.0, 4921.0, 16500.0),
}

JET_COEFFICIENTS: dict[tuple[str, str], tuple[float, float, float, float, float]] = {
    # C-2: E (lb), F (lb/kt), Ga (lb/ft), Gb (lb/ft2), H (lb/degC); K1..K4 are
    # 'n/a' for both aircraft, so the EPR and N1 thrust forms are never exercised.
    ("JETW", "MaxClimb"): (16000.0, -4.0, 0.4, -1e-05, 0.0),
    ("JETW", "IdleApproach"): (1100.0, -6.5, 0.17, -1e-05, 0.0),
    ("JETW", "MaxTakeoff"): (25000.0, -25.0, 0.3, 1e-05, 0.0),
}

PROP_COEFFICIENTS: dict[tuple[str, str], tuple[float, float]] = {
    # C-3: propeller efficiency, installed net propulsive power (hp).
    ("PROP", "MaxClimb"): (0.85, 7800.0),
    ("PROP", "MaxTakeOff"): (0.85, 9500.0),
}

AERODYNAMIC_COEFFICIENTS: dict[
    tuple[str, str, str], tuple[float | None, float | None, float]
] = {
    # C-4: B (ft/lb), C for arrivals or D for departures (kt/sqrt(lb)), R.
    # B and C/D are undefined for a flap setting no take-off or landing step uses.
    ("JETW", "A", "5"): (None, None, 0.07),
    ("JETW", "A", "15"): (None, None, 0.075),
    ("JETW", "A", "25"): (None, 0.375, 0.1),
    ("JETW", "A", "30"): (None, 0.35, 0.12),
    ("JETW", "A", "ZERO"): (None, None, 0.055),
    ("JETW", "D", "1"): (None, None, 0.06),
    ("JETW", "D", "5"): (0.0075, 0.4, 0.07),
    ("JETW", "D", "ZERO"): (None, None, 0.055),
    ("PROP", "D", "17"): (0.0091, 0.365, 0.11),
    ("PROP", "D", "ZERO"): (None, None, 0.08),
}

DEFAULT_WEIGHTS: dict[tuple[str, str, str], float] = {
    # C-5: aircraft, operation (A/D), stage length -> weight (lb).
    ("JETF", "A", "1"): 143300.0,
    ("JETF", "D", "1"): 165347.0,
    ("JETW", "A", "1"): 143300.0,
    ("JETW", "D", "1"): 165347.0,
    ("PROP", "A", "1"): 143300.0,
    ("PROP", "D", "1"): 165347.0,
}

DepartureStep = tuple[
    int,
    str,
    str,
    str,
    float | None,
    float | None,
    float | None,
    float | None,
    float | None,
]
"""C-6.1 row: step number, step type, thrust rating, flap identifier, end-point
altitude (ft), rate of climb (ft/min), end-point CAS (kt), acceleration
percentage (%), track distance (ft)."""

# fmt: off
# One sheet row on one line: the formatter would explode the long rows and
# a transcription is only auditable while it still looks like the table.
DEPARTURE_STEPS: dict[tuple[str, str], tuple[DepartureStep, ...]] = {
    ("JETW", "ICAO_B"): (
        (1, "Takeoff", "MaxTakeoff", "5", None, None, None, None, None),
        (2, "Climb", "MaxTakeoff", "5", 1000.0, None, None, None, None),
        (3, "Accelerate", "MaxClimb", "5", None, 984.3, 210.6, None, None),
        (4, "Accelerate", "MaxClimb", "1", None, 918.6, 226.8, None, None),
        (5, "Climb", "MaxClimb", "ZERO", 3000.0, None, None, None, None),
        (6, "Accelerate", "MaxClimb", "ZERO", None, 869.4, 251.1, None, None),
        (7, "Climb", "MaxClimb", "ZERO", 5500.0, None, None, None, None),
        (8, "Climb", "MaxClimb", "ZERO", 7500.0, None, None, None, None),
        (9, "Climb", "MaxClimb", "ZERO", 10000.0, None, None, None, None),
    ),
    ("JETW", "ICAO_A"): (
        (1, "Takeoff", "MaxTakeoff", "5", None, None, None, None, None),
        (2, "Climb", "MaxTakeoff", "5", 1500.0, None, None, None, None),
        (3, "Climb", "MaxClimb", "5", 3000.0, None, None, None, None),
        (4, "Accelerate", "MaxClimb", "5", None, 984.3, 210.6, None, None),
        (5, "Accelerate", "MaxClimb", "1", None, 918.6, 226.8, None, None),
        (6, "Accelerate", "MaxClimb", "ZERO", None, 869.4, 251.1, None, None),
        (7, "Climb", "MaxClimb", "ZERO", 5500.0, None, None, None, None),
        (8, "Climb", "MaxClimb", "ZERO", 7500.0, None, None, None, None),
        (9, "Climb", "MaxClimb", "ZERO", 10000.0, None, None, None, None),
    ),
    ("JETW", "ICAO_A_ESF"): (
        (1, "Takeoff", "MaxTakeOff", "5", None, None, None, None, None),
        (2, "Climb", "MaxTakeOff", "5", 1500.0, None, None, None, None),
        (3, "Climb", "MaxClimb", "5", 3000.0, None, None, None, None),
        (4, "Accelerate", "MaxClimb", "5", None, None, 210.6, 53.7986, None),
        (5, "Accelerate", "MaxClimb", "1", None, None, 226.8, 65.4993, None),
        (6, "Accelerate", "MaxClimb", "ZERO", None, None, 251.1, 71.118, None),
        (7, "Climb", "MaxClimb", "ZERO", 5500.0, None, None, None, None),
        (8, "Climb", "MaxClimb", "ZERO", 7500.0, None, None, None, None),
        (9, "Climb", "MaxClimb", "ZERO", 10000.0, None, None, None, None),
    ),
    ("JETW", "ICAO_A_Deep_cutback"): (
        (1, "Takeoff", "MaxTakeoff", "5", None, None, None, None, None),
        (2, "Climb", "MaxTakeoff", "5", 800.0, None, None, None, None),
        (3, "Climb", "MinimumThrust", "5", 2000.0, None, None, None, None),
        (4, "Climb", "MaxClimb", "5", 3000.0, None, None, None, None),
        (5, "Accelerate", "MaxClimb", "5", None, 984.3, 210.6, None, None),
        (6, "Accelerate", "MaxClimb", "1", None, 918.6, 226.8, None, None),
        (7, "Accelerate", "MaxClimb", "ZERO", None, 869.4, 251.1, None, None),
        (8, "Climb", "MaxClimb", "ZERO", 5500.0, None, None, None, None),
        (9, "Climb", "MaxClimb", "ZERO", 7500.0, None, None, None, None),
        (10, "Climb", "MaxClimb", "ZERO", 10000.0, None, None, None, None),
    ),
    ("JETW", "ICAO_A_Level_constant_speed"): (
        (1, "Takeoff", "MaxTakeoff", "5", None, None, None, None, None),
        (2, "Climb", "MaxTakeoff", "5", 1500.0, None, None, None, None),
        (3, "Climb", "MaxClimb", "5", 3000.0, None, None, None, None),
        (4, "Level", "AdaptedThrust", "5", None, None, None, None, 10000.0),
        (5, "Accelerate", "MaxClimb", "5", None, 984.3, 210.6, None, None),
        (6, "Accelerate", "MaxClimb", "1", None, 918.6, 226.8, None, None),
        (7, "Accelerate", "MaxClimb", "ZERO", None, 869.4, 251.1, None, None),
        (8, "Climb", "MaxClimb", "ZERO", 5500.0, None, None, None, None),
        (9, "Climb", "MaxClimb", "ZERO", 7500.0, None, None, None, None),
        (10, "Climb", "MaxClimb", "ZERO", 10000.0, None, None, None, None),
    ),
    ("JETW", "ICAO_A_Level_ROC"): (
        (1, "Takeoff", "MaxTakeoff", "5", None, None, None, None, None),
        (2, "Climb", "MaxTakeoff", "5", 1500.0, None, None, None, None),
        (3, "Climb", "MaxClimb", "5", 3000.0, None, None, None, None),
        (4, "Level-Accelerate", "MaxClimb", "5", None, 0.0, 210.6, None, None),
        (5, "Level-Accelerate", "MaxClimb", "1", None, 0.0, 226.8, None, None),
        (6, "Accelerate", "MaxClimb", "ZERO", None, 869.4, 251.1, None, None),
        (7, "Climb", "MaxClimb", "ZERO", 5500.0, None, None, None, None),
        (8, "Climb", "MaxClimb", "ZERO", 7500.0, None, None, None, None),
        (9, "Climb", "MaxClimb", "ZERO", 10000.0, None, None, None, None),
    ),
    ("JETW", "ICAO_A_Level_ESF"): (
        (1, "Takeoff", "MaxTakeoff", "5", None, None, None, None, None),
        (2, "Climb", "MaxTakeoff", "5", 1500.0, None, None, None, None),
        (3, "Climb", "MaxClimb", "5", 3000.0, None, None, None, None),
        (4, "Level-Accelerate", "MaxClimb", "5", None, None, 210.6, 100.0, None),
        (5, "Level-Accelerate", "MaxClimb", "1", None, None, 226.8, 100.0, None),
        (6, "Accelerate", "MaxClimb", "ZERO", None, None, 251.1, 71.118, None),
        (7, "Climb", "MaxClimb", "ZERO", 5500.0, None, None, None, None),
        (8, "Climb", "MaxClimb", "ZERO", 7500.0, None, None, None, None),
        (9, "Climb", "MaxClimb", "ZERO", 10000.0, None, None, None, None),
    ),
    ("PROP", "ICAO_A"): (
        (1, "Takeoff", "MaxTakeoff", "17", None, None, None, None, None),
        (2, "Climb", "MaxTakeoff", "17", 1500.0, None, None, None, None),
        (3, "Climb", "MaxClimb", "17", 3000.0, None, None, None, None),
        (4, "Accelerate", "MaxClimb", "17", None, 1000.7, 151.2, None, None),
        (5, "Accelerate", "MaxClimb", "ZERO", None, 1000.7, 162.0, None, None),
        (6, "Accelerate", "MaxClimb", "ZERO", None, 702.1, 199.8, None, None),
        (7, "Climb", "MaxClimb", "ZERO", 5500.0, None, None, None, None),
        (8, "Climb", "MaxClimb", "ZERO", 7500.0, None, None, None, None),
        (9, "Climb", "MaxClimb", "ZERO", 10000.0, None, None, None, None),
    ),
    ("PROP", "ICAO_A_ESF"): (
        (1, "Takeoff", "MaxTakeoff", "17", None, None, None, None, None),
        (2, "Climb", "MaxTakeoff", "17", 1500.0, None, None, None, None),
        (3, "Climb", "MaxClimb", "17", 3000.0, None, None, None, None),
        (4, "Accelerate", "MaxClimb", "17", None, None, 151.2, 30.0, None),
        (5, "Accelerate", "MaxClimb", "ZERO", None, None, 162.0, 26.0, None),
        (6, "Accelerate", "MaxClimb", "ZERO", None, None, 199.8, 30.0, None),
        (7, "Climb", "MaxClimb", "ZERO", 5500.0, None, None, None, None),
        (8, "Climb", "MaxClimb", "ZERO", 7500.0, None, None, None, None),
        (9, "Climb", "MaxClimb", "ZERO", 10000.0, None, None, None, None),
    ),
}

ApproachStep = tuple[
    int, str, str, float | None, float | None, float | None,
    float | None, float | None, float | None,
]
"""C-6.2 row: step number, step type, flap identifier, start altitude (ft),
start CAS (kt), descent angle (deg) -- or the ground distance (ft) of a
Level-Idle step -- touchdown roll (ft), track distance (ft), start thrust as a
percentage of maximum sea-level static thrust."""

APPROACH_STEPS: dict[tuple[str, str], tuple[ApproachStep, ...]] = {
    ("JETW", "Descend"): (
        (1, "Descend", "ZERO", 6000.0, 250.0, 2.8, None, None, None),
        (2, "Descend", "5", 3000.0, 180.0, 3.0, None, None, None),
        (3, "Descend", "25", 1500.0, 150.0, 3.0, None, None, None),
        (4, "Descend", "30", 1000.0, 135.0, 3.0, None, None, None),
        (5, "Descend", "30", 49.9, 135.0, 3.0, None, None, None),
        (6, "Land", "30", None, None, None, 304.1338582677165, None, None),
        (7, "Decelerate", "-NONE-", None, 129.58963282937364, None, None, 3937.007874015748, 40.0),
        (8, "Decelerate", "-NONE-", None, 26.997840172786177, None, None, 0.0, 10.0),
    ),
    ("JETW", "Descend2"): (
        (1, "Descend", "ZERO", 6000.0, 250.0, 2.8, None, None, None),
        (2, "Descend", "ZERO", 5000.0, 230.0, 3.0, None, None, None),
        (3, "Descend", "5", 3000.0, 180.0, 3.0, None, None, None),
        (4, "Descend", "25", 1500.0, 150.0, 3.0, None, None, None),
        (5, "Descend", "30", 1000.0, 135.0, 3.0, None, None, None),
        (6, "Descend", "30", 49.9, 135.0, 3.0, None, None, None),
        (7, "Land", "30", None, None, None, 304.1338582677165, None, None),
        (8, "Decelerate", "-NONE-", None, 129.58963282937364, None, None, 3937.007874015748, 40.0),
        (9, "Decelerate", "-NONE-", None, 26.997840172786177, None, None, 0.0, 10.0),
    ),
    ("JETW", "Level_Decel"): (
        (1, "Descend", "ZERO", 6000.0, 250.0, 2.8, None, None, None),
        (2, "Level-Decel", "ZERO", 3000.0, 250.0, None, None, 26246.719160104985, None),
        (3, "Level-Decel", "ZERO", 3000.0, 188.98488120950324, None, None, 3854.9868766404197, None),
        (4, "Descend-Decel", "15", 3000.0, 178.18574514038875, 3.0, None, None, None),
        (5, "Descend-Decel", "25", 2460.6299212598424, 167.38660907127428, 3.0, None, None, None),
        (6, "Descend-Decel", "25", 1968.503937007874, 140.3887688984881, 3.0, None, None, None),
        (7, "Descend", "30", 1640.4199475065616, 134.98920086393088, 3.0, None, None, None),
        (8, "Descend", "30", 49.86876640419947, 134.98920086393088, 3.0, None, None, None),
        (9, "Land", "30", None, None, None, 304.1338582677165, None, None),
        (10, "Decelerate", "-NONE-", None, 129.58963282937364, None, None, 3937.007874015748, 40.0),
        (11, "Decelerate", "-NONE-", None, 26.997840172786177, None, None, 0.0, 10.0),
    ),
    ("JETW", "Level_Idle"): (
        (1, "Descend", "ZERO", 6000.0, 250.0, 2.8, None, None, None),
        (2, "Level-Idle", "ZERO", 3000.0, 250.0, 16500.0, None, None, None),
        (3, "Level-Idle", "ZERO", 3000.0, 188.98488120950324, 5000.0, None, None, None),
        (4, "Descend-Idle", "15", 3000.0, 178.18574514038875, 3.0, None, None, None),
        (5, "Descend-Idle", "25", 2460.6299212598424, 167.38660907127428, 3.0, None, None, None),
        (6, "Descend-Idle", "25", 1968.503937007874, 140.3887688984881, 3.0, None, None, None),
        (7, "Descend", "30", 1640.4199475065616, 134.98920086393088, 3.0, None, None, None),
        (8, "Descend", "30", 49.86876640419947, 134.98920086393088, 3.0, None, None, None),
        (9, "Land", "30", None, None, None, 304.1338582677165, None, None),
        (10, "Decelerate", "-NONE-", None, 129.58963282937364, None, None, 3937.007874015748, 40.0),
        (11, "Decelerate", "-NONE-", None, 26.997840172786177, None, None, 0.0, 10.0),
    ),
    ("JETW", "Descend_Decel"): (
        (1, "Descend-Decel", "ZERO", 6000.0, 250.0, 2.8, None, None, None),
        (2, "Descend-Decel", "5", 3000.0, 180.0, 3.0, None, None, None),
        (3, "Descend-Decel", "25", 1500.0, 150.0, 3.0, None, None, None),
        (4, "Descend-Decel", "30", 1000.0, 135.0, 3.0, None, None, None),
        (5, "Descend-Decel", "30", 49.9, 135.0, 3.0, None, None, None),
        (6, "Land", "30", None, None, None, 304.1338582677165, None, None),
        (7, "Decelerate", "-NONE-", None, 129.58963282937364, None, None, 3937.007874015748, 40.0),
        (8, "Decelerate", "-NONE-", None, 26.997840172786177, None, None, 0.0, 10.0),
    ),
}
# fmt: on

ARRIVAL_CASES: dict[str, tuple[str, str, float, float, float, float, float]] = {
    # C-7: aircraft, procedure, airfield elevation (ft), temperature (degC),
    # headwind (kt), sea-level pressure (in-Hg), local airfield pressure (in-Hg).
    "2A": ("JETW", "Descend", 0.0, 15.0, 0.0, 29.92, 29.92),
    "2B": ("JETW", "Level_Decel", 0.0, 15.0, 0.0, 29.92, 29.92),
    "2C": ("JETW", "Level_Idle", 0.0, 15.0, 0.0, 29.92, 29.92),
    "2D": ("JETW", "Descend_Decel", 0.0, 15.0, 0.0, 29.92, 29.92),
    "2E": ("JETW", "Descend2", 0.0, 15.0, 0.0, 29.92, 29.92),
    "6A": ("JETW", "Descend", 0.0, 15.0, 8.0, 29.92, 29.92),
    "6B": ("JETW", "Level_Decel", 0.0, 15.0, 8.0, 29.92, 29.92),
    "6C": ("JETW", "Level_Idle", 0.0, 15.0, 8.0, 29.92, 29.92),
    "6E": ("JETW", "Descend2", 0.0, 15.0, 8.0, 29.92, 29.92),
}

DEPARTURE_CASES: dict[int, tuple[str, str, float, float, float, float]] = {
    # C-8: aircraft, procedure, airfield elevation (ft), temperature (degC),
    # headwind (kt), sea-level pressure (in-Hg). No separate local pressure
    # column: the departure cases set the airfield pressure from the elevation.
    1: ("JETW", "ICAO_B", 0.0, 15.0, 0.0, 29.92),
    6: ("JETW", "ICAO_A", 0.0, 15.0, 8.0, 29.92),
    10: ("JETW", "ICAO_A", 0.0, 15.0, -5.0, 29.92),
    26: ("JETW", "ICAO_A", 2000.0, 15.0, 8.0, 29.92),
    42: ("JETW", "ICAO_A", 0.0, 40.0, 0.0, 30.71),
    53: ("JETW", "ICAO_B", 5000.0, 40.0, 0.0, 29.92),
    54: ("JETW", "ICAO_A", 5000.0, 40.0, 0.0, 29.92),
    62: ("JETW", "ICAO_A_ESF", 0.0, 15.0, 8.0, 29.92),
    66: ("JETW", "ICAO_A_ESF", 5000.0, 40.0, 0.0, 29.92),
    74: ("JETW", "ICAO_A_Deep_cutback", 0.0, 15.0, 8.0, 29.92),
    80: ("JETW", "ICAO_A_Level_constant_speed", 0.0, 15.0, 8.0, 29.92),
    82: ("JETW", "ICAO_A_Level_ROC", 0.0, 15.0, 8.0, 29.92),
    84: ("JETW", "ICAO_A_Level_ESF", 0.0, 15.0, 8.0, 29.92),
    8: ("PROP", "ICAO_A", 0.0, 15.0, 8.0, 29.92),
    28: ("PROP", "ICAO_A", 2000.0, 15.0, 8.0, 29.92),
    56: ("PROP", "ICAO_A", 5000.0, 40.0, 0.0, 29.92),
    68: ("PROP", "ICAO_A_ESF", 5000.0, 40.0, 0.0, 29.92),
}

ProfilePoint = tuple[int, float, float, float, float]
"""D-1/D-2 row: point number, distance (ft), altitude (ft), true airspeed (kt),
corrected net thrust per engine (lb). Every column is rounded to 0.1."""

ARRIVAL_RESULTS: dict[str, tuple[ProfilePoint, ...]] = {
    # D-1: 124 points. Distance is negative before touchdown, which sits at 0 ft.
    "2A": (
        (1, -118582.9, 6000.0, 273.4, 533.1),
        (2, -58243.4, 3048.9, 189.9, 477.6),
        (3, -57243.4, 3000.0, 188.2, 1342.5),
        (4, -29621.7, 1552.4, 154.7, 1273.0),
        (5, -28621.7, 1500.0, 153.3, 3504.2),
        (6, -20081.1, 1052.4, 138.8, 3447.5),
        (7, -19081.1, 1000.0, 137.0, 4903.1),
        (8, -952.1, 49.9, 135.1, 4737.0),
        (9, -476.1, 24.9, 133.8, 4732.7),
        (10, 0.0, 0.0, 132.5, 4724.1),
        (11, 304.1, 0.0, 129.6, 10000.0),
        (12, 4241.1, 0.0, 27.0, 2500.0),
    ),
    "2B": (
        (1, -148684.6, 6000.0, 273.4, 533.1),
        (2, -88345.1, 3048.9, 261.5, 477.6),
        (3, -87345.1, 3000.0, 261.3, 451.1),
        (4, -61098.4, 3000.0, 197.6, 418.4),
        (5, -58243.4, 3000.0, 189.3, 418.4),
        (6, -57243.4, 3000.0, 186.3, 235.3),
        (7, -47951.0, 2513.0, 174.9, 231.1),
        (8, -46951.0, 2460.6, 173.6, 313.7),
        (9, -37561.2, 1968.5, 144.5, 2700.0),
        (10, -32300.7, 1692.8, 139.3, 2672.9),
        (11, -31300.7, 1640.4, 138.3, 5018.9),
        (12, -952.1, 49.9, 135.1, 4737.0),
        (13, -476.1, 24.9, 133.8, 4732.7),
        (14, 0.0, 0.0, 132.5, 4724.1),
        (15, 304.1, 0.0, 129.6, 10000.0),
        (16, 4241.1, 0.0, 27.0, 2500.0),
    ),
    "2C": (
        (1, -140082.9, 6000.0, 273.4, 533.1),
        (2, -79743.4, 3048.9, 261.5, 477.6),
        (3, -78743.4, 3000.0, 261.3, -105.0),
        (4, -62243.4, 3000.0, 197.6, 291.5),
        (5, -57243.4, 3000.0, 186.3, 361.7),
        (6, -46951.0, 2460.6, 173.6, 369.7),
        (7, -37561.2, 1968.5, 144.5, 483.3),
        (8, -32300.7, 1692.8, 139.3, 475.9),
        (9, -31300.7, 1640.4, 138.3, 5018.9),
        (10, -952.1, 49.9, 135.1, 4737.0),
        (11, -476.1, 24.9, 133.8, 4732.7),
        (12, 0.0, 0.0, 132.5, 4724.1),
        (13, 304.1, 0.0, 129.6, 10000.0),
        (14, 4241.1, 0.0, 27.0, 2500.0),
    ),
    "2D": (
        (1, -118582.9, 6000.0, 273.4, -1993.3),
        (2, -58243.4, 3048.9, 189.9, -1785.5),
        (3, -57243.4, 3000.0, 188.2, -63.8),
        (4, -29621.7, 1552.4, 154.7, -60.5),
        (5, -28621.7, 1500.0, 153.3, 1931.7),
        (6, -20081.1, 1052.4, 138.8, 1900.4),
        (7, -19081.1, 1000.0, 137.0, 4921.4),
        (8, -952.1, 49.9, 135.1, 2520.7),
        (9, -476.1, 24.9, 133.8, 2518.5),
        (10, 0.0, 0.0, 132.5, 4724.1),
        (11, 304.1, 0.0, 129.6, 10000.0),
        (12, 4241.1, 0.0, 27.0, 2500.0),
    ),
    "2E": (
        (1, -115852.2, 6000.0, 273.4, 533.1),
        (2, -96405.7, 5048.9, 249.1, 514.4),
        (3, -95405.7, 5000.0, 247.8, 199.2),
        (4, -58243.4, 3052.4, 190.0, 185.3),
        (5, -57243.4, 3000.0, 188.2, 1342.5),
        (6, -29621.7, 1552.4, 154.7, 1273.0),
        (7, -28621.7, 1500.0, 153.3, 3504.2),
        (8, -20081.1, 1052.4, 138.8, 3447.5),
        (9, -19081.1, 1000.0, 137.0, 4903.1),
        (10, -952.1, 49.9, 135.1, 4737.0),
        (11, -476.1, 24.9, 133.8, 4732.7),
        (12, 0.0, 0.0, 132.5, 4724.1),
        (13, 304.1, 0.0, 129.6, 10000.0),
        (14, 4241.1, 0.0, 27.0, 2500.0),
    ),
    "6A": (
        (1, -118582.9, 6000.0, 273.4, 677.1),
        (2, -58243.4, 3048.9, 189.9, 606.5),
        (3, -57243.4, 3000.0, 188.2, 1534.0),
        (4, -29621.7, 1552.4, 154.7, 1454.6),
        (5, -28621.7, 1500.0, 153.3, 3721.7),
        (6, -20081.1, 1052.4, 138.8, 3661.5),
        (7, -19081.1, 1000.0, 137.0, 5140.4),
        (8, -952.1, 49.9, 135.1, 4966.3),
        (9, -476.1, 24.9, 133.8, 4961.8),
        (10, 0.0, 0.0, 132.5, 4957.3),
        (11, 304.1, 0.0, 129.6, 10000.0),
        (12, 4241.1, 0.0, 27.0, 2500.0),
    ),
    "6B": (
        (1, -148684.6, 6000.0, 273.4, 677.1),
        (2, -88345.1, 3048.9, 261.5, 606.5),
        (3, -87345.1, 3000.0, 261.3, 588.7),
        (4, -61098.4, 3000.0, 197.6, 584.2),
        (5, -58243.4, 3000.0, 189.3, 584.2),
        (6, -57243.4, 3000.0, 186.3, 305.1),
        (7, -47951.0, 2513.0, 174.9, 299.7),
        (8, -46951.0, 2460.6, 173.6, 485.5),
        (9, -37561.2, 1968.5, 144.5, 2754.3),
        (10, -32300.7, 1692.8, 139.3, 2726.6),
        (11, -31300.7, 1640.4, 138.3, 5261.9),
        (12, -952.1, 49.9, 135.1, 4966.3),
        (13, -476.1, 24.9, 133.8, 4961.8),
        (14, 0.0, 0.0, 132.5, 4957.3),
        (15, 304.1, 0.0, 129.6, 10000.0),
        (16, 4241.1, 0.0, 27.0, 2500.0),
    ),
    "6C": (
        (1, -140082.9, 6000.0, 273.4, 677.1),
        (2, -79743.4, 3048.9, 261.5, 606.5),
        (3, -78743.4, 3000.0, 261.3, -105.0),
        (4, -62243.4, 3000.0, 197.6, 291.5),
        (5, -57243.4, 3000.0, 186.3, 361.7),
        (6, -46951.0, 2460.6, 173.6, 369.7),
        (7, -37561.2, 1968.5, 144.5, 483.3),
        (8, -32300.7, 1692.8, 139.3, 475.9),
        (9, -31300.7, 1640.4, 138.3, 5261.9),
        (10, -952.1, 49.9, 135.1, 4966.3),
        (11, -476.1, 24.9, 133.8, 4961.8),
        (12, 0.0, 0.0, 132.5, 4957.3),
        (13, 304.1, 0.0, 129.6, 10000.0),
        (14, 4241.1, 0.0, 27.0, 2500.0),
    ),
    "6E": (
        (1, -115852.2, 6000.0, 273.4, 677.1),
        (2, -96405.7, 5048.9, 249.1, 653.3),
        (3, -95405.7, 5000.0, 247.8, 360.7),
        (4, -58243.4, 3052.4, 190.0, 335.5),
        (5, -57243.4, 3000.0, 188.2, 1534.0),
        (6, -29621.7, 1552.4, 154.7, 1454.6),
        (7, -28621.7, 1500.0, 153.3, 3721.7),
        (8, -20081.1, 1052.4, 138.8, 3661.5),
        (9, -19081.1, 1000.0, 137.0, 5140.4),
        (10, -952.1, 49.9, 135.1, 4966.3),
        (11, -476.1, 24.9, 133.8, 4961.8),
        (12, 0.0, 0.0, 132.5, 4957.3),
        (13, 304.1, 0.0, 129.6, 10000.0),
        (14, 4241.1, 0.0, 27.0, 2500.0),
    ),
}

DEPARTURE_RESULTS: dict[int, tuple[ProfilePoint, ...]] = {
    # D-2: 190 points. Distance is measured from the start of the take-off roll.
    1: (
        (1, 0.0, 0.0, 0.0, 25000.0),
        (2, 5417.3, 0.0, 162.7, 20933.7),
        (3, 11096.6, 1000.0, 165.1, 21243.7),
        (4, 12096.6, 1051.0, 169.2, 15742.9),
        (5, 25047.9, 1711.7, 216.0, 15813.0),
        (6, 29329.0, 1887.1, 233.2, 15812.0),
        (7, 39205.0, 3000.0, 237.1, 16202.8),
        (8, 45897.4, 3233.6, 263.4, 16184.5),
        (9, 67096.3, 5500.0, 272.6, 16893.1),
        (10, 87063.6, 7500.0, 281.0, 17433.1),
        (11, 114187.6, 10000.0, 292.2, 17995.6),
    ),
    6: (
        (1, 0.0, 0.0, 0.0, 25000.0),
        (2, 4897.5, 0.0, 162.7, 20933.7),
        (3, 13051.3, 1500.0, 166.3, 21406.2),
        (4, 14051.3, 1612.3, 166.6, 15968.3),
        (5, 26405.0, 3000.0, 170.0, 16459.4),
        (6, 41657.3, 3787.5, 222.8, 16529.2),
        (7, 46335.6, 3980.2, 240.6, 16526.5),
        (8, 53213.4, 4224.6, 267.4, 16507.0),
        (9, 64922.7, 5500.0, 272.6, 16893.1),
        (10, 84249.5, 7500.0, 281.0, 17433.1),
        (11, 110504.4, 10000.0, 292.2, 17995.6),
    ),
    8: (
        (1, 0.0, 0.0, 0.0, 17736.5),
        (2, 7013.5, 0.0, 148.4, 17736.5),
        (3, 21469.1, 1500.0, 151.7, 18321.1),
        (4, 22469.1, 1561.1, 151.9, 15062.6),
        (5, 46012.5, 3000.0, 155.2, 15543.6),
        (6, 48052.0, 3079.1, 158.2, 15284.3),
        (7, 56833.4, 3621.9, 170.9, 14436.9),
        (8, 92593.0, 4924.7, 215.0, 12048.4),
        (9, 106449.4, 5500.0, 216.9, 12204.0),
        (10, 157554.2, 7500.0, 223.6, 12766.0),
        (11, 228916.0, 10000.0, 232.5, 13517.6),
    ),
    10: (
        (1, 0.0, 0.0, 0.0, 25000.0),
        (2, 5755.5, 0.0, 162.7, 20933.7),
        (3, 14609.5, 1500.0, 166.3, 21406.2),
        (4, 15609.5, 1603.6, 166.5, 15965.1),
        (5, 29094.7, 3000.0, 170.0, 16459.4),
        (6, 45389.7, 3787.5, 222.8, 16529.2),
        (7, 50339.7, 3980.2, 240.6, 16526.5),
        (8, 57580.5, 4224.6, 267.4, 16507.0),
        (9, 69920.8, 5500.0, 272.6, 16893.1),
        (10, 90288.3, 7500.0, 281.0, 17433.1),
        (11, 117955.5, 10000.0, 292.2, 17995.6),
    ),
    26: (
        (1, 0.0, 0.0, 0.0, 25640.0),
        (2, 5496.8, 0.0, 168.7, 21573.7),
        (3, 14161.4, 1500.0, 172.5, 22106.2),
        (4, 15161.4, 1606.2, 172.8, 16661.8),
        (5, 28281.4, 3000.0, 176.5, 17099.4),
        (6, 46386.9, 3899.6, 231.6, 17169.4),
        (7, 51887.0, 4117.4, 250.3, 17165.5),
        (8, 59941.7, 4392.4, 278.3, 17143.9),
        (9, 70809.7, 5500.0, 283.1, 17433.1),
        (10, 91604.9, 7500.0, 292.0, 17893.1),
        (11, 120239.0, 10000.0, 303.8, 18355.6),
    ),
    28: (
        (1, 0.0, 0.0, 0.0, 18393.8),
        (2, 7822.5, 0.0, 153.9, 18393.8),
        (3, 23455.1, 1500.0, 157.4, 19007.3),
        (4, 24455.1, 1554.8, 157.5, 15624.8),
        (5, 50807.7, 3000.0, 161.0, 16132.1),
        (6, 52992.3, 3070.7, 164.2, 15860.3),
        (7, 62435.0, 3597.1, 177.4, 14977.6),
        (8, 100680.6, 4797.9, 222.8, 12475.2),
        (9, 119609.3, 5500.0, 225.2, 12674.5),
        (10, 177340.3, 7500.0, 232.4, 13265.5),
        (11, 258971.4, 10000.0, 241.8, 14056.5),
    ),
    42: (
        (1, 0.0, 0.0, 0.0, 23170.7),
        (2, 6123.4, 0.0, 167.4, 19104.4),
        (3, 15516.2, 1500.0, 171.1, 19648.0),
        (4, 16516.2, 1594.8, 171.4, 14548.5),
        (5, 31345.8, 3000.0, 175.1, 14874.5),
        (6, 53058.2, 4043.6, 230.3, 14924.7),
        (7, 59399.3, 4287.7, 248.9, 14916.5),
        (8, 68545.0, 4592.1, 276.9, 14889.9),
        (9, 78395.7, 5500.0, 280.8, 15100.5),
        (10, 101226.2, 7500.0, 289.6, 15564.4),
        (11, 132260.5, 10000.0, 301.3, 16144.2),
    ),
    53: (
        (1, 0.0, 0.0, 0.0, 23170.7),
        (2, 9318.2, 0.0, 185.9, 19104.4),
        (3, 17900.2, 1000.0, 188.8, 19466.8),
        (4, 18900.2, 1044.0, 190.1, 14416.8),
        (5, 77597.1, 3626.3, 254.8, 14827.9),
        (6, 91988.4, 4127.4, 276.7, 14879.4),
        (7, 111375.3, 4708.3, 309.2, 14916.9),
        (8, 123762.4, 5500.0, 313.1, 15100.5),
        (9, 157006.6, 7500.0, 323.5, 15564.4),
        (10, 203174.2, 10000.0, 337.2, 16144.2),
    ),
    54: (
        (1, 0.0, 0.0, 0.0, 23170.7),
        (2, 9318.2, 0.0, 185.9, 19104.4),
        (3, 22289.0, 1500.0, 190.3, 19648.0),
        (4, 23289.0, 1563.4, 190.5, 14541.3),
        (5, 45938.3, 3000.0, 194.9, 14874.5),
        (6, 114395.3, 5689.8, 263.4, 15306.5),
        (7, 132898.4, 6313.0, 286.6, 15386.3),
        (8, 157312.6, 7019.0, 321.0, 15452.8),
        (9, 165587.7, 7500.0, 323.5, 15564.4),
        (10, 211755.2, 10000.0, 337.2, 16144.2),
    ),
    56: (
        (1, 0.0, 0.0, 0.0, 18652.1),
        (2, 11696.0, 0.0, 169.6, 18652.1),
        (3, 32420.8, 1500.0, 173.6, 19277.6),
        (4, 33420.8, 1537.1, 173.7, 15840.9),
        (5, 72863.9, 3000.0, 177.8, 16364.5),
        (6, 75594.6, 3042.9, 181.3, 16078.9),
        (7, 87502.6, 3506.4, 195.7, 15163.6),
        (8, 134570.8, 4367.1, 244.7, 12535.1),
        (9, 179690.3, 5500.0, 249.2, 12861.1),
        (10, 268458.9, 7500.0, 257.4, 13464.3),
        (11, 400814.3, 10000.0, 268.3, 14272.0),
    ),
    62: (
        (1, 0.0, 0.0, 0.0, 25000.0),
        (2, 4897.5, 0.0, 162.7, 20933.7),
        (3, 13051.3, 1500.0, 166.3, 21406.2),
        (4, 14051.3, 1612.3, 166.6, 15968.3),
        (5, 26405.0, 3000.0, 170.0, 16459.4),
        (6, 41657.3, 3787.5, 222.8, 16529.2),
        (7, 46335.7, 3980.2, 240.6, 16526.5),
        (8, 53213.5, 4224.6, 267.4, 16507.0),
        (9, 64922.8, 5500.0, 272.6, 16893.1),
        (10, 84249.5, 7500.0, 281.0, 17433.1),
        (11, 110504.4, 10000.0, 292.2, 17995.6),
    ),
    66: (
        (1, 0.0, 0.0, 0.0, 23170.7),
        (2, 9318.2, 0.0, 185.9, 19104.4),
        (3, 22289.0, 1500.0, 190.3, 19648.0),
        (4, 23289.0, 1563.4, 190.5, 14541.3),
        (5, 45938.3, 3000.0, 194.9, 14874.5),
        (6, 82888.8, 4059.8, 256.6, 14928.5),
        (7, 93733.5, 4320.1, 277.5, 14924.0),
        (8, 109363.8, 4650.8, 308.9, 14903.5),
        (9, 122635.3, 5500.0, 313.1, 15100.5),
        (10, 155879.6, 7500.0, 323.5, 15564.4),
        (11, 202047.1, 10000.0, 337.2, 16144.2),
    ),
    68: (
        (1, 0.0, 0.0, 0.0, 18652.1),
        (2, 11696.0, 0.0, 169.6, 18652.1),
        (3, 32420.8, 1500.0, 173.6, 19277.6),
        (4, 33420.8, 1537.1, 173.7, 15840.9),
        (5, 72863.9, 3000.0, 177.8, 16364.5),
        (6, 75594.6, 3042.9, 181.3, 16078.9),
        (7, 87502.6, 3506.4, 195.7, 15163.6),
        (8, 134570.8, 4367.1, 244.7, 12535.1),
        (9, 179690.3, 5500.0, 249.2, 12861.1),
        (10, 268458.9, 7500.0, 257.4, 13464.3),
        (11, 400814.3, 10000.0, 268.3, 14272.0),
    ),
    74: (
        (1, 0.0, 0.0, 0.0, 25000.0),
        (2, 4897.5, 0.0, 162.7, 20933.7),
        (3, 9199.4, 800.0, 164.6, 21180.1),
        (4, 10199.4, 895.1, 164.8, 13985.2),
        (5, 21814.3, 2000.0, 167.5, 14560.7),
        (6, 22814.3, 2111.6, 167.8, 16149.5),
        (7, 30772.3, 3000.0, 170.0, 16459.4),
        (8, 46024.6, 3787.5, 222.8, 16529.1),
        (9, 50703.1, 3980.2, 240.6, 16526.5),
        (10, 57580.8, 4224.6, 267.4, 16507.0),
        (11, 69290.1, 5500.0, 272.6, 16893.1),
        (12, 88616.9, 7500.0, 281.0, 17433.1),
        (13, 114871.8, 10000.0, 292.2, 17995.6),
    ),
    80: (
        (1, 0.0, 0.0, 0.0, 25000.0),
        (2, 4897.5, 0.0, 162.7, 20933.7),
        (3, 13051.3, 1500.0, 166.3, 21406.2),
        (4, 14051.3, 1612.3, 166.6, 15968.3),
        (5, 26405.0, 3000.0, 170.0, 16459.4),
        (6, 27405.0, 3000.0, 170.0, 6457.1),
        (7, 36405.0, 3000.0, 170.0, 6457.1),
        (8, 37405.0, 3051.6, 174.0, 16462.3),
        (9, 51657.3, 3787.5, 222.8, 16529.2),
        (10, 56335.6, 3980.2, 240.6, 16526.5),
        (11, 63213.4, 4224.6, 267.4, 16507.0),
        (12, 74922.7, 5500.0, 272.6, 16893.1),
        (13, 94249.5, 7500.0, 281.0, 17433.1),
        (14, 120504.4, 10000.0, 292.2, 17995.6),
    ),
    82: (
        (1, 0.0, 0.0, 0.0, 25000.0),
        (2, 4897.5, 0.0, 162.7, 20933.7),
        (3, 13051.3, 1500.0, 166.3, 21406.2),
        (4, 14051.3, 1612.3, 166.6, 15968.3),
        (5, 26405.0, 3000.0, 170.0, 16459.4),
        (6, 34068.5, 3000.0, 220.2, 16267.6),
        (7, 36876.0, 3000.0, 237.1, 16202.8),
        (8, 43354.6, 3233.6, 263.4, 16184.5),
        (9, 63873.0, 5500.0, 272.6, 16893.1),
        (10, 83199.8, 7500.0, 281.0, 17433.1),
        (11, 109454.7, 10000.0, 292.2, 17995.6),
    ),
    84: (
        (1, 0.0, 0.0, 0.0, 25000.0),
        (2, 4897.5, 0.0, 162.7, 20933.7),
        (3, 13051.3, 1500.0, 166.3, 21406.2),
        (4, 14051.3, 1612.3, 166.6, 15968.3),
        (5, 26405.0, 3000.0, 170.0, 16459.4),
        (6, 34068.5, 3000.0, 220.2, 16267.6),
        (7, 36876.0, 3000.0, 237.1, 16202.8),
        (8, 43383.7, 3237.0, 263.4, 16185.6),
        (9, 63872.8, 5500.0, 272.6, 16893.1),
        (10, 83199.5, 7500.0, 281.0, 17433.1),
        (11, 109454.4, 10000.0, 292.2, 17995.6),
    ),
}
