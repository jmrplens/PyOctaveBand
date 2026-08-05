#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Programme loudness and true peak: the EBU compliance test signals.

ITU-R BS.1770 defines the loudness meter and anchors it on a single 997 Hz
sine; EBU Tech 3341 and Tech 3342 define what a conforming meter must
report for a list of synthetic signals, with the tolerance each reading is
allowed. The cases are the standard's own acceptance test, so they are
transcribed as cases - signal description, expected reading, tolerance -
rather than as bare numbers.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# ITU-R BS.1770-5 / EBU R 128 - programme loudness and true peak.
# BS.1770-5 Annex 1 anchors a 0 dB FS 997 Hz sine on one front channel at
# -3.01 LKFS. EBU Tech 3341 Table 1 gives the 'minimum requirements' test
# signals (stereo 1 kHz sines unless noted, per-channel peak levels in dBFS)
# with +/-0.1 LU tolerance for loudness and +0.2/-0.4 dB for true peak;
# EBU Tech 3342 Table 1 gives the loudness-range signals (20 s stereo 1 kHz
# tone steps) with +/-1 LU tolerance.
# ---------------------------------------------------------------------------
BS1770_ANCHOR_997_LKFS = -3.01  # LKFS

EBU_TECH3341_TOL_LU = 0.1
#: Tech 3341 Table 1 integrated-loudness cases 1-5:
#: (case, ((per-channel peak level dBFS, duration s), ...), expected I LUFS).
EBU_TECH3341_INTEGRATED_CASES: list[
    tuple[int, tuple[tuple[float, float], ...], float]
] = [
    (1, ((-23.0, 20.0),), -23.0),
    (2, ((-33.0, 20.0),), -33.0),
    (3, ((-36.0, 10.0), (-23.0, 60.0), (-36.0, 10.0)), -23.0),
    (
        4,
        ((-72.0, 10.0), (-36.0, 10.0), (-23.0, 60.0), (-36.0, 10.0), (-72.0, 10.0)),
        -23.0,
    ),
    (5, ((-26.0, 20.0), (-20.0, 20.1), (-26.0, 20.0)), -23.0),
]
#: Tech 3341 case 6: 5.0-channel sine, per-channel peak levels (L, R, C,
#: Ls, Rs), 20 s, expected I = -23.0 LUFS.
EBU_TECH3341_CASE6_LEVELS = (-28.0, -28.0, -24.0, -30.0, -30.0)
EBU_TECH3341_CASE6_EXPECTED = -23.0

#: Tech 3341 Table 1 true-peak cases 15-19:
#: (case, frequency as a fraction of fs, amplitude FFS, phase deg,
#: expected max true-peak level dBTP). Tolerance +0.2/-0.4 dB.
EBU_TECH3341_TRUE_PEAK_CASES: list[tuple[int, float, float, float, float]] = [
    (15, 1.0 / 4.0, 0.50, 0.0, -6.0),
    (16, 1.0 / 4.0, 0.50, 45.0, -6.0),
    (17, 1.0 / 6.0, 0.50, 60.0, -6.0),
    (18, 1.0 / 8.0, 0.50, 67.5, -6.0),
    (19, 1.0 / 4.0, 1.41, 45.0, 3.0),
]
EBU_TECH3341_TP_TOL_UP = 0.2  # dB
EBU_TECH3341_TP_TOL_DOWN = 0.4  # dB
#: Cases 20-23 (a single fs/4 period inside an fs/6 tone, synthesized at
#: 4 fs and downsampled with a 0-3 sample offset) all expect 0.0 dBTP.
EBU_TECH3341_TP_OFFSET_EXPECTED = 0.0

EBU_TECH3342_TOL_LU = 1.0
#: Tech 3342 Table 1 LRA cases 1-4: (case, per-channel peak level dBFS of
#: each 20 s tone segment, expected LRA LU).
EBU_TECH3342_LRA_CASES: list[tuple[int, tuple[float, ...], float]] = [
    (1, (-20.0, -30.0), 10.0),
    (2, (-20.0, -15.0), 5.0),
    (3, (-40.0, -20.0), 20.0),
    (4, (-50.0, -35.0, -20.0, -35.0, -50.0), 15.0),
]
