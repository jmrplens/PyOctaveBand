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

# ---------------------------------------------------------------------------
# ITU-R BS.468-4 clause 2 - the quasi-peak detector's eleven acceptance
# windows, transcribed from PDF p. 4 = printed p. 4 rendered at 300 dpi
# whole-page and 600 dpi cropped. Clause 2 prints no time constant and no
# transfer function, so these windows are the entire specification of the
# dynamics, and they are an acceptance region rather than a set of values:
# every reading inside a window conforms.
#
# The percentages are percentages of the reading the same 5 kHz tone gives
# steadily, not of full scale. Table 3's 100 % upper limit at 100 bursts per
# second settles it: under the other reading it would sit 1.94 dB above the
# steady tone, which no peak follower can reach.
#
# The percentage rows are primary and the dB rows are derived from them. All
# 33 cells were audited: exactly one is inconsistent with its own percentage,
# Table 2's 5 ms upper limit, and it is the dB cell that is wrong (see
# docs/ERRATA.md).
# ---------------------------------------------------------------------------
#: Table 2, single 5 kHz tone bursts: (burst duration ms, full 5 kHz periods,
#: lower limit %, reference reading %, upper limit %). The 1 ms column carries
#: the footnote "The Administration of the USSR intends to use burst duration
#: >= 5 ms"; it is met here anyway, because it is the only constraint on the
#: fast end of the ballistics.
BS468_TABLE2_SINGLE_BURSTS: list[tuple[float, int, float, float, float]] = [
    (1.0, 5, 13.5, 17.0, 21.4),
    (2.0, 10, 22.4, 26.6, 31.6),
    (5.0, 25, 34.0, 40.0, 46.0),
    (10.0, 50, 41.0, 48.0, 55.0),
    (20.0, 100, 44.0, 52.0, 60.0),
    (50.0, 250, 50.0, 59.0, 68.0),
    (100.0, 500, 58.0, 68.0, 78.0),
    (200.0, 1000, 68.0, 80.0, 92.0),
]

#: Table 3, repetitive 5 ms (25-cycle) 5 kHz bursts: (bursts per second,
#: lower limit %, reference reading %, upper limit %). Duty cycles 1, 5 and
#: 50 %. The 100 per second window is the sharpest test in the document,
#: 0.537 dB wide, and its upper limit is the physical ceiling: the train
#: "may reach but not exceed the steady reading".
BS468_TABLE3_BURST_TRAINS: list[tuple[float, float, float, float]] = [
    (2.0, 43.0, 48.0, 53.0),
    (10.0, 72.0, 77.0, 82.0),
    (100.0, 94.0, 97.0, 100.0),
]

#: Carrier of every clause 2.1, 2.2 and 2.3 stimulus, in Hz.
BS468_BURST_HZ = 5000.0

#: Clause 2.6: the r.m.s. voltage of the 1 kHz sine that reads 0 dBqps.
BS468_CALIBRATION_V = 0.775

#: Clause 2.3: isolated bursts of this duration, in ms (3 whole 5 kHz
#: periods), stepped down over 20 dB, must track within +-1 dB.
BS468_OVERLOAD_BURST_MS = 0.6
BS468_OVERLOAD_RANGE_DB = 20.0
BS468_OVERLOAD_TOL_DB = 1.0

#: Clause 2.4: the reading must not change by more than this when the
#: polarity of an asymmetrical signal is reversed, tested unweighted.
BS468_REVERSIBILITY_TOL_DB = 0.5

#: Clause 2.5: momentary excess reading on a suddenly applied 1 kHz tone.
BS468_OVERSWING_TOL_DB = 0.3
