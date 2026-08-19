#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Program loudness (ITU-R BS.1770 / EBU R 128).

The broadcast loudness chain: the BS.1770 K-weighting and mean-square gating,
the EBU Tech 3341 momentary, short-term and integrated loudness meters, and
the EBU Tech 3342 loudness range.

The oracles are the compliance test signals the two EBU documents tabulate,
generated here at the sampling rate they are specified at, so the expected
value is the number printed in the table rather than a recording.
"""

from __future__ import annotations

import numpy as np
import reference_data as ref

import phonometry as ph

from ..registry import _DELTA_PLACES, Outcome, _fmt, numeric, register

_PROGRAM_LOUDNESS = "Program loudness (ITU-R BS.1770 / EBU R 128)"
_EBU_FS = 48000


def _ebu_tone(level_dbfs: float, duration_s: float, freq: float = 1000.0) -> np.ndarray:
    """A sine with per-sample peak level ``level_dbfs`` re full scale."""
    t = np.arange(round(duration_s * _EBU_FS)) / _EBU_FS
    return np.asarray(10.0 ** (level_dbfs / 20.0) * np.sin(2 * np.pi * freq * t))


def _ebu_stereo_steps(segments: tuple[tuple[float, float], ...]) -> np.ndarray:
    """Concatenated 1 kHz tone steps applied in phase to both channels."""
    x = np.concatenate([_ebu_tone(lvl, dur) for lvl, dur in segments])
    return np.vstack([x, x])


@register(
    _PROGRAM_LOUDNESS,
    "ITU-R BS.1770-5 Annex 1",
    "997 Hz sine at 0 dB FS on the left channel, LKFS",
)
def _chk_bs1770_anchor() -> Outcome:
    x = np.vstack([_ebu_tone(0.0, 10.0, freq=997.0), np.zeros(10 * _EBU_FS)])
    computed = ph.broadcast.integrated_loudness(x, _EBU_FS)
    return numeric(ref.BS1770_ANCHOR_997_LKFS, computed, 0.01, unit="LKFS", places=2)


@register(
    _PROGRAM_LOUDNESS,
    "EBU Tech 3341:2023 Table 1 case 1",
    "Integrated loudness of the -23 dBFS stereo sine, LUFS",
)
def _chk_tech3341_case1() -> Outcome:
    _, segments, expected = ref.EBU_TECH3341_INTEGRATED_CASES[0]
    computed = ph.broadcast.integrated_loudness(
        _ebu_stereo_steps(segments), _EBU_FS
    )
    return numeric(expected, computed, ref.EBU_TECH3341_TOL_LU, unit="LUFS", places=2)


@register(
    _PROGRAM_LOUDNESS,
    "EBU Tech 3341:2023 Table 1 case 5",
    "Gated integrated loudness of the -26/-20/-26 dBFS steps, LUFS",
)
def _chk_tech3341_case5() -> Outcome:
    _, segments, expected = ref.EBU_TECH3341_INTEGRATED_CASES[4]
    computed = ph.broadcast.integrated_loudness(
        _ebu_stereo_steps(segments), _EBU_FS
    )
    return numeric(expected, computed, ref.EBU_TECH3341_TOL_LU, unit="LUFS", places=2)


@register(
    _PROGRAM_LOUDNESS,
    "EBU Tech 3341:2023 Table 1 case 6",
    "Integrated loudness of the 5.0-channel sine (Table 3 weights), LUFS",
)
def _chk_tech3341_case6() -> Outcome:
    x = np.vstack([_ebu_tone(lvl, 20.0) for lvl in ref.EBU_TECH3341_CASE6_LEVELS])
    computed = ph.broadcast.integrated_loudness(x, _EBU_FS)
    return numeric(
        ref.EBU_TECH3341_CASE6_EXPECTED,
        computed,
        ref.EBU_TECH3341_TOL_LU,
        unit="LUFS",
        places=2,
    )


def _ebu_true_peak_case(index: int) -> tuple[float, float]:
    """Synthesize a Tech 3341 true-peak tone case; return (expected, computed)."""
    _, freq_ratio, amplitude, phase_deg, expected = ref.EBU_TECH3341_TRUE_PEAK_CASES[
        index
    ]
    t = np.arange(_EBU_FS) / _EBU_FS
    x = amplitude * np.sin(
        2 * np.pi * (freq_ratio * _EBU_FS) * t + np.deg2rad(phase_deg)
    )
    n = int(0.01 * _EBU_FS)
    x[:n] *= np.linspace(0.0, 1.0, n)
    x[-n:] *= np.linspace(1.0, 0.0, n)
    return expected, float(ph.broadcast.true_peak_level(x, _EBU_FS))


def _true_peak_outcome(expected: float, computed: float) -> Outcome:
    """Tech 3341 true-peak verdict with its asymmetric +0.2/-0.4 dB window."""
    delta = computed - expected
    passed = -ref.EBU_TECH3341_TP_TOL_DOWN <= delta <= ref.EBU_TECH3341_TP_TOL_UP
    return Outcome(
        expected=(
            f"{_fmt(expected, 'dBTP', 2)} "
            f"(+{ref.EBU_TECH3341_TP_TOL_UP:g}/-{ref.EBU_TECH3341_TP_TOL_DOWN:g} dB)"
        ),
        computed=_fmt(computed, "dBTP", 2),
        delta=_fmt(delta, "dBTP", _DELTA_PLACES),
        passed=passed,
    )


@register(
    _PROGRAM_LOUDNESS,
    "EBU Tech 3341:2023 Table 1 case 15",
    "True-peak level of the fs/4 sine at 0.5 FFS, dBTP",
)
def _chk_tech3341_case15() -> Outcome:
    return _true_peak_outcome(*_ebu_true_peak_case(0))


@register(
    _PROGRAM_LOUDNESS,
    "EBU Tech 3341:2023 Table 1 case 19",
    "True-peak level of the fs/4 sine at 1.41 FFS, dBTP",
)
def _chk_tech3341_case19() -> Outcome:
    return _true_peak_outcome(*_ebu_true_peak_case(4))


@register(
    _PROGRAM_LOUDNESS,
    "EBU Tech 3342:2023 Table 1 case 1",
    "Loudness range of the -20/-30 dBFS tone steps, LU",
)
def _chk_tech3342_case1() -> Outcome:
    _, levels, expected = ref.EBU_TECH3342_LRA_CASES[0]
    x = _ebu_stereo_steps(tuple((lvl, 20.0) for lvl in levels))
    res = ph.broadcast.program_loudness(x, _EBU_FS)
    return numeric(
        expected, res.loudness_range, ref.EBU_TECH3342_TOL_LU, unit="LU", places=2
    )


@register(
    _PROGRAM_LOUDNESS,
    "EBU Tech 3342:2023 Table 1 case 3",
    "Loudness range of the -40/-20 dBFS tone steps, LU",
)
def _chk_tech3342_case3() -> Outcome:
    _, levels, expected = ref.EBU_TECH3342_LRA_CASES[2]
    x = _ebu_stereo_steps(tuple((lvl, 20.0) for lvl in levels))
    res = ph.broadcast.program_loudness(x, _EBU_FS)
    return numeric(
        expected, res.loudness_range, ref.EBU_TECH3342_TOL_LU, unit="LU", places=2
    )
