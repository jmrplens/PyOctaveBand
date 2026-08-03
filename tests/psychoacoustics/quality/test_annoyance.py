#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Tests for psychoacoustic annoyance (Fastl & Zwicker Eqs 16.2-16.4).

The PA model is exact; these tests anchor it on a hand-computed worked tuple and
on the formula's structural properties (the sharpness-term threshold, the
loudness weighting and monotonicity), plus the signal-based convenience.
"""

from __future__ import annotations

import math

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pytest
import reference_data as ref

from phonometry import (
    PsychoacousticAnnoyanceResult,
    psychoacoustic_annoyance,
    psychoacoustic_annoyance_from_signal,
)


def test_pa_matches_hand_worked_tuple() -> None:
    n5, s, f, r = ref.PA_WORKED_INPUT
    res = psychoacoustic_annoyance(n5, s, f, r)
    assert res.annoyance == pytest.approx(ref.PA_WORKED_VALUE, abs=1e-3)
    assert res.w_s == pytest.approx(ref.PA_WORKED_WS, abs=1e-4)
    assert res.w_fr == pytest.approx(ref.PA_WORKED_WFR, abs=1e-4)


def test_pa_reduces_to_loudness_without_terms() -> None:
    # S <= 1.75 acum and F = R = 0 -> wS = wFR = 0 -> PA = N5.
    res = psychoacoustic_annoyance(12.0, 1.5, 0.0, 0.0)
    assert res.w_s == 0.0
    assert res.w_fr == 0.0
    assert res.annoyance == pytest.approx(12.0)


def test_pa_sharpness_term_threshold_at_1_75() -> None:
    # wS is exactly zero at the 1.75 acum threshold and positive just above.
    assert psychoacoustic_annoyance(10.0, 1.75, 0.0, 0.0).w_s == 0.0
    assert psychoacoustic_annoyance(10.0, 1.7501, 0.0, 0.0).w_s > 0.0


def test_pa_sharpness_term_formula() -> None:
    res = psychoacoustic_annoyance(20.0, 3.0, 0.0, 0.0)
    expected = (3.0 - 1.75) * 0.25 * math.log10(20.0 + 10.0)
    assert res.w_s == pytest.approx(expected)


def test_pa_fluctuation_roughness_term_formula() -> None:
    res = psychoacoustic_annoyance(8.0, 1.0, 1.2, 0.7)
    expected = (2.18 / 8.0**0.4) * (0.4 * 1.2 + 0.6 * 0.7)
    assert res.w_fr == pytest.approx(expected)


def test_pa_increases_with_each_sensation() -> None:
    base = psychoacoustic_annoyance(10.0, 2.0, 0.5, 0.5).annoyance
    assert psychoacoustic_annoyance(11.0, 2.0, 0.5, 0.5).annoyance > base
    assert psychoacoustic_annoyance(10.0, 2.5, 0.5, 0.5).annoyance > base
    assert psychoacoustic_annoyance(10.0, 2.0, 0.9, 0.5).annoyance > base
    assert psychoacoustic_annoyance(10.0, 2.0, 0.5, 0.9).annoyance > base


def test_pa_silence_is_zero() -> None:
    assert psychoacoustic_annoyance(0.0, 3.0, 1.0, 1.0).annoyance == 0.0


@pytest.mark.parametrize("bad", [-1.0, math.nan, math.inf])
def test_pa_rejects_bad_inputs(bad: float) -> None:
    with pytest.raises(ValueError):
        psychoacoustic_annoyance(bad, 2.0, 0.5, 0.3)
    with pytest.raises(ValueError):
        psychoacoustic_annoyance(10.0, bad, 0.5, 0.3)
    with pytest.raises(ValueError):
        psychoacoustic_annoyance(10.0, 2.0, bad, 0.3)
    with pytest.raises(ValueError):
        psychoacoustic_annoyance(10.0, 2.0, 0.5, bad)


def test_pa_result_fields_and_plot() -> None:
    res = psychoacoustic_annoyance(15.0, 2.2, 0.4, 0.6)
    assert isinstance(res, PsychoacousticAnnoyanceResult)
    assert res.n5 == 15.0
    assert res.sharpness == 2.2
    assert res.fluctuation_strength == 0.4
    assert res.roughness == 0.6
    ax = res.plot()
    assert ax is not None


def test_pa_from_signal_runs_and_is_positive() -> None:
    # A 1 kHz tone at ~70 dB: the convenience should return a finite, positive
    # PA consistent with its loudness (the composite mixes model families).
    fs = 48000
    t = np.arange(int(fs * 1.5)) / fs
    x = np.sin(2 * np.pi * 1000.0 * t)
    x = x / np.sqrt(np.mean(x**2)) * 2e-5 * 10 ** (70.0 / 20)
    res = psychoacoustic_annoyance_from_signal(x, fs)
    assert isinstance(res, PsychoacousticAnnoyanceResult)
    assert np.isfinite(res.annoyance)
    assert res.annoyance > 0.0
    assert res.n5 > 0.0
