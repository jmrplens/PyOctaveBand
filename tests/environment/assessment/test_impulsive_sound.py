#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Tests for :mod:`phonometry.environment.assessment.impulsive_sound` (ISO/PAS 1996-3:2022).

ISO/PAS 1996-3 gives no worked numeric example, so the oracle is derived by
hand from the standard's own definitions and reproduced here (clean-room from
the standard, independent of the implementation):

* A linear ``LpAF`` ramp is a controlled onset whose level difference and onset
  rate are known exactly: ``LD = Le - Ls`` (Clause 3.4) and, since the
  least-squares slope of a straight ramp equals its slope, ``OR = LD / duration``
  (Clause 3.5). A 30 dB ramp over 0.30 s therefore has ``OR = 100 dB/s`` and
  ``LD = 30 dB``, giving ``P = 3*lg(100) + 2*lg(30) = 8.9542426`` (Formula 2)
  and ``KI = 1.8*(P - 5) = 7.1176366 dB`` (Formula 3).
* A steady interval has no gradient above 10 dB/s, so it has no onset and the
  adjustment is 0 dB (Clause 6).
* A very sudden, loud ramp yields a large prominence and a highly-impulsive
  category (Clause 7).

The prominence and adjustment formulae are shared with NT ACOU 112 and covered
in more detail in ``test_impulse_prominence``; here the emphasis is the
objective measurement chain of ISO/PAS 1996-3 (level history, onset detection,
merging and the least-squares onset rate).
"""

from __future__ import annotations

import dataclasses
import importlib
import math

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
from reference_data import (
    ISO1996_3_RAMP_ADJUSTMENT,
    ISO1996_3_RAMP_LEVEL_DIFFERENCE,
    ISO1996_3_RAMP_ONSET_RATE,
    ISO1996_3_RAMP_PROMINENCE,
)

iso = importlib.import_module("phonometry.environment.assessment.impulsive_sound")


# ---------------------------------------------------------------------------
# Helpers: build a controlled LpAF level history (a piecewise-linear ramp).
# ---------------------------------------------------------------------------

DT = 0.02  # 20 ms, within the 10-25 ms range of Clause 4.


def _ramp(
    level_start: float, level_difference: float, rise: float, *, pre=0.2, post=0.3
):
    """A flat-ramp-flat ``LpAF`` history sampled at :data:`DT`.

    The onset spans exactly ``rise`` seconds so its onset rate is
    ``level_difference / rise`` and its level difference is ``level_difference``.
    Boundaries are aligned to the sample grid.
    """
    n_pre = round(pre / DT)
    n_rise = round(rise / DT)
    n_post = round(post / DT)
    level_end = level_start + level_difference
    flat_lo = np.full(n_pre, level_start)
    ramp = level_start + level_difference * (np.arange(1, n_rise + 1) / n_rise)
    flat_hi = np.full(n_post, level_end)
    return np.concatenate([flat_lo, ramp, flat_hi])


# ---------------------------------------------------------------------------
# Analytic oracle: onset detection on a known ramp (Clauses 3.4, 3.5, 5, 6).
# ---------------------------------------------------------------------------


def test_ramp_onset_rate_and_level_difference() -> None:
    # 30 dB over 0.30 s -> OR = 100 dB/s, LD = 30 dB (exact by construction).
    levels = _ramp(40.0, 30.0, 0.30)
    onsets = iso.detect_onsets(levels, DT)
    assert len(onsets) == 1
    o = onsets[0]
    assert o.level_difference == pytest.approx(
        ISO1996_3_RAMP_LEVEL_DIFFERENCE, abs=1e-9
    )
    assert o.onset_rate == pytest.approx(ISO1996_3_RAMP_ONSET_RATE, abs=1e-9)
    assert o.qualifies


def test_ramp_prominence_and_adjustment() -> None:
    # P = 3*lg(100) + 2*lg(30) = 8.9542426; KI = 1.8*(P - 5) = 7.1176366 dB.
    levels = _ramp(40.0, 30.0, 0.30)
    o = iso.detect_onsets(levels, DT)[0]
    assert o.prominence == pytest.approx(ISO1996_3_RAMP_PROMINENCE, abs=1e-9)
    assert float(iso.impulse_adjustment(o.prominence)) == pytest.approx(
        ISO1996_3_RAMP_ADJUSTMENT, abs=1e-9
    )
    # Cross-check against a fully independent hand evaluation.
    p_hand = 3.0 * math.log10(100.0) + 2.0 * math.log10(30.0)
    assert o.prominence == pytest.approx(p_hand, abs=1e-9)
    assert float(iso.impulse_adjustment(o.prominence)) == pytest.approx(
        1.8 * (p_hand - 5.0)
    )


def test_strongly_impulsive_near_cap() -> None:
    # 40 dB over 0.16 s -> OR = 250 dB/s, LD = 40 dB.
    # P = 3*lg(250) + 2*lg(40) = 10.397940; KI = 1.8*(P - 5) = 9.716291 dB.
    levels = _ramp(20.0, 40.0, 0.16)
    o = iso.detect_onsets(levels, DT)[0]
    assert o.onset_rate == pytest.approx(250.0, abs=1e-9)
    p_hand = 3.0 * math.log10(250.0) + 2.0 * math.log10(40.0)
    ki = float(iso.impulse_adjustment(o.prominence))
    assert o.prominence == pytest.approx(p_hand, abs=1e-9)
    assert ki == pytest.approx(1.8 * (p_hand - 5.0), abs=1e-9)
    assert ki > iso.HIGHLY_IMPULSIVE_LIMIT  # highly impulsive (Clause 7)
    assert iso._categorise(ki) == "highly impulsive"


def test_regular_impulsive_category() -> None:
    # 10 dB over 0.50 s -> OR = 20 dB/s, LD = 10 dB.
    # P = 3*lg(20) + 2*lg(10) = 5.903090; KI = 1.8*0.903090 = 1.625562 dB.
    levels = _ramp(40.0, 10.0, 0.50)
    o = iso.detect_onsets(levels, DT)[0]
    ki = float(iso.impulse_adjustment(o.prominence))
    p_hand = 3.0 * math.log10(20.0) + 2.0 * math.log10(10.0)
    assert o.onset_rate == pytest.approx(20.0, abs=1e-9)
    assert ki == pytest.approx(1.8 * (p_hand - 5.0), abs=1e-9)
    assert 0.0 < ki <= iso.HIGHLY_IMPULSIVE_LIMIT
    assert iso._categorise(ki) == "regular impulsive"


def test_steady_levels_have_no_onset() -> None:
    levels = np.full(60, 55.0)
    assert iso.detect_onsets(levels, DT) == ()


def test_slow_rise_below_gradient_limit_does_not_qualify() -> None:
    # 5 dB over 1.0 s -> 5 dB/s, below the 10 dB/s onset gradient limit.
    levels = _ramp(40.0, 5.0, 1.0, pre=0.2, post=0.2)
    assert iso.detect_onsets(levels, DT) == ()


# ---------------------------------------------------------------------------
# Merging of onsets within 50 ms (Clause 4, procedure d; Clause 3.3).
# ---------------------------------------------------------------------------


def test_onsets_within_50ms_are_merged() -> None:
    # Two 15 dB ramps separated by a 40 ms plateau (< 50 ms) merge into one
    # onset spanning the whole rise (LD = 30 dB).
    parts = [
        np.full(round(0.2 / DT), 40.0),
        40.0 + 15.0 * (np.arange(1, round(0.1 / DT) + 1) / round(0.1 / DT)),
        np.full(round(0.04 / DT), 55.0),
        55.0 + 15.0 * (np.arange(1, round(0.1 / DT) + 1) / round(0.1 / DT)),
        np.full(round(0.2 / DT), 70.0),
    ]
    levels = np.concatenate(parts)
    onsets = iso.detect_onsets(levels, DT)
    assert len(onsets) == 1
    assert onsets[0].level_difference == pytest.approx(30.0, abs=1e-9)


def test_onsets_beyond_50ms_are_separate() -> None:
    # The same ramps separated by a 100 ms plateau stay as two onsets.
    parts = [
        np.full(round(0.2 / DT), 40.0),
        40.0 + 15.0 * (np.arange(1, round(0.1 / DT) + 1) / round(0.1 / DT)),
        np.full(round(0.10 / DT), 55.0),
        55.0 + 15.0 * (np.arange(1, round(0.1 / DT) + 1) / round(0.1 / DT)),
        np.full(round(0.2 / DT), 70.0),
    ]
    levels = np.concatenate(parts)
    assert len(iso.detect_onsets(levels, DT)) == 2


# ---------------------------------------------------------------------------
# Pass-by variant: onset rate from the upper half of the slope (3.5, Note 1).
# ---------------------------------------------------------------------------


def test_upper_half_onset_rate_uses_top_of_slope() -> None:
    # A convex onset: shallow at the bottom, steep at the top. The upper-half
    # onset rate is larger than the whole-onset least-squares rate.
    t = np.arange(0.0, 0.5 + DT / 2, DT)
    base = np.full(round(0.2 / DT), 40.0)
    span = t[t <= 0.30]
    curve = 40.0 + 30.0 * ((span / 0.30) ** 2)  # convex rise to 70 dB
    levels = np.concatenate([base, curve[1:], np.full(round(0.2 / DT), 70.0)])
    whole = iso.detect_onsets(levels, DT, onset_rate_method="least_squares")[0]
    upper = iso.detect_onsets(levels, DT, onset_rate_method="upper_half")[0]
    assert upper.onset_rate > whole.onset_rate
    assert upper.level_difference == pytest.approx(whole.level_difference)


# ---------------------------------------------------------------------------
# Full signal chain (Clause 4): calibrated pressure signal -> adjustment.
# ---------------------------------------------------------------------------

FS = 8000


def _tone(frequency: float, duration: float, amplitude: float) -> np.ndarray:
    t = np.arange(round(duration * FS)) / FS
    return amplitude * np.sin(2.0 * np.pi * frequency * t)


def test_steady_tone_is_not_impulsive() -> None:
    signal = _tone(1000.0, 2.0, 1.0)  # 1 Pa peak, ~91 dB, steady
    with pytest.warns(iso.ImpulsiveSoundWarning):
        result = iso.impulsive_sound_adjustment(signal, FS)
    assert result.onsets == ()
    assert result.adjustment == 0.0
    assert result.category == "not impulsive"
    # ~1 kHz A-weighting is ~0 dB, so LAeq ~= 20*lg((1/sqrt2)/2e-5) = 90.97 dB.
    assert result.laeq == pytest.approx(90.97, abs=0.3)
    assert result.adjusted_laeq == pytest.approx(result.laeq)


def test_impulsive_burst_is_detected_and_highly_impulsive() -> None:
    signal = np.concatenate([np.zeros(int(0.6 * FS)), _tone(1000.0, 1.4, 1.0)])
    result = iso.impulsive_sound_adjustment(signal, FS)
    assert len(result.onsets) >= 1
    assert result.governing_onset is not None
    assert result.governing_onset.onset_rate > iso.ONSET_GRADIENT_LIMIT
    assert result.adjustment > 0.0
    assert result.category == "highly impulsive"
    assert result.adjusted_laeq == pytest.approx(result.laeq + result.adjustment)


def test_calibration_offset_leaves_adjustment_unchanged() -> None:
    # Clause 8: the adjustment is insensitive to absolute calibration.
    signal = np.concatenate([np.zeros(int(0.6 * FS)), _tone(1000.0, 1.4, 1.0)])
    base = iso.impulsive_sound_adjustment(signal, FS)
    shifted = iso.impulsive_sound_adjustment(signal, FS, calibration_offset=12.0)
    assert shifted.adjustment == pytest.approx(base.adjustment)
    assert shifted.prominence == pytest.approx(base.prominence)
    assert shifted.laeq == pytest.approx(base.laeq + 12.0)


def test_level_history_interval_within_range() -> None:
    signal = _tone(1000.0, 1.0, 1.0)
    times, levels = iso.sound_pressure_level_history(signal, FS, dt=0.02)
    assert levels.shape == times.shape
    realised = times[1] - times[0]
    lo, hi = iso.SAMPLE_INTERVAL_RANGE
    assert lo <= realised <= hi


# ---------------------------------------------------------------------------
# Validation.
# ---------------------------------------------------------------------------


def test_rejects_sample_interval_outside_range() -> None:
    sig = _tone(1000.0, 0.5, 1.0)
    with pytest.raises(ValueError, match="10-25 ms"):
        iso.sound_pressure_level_history(sig, FS, dt=0.05)


def test_rejects_non_positive_fs() -> None:
    sig = _tone(1000.0, 0.5, 1.0)
    with pytest.raises(ValueError, match="fs must be positive"):
        iso.sound_pressure_level_history(sig, 0.0)


def test_rejects_empty_signal() -> None:
    empty = np.array([])
    with pytest.raises(ValueError, match="must not be empty"):
        iso.sound_pressure_level_history(empty, FS)


def test_detect_onsets_rejects_short_input() -> None:
    with pytest.raises(ValueError, match="at least two"):
        iso.detect_onsets([40.0], DT)


def test_detect_onsets_rejects_non_positive_dt() -> None:
    with pytest.raises(ValueError, match="dt must be positive"):
        iso.detect_onsets([40.0, 41.0], 0.0)


# ---------------------------------------------------------------------------
# Plotting (EN/ES language support).
# ---------------------------------------------------------------------------


def _result() -> iso.ImpulsiveSoundResult:
    levels = _ramp(40.0, 30.0, 0.30)
    times = np.arange(levels.size) * DT
    onsets = iso.detect_onsets(levels, DT)
    p = max(o.prominence for o in onsets)
    ki = float(iso.impulse_adjustment(p))
    return iso.ImpulsiveSoundResult(
        times=times,
        levels=levels,
        dt=DT,
        onsets=onsets,
        prominence=p,
        adjustment=ki,
        category=iso._categorise(ki),
        laeq=70.0,
        adjusted_laeq=70.0 + ki,
    )


def test_plot_default_is_english() -> None:
    ax = _result().plot()
    assert ax.get_xlabel() == "Time [s]"
    assert "ISO/PAS 1996-3" in ax.get_title()
    plt.close("all")


def test_plot_spanish_labels() -> None:
    ax = _result().plot(language="es")
    assert ax.get_xlabel() == "Tiempo [s]"
    plt.close("all")


def test_plot_unknown_language_raises() -> None:
    result = _result()
    with pytest.raises(ValueError, match="Unknown language"):
        result.plot(language="xx")
    plt.close("all")


def test_the_level_history_still_unpacks_as_the_pair_it_replaced() -> None:
    """The migration this change had to survive.

    Every caller of ``sound_pressure_level_history`` wrote
    ``times, levels = ...``, so the result is iterable and yields those two
    in that order, the same way DecayCurve replaced its own tuple return.
    """
    rng = np.random.default_rng(0)
    signal = 0.01 * rng.standard_normal(FS * 2)
    result = iso.sound_pressure_level_history(signal, FS, dt=0.02)

    times, levels = result
    assert np.array_equal(times, result.times)
    assert np.array_equal(levels, result.levels)
    assert result.dt == pytest.approx(0.02)


def test_the_level_history_draws_its_own_trace() -> None:
    """The thing a bare tuple could not do, and every other result here can."""
    import matplotlib as mpl

    mpl.use("Agg")
    rng = np.random.default_rng(0)
    signal = 0.01 * rng.standard_normal(FS * 2)
    axes = iso.sound_pressure_level_history(signal, FS, dt=0.02).plot()
    assert "dB" in axes.get_ylabel()
    assert axes.get_xlabel().startswith("Time")


def test_the_history_reports_the_interval_it_actually_used() -> None:
    """``dt`` has to be the spacing of ``times``, not the interval requested.

    The step is quantised to whole samples, so a target that is not an
    integer number of them cannot be honoured exactly. Measured before this
    was fixed: at 44,1 kHz with ``dt=0.021`` the result reported 0.021 while
    its own times were 0.020997732 apart.
    """
    rng = np.random.default_rng(0)
    odd_fs = 44_100
    result = iso.sound_pressure_level_history(
        0.01 * rng.standard_normal(odd_fs), odd_fs, dt=0.021
    )
    assert result.dt == pytest.approx(float(result.times[1] - result.times[0]))
    assert result.dt != pytest.approx(0.021, abs=1e-9)


# ---------------------------------------------------------------------------
# Construction guards.
# ---------------------------------------------------------------------------


def test_the_trace_must_run_over_one_time_axis() -> None:
    """A trace off its own time axis is refused when built, not when drawn.

    The figure is the only reader of the two arrays and it draws one against
    the other, so a length disagreement surfaces there as matplotlib's "x and
    y must have same first dimension" and two bare shapes, naming neither the
    field nor the result, and only for whoever plots. An extra axis is worse:
    it passes every length check and draws in silence, one line per column,
    laying a second trace and a repeated legend entry over the onsets.
    """
    signal = np.concatenate([np.zeros(int(0.6 * FS)), _tone(1000.0, 1.4, 1.0)])
    good = iso.impulsive_sound_adjustment(signal, FS)
    per_sample = "one value per sample"
    cases = (
        ("levels", good.levels[:-1], per_sample),
        ("levels", np.append(good.levels, good.levels[-1]), per_sample),
        ("times", good.times[:-1], per_sample),
        ("levels", np.column_stack([good.levels] * 2), "must have one axis"),
        ("times", np.column_stack([good.times] * 2), "must have one axis"),
    )
    for field, value, fragment in cases:
        with pytest.raises(ValueError, match=rf"'{field}'.*{fragment}"):
            dataclasses.replace(good, **{field: value})
