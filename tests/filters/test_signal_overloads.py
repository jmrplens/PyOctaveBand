#  Copyright (c) 2026. Jose Manuel Requena Plens
"""
Every public filter takes a ``phonometry.io.Signal`` in place of ``(x, fs)``.

The contract is the one ``signals.levels`` established and
``phonometry.io._resolve`` now holds for the whole library: the
object's rate wins and a disagreeing explicit one raises, a bare array still
demands its rate by name, and a calibrated Signal is filtered in pascals.

Every equality below is exact (``==`` / :func:`numpy.array_equal`, not
``approx``): the overload must resolve to the identical bare-array call, never
to a nearby number. The calibrated cases compare against the *pre-scaled*
array, which is the only reading of "in pascals" that cannot be satisfied by
accident.
"""

from __future__ import annotations

import numpy as np
import pytest

from phonometry.filters import (
    EQSection,
    LevelCalibration,
    OctaveFilterBank,
    ParametricEQ,
    TimeWeighting,
    WeightingFilter,
    linkwitz_riley,
    octave_filter,
    parametric_eq,
    time_weighting,
    weighting_filter,
)
from phonometry.io import Signal

FS = 48000
CAL = 2.0


def _tone(frequency: float = 1000.0, seconds: float = 1.0) -> np.ndarray:
    t = np.arange(int(FS * seconds)) / FS
    return np.sin(2 * np.pi * frequency * t)


# ---------------------------------------------------------------------------
# Sample rate
# ---------------------------------------------------------------------------


def test_the_signals_rate_is_used_when_none_is_given() -> None:
    x = _tone()
    assert np.array_equal(weighting_filter(Signal(x, FS)), weighting_filter(x, FS))


@pytest.mark.parametrize(
    ("func", "kwargs"),
    [
        (weighting_filter, {}),
        (time_weighting, {}),
        (linkwitz_riley, {"freq": 800.0}),
        (octave_filter, {"fraction": 3}),
        (parametric_eq, {"sections": EQSection("peaking", 1000.0, 3.0, 1.0)}),
    ],
)
def test_a_conflicting_rate_is_refused_a_matching_one_is_not(func, kwargs) -> None:
    sig = Signal(_tone(), FS)
    with pytest.raises(ValueError, match="conflicts with the Signal's own fs"):
        func(sig, FS + 1, **kwargs)
    # The same number twice is agreement, not a conflict. Outside the block:
    # this call must succeed, and inside it a failure would read as a pass.
    func(sig, FS, **kwargs)


@pytest.mark.parametrize(
    ("func", "kwargs"),
    [
        (weighting_filter, {}),
        (time_weighting, {}),
        (linkwitz_riley, {"freq": 800.0}),
        (octave_filter, {"fraction": 3}),
        (parametric_eq, {"sections": EQSection("peaking", 1000.0, 3.0, 1.0)}),
    ],
)
def test_a_bare_array_still_requires_fs(func, kwargs) -> None:
    x = _tone()
    with pytest.raises(ValueError, match="fs is required"):
        func(x, **kwargs)


# ---------------------------------------------------------------------------
# Calibration: a calibrated Signal is filtered in pascals
# ---------------------------------------------------------------------------


def test_weighting_filter_returns_pascals_for_a_calibrated_signal() -> None:
    x = _tone()
    assert np.array_equal(
        weighting_filter(Signal(x, FS, calibration_factor=CAL)),
        weighting_filter(CAL * x, FS),
    )


def test_time_weighting_envelope_is_in_pascals_squared() -> None:
    x = _tone()
    assert np.array_equal(
        time_weighting(Signal(x, FS, calibration_factor=CAL)),
        time_weighting(CAL * x, FS),
    )


def test_linkwitz_riley_splits_the_calibrated_signal() -> None:
    x = _tone()
    low, high = linkwitz_riley(Signal(x, FS, calibration_factor=CAL), freq=800.0)
    low_ref, high_ref = linkwitz_riley(CAL * x, FS, 800.0)
    assert np.array_equal(low, low_ref)
    assert np.array_equal(high, high_ref)


def test_parametric_eq_equalizes_in_pascals() -> None:
    x = _tone()
    section = EQSection("peaking", 1000.0, 3.0, 1.0)
    assert np.array_equal(
        parametric_eq(Signal(x, FS, calibration_factor=CAL), sections=section),
        parametric_eq(CAL * x, FS, section),
    )


def test_octave_filter_levels_use_the_signals_calibration() -> None:
    x = _tone()
    spl, _ = octave_filter(Signal(x, FS, calibration_factor=CAL), fraction=3)
    spl_ref, _ = octave_filter(CAL * x, FS, fraction=3)
    assert np.array_equal(spl, spl_ref)


def test_an_uncalibrated_signal_filters_exactly_like_the_bare_array() -> None:
    x = _tone()
    spl, _ = octave_filter(Signal(x, FS), fraction=3)
    spl_ref, _ = octave_filter(x, FS, fraction=3)
    assert np.array_equal(spl, spl_ref)


def test_the_explicit_bundle_wins_and_the_factor_is_not_squared() -> None:
    """The bank's own LevelCalibration is the explicit knob.

    Honouring both it and the object would multiply the samples by the
    factor and then convert them again -- a level 6 dB out for CAL=2, which
    looks entirely plausible on a plot.
    """
    x = _tone()
    spl, _ = octave_filter(
        Signal(x, FS, calibration_factor=CAL),
        fraction=3,
        calibration=LevelCalibration(factor=CAL),
    )
    spl_ref, _ = octave_filter(
        x, FS, fraction=3, calibration=LevelCalibration(factor=CAL)
    )
    assert np.array_equal(spl, spl_ref)


def test_dbfs_ignores_the_signals_calibration() -> None:
    """dBFS is relative to digital full scale whatever the object carries."""
    x = _tone()
    dbfs = LevelCalibration(dbfs=True)
    spl, _ = octave_filter(
        Signal(x, FS, calibration_factor=123.0), fraction=3, calibration=dbfs
    )
    spl_ref, _ = octave_filter(x, FS, fraction=3, calibration=dbfs)
    assert np.array_equal(spl, spl_ref)


# ---------------------------------------------------------------------------
# The pre-designed objects take the same door
# ---------------------------------------------------------------------------


def test_the_bank_refuses_a_signal_recorded_at_another_rate() -> None:
    """A pre-designed object gets its own refusal, not the fs-argument one.

    "pass one or the other" is impossible advice here: the rate came from
    the constructor and ``filter()`` has no argument to drop.
    """
    bank = OctaveFilterBank(fs=FS, fraction=3)
    foreign = Signal(_tone(), FS // 2)
    with pytest.raises(ValueError, match="designed for 48000 Hz"):
        bank.filter(foreign)


def test_the_bank_honours_the_signals_calibration() -> None:
    x = _tone()
    bank = OctaveFilterBank(fs=FS, fraction=3)
    spl, _ = bank.filter(Signal(x, FS, calibration_factor=CAL))
    spl_ref, _ = bank.filter(CAL * x)
    assert np.array_equal(spl, spl_ref)


def test_a_bank_with_its_own_factor_does_not_apply_the_objects_too() -> None:
    x = _tone()
    bank = OctaveFilterBank(fs=FS, fraction=3, calibration=LevelCalibration(factor=CAL))
    spl, _ = bank.filter(Signal(x, FS, calibration_factor=CAL))
    spl_ref, _ = bank.filter(x)
    assert np.array_equal(spl, spl_ref)


def test_the_eq_cascade_refuses_a_signal_recorded_at_another_rate() -> None:
    eq = ParametricEQ(FS, EQSection("peaking", 1000.0, 3.0, 1.0))
    foreign = Signal(_tone(), FS // 2)
    with pytest.raises(ValueError, match="designed for 48000 Hz"):
        eq.filter(foreign)


def test_the_eq_cascade_honours_the_signals_calibration() -> None:
    x = _tone()
    eq = ParametricEQ(FS, EQSection("peaking", 1000.0, 3.0, 1.0))
    assert np.array_equal(
        eq.filter(Signal(x, FS, calibration_factor=CAL)), eq.filter(CAL * x)
    )


# ---------------------------------------------------------------------------
# Multichannel and the required arguments that sit behind fs
# ---------------------------------------------------------------------------


def test_a_multichannel_signal_filters_per_channel() -> None:
    x = np.stack([_tone(1000), 0.5 * _tone(1000)])
    spl, _ = octave_filter(Signal(x, FS, calibration_factor=CAL), fraction=3)
    spl_ref, _ = octave_filter(CAL * x, FS, fraction=3)
    assert spl.shape == spl_ref.shape
    assert np.array_equal(spl, spl_ref)


def test_the_arguments_behind_fs_are_still_required() -> None:
    """``freq`` and ``sections`` default to None only so ``fs`` can precede them."""
    sig = Signal(_tone(), FS)
    with pytest.raises(ValueError, match="'freq' is required"):
        linkwitz_riley(sig)
    with pytest.raises(ValueError, match="'sections' is required"):
        parametric_eq(sig)


def test_the_positional_call_is_unchanged() -> None:
    """``linkwitz_riley(x, fs, freq)`` still reads as it always did."""
    x = _tone()
    low, high = linkwitz_riley(x, FS, 800.0)
    low_kw, high_kw = linkwitz_riley(x, fs=FS, freq=800.0)
    assert np.array_equal(low, low_kw)
    assert np.array_equal(high, high_kw)


# ---------------------------------------------------------------------------
# The two entry points to the same computation must not disagree
#
# Each pair below is a free function and the object that backs it. Before the
# object learned the contract, handing the same calibrated Signal to both
# returned levels 20*log10(factor) apart -- 26 dB for a 0.05 measurement-mic
# factor -- with nothing in either result to show which one was in pascals.
# ---------------------------------------------------------------------------

MIC = 0.05  # a plain measurement-microphone factor, Pa per digital unit


def test_the_weighting_object_agrees_with_its_function() -> None:
    x = _tone()
    sig = Signal(x, FS, calibration_factor=MIC)
    assert np.array_equal(
        WeightingFilter(FS, "A").filter(sig), weighting_filter(sig, curve="A")
    )


def test_the_time_weighting_object_agrees_with_its_function() -> None:
    x = _tone()
    sig = Signal(x, FS, calibration_factor=MIC)
    assert np.array_equal(TimeWeighting(FS).process(sig), time_weighting(sig))


def test_the_spectrogram_agrees_with_the_filter_on_the_same_bank() -> None:
    """Band levels off one bank must not depend on which method read them."""
    x = _tone(1000.0, seconds=2.0)
    sig = Signal(x, FS, calibration_factor=MIC)
    bank = OctaveFilterBank(fs=FS, fraction=3)
    spl, freq = bank.filter(sig)
    levels, _, _ = bank.spectrogram(sig, window_time=0.125)
    # Energy-average the frames back to the whole-record level.
    averaged = 10 * np.log10(np.mean(10 ** (levels / 10), axis=-1))
    peak = int(np.argmax(spl))
    assert freq[peak] == pytest.approx(1000.0, rel=0.01)
    assert averaged[peak] == pytest.approx(spl[peak], abs=0.1)


def test_the_spectrogram_returns_pascal_levels_for_a_calibrated_signal() -> None:
    x = _tone()
    bank = OctaveFilterBank(fs=FS, fraction=3)
    levels, _, _ = bank.spectrogram(Signal(x, FS, calibration_factor=MIC))
    ref, _, _ = bank.spectrogram(MIC * x)
    assert np.array_equal(levels, ref)


@pytest.mark.parametrize(
    ("build", "call", "what"),
    [
        (lambda: OctaveFilterBank(fs=FS, fraction=3), "spectrogram", "filter bank"),
        (lambda: WeightingFilter(FS, "A"), "filter", "weighting filter"),
        (lambda: TimeWeighting(FS), "process", "time weighting"),
    ],
)
def test_the_pre_designed_objects_refuse_a_foreign_rate(build, call, what) -> None:
    obj = build()
    method = getattr(obj, call)
    foreign = Signal(_tone(), FS // 2)
    with pytest.raises(ValueError, match=f"this {what} was designed for"):
        method(foreign)
