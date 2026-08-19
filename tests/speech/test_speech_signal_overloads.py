#  Copyright (c) 2026. Jose Manuel Requena Plens
"""
The speech measurements take a ``Signal`` in place of ``(x, fs)``.

Same contract as the rest of the library, held by
``phonometry.io._resolve``: the object supplies the rate when ``fs`` is
omitted and a disagreeing explicit one raises, a bare array still demands
its rate by name, and a calibrated Signal is analysed in pascals.

The calibration is a no-op on every number these return, and for a reason
worth writing down rather than measuring twice: STOI normalises each band
of each segment before correlating, and the STI modulation indices are
each divided by the total intensity of their own band. A factor common to
the record therefore cancels inside the definition. The absolute levels
the STI noise corrections need do not come from the samples at all -- they
arrive as ``level`` and ``ambient``, in dB.

So these tests assert the factor is *applied* (the calibrated call equals
the pre-scaled array exactly, which a function ignoring the factor could
not pass) and leave what it does to the definitions.
"""

from __future__ import annotations

import numpy as np
import pytest
from signal_contract import assert_same

from phonometry.io import Signal
from phonometry.speech import (
    sti_from_impulse_response,
    stipa,
    stipa_signal,
    stoi,
)

FS = 48000
STOI_FS = 10000
CAL = 3.0


def _impulse_response(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(FS * 2) / FS
    ir = np.exp(-t * 8.0) * rng.standard_normal(t.size)
    ir[0] += 1.0
    return ir


def _speech(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.asarray(rng.standard_normal(STOI_FS * 3))


_IR = _impulse_response()
_STIPA = stipa_signal(FS, seconds=18.0, seed=1)
_CLEAN = _speech(0)
_DEGRADED = _CLEAN + 0.3 * _speech(1)

SOLO = [(sti_from_impulse_response, _IR), (stipa, _STIPA)]
SOLO_IDS = [f.__name__ for f, _ in SOLO]


# ---------------------------------------------------------------------------
# The single-record measurements
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("func", "record"), SOLO, ids=SOLO_IDS)
def test_an_uncalibrated_signal_computes_the_bare_array_result(func, record) -> None:
    assert_same(func(Signal(record, FS)), func(record, FS))


@pytest.mark.parametrize(("func", "record"), SOLO, ids=SOLO_IDS)
def test_a_conflicting_rate_is_refused_a_matching_one_is_not(func, record) -> None:
    sig = Signal(record, FS)
    with pytest.raises(ValueError, match="conflicts with the Signal's own fs"):
        func(sig, FS + 1)
    # The same number twice is agreement, not a conflict.
    func(sig, FS)


@pytest.mark.parametrize(("func", "record"), SOLO, ids=SOLO_IDS)
def test_a_bare_array_still_requires_fs(func, record) -> None:
    with pytest.raises(ValueError, match="fs is required"):
        func(record)


@pytest.mark.parametrize(("func", "record"), SOLO, ids=SOLO_IDS)
def test_a_calibrated_signal_is_analysed_in_pascals(func, record) -> None:
    assert_same(
        func(Signal(record, FS, calibration_factor=CAL)), func(CAL * record, FS)
    )


# ---------------------------------------------------------------------------
# The pair
# ---------------------------------------------------------------------------


def test_stoi_takes_the_rate_from_either_side() -> None:
    reference = stoi(_CLEAN, _DEGRADED, STOI_FS)
    assert_same(stoi(Signal(_CLEAN, STOI_FS), _DEGRADED), reference)
    assert_same(stoi(_CLEAN, Signal(_DEGRADED, STOI_FS)), reference)
    assert_same(
        stoi(Signal(_CLEAN, STOI_FS), Signal(_DEGRADED, STOI_FS)), reference
    )


def test_stoi_refuses_two_signals_at_different_rates() -> None:
    clean = Signal(_CLEAN, STOI_FS)
    degraded = Signal(_DEGRADED, STOI_FS // 2)
    with pytest.raises(ValueError, match="recorded at different rates"):
        stoi(clean, degraded)


def test_stoi_still_requires_fs_for_a_pair_of_bare_arrays() -> None:
    with pytest.raises(ValueError, match="fs is required"):
        stoi(_CLEAN, _DEGRADED)


def test_a_calibrated_stoi_pair_is_analysed_in_pascals() -> None:
    assert_same(
        stoi(
            Signal(_CLEAN, STOI_FS, calibration_factor=CAL),
            Signal(_DEGRADED, STOI_FS, calibration_factor=CAL),
        ),
        stoi(CAL * _CLEAN, CAL * _DEGRADED, STOI_FS),
    )


# ---------------------------------------------------------------------------
# The overload heads, which must not be stricter than the function
# ---------------------------------------------------------------------------


_LEVEL = [60.0] * 7
_AMBIENT = [30.0] * 7


def test_the_noise_corrections_still_take_their_levels_positionally() -> None:
    """A Signal must not cost the caller the positional form.

    ``level`` and ``ambient`` travel together and the overload heads say
    so; the head that takes them positionally has to stay reachable when
    the rate comes from the object instead of the argument list, which
    means passing ``None`` where the rate used to go.
    """
    positional = stipa(Signal(_STIPA, FS), None, None, _LEVEL, _AMBIENT)
    assert_same(positional, stipa(_STIPA, FS, None, _LEVEL, _AMBIENT))
    assert_same(positional, stipa(Signal(_STIPA, FS), level=_LEVEL, ambient=_AMBIENT))


def test_a_reference_at_another_rate_is_refused() -> None:
    """The reference is a second take of the same signal, so it shares a rate.

    Measured before the refusal existed: a reference recorded at half the
    rate was processed on the measurement's rate and returned STI 1.000, a
    perfect score for a mismatch, instead of saying anything was wrong.
    """
    measured = Signal(_STIPA, FS)
    half = Signal(stipa_signal(FS // 2, seconds=18.0, seed=2), FS // 2)
    with pytest.raises(ValueError, match="recorded at different rates"):
        stipa(measured, reference=half)


def test_the_reference_recording_is_read_and_then_cancels() -> None:
    """Its factor is applied like any other record's, and then divides out.

    What is taken from the reference is a modulation depth, and those are
    normalised by the total intensity of their own band, so the calibrated
    call has to match both the pre-scaled one and the bare one.
    """
    reference = stipa_signal(FS, seconds=18.0, seed=2)
    calibrated = stipa(
        Signal(_STIPA, FS), reference=Signal(reference, FS, calibration_factor=CAL)
    )
    assert_same(calibrated, stipa(_STIPA, FS, CAL * reference))
    assert calibrated.sti == pytest.approx(stipa(_STIPA, FS, reference).sti)
