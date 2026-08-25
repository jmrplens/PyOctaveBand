#  Copyright (c) 2026. Jose Manuel Requena Plens
"""The room measurements take a ``Signal`` in place of ``(ir, fs)``.

Same contract as the rest of the library, held by
``phonometry.io._resolve``. What is particular here is that the room
surface splits in two over how much it needs the rate.

:func:`decay_curve` and :func:`room_parameters` integrate over time and
read decay times off the curve, so the rate is load-bearing and a bare
array without one is refused as everywhere else.

The three deconvolutions are the opposite: the recovery is sample-rate
agnostic and the rate is only stored on the result so a plot can label its
time axis in seconds. A bare array without one has always been a legal
call there and stays legal, so what a Signal buys is that the label
arrives by itself. The single exception is ``method="farina"``, which
rebuilds the analytic inverse filter and genuinely needs the rate; the
overload heads say so and the refusal names the reason.
"""

from __future__ import annotations

from dataclasses import fields
from typing import TYPE_CHECKING

import numpy as np
import pytest
from signal_contract import assert_same

from phonometry.io import Signal
from phonometry.room import (
    decay_curve,
    golay_impulse_response,
    golay_pair,
    impulse_response,
    mls_impulse_response,
    mls_signal,
    room_parameters,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Callable

    from phonometry.room import DecayCurve, ImpulseResponseResult, RoomAcousticsResult

    #: The two measurements that read time off the record (rate mandatory).
    _TimedMeasurement = Callable[..., DecayCurve | RoomAcousticsResult]
    #: The three deconvolutions, where the rate is only a label.
    _Deconvolution = Callable[..., ImpulseResponseResult]

FS = 8000
CAL = 3.0


def _impulse_response(seed: int = 0) -> np.ndarray:
    """A decaying broadband impulse response with a clean direct sound."""
    rng = np.random.default_rng(seed)
    t = np.arange(FS * 2) / FS
    ir = np.exp(-t * 6.0) * rng.standard_normal(t.size)
    ir[0] += 1.0
    return ir


_IR = _impulse_response()

# The two that read time off the record, so the rate is mandatory.
TIMED = [(decay_curve, {}), (room_parameters, {"limits": (500.0, 1000.0)})]
TIMED_IDS = [f.__name__ for f, _ in TIMED]


# ---------------------------------------------------------------------------
# The measurements that need the rate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("func", "kwargs"), TIMED, ids=TIMED_IDS)
def test_an_uncalibrated_signal_computes_the_bare_array_result(
    func: _TimedMeasurement, kwargs: dict[str, tuple[float, float]]
) -> None:
    assert_same(func(Signal(_IR, FS), **kwargs), func(_IR, FS, **kwargs))


@pytest.mark.parametrize(("func", "kwargs"), TIMED, ids=TIMED_IDS)
def test_a_conflicting_rate_is_refused_a_matching_one_is_not(
    func: _TimedMeasurement, kwargs: dict[str, tuple[float, float]]
) -> None:
    sig = Signal(_IR, FS)
    with pytest.raises(ValueError, match="conflicts with the Signal's own fs"):
        func(sig, FS + 1, **kwargs)
    # The same number twice is agreement, not a conflict.
    func(sig, FS, **kwargs)


@pytest.mark.parametrize(("func", "kwargs"), TIMED, ids=TIMED_IDS)
def test_a_bare_array_still_requires_fs(
    func: _TimedMeasurement, kwargs: dict[str, tuple[float, float]]
) -> None:
    with pytest.raises(ValueError, match="fs is required"):
        func(_IR, **kwargs)


@pytest.mark.parametrize(("func", "kwargs"), TIMED, ids=TIMED_IDS)
def test_a_calibrated_signal_is_scaled_and_then_cancels(
    func: _TimedMeasurement, kwargs: dict[str, tuple[float, float]]
) -> None:
    """The factor is applied and then cancels, which is not the same as ignored.

    Against the pre-scaled array the equality is exact: the samples really
    were multiplied, and this is the only reading of "analysed in pascals"
    that a function ignoring the factor could not also pass.

    Against the bare array it is exact only in the mathematics. The decay
    curve is normalised and the parameters are ratios of energies, so the
    factor divides out of every quantity -- but it divides out of a
    *backward cumulative sum*, and its tail is where the summands cancel
    against each other. Measured on this record the two curves part by
    5e-5 dB, and only 90 dB down. The tolerance below is a thousandth of a
    decibel and a microsecond of decay time, which is far under anything a
    measurement could resolve; asserting bit equality instead would be
    asserting something about the rounding, not about the contract.
    """
    calibrated = func(Signal(_IR, FS, calibration_factor=CAL), **kwargs)
    assert_same(calibrated, func(CAL * _IR, FS, **kwargs))
    bare = func(_IR, FS, **kwargs)
    for field in fields(calibrated):
        value = getattr(calibrated, field.name)
        if isinstance(value, np.ndarray) and value.dtype.kind == "f":
            assert value == pytest.approx(
                getattr(bare, field.name), rel=1e-5, abs=1e-3, nan_ok=True
            ), field.name


# ---------------------------------------------------------------------------
# The deconvolutions, where the rate is only a label
# ---------------------------------------------------------------------------


_SWEEP_KWARGS = {"method": "spectral"}
_REFERENCE = np.concatenate([[1.0], np.zeros(FS - 1)])
_RECORDED = np.convolve(_REFERENCE, _IR[:FS])[: _REFERENCE.size]

_MLS = mls_signal(12)
_MLS_RECORDED = np.tile(_MLS, 2)
_PAIR = golay_pair(6)
_GOLAY_A = np.tile(_PAIR[0], 2)
_GOLAY_B = np.tile(_PAIR[1], 2)

DECONVOLUTIONS = [
    (impulse_response, (_RECORDED, _REFERENCE), _SWEEP_KWARGS),
    (mls_impulse_response, (_MLS_RECORDED, _MLS), {}),
    (golay_impulse_response, (_GOLAY_A, _GOLAY_B), {"pair": _PAIR}),
]
DECONVOLUTION_IDS = [f.__name__ for f, _, _ in DECONVOLUTIONS]


@pytest.mark.parametrize(
    ("func", "records", "kwargs"), DECONVOLUTIONS, ids=DECONVOLUTION_IDS
)
def test_a_deconvolution_keeps_its_rate_optional(
    func: _Deconvolution,
    records: tuple[np.ndarray, np.ndarray],
    kwargs: dict[str, str | tuple[np.ndarray, np.ndarray]],
) -> None:
    """The rate labels the axis; without one the recovery is still defined."""
    first, second = records
    result = func(first, second, **kwargs)
    assert result.fs is None


@pytest.mark.parametrize(
    ("func", "records", "kwargs"), DECONVOLUTIONS, ids=DECONVOLUTION_IDS
)
def test_a_deconvolution_takes_the_label_from_the_signal(
    func: _Deconvolution,
    records: tuple[np.ndarray, np.ndarray],
    kwargs: dict[str, str | tuple[np.ndarray, np.ndarray]],
) -> None:
    first, second = records
    assert_same(
        func(Signal(first, FS), second, **kwargs),
        func(first, second, fs=FS, **kwargs),
    )


@pytest.mark.parametrize(
    ("func", "records", "kwargs"), DECONVOLUTIONS, ids=DECONVOLUTION_IDS
)
def test_a_deconvolution_refuses_a_conflicting_rate(
    func: _Deconvolution,
    records: tuple[np.ndarray, np.ndarray],
    kwargs: dict[str, str | tuple[np.ndarray, np.ndarray]],
) -> None:
    first, second = records
    sig = Signal(first, FS)
    with pytest.raises(ValueError, match="conflicts with the Signal's own fs"):
        func(sig, second, fs=FS + 1, **kwargs)


@pytest.mark.parametrize(
    ("func", "records", "kwargs"), DECONVOLUTIONS, ids=DECONVOLUTION_IDS
)
def test_a_calibrated_recording_deconvolves_in_pascals(
    func: _Deconvolution,
    records: tuple[np.ndarray, np.ndarray],
    kwargs: dict[str, str | tuple[np.ndarray, np.ndarray]],
) -> None:
    first, second = records
    assert_same(
        func(Signal(first, FS, calibration_factor=CAL), second, **kwargs),
        func(CAL * first, second, fs=FS, **kwargs),
    )


def test_golay_reads_the_second_recording_too() -> None:
    """The parametrised cases above only ever wrap the first record.

    Golay is the one here with two recordings, so a regression that looked
    at ``recorded_a`` and ignored ``recorded_b`` would pass every case above.
    """
    assert_same(
        golay_impulse_response(_GOLAY_A, Signal(_GOLAY_B, FS), pair=_PAIR),
        golay_impulse_response(_GOLAY_A, _GOLAY_B, fs=FS, pair=_PAIR),
    )
    assert_same(
        golay_impulse_response(
            _GOLAY_A, Signal(_GOLAY_B, FS, calibration_factor=CAL), pair=_PAIR
        ),
        golay_impulse_response(_GOLAY_A, CAL * _GOLAY_B, fs=FS, pair=_PAIR),
    )
    second = Signal(_GOLAY_B, FS)
    with pytest.raises(ValueError, match="conflicts with the Signal's own fs"):
        golay_impulse_response(_GOLAY_A, second, fs=FS + 1, pair=_PAIR)


def test_two_recordings_at_different_rates_are_refused() -> None:
    """Golay takes two recordings, and they have to be the same measurement."""
    first, second = Signal(_GOLAY_A, FS), Signal(_GOLAY_B, FS // 2)
    with pytest.raises(ValueError, match="recorded at different rates"):
        golay_impulse_response(first, second, pair=_PAIR)


def test_the_farina_method_still_demands_the_rate() -> None:
    """The one branch that reads the rate rather than storing it.

    It rebuilds the analytic inverse filter, whose band edges are in hertz,
    so a missing rate is refused with that reason rather than defaulted.
    """
    with pytest.raises(ValueError, match="method='farina' requires fs"):
        impulse_response(
            _RECORDED, _REFERENCE, method="farina", f_range=(100.0, 3000.0)
        )
