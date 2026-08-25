#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Every electroacoustic measurement takes a ``Signal`` in place of ``(x, fs)``.

Same contract as ``signals``, ``filters`` and ``signals.levels``, held by
``phonometry.io._resolve``: the object supplies the rate when ``fs`` is
omitted and a disagreeing explicit one raises, a bare array still demands
its rate by name, and a calibrated Signal is analysed in pascals.

Two rules belong to this surface in particular.

The distortion metrics are *ratios* of amplitudes read off one record, so
the calibration factor divides out. The test still compares the calibrated
call against the pre-scaled array rather than against the bare one: that is
the only reading of "the samples are calibrated first" which a function
that ignored the factor could not also pass.

``dynamic_range`` and ``idle_channel_noise`` are the documented exemption.
They are referenced to digital full scale rather than to 20 uPa, so a
calibrated Signal must reach them *uncalibrated*: applying the factor would
move a full-scale reading by ``20 lg(factor)`` and call it dBFS. Their test
is therefore the opposite of every other one here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from signal_contract import assert_same

from phonometry.electroacoustics import (
    coherence,
    difference_frequency_distortion,
    dynamic_intermodulation_distortion,
    dynamic_range,
    harmonic_analysis,
    harmonic_distortion,
    idle_channel_noise,
    modulation_distortion,
    sinad,
    swept_sine_distortion,
    synchronized_sweep_signal,
    thd,
    thd_plus_noise,
    total_difference_frequency_distortion,
    transfer_function,
    weighted_thd,
)
from phonometry.io import Signal

if TYPE_CHECKING:
    from collections.abc import Callable

FS = 48000
CAL = 3.0
_SECONDS = 0.5


def _tone(*harmonics: tuple[float, float], seed: int = 0) -> np.ndarray:
    """A tone complex plus a little noise, so every metric has something to see."""
    t = np.arange(int(FS * _SECONDS)) / FS
    out = np.zeros(t.size)
    for frequency, amplitude in harmonics:
        out += amplitude * np.sin(2.0 * np.pi * frequency * t)
    rng = np.random.default_rng(seed)
    return out + 1e-4 * rng.standard_normal(t.size)


_HARMONIC = _tone((1000.0, 1.0), (2000.0, 0.05), (3000.0, 0.01))
_MODULATION = _tone((250.0, 0.25), (8000.0, 1.0))
_DIFFERENCE = _tone((8000.0, 1.0), (11950.0, 1.0))
_DIM = _tone((15000.0, 1.0), (3150.0, 0.5))

# The solo consumers: one record, one rate, calibration applied.
SOLO = [
    (thd, _HARMONIC, {}),
    (harmonic_distortion, _HARMONIC, {"fundamental": 1000.0, "order": 2}),
    (thd_plus_noise, _HARMONIC, {}),
    (sinad, _HARMONIC, {}),
    (weighted_thd, _HARMONIC, {}),
    (harmonic_analysis, _HARMONIC, {}),
    (modulation_distortion, _MODULATION, {"f_low": 250.0, "f_high": 8000.0}),
    (
        difference_frequency_distortion,
        _DIFFERENCE,
        {"f1": 8000.0, "f2": 11950.0},
    ),
    (total_difference_frequency_distortion, _DIFFERENCE, {}),
    (dynamic_intermodulation_distortion, _DIM, {}),
]
SOLO_IDS = [f.__name__ for f, _, _ in SOLO]

# The exemption: referenced to digital full scale, so never calibrated.
FULL_SCALE = [(dynamic_range, _HARMONIC, {}), (idle_channel_noise, _HARMONIC, {})]
FULL_SCALE_IDS = [f.__name__ for f, _, _ in FULL_SCALE]

# The pairwise ones: two records that must share a rate.
PAIRS = [(transfer_function, {}), (coherence, {})]
PAIR_IDS = [f.__name__ for f, _ in PAIRS]


# ---------------------------------------------------------------------------
# The rate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("func", "record", "kwargs"), SOLO + FULL_SCALE, ids=SOLO_IDS + FULL_SCALE_IDS
)
def test_an_uncalibrated_signal_computes_the_bare_array_result(
    func: Callable[..., object], record: np.ndarray, kwargs: dict[str, float]
) -> None:
    assert_same(func(Signal(record, FS), **kwargs), func(record, FS, **kwargs))


@pytest.mark.parametrize(
    ("func", "record", "kwargs"), SOLO + FULL_SCALE, ids=SOLO_IDS + FULL_SCALE_IDS
)
def test_a_conflicting_rate_is_refused_a_matching_one_is_not(
    func: Callable[..., object], record: np.ndarray, kwargs: dict[str, float]
) -> None:
    sig = Signal(record, FS)
    with pytest.raises(ValueError, match="conflicts with the Signal's own fs"):
        func(sig, FS + 1, **kwargs)
    # The same number twice is agreement, not a conflict.
    func(sig, FS, **kwargs)


@pytest.mark.parametrize(
    ("func", "record", "kwargs"), SOLO + FULL_SCALE, ids=SOLO_IDS + FULL_SCALE_IDS
)
def test_a_bare_array_still_requires_fs(
    func: Callable[..., object], record: np.ndarray, kwargs: dict[str, float]
) -> None:
    with pytest.raises(ValueError, match="fs is required"):
        func(record, **kwargs)


@pytest.mark.parametrize(
    ("func", "record", "kwargs"), SOLO + FULL_SCALE, ids=SOLO_IDS + FULL_SCALE_IDS
)
def test_a_multichannel_signal_is_refused_by_name(
    func: Callable[..., object], record: np.ndarray, kwargs: dict[str, float]
) -> None:
    """These metrics are defined on one record, and say so."""
    block = Signal(np.stack([record, record]), FS)
    with pytest.raises(ValueError, match="one-dimensional"):
        func(block, **kwargs)


# ---------------------------------------------------------------------------
# The calibration
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("func", "record", "kwargs"), SOLO, ids=SOLO_IDS)
def test_a_calibrated_signal_is_analysed_in_pascals(
    func: Callable[..., object], record: np.ndarray, kwargs: dict[str, float]
) -> None:
    assert_same(
        func(Signal(record, FS, calibration_factor=CAL), **kwargs),
        func(CAL * record, FS, **kwargs),
    )


@pytest.mark.parametrize(("func", "record", "kwargs"), FULL_SCALE, ids=FULL_SCALE_IDS)
def test_a_full_scale_quantity_never_sees_the_calibration(
    func: Callable[..., object], record: np.ndarray, kwargs: dict[str, float]
) -> None:
    """The exemption, asserted from both sides.

    The calibrated call must equal the *bare* one, and must differ from the
    pre-scaled one: a reading referenced to full scale that moved with the
    factor would be reporting ``20 lg(factor)`` of offset under a
    full-scale name.
    """
    calibrated = func(Signal(record, FS, calibration_factor=CAL), **kwargs)
    assert_same(calibrated, func(record, FS, **kwargs))
    assert calibrated != pytest.approx(func(CAL * record, FS, **kwargs))


# ---------------------------------------------------------------------------
# The pairs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("func", "kwargs"), PAIRS, ids=PAIR_IDS)
def test_a_pair_takes_the_rate_from_either_side(
    func: Callable[..., object], kwargs: dict[str, float]
) -> None:
    x, y = _HARMONIC, _MODULATION
    reference = func(x, y, FS, **kwargs)
    assert_same(func(Signal(x, FS), y, **kwargs), reference)
    assert_same(func(x, Signal(y, FS), **kwargs), reference)
    assert_same(func(Signal(x, FS), Signal(y, FS), **kwargs), reference)


@pytest.mark.parametrize(("func", "kwargs"), PAIRS, ids=PAIR_IDS)
def test_two_signals_at_different_rates_are_refused(
    func: Callable[..., object], kwargs: dict[str, float]
) -> None:
    first, second = Signal(_HARMONIC, FS), Signal(_MODULATION, FS // 2)
    with pytest.raises(ValueError, match="recorded at different rates"):
        func(first, second, **kwargs)


@pytest.mark.parametrize(("func", "kwargs"), PAIRS, ids=PAIR_IDS)
def test_a_pair_of_bare_arrays_still_requires_fs(
    func: Callable[..., object], kwargs: dict[str, float]
) -> None:
    with pytest.raises(ValueError, match="fs is required"):
        func(_HARMONIC, _MODULATION, **kwargs)


@pytest.mark.parametrize(("func", "kwargs"), PAIRS, ids=PAIR_IDS)
def test_a_calibrated_pair_is_analysed_in_pascals(
    func: Callable[..., object], kwargs: dict[str, float]
) -> None:
    """Two different factors, because a shared one cancels from both of these.

    The transfer function is a ratio of the two records and the coherence is
    normalised, so calibrating both by the same amount changes nothing and a
    function that ignored the pair's calibration entirely would pass. Giving
    the two records different factors is what tells them apart.
    """
    x, y = _HARMONIC, _MODULATION
    assert_same(
        func(
            Signal(x, FS, calibration_factor=CAL),
            Signal(y, FS, calibration_factor=2.0 * CAL),
            **kwargs,
        ),
        func(CAL * x, 2.0 * CAL * y, FS, **kwargs),
    )


def test_the_transfer_function_moves_with_an_unequal_pair() -> None:
    """And the direction is the documented one: H comes out as (c_y/c_x) H."""
    x, y = _HARMONIC, _MODULATION
    plain = transfer_function(x, y, FS)
    scaled = transfer_function(x, 3.0 * y, FS)
    assert np.allclose(scaled.response, 3.0 * plain.response)
    from_objects = transfer_function(
        Signal(x, FS), Signal(y, FS, calibration_factor=3.0)
    )
    assert np.allclose(from_objects.response, scaled.response)


# ---------------------------------------------------------------------------
# The swept sine, whose excitation parameters travel with the recording
# ---------------------------------------------------------------------------


_F1, _F2 = 100.0, 5000.0
_SWEEP = synchronized_sweep_signal(FS, _F1, _F2, 1.0)
_RECORDED = _SWEEP + 0.02 * _SWEEP**2 + 0.005 * _SWEEP**3
_SWEEP_KWARGS = {"f1": _F1, "f2": _F2, "seconds": 1.0, "n_harmonics": 3}


def test_swept_sine_takes_the_signal_the_reader_returns() -> None:
    assert_same(
        swept_sine_distortion(Signal(_RECORDED, FS), **_SWEEP_KWARGS),
        swept_sine_distortion(_RECORDED, FS, **_SWEEP_KWARGS),
    )


def test_swept_sine_refuses_a_conflicting_rate() -> None:
    sig = Signal(_RECORDED, FS)
    with pytest.raises(ValueError, match="conflicts with the Signal's own fs"):
        swept_sine_distortion(sig, FS + 1, **_SWEEP_KWARGS)


def test_swept_sine_still_requires_fs_for_a_bare_array() -> None:
    with pytest.raises(ValueError, match="fs is required"):
        swept_sine_distortion(_RECORDED, **_SWEEP_KWARGS)


def test_a_calibrated_swept_sine_is_analysed_in_pascals() -> None:
    assert_same(
        swept_sine_distortion(
            Signal(_RECORDED, FS, calibration_factor=CAL), **_SWEEP_KWARGS
        ),
        swept_sine_distortion(CAL * _RECORDED, FS, **_SWEEP_KWARGS),
    )
