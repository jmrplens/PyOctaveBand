#  Copyright (c) 2026. Jose Manuel Requena Plens
"""The psychoacoustic models take a ``Signal`` in place of ``(x, fs)``.

Same contract as the rest of the library, held by
``phonometry.io._resolve``. This is the surface where it matters most, and
for a reason the other packages do not share: a loudness, a sharpness, a
roughness and a fluctuation strength are all defined on *absolute* sound
pressure. Handing one an uncalibrated record does not scale its answer, it
silently reinterprets one digital unit as one pascal and returns a number
for a sound that was never measured. Reading the factor off the object the
file reader returned is how that stops being a step the caller can forget.

So the assertions here split in two.

For the level-dependent models the test compares the calibrated call
against the *pre-scaled* array, which is the only reading of "analysed in
pascals" a model that ignored the factor could not also pass, and pins that
the answer really does move when the factor does.

``tone_to_noise_ratio`` and ``prominence_ratio`` are the exception, and by
definition rather than by exemption: both compare a tone against the
masking noise inside the same spectrum, so a common factor cancels. They
are asserted from both sides.

Three of these carry a ``calibration_factor`` of their own. There the
precedence is the documented one and the opposite of the rate's: an
explicit argument wins over the object, because a caller passing one knows
something the file does not.
"""

from __future__ import annotations

import numpy as np
import pytest
from signal_contract import assert_same

from phonometry.io import Signal
from phonometry.psychoacoustics import (
    fluctuation_strength,
    fluctuation_strength_ecma,
    loudness_ecma,
    loudness_moore_glasberg,
    loudness_moore_glasberg_time,
    loudness_zwicker,
    prominence_ratio,
    psychoacoustic_annoyance_from_signal,
    roughness_ecma,
    sharpness_din,
    tonality_ecma,
    tone_to_noise_ratio,
)

FS = 48000
CAL = 4.0


def _tone(seconds: float = 1.0, seed: int = 0) -> np.ndarray:
    """A 1 kHz tone with a little noise, at about 60 dB SPL read as pascals."""
    t = np.arange(int(FS * seconds)) / FS
    rng = np.random.default_rng(seed)
    return 0.02 * np.sin(2.0 * np.pi * 1000.0 * t) + 1e-3 * rng.standard_normal(t.size)


def _modulated(seconds: float = 1.0) -> np.ndarray:
    """A 70 Hz amplitude modulation, where the roughness model reads a maximum.

    A steady tone has no roughness at all, so it would report 0.000 asper
    whatever the calibration, and could not tell a model that reads the
    factor from one that ignores it.
    """
    t = np.arange(int(FS * seconds)) / FS
    carrier = np.sin(2.0 * np.pi * 1000.0 * t)
    return 0.02 * (1.0 + np.sin(2.0 * np.pi * 70.0 * t)) * carrier


_TONE = _tone()
_MODULATED = _modulated()

# The level-dependent models: the record has to be in pascals. The short
# 1 s record is what keeps the ECMA-418-2 pair inside a couple of seconds.
LEVELLED = [
    (loudness_zwicker, {"stationary": True}),
    (sharpness_din, {}),
    (loudness_ecma, {}),
    (loudness_moore_glasberg, {}),
    (loudness_moore_glasberg_time, {}),
    (tonality_ecma, {}),
    (roughness_ecma, {}),
]
LEVELLED_IDS = [f.__name__ for f, _ in LEVELLED]

# The two that compare a tone against its own noise floor.
RATIOS = [(tone_to_noise_ratio, {}), (prominence_ratio, {})]
RATIO_IDS = [f.__name__ for f, _ in RATIOS]

ALL = LEVELLED + RATIOS
ALL_IDS = LEVELLED_IDS + RATIO_IDS


# ---------------------------------------------------------------------------
# The rate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("func", "kwargs"), ALL, ids=ALL_IDS)
def test_an_uncalibrated_signal_computes_the_bare_array_result(func, kwargs) -> None:
    assert_same(func(Signal(_TONE, FS), **kwargs), func(_TONE, FS, **kwargs))


@pytest.mark.parametrize(("func", "kwargs"), ALL, ids=ALL_IDS)
def test_a_conflicting_rate_is_refused_a_matching_one_is_not(func, kwargs) -> None:
    sig = Signal(_TONE, FS)
    with pytest.raises(ValueError, match="conflicts with the Signal's own fs"):
        func(sig, FS + 1, **kwargs)
    # The same number twice is agreement, not a conflict.
    func(sig, FS, **kwargs)


@pytest.mark.parametrize(("func", "kwargs"), ALL, ids=ALL_IDS)
def test_a_bare_array_still_requires_fs(func, kwargs) -> None:
    with pytest.raises(ValueError, match="fs is required"):
        func(_TONE, **kwargs)


# ---------------------------------------------------------------------------
# The calibration
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("func", "kwargs"), LEVELLED, ids=LEVELLED_IDS)
def test_a_calibrated_signal_is_analysed_in_pascals(func, kwargs) -> None:
    """And the answer moves with the factor, because the model is not a ratio."""
    record = _MODULATED if func is roughness_ecma else _TONE
    calibrated = func(Signal(record, FS, calibration_factor=CAL), **kwargs)
    assert_same(calibrated, func(CAL * record, FS, **kwargs))
    bare = func(record, FS, **kwargs)
    value, bare_value = _scalar(calibrated), _scalar(bare)
    assert value != pytest.approx(bare_value), (
        "a level-dependent model that answered the same either way would be "
        "ignoring the calibration"
    )


@pytest.mark.parametrize(("func", "kwargs"), RATIOS, ids=RATIO_IDS)
def test_a_tone_ratio_does_not_move_with_the_factor(func, kwargs) -> None:
    """The factor is applied and then cancels, which is not the same as ignored."""
    calibrated = func(Signal(_TONE, FS, calibration_factor=CAL), **kwargs)
    assert_same(calibrated, func(CAL * _TONE, FS, **kwargs))
    assert calibrated.ratio_db == pytest.approx(func(_TONE, FS, **kwargs).ratio_db)


def _scalar(result: object) -> float:
    """The one number a model reports, whatever it calls it."""
    if isinstance(result, (int, float)):
        return float(result)
    for name in (
        "loudness",
        "sharpness",
        "fluctuation_strength",
        "annoyance",
        "roughness",
        "tonality",
        "long_term_loudness",
    ):
        if hasattr(result, name):
            return float(np.asarray(getattr(result, name)).ravel()[0])
    raise AssertionError(f"no scalar on {type(result).__name__}")


# ---------------------------------------------------------------------------
# The models that carry a factor of their own
# ---------------------------------------------------------------------------


OWN_FACTOR = [
    (loudness_zwicker, {"stationary": True}),
    (sharpness_din, {}),
    (psychoacoustic_annoyance_from_signal, {}),
]
OWN_FACTOR_IDS = [f.__name__ for f, _ in OWN_FACTOR]


@pytest.mark.parametrize(("func", "kwargs"), OWN_FACTOR, ids=OWN_FACTOR_IDS)
def test_an_explicit_factor_wins_over_the_objects(func, kwargs) -> None:
    """The opposite precedence to the rate's, and deliberately so.

    A caller who passes a factor knows something the file does not: a
    re-calibration made after it was written, or a deliberate what-if. The
    rate is a fact of the recording and cannot be overridden that way, so
    a disagreement there raises instead.
    """
    record = _tone(seconds=2.0)
    explicit = func(
        Signal(record, FS, calibration_factor=CAL),
        calibration_factor=2.0,
        **kwargs,
    )
    assert_same(explicit, func(2.0 * record, FS, **kwargs))


def test_a_two_channel_signal_is_split_after_the_factor_is_applied() -> None:
    """The one model here that takes two ears, so the split has to see pascals.

    ``loudness_moore_glasberg_time`` divides the record into left and right
    before running either ear. Applying the factor after that split, or not
    at all, would report a binaural loudness for a sound at the wrong level.
    """
    left, right = _tone(seed=1), _tone(seed=2)
    stereo = np.stack([left, right])
    assert_same(
        loudness_moore_glasberg_time(Signal(stereo, FS, calibration_factor=CAL)),
        loudness_moore_glasberg_time(CAL * stereo, FS),
    )
    mono = loudness_moore_glasberg_time(Signal(left, FS, calibration_factor=CAL))
    binaural = loudness_moore_glasberg_time(
        Signal(np.stack([left, left]), FS, calibration_factor=CAL)
    )
    assert_same(mono, binaural)


def test_a_non_positive_explicit_factor_is_refused() -> None:
    """The convenience applies the factor itself, so it has to check it itself.

    It scales the record and hands the result to ``loudness_zwicker`` with
    no factor, so the check that function performs never sees the explicit
    one. Measured before this refusal existed: a factor of 0.0 silenced the
    signal and reported an annoyance of 0.0.
    """
    for bad in (0.0, -1.0):
        with pytest.raises(ValueError, match="calibration_factor"):
            psychoacoustic_annoyance_from_signal(_TONE, FS, calibration_factor=bad)


def test_a_bare_array_with_no_factor_is_still_read_as_pascals() -> None:
    """What the bare-array signature has always computed, unchanged."""
    assert_same(
        loudness_zwicker(_TONE, FS, stationary=True),
        loudness_zwicker(_TONE, FS, stationary=True, calibration_factor=1.0),
    )


@pytest.mark.parametrize(
    "func",
    [fluctuation_strength, fluctuation_strength_ecma],
    ids=["fluctuation_strength", "fluctuation_strength_ecma"],
)
def test_the_fluctuation_models_take_the_signal_too(func) -> None:
    """They are analysed in 2 s frames, so they get their own longer record."""
    record = _tone(seconds=2.0)
    assert_same(func(Signal(record, FS)), func(record, FS))
    assert_same(
        func(Signal(record, FS, calibration_factor=CAL)), func(CAL * record, FS)
    )
    with pytest.raises(ValueError, match="fs is required"):
        func(record)
