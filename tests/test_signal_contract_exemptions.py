#  Copyright (c) 2026. Jose Manuel Requena Plens
"""The last packages on the ``Signal`` contract, and the four exemptions.

Everything in ``phonometry`` that consumes a recording now takes the object
the reader returns in place of the ``(x, fs)`` pair. This file covers the
remainder -- broadcast, building, emission, environment, materials,
metrology, underwater and vibration -- and it is where the exemptions live,
so it is organised around them rather than around packages.

The rule is one rule: a calibrated Signal presents its samples in pascals.
The exemptions are the cases where pascals are the wrong thing to hand a
function, and each is a different reason rather than a special case:

* **Referenced to digital full scale.** LUFS, LU and dBTP count from a
  full-scale sine, not from 20 uPa. Scaling the samples would move every
  reading by ``20 lg(factor)`` and still call it by its full-scale name.
* **Not a pressure at all.** A whole-body vibration record is an
  acceleration in m/s2 and a heavy-impact record is a force in newtons.
  A digital-to-pascal factor is not a unit conversion either of them wants.
* **Derives the factor.** ``sensitivity`` is what *produces* a
  digital-to-pascal factor from a calibrator take. Applying one first
  would calibrate the calibration.

Each exemption is asserted from both sides: the calibrated call must equal
the **bare** one, and must differ from the **pre-scaled** one. Only the
second half can tell a deliberate exemption from a function that simply
forgot to look.

The rate has no exemptions. Every function here resolves it the same way,
including the ones where it is optional because it only labels an axis.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
import pytest
from signal_contract import assert_same

from phonometry.broadcast import (
    integrated_loudness,
    k_weighting,
    program_loudness,
    true_peak_level,
)
from phonometry.building import impact_force_exposure_level
from phonometry.emission import sound_intensity
from phonometry.environment import (
    impulsive_sound_adjustment,
    sound_pressure_level_history,
)
from phonometry.io import Signal
from phonometry.materials import insitu_absorption_spectrum
from phonometry.metrology import sensitivity
from phonometry.underwater import (
    pile_strike_metrics,
    single_strike_sel,
    sound_exposure_level,
    strike_sel_spectrum,
)
from phonometry.vibration import (
    apply_weighting,
    running_rms,
    spinal_response,
    vibration_dose_value,
)

if TYPE_CHECKING:
    from collections.abc import Callable

FS = 48000
CAL = 5.0


def _record(seconds: float = 1.0, seed: int = 0) -> np.ndarray:
    """A tone plus noise: every metric here has something to read."""
    t = np.arange(int(FS * seconds)) / FS
    rng = np.random.default_rng(seed)
    return 0.1 * np.sin(2.0 * np.pi * 1000.0 * t) + 0.01 * rng.standard_normal(t.size)


_RECORD = _record()
_SHORT = _record(seconds=0.05)


# ---------------------------------------------------------------------------
# The rule: a calibrated Signal is analysed in pascals
# ---------------------------------------------------------------------------

IN_PASCALS = [
    (sound_exposure_level, _RECORD, {}),
    (single_strike_sel, _RECORD, {}),
    (strike_sel_spectrum, _RECORD, {}),
    (sound_pressure_level_history, _RECORD, {}),
]
IN_PASCALS_IDS = [f.__name__ for f, _, _ in IN_PASCALS]


@pytest.mark.parametrize(("func", "record", "kwargs"), IN_PASCALS, ids=IN_PASCALS_IDS)
def test_a_calibrated_signal_is_analysed_in_pascals(
    func: Callable[..., object], record: np.ndarray, kwargs: dict[str, object]
) -> None:
    """A calibrated Signal reads the same as the samples already in pascals."""
    assert_same(
        func(Signal(record, FS, calibration_factor=CAL), **kwargs),
        func(CAL * record, FS, **kwargs),
    )


@pytest.mark.parametrize(("func", "record", "kwargs"), IN_PASCALS, ids=IN_PASCALS_IDS)
def test_an_uncalibrated_signal_computes_the_bare_array_result(
    func: Callable[..., object], record: np.ndarray, kwargs: dict[str, object]
) -> None:
    """With no factor to apply, the object and the bare array agree."""
    assert_same(func(Signal(record, FS), **kwargs), func(record, FS, **kwargs))


def test_the_two_microphone_probe_takes_the_rate_from_either_side() -> None:
    """An intensity probe is a pair, and the two records are one measurement."""
    p1, p2 = _record(seed=1), _record(seed=2)
    reference = sound_intensity(p1, p2, FS, spacing=0.012)
    assert_same(sound_intensity(Signal(p1, FS), p2, spacing=0.012), reference)
    assert_same(sound_intensity(p1, Signal(p2, FS), spacing=0.012), reference)
    first, second = Signal(p1, FS), Signal(p2, FS // 2)
    with pytest.raises(
        ValueError, match="'p1' and 'p2' are Signals recorded at different rates"
    ):
        sound_intensity(first, second, spacing=0.012)
    with pytest.raises(ValueError, match=r"fs is required when 'p1' is a bare array"):
        sound_intensity(p1, p2, spacing=0.012)


def test_the_probe_intensity_goes_as_the_product_of_the_two_pressures() -> None:
    """Both microphones calibrated: the intensity carries the factor squared."""
    p1, p2 = _record(seed=1), _record(seed=2)
    assert_same(
        sound_intensity(
            Signal(p1, FS, calibration_factor=CAL),
            Signal(p2, FS, calibration_factor=CAL),
            spacing=0.012,
        ),
        sound_intensity(CAL * p1, CAL * p2, FS, spacing=0.012),
    )


def test_a_ratio_of_two_records_still_applies_both_factors() -> None:
    """ "Cancels" has to mean applied and divided out, not skipped.

    The in-situ absorption is the ratio of two measured impulse responses,
    so a factor the two share leaves it where it was. That is only true
    because both are scaled: the third assertion is the one that fails if
    the samples never see the factor at all.
    """
    incident = np.zeros(2048)
    incident[100] = 1.0
    reflected = 0.4 * np.roll(incident, 96)
    both = insitu_absorption_spectrum(
        Signal(incident, FS, calibration_factor=CAL),
        Signal(reflected, FS, calibration_factor=CAL),
    )
    assert_same(both, insitu_absorption_spectrum(CAL * incident, CAL * reflected, FS))
    bare = insitu_absorption_spectrum(incident, reflected, FS)
    assert np.allclose(both.absorption, bare.absorption, equal_nan=True)
    one_only = insitu_absorption_spectrum(
        Signal(incident, FS, calibration_factor=CAL), reflected
    )
    assert not np.allclose(one_only.absorption, bare.absorption, equal_nan=True)


def test_a_delegating_chain_applies_the_factor_once() -> None:
    """The failure mode a function that calls another one invites.

    ``impulsive_sound_adjustment`` resolves the Signal and then hands the
    samples to ``sound_pressure_level_history``, which resolves too. If the
    inner call saw the object rather than the resolved array, the factor
    would land twice and every reported level would be 20 lg(factor) too
    high. The third comparison is what would catch that.
    """
    rng = np.random.default_rng(0)
    record = 0.01 * rng.standard_normal(FS * 2)
    record[FS : FS + 2000] += 0.5 * np.exp(-np.arange(2000) / 300.0)
    once = impulsive_sound_adjustment(Signal(record, FS, calibration_factor=CAL))
    assert once.laeq == pytest.approx(impulsive_sound_adjustment(CAL * record, FS).laeq)
    assert once.laeq != pytest.approx(
        impulsive_sound_adjustment(CAL**2 * record, FS).laeq
    )


def test_a_bundling_chain_applies_the_factor_once() -> None:
    """The other shape of the same risk, on the underwater side.

    ``pile_strike_metrics`` resolves the object and then feeds four other
    metrics from the samples it got. Passing the object on instead of the
    resolved array would square the factor in every one of them.
    """
    strike = 0.1 * np.exp(-np.arange(4800) / 500.0) * _record(seconds=0.1)[:4800]
    once = pile_strike_metrics(Signal(strike, FS, calibration_factor=CAL))
    assert once.single_strike_sel == pytest.approx(
        pile_strike_metrics(CAL * strike, FS).single_strike_sel
    )
    assert once.single_strike_sel != pytest.approx(
        pile_strike_metrics(CAL**2 * strike, FS).single_strike_sel
    )


# ---------------------------------------------------------------------------
# Exemption 1: referenced to digital full scale
# ---------------------------------------------------------------------------

FULL_SCALE = [
    (k_weighting, {}),
    (true_peak_level, {}),
    (integrated_loudness, {}),
    (program_loudness, {}),
]
FULL_SCALE_IDS = [f.__name__ for f, _ in FULL_SCALE]


@pytest.mark.parametrize(("func", "kwargs"), FULL_SCALE, ids=FULL_SCALE_IDS)
def test_a_full_scale_reading_never_sees_the_calibration(
    func: Callable[..., object], kwargs: dict[str, object]
) -> None:
    """LUFS and dBTP are counted from full scale, so pascals would shift them."""
    calibrated = func(Signal(_RECORD, FS, calibration_factor=CAL), **kwargs)
    assert_same(calibrated, func(_RECORD, FS, **kwargs))
    pre_scaled = func(CAL * _RECORD, FS, **kwargs)
    with pytest.raises(AssertionError, match=r"result.*differs"):
        assert_same(calibrated, pre_scaled)


@pytest.mark.parametrize(("func", "kwargs"), FULL_SCALE, ids=FULL_SCALE_IDS)
def test_a_full_scale_reading_still_resolves_the_rate(
    func: Callable[..., object], kwargs: dict[str, object]
) -> None:
    """Exempt from the calibration, not from the rate: it still takes one."""
    sig = Signal(_RECORD, FS)
    assert_same(func(sig, **kwargs), func(_RECORD, FS, **kwargs))
    with pytest.raises(ValueError, match=r"fs=\d+ conflicts with the Signal's own fs"):
        func(sig, FS + 1, **kwargs)
    with pytest.raises(ValueError, match=r"fs is required when 'x' is a bare array"):
        func(_RECORD, **kwargs)


# ---------------------------------------------------------------------------
# Exemption 2: the quantity is not a pressure
# ---------------------------------------------------------------------------

NOT_PRESSURE = [
    (apply_weighting, {"name": "Wk"}),
    (running_rms, {}),
    (vibration_dose_value, {}),
    (spinal_response, {}),
    (impact_force_exposure_level, {}),
]
NOT_PRESSURE_IDS = [f.__name__ for f, _ in NOT_PRESSURE]


@pytest.mark.parametrize(("func", "kwargs"), NOT_PRESSURE, ids=NOT_PRESSURE_IDS)
def test_a_non_pressure_record_never_sees_the_calibration(
    func: Callable[..., object], kwargs: dict[str, str]
) -> None:
    """An acceleration in m/s2 and a force in N are not pascals waiting to be."""
    calibrated = func(Signal(_RECORD, FS, calibration_factor=CAL), **kwargs)
    assert_same(calibrated, func(_RECORD, FS, **kwargs))
    pre_scaled = func(CAL * _RECORD, FS, **kwargs)
    with pytest.raises(AssertionError, match=r"result.*differs"):
        assert_same(calibrated, pre_scaled)


@pytest.mark.parametrize(("func", "kwargs"), NOT_PRESSURE, ids=NOT_PRESSURE_IDS)
def test_a_non_pressure_record_still_resolves_the_rate(
    func: Callable[..., object], kwargs: dict[str, str]
) -> None:
    """An acceleration is not a pressure, and it still needs a sample rate."""
    sig = Signal(_RECORD, FS)
    assert_same(func(sig, **kwargs), func(_RECORD, FS, **kwargs))
    with pytest.raises(
        ValueError, match=r"(fs|sample_rate)=\d+ conflicts with the Signal's own fs"
    ):
        func(sig, FS + 1, **kwargs)
    with pytest.raises(
        ValueError, match=r"(fs|sample_rate) is required when '\w+' is a bare array"
    ):
        func(_RECORD, **kwargs)


# ---------------------------------------------------------------------------
# Exemption 3: the function derives the factor
# ---------------------------------------------------------------------------


def test_the_calibrator_take_is_never_pre_calibrated() -> None:
    """``sensitivity`` produces a factor; folding one in would square it."""
    tone = _record(seconds=3.0)
    calibrated = sensitivity(Signal(tone, FS, calibration_factor=CAL))
    assert calibrated == pytest.approx(sensitivity(tone, fs=FS))
    assert calibrated != pytest.approx(sensitivity(CAL * tone, fs=FS))


def test_the_calibrator_take_gets_its_validation_from_the_object() -> None:
    """Its rate is optional, and a Signal is what makes the check possible.

    Without a rate the stability validation of IEC 60942 is skipped, which
    has always been a legal call and stays one. Reading the rate off the
    object is what turns a read take into a validated one for free, and the
    warning below is the evidence that the validation really did run: the
    same short take passed as a bare array reaches no check at all.
    """
    short = _record(seconds=0.5)
    take = Signal(short, FS)
    with pytest.warns(UserWarning, match="too short to validate"):
        from_object = sensitivity(take)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        unvalidated = sensitivity(short)
    assert from_object == pytest.approx(unvalidated)
    with pytest.raises(ValueError, match=r"fs=\d+ conflicts with the Signal's own fs"):
        sensitivity(take, fs=FS + 1)


def test_a_calibrator_take_without_a_rate_is_still_a_legal_call() -> None:
    """The function that derives the factor cannot be asked to apply one."""
    assert sensitivity(_SHORT) > 0.0
