#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Every signal-analysis estimate takes a ``Signal`` in place of ``(x, fs)``.

Same contract as ``signals.levels`` and ``filters``, held by
``phonometry.io._resolve``: the object supplies the rate when ``fs`` is
omitted and a disagreeing explicit one raises, a bare array still demands its
rate by name, and a calibrated Signal is analysed in pascals.

Two extra rules belong to this surface and are asserted here. These estimates
are defined on one record, so a multichannel Signal is refused by the same
complaint a 2-D array gets rather than being reduced to something. And where a
function takes two records, both may be Signals -- a measurement against a
reference -- in which case they must agree about the rate, since the whole
point of the estimate is to compare them in time.

Every equality is exact: the overload must resolve to the identical
bare-array call, never to a nearby number. The calibrated cases compare
against the *pre-scaled* array, the only reading of "in pascals" that cannot
be satisfied by accident.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from signal_contract import assert_same

from phonometry.io import Signal
from phonometry.metrology import (
    level_crossing_rate,
    peak_statistics,
    stationarity_test,
)
from phonometry.signals import (
    align_impulse_responses,
    cepstrum,
    coherent_output_spectrum,
    correlation,
    cross_spectral_density,
    echo_detection,
    envelope,
    envelope_spectrum,
    impulse_response_delay,
    lifter,
    miso_coherence,
    multitaper_psd,
    power_spectral_density,
    resample_signal,
    spectrogram,
    time_delay,
    time_synchronous_average,
    zoom_fft,
)

if TYPE_CHECKING:
    from collections.abc import Callable

FS = 8000
CAL = 3.0


def _record(seed: int = 0, n: int = 4096) -> np.ndarray:
    """A deterministic broadband record: every estimate here has something to see."""
    rng = np.random.default_rng(seed)
    return np.asarray(rng.standard_normal(n))


def _tone(frequency: float = 500.0, n: int = 4096) -> np.ndarray:
    return np.sin(2 * np.pi * frequency * np.arange(n) / FS)


# The solo consumers: one record, one rate.
SOLO = [
    (power_spectral_density, {}),
    (spectrogram, {}),
    (zoom_fft, {"f_min": 100.0, "f_max": 1000.0}),
    (cepstrum, {}),
    (lifter, {"cutoff": 0.002}),
    (echo_detection, {}),
    (envelope, {}),
    (envelope_spectrum, {}),
    (multitaper_psd, {}),
    (time_synchronous_average, {"period": 0.01}),
    (resample_signal, {"fs_new": 4000.0}),
    (stationarity_test, {}),
    (level_crossing_rate, {}),
    (peak_statistics, {}),
]
SOLO_IDS = [f.__name__ for f, _ in SOLO]

# The pairwise ones: two records that must share a rate. One shared guard
# refuses every mismatched pair, so the third field is the only thing that
# says which pair of arguments the call asked it to check.
_PAIRS_WITH_ARGUMENT_NAMES = [
    (cross_spectral_density, {}, ("x", "y")),
    (coherent_output_spectrum, {}, ("x", "y")),
    (time_delay, {}, ("x", "y")),
    (align_impulse_responses, {}, ("ir", "reference")),
]
PAIRS = [(func, kwargs) for func, kwargs, _ in _PAIRS_WITH_ARGUMENT_NAMES]
PAIR_IDS = [f.__name__ for f, _ in PAIRS]


# ---------------------------------------------------------------------------
# The rate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("func", "kwargs"), SOLO, ids=SOLO_IDS)
def test_an_uncalibrated_signal_computes_the_bare_array_result(
    func: Callable[..., object], kwargs: dict[str, float]
) -> None:
    x = _record()
    assert_same(func(Signal(x, FS), **kwargs), func(x, FS, **kwargs))


@pytest.mark.parametrize(("func", "kwargs"), SOLO, ids=SOLO_IDS)
def test_a_conflicting_rate_is_refused_a_matching_one_is_not(
    func: Callable[..., object], kwargs: dict[str, float]
) -> None:
    sig = Signal(_record(), FS)
    with pytest.raises(ValueError, match="conflicts with the Signal's own fs"):
        func(sig, FS + 1, **kwargs)
    # The same number twice is agreement, not a conflict.
    func(sig, FS, **kwargs)


@pytest.mark.parametrize(("func", "kwargs"), SOLO, ids=SOLO_IDS)
def test_a_bare_array_still_requires_fs(
    func: Callable[..., object], kwargs: dict[str, float]
) -> None:
    x = _record()
    with pytest.raises(ValueError, match="fs is required"):
        func(x, **kwargs)


@pytest.mark.parametrize(("func", "kwargs"), SOLO, ids=SOLO_IDS)
def test_a_multichannel_signal_is_refused_by_name(
    func: Callable[..., object], kwargs: dict[str, float]
) -> None:
    """These estimates are defined on one record, and say so.

    Reducing a multichannel Signal to a mono one -- picking a channel, or
    averaging -- would answer a question the caller did not ask.
    """
    block = Signal(np.stack([_record(0), _record(1)]), FS)
    with pytest.raises(ValueError, match=r"'x' must be one-dimensional"):
        func(block, **kwargs)


# ---------------------------------------------------------------------------
# The calibration
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("func", "kwargs"), SOLO, ids=SOLO_IDS)
def test_a_calibrated_signal_is_analysed_in_pascals(
    func: Callable[..., object], kwargs: dict[str, float]
) -> None:
    x = _record()
    assert_same(
        func(Signal(x, FS, calibration_factor=CAL), **kwargs),
        func(CAL * x, FS, **kwargs),
    )


# ---------------------------------------------------------------------------
# The pairs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("func", "kwargs"), PAIRS, ids=PAIR_IDS)
def test_a_pair_takes_the_rate_from_either_side(
    func: Callable[..., object], kwargs: dict[str, float]
) -> None:
    x, y = _record(0), _record(1)
    reference = func(x, y, FS, **kwargs)
    assert_same(func(Signal(x, FS), y, **kwargs), reference)
    assert_same(func(x, Signal(y, FS), **kwargs), reference)
    assert_same(func(Signal(x, FS), Signal(y, FS), **kwargs), reference)


@pytest.mark.parametrize(
    ("func", "kwargs", "names"), _PAIRS_WITH_ARGUMENT_NAMES, ids=PAIR_IDS
)
def test_two_signals_at_different_rates_are_refused(
    func: Callable[..., object], kwargs: dict[str, float], names: tuple[str, str]
) -> None:
    """Nothing here can say which rate is the truth, so neither is chosen."""
    x, y = _record(0), _record(1)
    first, second = Signal(x, FS), Signal(y, FS // 2)
    with pytest.raises(
        ValueError,
        match=rf"'{names[0]}' and '{names[1]}' are Signals recorded at different",
    ):
        func(first, second, **kwargs)


@pytest.mark.parametrize(("func", "kwargs"), PAIRS, ids=PAIR_IDS)
def test_a_pair_of_bare_arrays_still_requires_fs(
    func: Callable[..., object], kwargs: dict[str, float]
) -> None:
    x, y = _record(0), _record(1)
    with pytest.raises(ValueError, match="fs is required"):
        func(x, y, **kwargs)


@pytest.mark.parametrize(("func", "kwargs"), PAIRS, ids=PAIR_IDS)
def test_a_calibrated_pair_is_analysed_in_pascals(
    func: Callable[..., object], kwargs: dict[str, float]
) -> None:
    x, y = _record(0), _record(1)
    assert_same(
        func(
            Signal(x, FS, calibration_factor=CAL),
            Signal(y, FS, calibration_factor=CAL),
            **kwargs,
        ),
        func(CAL * x, CAL * y, FS, **kwargs),
    )


# ---------------------------------------------------------------------------
# The functions whose shape is its own decision
# ---------------------------------------------------------------------------


def test_correlation_keeps_its_rate_optional_for_bare_arrays() -> None:
    """The one estimate here that means something without a rate.

    Its lag axis is in samples when ``fs`` is 1.0, which is what
    ``correlation(a, b)`` has always returned, so the Signal contract must
    not turn that call into an error.
    """
    x, y = _record(0), _record(1)
    assert_same(correlation(x, y), correlation(x, y, 1.0))
    assert_same(correlation(Signal(x, FS), y), correlation(x, y, FS))
    at_fs, at_half = Signal(x, FS), Signal(y, FS // 2)
    with pytest.raises(
        ValueError, match=r"'x' and 'y' are Signals recorded at different"
    ):
        correlation(at_fs, at_half)


def test_impulse_response_delay_takes_the_rate_from_the_ir_or_the_reference() -> None:
    ir, ref = _record(0, 512), _record(1, 512)
    expected = impulse_response_delay(ir, FS, reference=ref)
    assert impulse_response_delay(Signal(ir, FS), reference=ref) == expected
    assert impulse_response_delay(ir, reference=Signal(ref, FS)) == expected
    with pytest.raises(ValueError, match="fs is required"):
        impulse_response_delay(ir)


def test_miso_takes_a_multichannel_signal_as_the_input_records() -> None:
    """Here the channels *are* the q input records, so 2-D is the natural form."""
    a, b, out = _record(0), _record(1), _record(2)
    block = Signal(np.stack([a, b]), FS, calibration_factor=CAL)
    assert_same(
        miso_coherence(block, Signal(out, FS, calibration_factor=CAL)),
        miso_coherence([CAL * a, CAL * b], CAL * out, FS),
    )


def test_miso_refuses_an_output_recorded_at_another_rate() -> None:
    a, b, out = _record(0), _record(1), _record(2)
    block = Signal(np.stack([a, b]), FS)
    at_half = Signal(out, FS // 2)
    with pytest.raises(
        ValueError, match=r"'inputs' and 'output' are Signals recorded at different"
    ):
        miso_coherence(block, at_half)


def test_miso_refuses_two_inputs_recorded_at_different_rates() -> None:
    """Every input, not just the first one the list happens to hold.

    Nothing downstream would catch it: two records at different rates can
    have the same number of samples, so the equal-length check passes, and
    the conditioning then cross-spectra them on a single frequency axis
    that is right for at most one of them. Before this was checked, the
    call returned an answer whose axis depended on the list order.
    """
    a, b, out = _record(0), _record(1), _record(2)
    first, second = Signal(a, FS), Signal(b, FS // 2)
    at_fs, at_half = Signal(out, FS), Signal(out, FS // 2)
    assert first.data.shape == second.data.shape  # the length check is blind
    with pytest.raises(
        ValueError, match=r"'inputs' holds Signals recorded at different rates"
    ):
        miso_coherence([first, second], at_fs)
    with pytest.raises(
        ValueError, match=r"'inputs' holds Signals recorded at different rates"
    ):
        miso_coherence([second, first], at_half)


def test_miso_takes_agreeing_input_signals() -> None:
    """The refusal above must not cost the ordinary case."""
    a, b, out = _record(0), _record(1), _record(2)
    assert_same(
        miso_coherence([Signal(a, FS), Signal(b, FS)], Signal(out, FS)),
        miso_coherence([a, b], out, FS),
    )


def test_the_arguments_behind_fs_are_keyword_only_and_required() -> None:
    """They sit behind an optional ``fs``, so Python itself enforces them.

    A ``= None`` default there would be a signature that lies: the call does
    need the value, and the reader would have to run it to find out.
    """
    sig = Signal(_record(), FS)
    for call, missing in (
        (zoom_fft, r"'f_min'.*'f_max'"),
        (lifter, r"'cutoff'"),
        (time_synchronous_average, r"'period'"),
        (resample_signal, r"'fs_new'"),
    ):
        with pytest.raises(
            TypeError,
            match=rf"{call.__name__}\(\).*required keyword-only arguments?: {missing}",
        ):
            call(sig)  # type: ignore[call-arg]


def test_the_positional_call_is_unchanged() -> None:
    """``zoom_fft(x, fs, f_min=f_min, f_max=f_max)`` still reads as it always did."""
    x = _tone()
    assert_same(
        zoom_fft(x, FS, f_min=100.0, f_max=1000.0),
        zoom_fft(x, fs=FS, f_min=100.0, f_max=1000.0),
    )
