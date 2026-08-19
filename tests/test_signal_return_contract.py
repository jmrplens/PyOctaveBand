#  Copyright (c) 2026. Jose Manuel Requena Plens
"""
The return side of the ``Signal`` contract: what a transform hands back.

Until now the contract ran one way. A :class:`~phonometry.io.Signal` went
into a transform and a bare array came out, so a chain of them lost the rate
and the calibration at the first step and the caller had to rebuild the
object by hand. The transforms whose output is itself a pressure record now
return a Signal when they are given one.

The whole risk of that change is one number, and it is what this file is for.

A calibrated Signal presents its samples in pascals, so the samples a
transform receives have ALREADY been scaled. If the Signal it hands back
carried the input's factor forward, every function downstream would scale
them a second time. That is not an exception anyone would notice: it is a
quiet offset of ``20 lg(factor)`` on a number that still looks computed,
through 49 sites in 35 modules, and it survives a trip to disk because
``write(sidecar=True)`` stamps the factor into the sidecar.

Measured before the rule below existed: the A-weighted level of a record
calibrated at 20.0 came back 26.02 dB low, and the full test suite passed.

So the rule is: the returned Signal carries ``calibration_factor=1.0`` when
the input carried one, and ``None`` when it did not. One means "calibrated,
and the conversion is already done", which is the only value that is right in
all three places at once: the level downstream, the axis label on
``Signal.plot()``, and the factor written to a sidecar.

The tests below take two hops on purpose. One hop cannot see this defect,
which is exactly why the existing 160 tests of the contract did not.
"""

from __future__ import annotations

import numpy as np
import pytest

from phonometry.filters import parametric_eq, weighting_filter
from phonometry.io import Signal
from phonometry.signals import leq

FS = 48000
#: A factor big enough that a second application is unmistakable: applying it
#: twice moves a level by 20 lg(20) = 26.02 dB.
CAL = 20.0


def _record(seconds: float = 1.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(int(FS * seconds)) / FS
    return 0.05 * np.sin(2.0 * np.pi * 1000.0 * t) + 0.01 * rng.standard_normal(t.size)


_RECORD = _record()

def _sections():
    """One peaking section, so ``parametric_eq`` has something to do."""
    from phonometry.filters import EQSection

    return [EQSection(filter_type="peaking", f0=1000.0, gain_db=3.0, q=1.0)]


#: The transforms whose output is a pressure record, each with the extra
#: arguments it needs. Kept here rather than derived from the source, so that
#: a function leaving the eligible set has to be removed here deliberately.
TRANSFORMS = [
    ("weighting_filter", lambda: {"curve": "A"}),
    ("parametric_eq", lambda: {"sections": _sections()}),
]
IDS = [name for name, _ in TRANSFORMS]


def _call(name, x, extra, fs=None):
    """Call the transform by name, with or without an explicit rate."""
    func = {"weighting_filter": weighting_filter, "parametric_eq": parametric_eq}[name]
    return func(x, **extra()) if fs is None else func(x, fs, **extra())


@pytest.mark.parametrize(("name", "extra"), TRANSFORMS, ids=IDS)
def test_a_second_hop_does_not_apply_the_factor_again(name, extra) -> None:
    """The defect this file exists for, asserted where it would appear.

    The level of the transformed record has to be the level of the
    transformed pascals. Carrying the input's factor onto the result makes
    this 20 lg(CAL) too high, and nothing else in the suite would notice.
    """
    from_object = leq(_call(name, Signal(_RECORD, FS, calibration_factor=CAL), extra))
    pre_scaled = leq(_call(name, Signal(CAL * _RECORD, FS), extra))
    assert from_object == pytest.approx(pre_scaled, abs=1e-9)


@pytest.mark.parametrize(("name", "extra"), TRANSFORMS, ids=IDS)
def test_the_returned_signal_says_its_conversion_is_done(name, extra) -> None:
    """One, not the input's factor, and not None either.

    None would give the same level but make ``Signal.plot()`` label a record
    of pascals as digital full scale, and ``write(sidecar=True)`` refuses it.
    """
    out = _call(name, Signal(_RECORD, FS, calibration_factor=CAL), extra)
    assert isinstance(out, Signal)
    assert out.calibration_factor == 1.0


@pytest.mark.parametrize(("name", "extra"), TRANSFORMS, ids=IDS)
def test_an_uncalibrated_record_stays_uncalibrated(name, extra) -> None:
    """The object never invents a calibration it was not given."""
    out = _call(name, Signal(_RECORD, FS), extra)
    assert isinstance(out, Signal)
    assert out.calibration_factor is None


@pytest.mark.parametrize(("name", "extra"), TRANSFORMS, ids=IDS)
def test_a_bare_array_still_comes_back_bare(name, extra) -> None:
    """The rule is conditional, so nothing that passes arrays today changes.

    Every call site inside the library resolves its input to an array before
    calling, and uses the returned value as an array.
    """
    out = _call(name, _RECORD, extra, fs=FS)
    assert isinstance(out, np.ndarray)
    assert not isinstance(out, Signal)


@pytest.mark.parametrize(("name", "extra"), TRANSFORMS, ids=IDS)
def test_the_samples_are_the_same_ones_the_bare_call_computes(name, extra) -> None:
    """Wrapping changes what comes back, never what was computed."""
    wrapped = _call(name, Signal(_RECORD, FS), extra)
    bare = _call(name, _RECORD, extra, fs=FS)
    assert np.array_equal(np.asarray(wrapped), np.asarray(bare))


@pytest.mark.parametrize(("name", "extra"), TRANSFORMS, ids=IDS)
def test_the_metadata_travels_with_the_samples(name, extra) -> None:
    """The rate and the labels are what the chain was losing."""
    labelled = Signal(
        np.stack([_RECORD, _RECORD]), FS, channel_labels=("left", "right")
    )
    out = _call(name, labelled, extra)
    assert out.fs == FS
    assert out.channel_labels == ("left", "right")
