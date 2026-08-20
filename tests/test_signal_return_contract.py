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
    """A seeded 1 kHz tone with a little noise, so levels are reproducible."""
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


# ---------------------------------------------------------------------------
# The squared one, which cannot be a Signal
# ---------------------------------------------------------------------------


def test_a_mean_square_envelope_is_not_a_signal() -> None:
    """The rule that decides what a Signal may hold, asserted where it bites.

    ``time_weighting`` returns the running mean SQUARE. Measured, its output
    scales as the square of a gain on the record, so it is in pascals
    squared: handing it back as a Signal would label a Pa2 quantity as a
    pressure record, which is the lie the calibration contract exists to
    prevent. It gets a result object of its own instead.
    """
    from phonometry.filters import TimeWeightedEnvelope, time_weighting

    plain = np.asarray(time_weighting(Signal(_RECORD, FS), mode="fast"))
    doubled = np.asarray(time_weighting(Signal(2.0 * _RECORD, FS), mode="fast"))
    finite = np.isfinite(plain) & (plain != 0)
    assert np.allclose(doubled[finite] / plain[finite], 4.0)

    out = time_weighting(Signal(_RECORD, FS), mode="fast")
    assert isinstance(out, TimeWeightedEnvelope)
    assert not isinstance(out, Signal)


def test_the_envelope_stands_in_for_the_array_it_replaced() -> None:
    """Every caller that only wanted the numbers must not notice the change."""
    from phonometry.filters import time_weighting

    out = time_weighting(Signal(_RECORD, FS), mode="fast")
    bare = time_weighting(_RECORD, FS, mode="fast")
    assert np.array_equal(np.asarray(out), bare)
    assert out.shape == bare.shape
    assert out.ndim == bare.ndim
    assert out.size == bare.size
    assert out.dtype == bare.dtype
    assert len(out) == len(bare)
    assert np.array_equal(out[..., -1], bare[..., -1])


def test_the_envelope_knows_whether_its_level_means_anything() -> None:
    """A level needs pascals; without them the trace is full-scale squared."""
    from phonometry.filters import time_weighting

    assert time_weighting(
        Signal(_RECORD, FS, calibration_factor=CAL), mode="fast"
    ).calibrated
    assert not time_weighting(Signal(_RECORD, FS), mode="fast").calibrated


def test_a_bare_array_still_gets_the_bare_envelope() -> None:
    """The six places that divide the envelope in place keep working."""
    from phonometry.filters import time_weighting

    out = time_weighting(_RECORD, FS, mode="fast")
    assert isinstance(out, np.ndarray)
    out /= float(np.max(out))  # in place, which a frozen result cannot do
    assert np.isclose(np.max(out), 1.0)


def test_the_object_forms_of_the_filters_wrap_too() -> None:
    """The class API and the function must not disagree about a recording.

    ``WeightingFilter.filter`` and ``ParametricEQ.filter`` are what the
    functions delegate to, so a Signal handed to either has to come back the
    same way. Before this they dropped the rate on the class path only.
    """
    from phonometry.filters import ParametricEQ, WeightingFilter

    sig = Signal(_RECORD, FS, calibration_factor=CAL)
    weighted = WeightingFilter(FS, "A").filter(sig)
    assert isinstance(weighted, Signal)
    assert weighted.calibration_factor == 1.0
    assert np.array_equal(
        np.asarray(weighted), np.asarray(weighting_filter(sig, curve="A"))
    )

    equalized = ParametricEQ(FS, _sections()).filter(sig)
    assert isinstance(equalized, Signal)
    assert equalized.calibration_factor == 1.0

    assert isinstance(WeightingFilter(FS, "A").filter(_RECORD), np.ndarray)


def test_the_envelope_reaches_ndarray_methods_through_asarray() -> None:
    """The boundary of what the result stands in for, stated and pinned.

    It forwards the array protocol and the geometry attributes, which is the
    same boundary :class:`~phonometry.room.ImpulseResponseResult` draws, so
    ``.mean()`` and the rest of the ndarray methods are reached through
    :func:`numpy.asarray` rather than off the object.
    """
    from phonometry.filters import time_weighting

    out = time_weighting(Signal(_RECORD, FS), mode="fast")
    assert not hasattr(out, "mean")
    assert np.asarray(out).mean() == pytest.approx(
        time_weighting(_RECORD, FS, mode="fast").mean()
    )


def test_the_envelope_plot_says_which_quantity_it_drew() -> None:
    """Both branches of the renderer, and neither claims to be an L_pAF.

    Time weighting alone is not A-weighting: the level is ``L_pAF`` only
    when the record was A-weighted first, and an uncalibrated record has no
    reference to count decibels from at all.
    """
    import matplotlib as mpl

    mpl.use("Agg")
    from phonometry.filters import time_weighting

    calibrated = time_weighting(
        Signal(_RECORD, FS, calibration_factor=CAL), mode="fast"
    ).plot()
    assert calibrated.get_ylabel() == "Sound pressure level [dB re 20 uPa]"
    assert "Fast" in calibrated.get_title()

    plain = time_weighting(Signal(_RECORD, FS), mode="slow").plot()
    assert "FS" in plain.get_ylabel()
    assert "dB" not in plain.get_ylabel()


def test_the_envelope_plot_labels_each_channel() -> None:
    """The multichannel branch, which the single-channel cases never reach."""
    import matplotlib as mpl

    mpl.use("Agg")
    from phonometry.filters import time_weighting

    stereo = Signal(np.stack([_RECORD, 2.0 * _RECORD]), FS, calibration_factor=CAL)
    axes = time_weighting(stereo, mode="fast").plot()
    assert len(axes.get_lines()) == 2
    assert axes.get_legend() is not None


def test_the_envelope_plot_is_translated() -> None:
    """Every fixed string this renderer draws has a Spanish counterpart."""
    import matplotlib as mpl

    mpl.use("Agg")
    from phonometry.filters import time_weighting

    axes = time_weighting(
        Signal(_RECORD, FS, calibration_factor=CAL), mode="fast"
    ).plot(language="es")
    assert axes.get_ylabel().startswith("Nivel de presión sonora")
    assert axes.get_xlabel() == "Tiempo [s]"


def test_the_band_waveforms_come_back_as_signals_only_when_the_bank_read_the_factor() -> (
    None
):
    """Both sides of the gate on the filter bank's band output.

    ``sigbands=True`` hands back one waveform per band, and each one is a
    pressure record in its own right. It is a Signal only when the bank
    filtered pascals: that is, when it took the object's calibration rather
    than one of its own or a dBFS reading. A dBFS bank filtered digital
    full-scale units, so returning those as a Signal would put a pascal
    label on a number that is not one.

    The factor on the returned band is 1.0 for the reason this whole file
    exists: the samples are already in pascals, so a factor that is not 1
    would be applied a second time downstream.
    """
    from phonometry.filters import LevelCalibration, octave_filter

    record = Signal(_RECORD, FS, calibration_factor=CAL)
    _spl, _freq, bands = octave_filter(record, sigbands=True, detrend=False)
    assert all(isinstance(band, Signal) for band in bands)
    assert {band.fs for band in bands} == {FS}
    assert {band.calibration_factor for band in bands} == {1.0}

    _spl, _freq, digital = octave_filter(
        record,
        sigbands=True,
        detrend=False,
        calibration=LevelCalibration(dbfs=True),
    )
    assert not any(isinstance(band, Signal) for band in digital)
    assert all(isinstance(band, np.ndarray) for band in digital)


def test_an_empty_block_keeps_the_carried_state_and_the_wrapper() -> None:
    """The block integrator's empty-input path, on both input shapes.

    A stream can hand a block-wise integrator an empty block, and that must
    neither reset the state it carries nor change the type it returns: the
    block after it has to continue exactly where the one before left off.
    Checked against a run with no empty block in the middle, which is the
    only thing that can tell a kept state from a forgotten one. The empty
    envelope itself comes back wrapped for a Signal and bare for an array,
    like every other call.
    """
    from phonometry.filters import TimeWeightedEnvelope, TimeWeighting

    first, second = _RECORD[:2400], _RECORD[2400:4800]

    straight = TimeWeighting(FS, mode="fast")
    straight.process(Signal(first, FS))
    continued = np.asarray(straight.process(Signal(second, FS)))

    interrupted = TimeWeighting(FS, mode="fast")
    interrupted.process(Signal(first, FS))
    empty = interrupted.process(Signal(np.zeros((1, 0)), FS))
    assert isinstance(empty, TimeWeightedEnvelope)
    assert np.asarray(empty).shape[-1] == 0
    assert np.array_equal(
        np.asarray(interrupted.process(Signal(second, FS))), continued
    )

    bare = TimeWeighting(FS, mode="fast")
    bare.process(first)
    assert isinstance(bare.process(np.zeros(0)), np.ndarray)


def test_the_envelope_plot_takes_a_caller_supplied_label() -> None:
    """The same keyword collision as the waveform renderer, same resolution."""
    import matplotlib as mpl

    mpl.use("Agg")
    from phonometry.filters import time_weighting

    stereo = Signal(np.stack([_RECORD, 2.0 * _RECORD]), FS, calibration_factor=CAL)
    axes = time_weighting(stereo, mode="fast").plot(label="mine")
    assert [line.get_label() for line in axes.get_lines()] == ["mine", "mine"]
