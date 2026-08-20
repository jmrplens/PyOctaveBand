#  Copyright (c) 2026. Jose Manuel Requena Plens
"""
Narrowing a ``Signal`` without losing what makes it one.

Indexing a Signal yields its samples, which is what makes the object a
drop-in for the array it stands for. The cost is that ``sig[0]`` and
``sig[:, a:b]`` hand back a bare array: the rate, the calibration, the
channel labels and the provenance are gone, and the caller who wanted one
channel of a multichannel take has to rebuild the object by hand and get the
calibration right while doing it.

:meth:`~phonometry.io.Signal.pick` and :meth:`~phonometry.io.Signal.crop`
are the two narrowings that keep it. They are named for the record they
operate on rather than for the array: MNE, which stores the same
``(channels, samples)`` beside a rate, spells them the same way.

The edges of ``crop`` follow the half-open convention of a Python slice, so
cropping a record in two at the same instant partitions it: the assertion
below is that the two halves add back up to the whole, with nothing counted
twice and nothing dropped.
"""

from __future__ import annotations

import numpy as np
import pytest

from phonometry.io import Signal

FS = 48000
CAL = 3.0


def _stereo() -> Signal:
    """A calibrated two-channel second, the right channel twice the left."""
    t = np.arange(FS) / FS
    left = 0.05 * np.sin(2.0 * np.pi * 1000.0 * t)
    return Signal(
        np.stack([left, 2.0 * left]),
        FS,
        calibration_factor=CAL,
        channel_labels=("left", "right"),
    )


def test_pick_keeps_the_rate_the_calibration_and_the_label() -> None:
    """What indexing drops is exactly what this exists to keep."""
    one = _stereo().pick(0)
    assert one.n_channels == 1
    assert one.fs == FS
    assert one.calibration_factor == CAL
    assert one.channel_labels == ("left",)


def test_pick_takes_the_channels_in_the_order_asked_for() -> None:
    """A reordering is a legitimate pick, so the labels follow the samples."""
    swapped = _stereo().pick([1, 0])
    assert swapped.channel_labels == ("right", "left")
    assert np.array_equal(np.asarray(swapped)[0], _stereo().data[1])


def test_pick_refuses_a_channel_that_is_not_there() -> None:
    """An index past the last channel is a mistake, not an empty result."""
    stereo = _stereo()
    with pytest.raises(IndexError, match="out of range"):
        stereo.pick(2)


def test_crop_cuts_at_the_seconds_asked_for() -> None:
    """The span in seconds, and the rate and the factor that come with it."""
    cropped = _stereo().crop(0.25, 0.75)
    assert cropped.n_samples == FS // 2
    assert cropped.duration == pytest.approx(0.5)
    assert cropped.fs == FS
    assert cropped.calibration_factor == CAL


def test_crop_is_half_open_so_two_halves_make_the_whole() -> None:
    """The property that makes the convention worth stating.

    Cropping ``[0, t)`` and ``[t, end)`` has to partition the record: no
    sample in both halves, none in neither.
    """
    whole = _stereo()
    first, second = whole.crop(tmax=0.4), whole.crop(tmin=0.4)
    assert first.n_samples + second.n_samples == whole.n_samples
    rejoined = np.concatenate([np.asarray(first), np.asarray(second)], axis=-1)
    assert np.array_equal(rejoined, np.asarray(whole))


def test_crop_defaults_to_the_records_own_edges() -> None:
    """Both edges omitted means the whole record, sample for sample."""
    whole = _stereo()
    assert np.array_equal(np.asarray(whole.crop()), np.asarray(whole))


def test_crop_refuses_an_empty_or_backwards_span() -> None:
    """A zero-length span, a reversed one and a negative start all raise."""
    whole = _stereo()
    with pytest.raises(ValueError, match="greater than"):
        whole.crop(0.5, 0.5)
    with pytest.raises(ValueError, match="greater than"):
        whole.crop(0.75, 0.25)
    with pytest.raises(ValueError, match="must not be negative"):
        whole.crop(-1.0, 0.5)


def test_the_two_narrowings_compose() -> None:
    """The point of both: a chain that stays a measurement."""
    part = _stereo().pick(1).crop(0.1, 0.2)
    assert part.n_channels == 1
    assert part.channel_labels == ("right",)
    assert part.n_samples == pytest.approx(FS * 0.1, abs=1)
    assert part.calibration_factor == CAL


def test_the_decibel_scale_needs_a_calibration_to_mean_anything() -> None:
    """Without a factor the samples are full-scale units, not pressures."""
    import matplotlib as mpl

    mpl.use("Agg")
    one = _stereo().pick(0)
    axes = one.plot(scale="db")
    assert "dB" in axes.get_ylabel()
    bare = Signal(np.asarray(one), FS)
    with pytest.raises(ValueError, match="needs a calibrated Signal"):
        bare.plot(scale="db")


def test_an_unknown_scale_is_refused_by_name() -> None:
    """A misspelt scale names the two that exist rather than drawing one."""
    import matplotlib as mpl

    mpl.use("Agg")
    one = _stereo().pick(0)
    with pytest.raises(ValueError, match="scale must be"):
        one.plot(scale="log")


def test_crop_excludes_the_sample_before_the_start() -> None:
    """The half-open convention, at the edge where rounding got it wrong.

    Measured before the fix: at 10 Hz, ``crop(0.01, 0.15)`` returned the
    sample at t = 0.0, which is before ``tmin``, because the edge was
    rounded to the nearest sample instead of taken as a lower bound.
    """
    ramp = Signal(np.arange(20, dtype=float), 10)
    assert np.array_equal(np.asarray(ramp.crop(0.01, 0.15)), [1.0])
    assert np.array_equal(np.asarray(ramp.crop(0.0, 0.5)), [0.0, 1.0, 2.0, 3.0, 4.0])


def test_crop_keeps_an_edge_that_lands_exactly_on_a_sample() -> None:
    """The other half of the convention: tmin is included, tmax is not."""
    ramp = Signal(np.arange(10, dtype=float), 10)
    assert np.array_equal(np.asarray(ramp.crop(0.2, 0.5)), [2.0, 3.0, 4.0])


def test_the_decibel_waveform_is_not_called_a_level() -> None:
    """It is 20 lg|p| per sample, which is a waveform in dB and not an L_p.

    An L_p is defined on a mean square over a stated time weighting. The
    axis says sound pressure in decibels, and the level trace lives on
    :class:`~phonometry.filters.TimeWeightedEnvelope` instead.
    """
    import matplotlib as mpl

    mpl.use("Agg")
    axes = _stereo().pick(0).plot(scale="db")
    assert axes.get_ylabel() == "Sound pressure [dB re 20 uPa]"
    assert "level" not in axes.get_ylabel().lower()


def test_the_decibel_axis_is_translated() -> None:
    """Every fixed string this package draws has a Spanish counterpart."""
    import matplotlib as mpl

    mpl.use("Agg")
    axes = _stereo().pick(0).plot(scale="db", language="es")
    assert axes.get_ylabel() == "Presión sonora [dB re 20 uPa]"


def test_crop_refuses_a_time_that_is_not_a_finite_number() -> None:
    """A NaN or an infinity is not an edge; it is a missing edge.

    Left to arithmetic, ``nan * fs`` compares false against everything and
    the span would silently become empty or the whole record. The refusal
    names the parameters so the caller knows which one it handed in.
    """
    whole = _stereo()
    with pytest.raises(ValueError, match="finite times in seconds"):
        whole.crop(float("nan"), 0.5)
    with pytest.raises(ValueError, match="finite times in seconds"):
        whole.crop(0.0, float("inf"))


def test_crop_refuses_a_span_too_narrow_to_hold_a_sample() -> None:
    """A span can be well formed in seconds and still hold nothing.

    At 10 Hz the samples sit 100 ms apart, so [0.11, 0.19) s falls between
    two of them: ``tmax`` is greater than ``tmin``, and yet there is no
    sample to return. Returning an empty Signal would push the surprise
    downstream, so the refusal quotes the span and the rate.
    """
    sparse = Signal(np.arange(5.0)[None, :], 10)
    with pytest.raises(ValueError, match="holds no samples at 10 Hz"):
        sparse.crop(0.11, 0.19)


def test_a_caller_supplied_label_reaches_the_lines_instead_of_crashing() -> None:
    """``label`` is an ordinary matplotlib keyword, and it arrives by kwargs.

    Measured before the fix: ``plot(label="mine")`` on a two-channel record
    raised ``TypeError: got multiple values for keyword argument 'label'``,
    because the renderer passed its generated label positionally alongside
    the caller's. The caller's wins now, and the generated one is still
    there when they say nothing.
    """
    import matplotlib as mpl

    mpl.use("Agg")
    theirs = _stereo().plot(label="mine")
    assert [line.get_label() for line in theirs.get_lines()] == ["mine", "mine"]
    ours = _stereo().plot()
    assert [line.get_label() for line in ours.get_lines()] == ["left", "right"]
