#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Resolving an ``(x, fs)`` argument pair that may arrive as a ``Signal``.

A function that consumes a recording should take a
:class:`phonometry.io.Signal` in place of the bare ``(x, fs)`` pair: the
object read from a measurement file already knows its sample rate and,
when calibrated, its digital-to-pascal factor, so asking the caller to
repeat either is asking for a transcription error. The helpers here are
that contract, in one place, so that the surfaces adopting it cannot
drift apart. Every function in the library that consumes a recording is
on it.

They live beside :class:`Signal` rather than in ``_internal`` because that
tree is a leaf every package may import: a resolver there would have to
import ``phonometry.io``, and importing a submodule runs its package
``__init__``, so ``_internal`` would sit at the wrong end of an import
cycle the moment ``io`` reached back for a toolbox helper.

The rules, identical everywhere:

- **Sample rate.** A :class:`~phonometry.io.Signal` brings its own; an
  explicit one that disagrees is refused rather than arbitrated, because
  the rate is a fact of the recording and silently trusting either side
  mis-times every filter downstream. A bare array knows nothing about
  time, so there the argument stays mandatory -- except where the rate is
  only a label for the result's time axis and the computation never reads
  it, which is :func:`resolve_optional_fs` and stays callable without
  one.
- **Samples.** What is not a signal is refused here, by the name the
  entry point gave it, and as a ``ValueError`` because that is what the
  entry points document: ``None``, a string, a dict, a ragged list of
  lists, and a ``NaN`` or an infinity among the samples. The last of
  those is a refusal rather than a passing-on because a single
  non-finite sample survives every filter and poisons every band
  computed from it, so a call that lost one sample comes back with a
  whole spectrum of ``NaN`` and nothing said. See
  :func:`phonometry._internal.utils._typesignal`, which is where all of
  it happens.
- **Calibration.** A calibrated Signal presents its samples in pascals.
  One rule rather than a per-function taxonomy: it is what keeps every
  surface on this contract talking about the same signal, instead of one
  reading pascals and its neighbour digital units. What that buys
  depends on what the call returns -- a waveform in pascals, a squared
  envelope in Pa2, a level in dB SPL re 20 uPa -- but the samples that
  enter the computation are the same either way. Bare arrays are
  untouched: they are taken to be in whatever unit they always were.
- **Exemptions**, taken with ``calibrate=False``, each said out loud in
  the docstring of the surface that takes it. Anything referenced to
  digital full scale rather than to 20 uPa (a ``dbfs=True`` path, and
  the EBU R 128 / ITU-R BS.1770 loudness family when it adopts this
  contract) must not see pascals, or it reports ``20*log10(factor)``
  of offset under a full-scale name. Neither must a function that
  *derives* the factor from a calibrator take, nor one whose quantity is
  not a pressure at all (acceleration in m/s2, force in N).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import overload

import numpy as np

from .._internal.utils import _typesignal
from ._signal import Signal

#: What a function that consumes a recording accepts for its signal
#: argument. ``Sequence[float]`` rather than ``list[float]`` so that a
#: surface already typed for tuples does not have to narrow its signature
#: to adopt the contract; ``str`` is a ``Sequence[str]`` and so stays out.
SignalInput = Signal | Sequence[float] | np.ndarray

# The rate type parameter below is value-restricted to ``(int, float)``
# rather than bound to ``float`` so that a module typing its rate as ``int``
# gets an ``int`` back: a plain ``-> float`` would leak a float into every
# ``int``-typed sample count downstream, and a pair of ``int`` / ``float``
# overloads overlap (``int`` is assignable to ``float``) with return types
# mypy rejects as incompatible.


def resolve_optional_fs[Rate: (int, float)](
    x: SignalInput,
    fs: Rate | None,
    *,
    rate: str = "fs",
) -> Rate | None:
    """Resolve a rate that the call can also do without.

    The sibling of :func:`resolve_fs` for the surfaces where the rate is
    only a label: a deconvolution is sample-rate agnostic, and the rate
    it is handed goes onto the result so a plot can put seconds on its
    time axis. A bare array without one is therefore legal, and stays
    legal. What a Signal buys here is that the label arrives by itself
    instead of being retyped, and the disagreement is still refused for
    the same reason as everywhere else.

    :param x: The signal argument, a Signal or a bare array.
    :param fs: The explicitly passed rate, or ``None``.
    :param rate: Name of the rate parameter, for the error message.
    :return: The resolved sample rate, or ``None`` if there is none.
    :raises ValueError: If a Signal is given an explicit rate that
        disagrees with its own.
    """
    if isinstance(x, Signal):
        if fs is not None and fs != x.fs:
            msg = (
                f"{rate}={fs} conflicts with the Signal's own fs={x.fs}; "
                "pass one or the other, not a disagreement"
            )
            raise ValueError(msg)
        return x.fs
    return fs


def resolve_fs[Rate: (int, float)](
    x: SignalInput,
    fs: Rate | None,
    *,
    name: str = "x",
    rate: str = "fs",
) -> Rate:
    """Resolve the sample rate from the argument or from the Signal itself.

    :param x: The signal argument, a Signal or a bare array.
    :param fs: The explicitly passed rate, or ``None``.
    :param name: Name of the signal parameter, for the error message.
    :param rate: Name of the rate parameter, for the error message.
    :return: The resolved sample rate.
    :raises ValueError: If a Signal is given an explicit rate that
        disagrees with its own, or a bare array is given none.
    """
    resolved = resolve_optional_fs(x, fs, rate=rate)
    if resolved is None:
        msg = f"{rate} is required when '{name}' is a bare array"
        raise ValueError(msg)
    return resolved


def refuse_foreign_rate(x: SignalInput, fs: float, what: str) -> None:
    """Refuse a Signal recorded at a rate this object was not designed for.

    :func:`resolve_fs` tells the caller to pass one rate or the other,
    which is sound advice where there is an ``fs`` argument to drop. On a
    pre-designed object -- a filter bank, an EQ cascade -- the rate came
    from the constructor and the call has no argument to arbitrate, so
    the advice would be impossible to follow.

    :param x: The signal argument, a Signal or a bare array.
    :param fs: The rate the object was designed for.
    :param what: What the object is, for the message ("filter bank").
    :raises ValueError: If *x* is a Signal at a different rate.
    """
    if isinstance(x, Signal) and x.fs != fs:
        msg = (
            f"the Signal was recorded at {x.fs} Hz but this {what} was "
            f"designed for {fs:g} Hz; resample the Signal, or build a "
            f"{what} for its own rate"
        )
        raise ValueError(msg)


def resolve_calibration(x: SignalInput, calibration_factor: float | None) -> float:
    """Resolve the digital-to-pascal factor by the documented precedence.

    For the functions that carry a calibration knob of their own: an
    explicit argument always wins -- the caller knows more than the
    object (a re-calibration after the file was written, a deliberate
    what-if). Otherwise a calibrated :class:`~phonometry.io.Signal`
    supplies the factor it carries. Otherwise 1.0: digital units
    straight through, which is what the bare-array signatures have
    always computed when no factor was given.

    :param x: The signal argument, a Signal or a bare array.
    :param calibration_factor: The explicitly passed factor, or ``None``.
    :return: The factor to apply.
    """
    if calibration_factor is not None:
        return calibration_factor
    if isinstance(x, Signal) and x.calibration_factor is not None:
        return x.calibration_factor
    return 1.0


def _scaled(samples: np.ndarray, factor: float, name: str) -> np.ndarray:
    """Scale *samples* by *factor*, refusing a product that leaves the doubles.

    Both operands are finite by the time they reach here: the samples were
    checked and :class:`~phonometry.io.Signal` requires a positive finite
    factor. Their product need not be, and it used to leave as an ``inf``
    that nothing had refused, since the finiteness check runs before the
    multiplication and cannot see past it.

    The overflow is caught in the multiplication itself rather than by
    measuring the result, so this costs no second pass over the recording:
    numpy already detects it and ``errstate`` only decides what to do about
    it. Nothing legitimate reaches it. The loudest quantity acoustics
    measures is a blast overpressure of a few bar, and a double holds
    something like ten to the three hundred and two times that, so a product
    that overflows is a calibration chain built wrong rather than a scale
    that will not fit.

    :param samples: The validated samples.
    :param factor: The calibration factor, positive and finite.
    :param name: Name of the signal parameter, for the error message.
    :return: The scaled samples.
    :raises ValueError: If the product overflows to infinity.
    """
    try:
        with np.errstate(over="raise"):
            return samples * factor
    except FloatingPointError:
        peak = float(np.max(np.abs(samples)))
        msg = (
            f"'{name}': the calibration factor overflows the samples it "
            f"scales; {factor!r} times a peak of {peak!r} leaves the range a "
            "double can hold. Both are finite on their own, so check the "
            "chain rather than either end of it."
        )
        raise ValueError(msg) from None


def resolve_samples(
    x: SignalInput, *, calibrate: bool = True, name: str = "x"
) -> np.ndarray:
    """The float64 samples of the input, in pascals when it can know.

    A :class:`~phonometry.io.Signal` contributes its array view (1-D for
    one channel, ``(channels, samples)`` for several), so a mono Signal
    yields the same result a mono array does. Both forms go through
    :func:`_typesignal`, which is where an input that is not a signal at
    all -- ``None``, a string, a ragged list, a ``NaN`` sample -- is
    refused by the name this call gives it; only the factor below tells
    the two apart.

    :param x: The signal argument, a Signal or a bare array.
    :param calibrate: Whether a calibrated Signal's factor is applied.
        ``False`` for the functions that own the factor themselves (they
        pair this with :func:`resolve_calibration`) and for the
        documented exemptions in the module docstring.
    :param name: Name of the signal parameter, for the error messages.
    :return: The samples, float64.
    :raises ValueError: If *x* is ``None``, cannot be read as numbers, or
        carries a non-finite sample.
    """
    samples = _typesignal(x, name=name)
    if calibrate and isinstance(x, Signal) and x.calibration_factor is not None:
        return _scaled(samples, x.calibration_factor, name)
    return samples


def apply_calibration(
    x: SignalInput, samples: np.ndarray, *, name: str = "x"
) -> np.ndarray:
    """Scale already-validated samples to pascals, if *x* carried a factor.

    The two-step sibling of :func:`resolve_samples`, for the functions that
    validate their input through a checker of their own. Calling
    :func:`resolve_samples` there instead would route the input through a
    second coercion and change which complaint a malformed array gets: a
    0-d input would stop being "must be one-dimensional" and start being
    "signal too short", for no reason a caller could act on.

    :param x: The signal argument as the caller passed it.
    :param samples: The validated samples derived from it.
    :param name: Name of the signal parameter, for the error message.
    :return: The samples, in pascals when the object knew how.
    :raises ValueError: If the factor overflows the samples it scales.
    """
    if isinstance(x, Signal) and x.calibration_factor is not None:
        return _scaled(samples, x.calibration_factor, name)
    return samples


@overload
def like_input(x: Signal, samples: np.ndarray, fs: int | None = ...) -> Signal: ...


@overload
def like_input(
    x: Sequence[float] | np.ndarray, samples: np.ndarray, fs: int | None = ...
) -> np.ndarray: ...


def like_input(
    x: SignalInput, samples: np.ndarray, fs: int | None = None
) -> Signal | np.ndarray:
    """Hand the samples back as a Signal when they arrived as one.

    The return side of the contract, and the mirror of
    :func:`resolve_samples`. A transform whose output is itself a record of
    pressure gives back a :class:`~phonometry.io.Signal` when it was given
    one, so a chain of them keeps the rate and the calibration instead of
    dropping them at the first step; a bare array in still gives a bare
    array out, so nothing that passes arrays today changes.

    The factor is the whole risk. A calibrated Signal presents its samples
    in pascals, so *samples* has ALREADY been scaled: carrying the input's
    factor forward would have every function downstream scale it a second
    time, which is a silent offset of ``20 lg(factor)`` through the 49
    sites that apply one, and it survives to disk because
    :func:`phonometry.io.write` stamps the factor into the sidecar. The
    returned object therefore says ``1.0``, meaning calibrated with the
    conversion already done, and ``None`` when the input had no calibration
    to begin with, because the object never invents one.

    :param x: The signal argument as the caller received it.
    :param samples: The samples the transform computed from it.
    :param fs: The rate of *samples* when the transform changed it; by
        default the input's own rate.
    :return: A Signal when *x* was one, otherwise *samples* unchanged.
    """
    if not isinstance(x, Signal):
        return samples
    return Signal(
        data=samples,
        fs=x.fs if fs is None else fs,
        calibration_factor=1.0 if x.calibration_factor is not None else None,
        channel_labels=x.channel_labels,
        provenance=x.provenance,
        source=x.source,
    )


def _agreed_pair(x: SignalInput, y: SignalInput, names: tuple[str, str]) -> SignalInput:
    """The member of the pair that knows the rate, once they agree.

    :param x: The first signal argument.
    :param y: The second signal argument.
    :param names: Names of the two signal parameters, for the message.
    :return: *x* when it is a Signal, otherwise *y*.
    :raises ValueError: If both are Signals recorded at different rates.
    """
    if isinstance(x, Signal) and isinstance(y, Signal) and x.fs != y.fs:
        msg = (
            f"'{names[0]}' and '{names[1]}' are Signals recorded at "
            f"different rates ({x.fs} Hz and {y.fs} Hz); resample one of "
            "them before comparing them"
        )
        raise ValueError(msg)
    return x if isinstance(x, Signal) else y


def resolve_pair_fs[Rate: (int, float)](
    x: SignalInput,
    y: SignalInput,
    fs: Rate | None,
    *,
    names: tuple[str, str] = ("x", "y"),
    rate: str = "fs",
) -> Rate:
    """Resolve one sample rate from two signal arguments.

    Two recordings are only comparable at one rate, so two Signals that
    disagree are refused for the same reason a Signal and an explicit
    argument are: nothing here can say which of them is the truth, and
    picking one silently mis-times the result. Either one may be a bare
    array -- pairing a read measurement with a synthesized reference is
    the common case -- and then the Signal supplies the rate.

    :param x: The first signal argument.
    :param y: The second signal argument.
    :param fs: The explicitly passed rate, or ``None``.
    :param names: Names of the two signal parameters, for the messages.
    :param rate: Name of the rate parameter, for the messages.
    :return: The resolved sample rate.
    :raises ValueError: On any disagreement, or when nothing knows the rate.
    """
    known = _agreed_pair(x, y, names)
    return resolve_fs(known, fs, name=names[0], rate=rate)


def resolve_optional_pair_fs[Rate: (int, float)](
    x: SignalInput,
    y: SignalInput,
    fs: Rate | None,
    *,
    names: tuple[str, str] = ("x", "y"),
    rate: str = "fs",
) -> Rate | None:
    """Resolve a rate from two signal arguments that can also do without.

    :func:`resolve_optional_fs` for a pair: same refusals, and a pair of
    bare arrays with no rate is still a legal call.

    :param x: The first signal argument.
    :param y: The second signal argument.
    :param fs: The explicitly passed rate, or ``None``.
    :param names: Names of the two signal parameters, for the messages.
    :param rate: Name of the rate parameter, for the messages.
    :return: The resolved sample rate, or ``None`` if nothing knows it.
    :raises ValueError: On any disagreement.
    """
    return resolve_optional_fs(_agreed_pair(x, y, names), fs, rate=rate)
