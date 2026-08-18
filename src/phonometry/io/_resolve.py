#  Copyright (c) 2026. Jose Manuel Requena Plens
"""
Resolving an ``(x, fs)`` argument pair that may arrive as a ``Signal``.

A function that consumes a recording should take a
:class:`phonometry.io.Signal` in place of the bare ``(x, fs)`` pair: the
object read from a measurement file already knows its sample rate and,
when calibrated, its digital-to-pascal factor, so asking the caller to
repeat either is asking for a transcription error. The helpers here are
that contract, in one place, so that the surfaces adopting it cannot
drift apart. ``signals.levels`` and ``filters`` are on it; the rest of
the library still takes the bare pair, and moves over surface by
surface.

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
  time, so there the argument stays mandatory.
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

import numpy as np

from .._internal.utils import _typesignal
from ._signal import Signal

#: What a function that consumes a recording accepts for its signal argument.
SignalInput = Signal | list[float] | np.ndarray

# The rate type parameter below is value-restricted to ``(int, float)``
# rather than bound to ``float`` so that a module typing its rate as ``int``
# gets an ``int`` back: a plain ``-> float`` would leak a float into every
# ``int``-typed sample count downstream, and a pair of ``int`` / ``float``
# overloads overlap (``int`` is assignable to ``float``) with return types
# mypy rejects as incompatible.


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
    if isinstance(x, Signal):
        if fs is not None and fs != x.fs:
            raise ValueError(
                f"{rate}={fs} conflicts with the Signal's own fs={x.fs}; "
                "pass one or the other, not a disagreement"
            )
        return x.fs
    if fs is None:
        raise ValueError(f"{rate} is required when '{name}' is a bare array")
    return fs


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
        raise ValueError(
            f"the Signal was recorded at {x.fs} Hz but this {what} was "
            f"designed for {fs:g} Hz; resample the Signal, or build a "
            f"{what} for its own rate"
        )


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


def resolve_samples(x: SignalInput, *, calibrate: bool = True) -> np.ndarray:
    """The float64 samples of the input, in pascals when it can know.

    A :class:`~phonometry.io.Signal` contributes its array view (1-D for
    one channel, ``(channels, samples)`` for several), so a mono Signal
    yields the same result a mono array does; bare input passes through
    :func:`_typesignal` untouched.

    :param x: The signal argument, a Signal or a bare array.
    :param calibrate: Whether a calibrated Signal's factor is applied.
        ``False`` for the functions that own the factor themselves (they
        pair this with :func:`resolve_calibration`) and for the
        documented exemptions in the module docstring.
    :return: The samples, float64.
    """
    if not isinstance(x, Signal):
        return _typesignal(x)
    samples = _typesignal(np.asarray(x))
    if calibrate and x.calibration_factor is not None:
        return samples * x.calibration_factor
    return samples
