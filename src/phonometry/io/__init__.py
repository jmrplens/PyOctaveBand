#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Measurement audio files: read, write, stream and convert without touching a level.

Every function here treats an audio file as a measurement record rather than
as material to be played back, which fixes the defaults: the native sample
rate is kept (no resampling on load), channels are never mixed down, samples
are never normalized, and integer PCM is scaled by exactly :math:`2^{B-1}`
into float64 -- a power of two, so the scaling is exact in binary floating
point, and a constant that cancels out of every calibrated level because the
calibrator tone is read through the same path (the derivation lives with the
WAV reader's source).

:func:`read` returns a :class:`Signal`: the samples as ``(channels, samples)``
float64 together with the sample rate, the calibration, the channel labels,
the ``bext`` broadcast provenance (EBU Tech 3285) and the origin record --
one immutable object that any ``(x, fs, ...)`` function of the library
accepts today via :func:`numpy.asarray`. The base install reads every linear
WAV a sound level meter or field recorder writes (PCM 16/24/32-bit, IEEE
float, ``WAVE_FORMAT_EXTENSIBLE``, RF64/BW64 past 4 GiB); the ``[audio]``
extra (python-soundfile, which bundles the LGPL libsndfile) adds FLAC, AIFF,
Ogg/Opus and MP3, and lossy sources raise :class:`LossyCompressionWarning`
because a level computed from a lossy codec is not metrologically defensible.

:func:`info` answers from the headers alone -- format, rate, channels, valid
bits, duration, ``bext``, cue points -- without decoding a single sample, so
it is safe on a 12-hour RF64. :func:`read_blocks` streams the same samples
:func:`read` would return, block by block, into the library's stateful
filters. :func:`write` produces WAV/BWF (and FLAC with the extra) with exact
integer codes, loud clipping (:class:`ClippingWarning`), optional TPDF dither
at 16 bits, a ``bext`` chunk written field by field, and never a silent
normalization. :func:`convert` moves a measurement between lossless
containers with samples, provenance and sidecar intact, and the calibration
travels in a versioned JSON sidecar (:class:`CalibrationSidecar`) next to the
audio, where the audio formats themselves have no field for it.
"""

from __future__ import annotations

from ._backends import LossyCompressionWarning, info, read
from ._blocks import read_blocks
from ._chunks import BroadcastMetadata, CuePoint
from ._convert import convert
from ._sidecar import (
    CalibrationSidecar,
    read_sidecar,
    sidecar_path,
    write_sidecar,
)
from ._signal import Signal, SignalOrigin
from ._wav import AudioFileInfo
from ._write import ClippingWarning, write

__all__ = [
    "AudioFileInfo",
    "BroadcastMetadata",
    "CalibrationSidecar",
    "ClippingWarning",
    "CuePoint",
    "LossyCompressionWarning",
    "Signal",
    "SignalOrigin",
    "convert",
    "info",
    "read",
    "read_blocks",
    "read_sidecar",
    "sidecar_path",
    "write",
    "write_sidecar",
]
