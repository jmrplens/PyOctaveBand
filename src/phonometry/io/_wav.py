#  Copyright (c) 2026. Jose Manuel Requena Plens
r"""Base-install WAV reading: scipy decodes the samples, the walker the rest.

:mod:`scipy.io.wavfile` already decodes every linear WAV variant measurement
equipment writes -- PCM 1-64 bit (24-bit arrives as ``int32`` with the sample
left-justified in the container), IEEE float 32/64, WAVE_FORMAT_EXTENSIBLE
(it resolves the codec from the SubFormat GUID) and RF64 (it honours
``ds64``) -- so the base install needs no new dependency to read the linear
recordings of an NTi XL2 Extended Acoustic Pack, a Svantek SVAN or a field
recorder. What scipy does not do is keep the metadata (it skips ``bext`` and
friends with a ``WavFileWarning``) or choose a float scaling; this module
adds both and returns a :class:`~phonometry.io.Signal`.

**Integer scaling.** Integer samples are divided by :math:`2^{B-1}` for a
``B``-bit container (128 for the unsigned 8-bit convention, 32768 for 16-bit,
:math:`2^{31}` for 24-bit-in-``int32`` -- scipy's left-justification makes
that identical to dividing the 24-bit value by :math:`2^{23}`). Two reasons,
in order of weight:

* **The choice cancels out of every calibrated result.** phonometry's
  calibration model derives ``calibration_factor`` from a recording of the
  calibrator tone (:func:`phonometry.metrology.sensitivity`, IEC 60942) *read
  through this same reader*. Write the reader's scaling as :math:`x = n / D`
  for divisor ``D``: the calibrator recording gives
  ``calibration_factor = p_ref / rms(n_cal / D) = D * p_ref / rms(n_cal)``, and
  a measurement then reads
  ``x * calibration_factor = (n / D) * D * p_ref / rms(n_cal)`` -- ``D``
  divides out exactly. The perennial 32767-vs-32768 debate is therefore moot
  here; even uncalibrated, the difference is
  ``20*log10(32768/32767) = 0.00027 dB``, some 380 times smaller than the
  0.1 dB indication resolution of an IEC 61672 sound level meter.
* **A power of two is exact in binary floating point.** Dividing by
  :math:`2^{B-1}` only shifts the exponent, so every integer sample maps to
  a float64 it round-trips from, the most negative code lands exactly on
  -1.0, and the scaling itself adds zero rounding error. It is also the
  convention of libsndfile and MATLAB's ``audioread``, so levels agree with
  those tools to the last bit.

Nothing else touches the samples: no resampling (the library's own
polyphase resampler exists for whoever wants one *after* the import), no
normalisation (the python-acoustics ``normalize=True`` default destroyed
absolute level on every load, which is the exact mistake a metrology
library cannot make), no dithering, no channel mixing.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from scipy.io import wavfile

from ._chunks import (
    WAVE_FORMAT_IEEE_FLOAT,
    WAVE_FORMAT_PCM,
    WavChunks,
    parse_wav_chunks,
)
from ._signal import Signal, SignalOrigin

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ._chunks import BroadcastMetadata, CuePoint


@dataclass(frozen=True)
class AudioFileInfo:
    """What :func:`phonometry.io.info` learned without reading the samples.

    ``bit_depth`` is the *valid* bits per sample where the notion applies
    (for an EXTENSIBLE container the ``wValidBitsPerSample`` field, so a
    20-in-24-bit file reports 20) and ``None`` for codecs that decode to
    float. ``lossy`` marks codecs that cannot be trusted with the recorded
    waveform: the transcribed lossy/companded tags, and any codec outside
    the transcription that is not linear PCM/IEEE float, which fails
    closed as suspect rather than passing as clean (the
    ``FormatChunk.lossy`` property explains the bias);
    :func:`phonometry.io.read` warns when asked to read one.
    ``channel_mask``/``channel_labels``, ``bext``, ``has_ixml`` and
    ``cue_points`` are WAV-family metadata and are ``None``/empty for other
    containers -- except ``bext``, which a FLAC can also carry in its
    APPLICATION ``riff`` block (see :mod:`phonometry.io._flac`).
    """

    path: str
    container: str
    format_name: str
    fs: int
    channels: int
    frames: int
    duration: float
    bit_depth: int | None
    lossy: bool
    channel_mask: int | None = None
    channel_labels: tuple[str, ...] | None = None
    bext: BroadcastMetadata | None = None
    has_ixml: bool = False
    cue_points: tuple[CuePoint, ...] = ()


def _scale_to_float(raw: NDArray[np.generic]) -> NDArray[np.float64]:
    """Scale scipy's decoded samples to float64 by the module convention.

    Float input passes through untouched (only recast); unsigned 8-bit is
    re-centred on its midpoint 128 before the division; signed integers are
    divided by ``2**(container bits - 1)``, which for scipy's left-justified
    24-bit-in-int32 equals dividing the 24-bit sample by ``2**23``. See the
    module docstring for why the divisor is this one and why the choice
    cannot bias a calibrated result.
    """
    if raw.dtype.kind == "f":
        # copy=True even when the input is float64 already: the input may be
        # scipy's read-only memmap, and the Signal must own real memory, not
        # keep the file handle alive behind a lazily-paged view.
        return raw.astype(np.float64, order="C", copy=True)
    if raw.dtype == np.uint8:
        return (raw.astype(np.float64, order="C") - 128.0) / 128.0
    scale = float(2 ** (8 * raw.dtype.itemsize - 1))
    return raw.astype(np.float64, order="C") / scale


def wav_info(path: str | Path, chunks: WavChunks | None = None) -> AudioFileInfo:
    """Describe a WAV/BWF/RF64 file from its headers alone.

    A single pass of the chunk walker: the samples are never read, so the
    cost is independent of file size (an overnight RF64 answers as fast as
    a two-second snippet). Everything scipy would have dropped is kept:
    the EXTENSIBLE channel mask (and the loudspeaker labels it implies),
    valid bits, ``bext`` provenance, ``iXML`` presence and cue markers.

    :param path: The file to describe.
    :param chunks: An already-walked :class:`WavChunks`, to avoid a second
        pass when the caller has one.
    :return: The file's description.
    """
    parsed = parse_wav_chunks(path) if chunks is None else chunks
    fmt = parsed.fmt
    frames = parsed.frames
    lossless = fmt.resolved_tag in (WAVE_FORMAT_PCM, WAVE_FORMAT_IEEE_FLOAT)
    return AudioFileInfo(
        path=str(path),
        container=parsed.container,
        format_name=fmt.format_name,
        fs=fmt.fs,
        channels=fmt.channels,
        frames=frames,
        duration=frames / float(fmt.fs) if fmt.fs else 0.0,
        bit_depth=(fmt.valid_bits or fmt.bits_per_sample) if lossless else None,
        lossy=fmt.lossy,
        channel_mask=fmt.channel_mask,
        channel_labels=fmt.channel_labels(),
        bext=parsed.bext,
        has_ixml=parsed.has_ixml,
        cue_points=parsed.cue_points,
    )


def read_wav(
    path: str | Path,
    chunks: WavChunks | None = None,
    *,
    calibration_factor: float | None = None,
) -> Signal:
    """Read a linear (PCM or IEEE float) WAV into a :class:`Signal`.

    scipy decodes the samples -- memory-mapped where the container allows
    it (everything but 24-bit, whose 3-byte stride no numpy dtype can map),
    so the integer file is paged in rather than duplicated while the float64
    copy is built -- and its ``WavFileWarning`` about skipped chunks is
    suppressed, because the chunk walker read those chunks already: the
    ``bext`` provenance and EXTENSIBLE channel labels land on the returned
    object instead of a warning about their loss. Samples are scaled per
    the module convention and transposed to the library's
    ``(channels, samples)`` axis order. A ``BW64`` container (ITU-R
    BS.2088) shares RF64's ``ds64`` semantics but not the fourcc scipy
    recognises, so its payload is decoded in-house by the block reader's
    linear decoder instead of through scipy.

    Before any sample is decoded, the ``data`` payload the header promises
    is checked against the bytes the file actually holds. The walker never
    verifies this (``info()`` on a header-only forgery is a feature, and it
    seeks past the payload rather than reading it), and scipy cannot be
    relied on to: its non-mmap path -- 24-bit, the depth field recorders
    actually write -- tolerates a short read and returns the frames it
    found. A truncated recording (interrupted copy, full card) would then
    come back as a shorter file that looks right, and a level integrated
    over it (Leq, SEL) would be a wrong number with nothing wrong-looking
    about it. Refusing loudly here matches :func:`~phonometry.io.read_blocks`,
    which already raises for the same defect.

    :param path: The file to read.
    :param chunks: An already-walked :class:`WavChunks` for this file, to
        avoid a second pass when the caller (the backend dispatcher) has one.
    :param calibration_factor: Digital-to-pascal multiplier to attach to the
        signal, when known. Never invented: without one the signal stays in
        full-scale units and says so.
    :return: The signal with its metadata.
    :raises ValueError: If the file's codec is not linear PCM/float (those
        files go through the ``[audio]`` backend, which decodes them), or
        the file ends before the ``data`` payload its header declares.
    """
    parsed = parse_wav_chunks(path) if chunks is None else chunks
    fmt = parsed.fmt
    if fmt.resolved_tag not in (WAVE_FORMAT_PCM, WAVE_FORMAT_IEEE_FLOAT):
        raise ValueError(
            f"{path}: {fmt.format_name} is not linear PCM/float; "
            "read it through the [audio] backend"
        )
    payload_end = parsed.data_offset + parsed.data_size
    actual = Path(path).stat().st_size
    if actual < payload_end:
        raise ValueError(
            f"{path}: data chunk claims {parsed.data_size} bytes "
            f"({parsed.frames} frames) but the file ends "
            f"{payload_end - actual} bytes short; refusing to read a "
            "truncated recording as if it were complete"
        )
    if parsed.container == "BW64":
        # scipy honours ds64 for the RF64 fourcc but rejects BW64 (ITU-R
        # BS.2088), which differs from RF64 in that four-byte tag alone;
        # the payload the walker already located decodes in-house through
        # the block reader's linear decoder instead.
        from ._blocks import _decode_linear_frames

        with Path(path).open("rb") as fh:
            fh.seek(parsed.data_offset)
            raw_bytes = fh.read(parsed.frames * fmt.block_align)
        data = np.atleast_2d(_decode_linear_frames(raw_bytes, fmt))
        fs = fmt.fs
    else:
        with warnings.catch_warnings():
            # The walker already kept what scipy is about to skip; warning
            # about the skip would tell the user their metadata was lost
            # when it was not.
            warnings.simplefilter("ignore", wavfile.WavFileWarning)
            fs, raw = wavfile.read(str(path), mmap=fmt.bits_per_sample != 24)
        # scipy returns (frames,) or (frames, channels); the transpose to
        # the library's (channels, samples) is taken before the float
        # conversion so the conversion itself materialises the C-contiguous
        # layout the axis=-1 reductions want, costing one copy instead of
        # two.
        data = np.atleast_2d(_scale_to_float(raw.T))
    return Signal(
        data=data,
        fs=int(fs),
        calibration_factor=calibration_factor,
        channel_labels=fmt.channel_labels(),
        provenance=parsed.bext,
        source=SignalOrigin(
            path=str(path),
            container=parsed.container,
            format_name=fmt.format_name,
            bit_depth=fmt.valid_bits or fmt.bits_per_sample,
            lossy=False,
        ),
    )
