#  Copyright (c) 2026. Jose Manuel Requena Plens
r"""
Backend dispatch: base install for the WAV family, ``[audio]`` for the rest.

The base install (NumPy + SciPy) reads every *linear* WAV a measurement
chain produces -- PCM at any depth, IEEE float, EXTENSIBLE, BWF metadata,
RF64/BW64 -- which is where essentially all metrology-grade audio lives
(see :mod:`phonometry.io._wav`). Everything else goes through
python-soundfile (libsndfile), installed as the optional ``[audio]`` extra:
FLAC, AIFF, Ogg Vorbis, Opus, MP3, and the compressed WAV codecs (ADPCM,
A-law/mu-law) that some sound level meters write for listening copies.
Files are told apart by their leading magic bytes, not their extension: a
mislabelled file dispatches by what it is, and the error for a missing
backend can name the actual format.

Lossy sources are read, not refused, but never silently: a decoder for
Ogg/Opus/MP3/ADPCM reconstructs an *approximation* of the recorded
waveform, so a level computed from it is not defensible as a measurement
(and the NTi XL2's default 4-bit ADPCM recording format exists precisely
as a listening record, with the manual reserving linear WAV for
post-processing). :func:`read` emits :class:`LossyCompressionWarning`
(a :class:`~phonometry.PhonometryWarning`) and stamps ``lossy=True`` on the
signal's source metadata, so the fact survives past the warning.

soundfile decodes to ``(frames, channels)`` float64, already scaled by the
libsndfile convention (:math:`\div 2^{B-1}`), consistent with the base
reader's scaling -- and, like there, any fixed scaling choice cancels out
of calibrated results because the calibrator tone is read through the same
path. When a compressed *WAV* goes through soundfile, the chunk walker
still contributes what libsndfile does not expose: the ``bext`` provenance
and the EXTENSIBLE channel labels ride along on the returned signal.
"""

from __future__ import annotations

import warnings
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from .._internal.warnings import PhonometryWarning
from ._chunks import WAVE_FORMAT_IEEE_FLOAT, WAVE_FORMAT_PCM, parse_wav_chunks
from ._flac import read_flac_bext
from ._sidecar import read_sidecar
from ._signal import Signal, SignalOrigin
from ._wav import AudioFileInfo, read_wav, wav_info

if TYPE_CHECKING:
    from ._chunks import WavChunks


class LossyCompressionWarning(PhonometryWarning):
    """Warns that samples came through a lossy codec: levels are not defensible."""


#: Leading magic bytes -> format name, for dispatch and for error messages
#: that name what the file actually is. The WAV family is recognised
#: separately (fourcc + ``WAVE`` form type).
_MAGIC: tuple[tuple[bytes, str], ...] = (
    (b"fLaC", "FLAC"),
    (b"OggS", "Ogg (Vorbis/Opus)"),
    (b"ID3", "MPEG audio (MP3)"),
    (b"FORM", "AIFF"),
    (b".snd", "AU"),
    (b"caff", "CAF"),
)

_WAV_FOURCC = (b"RIFF", b"RF64", b"BW64")

#: soundfile subtype substrings that mark a lossy or companded codec.
#: libsndfile subtypes are stable identifiers ('VORBIS', 'OPUS',
#: 'MPEG_LAYER_III', 'IMA_ADPCM', 'MS_ADPCM', 'ULAW', 'ALAW', 'GSM610',
#: 'G721_32', ...); everything PCM/FLOAT/DOUBLE (including inside FLAC,
#: which is lossless compression of PCM) is not matched.
_LOSSY_SUBTYPE_MARKS = (
    "VORBIS", "OPUS", "MPEG", "ADPCM", "ULAW", "ALAW", "GSM", "G721",
    "G723", "DWVW", "VOX",
)

#: soundfile subtype -> valid bits per sample, where the notion applies.
_SUBTYPE_BITS = {
    "PCM_S8": 8, "PCM_U8": 8, "PCM_16": 16, "PCM_24": 24, "PCM_32": 32,
    "FLOAT": 32, "DOUBLE": 64,
}

_AUDIO_EXTRA_HINT = (
    "Install the [audio] extra with: pip install phonometry[audio]"
)


def _sniff(path: str | Path) -> str | None:
    """Identify a file by its magic bytes: ``None`` for the WAV family.

    :raises ValueError: If the leading bytes match no known audio format.
    """
    with Path(path).open("rb") as fh:
        head = fh.read(12)
    if head[:4] in _WAV_FOURCC and head[8:12] == b"WAVE":
        return None
    for magic, name in _MAGIC:
        if head.startswith(magic):
            return name
    # An MP3 without an ID3 tag starts straight at a frame sync
    # (11 set bits: 0xFF 0xE0).
    if len(head) >= 2 and head[0] == 0xFF and head[1] & 0xE0 == 0xE0:
        return "MPEG audio (MP3)"
    raise ValueError(f"{path}: not a recognised audio file")


def _import_soundfile(format_name: str) -> Any:
    """Import python-soundfile or explain which extra provides it."""
    try:
        import soundfile
    except ImportError as exc:
        raise ImportError(
            f"Reading {format_name} requires python-soundfile, which "
            f"phonometry ships as the [audio] extra. {_AUDIO_EXTRA_HINT}"
        ) from exc
    return soundfile


def _warn_lossy(format_name: str, path: str | Path) -> None:
    """Emit the metrology warning for a lossy source."""
    warnings.warn(
        f"{path}: {format_name} is a lossy format; the decoder reconstructs "
        "an approximation of the recorded waveform, so levels computed from "
        "it are not metrologically defensible. Use the linear PCM recording "
        "for analysis; lossy copies are for listening.",
        LossyCompressionWarning,
        stacklevel=3,
    )


def _soundfile_lossy(subtype: str) -> bool:
    """Whether a libsndfile subtype identifier names a lossy codec."""
    return any(mark in subtype.upper() for mark in _LOSSY_SUBTYPE_MARKS)


def _read_soundfile(
    path: str | Path,
    format_name: str,
    *,
    calibration_factor: float | None,
    chunks: WavChunks | None = None,
) -> Signal:
    """Decode through soundfile, warning and stamping lossy sources.

    ``chunks`` carries the base walker's harvest for a compressed WAV, so
    the ``bext`` provenance and channel labels libsndfile does not expose
    still reach the signal.
    """
    sf = _import_soundfile(format_name)
    file_info = sf.info(str(path))
    lossy = _soundfile_lossy(str(file_info.subtype))
    if lossy:
        _warn_lossy(f"{format_name} ({file_info.subtype})", path)
    data, fs = sf.read(str(path), dtype="float64", always_2d=True)
    if chunks is not None:
        provenance = chunks.bext
    elif format_name == "FLAC":
        # A FLAC written by phonometry (or by flac --keep-foreign-metadata)
        # carries the bext chunk in an APPLICATION 'riff' block.
        provenance = read_flac_bext(path)
    else:
        provenance = None
    return Signal(
        data=np.ascontiguousarray(data.T),
        fs=int(fs),
        calibration_factor=calibration_factor,
        channel_labels=chunks.fmt.channel_labels() if chunks is not None else None,
        provenance=provenance,
        source=SignalOrigin(
            path=str(path),
            container=chunks.container if chunks is not None else str(file_info.format),
            format_name=str(file_info.subtype),
            bit_depth=_SUBTYPE_BITS.get(str(file_info.subtype)),
            lossy=lossy,
        ),
    )


def read(path: str | Path, *, calibration_factor: float | None = None) -> Signal:
    """Read an audio file into a :class:`~phonometry.io.Signal`.

    The samples come back float64, at the file's native rate, in the
    library's ``(channels, samples)`` order, scaled by :math:`2^{B-1}` for
    integer sources (exact in binary floating point, and any fixed scaling
    cancels out of calibrated results -- see :mod:`phonometry.io._wav`).
    Nothing else is done to them: no resampling, no normalisation, no
    channel mixing -- each of which would silently change the level or the
    timing that a measurement depends on, and each of which is a documented
    default somewhere in the ecosystem (librosa resamples to 22050 Hz and
    mixes to mono; python-acoustics normalised on load).

    The base install reads the whole linear WAV family: PCM at any depth,
    IEEE float, WAVE_FORMAT_EXTENSIBLE (with its channel labels), BWF
    ``bext`` provenance, RF64/BW64 long recordings. With the ``[audio]``
    extra (python-soundfile), FLAC, AIFF, Ogg Vorbis, Opus, MP3 and
    compressed WAV codecs (ADPCM, A-law/mu-law) decode too; without it,
    those raise :class:`ImportError` naming the extra. A lossy source is
    read but announced with :class:`LossyCompressionWarning` and stamped
    ``lossy=True`` on ``signal.source``: a level computed from an
    approximation of the waveform is not a measurement.

    A calibration sidecar beside the file
    (:mod:`phonometry.io._sidecar`) is applied automatically: its
    ``calibration_factor`` fills in when the argument here is ``None``
    (the explicit argument always wins -- the caller knows more than a
    file on disk), and its curated ``channel_labels`` replace labels
    derived from the file's channel mask.

    :param path: The file to read.
    :param calibration_factor: Digital-to-pascal multiplier to attach to
        the returned signal when the calibration is known (typically from
        :func:`phonometry.metrology.sensitivity` on a calibrator recording read
        through this same function). ``None`` (the default) takes the
        sidecar's factor when a sidecar exists, and otherwise leaves the
        signal in digital full-scale units.
    :return: The signal with its metadata.
    :raises ValueError: If the file matches no known audio format, or a
        sidecar exists but is invalid.
    :raises ImportError: If the format needs the ``[audio]`` extra and it
        is not installed.
    """
    sidecar = read_sidecar(path)
    if sidecar is not None and calibration_factor is None:
        calibration_factor = sidecar.calibration_factor
    format_name = _sniff(path)
    if format_name is None:
        chunks = parse_wav_chunks(path)
        if chunks.fmt.resolved_tag in (WAVE_FORMAT_PCM, WAVE_FORMAT_IEEE_FLOAT):
            signal = read_wav(path, chunks, calibration_factor=calibration_factor)
        else:
            signal = _read_soundfile(
                path,
                f"WAV {chunks.fmt.format_name}",
                calibration_factor=calibration_factor,
                chunks=chunks,
            )
    else:
        signal = _read_soundfile(
            path, format_name, calibration_factor=calibration_factor
        )
    if sidecar is not None and sidecar.channel_labels is not None:
        signal = replace(signal, channel_labels=sidecar.channel_labels)
    return signal


def info(path: str | Path) -> AudioFileInfo:
    """Describe an audio file without reading its samples.

    For the WAV family this is a single pass over the chunk headers on the
    base install, whatever the codec inside -- format, rate, channels,
    frames and duration, valid bits, lossy flag, EXTENSIBLE channel mask
    and labels, ``bext`` provenance, ``iXML`` presence and cue markers --
    with a cost independent of file size. Other containers are described
    through the ``[audio]`` extra's header reader. ``info`` never warns
    about lossy content (inspecting a file is not measuring from it);
    the ``lossy`` field carries the fact instead.

    :param path: The file to describe.
    :return: The file's description.
    :raises ValueError: If the file matches no known audio format.
    :raises ImportError: If the container needs the ``[audio]`` extra and
        it is not installed.
    """
    format_name = _sniff(path)
    if format_name is None:
        return wav_info(path)
    sf = _import_soundfile(format_name)
    file_info = sf.info(str(path))
    frames = int(file_info.frames)
    fs = int(file_info.samplerate)
    return AudioFileInfo(
        path=str(path),
        container=str(file_info.format),
        format_name=str(file_info.subtype),
        fs=fs,
        channels=int(file_info.channels),
        frames=frames,
        duration=frames / float(fs) if fs else 0.0,
        bit_depth=_SUBTYPE_BITS.get(str(file_info.subtype)),
        lossy=_soundfile_lossy(str(file_info.subtype)),
        bext=read_flac_bext(path) if format_name == "FLAC" else None,
    )
