#  Copyright (c) 2026. Jose Manuel Requena Plens
r"""
Converting a measurement file without losing what makes it a measurement.

Generic converters treat audio as audio: they resample when convenient,
normalise when helpful, and drop every chunk they do not recognise --
which is precisely the originator, the sample-accurate timecode, the
coding history and the calibration that separate a measurement record
from a sound. :func:`convert` is the composition of this module's own
reader and writers under a contract none of those tools offer:

* **Samples pass through at full precision.** No resampling, no
  normalisation, no channel mixing, ever. Integer sources land on the
  same codes (the float64 intermediary holds any code up to 53 bits
  exactly, and the writer's quantisation is the exact inverse of the
  reader's scaling), so WAV to FLAC and back is bit-identical. A
  narrower explicit ``subtype`` quantises by the documented rounding
  with the clipping count accumulated across the whole stream and
  reported once.
* **The provenance travels whole.** The ``bext`` chunk is carried field
  for field -- into the WAV family natively, into FLAC via the
  APPLICATION ``riff`` block of :mod:`._flac` -- with its CodingHistory
  extended (never replaced) by one EBU R98 line naming this conversion.
  A calibration sidecar (:mod:`._sidecar`) beside the source is
  validated and copied beside the destination byte for byte. What is
  *not* carried is noted honestly: ``cue`` markers, ``iXML`` and
  EXTENSIBLE channel-mask labels stay behind (the sidecar's curated
  labels are the supported vehicle for channel naming across formats),
  and a lossy *source's* nature is announced by the read-time warning
  but cannot be stamped into the output container -- keep the original
  file; the conversion is a copy of a reconstruction.
* **Lossy targets do not exist here.** Writing MP3/Ogg/Opus from a
  metrology library is the decided policy this module enforces at the
  suffix, before a single sample is read.
* **Memory stays flat.** The samples stream through
  :func:`~phonometry.io.read_blocks` into the streaming WAV writer (the
  frame count is known from the headers, so the RF64-aware header is
  exact up front; for a lossy source, whose decoder may deliver a few
  frames more or fewer than the container declares, the writer
  reconciles the header with the streamed count at close) or into an
  open ``SoundFile``, one block at a time: an hour of RF64 converts in
  block-sized memory.

The default target ``subtype`` preserves what the source had: integer
depths map to the smallest of PCM_16/24/32 that holds them, float
sources keep their float width, and decoded lossy sources become FLOAT
(the decoder hands floats; there are no original codes to preserve).
FLAC targets hold integers up to 24 bits only, so sources that do not
fit (float, 32-bit int) require an explicit ``subtype`` -- the caller,
not the library, accepts that narrowing.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ._backends import _import_soundfile, _sniff, info
from ._blocks import read_blocks
from ._chunks import (
    WAVE_FORMAT_IEEE_FLOAT,
    WAVE_FORMAT_PCM,
    BroadcastMetadata,
    parse_wav_chunks,
)
from ._flac import embed_flac_bext, read_flac_bext
from ._sidecar import read_sidecar, sidecar_path
from ._signal import Signal
from ._write import (
    _SUBTYPE_BITS,
    _SUBTYPE_SPEC,
    _WAV_SUFFIXES,
    _quantise,
    _resolve_bext,
    _warn_clipped,
    frame_riff_chunk,
    write_wav_stream,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from ._wav import AudioFileInfo

#: Suffixes that name lossy targets, refused by decided policy (writing
#: approximations is outside this API; the read side accepts them with a
#: warning, which is a different thing).
_LOSSY_TARGET_SUFFIXES = frozenset(
    {".mp3", ".ogg", ".oga", ".opus", ".m4a", ".aac", ".wma"}
)

#: soundfile subtype -> (numeric kind, bits) for default-subtype selection.
_SF_SUBTYPE_TRAITS: dict[str, tuple[str, int]] = {
    "PCM_S8": ("int", 8), "PCM_U8": ("int", 8), "PCM_16": ("int", 16),
    "PCM_24": ("int", 24), "PCM_32": ("int", 32),
    "FLOAT": ("float", 32), "DOUBLE": ("float", 64),
}


def _source_traits(
    path: str | Path,
) -> tuple[str, int, BroadcastMetadata | None, int, int, int]:
    """What the target choice needs to know about the source.

    Returns ``(kind, bits, bext, fs, channels, frames)`` where ``kind``
    is ``"int"``, ``"float"`` or ``"decoded"`` (a codec whose decoder
    hands floats with no original integer codes behind them -- the lossy
    family) and ``bits`` is the container depth (0 where the notion does
    not apply).
    """
    format_name = _sniff(path)
    if format_name is None:
        chunks = parse_wav_chunks(path)
        fmt = chunks.fmt
        if fmt.resolved_tag == WAVE_FORMAT_PCM:
            kind, bits = "int", fmt.bits_per_sample
        elif fmt.resolved_tag == WAVE_FORMAT_IEEE_FLOAT:
            kind, bits = "float", fmt.bits_per_sample
        else:
            kind, bits = "decoded", 0
        return kind, bits, chunks.bext, fmt.fs, fmt.channels, chunks.frames
    sf = _import_soundfile(format_name)
    file_info = sf.info(str(path))
    kind, bits = _SF_SUBTYPE_TRAITS.get(str(file_info.subtype), ("decoded", 0))
    bext = read_flac_bext(path) if format_name == "FLAC" else None
    return (
        kind, bits, bext, int(file_info.samplerate),
        int(file_info.channels), int(file_info.frames),
    )


def _default_subtype(kind: str, bits: int, *, flac: bool) -> str:
    """The subtype that preserves the source, or a demand to choose one."""
    if kind == "int":
        if bits <= 16:
            depth = 16
        elif bits <= 24:
            depth = 24
        else:
            depth = 32
        if flac and depth > 24:
            raise ValueError(
                "FLAC holds integer PCM up to 24 bits (libsndfile); a "
                f"{bits}-bit integer source cannot pass through whole. "
                "Pass subtype='PCM_24' to accept the truncation explicitly."
            )
        return f"PCM_{depth}"
    if flac:
        raise ValueError(
            "FLAC holds integer PCM only; converting a "
            f"{'float' if kind == 'float' else 'lossy-decoded'} source "
            "quantises it, so choose subtype='PCM_16' or 'PCM_24' "
            "explicitly rather than have the library narrow it silently."
        )
    if kind == "float":
        return "FLOAT" if bits == 32 else "DOUBLE"
    return "FLOAT"


def _quantised_blocks(
    src: str | Path,
    dst: str | Path,
    subtype: str,
    block_size: int,
    total_samples: int,
) -> Iterator[np.ndarray]:
    """Stream source blocks quantised for the target, counting every clip.

    The clipping count accumulates across the whole file and is reported
    once at the end -- a warning per block would drown the terminal and
    say less than one total.
    """
    is_float_target = _SUBTYPE_SPEC[subtype][0] != 0x0001
    bits = _SUBTYPE_BITS[subtype]
    clipped_total = 0
    peak_dbfs = float("-inf")
    for block in read_blocks(src, block_size):
        block2d = np.atleast_2d(block)
        if is_float_target:
            yield block2d
            continue
        codes, clipped, peak = _quantise(block2d, bits, None)
        clipped_total += clipped
        peak_dbfs = max(peak_dbfs, peak)
        yield codes
    if clipped_total:
        _warn_clipped(dst, subtype, clipped_total, total_samples, peak_dbfs)


def _check_conversion_paths(src: str | Path, dst: str | Path) -> None:
    """Refuse lossy and unknown targets, and a self-overwriting pair."""
    source, target = Path(src), Path(dst)
    suffix = target.suffix.lower()
    if suffix in _LOSSY_TARGET_SUFFIXES:
        raise ValueError(
            f"{dst}: writing lossy formats is outside this API by policy; "
            "a metrology library does not export approximations of its "
            "own data. Convert to WAV or FLAC instead."
        )
    if suffix not in _WAV_SUFFIXES and suffix != ".flac":
        raise ValueError(
            f"{dst}: unsupported target {target.suffix!r}; convert() "
            "produces the WAV family (.wav, .wave, .bwf) and, with the "
            "[audio] extra, FLAC."
        )
    if source.resolve() == target.resolve():
        raise ValueError(f"{src}: source and destination are the same file")


def _check_resolved_subtype(resolved: str, *, flac: bool) -> None:
    """Refuse a subtype the chosen container cannot hold."""
    if flac and resolved not in ("PCM_16", "PCM_24"):
        raise ValueError(
            f"FLAC holds integer PCM up to 24 bits; subtype {resolved!r} "
            "is not available (choose PCM_16 or PCM_24)"
        )
    if not flac and resolved not in _SUBTYPE_SPEC:
        raise ValueError(
            f"unknown subtype {resolved!r}; the WAV writer offers "
            f"{sorted(_SUBTYPE_SPEC)}"
        )


def _stream_to_flac(
    target: Path,
    blocks: Iterator[np.ndarray],
    fs: int,
    channels: int,
    resolved: str,
    bext_payload: bytes | None,
) -> None:
    """Stream quantised blocks into a FLAC file, then embed the bext."""
    sf = _import_soundfile("FLAC")
    with sf.SoundFile(
        str(target), "w", samplerate=fs, channels=channels,
        subtype=resolved, format="FLAC",
    ) as out:
        for codes in blocks:
            narrowed = (
                codes.astype(np.int16) if resolved == "PCM_16"
                else np.left_shift(codes.astype(np.int32), 8)
            )
            out.write(np.ascontiguousarray(narrowed.T))
    if bext_payload is not None:
        embed_flac_bext(target, bext_payload)


def convert(
    src: str | Path,
    dst: str | Path,
    *,
    subtype: str | None = None,
    block_size: int = 1 << 20,
) -> AudioFileInfo:
    """Convert a measurement file, keeping samples, bext and sidecar whole.

    See the module docstring for the full contract. Typical uses: an
    overnight RF64 archived to FLAC at a third of the size and back
    again bit for bit; a 32-bit-float working file flattened to a
    24-bit BWF for delivery; a recorder's ADPCM listening copy expanded
    to linear WAV (with the lossy warning, and the advice to keep the
    original) so tools that only read PCM can open it.

    :param src: The file to read (any format the readers accept).
    :param dst: The file to write; the suffix picks the container (WAV
        family or, with the ``[audio]`` extra, FLAC). Lossy suffixes are
        refused by policy.
    :param subtype: Target sample format; ``None`` preserves the
        source's depth (the module docstring's mapping). An explicitly
        narrower choice quantises with the clipping count reported.
    :param block_size: Frames per streamed block; the memory ceiling.
    :return: :func:`phonometry.io.info` of the written file.
    :raises ValueError: For a lossy or unknown target suffix, source and
        destination naming the same file, a FLAC target that cannot hold
        the source without an explicit ``subtype``, or an invalid
        ``block_size``.
    :raises ImportError: If source or target needs the ``[audio]`` extra
        and it is not installed.
    """
    source, target = Path(src), Path(dst)
    _check_conversion_paths(src, dst)

    flac = target.suffix.lower() == ".flac"
    kind, bits, bext, fs, channels, frames = _source_traits(source)
    resolved = (
        _default_subtype(kind, bits, flac=flac) if subtype is None else subtype
    )
    _check_resolved_subtype(resolved, flac=flac)

    # The conversion's own CodingHistory line, appended under the carried
    # trail. The resolver keys on a Signal's provenance, so the carried
    # chunk rides a minimal one; it never invents a chunk for a bext-less
    # source, and the zero-sample probe only supplies the channel count
    # for the R98 M= mode.
    probe = np.zeros((channels, 0))
    carrier = Signal(data=np.zeros((channels, 1)), fs=fs, provenance=bext)
    bext_payload = _resolve_bext(
        carrier, None, probe, fs, resolved,
        algorithm="FLAC" if flac else "PCM",
    )

    blocks = _quantised_blocks(source, target, resolved, block_size,
                               frames * channels)
    if flac:
        _stream_to_flac(target, blocks, fs, channels, resolved, bext_payload)
    else:
        write_wav_stream(
            target, blocks, fs, channels, resolved, frames,
            metadata_chunks=(
                frame_riff_chunk(b"bext", bext_payload)
                if bext_payload is not None else b""
            ),
            # A lossy container's declared frame count is a promise its
            # decoder need not keep (decoder delay and padding), so the
            # writer reconciles the header with what actually streamed.
            frames_are_estimate=kind == "decoded",
        )

    source_sidecar = read_sidecar(source)
    if source_sidecar is not None:
        # Validated just above; carried byte for byte, the most faithful copy.
        shutil.copyfile(sidecar_path(source), sidecar_path(target))
    return info(target)
