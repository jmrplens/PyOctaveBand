#  Copyright (c) 2026. Jose Manuel Requena Plens
r"""RIFF/RF64 chunk walker for WAVE measurement files, in pure ``struct``.

:mod:`scipy.io.wavfile` decodes the samples of every WAV variant a
measurement chain produces (PCM 1-64 bit, IEEE float, WAVE_FORMAT_EXTENSIBLE,
RF64), but it steps over every chunk it does not need with a
``WavFileWarning`` -- and the skipped chunks are exactly where a recording's
metadata lives. The ``bext`` broadcast extension that field recorders and
broadcast tools write (originator, sample-accurate midnight timecode, coding
history) is the only standardised provenance vehicle measurement equipment
already emits; the EXTENSIBLE ``fmt`` fields scipy discards after resolving
the format tag (channel mask, valid bits) say which loudspeaker position each
channel is and how many of a container's bits are real; ``cue`` points mark
events; ``iXML`` carries production metadata. This module walks the file once
with :mod:`struct` alone and keeps all of it, so the base install
(NumPy + SciPy, no new dependency) reads a Broadcast Wave file whole.

Layout facts and their sources:

* **RIFF/WAVE**: a little-endian sequence of ``(fourcc, uint32 size)`` chunk
  headers, each payload padded to an even byte boundary (the pad byte is not
  counted in ``size``). From the Microsoft/IBM *Multimedia Programming
  Interface and Data Specifications 1.0* (1991), the document that defined
  the format.
* **``fmt``**: ``wFormatTag, nChannels, nSamplesPerSec, nAvgBytesPerSec,
  nBlockAlign, wBitsPerSample`` as ``<HHIIHH``. ``WAVE_FORMAT_EXTENSIBLE``
  (tag 0xFFFE, from Microsoft's ``mmreg.h``/``ksmedia.h``) appends
  ``cbSize=22``: ``wValidBitsPerSample`` (how many of the container's bits
  carry signal, e.g. 20 valid bits in a 24-bit container),
  ``dwChannelMask`` (a bit per loudspeaker position, see
  :data:`CHANNEL_MASK_LABELS`) and a 16-byte ``SubFormat`` GUID whose first
  two bytes are the real format tag when the GUID has the standard media
  tail (``????????-0000-0010-8000-00AA00389B71``). Microsoft's guidance
  requires 0xFFFE whenever there are more than two channels, more than
  16 bits, or a channel mask, so multichannel and 24-bit field recordings
  routinely arrive in this form.
* **RF64/BW64** (EBU Tech 3306; superseded by ITU-R BS.2088, whose files use
  the ``BW64`` fourcc): the opening fourcc changes from ``RIFF`` and a
  mandatory ``ds64`` chunk, first after ``WAVE``, carries the real sizes as
  64-bit integers (``riffSize, dataSize, sampleCount`` then a table of
  further oversized chunks); the 32-bit size fields it replaces are set to
  0xFFFFFFFF. Field recorders auto-promote to RF64 when a take passes
  4 GiB -- under 4 h of stereo 24-bit/48 kHz -- so a long environmental
  recording is RF64, not an exotic case.
* **``fact``**: a single ``uint32`` number of frames (samples per channel),
  mandatory for compressed formats where ``data size / block align`` no
  longer counts frames.
* **``bext`` v2** (EBU Tech 3285, 2011): a fixed 602-byte prefix followed by
  the variable-length ASCII ``CodingHistory``; the field-by-field offset
  table lives in the :class:`BroadcastMetadata` docstring.
* **``cue``**: ``dwCuePoints`` then 24-byte records ``dwName, dwPosition,
  fccChunk, dwChunkStart, dwBlockStart, dwSampleOffset`` (same 1991
  specification as RIFF itself).

The walker never touches the sample data: it records the ``data`` payload's
offset and true size (via ``ds64`` for RF64) and leaves decoding to the
reader, so ``info()`` on a multi-gigabyte overnight recording costs a few
kilobytes of header reads.
"""

from __future__ import annotations

import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from typing import BinaryIO

#: RIFF format tags (``wFormatTag`` in ``fmt``, from Microsoft ``mmreg.h``).
WAVE_FORMAT_PCM = 0x0001
WAVE_FORMAT_ADPCM = 0x0002
WAVE_FORMAT_IEEE_FLOAT = 0x0003
WAVE_FORMAT_ALAW = 0x0006
WAVE_FORMAT_MULAW = 0x0007
WAVE_FORMAT_IMA_ADPCM = 0x0011
WAVE_FORMAT_GSM610 = 0x0031
WAVE_FORMAT_MPEGLAYER3 = 0x0055
WAVE_FORMAT_EXTENSIBLE = 0xFFFE

#: Format tags whose decoder reconstructs an approximation of the recorded
#: waveform: companded 8-bit telephone codecs, block ADPCM (what an NTi XL2
#: writes by default), GSM and MP3-in-WAV. Reading them is possible (via the
#: ``[audio]`` extra) but levels computed from them are not metrologically
#: defensible, which is why the reader warns.
LOSSY_FORMAT_TAGS: frozenset[int] = frozenset(
    {
        WAVE_FORMAT_ADPCM,
        WAVE_FORMAT_ALAW,
        WAVE_FORMAT_MULAW,
        WAVE_FORMAT_IMA_ADPCM,
        WAVE_FORMAT_GSM610,
        WAVE_FORMAT_MPEGLAYER3,
    }
)

_FORMAT_TAG_NAMES: dict[int, str] = {
    WAVE_FORMAT_PCM: "PCM",
    WAVE_FORMAT_ADPCM: "MS ADPCM",
    WAVE_FORMAT_IEEE_FLOAT: "IEEE float",
    WAVE_FORMAT_ALAW: "A-law",
    WAVE_FORMAT_MULAW: "mu-law",
    WAVE_FORMAT_IMA_ADPCM: "IMA ADPCM",
    WAVE_FORMAT_GSM610: "GSM 6.10",
    WAVE_FORMAT_MPEGLAYER3: "MPEG Layer III",
}

#: Tail (bytes 2-15) of the KSDATAFORMAT_SUBTYPE media GUIDs: a SubFormat of
#: ``XXXX0000-0000-0010-8000-00AA00389B71`` encodes plain format tag XXXX
#: (little-endian) in an EXTENSIBLE header (Microsoft ``ksmedia.h``).
_KSDATAFORMAT_TAIL = b"\x00\x00\x00\x00\x10\x00\x80\x00\x00\xaa\x00\x38\x9b\x71"

#: Tail (bytes 4-15) of ``SUBTYPE_AMBISONIC_B_FORMAT_PCM``/``_IEEE_FLOAT``
#: ({0000000n-0721-11D3-8644-C8C1CA000000}, n = 1 PCM, 3 float; Data2/Data3
#: little-endian on the wire): the ``.amb`` convention for Furse-Malham
#: B-format. The samples are ordinary PCM/float; only the channel meaning
#: differs, so the tag resolves to the plain codec and the ambisonic identity
#: is kept in :attr:`FormatChunk.ambisonic`.
_AMBISONIC_TAIL = b"\x21\x07\xd3\x11\x86\x44\xc8\xc1\xca\x00\x00\x00"

#: ``dwChannelMask`` bit -> loudspeaker position, in bit order (Microsoft
#: ``mmreg.h`` SPEAKER_* constants). A set bit assigns the next interleaved
#: channel to that position, lowest bit first, so a mask of 0x3 labels a
#: stereo pair front left, front right.
CHANNEL_MASK_LABELS: tuple[str, ...] = (
    "FL",
    "FR",
    "FC",
    "LFE",
    "BL",
    "BR",
    "FLC",
    "FRC",
    "BC",
    "SL",
    "SR",
    "TC",
    "TFL",
    "TFC",
    "TFR",
    "TBL",
    "TBC",
    "TBR",
)

#: ``struct`` layout of the fixed part of ``bext`` v2 (EBU Tech 3285, 2011).
#: 602 bytes; every version writes the full prefix, later versions carve
#: their new fields out of what was reserved space, which is why one layout
#: reads v0, v1 and v2 chunks alike.
_BEXT_FIXED = struct.Struct("<256s32s32s10s8sIIH64shhhhh180s")

#: EBU Tech 3285 sentinel for a loudness field that carries no value: the
#: maximum positive ``int16``. Tech 3285 requires unset loudness fields of a
#: version 2 chunk to hold 0x7FFF, and the reader maps it back to ``None``
#: rather than reporting an absurd 327.67 LUFS.
_LOUDNESS_UNSET = 0x7FFF

#: The ``bext`` version (EBU Tech 3285 v2, 2011) from which the five
#: loudness fields carry values; below it those bytes are still reserved
#: space and the reader returns the fields as ``None``.
_BEXT_LOUDNESS_MIN_VERSION = 2

#: Chunk IDs (fourccs) of the wire vocabulary the walker speaks. ``fmt``
#: and ``cue`` carry the pad space RIFF requires to make 4 bytes (a
#: re-typed ``b"fmt"`` would silently never match) and ``iXML``'s mixed
#: case is likewise the spec's own.
_FMT_CHUNK = b"fmt "  # wFormatTag/channels/rate/bits (1991 RIFF spec)
_DS64_CHUNK = b"ds64"  # RF64/BW64 64-bit sizes (EBU Tech 3306)
_BEXT_CHUNK = b"bext"  # EBU Tech 3285 broadcast extension
_FACT_CHUNK = b"fact"  # uint32 frame count, mandatory for compressed formats
_CUE_CHUNK = b"cue "  # event markers a recorder writes (1991 RIFF spec)
_DATA_CHUNK = b"data"  # sample payload: located, deliberately never read
_IXML_CHUNK = b"iXML"  # production metadata, recorded only as a presence flag

#: RIFF form-type fourcc at bytes 8-12 of the prelude: declares that this
#: RIFF/RF64/BW64 container holds a WAVE form (1991 Multimedia Programming
#: Interface spec).
_WAVE_FORM_TYPE = b"WAVE"

#: Byte size of the RIFF prelude: 4-byte container fourcc, uint32 outer
#: size, 4-byte ``WAVE`` form type.
_RIFF_PRELUDE_BYTES = 12

#: Byte size of one RIFF chunk header, the ``(fourcc, uint32 size)`` pair.
_CHUNK_HEADER_BYTES = 8

#: Byte width of one little-endian ``uint32`` field, e.g. the cue chunk's
#: leading record count or the ``fact`` chunk's frame count.
_UINT32_BYTES = 4

#: Byte size of the six mandatory ``fmt`` fields (``<HHIIHH``), the
#: minimum legal ``fmt`` chunk payload (1991 RIFF spec).
_FMT_BASE_BYTES = 16

#: Minimum ``fmt`` payload for WAVE_FORMAT_EXTENSIBLE: the 16-byte base
#: plus the 2-byte ``cbSize`` plus the 22-byte extension (valid bits,
#: channel mask, 16-byte SubFormat GUID).
_FMT_EXTENSIBLE_BYTES = 40

#: Byte size of the three mandatory 64-bit fields of a ``ds64`` chunk
#: (``riffSize, dataSize, sampleCount`` as ``<QQQ``, EBU Tech 3306);
#: distinct from the 24-byte cue-point record, which must not share it.
_DS64_FIXED_BYTES = 24

#: The RF64 size sentinel of EBU Tech 3306: a 32-bit chunk-size field set
#: to 0xFFFFFFFF defers the true 64-bit size to the ``ds64`` chunk.
_RF64_SIZE_SENTINEL = 0xFFFFFFFF


@dataclass(frozen=True)
class FormatChunk:
    """Decoded ``fmt`` chunk, including the EXTENSIBLE fields scipy drops.

    ``format_tag`` is the raw ``wFormatTag``; for a ``WAVE_FORMAT_EXTENSIBLE``
    header :attr:`resolved_tag` is the codec extracted from the SubFormat
    GUID, and equals ``format_tag`` otherwise. ``bits_per_sample`` is the
    container width; ``valid_bits`` (EXTENSIBLE only) how many of those bits
    carry signal -- e.g. a 20-bit converter recorded into a 24-bit container
    reports ``bits_per_sample=24, valid_bits=20``.
    """

    format_tag: int
    channels: int
    fs: int
    byte_rate: int
    block_align: int
    bits_per_sample: int
    valid_bits: int | None = None
    channel_mask: int | None = None
    subformat: bytes | None = None

    @property
    def resolved_tag(self) -> int:
        """The effective codec tag, seen through an EXTENSIBLE GUID.

        A SubFormat GUID with the standard media tail encodes the plain
        format tag in its first two little-endian bytes; the ambisonic
        B-format GUIDs use their own tail but the same first-bytes
        convention (0x0001 PCM, 0x0003 IEEE float). An unrecognised GUID
        keeps the raw 0xFFFE so callers fail loudly instead of misreading.
        """
        if self.format_tag != WAVE_FORMAT_EXTENSIBLE or self.subformat is None:
            return self.format_tag
        if self.subformat[2:] == _KSDATAFORMAT_TAIL or self.ambisonic:
            return int.from_bytes(self.subformat[:2], "little")
        return self.format_tag

    @property
    def ambisonic(self) -> bool:
        """Whether the SubFormat GUID marks Furse-Malham B-format (``.amb``)."""
        return self.subformat is not None and self.subformat[4:] == _AMBISONIC_TAIL

    @property
    def format_name(self) -> str:
        """Human-readable codec name for the resolved tag."""
        name = _FORMAT_TAG_NAMES.get(
            self.resolved_tag, f"tag 0x{self.resolved_tag:04X}"
        )
        return f"ambisonic B-format {name}" if self.ambisonic else name

    @property
    def lossy(self) -> bool:
        """Whether the resolved codec cannot be trusted with the waveform.

        True for the transcribed lossy tags of :data:`LOSSY_FORMAT_TAGS`,
        and equally for any tag outside that transcription that is not
        linear PCM or IEEE float: the RIFF registry holds hundreds of
        codecs (MPEG audio 0x0050, AC-3, WMA, ...) and nearly all of them
        are lossy, so an unrecognised codec fails closed as suspect
        instead of passing as clean. The bias is deliberate: reporting
        the rare lossless stranger (WMA Lossless, 0x0163) as lossy
        provokes scrutiny of a file this library cannot vouch for, where
        the opposite error would launder an MPEG recording into a
        defensible-looking level.
        """
        return self.resolved_tag not in (WAVE_FORMAT_PCM, WAVE_FORMAT_IEEE_FLOAT)

    def channel_labels(self) -> tuple[str, ...] | None:
        """Loudspeaker-position labels from ``dwChannelMask``, if usable.

        Returns one label per channel, lowest mask bit first, exactly when
        the mask's population count matches the channel count; a zero mask
        (which EXTENSIBLE files legitimately write to say "no positions
        assigned") or a mismatched one yields ``None`` rather than a wrong
        assignment.
        """
        if not self.channel_mask:
            return None
        labels = [
            label
            for bit, label in enumerate(CHANNEL_MASK_LABELS)
            if self.channel_mask >> bit & 1
        ]
        if len(labels) != self.channels:
            return None
        return tuple(labels)


@dataclass(frozen=True)
class BroadcastMetadata:
    r"""The ``bext`` broadcast extension chunk of EBU Tech 3285 (v2, 2011).

    Field-by-field transcription of the Tech 3285 ``BROADCAST_EXT``
    structure. Offsets are into the chunk payload; the fixed part is 602
    bytes in *every* version, because each revision claimed its new fields
    from the tail of the original reserved block:

    =======  ====  =============================  ==========================
    Offset   Size  Field                          Held here as
    =======  ====  =============================  ==========================
    0        256   Description                    ``description``
    256      32    Originator                     ``originator``
    288      32    OriginatorReference            ``originator_reference``
    320      10    OriginationDate (yyyy-mm-dd)   ``origination_date``
    330      8     OriginationTime (hh-mm-ss)     ``origination_time``
    338      4     TimeReferenceLow               ``time_reference`` (low)
    342      4     TimeReferenceHigh              ``time_reference`` (high)
    346      2     Version (0, 1 or 2)            ``version``
    348      64    UMID (SMPTE ST 330), v >= 1    ``umid``
    412      2     LoudnessValue, v2              ``loudness_value``
    414      2     LoudnessRange, v2              ``loudness_range``
    416      2     MaxTruePeakLevel, v2           ``max_true_peak_level``
    418      2     MaxMomentaryLoudness, v2       ``max_momentary_loudness``
    420      2     MaxShortTermLoudness, v2       ``max_short_term_loudness``
    422      180   Reserved (zeros)               (dropped)
    602      var   CodingHistory (EBU R98)        ``coding_history``
    =======  ====  =============================  ==========================

    * ``time_reference`` is the two halves joined into one unsigned 64-bit
      count of **samples since midnight** of the origination date: the
      absolute instant of the file's first sample at single-sample
      resolution, which is what ties an environmental recording to wall
      time. Divide by the sample rate for seconds since midnight.
    * The five v2 loudness fields are stored on disk as the int16 of
      100 x value rounded with ties away from zero, the Tech 3285 2.4
      rule (LUFS for the loudness values and
      maxima, LU for the range, dBTP for the true peak); they are returned
      already divided by 100. Tech 3285 fills an unset field with 0x7FFF,
      which is returned as ``None``, as are all five for ``version < 2``
      (where those bytes are still reserved space). ``umid`` is likewise
      ``None`` for ``version < 1``.
    * The strings are fixed-size ASCII fields padded with NUL; padding is
      stripped and stray non-ASCII bytes are decoded permissively
      (latin-1) rather than making an otherwise sound file unreadable.
      ``coding_history`` keeps its internal CR/LF line structure (one line
      per coding step, EBU R98 format ``A=PCM,F=48000,W=24,M=stereo,T=...``);
      only trailing NUL padding and whitespace are stripped.
    """

    description: str
    originator: str
    originator_reference: str
    origination_date: str
    origination_time: str
    time_reference: int
    version: int
    umid: bytes | None
    loudness_value: float | None
    loudness_range: float | None
    max_true_peak_level: float | None
    max_momentary_loudness: float | None
    max_short_term_loudness: float | None
    coding_history: str


@dataclass(frozen=True)
class CuePoint:
    """One record of the ``cue`` chunk (1991 RIFF specification).

    For the plain ``data``-chunk case (no playlist, no wave list), which is
    what marker-writing recorders produce, ``sample_offset`` is the marker's
    position in samples from the start of the data.
    """

    cue_id: int
    position: int
    chunk_id: bytes
    chunk_start: int
    block_start: int
    sample_offset: int


@dataclass(frozen=True)
class Ds64Chunk:
    """The 64-bit sizes of an RF64/BW64 file (EBU Tech 3306 ``ds64``).

    ``sample_count`` mirrors the ``fact`` chunk's frame count for compressed
    payloads; writers of PCM files may leave it zero, so it is a fallback,
    not an authority, for frame counting.
    """

    riff_size: int
    data_size: int
    sample_count: int


#: The container the RIFF prelude declares: classic 32-bit "WAV" (``RIFF``
#: fourcc), or the 64-bit "RF64" (EBU Tech 3306) / "BW64" (ITU-R BS.2088).
WavContainer = Literal["WAV", "RF64", "BW64"]


@dataclass(frozen=True)
class WavChunks:
    """Everything the walker gathered from one pass over the headers."""

    container: WavContainer
    fmt: FormatChunk
    data_offset: int
    data_size: int
    ds64: Ds64Chunk | None = None
    bext: BroadcastMetadata | None = None
    fact_frames: int | None = None
    cue_points: tuple[CuePoint, ...] = ()
    has_ixml: bool = False

    @property
    def frames(self) -> int:
        """Frames (samples per channel) in the data payload.

        For PCM and IEEE float the exact count is
        ``data size / block align``. For compressed codecs a block no longer
        holds one frame per ``block_align`` bytes, so the ``fact`` chunk
        (mandatory there) is believed first, then a nonzero ``ds64``
        sample count, with the block-align quotient as the last resort.
        """
        by_blocks = (
            self.data_size // self.fmt.block_align if self.fmt.block_align else 0
        )
        if self.fmt.resolved_tag in (WAVE_FORMAT_PCM, WAVE_FORMAT_IEEE_FLOAT):
            return by_blocks
        if self.fact_frames:
            return self.fact_frames
        if self.ds64 is not None and self.ds64.sample_count:
            return self.ds64.sample_count
        return by_blocks


def _ascii_field(raw: bytes) -> str:
    """Decode a fixed-size NUL-padded ASCII field permissively."""
    return raw.split(b"\x00", 1)[0].decode("latin-1")


def _loudness(raw: int) -> float | None:
    """Decode one ``int16 = 100 x value`` loudness field (unset -> None)."""
    return None if raw == _LOUDNESS_UNSET else raw / 100.0


def _parse_bext(payload: bytes) -> BroadcastMetadata:
    """Decode a ``bext`` payload laid out per the class docstring's table."""
    if len(payload) < _BEXT_FIXED.size:
        msg = (
            f"bext chunk is {len(payload)} bytes; the EBU Tech 3285 fixed "
            f"part is {_BEXT_FIXED.size}"
        )
        raise ValueError(msg)
    (
        description,
        originator,
        originator_reference,
        origination_date,
        origination_time,
        time_low,
        time_high,
        version,
        umid,
        loudness_value,
        loudness_range,
        max_true_peak,
        max_momentary,
        max_short_term,
        _reserved,
    ) = _BEXT_FIXED.unpack_from(payload)
    coding_history = (
        payload[_BEXT_FIXED.size :].split(b"\x00", 1)[0].decode("latin-1").rstrip()
    )
    v2 = version >= _BEXT_LOUDNESS_MIN_VERSION
    return BroadcastMetadata(
        description=_ascii_field(description),
        originator=_ascii_field(originator),
        originator_reference=_ascii_field(originator_reference),
        origination_date=_ascii_field(origination_date),
        origination_time=_ascii_field(origination_time),
        time_reference=time_high << 32 | time_low,
        version=version,
        umid=umid if version >= 1 else None,
        loudness_value=_loudness(loudness_value) if v2 else None,
        loudness_range=_loudness(loudness_range) if v2 else None,
        max_true_peak_level=_loudness(max_true_peak) if v2 else None,
        max_momentary_loudness=_loudness(max_momentary) if v2 else None,
        max_short_term_loudness=_loudness(max_short_term) if v2 else None,
        coding_history=coding_history,
    )


def _parse_fmt(payload: bytes) -> FormatChunk:
    """Decode ``fmt``, including the EXTENSIBLE extension when present."""
    if len(payload) < _FMT_BASE_BYTES:
        msg = f"fmt chunk is {len(payload)} bytes; at least 16 required"
        raise ValueError(msg)
    tag, channels, fs, byte_rate, block_align, bits = struct.unpack_from(
        "<HHIIHH", payload
    )
    valid_bits: int | None = None
    channel_mask: int | None = None
    subformat: bytes | None = None
    if tag == WAVE_FORMAT_EXTENSIBLE:
        if len(payload) < _FMT_EXTENSIBLE_BYTES:
            msg = (
                f"WAVE_FORMAT_EXTENSIBLE fmt chunk is {len(payload)} bytes; "
                "the 22-byte extension requires 40"
            )
            raise ValueError(msg)
        # cbSize (uint16 at offset 16) precedes the extension proper.
        valid_bits, channel_mask = struct.unpack_from("<HI", payload, 18)
        subformat = payload[24:_FMT_EXTENSIBLE_BYTES]
    return FormatChunk(
        format_tag=tag,
        channels=channels,
        fs=fs,
        byte_rate=byte_rate,
        block_align=block_align,
        bits_per_sample=bits,
        valid_bits=valid_bits,
        channel_mask=channel_mask,
        subformat=subformat,
    )


def _parse_cue(payload: bytes) -> tuple[CuePoint, ...]:
    """Decode the ``cue`` chunk's fixed-size records.

    The record count is checked against the payload's actual length
    before any record is unpacked, so a chunk that declares more points
    than it holds raises the module's documented :exc:`ValueError`
    instead of letting :func:`struct.unpack_from` escape with its own
    :exc:`struct.error` (which a caller filtering on ``ValueError``, the
    type every other malformed-chunk path here raises, would not catch).
    """
    if len(payload) < _UINT32_BYTES:
        msg = f"cue chunk is {len(payload)} bytes; its uint32 record count requires 4"
        raise ValueError(msg)
    (count,) = struct.unpack_from("<I", payload)
    needed = _UINT32_BYTES + 24 * count
    if len(payload) < needed:
        msg = (
            f"cue chunk declares {count} points but is {len(payload)} "
            f"bytes; the 24-byte records require {needed}"
        )
        raise ValueError(msg)
    points = []
    for i in range(count):
        cue_id, position, chunk_id, chunk_start, block_start, sample_offset = (
            struct.unpack_from("<II4sIII", payload, _UINT32_BYTES + 24 * i)
        )
        points.append(
            CuePoint(
                cue_id=cue_id,
                position=position,
                chunk_id=chunk_id,
                chunk_start=chunk_start,
                block_start=block_start,
                sample_offset=sample_offset,
            )
        )
    return tuple(points)


def _parse_ds64(payload: bytes, path: str | Path) -> Ds64Chunk:
    """Decode the three mandatory 64-bit sizes of a ``ds64`` chunk."""
    if len(payload) < _DS64_FIXED_BYTES:
        msg = (
            f"{path}: ds64 chunk is {len(payload)} bytes; the "
            "riffSize, dataSize and sampleCount fields require 24"
        )
        raise ValueError(msg)
    riff_size, chunk_data_size, sample_count = struct.unpack_from("<QQQ", payload)
    return Ds64Chunk(
        riff_size=riff_size,
        data_size=chunk_data_size,
        sample_count=sample_count,
    )


def _parse_fact(payload: bytes, path: str | Path) -> int:
    """Decode the ``fact`` chunk's single uint32 frame count."""
    if len(payload) < _UINT32_BYTES:
        msg = (
            f"{path}: fact chunk is {len(payload)} bytes; its "
            "uint32 frame count requires 4"
        )
        raise ValueError(msg)
    (fact_frames,) = struct.unpack_from("<I", payload)
    return int(fact_frames)


def _read_wave_header(fh: BinaryIO, path: str | Path) -> WavContainer:
    """Check the 12-byte RIFF prelude and name the container it declares."""
    header = fh.read(_RIFF_PRELUDE_BYTES)
    if len(header) < _RIFF_PRELUDE_BYTES or header[8:12] != _WAVE_FORM_TYPE:
        msg = f"{path}: not a WAVE file"
        raise ValueError(msg)
    fourcc = header[:4]
    containers: dict[bytes, WavContainer] = {
        b"RIFF": "WAV",
        b"RF64": "RF64",
        b"BW64": "BW64",
    }
    if fourcc not in containers:
        msg = (
            f"{path}: unknown container fourcc {fourcc!r} (expected RIFF, RF64 or BW64)"
        )
        raise ValueError(msg)
    return containers[fourcc]


def _check_chunk_extent(
    chunk_id: bytes, size: int, offset: int, file_size: int, path: str | Path
) -> None:
    """Refuse a chunk whose declared payload runs past the end of the file.

    Checked before a byte of the payload is read or skipped: a truncated
    file or a lying size field must produce this clear refusal, not a
    short read interpreted somewhere deep inside a field parser (or, for
    a skipped chunk, a seek past EOF that silently succeeds). The pad
    byte is deliberately not demanded: some writers omit the final pad
    of an odd last chunk, and the payload itself is whole either way.
    """
    if offset + size > file_size:
        msg = (
            f"{path}: chunk {chunk_id!r} claims {size} bytes but only "
            f"{file_size - offset} remain in the file"
        )
        raise ValueError(msg)


def _read_chunk_payload(
    fh: BinaryIO, chunk_id: bytes, size: int, path: str | Path
) -> bytes:
    """Read one chunk's payload whole, honouring RIFF word alignment.

    The extent was already checked against the file size; the short-read
    guard stays for the race where the file shrinks under the walker.
    """
    payload = fh.read(size)
    if len(payload) < size:
        msg = (
            f"{path}: chunk {chunk_id!r} claims {size} bytes but the "
            f"file ends after {len(payload)}"
        )
        raise ValueError(msg)
    if size % 2:
        fh.seek(1, 1)  # chunks are word-aligned; the pad byte is not counted
    return payload


def _true_data_size(
    size: int, container: WavContainer, ds64: Ds64Chunk | None, path: str | Path
) -> int:
    """Resolve the ``data`` size, through ``ds64`` for the RF64 sentinel."""
    if size != _RF64_SIZE_SENTINEL or container == "WAV":
        return size
    if ds64 is None:
        msg = (
            f"{path}: RF64 data chunk defers its size to a "
            "ds64 chunk that never appeared"
        )
        raise ValueError(msg)
    return ds64.data_size


#: Chunk kinds whose payload the walker decodes. Everything else is
#: skipped by seeking, never read: a ``JUNK`` (or any unknown) chunk can
#: legitimately be huge, and loading it would break the walker's promise
#: that header inspection costs kilobytes whatever the file size.
_DECODED_CHUNKS = frozenset(
    {_FMT_CHUNK, _DS64_CHUNK, _BEXT_CHUNK, _FACT_CHUNK, _CUE_CHUNK}
)


@dataclass
class _WalkedChunks:
    """The non-``data`` chunks gathered so far, mutable while walking."""

    fmt: FormatChunk | None = None
    ds64: Ds64Chunk | None = None
    bext: BroadcastMetadata | None = None
    fact_frames: int | None = None
    cue_points: tuple[CuePoint, ...] = ()
    has_ixml: bool = False

    def absorb(self, chunk_id: bytes, payload: bytes, path: str | Path) -> None:
        """Decode one chunk of :data:`_DECODED_CHUNKS` into its slot."""
        if chunk_id == _FMT_CHUNK:
            self.fmt = _parse_fmt(payload)
        elif chunk_id == _DS64_CHUNK:
            self.ds64 = _parse_ds64(payload, path)
        elif chunk_id == _BEXT_CHUNK:
            self.bext = _parse_bext(payload)
        elif chunk_id == _FACT_CHUNK:
            self.fact_frames = _parse_fact(payload, path)
        elif chunk_id == _CUE_CHUNK:
            self.cue_points = _parse_cue(payload)


def parse_wav_chunks(path: str | Path) -> WavChunks:
    """Walk a WAV/BWF/RF64/BW64 file's chunks without reading the samples.

    One sequential pass over the chunk headers: ``fmt`` (with the
    EXTENSIBLE fields), ``ds64``, ``bext``, ``fact``, ``cue`` and the
    presence of ``iXML`` are decoded; the ``data`` payload is located
    (offset and true size, resolved through ``ds64`` when the 32-bit field
    holds the RF64 sentinel 0xFFFFFFFF) but never read, so the cost is
    independent of file size. Unknown chunks are skipped without comment
    (unlike :func:`scipy.io.wavfile.read` there is nothing to warn about,
    because nothing understood is being dropped) and without being read,
    so a huge ``JUNK`` or ``iXML`` costs a seek, not its size in memory.
    Every chunk's declared extent except ``data``'s is checked against
    the file size before its payload is touched, so a truncated file or
    a lying size field refuses loudly up front; the ``data`` extent is
    exempt on purpose (describing a header-only forgery is a feature the
    tests rely on) and is verified by the sample readers instead.

    :param path: The file to walk.
    :return: The decoded headers as a :class:`WavChunks`.
    :raises ValueError: If the file is not RIFF/RF64/BW64 + WAVE, a
        mandatory chunk is missing or malformed, a non-``data`` chunk
        declares more bytes than the file holds, or an RF64 file lacks
        the ``ds64`` its data size lives in.
    """
    seen = _WalkedChunks()
    data_offset: int | None = None
    data_size: int | None = None

    with Path(path).open("rb") as fh:
        file_size = os.fstat(fh.fileno()).st_size
        container = _read_wave_header(fh, path)
        while True:
            chunk_header = fh.read(_CHUNK_HEADER_BYTES)
            if len(chunk_header) < _CHUNK_HEADER_BYTES:
                break
            chunk_id = chunk_header[:4]
            (size,) = struct.unpack("<I", chunk_header[4:])
            if chunk_id == _DATA_CHUNK:
                size = _true_data_size(size, container, seen.ds64, path)
                data_offset = fh.tell()
                data_size = size
                # Seek, never read: the payload may be gigabytes. Its
                # extent is deliberately not checked against the file
                # either -- info() on a header-only forgery is a feature,
                # and the sample readers verify the payload before
                # decoding a frame of it.
                fh.seek(size + size % 2, 1)
                continue
            _check_chunk_extent(chunk_id, size, fh.tell(), file_size, path)
            if chunk_id in _DECODED_CHUNKS:
                seen.absorb(
                    chunk_id,
                    _read_chunk_payload(fh, chunk_id, size, path),
                    path,
                )
                continue
            if chunk_id == _IXML_CHUNK:
                seen.has_ixml = True
            # Skipped, not read: unknown chunks (and iXML, kept only as a
            # presence flag) can be arbitrarily large.
            fh.seek(size + size % 2, 1)

    if seen.fmt is None:
        msg = f"{path}: no fmt chunk found"
        raise ValueError(msg)
    if data_offset is None or data_size is None:
        msg = f"{path}: no data chunk found"
        raise ValueError(msg)
    return WavChunks(
        container=container,
        fmt=seen.fmt,
        data_offset=data_offset,
        data_size=data_size,
        ds64=seen.ds64,
        bext=seen.bext,
        fact_frames=seen.fact_frames,
        cue_points=seen.cue_points,
        has_ixml=seen.has_ixml,
    )
