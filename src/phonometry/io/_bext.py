#  Copyright (c) 2026. Jose Manuel Requena Plens
r"""
Writing the ``bext`` broadcast extension chunk (EBU Tech 3285 v2, 2011).

No maintained Python package writes ``bext`` (wavinfo only reads it;
python-soundfile does not expose libsndfile's ``SFC_SET_BROADCAST_INFO``),
so the chunk that carries a recording's provenance -- originator,
sample-accurate midnight timecode, coding history, R 128 loudness -- is
serialised here, field by field at the offsets of the Tech 3285 table
transcribed in :class:`~phonometry.io._chunks.BroadcastMetadata`. The
serialiser is the exact inverse of that reader: what one writes the other
returns, and the round trip is pinned by tests against the same
independently transcribed offset table.

Conventions the serialiser enforces, each from the Tech 3285 text:

* **Version gating.** The UMID bytes exist from version 1 and the five
  loudness fields from version 2; below those versions the bytes are
  reserved space that "shall be set to zero". Metadata that carries a
  UMID or loudness under a version too old to hold them is refused
  loudly -- writing the bytes anyway would produce a chunk that lies
  about itself, and zeroing them would silently discard data.
* **Loudness encoding.** Each field is ``int16 = round(100 x value)``.
  0x7FFF is Tech 3285's sentinel for "no value here" and is what ``None``
  serialises to, so it can never collide with a measurement; a real value
  therefore clamps to at most 0x7FFE (327.66) and at least -32768
  (-327.68), and an infinite loudness (digital silence measures
  -inf LUFS) pins to the nearest representable end rather than crashing
  the export of a silent calibration gap.
* **Strings.** Fixed-size fields, NUL-padded by :mod:`struct`; content
  that does not fit its field raises instead of truncating (a provenance
  field with the end cut off is corrupted provenance, and the caller is
  the only one who knows what to shorten). Encoding is latin-1,
  mirroring the reader's permissive decode.
* **CodingHistory is extended, never replaced.** EBU R98 defines the
  field as an audit trail: one CR/LF-terminated line per coding step in
  the ``A=<algorithm>,F=<rate>,W=<bits>,M=<mode>,T=<free text>`` grammar,
  each writer *appending* its line under those of every writer before
  it. :func:`extend_coding_history` implements exactly that append, and
  the write path runs every chunk through it -- including a chunk read
  from another recorder's file, whose history therefore survives intact
  with phonometry's line beneath it. The ``M=`` parameter only exists in
  R98 for ``mono``/``stereo``/``dual-mono``/``joint-stereo``; for other
  channel counts the parameter is omitted rather than invented.
* **Fresh chunks claim nothing.** A chunk phonometry creates from scratch
  (exporting a generated signal) sets ``Originator`` to the library and
  version, stamps the origination date and time, and leaves
  ``OriginatorReference`` empty: EBU R99's USID wants country and
  organisation codes this library cannot know, and an invented unique ID
  is worse than an absent one. ``TimeReference`` stays zero -- it means
  "first sample at midnight", and pretending to know the capture instant
  of a signal that was never captured would be false provenance.

The five R 128 loudness fields can be *measured* at export by the
library's own ITU-R BS.1770 implementation
(:func:`phonometry.broadcast.program_loudness`), producing a chunk no
generic I/O tool can write: integrated loudness, range, true peak and the
momentary/short-term maxima of the actual samples being written. Per the
module's design decisions the measurement runs only when asked for
(``bext="loudness"``): it costs a full pass over the signal.
"""

from __future__ import annotations

import math
from dataclasses import replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from ._chunks import _BEXT_FIXED, _LOUDNESS_UNSET, BroadcastMetadata

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

#: (attribute, Tech 3285 field name, size) of the fixed-size string fields.
_STRING_FIELDS: tuple[tuple[str, str, int], ...] = (
    ("description", "Description", 256),
    ("originator", "Originator", 32),
    ("originator_reference", "OriginatorReference", 32),
    ("origination_date", "OriginationDate", 10),
    ("origination_time", "OriginationTime", 8),
)


def _encode_loudness(value: float | None) -> int:
    """Encode one loudness field as ``int16 = round(100 x value)``.

    ``None`` becomes the Tech 3285 unset sentinel 0x7FFF; finite values
    clamp to [-32768, 0x7FFE] so they can never alias the sentinel; an
    infinity (silence measures -inf LUFS) pins to the nearest end.
    """
    if value is None:
        return _LOUDNESS_UNSET
    if not math.isfinite(value):
        return -0x8000 if value < 0 else _LOUDNESS_UNSET - 1
    return max(-0x8000, min(_LOUDNESS_UNSET - 1, round(value * 100)))


def serialize_bext(meta: BroadcastMetadata) -> bytes:
    """Serialise the chunk payload, the exact inverse of the reader.

    Fixed 602-byte prefix at the offsets of the Tech 3285 table in the
    :class:`~phonometry.io._chunks.BroadcastMetadata` docstring, then the
    variable-length CodingHistory. See the module docstring for the
    version gating, the loudness encoding and the string policy.

    :raises ValueError: If a string overflows its field, the version
        cannot hold a field that carries data (UMID needs version >= 1,
        loudness version >= 2), or ``time_reference`` does not fit its
        unsigned 64 bits.
    """
    encoded: list[bytes] = []
    for attribute, name, size in _STRING_FIELDS:
        raw = str(getattr(meta, attribute)).encode("latin-1")
        if len(raw) > size:
            raise ValueError(
                f"bext {name} is {len(raw)} bytes; the Tech 3285 field "
                f"holds {size}"
            )
        encoded.append(raw)
    if not 0 <= meta.time_reference < 2**64:
        raise ValueError(
            "bext TimeReference is an unsigned 64-bit sample count; got "
            f"{meta.time_reference}"
        )
    if meta.umid is not None and meta.version < 1:
        raise ValueError(
            f"bext version {meta.version} has no UMID field (version >= 1); "
            "raise the version instead of losing the UMID"
        )
    loudness = (
        meta.loudness_value, meta.loudness_range, meta.max_true_peak_level,
        meta.max_momentary_loudness, meta.max_short_term_loudness,
    )
    if meta.version < 2 and any(value is not None for value in loudness):
        raise ValueError(
            f"bext version {meta.version} has no loudness fields "
            "(version >= 2); raise the version instead of losing them"
        )
    umid = meta.umid or b""
    if len(umid) > 64:
        raise ValueError(f"bext UMID is {len(umid)} bytes; the field holds 64")
    history = meta.coding_history.encode("latin-1")
    return _BEXT_FIXED.pack(
        *encoded,
        meta.time_reference & 0xFFFFFFFF,
        meta.time_reference >> 32,
        meta.version,
        umid,
        *(
            (_encode_loudness(value) for value in loudness)
            if meta.version >= 2
            else (0, 0, 0, 0, 0)
        ),
        b"",  # the 180 reserved bytes: zeros in every version
    ) + history


def coding_history_line(
    *, fs: int, bits: int, channels: int, algorithm: str = "PCM"
) -> str:
    """One EBU R98 CodingHistory line for a file phonometry is writing.

    The ``A=<algorithm>,F=<sampling rate>,W=<word length>,M=<mode>,
    T=<free text>`` grammar of EBU R98, with ``T=`` naming the writer the
    way R98's own examples do. ``M=`` exists only for the R98 modes
    (mono/stereo/...); other channel counts omit the parameter rather
    than invent a value the grammar does not define.
    """
    from .._version import __version__

    mode = {1: "mono", 2: "stereo"}.get(channels)
    parts = [f"A={algorithm}", f"F={fs}", f"W={bits}"]
    if mode is not None:
        parts.append(f"M={mode}")
    parts.append(f"T=phonometry {__version__}")
    return ",".join(parts)


def extend_coding_history(existing: str, line: str) -> str:
    """Append one line to a CodingHistory, preserving everything above it.

    Each line ends CR/LF per EBU R98; the reader strips the trailing
    terminator, so it is restored here before the new line goes beneath
    the existing audit trail.
    """
    if not existing:
        return line + "\r\n"
    return existing.rstrip("\r\n") + "\r\n" + line + "\r\n"


def fresh_metadata() -> BroadcastMetadata:
    """A version 2 chunk for a file that has no provenance to carry.

    Stamps the library as Originator with the origination instant, and
    deliberately claims nothing else -- see the module docstring for why
    OriginatorReference stays empty and TimeReference zero.
    """
    from .._version import __version__

    # Local wall-clock time: Tech 3285's OriginationDate/Time carry the
    # local instant of creation, with no timezone field to say otherwise.
    now = datetime.now(UTC).astimezone()
    return BroadcastMetadata(
        description="",
        originator=f"phonometry {__version__}"[:32],
        originator_reference="",
        origination_date=now.strftime("%Y-%m-%d"),
        origination_time=now.strftime("%H-%M-%S"),
        time_reference=0,
        version=2,
        umid=None,
        loudness_value=None,
        loudness_range=None,
        max_true_peak_level=None,
        max_momentary_loudness=None,
        max_short_term_loudness=None,
        coding_history="",
    )


def with_measured_loudness(
    meta: BroadcastMetadata, data: NDArray[np.float64], fs: int
) -> BroadcastMetadata:
    """Fill the five R 128 fields by measuring the samples being written.

    Runs the library's own ITU-R BS.1770 / EBU R 128 implementation
    (:func:`phonometry.broadcast.program_loudness`) over the full-scale
    data: integrated loudness (LUFS), loudness range (LU), maximum true
    peak (dBTP) and the momentary/short-term maxima (LUFS) -- measured
    values, not copied ones. The version is raised to 2, where the fields
    exist. The serialiser later quantises each to the chunk's hundredth
    resolution; the returned metadata keeps the unquantised measurement.
    """
    from ..broadcast import program_loudness

    result = program_loudness(data[0] if data.shape[0] == 1 else data, fs)
    return replace(
        meta,
        version=max(meta.version, 2),
        loudness_value=result.integrated,
        loudness_range=result.loudness_range,
        max_true_peak_level=result.true_peak,
        max_momentary_loudness=result.max_momentary,
        max_short_term_loudness=result.max_short_term,
    )
