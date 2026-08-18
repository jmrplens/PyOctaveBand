#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Broadcast Wave metadata (EBU Tech 3285 bext / ITU-R BS.2088 64-bit).

The ``bext`` chunk phonometry writes and reads is a byte-for-byte contract:
EBU Tech 3285 v2 (2011) prints the ``BROADCAST_EXT`` structure field by
field (2.3, pp. 9-10), defines the loudness integer encoding with worked
examples down to the carried hexadecimal (2.4, pp. 12-13), and gates the
UMID and loudness fields on the chunk's version (1.1, p. 8). The
CodingHistory rows it carries follow EBU R 98-1999, whose Appendix 1
prints the ``A=,F=,W=,M=,T=`` grammar and a literal 48 kHz 16-bit stereo
PCM row. Past 4 GiB the container itself changes: Recommendation
ITU-R BS.2088-2 (11/2025) specifies the 64-bit layout - the ``ds64``
chunk's DWORD sequence (Annex 1, 4.1-4.2) and the 0xFFFFFFFF sentinel
that defers a 32-bit size field to it (2.4, 3.2).

Every expected value here is read from ``tests/reference_data``'s
transcription of those documents; the computed side is a file the library
wrote (or, for the 64-bit reader rule, a minimal container assembled from
the transcribed layout) inspected at the transcribed offsets and read back
through the public reader.
"""

from __future__ import annotations

import dataclasses
import functools
import struct
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import reference_data as ref

from phonometry.io import BroadcastMetadata, info, read, write
from phonometry.io._write import build_wav_header

from ..registry import Outcome, register

_BWF = "Broadcast Wave metadata (EBU Tech 3285 / ITU-R BS.2088)"

#: 13:30:00 at 96 kHz: a first-sample count since midnight that no longer
#: fits 32 bits, so both TimeReference words must carry weight.
_TIME_REFERENCE = 13 * 3600 * 96000 + 30 * 60 * 96000  # 4 665 600 000
_UMID = bytes(range(64))

#: On-the-hundredth loudness values for the layout check, so the expected
#: int16 is exact arithmetic (value x 100) with no rounding in play.
_LOUDNESS_ON_GRID: dict[str, tuple[float, int]] = {
    "LoudnessValue": (-23.11, -2311),
    "LoudnessRange": (5.4, 540),
    "MaxTruePeakLevel": (-1.02, -102),
    "MaxMomentaryLoudness": (-19.87, -1987),
    "MaxShortTermLoudness": (-21.4, -2140),
}


def _metadata(**overrides: Any) -> BroadcastMetadata:
    """A fully-populated v2 chunk with distinctive, realistic values."""
    base = BroadcastMetadata(
        description="facade measurement, mic position 2",
        originator="NTi XL2",
        originator_reference="ESUPV0000000000000000000META0002",
        origination_date="2026-08-17",
        origination_time="13-30-00",
        time_reference=_TIME_REFERENCE,
        version=2,
        umid=_UMID,
        loudness_value=_LOUDNESS_ON_GRID["LoudnessValue"][0],
        loudness_range=_LOUDNESS_ON_GRID["LoudnessRange"][0],
        max_true_peak_level=_LOUDNESS_ON_GRID["MaxTruePeakLevel"][0],
        max_momentary_loudness=_LOUDNESS_ON_GRID["MaxMomentaryLoudness"][0],
        max_short_term_loudness=_LOUDNESS_ON_GRID["MaxShortTermLoudness"][0],
        coding_history=ref.EBU_R98_EXAMPLE2_PRIOR_LINE + "\r\n",
    )
    return dataclasses.replace(base, **overrides) if overrides else base


def _bext_payload(blob: bytes) -> bytes:
    """Extract the raw bext payload from a written file's bytes.

    A deliberately independent 10-line chunk walk (fourcc + uint32 size,
    even-byte alignment), so the bytes under test are located without the
    parser whose layout is being checked.
    """
    pos = 12
    while pos + 8 <= len(blob):
        chunk_id = blob[pos:pos + 4]
        (size,) = struct.unpack_from("<I", blob, pos + 4)
        if chunk_id == b"bext":
            return blob[pos + 8:pos + 8 + size]
        pos += 8 + size + size % 2
    raise ValueError("no bext chunk in the written file")


def _write_and_capture(
    meta: BroadcastMetadata,
    *,
    fs: int = 96000,
    channels: int = 1,
) -> tuple[bytes, BroadcastMetadata]:
    """Write a file carrying ``meta``; return (raw bext payload, reread)."""
    data = np.zeros(16 if channels == 1 else (channels, 16))
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "capture.wav"
        write(path, data, fs, subtype="PCM_16", bext=meta)
        reread = read(path).provenance
        payload = _bext_payload(path.read_bytes())
    if reread is None:
        raise ValueError("written bext did not read back")
    return payload, reread


@functools.cache
def _canonical() -> tuple[bytes, BroadcastMetadata]:
    """The fully-populated v2 file every Tech 3285 check inspects."""
    return _write_and_capture(_metadata())


def _field_bytes(payload: bytes, field: str) -> bytes:
    """Slice one fixed field out of a payload at its transcribed offset."""
    for name, offset, size in ref.TECH3285_BEXT_FIELDS:
        if name == field:
            return payload[offset:offset + size]
    raise KeyError(field)


def _int16_at(payload: bytes, field: str) -> int:
    return int.from_bytes(_field_bytes(payload, field), "little", signed=True)


def _uint16_at(payload: bytes, field: str) -> int:
    return int.from_bytes(_field_bytes(payload, field), "little")


def _expected_fixed_part() -> dict[str, bytes]:
    """The 602 bytes of the canonical chunk, assembled from the table.

    Every slice is built from the transcribed (offset, size) rows and the
    field semantics of Tech 3285 pp. 10-11, independently of the writer's
    serialiser: NUL-padded latin-1 strings, the TimeReference split into
    little-endian low/high DWORDs, 0002h in Version, the UMID verbatim,
    each loudness value as the int16 of 100 x value (all five sit on the
    hundredth grid, so the product is exact), and 180 zero bytes.
    """
    meta = _metadata()
    expected: dict[str, bytes] = {}
    for name, _offset, size in ref.TECH3285_BEXT_FIELDS:
        if name in _LOUDNESS_ON_GRID:
            expected[name] = _LOUDNESS_ON_GRID[name][1].to_bytes(
                2, "little", signed=True
            )
        elif name == "TimeReferenceLow":
            expected[name] = (_TIME_REFERENCE & 0xFFFFFFFF).to_bytes(4, "little")
        elif name == "TimeReferenceHigh":
            expected[name] = (_TIME_REFERENCE >> 32).to_bytes(4, "little")
        elif name == "Version":
            expected[name] = ref.TECH3285_VERSION_2.to_bytes(2, "little")
        elif name == "UMID":
            expected[name] = _UMID
        elif name == "Reserved":
            expected[name] = bytes(size)
        else:
            attribute = {
                "Description": "description",
                "Originator": "originator",
                "OriginatorReference": "originator_reference",
                "OriginationDate": "origination_date",
                "OriginationTime": "origination_time",
            }[name]
            expected[name] = str(getattr(meta, attribute)).encode(
                "latin-1"
            ).ljust(size, b"\x00")
    return expected


@register(
    _BWF,
    "EBU Tech 3285:2011 (2.3)",
    "bext fixed part: every field at its cumulative offset, 602 bytes",
)
def _chk_bext_layout() -> Outcome:
    payload, _ = _canonical()
    expected = _expected_fixed_part()
    matching = sum(
        1
        for name, offset, size in ref.TECH3285_BEXT_FIELDS
        if payload[offset:offset + size] == expected[name]
    )
    total = len(ref.TECH3285_BEXT_FIELDS)
    size_ok = len(payload) >= ref.TECH3285_BEXT_FIXED_SIZE
    return Outcome(
        expected=(
            f"{total} fields byte-identical at offsets 0..422, fixed part "
            f"{ref.TECH3285_BEXT_FIXED_SIZE} B"
        ),
        computed=(
            f"{matching}/{total} byte-identical, "
            f"{ref.TECH3285_BEXT_FIXED_SIZE} B + CodingHistory"
        ),
        delta="-",
        passed=matching == total and size_ok,
    )


@register(
    _BWF,
    "EBU Tech 3285:2011 (2.3)",
    "bext round trip: written metadata returns identically through the reader",
)
def _chk_bext_round_trip() -> Outcome:
    _, reread = _canonical()
    meta = _metadata()
    scalar_fields = (
        "description", "originator", "originator_reference",
        "origination_date", "origination_time", "time_reference", "version",
        "umid", "loudness_value", "loudness_range", "max_true_peak_level",
        "max_momentary_loudness", "max_short_term_loudness",
    )
    identical = sum(
        1 for f in scalar_fields if getattr(reread, f) == getattr(meta, f)
    )
    # The CodingHistory is required to differ: the writer appends its own
    # R 98 row beneath the prior trail (checked in its own rows below).
    extended = reread.coding_history.startswith(ref.EBU_R98_EXAMPLE2_PRIOR_LINE)
    return Outcome(
        expected=f"{len(scalar_fields)} fields identical, CodingHistory extended",
        computed=(
            f"{identical}/{len(scalar_fields)} identical, history "
            f"{'extended' if extended else 'NOT extended'}"
        ),
        delta="-",
        passed=identical == len(scalar_fields) and extended,
    )


def _loudness_examples_outcome(rows: tuple[int, ...], fields: tuple[str, ...]) -> Outcome:
    """Write one file carrying three worked-example values; compare on disk."""
    examples = [ref.TECH3285_LOUDNESS_EXAMPLES[i] for i in rows]
    overrides = {
        "loudness_value": None, "loudness_range": None,
        "max_true_peak_level": None, "max_momentary_loudness": None,
        "max_short_term_loudness": None,
    }
    attribute = {
        "LoudnessValue": "loudness_value",
        "LoudnessRange": "loudness_range",
        "MaxTruePeakLevel": "max_true_peak_level",
        "MaxMomentaryLoudness": "max_momentary_loudness",
        "MaxShortTermLoudness": "max_short_term_loudness",
    }
    for (value, _dec, _hexa), field in zip(examples, fields):
        overrides[attribute[field]] = value
    payload, _ = _write_and_capture(_metadata(**overrides))
    stored = [_int16_at(payload, field) for field in fields]
    expected_txt = ", ".join(
        f"{value} -> {hexa:04X}h ({dec})" for value, dec, hexa in examples
    )
    computed_txt = ", ".join(
        f"{got & 0xFFFF:04X}h ({got})" for got in stored
    )
    passed = all(got == dec for got, (_v, dec, _h) in zip(stored, examples))
    return Outcome(expected=expected_txt, computed=computed_txt, delta="-", passed=passed)


@register(
    _BWF,
    "EBU Tech 3285:2011 (2.4)",
    "Loudness int16 = 100 x value, ties away from zero: negative examples",
)
def _chk_loudness_examples_negative() -> Outcome:
    return _loudness_examples_outcome(
        (0, 1, 2),
        ("LoudnessValue", "MaxMomentaryLoudness", "MaxShortTermLoudness"),
    )


@register(
    _BWF,
    "EBU Tech 3285:2011 (2.4)",
    "Loudness int16 = 100 x value, ties away from zero: positive examples",
)
def _chk_loudness_examples_positive() -> Outcome:
    return _loudness_examples_outcome(
        (3, 4, 5),
        ("LoudnessValue", "LoudnessRange", "MaxTruePeakLevel"),
    )


@register(
    _BWF,
    "EBU Tech 3285:2011 (2.4)",
    "Unused loudness parameters: 7FFFh on disk, None through the reader",
)
def _chk_loudness_unset_sentinel() -> Outcome:
    payload, reread = _write_and_capture(_metadata(
        loudness_value=None, loudness_range=None, max_true_peak_level=None,
        max_momentary_loudness=None, max_short_term_loudness=None,
    ))
    fields = (
        "LoudnessValue", "LoudnessRange", "MaxTruePeakLevel",
        "MaxMomentaryLoudness", "MaxShortTermLoudness",
    )
    on_disk = [_uint16_at(payload, field) for field in fields]
    reread_none = (
        reread.loudness_value is None and reread.loudness_range is None
        and reread.max_true_peak_level is None
        and reread.max_momentary_loudness is None
        and reread.max_short_term_loudness is None
    )
    sentinel = ref.TECH3285_LOUDNESS_UNSET
    return Outcome(
        expected=f"{sentinel:04X}h x 5 on disk; None x 5 reread",
        computed=(
            ", ".join(f"{value:04X}h" for value in on_disk)
            + ("; None x 5" if reread_none else "; a value leaked through")
        ),
        delta="-",
        passed=all(value == sentinel for value in on_disk) and reread_none,
    )


@register(
    _BWF,
    "EBU Tech 3285:2011 (2.4)",
    "Out-of-range loudness clamps to 7FFEh/8000h, never the 7FFFh sentinel",
)
def _chk_loudness_clamp() -> Outcome:
    # 327.9 dBTP would encode above int16; digital silence measures
    # -inf LUFS. Tech 3285 reserves 7FFFh for "not being used", so a real
    # value must never land there: the serialiser's documented rule pins
    # the overflow one code below the sentinel and the underflow to 8000h.
    payload, reread = _write_and_capture(_metadata(
        max_true_peak_level=327.9,
        max_momentary_loudness=float("-inf"),
    ))
    over = _uint16_at(payload, "MaxTruePeakLevel")
    under = _uint16_at(payload, "MaxMomentaryLoudness")
    sentinel = ref.TECH3285_LOUDNESS_UNSET
    passed = (
        over == sentinel - 1
        and under == 0x8000
        and reread.max_true_peak_level == (sentinel - 1) / 100
        and reread.max_momentary_loudness == -327.68
    )
    return Outcome(
        expected=f"327.9 -> {sentinel - 1:04X}h (not the sentinel); -inf -> 8000h",
        computed=f"{over:04X}h; {under:04X}h",
        delta="-",
        passed=passed,
    )


@register(
    _BWF,
    "EBU Tech 3285:2011 (2.3)",
    "TimeReference: 64-bit first-sample count split low/high at 338/342",
)
def _chk_time_reference_64_bit() -> Outcome:
    payload, reread = _canonical()
    low = int.from_bytes(_field_bytes(payload, "TimeReferenceLow"), "little")
    high = int.from_bytes(_field_bytes(payload, "TimeReferenceHigh"), "little")
    rebuilt = high << 32 | low
    passed = (
        rebuilt == _TIME_REFERENCE
        and _TIME_REFERENCE > 0xFFFFFFFF  # the count genuinely needs 64 bits
        and reread.time_reference == _TIME_REFERENCE
    )
    return Outcome(
        expected=(
            f"low {_TIME_REFERENCE & 0xFFFFFFFF} @ 338, "
            f"high {_TIME_REFERENCE >> 32} @ 342 "
            f"(13:30:00 at 96 kHz = {_TIME_REFERENCE} samples)"
        ),
        computed=f"low {low}, high {high} -> {rebuilt} samples, reread equal",
        delta="-",
        passed=passed,
    )


@functools.cache
def _r98_history() -> tuple[bytes, BroadcastMetadata]:
    """A 48 kHz 16-bit stereo write: R 98's own Example 1 configuration."""
    return _write_and_capture(_metadata(), fs=48000, channels=2)


@register(
    _BWF,
    "EBU Tech 3285:2011 (2.3): CodingHistory row per EBU R 98 Appendix 1",
    "Appended row is A=PCM,F=48000,W=16,M=stereo,T=... + CR/LF (Example 1)",
)
def _chk_coding_history_grammar() -> Outcome:
    payload, reread = _r98_history()
    rows = reread.coding_history.split("\r\n")
    appended = rows[-1]
    raw_history = payload[ref.TECH3285_BEXT_FIXED_SIZE:].split(b"\x00", 1)[0]
    prefix = ref.EBU_R98_EXAMPLE1_PCM_PREFIX
    free_text = appended[len(prefix):] if appended.startswith(prefix) else ""
    passed = (
        appended.startswith(prefix)
        and free_text != "" and "," not in free_text  # T= free string, no commas
        and raw_history.endswith(b"\r\n")  # every row CR/LF-terminated
    )
    shown = appended.split(",T=")[0] + ",T=(free text)" if appended else "(empty)"
    return Outcome(
        expected=prefix + "(free text, no commas) + CR/LF",
        computed=f"{shown} + CR/LF" if passed else f"{shown!r}",
        delta="-",
        passed=passed,
    )


@register(
    _BWF,
    "EBU Tech 3285:2011 (2.3): CodingHistory row per EBU R 98 Appendix 1",
    "Prior coding row preserved verbatim, new row added beneath it",
)
def _chk_coding_history_appends() -> Outcome:
    _, reread = _r98_history()
    rows = reread.coding_history.split("\r\n")
    passed = (
        len(rows) == 2
        and rows[0] == ref.EBU_R98_EXAMPLE2_PRIOR_LINE
        and rows[1].startswith("A=PCM,")
    )
    return Outcome(
        expected="2 rows: Example 2's A/D row intact above, the writer's beneath",
        computed=(
            f"{len(rows)} rows, prior row "
            + ("byte-identical" if rows and rows[0] == ref.EBU_R98_EXAMPLE2_PRIOR_LINE
               else "altered")
        ),
        delta="-",
        passed=passed,
    )


@functools.cache
def _v1_capture() -> tuple[bytes, BroadcastMetadata]:
    """A version 1 chunk: UMID present, no loudness (those bytes reserved)."""
    return _write_and_capture(_metadata(
        version=1, loudness_value=None, loudness_range=None,
        max_true_peak_level=None, max_momentary_loudness=None,
        max_short_term_loudness=None,
    ))


def _refused(**overrides: object) -> bool:
    """Whether the writer refuses to serialise the metadata."""
    try:
        _write_and_capture(_metadata(**overrides))
    except ValueError:
        return True
    return False


@register(
    _BWF,
    "EBU Tech 3285:2011 (1.1/2.3)",
    "UMID exists from v1 (64 of the 254 reserved bytes): v0 refused, v1 at 348",
)
def _chk_umid_version_gate() -> Outcome:
    refused_v0 = _refused(
        version=0, loudness_value=None, loudness_range=None,
        max_true_peak_level=None, max_momentary_loudness=None,
        max_short_term_loudness=None,
    )
    payload, reread = _v1_capture()
    version_word = _uint16_at(payload, "Version")
    umid_ok = _field_bytes(payload, "UMID") == _UMID and reread.umid == _UMID
    passed = refused_v0 and version_word == ref.TECH3285_VERSION_1 and umid_ok
    return Outcome(
        expected="v0+UMID refused; v1: Version=0001h, UMID verbatim at 348",
        computed=(
            f"v0 {'refused' if refused_v0 else 'ACCEPTED'}; "
            f"v1: Version={version_word:04X}h, UMID "
            + ("verbatim" if umid_ok else "altered")
        ),
        delta="-",
        passed=passed,
    )


@register(
    _BWF,
    "EBU Tech 3285:2011 (1.1/2.3)",
    "Loudness exists from v2 (10 of the 190 reserved bytes): v1 refused/zeroed",
)
def _chk_loudness_version_gate() -> Outcome:
    refused_v1 = _refused(version=1)
    payload, reread = _v1_capture()
    offsets = {name: offset for name, offset, _size in ref.TECH3285_BEXT_FIELDS}
    start = offsets["LoudnessValue"]
    loudness_area = payload[start:start + ref.TECH3285_LOUDNESS_BYTES]
    zeroed = loudness_area == bytes(ref.TECH3285_LOUDNESS_BYTES)
    reread_none = reread.loudness_value is None and reread.loudness_range is None
    v2_payload, _ = _canonical()
    v2_carries = _int16_at(v2_payload, "LoudnessValue") == _LOUDNESS_ON_GRID[
        "LoudnessValue"
    ][1]
    passed = refused_v1 and zeroed and reread_none and v2_carries
    return Outcome(
        expected="v1+loudness refused; v1 writes 10 zero bytes, reads None; v2 carries",
        computed=(
            f"v1 {'refused' if refused_v1 else 'ACCEPTED'}; bytes "
            + ("zeroed" if zeroed else "nonzero")
            + (", None reread" if reread_none else ", value leaked")
            + (", v2 carries" if v2_carries else ", v2 empty")
        ),
        delta="-",
        passed=passed,
    )


# ---------------------------------------------------------------------------
# ITU-R BS.2088-2: the 64-bit container.
# ---------------------------------------------------------------------------

#: 24 h of stereo 24-bit at 48 kHz: 24.9 GB of samples, well past the 4-GiB
#: RIFF limit, so the header must promote to 64-bit addressing.
_LONG_FRAMES = 48000 * 86400
_LONG_DATA_BYTES = _LONG_FRAMES * 6


@functools.cache
def _promoted_header() -> bytes:
    return build_wav_header(
        fs=48000, channels=2, subtype="PCM_24", frames=_LONG_FRAMES
    )


def _ds64_u64(payload: bytes, low_field: str) -> int:
    """Join a low/high DWORD pair of the ds64 payload into its uint64."""
    offsets = {name: offset for name, offset, _size in ref.BS2088_DS64_FIELDS}
    low = int.from_bytes(payload[offsets[low_field]:offsets[low_field] + 4], "little")
    high_field = low_field.replace("Low", "High")
    high = int.from_bytes(
        payload[offsets[high_field]:offsets[high_field] + 4], "little"
    )
    return high << 32 | low


@register(
    _BWF,
    "ITU-R BS.2088-2 Annex 1 (4.1/4.2)",
    "ds64 first after WAVE: bw64Size/dataSize u64 pairs at 0/8, table at 24",
)
def _chk_ds64_geometry() -> Outcome:
    header = _promoted_header()
    first_id = header[12:16]
    (ck_size,) = struct.unpack_from("<I", header, 16)
    payload = header[20:20 + ck_size]
    riff_size = _ds64_u64(payload, "bw64SizeLow")
    data_size = _ds64_u64(payload, "dataSizeLow")
    offsets = {name: offset for name, offset, _size in ref.BS2088_DS64_FIELDS}
    (table_length,) = struct.unpack_from("<I", payload, offsets["tableLength"])
    # The outer size counts everything after its own 8-byte header: the
    # written header minus those 8 bytes, plus the sample payload.
    expected_riff = len(header) - 8 + _LONG_DATA_BYTES
    passed = (
        first_id == b"ds64"
        and ck_size >= ref.BS2088_DS64_MIN_SIZE
        and riff_size == expected_riff
        and data_size == _LONG_DATA_BYTES
        and table_length == 0
    )
    return Outcome(
        expected=(
            f"ds64 first, >= {ref.BS2088_DS64_MIN_SIZE} B: riffSize "
            f"{expected_riff}, dataSize {_LONG_DATA_BYTES}, tableLength 0"
        ),
        computed=(
            f"{first_id.decode('latin-1')} @ 12, {ck_size} B: riffSize "
            f"{riff_size}, dataSize {data_size}, tableLength {table_length}"
        ),
        delta="-",
        passed=passed,
    )


@register(
    _BWF,
    "ITU-R BS.2088-2 Annex 1 (3.2/2.4)",
    "Promoted header: FFFFFFFFh sentinel in the outer and data size fields",
)
def _chk_size_sentinels() -> Outcome:
    header = _promoted_header()
    (outer,) = struct.unpack_from("<I", header, 4)
    (data_field,) = struct.unpack_from("<I", header, len(header) - 4)
    form_type = header[8:12]
    sentinel = ref.BS2088_SIZE_SENTINEL
    passed = (
        outer == sentinel and data_field == sentinel and form_type == b"WAVE"
        and header[len(header) - 8:len(header) - 4] == b"data"
    )
    return Outcome(
        expected=f"outer size = data size = {sentinel:08X}h, form type WAVE",
        computed=f"outer {outer:08X}h, data {data_field:08X}h, {form_type.decode('latin-1')}",
        delta="-",
        passed=passed,
    )


def _sixty_four_bit_file(fourcc: bytes) -> bytes:
    """A minimal 64-bit container assembled from the transcribed layout.

    48 frames of 16-bit mono PCM at 48 kHz whose 32-bit data size field
    holds the 0xFFFFFFFF sentinel, so the real size is only knowable
    through ds64 - the BS.2088-2 2.4 deferral rule in its smallest
    possible file, under either the BW64 fourcc (3.2) or Tech 3306's RF64
    (which BS.2088 Annex 1 builds on).
    """
    samples = bytes(96)  # 48 frames x 2 bytes
    fmt = struct.pack("<HHIIHH", 1, 1, 48000, 96000, 2, 16)
    sentinel = ref.BS2088_SIZE_SENTINEL
    # The outer 64-bit size counts the WAVE form type plus every chunk:
    # ds64 (8 + 28), fmt (8 + 16) and data (8 + 96).
    riff_size = 4 + 8 + ref.BS2088_DS64_MIN_SIZE + 8 + len(fmt) + 8 + len(samples)
    ds64 = struct.pack("<QQQI", riff_size, len(samples), 0, 0)
    body = (
        b"ds64" + struct.pack("<I", len(ds64)) + ds64
        + b"fmt " + struct.pack("<I", len(fmt)) + fmt
        + b"data" + struct.pack("<I", sentinel) + samples
    )
    return fourcc + struct.pack("<I", sentinel) + b"WAVE" + body


@register(
    _BWF,
    "ITU-R BS.2088-2 Annex 1 (2.4/3.1)",
    "Reader resolves the data size through ds64; BW64 and RF64 fourccs alike",
)
def _chk_reader_ds64_resolution() -> Outcome:
    results: dict[str, tuple[str, int]] = {}
    with tempfile.TemporaryDirectory() as tmp:
        for fourcc in (b"RF64", ref.BS2088_BW64_FOURCC):
            path = Path(tmp) / f"{fourcc.decode('latin-1').lower()}.wav"
            path.write_bytes(_sixty_four_bit_file(fourcc))
            described = info(path)
            results[fourcc.decode("latin-1")] = (
                described.container, described.frames
            )
    passed = results == {"RF64": ("RF64", 48), "BW64": ("BW64", 48)}
    return Outcome(
        expected="RF64: 48 frames, BW64: 48 frames (via ds64 dataSize = 96 B)",
        computed="; ".join(
            f"{name} read as {container}, {frames} frames"
            for name, (container, frames) in results.items()
        ),
        delta="-",
        passed=passed,
    )
