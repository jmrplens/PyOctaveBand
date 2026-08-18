#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Tests for the RIFF/RF64 chunk walker against hand-assembled files.

The ``bext`` tests are transcription oracles: the chunk is assembled byte by
byte at the offsets of the EBU Tech 3285 (2011) field table, written down
here independently of the reader's own layout, so a transcription slip on
either side surfaces as a field-level mismatch instead of cancelling out.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest
from wav_forge import (
    chunk,
    extensible_fmt_payload,
    fmt_payload,
    pcm_data,
    pcm_wav,
    rf64_wav,
    riff_wave,
)

from phonometry.io import _chunks
from phonometry.io._chunks import WAVE_FORMAT_IMA_ADPCM, parse_wav_chunks

TONE = np.array([0, 8000, -8000, 32000], dtype=np.int64)


def _write(tmp_path: Path, image: bytes) -> Path:
    path = tmp_path / "forged.wav"
    path.write_bytes(image)
    return path


# ---------------------------------------------------------------------------
# bext v2: byte-by-byte transcription oracle from the Tech 3285 field table
# ---------------------------------------------------------------------------

#: (offset, size) of every fixed bext field, copied from EBU Tech 3285
#: (2011) independently of src/phonometry/io/_chunks.py. CodingHistory
#: follows the 602-byte fixed part.
BEXT_OFFSETS = {
    "Description": (0, 256),
    "Originator": (256, 32),
    "OriginatorReference": (288, 32),
    "OriginationDate": (320, 10),
    "OriginationTime": (330, 8),
    "TimeReferenceLow": (338, 4),
    "TimeReferenceHigh": (342, 4),
    "Version": (346, 2),
    "UMID": (348, 64),
    "LoudnessValue": (412, 2),
    "LoudnessRange": (414, 2),
    "MaxTruePeakLevel": (416, 2),
    "MaxMomentaryLoudness": (418, 2),
    "MaxShortTermLoudness": (420, 2),
    "Reserved": (422, 180),
}
BEXT_FIXED_SIZE = 602


def _place(buf: bytearray, field: str, raw: bytes) -> None:
    """Write ``raw`` into ``buf`` at the field's Tech 3285 offset."""
    offset, size = BEXT_OFFSETS[field]
    assert len(raw) <= size, f"{field}: {len(raw)} bytes exceed the field's {size}"
    buf[offset:offset + len(raw)] = raw


def _bext_payload(
    *,
    version: int = 2,
    time_low: int = 0,
    time_high: int = 0,
    umid: bytes = b"",
    loudness: dict[str, int] | None = None,
    coding_history: bytes = b"",
) -> bytearray:
    """Assemble a bext payload at the offsets of the table above."""
    buf = bytearray(BEXT_FIXED_SIZE)
    _place(buf, "Description", b"Facade measurement, position 1")
    _place(buf, "Originator", b"phonometry test rig")
    _place(buf, "OriginatorReference", b"ESJMR0123456789012345678901234")
    _place(buf, "OriginationDate", b"2026-08-18")
    _place(buf, "OriginationTime", b"03-15-00")
    _place(buf, "TimeReferenceLow", struct.pack("<I", time_low))
    _place(buf, "TimeReferenceHigh", struct.pack("<I", time_high))
    _place(buf, "Version", struct.pack("<H", version))
    _place(buf, "UMID", umid)
    for field, raw_value in (loudness or {}).items():
        _place(buf, field, struct.pack("<h", raw_value))
    return buf + coding_history


def _read_bext(tmp_path: Path, payload: bytes):
    image = pcm_wav(TONE, extra_chunks=chunk(b"bext", bytes(payload)))
    result = parse_wav_chunks(_write(tmp_path, image)).bext
    assert result is not None
    return result


def test_bext_fields_read_at_tech3285_offsets(tmp_path: Path) -> None:
    """Every fixed field decodes from exactly its documented offset."""
    history = b"A=PCM,F=48000,W=24,M=stereo,T=field recorder\r\n"
    bext = _read_bext(tmp_path, _bext_payload(
        time_low=0x0002D2F0,  # 48000 Hz * 3.86 s, low half
        time_high=0x00000001,
        umid=bytes(range(64)),
        loudness={
            "LoudnessValue": -2300,        # -23.00 LUFS
            "LoudnessRange": 550,          # 5.50 LU
            "MaxTruePeakLevel": -102,      # -1.02 dBTP
            "MaxMomentaryLoudness": -1875, # -18.75 LUFS
            "MaxShortTermLoudness": -2010, # -20.10 LUFS
        },
        coding_history=history,
    ))
    assert bext.description == "Facade measurement, position 1"
    assert bext.originator == "phonometry test rig"
    assert bext.originator_reference == "ESJMR0123456789012345678901234"
    assert bext.origination_date == "2026-08-18"
    assert bext.origination_time == "03-15-00"
    assert bext.time_reference == (1 << 32) + 0x0002D2F0
    assert bext.version == 2
    assert bext.umid == bytes(range(64))
    assert bext.loudness_value == pytest.approx(-23.00)
    assert bext.loudness_range == pytest.approx(5.50)
    assert bext.max_true_peak_level == pytest.approx(-1.02)
    assert bext.max_momentary_loudness == pytest.approx(-18.75)
    assert bext.max_short_term_loudness == pytest.approx(-20.10)
    assert bext.coding_history == history.decode().rstrip()


def test_bext_version_gates_umid_and_loudness(tmp_path: Path) -> None:
    """v0 reports neither UMID nor loudness; v1 only the UMID.

    The bytes are present in every version (the fixed part is always 602
    bytes); what the version changes is whether they mean anything, and the
    reader must not report reserved padding as data.
    """
    v0 = _read_bext(tmp_path, _bext_payload(version=0, umid=b"\x42" * 64))
    assert v0.umid is None
    assert v0.loudness_value is None
    v1 = _read_bext(tmp_path, _bext_payload(version=1, umid=b"\x42" * 64,
                                            loudness={"LoudnessValue": -2300}))
    assert v1.umid == b"\x42" * 64
    assert v1.loudness_value is None
    assert v1.loudness_range is None


def test_bext_unset_loudness_sentinel_reads_as_none(tmp_path: Path) -> None:
    """Tech 3285 fills unset v2 loudness fields with 0x7FFF, not a value."""
    bext = _read_bext(tmp_path, _bext_payload(
        loudness={field: 0x7FFF for field in (
            "LoudnessValue", "LoudnessRange", "MaxTruePeakLevel",
            "MaxMomentaryLoudness", "MaxShortTermLoudness",
        )},
    ))
    assert bext.loudness_value is None
    assert bext.loudness_range is None
    assert bext.max_true_peak_level is None
    assert bext.max_momentary_loudness is None
    assert bext.max_short_term_loudness is None


def test_bext_shorter_than_fixed_part_is_rejected(tmp_path: Path) -> None:
    image = pcm_wav(TONE, extra_chunks=chunk(b"bext", bytes(100)))
    path = _write(tmp_path, image)
    with pytest.raises(ValueError, match="602"):
        parse_wav_chunks(path)


# ---------------------------------------------------------------------------
# fmt: EXTENSIBLE fields scipy drops
# ---------------------------------------------------------------------------

def test_extensible_fmt_keeps_mask_valid_bits_and_guid(tmp_path: Path) -> None:
    payload = extensible_fmt_payload(
        channels=2, bits=32, valid_bits=24, channel_mask=0x3,
    )
    image = riff_wave(
        chunk(b"fmt ", payload),
        chunk(b"data", pcm_data(np.array([1, -1, 2, -2]), 32)),
    )
    fmt = parse_wav_chunks(_write(tmp_path, image)).fmt
    assert fmt.format_tag == 0xFFFE
    assert fmt.resolved_tag == 0x0001
    assert fmt.format_name == "PCM"
    assert fmt.bits_per_sample == 32
    assert fmt.valid_bits == 24
    assert fmt.channel_mask == 0x3
    assert fmt.channel_labels() == ("FL", "FR")
    assert not fmt.ambisonic
    assert not fmt.lossy


def test_channel_labels_require_matching_population(tmp_path: Path) -> None:
    """A zero or mismatched mask must yield no labels, not wrong ones."""
    for mask, channels in ((0x0, 2), (0x7, 2)):
        payload = extensible_fmt_payload(channels=channels, channel_mask=mask)
        image = riff_wave(
            chunk(b"fmt ", payload),
            chunk(b"data", pcm_data(np.zeros(channels, dtype=np.int64), 16)),
        )
        fmt = parse_wav_chunks(_write(tmp_path, image)).fmt
        assert fmt.channel_labels() is None, hex(mask)


def test_ambisonic_guid_resolves_to_pcm_and_is_flagged(tmp_path: Path) -> None:
    """A FuMa ``.amb`` GUID decodes as PCM but keeps its ambisonic identity."""
    ambisonic_tail = b"\x21\x07\xd3\x11\x86\x44\xc8\xc1\xca\x00\x00\x00"
    payload = extensible_fmt_payload(
        channels=4, bits=16, channel_mask=0, sub_tag=0x0001,
        guid_tail=b"\x00\x00" + ambisonic_tail,
    )
    image = riff_wave(
        chunk(b"fmt ", payload),
        chunk(b"data", pcm_data(np.zeros(4, dtype=np.int64), 16)),
    )
    fmt = parse_wav_chunks(_write(tmp_path, image)).fmt
    assert fmt.ambisonic
    assert fmt.resolved_tag == 0x0001
    assert fmt.format_name == "ambisonic B-format PCM"


def test_unknown_extensible_guid_keeps_the_raw_tag(tmp_path: Path) -> None:
    """An unrecognised SubFormat must not be silently read as its first bytes."""
    payload = extensible_fmt_payload(
        channels=1, sub_tag=0x0001, guid_tail=b"\xde\xad" * 7,
    )
    image = riff_wave(
        chunk(b"fmt ", payload),
        chunk(b"data", pcm_data(np.zeros(1, dtype=np.int64), 16)),
    )
    fmt = parse_wav_chunks(_write(tmp_path, image)).fmt
    assert fmt.resolved_tag == 0xFFFE
    assert not fmt.ambisonic


def test_unknown_codec_tags_fail_closed_as_lossy(tmp_path: Path) -> None:
    """A wFormatTag outside the transcribed set must not pass as clean.

    The RIFF registry holds hundreds of codecs and nearly all of them are
    lossy; tag 0x0050 (WAVE_FORMAT_MPEG, MPEG-1 audio layer I/II) is a
    real one a transcoder can leave inside a WAV. If membership in the
    transcribed lossy set were the only path to ``lossy=True``, every
    untranscribed codec would be reported clean -- the wrong default for
    the field the docs present as the honest carrier of lossiness.
    """
    image = riff_wave(
        chunk(b"fmt ", fmt_payload(tag=0x0050)),
        chunk(b"fact", struct.pack("<I", 4)),
        chunk(b"data", bytes(8)),
    )
    fmt = parse_wav_chunks(_write(tmp_path, image)).fmt
    assert fmt.lossy
    assert fmt.format_name == "tag 0x0050"


# ---------------------------------------------------------------------------
# RF64 / ds64
# ---------------------------------------------------------------------------

def test_rf64_data_size_comes_from_ds64(tmp_path: Path) -> None:
    """The 0xFFFFFFFF sentinel defers to the 64-bit ds64 sizes."""
    parsed = parse_wav_chunks(_write(tmp_path, rf64_wav(TONE)))
    assert parsed.container == "RF64"
    assert parsed.ds64 is not None
    assert parsed.data_size == TONE.size * 2
    assert parsed.ds64.data_size == TONE.size * 2
    assert parsed.ds64.sample_count == TONE.size
    assert parsed.frames == TONE.size


def test_rf64_sentinel_without_ds64_is_rejected(tmp_path: Path) -> None:
    image = riff_wave(
        chunk(b"fmt ", fmt_payload()),
        chunk(b"data", pcm_data(TONE, 16), declared_size=0xFFFFFFFF),
        fourcc=b"RF64",
    )
    path = _write(tmp_path, image)
    with pytest.raises(ValueError, match="ds64"):
        parse_wav_chunks(path)


def test_bw64_fourcc_is_accepted(tmp_path: Path) -> None:
    image = rf64_wav(TONE).replace(b"RF64", b"BW64", 1)
    assert parse_wav_chunks(_write(tmp_path, image)).container == "BW64"


# ---------------------------------------------------------------------------
# cue, fact, iXML, alignment, errors
# ---------------------------------------------------------------------------

def test_cue_points_decode_all_record_fields(tmp_path: Path) -> None:
    records = struct.pack("<I", 2) + struct.pack(
        "<II4sIII", 1, 0, b"data", 0, 0, 4800
    ) + struct.pack("<II4sIII", 2, 1, b"data", 0, 0, 96000)
    image = pcm_wav(TONE, extra_chunks=chunk(b"cue ", records))
    points = parse_wav_chunks(_write(tmp_path, image)).cue_points
    assert len(points) == 2
    assert points[0].cue_id == 1
    assert points[0].chunk_id == b"data"
    assert points[0].sample_offset == 4800
    assert points[1].sample_offset == 96000


def test_ixml_presence_is_reported(tmp_path: Path) -> None:
    with_ixml = pcm_wav(
        TONE, extra_chunks=chunk(b"iXML", b"<BWFXML></BWFXML>")
    )
    assert parse_wav_chunks(_write(tmp_path, with_ixml)).has_ixml
    assert not parse_wav_chunks(_write(tmp_path, pcm_wav(TONE))).has_ixml


def test_fact_frames_govern_compressed_frame_count(tmp_path: Path) -> None:
    """For ADPCM the fact chunk counts frames; block division cannot."""
    image = riff_wave(
        chunk(b"fmt ", fmt_payload(tag=WAVE_FORMAT_IMA_ADPCM, bits=4,
                                   block_align=256)),
        chunk(b"fact", struct.pack("<I", 5000)),
        chunk(b"data", bytes(512)),
    )
    parsed = parse_wav_chunks(_write(tmp_path, image))
    assert parsed.fmt.lossy
    assert parsed.fact_frames == 5000
    assert parsed.frames == 5000


def test_odd_sized_chunk_keeps_the_walker_aligned(tmp_path: Path) -> None:
    """A pad byte after an odd payload is skipped, not read as a header."""
    image = pcm_wav(TONE, extra_chunks=chunk(b"iXML", b"<odd/>" + b"x"))
    parsed = parse_wav_chunks(_write(tmp_path, image))
    assert parsed.has_ixml
    assert parsed.frames == TONE.size


def test_skipped_chunks_are_seeked_past_never_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unknown chunks and iXML cost a seek, not their size in memory.

    The walker's promise is that header inspection costs kilobytes
    whatever the file holds; a recorder's multi-megabyte JUNK padding or
    iXML production dump must therefore be stepped over, with only the
    decoded kinds ever read whole.
    """
    read_ids: list[bytes] = []
    original = _chunks._read_chunk_payload

    def spy(fh: object, chunk_id: bytes, size: int, path: object) -> bytes:
        read_ids.append(chunk_id)
        return original(fh, chunk_id, size, path)  # type: ignore[arg-type]

    monkeypatch.setattr(_chunks, "_read_chunk_payload", spy)
    image = pcm_wav(
        TONE,
        extra_chunks=chunk(b"JUNK", bytes(1 << 16)) + chunk(b"iXML", b"<x/>"),
    )
    parsed = parse_wav_chunks(_write(tmp_path, image))
    assert parsed.has_ixml
    assert parsed.frames == TONE.size
    assert b"JUNK" not in read_ids
    assert b"iXML" not in read_ids
    assert b"fmt " in read_ids


def test_data_before_metadata_chunks_is_still_walked(tmp_path: Path) -> None:
    """Chunks after ``data`` (where some recorders put bext) are reached."""
    image = riff_wave(
        chunk(b"fmt ", fmt_payload()),
        chunk(b"data", pcm_data(TONE, 16)),
        chunk(b"bext", bytes(_bext_payload())),
    )
    parsed = parse_wav_chunks(_write(tmp_path, image))
    assert parsed.bext is not None
    assert parsed.frames == TONE.size


@pytest.mark.parametrize(
    ("image", "match"),
    [
        (b"RIFF\x04\x00\x00\x00JUNK", "not a WAVE"),
        (b"XXXX\x08\x00\x00\x00WAVEabcd", "unknown container"),
        (riff_wave(chunk(b"fmt ", fmt_payload())), "no data chunk"),
        (riff_wave(chunk(b"data", b"\x00\x00")), "no fmt chunk"),
        (riff_wave(chunk(b"fmt ", fmt_payload()[:8])), "at least 16"),
        (
            riff_wave(chunk(b"bext", bytes(4), declared_size=4000)),
            "remain in the file",
        ),
        (
            riff_wave(
                chunk(b"fmt ", fmt_payload()),
                chunk(b"data", b"\x00\x00"),
                chunk(b"JUNK", bytes(4), declared_size=4000),
            ),
            "remain in the file",
        ),
        (
            riff_wave(chunk(b"fmt ", fmt_payload()[:8], declared_size=16)),
            "remain in the file",
        ),
        (
            riff_wave(
                chunk(b"fmt ", fmt_payload()),
                chunk(b"cue ", struct.pack("<I", 3) + bytes(10)),
                chunk(b"data", b"\x00\x00"),
            ),
            "cue chunk declares 3 points",
        ),
        (
            riff_wave(
                chunk(b"ds64", bytes(10)),
                chunk(b"fmt ", fmt_payload()),
                chunk(b"data", b"\x00\x00"),
                fourcc=b"RF64",
            ),
            "ds64 chunk is 10 bytes",
        ),
        (
            riff_wave(
                chunk(b"fmt ", fmt_payload()),
                chunk(b"fact", b"\x00\x00"),
                chunk(b"data", b"\x00\x00"),
            ),
            "fact chunk is 2 bytes",
        ),
    ],
    ids=["junk-form", "junk-fourcc", "no-data", "no-fmt", "short-fmt",
         "truncated-chunk", "overdeclared-skipped-chunk", "truncated-fmt",
         "short-cue", "short-ds64", "short-fact"],
)
def test_malformed_files_fail_loudly(
    tmp_path: Path, image: bytes, match: str
) -> None:
    path = _write(tmp_path, image)
    with pytest.raises(ValueError, match=match):
        parse_wav_chunks(path)
