#  Copyright (c) 2026. Jose Manuel Requena Plens
"""In-test WAV forging helpers for the io tests.

Every byte written here is assembled by hand with :mod:`struct`, never by the
code under test and never by scipy, so a file the io tests read is an
independent transcription of the container specifications: RIFF/WAVE from the
Microsoft/IBM Multimedia Programming Interface and Data Specifications 1.0
(1991), WAVE_FORMAT_EXTENSIBLE from Microsoft's ``mmreg.h``/``ksmedia.h``
layout, RF64 from EBU Tech 3306, and ``bext`` from the EBU Tech 3285 (2011)
field table (the byte-by-byte offset assembly for the bext oracle lives in
the test module itself, next to the offsets it pins).

The forge writes what the tests ask for, including files no sane writer
produces (an RF64 holding two kilobytes behind a real ``ds64``), because the
point is exercising the reader's handling of the layout, not producing
archival audio.
"""

from __future__ import annotations

import struct

import numpy as np

#: Tail of the KSDATAFORMAT media GUIDs ({XXXX0000-0000-0010-8000-00AA00389B71}
#: with Data2/Data3 little-endian), transcribed from ksmedia.h independently
#: of the reader's copy.
KSDATAFORMAT_TAIL = b"\x00\x00\x00\x00\x10\x00\x80\x00\x00\xaa\x00\x38\x9b\x71"


def chunk(
    chunk_id: bytes, payload: bytes, *, declared_size: int | None = None
) -> bytes:
    """One RIFF chunk: header, payload, and the pad byte of an odd payload.

    ``declared_size`` forges a size field that differs from the payload (the
    RF64 0xFFFFFFFF sentinel); the pad byte still follows the real payload
    length, as the RIFF rule is about the bytes actually present.
    """
    size = len(payload) if declared_size is None else declared_size
    padding = b"\x00" if len(payload) % 2 else b""
    return chunk_id + struct.pack("<I", size) + payload + padding


def fmt_payload(
    *,
    tag: int = 0x0001,
    channels: int = 1,
    fs: int = 48000,
    bits: int = 16,
    block_align: int | None = None,
) -> bytes:
    """A 16-byte plain ``fmt`` payload."""
    block = channels * ((bits + 7) // 8) if block_align is None else block_align
    return struct.pack("<HHIIHH", tag, channels, fs, fs * block, block, bits)


def extensible_fmt_payload(
    *,
    channels: int,
    fs: int = 48000,
    bits: int = 16,
    valid_bits: int | None = None,
    channel_mask: int = 0,
    sub_tag: int = 0x0001,
    guid_tail: bytes = KSDATAFORMAT_TAIL,
) -> bytes:
    """A 40-byte WAVE_FORMAT_EXTENSIBLE ``fmt`` payload (cbSize = 22)."""
    block = channels * ((bits + 7) // 8)
    guid = struct.pack("<H", sub_tag) + guid_tail
    assert len(guid) == 16
    return struct.pack("<HHIIHH", 0xFFFE, channels, fs, fs * block, block, bits) + (
        struct.pack(
            "<HHI", 22, valid_bits if valid_bits is not None else bits, channel_mask
        )
        + guid
    )


def riff_wave(*chunks: bytes, fourcc: bytes = b"RIFF") -> bytes:
    """Wrap chunks into a RIFF/RF64/BW64 WAVE file image."""
    body = b"WAVE" + b"".join(chunks)
    size = 0xFFFFFFFF if fourcc in (b"RF64", b"BW64") else len(body)
    return fourcc + struct.pack("<I", size) + body


def pcm_data(samples: np.ndarray, bits: int) -> bytes:
    """Encode integer samples (frames, channels) or (n,) at a PCM depth.

    8-bit is unsigned per the WAV convention; 24-bit packs the low three
    bytes of each value little-endian; 16/32-bit write the matching numpy
    little-endian dtype directly.
    """
    flat = np.asarray(samples).reshape(-1)
    if bits == 8:
        return flat.astype("<u1").tobytes()
    if bits == 16:
        return flat.astype("<i2").tobytes()
    if bits == 24:
        return b"".join(int(v).to_bytes(4, "little", signed=True)[:3] for v in flat)
    if bits == 32:
        return flat.astype("<i4").tobytes()
    raise ValueError(f"unsupported PCM depth {bits}")


def pcm_wav(
    samples: np.ndarray,
    *,
    bits: int = 16,
    channels: int = 1,
    fs: int = 48000,
    extra_chunks: bytes = b"",
) -> bytes:
    """A complete PCM WAV image with optional extra chunks before ``data``."""
    return riff_wave(
        chunk(b"fmt ", fmt_payload(channels=channels, fs=fs, bits=bits)),
        extra_chunks,
        chunk(b"data", pcm_data(samples, bits)),
    )


def float_wav(
    samples: np.ndarray,
    *,
    bits: int = 32,
    channels: int = 1,
    fs: int = 48000,
) -> bytes:
    """A complete IEEE-float (tag 0x0003) WAV image."""
    dtype = "<f4" if bits == 32 else "<f8"
    return riff_wave(
        chunk(b"fmt ", fmt_payload(tag=0x0003, channels=channels, fs=fs, bits=bits)),
        chunk(b"data", np.asarray(samples).reshape(-1).astype(dtype).tobytes()),
    )


def rf64_wav(
    samples: np.ndarray,
    *,
    bits: int = 16,
    channels: int = 1,
    fs: int = 48000,
    sample_count: int | None = None,
    fourcc: bytes = b"RF64",
) -> bytes:
    """A small forged RF64: real ``ds64`` sizes, sentinel 32-bit fields.

    The layout is exactly what a field recorder writes past 4 GiB (fourcc
    ``RF64``, ``ds64`` first, ``data`` size 0xFFFFFFFF) with only the byte
    counts scaled down, so the reader's RF64 path is exercised without
    forging 4 GiB of payload. ``fourcc=b"BW64"`` forges the ITU-R BS.2088
    variant, which shares the ``ds64`` layout and differs in the tag alone.
    """
    payload = pcm_data(samples, bits)
    frames = len(payload) // (channels * ((bits + 7) // 8))
    count = frames if sample_count is None else sample_count
    fmt = chunk(b"fmt ", fmt_payload(channels=channels, fs=fs, bits=bits))
    data = chunk(b"data", payload, declared_size=0xFFFFFFFF)
    # riffSize = everything after the initial 8-byte RIFF header; assemble
    # once with a placeholder to measure it, then write the real value.
    placeholder = struct.pack("<QQQI", 0, len(payload), count, 0)
    body_size = len(riff_wave(chunk(b"ds64", placeholder), fmt, data)) - 8
    ds64 = struct.pack("<QQQI", body_size, len(payload), count, 0)
    return riff_wave(chunk(b"ds64", ds64), fmt, data, fourcc=fourcc)
