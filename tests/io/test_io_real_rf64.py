#  Copyright (c) 2026. Jose Manuel Requena Plens
"""A genuine RF64 from an independent producer (EBU libbw64, Apache-2.0).

The RF64 files elsewhere in this suite are forged by ``tests/wav_forge.py``,
so reader and forger share one understanding of the format. This file was
written by the EBU's own BW64/RF64 library (github.com/ebu/libbw64,
``tests/test_data/rect_24bit_rf64.wav``, byte-identical; see
``tests/data/audio/README.md``): a second, independent producer's idea of
``RF64`` + ``ds64``, which is what the reader must agree with in the field.

The expected numbers are read off the file's own header bytes (the ``ds64``
payload declares riffSize 132372 and dataSize 132300 = 22050 frames x 2
channels x 3 bytes) and off its content, a full-scale-adjacent rectangular
wave whose 24-bit codes are +/-6340995 exactly.
"""

from __future__ import annotations

import struct

import numpy as np
import oracle_data

from phonometry import io

_PATH = oracle_data.DATA / "audio" / "libbw64" / "rect_24bit_rf64.wav"


def test_file_is_a_genuine_rf64_with_ds64() -> None:
    """The fixture itself: RF64 FourCC and a ds64 declaring the sizes."""
    raw = _PATH.read_bytes()
    assert raw[:4] == b"RF64"
    assert raw[8:12] == b"WAVE"
    i = raw.find(b"ds64")
    assert i > 0
    riff_size, data_size = struct.unpack("<QQ", raw[i + 8 : i + 24])
    assert riff_size == 132372
    assert data_size == 132300
    assert len(raw) == riff_size + 8


def test_info_reads_the_geometry_from_ds64() -> None:
    meta = io.info(_PATH)
    assert meta.container == "RF64"
    assert meta.format_name == "PCM"
    assert meta.bit_depth == 24
    assert meta.fs == 44100
    assert meta.channels == 2
    # 132300 data bytes / (2 channels x 3 bytes) = 22050 frames = 0.5 s.
    assert meta.frames == 22050
    assert meta.duration == 0.5
    assert not meta.lossy


def test_read_decodes_the_rectangle_bit_exactly() -> None:
    x = np.asarray(io.read(_PATH))
    assert x.shape == (2, 22050)
    # Integer PCM is scaled by exactly 2**23, so the decoded float values
    # must be exact multiples of it: the rectangle's two levels and the
    # zero crossings, nothing else.
    codes = np.unique(np.round(x * 2**23)).astype(np.int64)
    assert codes.tolist() == [-6340995, 0, 6340995]
    assert np.array_equal(np.round(x * 2**23) / 2**23, x)
