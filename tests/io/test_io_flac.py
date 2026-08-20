#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Tests for FLAC writing and the bext carried in its APPLICATION block.

FLAC being lossless compression of integer PCM, every sample comparison
here is exact equality: the writer quantises in-house and hands libsndfile
finished codes, so a value that is off by one code means the pipeline
scaled where it promised not to. The bext round trip goes through the
phase-tested chunk parser, reading the APPLICATION block laid out per
RFC 9639 and the flac --keep-foreign-metadata convention.
"""

from __future__ import annotations

import builtins
import struct
from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np
import pytest

from phonometry import __version__
from phonometry.io import ClippingWarning, info, read, write
from phonometry.io._bext import fresh_metadata

if TYPE_CHECKING:
    from pathlib import Path

try:
    import soundfile as sf

    HAVE_SOUNDFILE = True
except ImportError:  # pragma: no cover - depends on the environment
    sf = None  # type: ignore[assignment]
    HAVE_SOUNDFILE = False

needs_soundfile = pytest.mark.skipif(
    not HAVE_SOUNDFILE, reason="the [audio] extra (soundfile) is not installed"
)

FS = 48000


@needs_soundfile
def test_flac_default_is_pcm24_and_bit_exact(tmp_path: Path) -> None:
    codes = np.array([0, 1, -1, 2**23 - 1, -(2**23), 123456], dtype=np.int64)
    path = tmp_path / "archive.flac"
    write(path, codes / 2**23, FS)
    assert np.asarray(read(path)).tolist() == (codes / 2**23).tolist()
    described = info(path)
    assert described.container == "FLAC"
    assert described.bit_depth == 24
    assert not described.lossy


@needs_soundfile
def test_flac_pcm16_stereo_roundtrips_exactly(tmp_path: Path) -> None:
    left = np.arange(-16, 16, dtype=np.int64) * 1000
    right = left[::-1]
    path = tmp_path / "s16.flac"
    write(path, np.stack([left, right]) / 32768, FS, subtype="PCM_16")
    sig = read(path)
    assert np.asarray(sig)[0].tolist() == (left / 32768).tolist()
    assert np.asarray(sig)[1].tolist() == (right / 32768).tolist()


@needs_soundfile
def test_int16_codes_pass_through_flac_bit_exact(tmp_path: Path) -> None:
    codes = np.array([0, 1, -1, 32767, -32768], dtype=np.int16)
    path = tmp_path / "codes.flac"
    write(path, codes, FS)
    assert np.asarray(read(path)).tolist() == (codes / 32768).tolist()


@needs_soundfile
def test_flac_refuses_what_it_cannot_hold(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="24 bits"):
        write(tmp_path / "x.flac", np.zeros(4, dtype=np.int32), FS)
    for subtype in ("PCM_32", "FLOAT", "DOUBLE"):
        with pytest.raises(ValueError, match="24 bits"):
            write(tmp_path / "x.flac", np.zeros(4), FS, subtype=subtype)


@needs_soundfile
def test_flac_clipping_is_saturated_and_announced(tmp_path: Path) -> None:
    path = tmp_path / "hot.flac"
    with pytest.warns(ClippingWarning, match="1 of 2"):
        write(path, np.array([1.5, -0.5]), FS, subtype="PCM_16")
    assert np.asarray(read(path)).tolist() == [32767 / 32768, -0.5]


@needs_soundfile
def test_bext_rides_the_application_block_field_for_field(
    tmp_path: Path,
) -> None:
    meta = replace(
        fresh_metadata(),
        description="reverberation room, source position B",
        originator="SVAN 979",
        time_reference=(3 << 32) | 999,
        umid=bytes(range(64)),
        loudness_value=-23.0,
        coding_history="A=PCM,F=48000,W=24,M=mono,T=recorder\r\n",
    )
    path = tmp_path / "carried.flac"
    write(path, np.full(16, 0.125), FS, bext=meta)
    got = read(path).provenance
    assert got is not None
    expected_history = (
        meta.coding_history + f"A=FLAC,F={FS},W=24,M=mono,T=phonometry {__version__}"
    )
    assert got == replace(meta, coding_history=expected_history)
    # info() surfaces the same chunk without decoding a single frame.
    described = info(path).bext
    assert described is not None
    assert described.originator == "SVAN 979"


@needs_soundfile
def test_application_block_bytes_match_the_reference_convention(
    tmp_path: Path,
) -> None:
    """The written APPLICATION block, read back byte for byte.

    Every other bext test here writes and reads through this same module,
    so a regression in the block layout -- the application ID, the
    verbatim fourcc + little-endian size framing, the block header --
    would round-trip invisibly (flipping the app ID's case survives
    exactly that way), and with it the module's central claim: that the
    reference ``flac`` tool recognises the block. The constants below are
    a second transcription, from RFC 9639 section 8 and the
    ``flac --keep-foreign-metadata`` convention, independent of the
    module's own -- the same pattern the bext offset oracle uses.
    """
    path = tmp_path / "bytes.flac"
    write(path, np.full(16, 0.25), FS, bext=fresh_metadata())
    blob = path.read_bytes()
    assert blob[:4] == b"fLaC"

    application = None
    pos = 4
    while True:
        header = blob[pos : pos + 4]
        # One byte: last-block flag in bit 7, type in bits 6-0; then a
        # 24-bit big-endian length (RFC 9639 section 8.1).
        block_type = header[0] & 0x7F
        length = int.from_bytes(header[1:4], "big")
        last = bool(header[0] & 0x80)
        if block_type == 2:  # APPLICATION
            assert application is None, "one APPLICATION block, not several"
            assert last, "the inserted block must close the metadata chain"
            application = blob[pos + 4 : pos + 4 + length]
        pos += 4 + length
        if last:
            break

    assert application is not None
    # flac --keep-foreign-metadata layout: the registered application ID
    # 'riff' (lowercase on the wire), then the RIFF chunk verbatim --
    # fourcc, uint32 little-endian size, payload, and a pad byte exactly
    # when the payload is odd.
    assert application[:4] == b"riff"
    assert application[4:8] == b"bext"
    (size,) = struct.unpack_from("<I", application, 8)
    assert len(application) == 12 + size + size % 2
    # The verbatim payload is a real Tech 3285 chunk: the 602-byte fixed
    # part (version at offset 346, here 2) plus the CodingHistory tail.
    assert size >= 602
    (version,) = struct.unpack_from("<H", application, 12 + 346)
    assert version == 2


@needs_soundfile
def test_flac_without_foreign_metadata_has_no_provenance(
    tmp_path: Path,
) -> None:
    path = tmp_path / "plain.flac"
    write(path, np.zeros(8), FS)
    assert read(path).provenance is None
    assert info(path).bext is None


@needs_soundfile
def test_embedded_block_survives_a_third_party_decoder(tmp_path: Path) -> None:
    """soundfile itself must still read the file after the block insert."""
    codes = np.arange(-64, 64, dtype=np.int64) * 65536
    path = tmp_path / "still_valid.flac"
    write(path, codes / 2**23, FS, bext=fresh_metadata())
    decoded, fs = sf.read(str(path), dtype="int32")
    assert fs == FS
    assert (decoded >> 8).tolist() == codes.tolist()


def test_flac_target_without_the_extra_names_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_import = builtins.__import__

    def blocked(name: str, *args: object, **kwargs: object) -> object:
        if name.split(".")[0] == "soundfile":
            msg = "blocked for the test"
            raise ImportError(msg)
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", blocked)
    with pytest.raises(ImportError, match=r"phonometry\[audio\]"):
        write(tmp_path / "x.flac", np.zeros(4), FS)
