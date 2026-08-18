#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Tests for convert(): samples, bext and sidecar across container trips.

The central contract test is the one no generic tool passes: a 24-bit BWF
with provenance and a calibration sidecar goes to FLAC and back, and the
samples are bit-identical, the bext static fields identical, the coding
history extended by exactly one line per hop, and the calibration still
beside the file. Small block sizes force real multi-block streaming so
the boundaries are exercised, not avoided.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from wav_forge import rf64_wav

from phonometry import __version__
from phonometry.io import (
    ClippingWarning,
    LossyCompressionWarning,
    convert,
    info,
    read,
    read_sidecar,
    sidecar_path,
    write,
    write_sidecar,
)
from phonometry.io._bext import fresh_metadata

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
def test_wav24_to_flac_and_back_loses_nothing(tmp_path: Path) -> None:
    """The contract test: samples bit-identical, provenance whole."""
    rng = np.random.default_rng(2026)
    codes = rng.integers(-(2**23), 2**23, 3000)
    meta = replace(
        fresh_metadata(),
        description="facade measurement, position 2",
        originator="SVAN 979",
        time_reference=(7 << 32) | 42,
        umid=bytes(range(64)),
        coding_history="A=PCM,F=48000,W=24,M=mono,T=recorder\r\n",
    )
    original = tmp_path / "meas.wav"
    write(original, codes / 2**23, FS, subtype="PCM_24", bext=meta)
    write_sidecar(original, 12.5, reference_spl=94.0)

    archived = tmp_path / "meas.flac"
    convert(original, archived, block_size=700)  # forces 5 blocks, odd tail
    restored = tmp_path / "restored.wav"
    convert(archived, restored, block_size=999)

    for path in (archived, restored):
        assert np.asarray(read(path)).tolist() == (codes / 2**23).tolist()

    line_written = f"A=PCM,F={FS},W=24,M=mono,T=phonometry {__version__}"
    line_flac = f"A=FLAC,F={FS},W=24,M=mono,T=phonometry {__version__}"
    got = read(restored).provenance
    assert got is not None
    # Every static field identical; the audit trail one line per hop.
    assert got == replace(
        meta,
        coding_history=(
            meta.coding_history
            + line_written + "\r\n" + line_flac + "\r\n" + line_written
        ),
    )
    # The calibration sidecar rode along byte for byte at each hop.
    for path in (archived, restored):
        carried = read_sidecar(path)
        assert carried is not None
        assert carried.calibration_factor == 12.5
        assert carried.reference_spl == 94.0
        assert sidecar_path(path).read_bytes() == (
            sidecar_path(original).read_bytes()
        )
    # And read() therefore returns the restored file already calibrated.
    assert read(restored).calibration_factor == 12.5


def test_float64_wav_conversion_is_the_identity(tmp_path: Path) -> None:
    rng = np.random.default_rng(31337)
    x = rng.standard_normal((2, 1500)) * 0.2
    src = tmp_path / "work.wav"
    write(src, x, FS, subtype="DOUBLE")
    dst = tmp_path / "copy.wav"
    result = convert(src, dst, block_size=64)
    assert np.asarray(read(dst)).tolist() == x.tolist()
    # The default subtype preserved the source's float width...
    assert result.format_name == "IEEE float"
    assert result.bit_depth == 64
    # ...and the return value is the destination's own description.
    assert result.path == str(dst)
    assert result.frames == 1500


def test_int16_wav_default_preserves_the_depth(tmp_path: Path) -> None:
    codes = np.array([0, 1, -1, 32767, -32768], dtype=np.int16)
    src = tmp_path / "a.wav"
    write(src, codes, FS)
    dst = tmp_path / "b.wav"
    assert convert(src, dst).bit_depth == 16
    assert np.asarray(read(dst)).tolist() == (codes / 32768).tolist()


def test_rf64_source_streams_through(tmp_path: Path) -> None:
    """The forged RF64 (real ds64, sentinel sizes) converts like any WAV."""
    codes = np.arange(-500, 500, dtype=np.int64) * 60
    src = tmp_path / "long.wav"
    src.write_bytes(rf64_wav(codes, bits=16))
    dst = tmp_path / "flat.wav"
    convert(src, dst, block_size=128)
    assert np.asarray(read(dst)).tolist() == (codes / 32768).tolist()


def test_explicit_narrowing_reports_the_accumulated_clipping(
    tmp_path: Path,
) -> None:
    """Hot samples spread over many blocks are counted once, in total."""
    x = np.zeros(1000)
    x[::100] = 1.5  # ten hot samples, one per streamed block
    src = tmp_path / "hot.wav"
    write(src, x, FS, subtype="DOUBLE")
    dst = tmp_path / "narrow.wav"
    with pytest.warns(ClippingWarning, match="10 of 1000"):
        convert(src, dst, subtype="PCM_16", block_size=100)
    assert np.asarray(read(dst)).max() == 32767 / 32768


@needs_soundfile
def test_lossy_source_converts_with_the_metrology_warning(
    tmp_path: Path,
) -> None:
    x = 0.3 * np.sin(2 * np.pi * 997 * np.arange(FS // 4) / FS)
    src = tmp_path / "note.mp3"
    sf.write(str(src), x, FS)
    dst = tmp_path / "expanded.wav"
    with pytest.warns(LossyCompressionWarning, match="not metrologically"):
        result = convert(src, dst)
    # The decoder hands floats; the default keeps them as FLOAT.
    assert result.format_name == "IEEE float"
    # A lossy container's declared count and its decoder's delivery may
    # differ by decoder delay/padding, so the contract is internal
    # consistency (the written header equals the streamed samples) plus
    # agreement with the source within one MP3 granule train (1152
    # frames), not exact equality with the source header.
    assert result.frames == np.asarray(read(dst)).shape[-1]
    assert abs(result.frames - info(src).frames) <= 1152


def test_lossy_targets_are_refused_before_reading_anything(
    tmp_path: Path,
) -> None:
    for name in ("y.mp3", "y.ogg", "y.opus", "y.m4a"):
        with pytest.raises(ValueError, match="outside this API by policy"):
            convert(tmp_path / "never-read.wav", tmp_path / name)
    with pytest.raises(ValueError, match="unsupported target"):
        convert(tmp_path / "never-read.wav", tmp_path / "y.xyz")


def test_same_file_is_refused(tmp_path: Path) -> None:
    src = tmp_path / "self.wav"
    write(src, np.zeros(8), FS)
    with pytest.raises(ValueError, match="same file"):
        convert(src, src)


@needs_soundfile
def test_flac_target_demands_an_explicit_choice_for_what_it_cannot_hold(
    tmp_path: Path,
) -> None:
    floats = tmp_path / "f.wav"
    write(floats, np.zeros(8), FS, subtype="DOUBLE")
    with pytest.raises(ValueError, match="integer PCM only"):
        convert(floats, tmp_path / "f.flac")
    ints32 = tmp_path / "i.wav"
    write(ints32, np.zeros(8, dtype=np.int32), FS)
    with pytest.raises(ValueError, match="cannot pass through whole"):
        convert(ints32, tmp_path / "i.flac")
    # The explicit choice is honoured.
    convert(floats, tmp_path / "ok.flac", subtype="PCM_24")
    assert info(tmp_path / "ok.flac").bit_depth == 24


def test_a_bextless_source_stays_bextless(tmp_path: Path) -> None:
    """convert never invents provenance it does not have."""
    src = tmp_path / "plain.wav"
    write(src, np.zeros(16), FS, subtype="PCM_16")
    dst = tmp_path / "copy.wav"
    assert convert(src, dst).bext is None
