#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Tests for read_blocks: block trains must equal the whole-file read.

The oracle is read() itself (whose scaling is pinned against hand-computed
values), compared sample for sample: streaming is an implementation
strategy, never a different answer. Files come from the wav_forge
hand-assembly helpers so every depth, the RF64 layout and awkward
geometries (blocks that do not divide the length, overlap) are covered.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from wav_forge import chunk, float_wav, fmt_payload, pcm_wav, rf64_wav, riff_wave

from phonometry.io import LossyCompressionWarning, read, read_blocks

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


def _write(tmp_path: Path, image: bytes, name: str = "forged.wav") -> Path:
    path = tmp_path / name
    path.write_bytes(image)
    return path


def _reassemble(path: Path, block_size: int) -> np.ndarray:
    blocks = [np.atleast_2d(b) for b in read_blocks(path, block_size)]
    return np.concatenate(blocks, axis=-1)


@pytest.mark.parametrize("bits", [8, 16, 24, 32])
def test_pcm_blocks_reassemble_the_whole_read(tmp_path: Path, bits: int) -> None:
    rng = np.random.default_rng(bits)
    codes = rng.integers(-(2 ** (bits - 1)), 2 ** (bits - 1), 61)
    if bits == 8:
        codes = codes + 2 ** (bits - 1)  # the WAV 8-bit unsigned convention
    path = _write(tmp_path, pcm_wav(codes, bits=bits))
    whole = np.atleast_2d(np.asarray(read(path)))
    # 61 frames over 16-frame blocks: three full blocks and a 13-frame tail.
    assert _reassemble(path, 16).tolist() == whole.tolist()


@pytest.mark.parametrize("bits", [32, 64])
def test_float_blocks_reassemble_the_whole_read(
    tmp_path: Path, bits: int
) -> None:
    rng = np.random.default_rng(bits)
    x = rng.standard_normal(45) * 0.25
    path = _write(tmp_path, float_wav(x, bits=bits))
    whole = np.atleast_2d(np.asarray(read(path)))
    assert _reassemble(path, 8).tolist() == whole.tolist()


def test_stereo_blocks_keep_channel_geometry(tmp_path: Path) -> None:
    frames = np.arange(-40, 40, dtype=np.int64).reshape(40, 2) * 400
    path = _write(tmp_path, pcm_wav(frames, bits=16, channels=2))
    whole = np.asarray(read(path))
    blocks = list(read_blocks(path, 16))
    assert all(b.shape[0] == 2 for b in blocks)
    assert [b.shape[-1] for b in blocks] == [16, 16, 8]
    assert np.concatenate(blocks, axis=-1).tolist() == whole.tolist()


def test_rf64_streams_through_the_ds64_sizes(tmp_path: Path) -> None:
    codes = np.arange(-24, 24, dtype=np.int64) * 512
    path = _write(tmp_path, rf64_wav(codes, bits=16))
    whole = np.atleast_2d(np.asarray(read(path)))
    assert _reassemble(path, 7).tolist() == whole.tolist()


def test_mono_blocks_are_one_dimensional(tmp_path: Path) -> None:
    path = _write(tmp_path, pcm_wav(np.zeros(10, dtype=np.int64), bits=16))
    assert all(b.ndim == 1 for b in read_blocks(path, 4))


def test_overlap_repeats_the_tail_of_each_block(tmp_path: Path) -> None:
    """Blocks advance by block_size - overlap, per the libsndfile rule."""
    codes = np.arange(10, dtype=np.int64) * 1000 - 5000
    path = _write(tmp_path, pcm_wav(codes, bits=16))
    whole = np.asarray(read(path))
    blocks = list(read_blocks(path, 4, overlap=2))
    assert [b.tolist() for b in blocks] == [
        whole[0:4].tolist(), whole[2:6].tolist(),
        whole[4:8].tolist(), whole[6:10].tolist(),
    ]


@needs_soundfile
def test_the_overlap_rule_matches_libsndfile_exactly(tmp_path: Path) -> None:
    """Both backends must yield the same block train for the same signal."""
    codes = np.arange(10, dtype=np.int64) * 1000 - 5000
    wav = _write(tmp_path, pcm_wav(codes, bits=16))
    flac = tmp_path / "same.flac"
    sf.write(str(flac), (codes / 32768), FS, subtype="PCM_16")
    for overlap in (0, 2):
        ours = [b.tolist() for b in read_blocks(wav, 4, overlap=overlap)]
        theirs = [b.tolist() for b in read_blocks(flac, 4, overlap=overlap)]
        assert ours == theirs


@needs_soundfile
def test_flac_blocks_reassemble_the_whole_read(tmp_path: Path) -> None:
    codes = np.arange(-30, 31, dtype=np.int64) * 100000
    path = tmp_path / "long.flac"
    sf.write(str(path), (codes << 8).astype(np.int32), FS, subtype="PCM_24")
    whole = np.atleast_2d(np.asarray(read(path)))
    assert _reassemble(path, 9).tolist() == whole.tolist()


@needs_soundfile
def test_lossy_sources_warn_when_streaming_begins(tmp_path: Path) -> None:
    x = 0.3 * np.sin(2 * np.pi * 1000 * np.arange(FS // 4) / FS)
    path = tmp_path / "note.mp3"
    sf.write(str(path), x, FS)
    with pytest.warns(LossyCompressionWarning, match="not metrologically"):
        next(iter(read_blocks(path, 1024)))


def test_invalid_geometry_is_refused(tmp_path: Path) -> None:
    path = _write(tmp_path, pcm_wav(np.zeros(8, dtype=np.int64), bits=16))
    with pytest.raises(ValueError, match="block_size"):
        read_blocks(path, 0)
    with pytest.raises(ValueError, match="overlap"):
        read_blocks(path, 4, overlap=4)
    with pytest.raises(ValueError, match="overlap"):
        read_blocks(path, 4, overlap=-1)


def test_a_truncated_data_chunk_fails_mid_stream(tmp_path: Path) -> None:
    """A data chunk shorter than its header claims is an error, not silence."""
    image = riff_wave(
        chunk(b"fmt ", fmt_payload(bits=16)),
        # Claims 40 bytes (20 frames) but carries only 10 frames.
        chunk(b"data", np.arange(10, dtype="<i2").tobytes(),
              declared_size=40),
    )
    path = _write(tmp_path, image)
    blocks = read_blocks(path, 8)
    next(blocks)  # frames 0-8 exist
    with pytest.raises(ValueError, match="bytes short"):
        list(blocks)


def test_blocks_agree_with_read_on_a_written_measurement(
    tmp_path: Path,
) -> None:
    """End to end through the module's own writer at 24-bit."""
    from phonometry.io import write

    rng = np.random.default_rng(99)
    x = rng.uniform(-0.9, 0.9, (2, 1000))
    path = tmp_path / "meas.wav"
    write(path, x, FS, subtype="PCM_24")
    whole = np.asarray(read(path))
    assert np.concatenate(
        list(read_blocks(path, 256)), axis=-1
    ).tolist() == whole.tolist()
