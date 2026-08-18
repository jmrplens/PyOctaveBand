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

from phonometry import filters, leq
from phonometry.io import LossyCompressionWarning, read, read_blocks, write

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
    blocks = iter(read_blocks(path, 1024))
    with pytest.warns(LossyCompressionWarning, match="not metrologically"):
        next(blocks)


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
    rng = np.random.default_rng(99)
    x = rng.uniform(-0.9, 0.9, (2, 1000))
    path = tmp_path / "meas.wav"
    write(path, x, FS, subtype="PCM_24")
    whole = np.asarray(read(path))
    assert np.concatenate(
        list(read_blocks(path, 256)), axis=-1
    ).tolist() == whole.tolist()


# ---------------------------------------------------------------------------
# The equivalence oracle: a level computed from the stream is the same
# number as the level computed from the whole file. This is the property
# that makes read_blocks usable for measurement at all -- streaming is an
# implementation strategy, never a different answer -- so it is asserted
# at float64 precision, not to some engineering tolerance.
# ---------------------------------------------------------------------------

def _streamed_leq(path: Path, block_size: int, overlap: int) -> float:
    """Accumulate a running Leq from the block stream, dB re 20 uPa.

    Energy metrics accumulate across blocks (the block-processing guide's
    rule); with ``overlap`` the repeated head of each later block is
    dropped so every frame is counted exactly once.
    """
    total, frames = 0.0, 0
    for i, block in enumerate(read_blocks(path, block_size, overlap=overlap)):
        fresh = block[..., overlap:] if i else block
        total += float(np.sum(np.asarray(fresh) ** 2))
        frames += fresh.shape[-1]
    return float(10 * np.log10((total / frames) / (2e-5) ** 2))


def _long_synthetic(seconds: float = 3.0) -> np.ndarray:
    """A tone in noise, long enough that many blocks tile it unevenly."""
    rng = np.random.default_rng(2026)
    t = np.arange(int(seconds * FS)) / FS
    return 0.05 * np.sin(2 * np.pi * 1000 * t) + 0.02 * rng.standard_normal(
        t.size
    )


@pytest.mark.parametrize("overlap", [0, 480])
def test_streamed_leq_equals_the_whole_file_leq(
    tmp_path: Path, overlap: int
) -> None:
    """Block-accumulated Leq == whole-file Leq, with and without overlap."""
    x = _long_synthetic()
    path = tmp_path / "night.wav"
    write(path, x, FS, subtype="DOUBLE")  # bit-exact container: no
    # quantisation muddies the comparison; the only difference left
    # between the two paths is summation order (~1e-13 dB).
    whole = leq(np.asarray(read(path)))
    # 4800-frame blocks do not divide 144000 - overlap tilings evenly.
    assert _streamed_leq(path, 4800, overlap) == pytest.approx(
        whole, abs=1e-9
    )


@pytest.mark.parametrize("overlap", [0, 1200])
def test_streamed_stateful_laeq_equals_the_single_pass(
    tmp_path: Path, overlap: int
) -> None:
    """read_blocks feeding a stateful WeightingFilter matches one pass.

    The stream drives the existing block machinery exactly as the
    block-processing guide prescribes (fresh frames only into the
    filter, state carried across calls); the reference is the same
    bilinear design run over the whole file in one call, which the
    guide promises is bit-identical to the concatenated stream.
    """
    x = _long_synthetic()
    path = tmp_path / "night.wav"
    write(path, x, FS, subtype="DOUBLE")

    aw = filters.WeightingFilter(FS, "A", stateful=True)
    total, frames = 0.0, 0
    for i, block in enumerate(read_blocks(path, 4800, overlap=overlap)):
        fresh = block[overlap:] if i else block
        y = aw.filter(fresh)
        total += float(np.sum(y**2))
        frames += y.shape[-1]
    streamed = 10 * np.log10((total / frames) / (2e-5) ** 2)

    offline = leq(
        filters.WeightingFilter(FS, "A", high_accuracy=False).filter(
            np.asarray(read(path))
        )
    )
    assert streamed == pytest.approx(offline, abs=1e-9)


def test_streamed_band_leq_through_a_stateful_bank(tmp_path: Path) -> None:
    """read_blocks feeds BlockProcessing(stateful=True) unchanged.

    The 1 kHz octave band of a stateful OctaveFilterBank, fed block by
    block from the file, accumulates to the same band Leq as the
    offline bank over the whole read.
    """
    x = _long_synthetic()
    path = tmp_path / "night.wav"
    write(path, x, FS, subtype="DOUBLE")

    bank = filters.OctaveFilterBank(
        FS,
        fraction=1,
        limits=[900, 1100],
        design=filters.FilterDesign(resample=False),
        block_processing=filters.BlockProcessing(stateful=True),
    )
    total, frames = 0.0, 0
    for block in read_blocks(path, 4800):
        band = bank.filter(
            block, sigbands=True, detrend=False, calculate_level=False
        )[2][0]
        total += float(np.sum(band**2))
        frames += band.shape[-1]
    streamed = 10 * np.log10((total / frames) / (2e-5) ** 2)

    offline_band = filters.OctaveFilterBank(
        FS,
        fraction=1,
        limits=[900, 1100],
        design=filters.FilterDesign(resample=False),
    ).filter(
        np.asarray(read(path)),
        sigbands=True,
        detrend=False,
        calculate_level=False,
    )[2][0]
    assert streamed == pytest.approx(
        leq(offline_band), abs=1e-9
    )


@needs_soundfile
@pytest.mark.parametrize("overlap", [0, 800])
def test_streamed_leq_matches_across_backends(
    tmp_path: Path, overlap: int
) -> None:
    """WAV (base decoder) and FLAC (libsndfile) stream the same Leq.

    The same 24-bit codes go into both containers, so the decoded
    samples are bit-identical and the streamed levels must agree with
    each other and with the whole-file read at float64 precision.
    """
    rng = np.random.default_rng(7)
    codes = (rng.uniform(-0.5, 0.5, FS) * 2**23).astype(np.int64)
    wav = _write(tmp_path, pcm_wav(codes, bits=24), "same.wav")
    flac = tmp_path / "same.flac"
    sf.write(str(flac), (codes << 8).astype(np.int32), FS, subtype="PCM_24")

    whole = leq(np.asarray(read(wav)))
    assert _streamed_leq(wav, 4800, overlap) == pytest.approx(whole, abs=1e-9)
    assert _streamed_leq(flac, 4800, overlap) == pytest.approx(whole, abs=1e-9)
