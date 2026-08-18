#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Tests for phonometry.io.write: exact codes, loud clipping, honest RF64.

The oracles are independent of the writer: expected byte images come from
tests/wav_forge.py (hand-assembled from the container specifications) or
from arithmetic done longhand in the test, and reread comparisons go
through the phase-tested reader whose scaling was itself pinned against
hand-computed values. Quantisation checks are exact equality -- the write
scaling is the exact inverse of the read scaling, so a value that is only
approximately right is wrong.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest
from wav_forge import pcm_data

from phonometry.io import ClippingWarning, Signal, info, read, write
from phonometry.io._chunks import parse_wav_chunks
from phonometry.io._write import build_wav_header, write_wav_stream

FS = 48000


# ---------------------------------------------------------------------------
# Round-trips per subtype: write with our writer, reread with our reader
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(("subtype", "bits"), [
    ("PCM_16", 16), ("PCM_24", 24), ("PCM_32", 32),
])
def test_integer_subtypes_roundtrip_grid_values_exactly(
    tmp_path: Path, subtype: str, bits: int
) -> None:
    """Values on the target grid survive write+read bit for bit."""
    scale = 2 ** (bits - 1)
    codes = np.array([0, 1, -1, scale // 2, scale - 1, -scale], dtype=np.int64)
    path = tmp_path / "grid.wav"
    write(path, codes / scale, FS, subtype=subtype)
    sig = read(path)
    assert np.asarray(sig).tolist() == (codes / scale).tolist()
    assert info(path).bit_depth == bits


def test_float_subtype_is_exact_for_24bit_sourced_values(tmp_path: Path) -> None:
    """The FLOAT default represents any 24-bit-container value exactly."""
    codes = np.array([0, 1, -1, 2**23 - 1, -(2**23), 5_000_001], dtype=np.int64)
    path = tmp_path / "f32.wav"
    write(path, codes / 2**23, FS)  # subtype defaults to FLOAT
    sig = read(path)
    assert np.asarray(sig).tolist() == (codes / 2**23).tolist()
    assert info(path).format_name == "IEEE float"


def test_double_subtype_roundtrips_float64_bit_for_bit(tmp_path: Path) -> None:
    rng = np.random.default_rng(1234)
    x = rng.standard_normal(1024) * 0.3
    path = tmp_path / "f64.wav"
    write(path, x, FS, subtype="DOUBLE")
    assert np.asarray(read(path)).tolist() == x.tolist()


def test_stereo_interleaving_keeps_channel_identity(tmp_path: Path) -> None:
    """Distinct per-channel content comes back on its own channel."""
    left = np.linspace(-0.5, 0.5, 32)
    right = -left
    path = tmp_path / "stereo.wav"
    write(path, np.stack([left, right]), FS, subtype="DOUBLE")
    sig = read(path)
    assert sig.n_channels == 2
    assert np.asarray(sig)[0].tolist() == left.tolist()
    assert np.asarray(sig)[1].tolist() == right.tolist()


def test_signal_input_supplies_fs_and_refuses_a_conflicting_one(
    tmp_path: Path,
) -> None:
    sig = Signal(data=np.zeros(8), fs=44100)
    path = tmp_path / "fromsignal.wav"
    write(path, sig)
    assert read(path).fs == 44100
    with pytest.raises(ValueError, match="conflicts"):
        write(path, sig, FS)


def test_bare_array_requires_fs(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="fs is required"):
        write(tmp_path / "nofs.wav", np.zeros(8))


# ---------------------------------------------------------------------------
# 24-bit: the packing scipy lacks, pinned against the forge's own packer
# ---------------------------------------------------------------------------

def test_pcm24_payload_matches_the_hand_packed_oracle(tmp_path: Path) -> None:
    """The in-house 3-byte packing equals wav_forge's independent packer."""
    codes = np.array([0, 1, -1, 2**23 - 1, -(2**23), 123456, -654321],
                     dtype=np.int64)
    path = tmp_path / "packed.wav"
    write(path, codes / 2**23, FS, subtype="PCM_24")
    parsed = parse_wav_chunks(path)
    payload = path.read_bytes()[
        parsed.data_offset:parsed.data_offset + parsed.data_size
    ]
    assert payload == pcm_data(codes, 24)


def test_pcm24_stereo_roundtrips_through_scipy_decoding(tmp_path: Path) -> None:
    """Interleaved 24-bit stereo rereads exactly (scipy left-justifies)."""
    left = np.arange(-8, 8, dtype=np.int64) * 40503
    right = left[::-1]
    path = tmp_path / "s24.wav"
    write(path, np.stack([left, right]) / 2**23, FS, subtype="PCM_24")
    sig = read(path)
    assert np.asarray(sig)[0].tolist() == (left / 2**23).tolist()
    assert np.asarray(sig)[1].tolist() == (right / 2**23).tolist()


# ---------------------------------------------------------------------------
# Integer pass-through: the archival path stores the codes themselves
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(("dtype", "bits"), [(np.int16, 16), (np.int32, 32)])
def test_integer_input_passes_codes_through_bit_exact(
    tmp_path: Path, dtype: type, bits: int
) -> None:
    codes = np.array([0, 1, -1, 2**(bits - 1) - 1, -(2**(bits - 1))],
                     dtype=dtype)
    path = tmp_path / "codes.wav"
    write(path, codes, FS)
    assert np.asarray(read(path)).tolist() == (codes / 2**(bits - 1)).tolist()


def test_integer_input_refuses_a_conflicting_subtype(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="pass-through"):
        write(tmp_path / "c.wav", np.zeros(4, dtype=np.int16), FS,
              subtype="PCM_32")


def test_integer_input_refuses_dither(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="nothing to dither"):
        write(tmp_path / "c.wav", np.zeros(4, dtype=np.int16), FS,
              dither="tpdf")


# ---------------------------------------------------------------------------
# Clipping: saturated, counted, announced; never rescaled, never wrapped
# ---------------------------------------------------------------------------

def test_clipping_saturates_counts_and_warns(tmp_path: Path) -> None:
    x = np.array([0.5, 1.5, -2.0, -0.25])
    path = tmp_path / "hot.wav"
    with pytest.warns(ClippingWarning, match=r"2 of 4 samples"):
        write(path, x, FS, subtype="PCM_16")
    got = np.asarray(read(path))
    # Saturation, not modulo wrap-around: the hot samples pin to the rails.
    assert got.tolist() == [0.5, 32767 / 32768, -1.0, -0.25]


def test_positive_full_scale_is_the_documented_one_lsb_clip(
    tmp_path: Path,
) -> None:
    """+1.0's ideal code is 2^(B-1), one above the top code: reported."""
    path = tmp_path / "edge.wav"
    with pytest.warns(ClippingWarning, match=r"1 of 3 samples"):
        write(path, np.array([-1.0, 0.0, 1.0]), FS, subtype="PCM_16")
    assert np.asarray(read(path)).tolist() == [-1.0, 0.0, 32767 / 32768]


def test_float_subtypes_carry_over_range_values_without_warning(
    tmp_path: Path,
) -> None:
    """IEEE float has no full-scale wall: +1.5 is stored, not clipped."""
    import warnings

    path = tmp_path / "hotfloat.wav"
    with warnings.catch_warnings():
        warnings.simplefilter("error", ClippingWarning)
        write(path, np.array([1.5, -1.25]), FS, subtype="DOUBLE")
    assert np.asarray(read(path)).tolist() == [1.5, -1.25]


# ---------------------------------------------------------------------------
# Dither: TPDF at 16 bits only, unbiased where plain rounding is biased
# ---------------------------------------------------------------------------

def test_tpdf_dither_renders_the_mean_code_unbiased(tmp_path: Path) -> None:
    """A constant between codes averages to itself only under dither.

    Plain rounding writes the same nearest code for every sample, a bias
    of -0.4 LSB for a value 0.4 above the code; TPDF dither spreads the
    codes so their mean converges on the true value (the first-moment
    independence of Lipshitz et al. 1992). With 20000 samples the standard
    error is about 0.005 LSB, so the 0.1 LSB tolerance is twenty sigma.
    The seeded ``rng`` makes the draw, and so the whole file, exact: a
    second write from an equally seeded generator is byte-identical.
    """
    value = 10000.4 / 32768
    x = np.full(20000, value)
    plain = tmp_path / "plain.wav"
    write(plain, x, FS, subtype="PCM_16")
    codes_plain = np.asarray(read(plain)) * 32768
    assert set(codes_plain.tolist()) == {10000.0}

    dithered = tmp_path / "dithered.wav"
    write(dithered, x, FS, subtype="PCM_16", dither="tpdf",
          rng=np.random.default_rng(1992))
    codes_dithered = np.asarray(read(dithered)) * 32768
    assert len(set(codes_dithered.tolist())) > 1
    assert abs(float(np.mean(codes_dithered)) - 10000.4) < 0.1

    again = tmp_path / "again.wav"
    write(again, x, FS, subtype="PCM_16", dither="tpdf",
          rng=np.random.default_rng(1992))
    assert again.read_bytes() == dithered.read_bytes()


def test_dither_is_refused_outside_pcm16(tmp_path: Path) -> None:
    for subtype in ("PCM_24", "PCM_32", "FLOAT", "DOUBLE"):
        with pytest.raises(ValueError, match="PCM_16"):
            write(tmp_path / "d.wav", np.zeros(4), FS, subtype=subtype,
                  dither="tpdf")
    with pytest.raises(ValueError, match="dither"):
        write(tmp_path / "d.wav", np.zeros(4), FS, dither="rectangular")
    # rng exists to seed the dither; alone it would promise a
    # reproducibility nothing delivers.
    silence = np.zeros(4)
    with pytest.raises(ValueError, match="rng"):
        write(tmp_path / "d.wav", silence, FS, rng=7)


# ---------------------------------------------------------------------------
# Containers: suffix dispatch and the RF64 promotion threshold
# ---------------------------------------------------------------------------

def test_lossy_and_unknown_targets_are_refused(tmp_path: Path) -> None:
    for name in ("copy.mp3", "copy.ogg", "copy.opus", "copy.xyz"):
        with pytest.raises(ValueError, match="WAV family"):
            write(tmp_path / name, np.zeros(4), FS)


def test_header_stays_riff_at_the_4gib_boundary() -> None:
    """The largest RIFF payload that fits 32 bits is still written as RIFF.

    Solved for the frame count whose mono PCM_16 header lands the RIFF
    payload exactly on 0xFFFFFFFF: payload = 4 + 24 (fmt) + 8 + 2*frames.
    """
    frames = (0xFFFFFFFF - 36) // 2
    header = build_wav_header(fs=FS, channels=1, subtype="PCM_16",
                              frames=frames)
    assert header[:4] == b"RIFF"
    assert struct.unpack_from("<I", header, 4)[0] == 0xFFFFFFFF - 1
    # One frame more (payload 0x100000001, past the sentinel) promotes.
    promoted = build_wav_header(fs=FS, channels=1, subtype="PCM_16",
                                frames=frames + 1)
    assert promoted[:4] == b"RF64"


def test_rf64_header_carries_the_real_sizes_in_ds64() -> None:
    """Past 4 GiB the header is EBU Tech 3306: ds64 first, sentinels after."""
    frames = 2**31  # 4 GiB of stereo PCM_16 data: 8 GiB payload
    header = build_wav_header(fs=FS, channels=2, subtype="PCM_16",
                              frames=frames)
    assert header[:4] == b"RF64"
    assert struct.unpack_from("<I", header, 4)[0] == 0xFFFFFFFF
    assert header[8:16] == b"WAVEds64"
    riff_size, data_size, frame_count = struct.unpack_from("<QQQ", header, 20)
    assert data_size == frames * 4
    assert frame_count == frames
    assert riff_size == len(header) - 8 + data_size
    # The 32-bit data size field holds the sentinel, not a truncated size.
    assert header.endswith(b"data" + b"\xff\xff\xff\xff")


def test_float_header_carries_fact_and_cbsize() -> None:
    """Non-PCM fmt gets cbSize=0 and a fact frame count (1991 RIFF spec)."""
    header = build_wav_header(fs=FS, channels=1, subtype="FLOAT", frames=7)
    assert struct.unpack_from("<I", header, 16)[0] == 18  # fmt size with cbSize
    fact_at = header.index(b"fact")
    assert struct.unpack_from("<II", header, fact_at + 4) == (4, 7)


@pytest.mark.parametrize("actual", [1000, 1300])
def test_streamed_header_reconciles_with_an_estimated_frame_count(
    tmp_path: Path, actual: int
) -> None:
    """With ``frames_are_estimate`` the header follows the streamed count.

    A lossy source's decoder may deliver a different count than its
    container declares (decoder delay and padding); the sizes on disk
    must then describe the samples that landed, not the declaration --
    in both directions, short and long of the promise of 1152.
    """
    path = tmp_path / "reconciled.wav"
    write_wav_stream(
        path, [np.zeros((1, actual))], FS, 1, "FLOAT", 1152,
        frames_are_estimate=True,
    )
    assert info(path).frames == actual
    assert read(path).n_samples == actual
    # The outer RIFF size agrees with the bytes actually on disk.
    (riff_size,) = struct.unpack_from("<I", path.read_bytes(), 4)
    assert path.stat().st_size == 8 + riff_size


def test_streamed_count_mismatch_without_an_estimate_still_raises(
    tmp_path: Path,
) -> None:
    """The exact-count contract keeps refusing a short or long stream."""
    path = tmp_path / "mismatch.wav"
    with pytest.raises(ValueError, match="inconsistent"):
        write_wav_stream(path, [np.zeros((1, 10))], FS, 1, "FLOAT", 12)


def test_odd_pcm24_payload_gets_its_pad_byte(tmp_path: Path) -> None:
    """A mono 24-bit file with odd frames stays word-aligned on disk."""
    path = tmp_path / "odd.wav"
    write(path, np.array([0.25, -0.25, 0.5]), FS, subtype="PCM_24")
    assert path.stat().st_size % 2 == 0
    parsed = parse_wav_chunks(path)
    assert parsed.data_size == 9  # the pad byte is not part of the payload
    assert read(path).n_samples == 3
