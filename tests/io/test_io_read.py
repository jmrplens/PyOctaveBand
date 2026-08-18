#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Tests for phonometry.io.read/info over forged files at every depth.

The WAV images are assembled by hand (tests/wav_forge.py), the expected
float values are computed by hand from the documented scaling (divide by
2**(bits-1), unsigned 8-bit re-centred on 128), and every comparison of
scaled integers is exact equality: the divisor is a power of two, so the
mapping must be bit-exact, not approximately right.

The ``[audio]`` lane (FLAC/MP3/compressed WAV through python-soundfile)
skips cleanly when the extra is not installed; the missing-extra error
message is tested by blocking the import, so it runs either way.
"""

from __future__ import annotations

import builtins
import struct
import warnings
from pathlib import Path

import numpy as np
import pytest
from wav_forge import (
    chunk,
    extensible_fmt_payload,
    float_wav,
    fmt_payload,
    pcm_data,
    pcm_wav,
    rf64_wav,
    riff_wave,
)

from phonometry import leq
from phonometry.io import LossyCompressionWarning, Signal, info, read

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


# ---------------------------------------------------------------------------
# Scaling: hand-computed values, exact equality
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("bits", "codes", "expected"),
    [
        # 8-bit WAV is unsigned around 128: (code - 128) / 128.
        (8, [192, 64, 128, 255], [0.5, -0.5, 0.0, 127 / 128]),
        (16, [16384, -32768, 0, 32767], [0.5, -1.0, 0.0, 32767 / 32768]),
        # 24-bit: scipy returns the sample left-justified in int32, so
        # dividing by 2**31 is dividing the 24-bit code by 2**23.
        (24, [2**22, -(2**23), 0, 2**23 - 1],
         [0.5, -1.0, 0.0, (2**23 - 1) / 2**23]),
        (32, [2**30, -(2**31), 0, 2**31 - 1],
         [0.5, -1.0, 0.0, (2**31 - 1) / 2**31]),
    ],
    ids=["pcm8", "pcm16", "pcm24", "pcm32"],
)
def test_pcm_scaling_matches_hand_computed_values(
    tmp_path: Path, bits: int, codes: list[int], expected: list[float]
) -> None:
    path = _write(tmp_path, pcm_wav(np.array(codes), bits=bits))
    sig = read(path)
    assert isinstance(sig, Signal)
    assert sig.fs == FS
    assert sig.dtype == np.float64
    assert np.asarray(sig).tolist() == expected
    assert sig.source is not None
    assert sig.source.bit_depth == bits
    assert not sig.source.lossy


@pytest.mark.parametrize("bits", [32, 64], ids=["float32", "float64"])
def test_float_wavs_pass_through_unscaled(tmp_path: Path, bits: int) -> None:
    values = [0.5, -0.25, 1.5, -2.0]  # float WAV admits |x| > 1; keep it
    path = _write(tmp_path, float_wav(np.array(values), bits=bits))
    sig = read(path)
    assert np.asarray(sig).tolist() == values
    assert sig.source is not None
    assert sig.source.format_name == "IEEE float"


def test_stereo_deinterleaves_to_channels_first(tmp_path: Path) -> None:
    """Interleaved L R L R frames must come back as (channels, samples)."""
    left = np.array([100, 200, 300])
    right = np.array([-100, -200, -300])
    interleaved = np.stack([left, right], axis=1)  # frames x channels
    path = _write(tmp_path, pcm_wav(interleaved, bits=16, channels=2))
    sig = read(path)
    assert sig.shape == (2, 3)
    np.testing.assert_array_equal(np.asarray(sig)[0], left / 32768)
    np.testing.assert_array_equal(np.asarray(sig)[1], right / 32768)
    assert sig.data.flags["C_CONTIGUOUS"]


def test_read_memory_is_owned_and_writable(tmp_path: Path) -> None:
    """The signal must own real memory, not a view of scipy's read-only mmap.

    16-bit goes through the memory-mapped scipy path and 24-bit through the
    plain one; both must come back writable and backed by heap memory, or
    downstream in-place processing would raise on one file and not another
    and a lingering memmap would pin the file open.
    """
    for bits in (16, 24):
        path = _write(tmp_path, pcm_wav(np.array([1, 2, 3]), bits=bits))
        sig = read(path)
        assert sig.data.flags.writeable, bits
        base = sig.data
        while isinstance(base, np.ndarray) and base.base is not None:
            base = base.base
        assert not isinstance(base, np.memmap), bits


# ---------------------------------------------------------------------------
# Calibration: the scaling constant cancels, as the docstring derives
# ---------------------------------------------------------------------------

def test_calibration_cancels_the_scaling_convention(tmp_path: Path) -> None:
    """Calibrator + measurement through the same reader fix absolute level.

    The rig quantizes with 32767 (a recorder's convention) while the reader
    divides by 32768: the mismatch, and any fixed reader scaling at all,
    must cancel once the calibration factor is derived from the calibrator
    tone read through the same path. The recovered absolute level must
    match the exact continuous-signal value to quantization noise, far
    inside 0.01 dB.
    """
    rng = np.random.default_rng(1234)
    t = np.arange(FS) / FS

    def record(x: np.ndarray) -> np.ndarray:
        """What a 16-bit recorder writes: dithered rounding at 32767."""
        return np.round(x * 32767 + rng.uniform(-0.5, 0.5, x.size)).astype(np.int64)

    cal_fs = 0.3        # calibrator tone at 0.3 of full scale...
    meas_fs = 0.05      # ...and the measured signal 15.56 dB below it
    cal = np.sin(2 * np.pi * 1000 * t)
    meas = np.sin(2 * np.pi * 250 * t)
    cal_sig = read(_write(tmp_path, pcm_wav(record(cal_fs * cal), bits=16), "cal.wav"))
    meas_sig = read(_write(tmp_path, pcm_wav(record(meas_fs * meas), bits=16), "m.wav"))

    # IEC 60942 sound calibrator at 94.0 dB SPL: p_rms = 2e-5 * 10**(94/20) Pa
    # (1.00237 Pa; 1.0 Pa would be 93.98 dB).
    p_cal = 2e-5 * 10 ** (94.0 / 20.0)
    factor = p_cal / float(np.sqrt(np.mean(np.asarray(cal_sig) ** 2)))
    measured = leq(meas_sig, calibration_factor=factor)
    expected = 94.0 + 20 * np.log10(meas_fs / cal_fs)
    assert measured == pytest.approx(expected, abs=0.01)


def test_calibration_factor_is_attached_not_applied(tmp_path: Path) -> None:
    path = _write(tmp_path, pcm_wav(np.array([16384]), bits=16))
    sig = read(path, calibration_factor=2.5)
    assert sig.calibration_factor == 2.5
    # The samples stay in digital full scale; calibration is metadata.
    assert np.asarray(sig).tolist() == [0.5]


# ---------------------------------------------------------------------------
# Metadata: EXTENSIBLE, RF64, bext, info()
# ---------------------------------------------------------------------------

def test_extensible_channel_mask_labels_the_channels(tmp_path: Path) -> None:
    image = riff_wave(
        chunk(b"fmt ", extensible_fmt_payload(channels=2, channel_mask=0x3)),
        chunk(b"data", pcm_data(np.array([1, -1, 2, -2]), 16)),
    )
    path = _write(tmp_path, image)
    sig = read(path)
    assert sig.channel_labels == ("FL", "FR")
    described = info(path)
    assert described.channel_mask == 0x3
    assert described.channel_labels == ("FL", "FR")


def test_extensible_valid_bits_are_the_reported_depth(tmp_path: Path) -> None:
    """A 20-in-24-bit EXTENSIBLE file reports 20 bits, not the container."""
    image = riff_wave(
        chunk(b"fmt ", extensible_fmt_payload(channels=1, bits=24, valid_bits=20)),
        chunk(b"data", pcm_data(np.array([2**22]), 24)),
    )
    path = _write(tmp_path, image)
    assert info(path).bit_depth == 20
    sig = read(path)
    assert sig.source is not None
    assert sig.source.bit_depth == 20
    assert np.asarray(sig).tolist() == [0.5]


def test_rf64_reads_through_the_ds64_sizes(tmp_path: Path) -> None:
    codes = np.arange(-1000, 1000, dtype=np.int64)
    path = _write(tmp_path, rf64_wav(codes, bits=16))
    sig = read(path)
    np.testing.assert_array_equal(np.asarray(sig), codes / 32768)
    assert sig.source is not None
    assert sig.source.container == "RF64"
    described = info(path)
    assert described.container == "RF64"
    assert described.frames == codes.size
    assert described.duration == pytest.approx(codes.size / FS)


def _minimal_bext(originator: bytes) -> bytes:
    """A minimal v2 bext payload: originator at offset 256, version at 346."""
    buf = bytearray(602)
    buf[256:256 + len(originator)] = originator
    struct.pack_into("<H", buf, 346, 2)
    return bytes(buf)


def test_bext_provenance_reaches_the_signal(tmp_path: Path) -> None:
    image = pcm_wav(
        np.array([0, 1]), extra_chunks=chunk(b"bext", _minimal_bext(b"XL2 rig"))
    )
    sig = read(_write(tmp_path, image))
    assert sig.provenance is not None
    assert sig.provenance.originator == "XL2 rig"


def test_info_describes_headers_without_the_samples(tmp_path: Path) -> None:
    cue = struct.pack("<I", 1) + struct.pack("<II4sIII", 1, 0, b"data", 0, 0, 7)
    image = pcm_wav(
        np.arange(10),
        extra_chunks=(
            chunk(b"bext", _minimal_bext(b"rig"))
            + chunk(b"cue ", cue)
            + chunk(b"iXML", b"<BWFXML/>")
        ),
    )
    described = info(_write(tmp_path, image))
    assert described.container == "WAV"
    assert described.format_name == "PCM"
    assert described.fs == FS
    assert described.channels == 1
    assert described.frames == 10
    assert described.duration == pytest.approx(10 / FS)
    assert described.bit_depth == 16
    assert not described.lossy
    assert described.bext is not None
    assert described.bext.originator == "rig"
    assert described.has_ixml
    assert described.cue_points[0].sample_offset == 7


def _forged_full_bext() -> bytes:
    """A v2 bext payload with every field set, at its Tech 3285 offset.

    Assembled by hand, independently of the reader's struct, so the test
    is a second transcription of the field table: description at 0,
    originator at 256, reference at 288, date at 320, time at 330, the
    64-bit TimeReference split into its two 32-bit halves at 338/342
    (the value used here does not fit 32 bits, so joining the halves is
    actually exercised), version at 346, and the five R128 loudness
    int16s (value x100) from 412, one left at the 0x7FFF unset sentinel.
    """
    buf = bytearray(602)
    buf[0:11] = b"Facade SW-2"
    buf[256:263] = b"XL2 rig"
    buf[288:298] = b"serial-042"
    buf[320:330] = b"2026-08-17"
    buf[330:338] = b"23:59:58"
    tref = 5_000_000_000
    struct.pack_into("<II", buf, 338, tref & 0xFFFFFFFF, tref >> 32)
    struct.pack_into("<H", buf, 346, 2)
    struct.pack_into("<h", buf, 412, -2310)  # loudness_value, -23.10 LUFS
    struct.pack_into("<h", buf, 414, 520)  # loudness_range, 5.20 LU
    struct.pack_into("<h", buf, 416, 0x7FFF)  # max_true_peak_level: unset
    struct.pack_into("<h", buf, 418, -1990)  # max_momentary_loudness
    struct.pack_into("<h", buf, 420, -2110)  # max_short_term_loudness
    return bytes(buf)


def test_info_answers_a_giant_rf64_from_its_headers_alone(
    tmp_path: Path,
) -> None:
    """The description of an unreadable overnight RF64 costs only headers.

    The forged file claims three billion stereo frames (12 GB of data,
    about 17 hours) through a real ``ds64``, carries a fully populated v2
    ``bext``, and then simply ends: not one sample byte exists. ``info``
    must still describe everything -- if it touched the data even once it
    would fail exactly as ``read`` does below -- which is the property
    that lets a survey tool catalogue a disk of overnight recordings
    without decoding a second of audio.
    """
    frames = 3_000_000_000
    data_bytes = frames * 4  # 16-bit stereo: block_align = 4
    riff_size = 12 + (8 + 28) + (8 + 40) + (8 + 602) + 8 + data_bytes
    image = riff_wave(
        chunk(b"ds64", struct.pack("<QQQI", riff_size, data_bytes, frames, 0)),
        chunk(b"fmt ", fmt_payload(channels=2, bits=16)),
        chunk(b"bext", _forged_full_bext()),
        b"data" + struct.pack("<I", 0xFFFFFFFF),  # header only, no payload
        fourcc=b"RF64",
    )
    path = _write(tmp_path, image, "overnight.wav")

    described = info(path)
    assert described.container == "RF64"
    assert described.format_name == "PCM"
    assert described.fs == FS
    assert described.channels == 2
    assert described.frames == frames
    assert described.duration == pytest.approx(frames / FS)
    assert described.bit_depth == 16
    assert not described.lossy

    bext = described.bext
    assert bext is not None
    assert bext.description == "Facade SW-2"
    assert bext.originator == "XL2 rig"
    assert bext.originator_reference == "serial-042"
    assert bext.origination_date == "2026-08-17"
    assert bext.origination_time == "23:59:58"
    assert bext.time_reference == 5_000_000_000
    assert bext.version == 2
    assert bext.loudness_value == pytest.approx(-23.10)
    assert bext.loudness_range == pytest.approx(5.20)
    assert bext.max_true_peak_level is None  # the 0x7FFF sentinel
    assert bext.max_momentary_loudness == pytest.approx(-19.90)
    assert bext.max_short_term_loudness == pytest.approx(-21.10)

    # The proof that no sample was touched: touching them is impossible.
    with pytest.raises(ValueError):
        read(path)


# ---------------------------------------------------------------------------
# Dispatch: magic bytes, missing extra, unknown files
# ---------------------------------------------------------------------------

def test_unrecognised_bytes_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "noise.bin"
    path.write_bytes(b"\x00\x01\x02\x03 not audio at all")
    with pytest.raises(ValueError, match="not a recognised audio file"):
        read(path)
    with pytest.raises(ValueError, match="not a recognised audio file"):
        info(path)


def test_missing_audio_extra_is_named_in_the_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without soundfile, non-WAV formats must name the extra that reads them.

    The import is blocked rather than relying on the environment, so the
    message is pinned whether or not the extra happens to be installed.
    """
    real_import = builtins.__import__

    def blocked(name: str, *args: object, **kwargs: object) -> object:
        if name == "soundfile" or name.startswith("soundfile."):
            raise ImportError("No module named 'soundfile'")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", blocked)

    mp3 = tmp_path / "note.mp3"
    mp3.write_bytes(b"ID3\x04\x00\x00\x00\x00\x00\x00" + bytes(64))
    with pytest.raises(ImportError, match=r"pip install phonometry\[audio\]"):
        read(mp3)
    with pytest.raises(ImportError, match="MPEG audio"):
        info(mp3)

    # A compressed WAV is recognised on the base install (the walker reads
    # its headers) but its samples need the extra too.
    alaw = riff_wave(
        chunk(b"fmt ", fmt_payload(tag=0x0006, bits=8)),
        chunk(b"fact", struct.pack("<I", 4)),
        chunk(b"data", bytes(4)),
    )
    with pytest.raises(ImportError, match=r"phonometry\[audio\]"):
        read(_write(tmp_path, alaw, "compressed.wav"))
    # info() on the same file needs no extra at all: headers are base work.
    described = info(_write(tmp_path, alaw, "compressed2.wav"))
    assert described.lossy
    assert described.format_name == "A-law"
    assert described.frames == 4


# ---------------------------------------------------------------------------
# The [audio] lane: lossy warning, FLAC, magic-over-extension
# ---------------------------------------------------------------------------

@needs_soundfile
def test_mp3_read_warns_and_stamps_lossy(tmp_path: Path) -> None:
    x = 0.4 * np.sin(2 * np.pi * 1000 * np.arange(FS // 2) / FS)
    path = tmp_path / "note.mp3"
    sf.write(str(path), x, FS)
    with pytest.warns(LossyCompressionWarning, match="not metrologically defensible"):
        sig = read(path)
    assert sig.fs == FS
    assert sig.source is not None
    assert sig.source.lossy
    assert info(path).lossy


@needs_soundfile
def test_flac_reads_without_lossy_warning(tmp_path: Path) -> None:
    codes = np.array([16384, -32768, 0, 32767], dtype=np.int64)
    path = tmp_path / "archive.flac"
    sf.write(str(path), codes / 32768, FS, subtype="PCM_16")
    with warnings.catch_warnings():
        warnings.simplefilter("error", LossyCompressionWarning)
        sig = read(path)
    # FLAC is lossless compression of PCM: bit-exact codes, same scaling.
    assert np.asarray(sig).tolist() == (codes / 32768).tolist()
    assert sig.source is not None
    assert not sig.source.lossy
    assert sig.source.container == "FLAC"
    assert info(path).bit_depth == 16


@needs_soundfile
def test_compressed_wav_keeps_the_walker_metadata(tmp_path: Path) -> None:
    """An A-law WAV decodes via soundfile but keeps base-walker bext."""
    x = 0.25 * np.sin(2 * np.pi * 1000 * np.arange(2400) / FS)
    path = tmp_path / "listening.wav"
    sf.write(str(path), x, FS, subtype="ALAW")
    # Append a bext chunk after data (a legal position the walker covers).
    raw = bytearray(path.read_bytes())
    if len(raw) % 2:
        raw += b"\x00"
    raw += chunk(b"bext", _minimal_bext(b"logger"))
    struct.pack_into("<I", raw, 4, len(raw) - 8)
    path.write_bytes(bytes(raw))

    with pytest.warns(LossyCompressionWarning, match="ALAW"):
        sig = read(path)
    assert sig.provenance is not None
    assert sig.provenance.originator == "logger"
    assert sig.source is not None
    assert sig.source.lossy
    assert sig.source.container == "WAV"


@needs_soundfile
def test_magic_bytes_beat_the_file_extension(tmp_path: Path) -> None:
    """A FLAC named .wav dispatches as FLAC: content decides, not the name."""
    path = tmp_path / "mislabelled.wav"
    sf.write(str(path), np.array([0.5, -0.5]), FS, format="FLAC", subtype="PCM_16")
    sig = read(path)
    assert sig.source is not None
    assert sig.source.container == "FLAC"
