#  Copyright (c) 2026. Jose Manuel Requena Plens
r"""Base-install WAV writing: scipy where it already serves, in-house where not.

:func:`scipy.io.wavfile.write` covers most of what a measurement export
needs -- PCM 16/32, IEEE float 32/64, and automatic RF64 promotion when the
file would pass the 4 GiB RIFF limit (verified in the scipy source: it
rebuilds the header as ``RF64`` + ``ds64`` when ``file size - 8`` exceeds
0xFFFFFFFF, exactly the EBU Tech 3306 layout the reader in this package
walks). What scipy cannot do is write 24-bit PCM (no numpy dtype has a
3-byte stride) or any metadata chunk; this module packs 24-bit in-house and
owns the header assembly for that path, so the base install writes every
linear depth a measurement chain consumes.

**Quantisation.** Float samples map to a ``B``-bit integer subtype as
``round(x * 2^(B-1))``, the exact inverse of the reader's ``n / 2^(B-1)``
scaling: an integer file read and rewritten at the same depth round-trips
bit for bit (``round(n / 2^(B-1) * 2^(B-1)) = n``, and the power of two
makes both directions exact in binary floating point). The one value the
inverse cannot represent is +1.0 itself, whose ideal code ``2^(B-1)`` sits
one above the top code ``2^(B-1) - 1``; it is saturated and counted as
clipped, because a sample written 1 LSB below where the convention says it
belongs is a fact the caller should hear about, not a rounding detail.

**Clipping is never silent.** Ideal codes outside the representable range
are saturated (never wrapped: a naive ``astype(np.int16)`` wraps modulo
2^16, turning +1.01 full scale into roughly -1.0) and reported through
:class:`ClippingWarning` with the count of affected samples and the signal
peak. Nothing is ever rescaled to avoid the clip: silent normalisation is
the python-acoustics default that destroyed absolute level on every export,
and absolute level is the quantity this library exists to preserve. The
float subtypes carry values beyond +/-1.0 losslessly (IEEE float has no
full-scale wall), so no warning applies there.

**Dither.** Optional and off by default, only for the 16-bit listening-copy
case: ``dither="tpdf"`` adds triangular-PDF noise of +/-1 LSB (the sum of
two independent uniform variates of +/-0.5 LSB) to the ideal code before
rounding. Lipshitz, Wannamaker & Vanderkooy ("Quantization and Dither: A
Theoretical Survey", J. Audio Eng. Soc. 40(5), 1992) show this is the
smallest dither that makes the first and second moments of the total
quantisation error independent of the signal, replacing correlated
distortion with a benign constant noise floor of variance
:math:`\Delta^2/4` (dither :math:`\Delta^2/6` plus quantiser
:math:`\Delta^2/12`), about 4.8 dB above the undithered quantiser alone.
That trade is right for audition and wrong for analysis -- dither *adds*
noise to data -- which is why it is opt-in, and refused entirely at 24 bits
or above, where the quantisation error (roughly -146 dBFS at 24 bits) is
already far below the noise floor of any microphone chain and the added
noise would buy nothing.

**24-bit packing.** A 24-bit PCM sample is three little-endian bytes.
NumPy cannot address a 3-byte stride, so each code is materialised as a
little-endian ``int32`` and the low three bytes are kept -- the exact
inverse of the read path, where scipy left-justifies the three bytes into
``int32`` (so ``raw24 / 2^23 == int32 / 2^31``). The container is written
as plain ``WAVE_FORMAT_PCM``: Microsoft's guidance prefers EXTENSIBLE
above 16 bits, but plain 24-bit PCM is what field recorders and libsndfile
emit in practice and what every reader (including scipy and this package)
accepts, so plainness buys interoperability and costs nothing that matters
here (a channel mask can be attached via ``Signal.channel_labels`` and the
sidecar instead).

Integer input arrays (``int16``/``int32``) are written as a bit-exact
pass-through of their codes -- the archival path, where the file must hold
exactly the integers that came from the converter -- and refuse a
conflicting ``subtype`` rather than resample the codes behind the caller's
back.
"""

from __future__ import annotations

import struct
import warnings
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Literal, overload

import numpy as np
from scipy.io import wavfile

from .._internal.warnings import PhonometryWarning
from ._bext import (
    coding_history_line,
    extend_coding_history,
    fresh_metadata,
    serialize_bext,
    with_measured_loudness,
)
from ._chunks import _RF64_SIZE_SENTINEL, BroadcastMetadata
from ._signal import Signal

if TYPE_CHECKING:
    from collections.abc import Iterable
    from typing import BinaryIO

    from numpy.typing import NDArray


class ClippingWarning(PhonometryWarning):
    """Warns that samples exceeded full scale and were saturated on write."""


#: subtype -> (fmt tag, bytes per sample). Tags from Microsoft ``mmreg.h``:
#: 0x0001 PCM, 0x0003 IEEE float. The subtype names follow libsndfile's
#: identifiers so the same vocabulary serves the base writer and the
#: ``[audio]`` extra.
_SUBTYPE_SPEC: dict[str, tuple[int, int]] = {
    "PCM_16": (0x0001, 2),
    "PCM_24": (0x0001, 3),
    "PCM_32": (0x0001, 4),
    "FLOAT": (0x0003, 4),
    "DOUBLE": (0x0003, 8),
}

#: Bytes per sample of PCM_16, read from ``_SUBTYPE_SPEC`` so the value keeps
#: living in one place: the per-sample width that separates the two
#: scipy-served integer depths (PCM_16 vs PCM_32) and selects the ``<i2``
#: dtype.
_PCM16_SAMPLE_BYTES = _SUBTYPE_SPEC["PCM_16"][1]

#: File suffixes the base writer owns (the WAV family; dispatch on write is
#: by declared target, unlike the read side's magic bytes -- a file being
#: created has no bytes to sniff).
_WAV_SUFFIXES = frozenset({".wav", ".wave", ".bwf"})

#: numpy integer dtypes accepted for bit-exact pass-through, and the
#: subtype their codes are.
_PASSTHROUGH_SUBTYPES = {np.dtype(np.int16): "PCM_16", np.dtype(np.int32): "PCM_32"}

#: Fourcc of the classic 32-bit WAV container: this file's declared size is
#: the uint32 at offset 4, unlike RF64/BW64 whose real size is the uint64
#: inside ``ds64`` at offset 20.
_RIFF_FOURCC = b"RIFF"


def _resolve_input(
    x: Signal | NDArray[np.generic] | list[float],
    fs: int | None,
) -> tuple[np.ndarray, int]:
    """Normalise the input to ``(channels, samples)`` plus a sample rate.

    A :class:`Signal` brings its own ``fs`` (a conflicting explicit one is
    an error, not a preference); a bare array follows the library's channel
    convention (1-D mono, 2-D ``(channels, samples)``) and must bring
    ``fs``. Integer arrays keep their dtype for the pass-through path;
    everything else becomes float64.
    """
    if isinstance(x, Signal):
        if fs is not None and int(fs) != x.fs:
            msg = (
                f"fs={fs} conflicts with the Signal's own fs={x.fs}; "
                "pass one or the other, not a disagreement"
            )
            raise ValueError(msg)
        return x.data, x.fs
    data = np.asarray(x)
    if fs is None:
        msg = "fs is required when writing a bare array"
        raise ValueError(msg)
    if int(fs) <= 0:
        msg = "fs must be positive"
        raise ValueError(msg)
    if data.dtype not in _PASSTHROUGH_SUBTYPES:
        data = np.asarray(data, dtype=np.float64)
    data = np.atleast_2d(data)
    if data.ndim != 2:  # noqa: PLR2004
        msg = f"x must be 1-D or (channels, samples); got {data.ndim}-D"
        raise ValueError(msg)
    return data, int(fs)


def _resolve_subtype(data: np.ndarray, subtype: str | None) -> str:
    """Pick or validate the subtype for the resolved input data.

    Float data defaults to ``FLOAT``: float32 represents every sample that
    came from a container of up to 24 bits *exactly* (a 24-bit code divided
    by 2^23 fits the 24-bit float32 mantissa with room to spare), halves
    the file against DOUBLE, and any tool reads it. Only float64-computed
    signals lose precision to it, and those callers ask for ``DOUBLE`` and
    get a bit-exact file. Integer input implies its own depth and refuses a
    conflicting request instead of resampling codes silently.
    """
    implied = _PASSTHROUGH_SUBTYPES.get(data.dtype)
    if implied is not None:
        if subtype is not None and subtype != implied:
            msg = (
                f"integer {data.dtype} input is a bit-exact pass-through of "
                f"{implied}; requesting subtype={subtype!r} would resample "
                "the codes. Convert to float64 first to choose a depth."
            )
            raise ValueError(msg)
        return implied
    if data.dtype.kind != "f":
        msg = (
            f"unsupported input dtype {data.dtype}: pass float data, or "
            "int16/int32 codes for bit-exact pass-through"
        )
        raise ValueError(msg)
    if subtype is None:
        return "FLOAT"
    if subtype not in _SUBTYPE_SPEC:
        msg = (
            f"unknown subtype {subtype!r}; the WAV writer offers "
            f"{sorted(_SUBTYPE_SPEC)}"
        )
        raise ValueError(msg)
    return subtype


def _quantise(
    data: NDArray[np.float64],
    bits: int,
    dither: str | None,
    rng: np.random.Generator | int | None = None,
) -> tuple[NDArray[np.int64], int, float]:
    """Map float samples to ``bits``-bit codes; count saturated samples.

    Returns the codes, the number of samples whose ideal code fell outside
    the representable range (saturated, never wrapped), and the signal peak
    in dBFS for the warning message. See the module docstring for the
    scaling, the +1.0 edge and the dither derivation. ``rng`` seeds the
    dither noise; ``None`` draws fresh entropy (see :func:`write` for why
    that is the right default).
    """
    scale = float(2 ** (bits - 1))
    ideal = data * scale
    if dither == "tpdf":
        generator = np.random.default_rng(rng)
        ideal = ideal + (generator.random(data.shape) - generator.random(data.shape))
    codes = np.rint(ideal)
    lo, hi = -scale, scale - 1.0
    clipped = int(np.count_nonzero((codes < lo) | (codes > hi)))
    peak = float(np.max(np.abs(data))) if data.size else 0.0
    peak_dbfs = 20.0 * np.log10(peak) if peak > 0 else float("-inf")
    return np.clip(codes, lo, hi).astype(np.int64), clipped, peak_dbfs


def _warn_clipped(
    path: str | Path, subtype: str, clipped: int, total: int, peak_dbfs: float
) -> None:
    """Report saturation loudly: count, total, and how far over the top."""
    warnings.warn(
        f"{path}: {clipped} of {total} samples exceed the full-scale range "
        f"of {subtype} (signal peak {peak_dbfs:+.2f} dBFS) and were "
        "saturated. Nothing was rescaled: scale the signal before writing "
        "if clipping is not intended.",
        ClippingWarning,
        stacklevel=4,
    )


def _pack_pcm24(codes_interleaved: NDArray[np.int64]) -> bytes:
    """Pack interleaved 24-bit codes as 3-byte little-endian samples.

    Each code becomes a little-endian ``int32`` whose low three bytes are
    the 24-bit sample (the value fits by construction: the quantiser
    saturates to +/-2^23); dropping every fourth byte of the int32 view is
    the vectorised form of "keep the low three".
    """
    as_int32 = codes_interleaved.astype("<i4")
    return np.ascontiguousarray(as_int32.view(np.uint8).reshape(-1, 4)[:, :3]).tobytes()


def build_wav_header(
    *,
    fs: int,
    channels: int,
    subtype: str,
    frames: int,
    metadata_chunks: bytes = b"",
) -> bytes:
    """Assemble a complete WAV/RF64 header for a known-length data payload.

    Everything up to and including the ``data`` chunk header, sized exactly
    (the payload length is ``frames`` times the block align, known before a
    single sample is written), so a writer can stream the samples straight
    after it with no seek-back patching. ``metadata_chunks`` (already
    RIFF-framed, e.g. a ``bext``) land between ``fmt`` and ``data``.

    The container promotes itself to RF64 under the same condition scipy's
    writer uses -- the RIFF payload (file size minus the 8-byte outer
    header) exceeding 0xFFFFFFFF -- with the EBU Tech 3306 layout: fourcc
    ``RF64``, outer size 0xFFFFFFFF, a ``ds64`` first after ``WAVE``
    carrying the real 64-bit sizes, and the ``data`` size field set to the
    0xFFFFFFFF sentinel. IEEE float files get the ``fact`` frame count and
    the ``cbSize=0`` tail on ``fmt`` that the 1991 RIFF specification
    requires of non-PCM formats (scipy writes both; readers exist that
    check).
    """
    tag, sample_bytes = _SUBTYPE_SPEC[subtype]
    block_align = channels * sample_bytes
    data_size = frames * block_align
    fmt_payload = struct.pack(
        "<HHIIHH",
        tag,
        channels,
        fs,
        fs * block_align,
        block_align,
        8 * sample_bytes,
    )
    if tag != 0x0001:
        fmt_payload += b"\x00\x00"  # cbSize = 0, mandatory for non-PCM fmt
    body = b"fmt " + struct.pack("<I", len(fmt_payload)) + fmt_payload
    if tag != 0x0001:
        body += b"fact" + struct.pack("<II", 4, frames)
    body += metadata_chunks
    # RIFF payload: the WAVE form type, every chunk, the data payload and
    # its pad byte; the outer 8-byte (fourcc + size) header is not counted.
    riff_payload = 4 + len(body) + 8 + data_size + data_size % 2
    if riff_payload <= _RF64_SIZE_SENTINEL:
        return (
            b"RIFF"
            + struct.pack("<I", riff_payload)
            + b"WAVE"
            + body
            + b"data"
            + struct.pack("<I", data_size)
        )
    ds64 = b"ds64" + struct.pack("<IQQQI", 28, riff_payload + 36, data_size, frames, 0)
    return (
        b"RF64"
        + struct.pack("<I", _RF64_SIZE_SENTINEL)
        + b"WAVE"
        + ds64
        + body
        + b"data"
        + struct.pack("<I", _RF64_SIZE_SENTINEL)
    )


def _encode_frames(data: np.ndarray, subtype: str) -> bytes:
    """Encode ``(channels, samples)`` data as interleaved sample bytes.

    Integer input must already be quantised codes; the transpose
    interleaves frames the way the container stores them.
    """
    interleaved = np.ascontiguousarray(data.T).reshape(-1)
    if subtype == "PCM_16":
        return interleaved.astype("<i2").tobytes()
    if subtype == "PCM_24":
        return _pack_pcm24(interleaved)
    if subtype == "PCM_32":
        return interleaved.astype("<i4").tobytes()
    if subtype == "FLOAT":
        return interleaved.astype("<f4").tobytes()
    return interleaved.astype("<f8").tobytes()


def frame_riff_chunk(chunk_id: bytes, payload: bytes) -> bytes:
    """Frame a payload as one RIFF chunk: header, payload, alignment pad.

    The pad byte of an odd payload is written but not counted in the size
    field, per the 1991 RIFF specification's word-alignment rule.
    """
    padding = b"\x00" if len(payload) % 2 else b""
    return chunk_id + struct.pack("<I", len(payload)) + payload + padding


def append_riff_chunk(path: str | Path, chunk_id: bytes, payload: bytes) -> None:
    """Append a chunk to a finished WAV/RF64 file and repair its size field.

    The path for files scipy wrote (its writer offers no metadata): the
    chunk lands after ``data`` -- a legal position, and the one the
    package's own walker and every RIFF reader that honours chunk headers
    handle -- the file is first padded to even length if the data payload
    left it odd (scipy writes no alignment pad), and the container's
    declared size is patched: the 32-bit RIFF size at offset 4, or for
    RF64/BW64 the 64-bit ``ds64`` riff size at offset 20 (the fixed
    position both scipy's writer and EBU Tech 3306 put it, ``ds64``
    mandatory first after ``WAVE``).
    """
    with Path(path).open("r+b") as fh:
        fourcc = fh.read(4)
        if fourcc not in (b"RIFF", b"RF64", b"BW64"):
            msg = f"{path}: not a RIFF/RF64/BW64 file"
            raise ValueError(msg)
        fh.seek(0, 2)
        if fh.tell() % 2:
            fh.write(b"\x00")
        fh.write(frame_riff_chunk(chunk_id, payload))
        riff_size = fh.tell() - 8
        if fourcc == _RIFF_FOURCC:
            if riff_size > _RF64_SIZE_SENTINEL:
                msg = (
                    f"{path}: appending {len(payload)} bytes overflows the "
                    "32-bit RIFF size; the file needed to be RF64"
                )
                raise ValueError(msg)
            fh.seek(4)
            fh.write(struct.pack("<I", riff_size))
        else:
            fh.seek(20)
            fh.write(struct.pack("<Q", riff_size))


#: subtype -> word length written to disk, for the CodingHistory ``W=``.
_SUBTYPE_BITS = {
    "PCM_16": 16,
    "PCM_24": 24,
    "PCM_32": 32,
    "FLOAT": 32,
    "DOUBLE": 64,
}


def _check_dither(dither: str | None, subtype: str) -> None:
    """Validate the dither request against the target subtype."""
    if dither is None:
        return
    if dither != "tpdf":
        msg = f"unknown dither {dither!r}; offered: 'tpdf'"
        raise ValueError(msg)
    if subtype != "PCM_16":
        msg = (
            "dither='tpdf' applies only when quantising to PCM_16: at "
            "24 bits and above the quantisation error is already far "
            "below any microphone chain's noise floor, and dithering "
            "float data only adds noise to it"
        )
        raise ValueError(msg)


def _resolve_bext(
    x: Signal | NDArray[np.generic] | list[float],
    bext: BroadcastMetadata | str | None,
    data: np.ndarray,
    fs: int,
    subtype: str,
    algorithm: str = "PCM",
) -> bytes | None:
    """Turn the ``bext`` argument into a serialised chunk payload, or None.

    ``None`` carries a :class:`Signal`'s provenance when it has one (a
    measurement's chain of custody should survive an export by default)
    and writes no chunk otherwise; ``"loudness"`` additionally measures
    the five R 128 fields with the library's BS.1770 implementation on
    the very samples being written; a :class:`BroadcastMetadata` is
    written as given. Whatever the source, the CodingHistory leaves with
    phonometry's line appended below the existing audit trail (EBU R98:
    extended, never replaced).
    """
    provenance = x.provenance if isinstance(x, Signal) else None
    meta: BroadcastMetadata | None
    if bext is None:
        meta = provenance
    elif isinstance(bext, BroadcastMetadata):
        meta = bext
    elif bext == "loudness":
        base = provenance if provenance is not None else fresh_metadata()
        if data.dtype.kind == "f":
            full_scale = np.asarray(data, dtype=np.float64)
        else:
            full_scale = data / float(2 ** (8 * data.dtype.itemsize - 1))
        meta = with_measured_loudness(base, full_scale, fs)
    else:
        msg = f"unknown bext {bext!r}: pass None, 'loudness' or a BroadcastMetadata"
        raise ValueError(msg)
    if meta is None:
        return None
    line = coding_history_line(
        fs=fs,
        bits=_SUBTYPE_BITS[subtype],
        channels=int(data.shape[0]),
        algorithm=algorithm,
    )
    meta = replace(
        meta, coding_history=extend_coding_history(meta.coding_history, line)
    )
    return serialize_bext(meta)


def _reconcile_streamed_header(
    fh: BinaryIO,
    path: str | Path,
    *,
    fs: int,
    channels: int,
    subtype: str,
    written: int,
    header_len: int,
    metadata_chunks: bytes,
) -> None:
    """Rewrite the header in place for the frame count actually streamed.

    Only for callers whose up-front count was an estimate: a lossy
    source's decoder can deliver a few frames more or fewer than its
    container declares (decoder delay and padding), and the sizes on
    disk must describe the samples that landed, not the declaration.
    The rebuilt header must occupy exactly the bytes of the original;
    the layout only changes across the RF64 promotion boundary, where a
    discrepancy this large is no codec quirk and refusing is honest.
    """
    patched = build_wav_header(
        fs=fs,
        channels=channels,
        subtype=subtype,
        frames=written,
        metadata_chunks=metadata_chunks,
    )
    if len(patched) != header_len:
        msg = (
            f"{path}: the {written} frames actually streamed move the "
            "header across the RF64 promotion boundary the estimated "
            "count chose; the estimate was too wrong to reconcile"
        )
        raise ValueError(msg)
    fh.seek(0)
    fh.write(patched)


def write_wav_stream(
    path: str | Path,
    blocks: Iterable[np.ndarray],
    fs: int,
    channels: int,
    subtype: str,
    frames: int,
    *,
    metadata_chunks: bytes = b"",
    frames_are_estimate: bool = False,
) -> None:
    """Stream ``(channels, n)`` blocks (codes or floats) into one WAV file.

    The in-house writer: because ``frames`` is known up front, the header
    (RF64 automatic past 4 GiB) is exact before the first sample lands
    and nothing is ever seeked back and patched -- each block is encoded,
    interleaved and appended, so an hour of RF64 flows through a
    block-sized memory window. The blocks carry quantised codes for the
    PCM subtypes and float samples for the float ones; no scaling happens
    here.

    :param frames_are_estimate: The caller's ``frames`` is a container
        declaration it cannot guarantee (a lossy source's decoder may
        deliver a slightly different count than its header claims). When
        the streamed count differs, the header is then rewritten in
        place at close so the sizes on disk describe what was actually
        written, instead of the mismatch being an error.
    :raises ValueError: If the blocks do not add up to exactly ``frames``
        frames and ``frames`` was not an estimate -- the header already
        promised that length, so a mismatch is a corrupt file, not a
        rounding detail.
    """
    header = build_wav_header(
        fs=fs,
        channels=channels,
        subtype=subtype,
        frames=frames,
        metadata_chunks=metadata_chunks,
    )
    written = 0
    odd = False
    with Path(path).open("wb") as fh:
        fh.write(header)
        for block in blocks:
            payload = _encode_frames(block, subtype)
            fh.write(payload)
            written += int(block.shape[-1])
            odd ^= bool(len(payload) % 2)
        if odd:
            fh.write(b"\x00")  # RIFF word alignment; not counted in the size
        if written != frames and frames_are_estimate:
            _reconcile_streamed_header(
                fh,
                path,
                fs=fs,
                channels=channels,
                subtype=subtype,
                written=written,
                header_len=len(header),
                metadata_chunks=metadata_chunks,
            )
            frames = written
    if written != frames:
        msg = (
            f"{path}: the block stream carried {written} frames but the "
            f"header promised {frames}; the file is inconsistent"
        )
        raise ValueError(msg)


def write_wav(
    path: str | Path,
    data: np.ndarray,
    fs: int,
    subtype: str,
    *,
    metadata_chunks: bytes = b"",
) -> None:
    """Write ``(channels, samples)`` data as one WAV file, in one block."""
    write_wav_stream(
        path,
        [data],
        fs,
        int(data.shape[0]),
        subtype,
        int(data.shape[-1]),
        metadata_chunks=metadata_chunks,
    )


def _check_target_suffix(path: str | Path, target: Path) -> None:
    """Refuse any destination outside the WAV family and FLAC."""
    suffix = target.suffix.lower()
    if suffix not in _WAV_SUFFIXES and suffix != ".flac":
        msg = (
            f"{path}: unsupported target {target.suffix!r}; the writer "
            "produces the WAV family (.wav, .wave, .bwf) and, with the "
            "[audio] extra, FLAC. Lossy formats are deliberately not "
            "offered: a metrology library does not export approximations "
            "of its own data."
        )
        raise ValueError(msg)


def _check_sidecar_request(
    x: Signal | NDArray[np.generic] | list[float], sidecar: bool
) -> None:
    """Refuse a sidecar request without a calibrated Signal behind it."""
    if sidecar and (not isinstance(x, Signal) or x.calibration_factor is None):
        msg = (
            "sidecar=True needs a Signal with a calibration_factor: the "
            "sidecar exists to carry a calibration, and inventing one "
            "is the single thing this library must never do"
        )
        raise ValueError(msg)


def _refuse_passthrough_dither(dither: str | None) -> None:
    """Integer input has nothing to dither; say so instead of ignoring."""
    if dither is not None:
        msg = (
            "dither applies to quantisation, and integer input is a "
            "bit-exact pass-through: there is nothing to dither"
        )
        raise ValueError(msg)


def _write_pcm24(
    path: str | Path,
    data: np.ndarray,
    rate: int,
    resolved: str,
    dither: str | None,
    rng: np.random.Generator | int | None,
    bext_payload: bytes | None,
) -> None:
    """Quantise to 24-bit codes and write through the in-house packer."""
    codes, clipped, peak_dbfs = _quantise(data, 24, dither, rng)
    if clipped:
        _warn_clipped(path, resolved, clipped, data.size, peak_dbfs)
    write_wav(
        path,
        codes,
        rate,
        resolved,
        metadata_chunks=(
            frame_riff_chunk(b"bext", bext_payload) if bext_payload is not None else b""
        ),
    )


def _write_through_scipy(
    path: str | Path,
    data: np.ndarray,
    rate: int,
    resolved: str,
    dither: str | None,
    rng: np.random.Generator | int | None,
) -> None:
    """Quantise (PCM) or narrow (float) and hand scipy finished samples."""
    tag, sample_bytes = _SUBTYPE_SPEC[resolved]
    if tag == 0x0001:
        codes, clipped, peak_dbfs = _quantise(data, 8 * sample_bytes, dither, rng)
        if clipped:
            _warn_clipped(path, resolved, clipped, data.size, peak_dbfs)
        _scipy_write(
            path,
            rate,
            codes.astype("<i2" if sample_bytes == _PCM16_SAMPLE_BYTES else "<i4"),
        )
    else:
        _scipy_write(
            path, rate, data if resolved == "DOUBLE" else data.astype(np.float32)
        )


@overload
def write(
    path: str | Path,
    x: Signal | NDArray[np.generic] | list[float],
    fs: int | None = ...,
    *,
    subtype: str | None = ...,
    bext: BroadcastMetadata | str | None = ...,
    dither: Literal["tpdf"],
    rng: np.random.Generator | int | None = ...,
    sidecar: bool = ...,
) -> None: ...


@overload
def write(
    path: str | Path,
    x: Signal | NDArray[np.generic] | list[float],
    fs: int | None = ...,
    *,
    subtype: str | None = ...,
    bext: BroadcastMetadata | str | None = ...,
    sidecar: bool = ...,
) -> None: ...


def write(
    path: str | Path,
    x: Signal | NDArray[np.generic] | list[float],
    fs: int | None = None,
    *,
    subtype: str | None = None,
    bext: BroadcastMetadata | str | None = None,
    dither: str | None = None,
    rng: np.random.Generator | int | None = None,
    sidecar: bool = False,
) -> None:
    """Write a signal to an audio file, without ever touching its level.

    The counterpart of :func:`phonometry.io.read` and the export path for
    everything the library generates (IEC 60268-1 test tones, sweeps, MLS,
    processed measurements). The WAV subtypes are ``PCM_16``, ``PCM_24``
    (packed in-house; scipy cannot), ``PCM_32``, ``FLOAT`` (float32, the
    default: exact for anything that came from a container of up to
    24 bits) and ``DOUBLE`` (float64, bit-exact for computed signals).
    Files past the 4 GiB RIFF limit promote to RF64 automatically. A
    ``.flac`` destination writes the lossless archive format through the
    ``[audio]`` extra (integer subtypes up to ``PCM_24``, the default
    there; see :func:`_write_flac`).

    What this function will never do, each a documented ecosystem default
    that destroys a measurement: normalise (python-acoustics ``to_wav``),
    resample, or mix channels. Samples that exceed full scale on an
    integer subtype are saturated and *reported* -- see
    :class:`ClippingWarning` and the module docstring for the exact
    quantisation math, the +1.0 edge, and why the scaling choice cancels
    out of calibrated results.

    :param path: Destination file; the suffix picks the container: the
        WAV family (``.wav``, ``.wave``, ``.bwf``) on the base install,
        ``.flac`` with the ``[audio]`` extra. Lossy suffixes are refused
        by decided policy.
    :param x: A :class:`Signal`, or an array in the library's channel
        convention (1-D mono, 2-D ``(channels, samples)``). Float data is
        quantised per ``subtype``; ``int16``/``int32`` arrays are written
        as a bit-exact pass-through of their codes.
    :param fs: Sample rate, Hz. Required for bare arrays; forbidden to
        disagree with a :class:`Signal`'s own rate.
    :param subtype: Target sample format (see above); ``None`` picks
        ``FLOAT`` for float data and the matching depth for integer data.
    :param bext: The broadcast provenance chunk (EBU Tech 3285; see
        :mod:`phonometry.io._bext`). ``None`` carries the
        :class:`Signal`'s own provenance when it has one and writes no
        chunk otherwise; ``"loudness"`` additionally measures the five
        R 128 fields with the library's BS.1770 implementation on the
        samples being written (one extra pass, which is why it is
        opt-in); a :class:`BroadcastMetadata` is written as given.
        Every written chunk's CodingHistory is extended -- never
        replaced -- with phonometry's ``A=,F=,W=,M=,T=`` line.
    :param dither: ``"tpdf"`` adds +/-1 LSB triangular-PDF dither before
        quantising to ``PCM_16`` (Lipshitz et al. 1992; see the module
        docstring); ``None`` (default) quantises plainly. Refused for any
        other subtype, where it would only add noise.
    :param rng: Randomness for the dither noise: a seeded
        :class:`numpy.random.Generator` (or an int seed) makes the
        written bytes reproducible, which is what a test or a pinned
        pipeline needs. ``None`` (default) draws fresh entropy on every
        write, and that default is deliberate: dither exists to
        decorrelate the quantisation error, and repeating one noise
        pattern across multiple writes would correlate the files with
        each other, undoing exactly what the dither is for. Refused
        without ``dither``, the only thing it seeds.
    :param sidecar: When true, also write the calibration sidecar
        (:mod:`phonometry.io._sidecar`) beside the file, carrying the
        :class:`Signal`'s ``calibration_factor`` and channel labels so
        :func:`phonometry.io.read` recovers the absolute level with no
        argument at all. Requires a calibrated :class:`Signal`: a sidecar
        without a calibration would be a promise with nothing behind it.
    :raises ValueError: For an unknown suffix or subtype, a missing or
        conflicting ``fs``, a dither request outside ``PCM_16``, an
        ``rng`` without a ``dither`` for it to seed, bext
        metadata that violates Tech 3285 (oversize field, version too old
        for a carried UMID or loudness), or a sidecar request without a
        calibrated :class:`Signal`.
    """
    target = Path(path)
    _check_target_suffix(path, target)
    _check_sidecar_request(x, sidecar)
    if rng is not None and dither is None:
        msg = (
            "rng seeds the TPDF dither noise and does nothing without it; "
            "pass dither='tpdf' alongside it or drop it"
        )
        raise ValueError(msg)
    data, rate = _resolve_input(x, fs)
    if target.suffix.lower() == ".flac":
        _write_flac(path, x, data, rate, subtype, bext, dither, rng)
        _write_signal_sidecar(path, x, sidecar)
        return
    resolved = _resolve_subtype(data, subtype)
    _check_dither(dither, resolved)
    bext_payload = _resolve_bext(x, bext, data, rate, resolved)
    if data.dtype in _PASSTHROUGH_SUBTYPES:
        _refuse_passthrough_dither(dither)
        _scipy_write(path, rate, data)
    elif resolved == "PCM_24":
        _write_pcm24(path, data, rate, resolved, dither, rng, bext_payload)
        _write_signal_sidecar(path, x, sidecar)
        return
    else:
        _write_through_scipy(path, data, rate, resolved, dither, rng)
    if bext_payload is not None:
        append_riff_chunk(path, b"bext", bext_payload)
    _write_signal_sidecar(path, x, sidecar)


def _write_signal_sidecar(
    path: str | Path,
    x: Signal | NDArray[np.generic] | list[float],
    sidecar: bool,
) -> None:
    """Write the calibration sidecar for a just-written Signal, on request.

    The precondition (a :class:`Signal` with a calibration) was validated
    before any bytes hit the disk, so this step cannot fail after the
    audio file already exists half-committed to the pair.
    """
    if not sidecar or not isinstance(x, Signal) or x.calibration_factor is None:
        return
    from ._sidecar import write_sidecar

    write_sidecar(
        path,
        x.calibration_factor,
        channel_labels=x.channel_labels,
    )


def _scipy_write(path: str | Path, fs: int, data: np.ndarray) -> None:
    """Write ``(channels, samples)`` data through scipy's writer.

    scipy fully serves four of the five subtypes (int16/int32 PCM, float
    32/64) including the automatic RF64 promotion past 4 GiB, so those
    paths lean on it; only the transpose to its ``(Nsamples, Nchannels)``
    interleaving and the 1-D mono convention are handled here.
    """
    frames_first = np.ascontiguousarray(data.T)
    wavfile.write(
        str(path), fs, frames_first[:, 0] if data.shape[0] == 1 else frames_first
    )


def _flac_passthrough_codes(
    data: np.ndarray, subtype: str | None, dither: str | None
) -> np.ndarray:
    """int16 codes pass through whole; a conflicting request is refused."""
    if subtype not in (None, "PCM_16"):
        msg = (
            "integer int16 input is a bit-exact pass-through of PCM_16; "
            f"requesting subtype={subtype!r} would resample the codes"
        )
        raise ValueError(msg)
    _refuse_passthrough_dither(dither)
    return data.astype(np.int16)


def _flac_quantised_codes(
    path: str | Path,
    data: np.ndarray,
    subtype: str | None,
    dither: str | None,
    rng: np.random.Generator | int | None,
) -> tuple[str, np.ndarray]:
    """Quantise float data to the integer codes a FLAC subtype stores."""
    if data.dtype.kind != "f":
        msg = (
            f"unsupported input dtype {data.dtype}: pass float data, "
            "or int16 codes for bit-exact pass-through"
        )
        raise ValueError(msg)
    resolved = "PCM_24" if subtype is None else subtype
    if resolved not in ("PCM_16", "PCM_24"):
        msg = (
            f"FLAC holds integer PCM up to 24 bits; subtype "
            f"{resolved!r} is not available (choose PCM_16 or PCM_24)"
        )
        raise ValueError(msg)
    _check_dither(dither, resolved)
    bits = _SUBTYPE_BITS[resolved]
    codes, clipped, peak_dbfs = _quantise(data, bits, dither, rng)
    if clipped:
        _warn_clipped(path, resolved, clipped, data.size, peak_dbfs)
    out = (
        codes.astype(np.int16)
        if resolved == "PCM_16"
        else np.left_shift(codes.astype(np.int32), 8)
    )
    return resolved, out


def _write_flac(
    path: str | Path,
    x: Signal | NDArray[np.generic] | list[float],
    data: np.ndarray,
    fs: int,
    subtype: str | None,
    bext: BroadcastMetadata | str | None,
    dither: str | None,
    rng: np.random.Generator | int | None,
) -> None:
    """Write a FLAC archive copy through the ``[audio]`` extra.

    FLAC stores integers only, up to 24 bits in libsndfile, so float data
    defaults to ``PCM_24`` (whose quantisation noise, roughly -146 dBFS,
    sits far below any microphone chain's floor) and the quantisation is
    done *here*, by the same documented rounding as the WAV paths, with
    libsndfile handed finished codes: ``int16`` directly, and 24-bit
    codes left-justified in ``int32``, which libsndfile's integer write
    path stores bit-exactly (its own float-to-int scaling never runs, so
    its conventions cannot disagree with ours). A ``bext`` rides in the
    FLAC APPLICATION ``riff`` block afterwards (:mod:`._flac`), with the
    CodingHistory line's ``A=FLAC`` naming the coding step.

    ``int16`` input passes its codes through bit-exact; ``int32`` input
    is refused rather than silently dropped to 24 bits.
    """
    from ._backends import _import_soundfile
    from ._flac import embed_flac_bext

    sf = _import_soundfile("FLAC")
    implied = _PASSTHROUGH_SUBTYPES.get(data.dtype)
    out: np.ndarray
    if implied == "PCM_32":
        msg = (
            f"{path}: FLAC stores integers up to 24 bits (libsndfile), so "
            "int32 codes cannot pass through bit-exact. Convert to float64 "
            "and choose subtype='PCM_24' to accept the 8-bit truncation "
            "explicitly."
        )
        raise ValueError(msg)
    if implied == "PCM_16":
        resolved = "PCM_16"
        out = _flac_passthrough_codes(data, subtype, dither)
    else:
        resolved, out = _flac_quantised_codes(path, data, subtype, dither, rng)
    bext_payload = _resolve_bext(x, bext, data, fs, resolved, algorithm="FLAC")
    frames_first = np.ascontiguousarray(out.T)
    sf.write(
        str(path),
        frames_first[:, 0] if out.shape[0] == 1 else frames_first,
        fs,
        subtype=resolved,
        format="FLAC",
    )
    if bext_payload is not None:
        embed_flac_bext(path, bext_payload)
