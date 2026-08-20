#  Copyright (c) 2026. Jose Manuel Requena Plens
r"""
Block streaming: feed hours of recording through a constant memory window.

A night of environmental monitoring at 48 kHz/24-bit/2 channels grows past
4 GiB on disk in under four hours and costs about 5.5 GiB of RAM *per
hour* once decoded to float64: whole-file :func:`phonometry.io.read` is
the wrong tool for it. :func:`read_blocks` yields the same samples as
``read`` -- identical scaling, identical channel order, sample for
sample -- in ``(channels, block)`` pieces sized by the caller, which is
exactly the shape the library's stateful block machinery consumes
(``BlockProcessing(stateful=True)`` on the filter banks and weightings;
see the block-processing guide): the loop over a memory array becomes a
loop over this generator and nothing else changes.

For the linear WAV family the decoder here is the module's own: the chunk
walker locates the data payload (through ``ds64`` for RF64, so a long
recording streams the same as a short one), each block is one seek and
one bounded read, and the bytes decode by the reader's documented
convention (divide by :math:`2^{B-1}`, unsigned 8-bit re-centred, 24-bit
assembled from its three little-endian bytes and sign-extended). No
memory map is involved: a map of a 100 GiB RF64 costs address space and
page-cache churn, while a seek-and-read of ``block x block_align`` bytes
costs precisely that many bytes, 24-bit included -- the depth no numpy
dtype can map. Everything else (compressed WAV, FLAC, Ogg, MP3) streams
through ``soundfile.blocks`` from the ``[audio]`` extra, with the same
lossy warning :func:`phonometry.io.read` gives: chopping an approximation
into blocks does not make its levels defensible.

**Overlap.** ``overlap`` frames of each block reappear at the start of
the next (windowed analyses want it; ``0`` for plain streaming). The
iteration rule is libsndfile's, matched exactly so both backends yield
identical block trains: blocks start every ``block_size - overlap``
frames, the final block may be short, and iteration ends when the frames
left over are only ones already delivered (``remaining <= overlap``).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ._backends import _import_soundfile, _sniff, _soundfile_lossy, _warn_lossy
from ._chunks import (
    WAVE_FORMAT_IEEE_FLOAT,
    WAVE_FORMAT_PCM,
    FormatChunk,
    parse_wav_chunks,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from numpy.typing import NDArray


def _decode_linear_frames(raw: bytes, fmt: FormatChunk) -> NDArray[np.float64]:
    """Decode interleaved linear PCM/float bytes to ``(channels, frames)``.

    The scaling convention of :mod:`phonometry.io._wav`, applied straight
    to the wire bytes: unsigned 8-bit re-centred on 128, signed integers
    divided by ``2**(bits - 1)``, floats recast. 24-bit samples are
    assembled from their three little-endian bytes and sign-extended via
    the two's-complement identity ``(v XOR 0x800000) - 0x800000``.
    """
    bits = fmt.bits_per_sample
    if fmt.resolved_tag == WAVE_FORMAT_IEEE_FLOAT:
        data = np.frombuffer(raw, dtype="<f4" if bits == 32 else "<f8")
        scaled = data.astype(np.float64)
    elif bits == 8:
        scaled = (np.frombuffer(raw, dtype=np.uint8).astype(np.float64) - 128.0) / 128.0
    elif bits == 24:
        triplets = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        values = (
            triplets[:, 0].astype(np.int32)
            | triplets[:, 1].astype(np.int32) << 8
            | triplets[:, 2].astype(np.int32) << 16
        )
        scaled = ((values ^ 0x800000) - 0x800000) / float(2**23)
    else:
        data = np.frombuffer(raw, dtype="<i2" if bits == 16 else "<i4")
        scaled = data.astype(np.float64) / float(2 ** (bits - 1))
    frames = scaled.reshape(-1, fmt.channels)
    return np.ascontiguousarray(frames.T)


def _squeeze(block: NDArray[np.float64]) -> NDArray[np.float64]:
    """Mono blocks go out 1-D, matching read()/Signal's array view."""
    return block[0] if block.shape[0] == 1 else block


def _iter_wav_blocks(
    path: str | Path, block_size: int, overlap: int
) -> Iterator[NDArray[np.float64]]:
    """The base-install block reader for linear WAV/BWF/RF64/BW64."""
    chunks = parse_wav_chunks(path)
    fmt = chunks.fmt
    total = chunks.frames
    step = block_size - overlap
    with Path(path).open("rb") as fh:
        start = 0
        while total - start > overlap or start == 0 and total > 0:
            n = min(block_size, total - start)
            fh.seek(chunks.data_offset + start * fmt.block_align)
            raw = fh.read(n * fmt.block_align)
            if len(raw) < n * fmt.block_align:
                raise ValueError(
                    f"{path}: data chunk ends {n * fmt.block_align - len(raw)} "
                    f"bytes short of frame {start + n}"
                )
            yield _squeeze(_decode_linear_frames(raw, fmt))
            start += step


def _iter_soundfile_blocks(
    path: str | Path, format_name: str, block_size: int, overlap: int
) -> Iterator[NDArray[np.float64]]:
    """The ``[audio]`` block reader: everything libsndfile can stream."""
    sf = _import_soundfile(format_name)
    subtype = str(sf.info(str(path)).subtype)
    if _soundfile_lossy(subtype):
        _warn_lossy(f"{format_name} ({subtype})", path)
    for block in sf.blocks(
        str(path),
        blocksize=block_size,
        overlap=overlap,
        dtype="float64",
        always_2d=True,
    ):
        yield _squeeze(np.ascontiguousarray(block.T))


def read_blocks(
    path: str | Path, block_size: int, *, overlap: int = 0
) -> Iterator[NDArray[np.float64]]:
    """Stream an audio file as float64 blocks of ``block_size`` frames.

    Yields what :func:`phonometry.io.read` would return for the same
    file, cut into consecutive ``(channels, block_size)`` pieces (1-D for
    mono, like the ``Signal`` array view; the last piece may be shorter),
    while holding only one block in memory -- the way an overnight RF64
    flows through ``BlockProcessing(stateful=True)`` filters without ever
    existing as an array. See the module docstring for the per-backend
    mechanics and the exact overlap rule.

    No calibration rides on bare blocks: apply ``calibration_factor``
    where the level is computed, as the block-processing guide shows.

    :param path: The file to stream.
    :param block_size: Frames per block (at least 1).
    :param overlap: Frames each block shares with its predecessor
        (``0 <= overlap < block_size``).
    :return: An iterator of float64 blocks.
    :raises ValueError: If the geometry is invalid, the file matches no
        known audio format, or the data chunk is shorter than its header
        claims.
    :raises ImportError: If the format needs the ``[audio]`` extra and it
        is not installed.
    """
    block_size = int(block_size)
    overlap = int(overlap)
    if block_size < 1:
        raise ValueError(f"block_size must be at least 1; got {block_size}")
    if not 0 <= overlap < block_size:
        raise ValueError(
            f"overlap must satisfy 0 <= overlap < block_size; got "
            f"{overlap} against {block_size}"
        )
    format_name = _sniff(path)
    if format_name is None:
        fmt = parse_wav_chunks(path).fmt
        if fmt.resolved_tag in (WAVE_FORMAT_PCM, WAVE_FORMAT_IEEE_FLOAT):
            return _iter_wav_blocks(path, block_size, overlap)
        return _iter_soundfile_blocks(
            path, f"WAV {fmt.format_name}", block_size, overlap
        )
    return _iter_soundfile_blocks(path, format_name, block_size, overlap)
