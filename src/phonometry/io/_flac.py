#  Copyright (c) 2026. Jose Manuel Requena Plens
r"""Carrying the ``bext`` provenance chunk inside a FLAC file.

FLAC is the one compressed format a measurement archive can defend
(lossless by construction, RFC 9639), but its metadata model has no
broadcast extension: converting a BWF to FLAC with any generic tool drops
the originator, the sample-accurate timecode and the coding history --
exactly the fields that make the file a measurement record rather than
audio. The FLAC reference implementation solved the general problem long
ago: ``flac --keep-foreign-metadata`` stores each foreign RIFF chunk in
its own APPLICATION metadata block under the registered application ID
``riff``, verbatim (fourcc, 32-bit little-endian size, payload), so that
``flac -d --keep-foreign-metadata`` can reassemble the original WAV byte
for byte. This module reads and writes the ``bext`` chunk in that same
convention: a file phonometry writes is one the reference ``flac`` tool
recognises as carrying foreign RIFF metadata, and a WAV-to-FLAC-to-WAV
trip through phonometry preserves the provenance whole.

The FLAC container layout involved (RFC 9639 section 8): the ``fLaC``
magic, then a chain of metadata blocks -- each headed by one byte (bit 7:
last-block flag, bits 6-0: type, APPLICATION = 2) and a 24-bit big-endian
length -- with STREAMINFO mandatory first, and the audio frames after the
block whose last-flag is set. libsndfile writes no APPLICATION blocks and
python-soundfile exposes none, so the block is inserted here by rewriting
the metadata chain (clear the old last-flag, append the new block, set
its flag) and streaming the untouched audio frames behind it; the frames
are copied, never decoded, so the cost is I/O only and the compressed
audio is byte-identical.
"""

from __future__ import annotations

import shutil
import struct
from pathlib import Path

from ._chunks import _BEXT_CHUNK, BroadcastMetadata, _parse_bext

#: The FLAC stream marker (RFC 9639 section 8): the four bytes every
#: FLAC file must open with, in the mixed case the spec mandates.
_FLAC_MAGIC = b"fLaC"

#: Byte size of a FLAC metadata block header -- the last-block-flag and
#: block-type byte plus a 24-bit big-endian length (RFC 9639 section 8).
_METADATA_BLOCK_HEADER_BYTES = 4

#: FLAC metadata block types (RFC 9639 section 8.1).
_APPLICATION = 2

#: The registered application ID the FLAC reference tools use for foreign
#: RIFF chunks (flac --keep-foreign-metadata).
_RIFF_APP_ID = b"riff"

#: Preamble bytes inside an APPLICATION payload before the foreign
#: chunk's data begins: the 4-byte ``riff`` application ID plus the
#: verbatim RIFF chunk header (4-byte fourcc + 4-byte 32-bit
#: little-endian size).
_FOREIGN_CHUNK_HEADER_BYTES = 12


def read_flac_bext(path: str | Path) -> BroadcastMetadata | None:
    """Extract a ``bext`` carried in a FLAC APPLICATION ``riff`` block.

    Walks the metadata chain reading only the block headers and the
    APPLICATION payloads (never the audio frames); returns ``None`` when
    no block carries a ``bext`` chunk -- an ordinary FLAC, or one whose
    foreign metadata is some other chunk.
    """
    with Path(path).open("rb") as fh:
        if fh.read(4) != _FLAC_MAGIC:
            msg = f"{path}: not a FLAC file"
            raise ValueError(msg)
        while True:
            header = fh.read(4)
            if len(header) < _METADATA_BLOCK_HEADER_BYTES:
                return None
            block_type = header[0] & 0x7F
            length = int.from_bytes(header[1:4], "big")
            if block_type == _APPLICATION:
                payload = fh.read(length)
                if (
                    payload[:4] == _RIFF_APP_ID
                    and payload[4:8] == _BEXT_CHUNK
                    and len(payload) >= _FOREIGN_CHUNK_HEADER_BYTES
                ):
                    (size,) = struct.unpack_from("<I", payload, 8)
                    return _parse_bext(
                        payload[
                            _FOREIGN_CHUNK_HEADER_BYTES : _FOREIGN_CHUNK_HEADER_BYTES
                            + size
                        ]
                    )
            else:
                fh.seek(length, 1)
            if header[0] & 0x80:  # that was the last metadata block
                return None


def embed_flac_bext(path: str | Path, bext_payload: bytes) -> None:
    """Insert the ``bext`` chunk into a finished FLAC file's metadata.

    Rewrites the metadata chain -- STREAMINFO and friends copied as they
    are, the old last-block flag cleared -- and appends one APPLICATION
    block in the ``flac --keep-foreign-metadata`` convention: application
    ID ``riff``, then the verbatim RIFF chunk (``bext`` fourcc, 32-bit
    little-endian size, payload, alignment pad byte if the payload is
    odd). The audio frames are streamed across untouched, and the rewrite
    lands in a sibling temporary file moved over the original only once
    complete, so a failure cannot leave a half-written measurement file
    in place.

    :raises ValueError: If the file is not FLAC, or the chunk exceeds the
        24-bit block length a FLAC metadata block can hold.
    """
    chunk = (
        _RIFF_APP_ID
        + _BEXT_CHUNK
        + struct.pack("<I", len(bext_payload))
        + bext_payload
        + (b"\x00" if len(bext_payload) % 2 else b"")
    )
    if len(chunk) >= 1 << 24:
        msg = (
            f"{path}: bext chunk of {len(bext_payload)} bytes exceeds the "
            "24-bit FLAC metadata block length"
        )
        raise ValueError(msg)
    source = Path(path)
    staging = source.with_name(source.name + ".phonometry-tmp")
    with source.open("rb") as src, staging.open("wb") as dst:
        magic = src.read(4)
        if magic != _FLAC_MAGIC:
            msg = f"{path}: not a FLAC file"
            raise ValueError(msg)
        dst.write(magic)
        while True:
            header = src.read(4)
            if len(header) < _METADATA_BLOCK_HEADER_BYTES:
                msg = f"{path}: truncated FLAC metadata chain"
                raise ValueError(msg)
            body = src.read(int.from_bytes(header[1:4], "big"))
            last = bool(header[0] & 0x80)
            # Clear the last-block flag: our block goes beneath this one.
            dst.write(bytes([header[0] & 0x7F]) + header[1:])
            dst.write(body)
            if last:
                break
        dst.write(bytes([0x80 | _APPLICATION]) + len(chunk).to_bytes(3, "big"))
        dst.write(chunk)
        # The audio frames, streamed: copied, never decoded or held whole.
        shutil.copyfileobj(src, dst)
    staging.replace(source)
