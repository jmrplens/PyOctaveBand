#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Broadcast Wave file layout: the published bext and ds64 geometry.

Where :mod:`.broadcast` transcribes the loudness *meter's* acceptance
signals, this module transcribes the loudness *file's* geometry: the field
sizes, encodings and sentinels a Broadcast Wave chunk is required to carry
on disk, as the standards bodies printed them.

Sources, cited line by line below:

* EBU Tech 3285 v2, *Specification of the Broadcast Wave Format (BWF)*,
  Geneva, May 2011 - the ``BROADCAST_EXT`` structure (2.3, pp. 9-10), the
  field semantics (pp. 10-12), the loudness integer encoding with its six
  worked examples (2.4, pp. 12-13) and the version history (1.1, p. 8).
* EBU R 98-1999, *Format for the <CodingHistory> field in Broadcast Wave
  Format files, BWF*, 1999 - the ``A=,F=,W=,M=,T=`` row grammar
  (Appendix 1, p. 2) and its printed examples (p. 3).
* Recommendation ITU-R BS.2088-2 (11/2025), *Long-form file format for the
  international exchange of audio programme materials with metadata*,
  Annex 1 - the BW64 top-level chunk (3.2, p. 9), the ``ds64`` structure
  (4.1, pp. 9-10; 4.2, pp. 10-11) and the 0xFFFFFFFF size-deferral rule
  (2.4, p. 5).

The offsets are not printed as numbers: the documents print the structs
field by field with their sizes, and every offset below is the running sum
of the printed sizes in the printed order. The transcription-guard tests
(``tests/io``) recompute what the documents let them recompute: the offsets
from the sizes, the fixed part against the reserved-byte accounting of
Tech 3285 1.1 (p. 8), the worked encoding examples against the 2.4
rounding formula, and the ``ds64`` minimum against its seven leading
DWORDs.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# EBU Tech 3285 v2 (May 2011) 2.3, pp. 9-10: the BROADCAST_EXT structure.
# (field name, offset, size in bytes) in the printed order; each offset is
# the running sum of the sizes above it, starting at 0.
# ---------------------------------------------------------------------------
TECH3285_BEXT_FIELDS: tuple[tuple[str, int, int], ...] = (
    ("Description", 0, 256),           # CHAR[256], p. 9
    ("Originator", 256, 32),           # CHAR[32], p. 9
    ("OriginatorReference", 288, 32),  # CHAR[32], p. 9
    ("OriginationDate", 320, 10),      # CHAR[10], p. 9
    ("OriginationTime", 330, 8),       # CHAR[8], p. 10
    ("TimeReferenceLow", 338, 4),      # DWORD, p. 10
    ("TimeReferenceHigh", 342, 4),     # DWORD, p. 10
    ("Version", 346, 2),               # WORD, p. 10
    ("UMID", 348, 64),                 # BYTE[64] (UMID_0..UMID_63), p. 10
    ("LoudnessValue", 412, 2),         # WORD, p. 10
    ("LoudnessRange", 414, 2),         # WORD, p. 10
    ("MaxTruePeakLevel", 416, 2),      # WORD, p. 10
    ("MaxMomentaryLoudness", 418, 2),  # WORD, p. 10
    ("MaxShortTermLoudness", 420, 2),  # WORD, p. 10
    ("Reserved", 422, 180),            # BYTE[180], zeros in v1/v2, pp. 10-11
)

#: Size of the fixed part (running sum of the sizes above); the
#: variable-length CodingHistory follows it.
TECH3285_BEXT_FIXED_SIZE = 602

#: Tech 3285 1.1 (p. 8) reserved-byte accounting, the printed arithmetic the
#: guard test recomputes the tail of the struct from: Version 1 took "64 of
#: the 254 reserved bytes" of Version 0 for the UMID, and Version 2 took
#: "10 of the 190 reserved bytes" of Version 1 for the loudness fields.
TECH3285_V0_RESERVED = 254    # p. 8 ("64 of the 254 reserved bytes")
TECH3285_UMID_SIZE = 64       # p. 8 / p. 11 (64 bytes, SMPTE 330M)
TECH3285_V1_RESERVED = 190    # p. 8 ("10 of the 190 reserved bytes")
TECH3285_LOUDNESS_BYTES = 10  # p. 8 (five WORDs)
TECH3285_V2_RESERVED = 180    # pp. 10-11 (Reserved[180], zeros)

#: Version field values (p. 11): "For Version 1 it shall be set to 0001h and
#: for Version 2 it shall be set to 0002h."
TECH3285_VERSION_1 = 0x0001
TECH3285_VERSION_2 = 0x0002

# ---------------------------------------------------------------------------
# Tech 3285 2.4 (pp. 12-13): loudness integer encoding. Each parameter is
# stored as a 16-bit signed integer of 100 x value, rounded to the nearest
# integer with ties away from zero ("integer part of (x + sgn(x) * 0.5)",
# p. 12). The two example tables print six (float, carried decimal, carried
# hexadecimal) rows, transcribed here verbatim.
# ---------------------------------------------------------------------------
#: (float value, value carried in BWF as decimal, as hexadecimal).
TECH3285_LOUDNESS_EXAMPLES: tuple[tuple[float, int, int], ...] = (
    (-22.644, -2264, 0xF728),  # p. 12 (negative numbers table)
    (-22.645, -2265, 0xF727),  # p. 12
    (-22.646, -2265, 0xF727),  # p. 12
    (12.764, 1276, 0x04FC),    # p. 13 (positive numbers table)
    (12.765, 1277, 0x04FD),    # p. 13
    (12.766, 1277, 0x04FD),    # p. 13
)

#: The unset sentinel (p. 13): "If any of the loudness parameters are not
#: being used then their 16-bit integer values shall be set to 7FFFh, which
#: is a value outside the range of the parameter values."
TECH3285_LOUDNESS_UNSET = 0x7FFF

# ---------------------------------------------------------------------------
# EBU R 98-1999 Appendix 1: the CodingHistory row grammar Tech 3285 2.3
# points at (its Note on p. 12). Comma-separated parameters, each row
# terminated by CR/LF (p. 2); B= exists only for MPEG coding, so a linear
# PCM row is A=,F=,W=,M=,T= exactly.
# ---------------------------------------------------------------------------
#: Parameter order of a PCM row per the Appendix 1 syntax table (p. 2).
EBU_R98_PCM_PARAMETERS: tuple[str, ...] = ("A", "F", "W", "M", "T")

#: The allowed M= modes (p. 2). Channel counts outside these have no mode.
EBU_R98_MODES: tuple[str, ...] = ("mono", "stereo", "dual-mono", "joint-stereo")

#: Example 1, line 1 (p. 3): a 48 kHz 16-bit stereo linear PCM recording's
#: row, up to its free-text T= parameter.
EBU_R98_EXAMPLE1_PCM_PREFIX = "A=PCM,F=48000,W=16,M=stereo,T="

#: Example 2, line 2 (p. 3): a prior coding step another writer left behind,
#: which a new application preserves while adding its own row beneath
#: (Tech 3285 p. 12: "Each new coding application shall add a new string").
EBU_R98_EXAMPLE2_PRIOR_LINE = "A=PCM,F=48000,W=18,M=stereo,T=NVision; NV1000; A/D"

# ---------------------------------------------------------------------------
# ITU-R BS.2088-2 (11/2025) Annex 1: the 64-bit container.
# ---------------------------------------------------------------------------
#: 3.2 (p. 9): the top-level chunk's ckID, and its ckSize, which "shall be
#: set to 0xFFFFFFFF to indicate that this size value is not used and the
#: <ds64> chunk shall be used for determining sizes". 2.4 (p. 5) states the
#: general rule: a 32-bit size field of 0xFFFFFFFF defers to the 64-bit
#: value in <ds64>; any other value is used as-is. The <ds64> chunk "has to
#: be the first chunk after the 'BW64 chunk'" (p. 5).
BS2088_BW64_FOURCC = b"BW64"
BS2088_SIZE_SENTINEL = 0xFFFFFFFF

#: 4.1 (pp. 9-10): the DataSize64Chunk payload, after its ckID/ckSize
#: header, opens with seven DWORDs, transcribed here as (field, offset
#: within the payload, size) with each offset the running sum of the
#: printed 4-byte widths. bw64Size low/high replaces the <BW64> (or <RIFF>)
#: size and dataSize low/high the <data> size, both little-endian low word
#: first (4.2, pp. 10-11); the dummy pair "shall be ignored when read, and
#: set to zero when writing" under BS.2088 and exists for compatibility
#: with EBU Tech 3306's RF64, which carries <fact> size information there
#: (4.2, p. 11).
BS2088_DS64_FIELDS: tuple[tuple[str, int, int], ...] = (
    ("bw64SizeLow", 0, 4),    # DWORD, p. 9
    ("bw64SizeHigh", 4, 4),   # DWORD, p. 9
    ("dataSizeLow", 8, 4),    # DWORD, p. 9
    ("dataSizeHigh", 12, 4),  # DWORD, p. 10
    ("dummyLow", 16, 4),      # DWORD, p. 10
    ("dummyHigh", 20, 4),     # DWORD, p. 10
    ("tableLength", 24, 4),   # DWORD, p. 10
)

#: Minimum <ds64> payload: the seven leading DWORDs, before the optional
#: ChunkSize64 table. 4.1/4.3 (pp. 10-11) print the same number for the
#: <JUNK> placeholder: its size "Shall be at least 28 to be a placeholder
#: for <ds64>".
BS2088_DS64_MIN_SIZE = 28
