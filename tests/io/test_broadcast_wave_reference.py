#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Transcription guards for the Broadcast Wave reference tables.

``tests/reference_data/broadcast_wave.py`` is a transcription: numbers
copied out of EBU Tech 3285 v2 (2011), EBU R 98-1999 and ITU-R BS.2088-2
(11/2025) that both the test suite and the conformance report treat as the
standard's voice. A digit slipped while copying would move every check
that reads it, so everything the documents let us recompute is recomputed
here: the offsets from the printed field sizes, the struct tail from the
printed reserved-byte accounting, the worked encoding examples from the
printed rounding formula, the ``ds64`` minimum from its seven printed
DWORDs, and the grammar example against the grammar's own parameter
table. The two independent transcriptions in this tree (this one and the
reader-side oracle of ``test_io_chunks``) are also pinned to each other.
"""

from __future__ import annotations

from decimal import Decimal

import pytest
import reference_data as ref
from test_io_chunks import BEXT_FIXED_SIZE, BEXT_OFFSETS


def test_bext_offsets_are_the_running_sum_of_the_printed_sizes() -> None:
    """Tech 3285 prints sizes, not offsets: the offsets must be their sum."""
    running = 0
    for name, offset, size in ref.TECH3285_BEXT_FIELDS:
        assert offset == running, f"{name}: offset {offset}, sum says {running}"
        running += size
    assert running == ref.TECH3285_BEXT_FIXED_SIZE == 602


def test_struct_tail_matches_the_reserved_byte_accounting() -> None:
    """Tech 3285 1.1 (p. 8) prints how each version carved the reserve.

    Version 0 reserved 254 bytes after the Version field; Version 1 took
    64 of them for the UMID; Version 2 took 10 of the remaining 190 for
    the loudness words, leaving the Reserved[180] the struct prints. Each
    printed number must reproduce the next, and the whole chain must
    close the 602-byte fixed part.
    """
    fields = {name: (offset, size) for name, offset, size in ref.TECH3285_BEXT_FIELDS}
    umid_offset, umid_size = fields["UMID"]
    assert umid_size == ref.TECH3285_UMID_SIZE
    # v0: everything after Version is reserved space.
    assert umid_offset + ref.TECH3285_V0_RESERVED == ref.TECH3285_BEXT_FIXED_SIZE
    # v1 carved the UMID off the front of it.
    assert ref.TECH3285_V0_RESERVED - ref.TECH3285_UMID_SIZE == ref.TECH3285_V1_RESERVED
    # v2 carved the five loudness WORDs off the front of what remained.
    loudness_words = [
        fields[name][1]
        for name in (
            "LoudnessValue",
            "LoudnessRange",
            "MaxTruePeakLevel",
            "MaxMomentaryLoudness",
            "MaxShortTermLoudness",
        )
    ]
    assert sum(loudness_words) == ref.TECH3285_LOUDNESS_BYTES
    assert (
        ref.TECH3285_V1_RESERVED - ref.TECH3285_LOUDNESS_BYTES
        == ref.TECH3285_V2_RESERVED
        == fields["Reserved"][1]
    )


@pytest.mark.parametrize(
    ("value", "carried_decimal", "carried_hex"), ref.TECH3285_LOUDNESS_EXAMPLES
)
def test_loudness_examples_recompute_from_the_printed_formula(
    value: float, carried_decimal: int, carried_hex: int
) -> None:
    """Tech 3285 2.4 defines integer part of (x + sgn(x) x 0.5), x = 100v.

    Recomputed in decimal arithmetic (the tables operate on the printed
    decimal literals, where 100 x 12.765 is exactly 1276.5; binary floats
    would beg the question the encoder's own tests answer). The
    hexadecimal column must be the int16 two's complement of the decimal
    one.
    """
    x = Decimal(str(value)) * 100
    half = Decimal("0.5") if x > 0 else Decimal("-0.5") if x < 0 else Decimal(0)
    recomputed = int(x + half)  # int() truncates: the "integer part"
    assert recomputed == carried_decimal
    assert carried_hex == carried_decimal & 0xFFFF


def test_unset_sentinel_sits_outside_every_valid_range() -> None:
    """7FFFh is int16 max, above the 270Fh (99.99) top of the valid ranges."""
    assert ref.TECH3285_LOUDNESS_UNSET == 2**15 - 1
    assert ref.TECH3285_LOUDNESS_UNSET > 0x270F


def test_r98_example_row_parses_under_its_own_grammar() -> None:
    """The Example 1 prefix must follow Appendix 1's parameter table."""
    parameters = ref.EBU_R98_EXAMPLE1_PCM_PREFIX.split(",")
    assert [p.split("=")[0] for p in parameters] == list(ref.EBU_R98_PCM_PARAMETERS)
    named = dict(p.split("=", 1) for p in parameters)
    assert named["A"] == "PCM"
    assert named["F"].isdigit()
    assert named["W"].isdigit()
    assert named["M"] in ref.EBU_R98_MODES
    assert named["T"] == ""  # the prefix stops where the free text begins
    # The prior-line example obeys the same grammar with a real T= text.
    prior = dict(p.split("=", 1) for p in ref.EBU_R98_EXAMPLE2_PRIOR_LINE.split(","))
    assert list(prior) == list(ref.EBU_R98_PCM_PARAMETERS)
    assert prior["M"] in ref.EBU_R98_MODES
    assert "," not in prior["T"]  # R 98: the free string contains no commas


def test_ds64_offsets_are_the_running_sum_of_seven_dwords() -> None:
    """BS.2088-2 4.1 prints seven leading DWORDs; 28 must be their sum."""
    running = 0
    for name, offset, size in ref.BS2088_DS64_FIELDS:
        assert offset == running, f"{name}: offset {offset}, sum says {running}"
        assert size == 4  # every leading field is a DWORD
        running += size
    assert running == ref.BS2088_DS64_MIN_SIZE == 28
    assert len(ref.BS2088_DS64_FIELDS) == 7


def test_the_two_independent_bext_transcriptions_agree() -> None:
    """This table and the reader-side oracle were transcribed separately.

    ``test_io_chunks.BEXT_OFFSETS`` was written down from Tech 3285 to pin
    the reader without sharing the writer's struct; ``reference_data``'s
    copy anchors the conformance report. Independent transcriptions of one
    printed table must be byte-for-byte the same table.
    """
    assert {
        name: (offset, size) for name, offset, size in ref.TECH3285_BEXT_FIELDS
    } == BEXT_OFFSETS
    assert ref.TECH3285_BEXT_FIXED_SIZE == BEXT_FIXED_SIZE
