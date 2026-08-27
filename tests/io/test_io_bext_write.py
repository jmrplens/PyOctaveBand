#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Round-trip tests for the bext writer against the phase-tested reader.

The reader was pinned field by field against a byte-offset oracle
transcribed independently from EBU Tech 3285 (tests/test_io_chunks.py), so
writing a chunk and reading it back with that reader checks the writer's
transcription without the two sides sharing a layout. The loudness fields
are cross-checked against the library's own BS.1770 implementation called
directly on the same samples the writer measured.
"""

from __future__ import annotations

import re
from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np
import pytest
import reference_data as ref

from phonometry import __version__
from phonometry.broadcast import program_loudness
from phonometry.io import BroadcastMetadata, read, write
from phonometry.io._bext import (
    _encode_loudness,
    coding_history_line,
    extend_coding_history,
    fresh_metadata,
)
from phonometry.io._chunks import parse_wav_chunks

if TYPE_CHECKING:
    from pathlib import Path

FS = 48000

#: A fully populated v2 chunk; every loudness value sits on the hundredth
#: grid, so the on-disk int16 = 100 x value encoding is exactly invertible
#: and the round-trip comparison can demand equality, not closeness.
FULL_META = BroadcastMetadata(
    description="courtyard measurement, position 3",
    originator="NTi XL2",
    originator_reference="ESUPV0000000000000000000META0001",
    origination_date="2026-08-18",
    origination_time="03-14-15",
    time_reference=(5 << 32) | 12345,  # exercises both 32-bit halves
    version=2,
    umid=bytes(range(64)),
    loudness_value=-23.12,
    loudness_range=6.5,
    max_true_peak_level=-1.02,
    max_momentary_loudness=-19.87,
    max_short_term_loudness=-21.4,
    coding_history="A=PCM,F=48000,W=24,M=stereo,T=recorder firmware 4.8\r\n",
)


def _expected_line(*, bits: int, channels: int = 1) -> str:
    mode = {1: ",M=mono", 2: ",M=stereo"}.get(channels, "")
    return f"A=PCM,F={FS},W={bits}{mode},T=phonometry {__version__}"


# ---------------------------------------------------------------------------
# Field-by-field round trips, on both writer paths (scipy append, in-house)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("subtype", "bits"),
    [
        ("PCM_16", 16),  # scipy writes the file; the chunk is appended
        ("PCM_24", 24),  # the in-house writer inlines the chunk before data
        ("DOUBLE", 64),  # float path: fact chunk present, append after data
    ],
)
def test_every_bext_field_survives_the_round_trip(
    tmp_path: Path, subtype: str, bits: int
) -> None:
    path = tmp_path / "provenance.wav"
    write(path, np.zeros(16), FS, subtype=subtype, bext=FULL_META)
    got = read(path).provenance
    assert got is not None
    # On disk the new line ends CR/LF; the reader strips the trailing
    # terminator (pinned in the phase-one chunk tests), so the reread
    # history is the original trail plus the writer's line, unterminated.
    expected_history = FULL_META.coding_history + _expected_line(bits=bits)
    assert got == replace(FULL_META, coding_history=expected_history)


def test_samples_survive_an_inline_odd_length_bext(tmp_path: Path) -> None:
    """An odd CodingHistory before data must not shift the data chunk."""
    meta = replace(FULL_META, coding_history="A=PCM,F=48000,W=24,T=x\r\n\r\n")
    codes = np.array([1, -1, 2**23 - 1, -(2**23)], dtype=np.int64)
    path = tmp_path / "odd_history.wav"
    write(path, codes / 2**23, FS, subtype="PCM_24", bext=meta)
    sig = read(path)
    assert np.asarray(sig).tolist() == (codes / 2**23).tolist()
    assert sig.provenance is not None
    assert sig.provenance.originator == FULL_META.originator


# ---------------------------------------------------------------------------
# CodingHistory: extended under the existing trail, never replaced
# ---------------------------------------------------------------------------


def test_provenance_is_carried_and_its_history_extended(tmp_path: Path) -> None:
    """Write bext, read it, write that Signal again: two lines accrue."""
    first = tmp_path / "gen1.wav"
    write(first, np.full(8, 0.25), FS, subtype="PCM_16", bext=FULL_META)
    generation1 = read(first)

    second = tmp_path / "gen2.wav"
    write(second, generation1, subtype="PCM_24")  # bext=None: carried
    generation2 = read(second).provenance
    assert generation2 is not None
    # Everything the recorder claimed is still there...
    assert generation2.originator == FULL_META.originator
    assert generation2.time_reference == FULL_META.time_reference
    assert generation2.umid == FULL_META.umid
    # ...and the audit trail grew one line per writing, oldest first (the
    # reader strips the final CR/LF terminator).
    assert generation2.coding_history == (
        FULL_META.coding_history
        + _expected_line(bits=16)
        + "\r\n"
        + _expected_line(bits=24)
    )


def test_bare_arrays_write_no_bext_by_default(tmp_path: Path) -> None:
    path = tmp_path / "plain.wav"
    write(path, np.zeros(8), FS)
    assert parse_wav_chunks(path).bext is None


def test_coding_history_line_follows_the_r98_grammar() -> None:
    assert coding_history_line(fs=44100, bits=16, channels=2) == (
        f"A=PCM,F=44100,W=16,M=stereo,T=phonometry {__version__}"
    )
    # R98 defines M= only for its named modes: other counts omit it.
    assert coding_history_line(fs=48000, bits=24, channels=4) == (
        f"A=PCM,F=48000,W=24,T=phonometry {__version__}"
    )


def test_extend_coding_history_restores_the_crlf_terminator() -> None:
    # The reader strips the trailing CR/LF; extension must restore it so
    # the old trail and the new line stay one-per-line on disk.
    assert extend_coding_history("A=PCM,F=48000,W=16,M=mono,T=a", "T=b") == (
        "A=PCM,F=48000,W=16,M=mono,T=a\r\nT=b\r\n"
    )
    assert extend_coding_history("", "T=b") == "T=b\r\n"


# ---------------------------------------------------------------------------
# bext="loudness": the R 128 fields are measured, not invented
# ---------------------------------------------------------------------------


def test_loudness_fields_match_bs1770_called_directly(tmp_path: Path) -> None:
    rng = np.random.default_rng(20260818)
    t = np.arange(4 * FS) / FS
    x = 0.05 * np.sin(2 * np.pi * 997 * t) + 0.01 * rng.standard_normal(t.size)
    path = tmp_path / "measured.wav"
    write(path, x, FS, subtype="DOUBLE", bext="loudness")
    got = read(path).provenance
    assert got is not None
    oracle = program_loudness(x, FS)
    # The chunk stores int16 hundredths: the reread value must equal the
    # direct measurement quantised to that grid, exactly.
    assert got.loudness_value == round(oracle.integrated * 100) / 100
    assert got.loudness_range == round(oracle.loudness_range * 100) / 100
    assert got.max_true_peak_level == round(oracle.true_peak * 100) / 100
    assert got.max_momentary_loudness == round(oracle.max_momentary * 100) / 100
    assert got.max_short_term_loudness == (round(oracle.max_short_term * 100) / 100)
    assert got.version == 2
    assert got.originator == f"phonometry {__version__}"[:32]
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", got.origination_date)
    assert re.fullmatch(r"\d{2}-\d{2}-\d{2}", got.origination_time)


def test_unset_loudness_reads_back_as_none(tmp_path: Path) -> None:
    """A v2 chunk without loudness writes the 0x7FFF sentinels."""
    path = tmp_path / "silent_fields.wav"
    write(path, np.zeros(8), FS, subtype="PCM_16", bext=fresh_metadata())
    got = read(path).provenance
    assert got is not None
    assert got.version == 2
    assert got.loudness_value is None
    assert got.loudness_range is None
    assert got.max_true_peak_level is None
    assert got.max_momentary_loudness is None
    assert got.max_short_term_loudness is None


def test_loudness_encoding_pins_its_edges() -> None:
    assert _encode_loudness(None) == 0x7FFF  # the unset sentinel
    assert _encode_loudness(-23.0) == -2300
    assert _encode_loudness(327.67) == 0x7FFE  # clamps off the sentinel
    assert _encode_loudness(float("-inf")) == -32768  # digital silence
    assert _encode_loudness(-400.0) == -32768


@pytest.mark.parametrize(
    ("value", "carried_decimal", "carried_hex"), ref.TECH3285_LOUDNESS_EXAMPLES
)
def test_loudness_encoding_reproduces_the_tech3285_worked_examples(
    value: float, carried_decimal: int, carried_hex: int
) -> None:
    """The six printed 2.4 example rows, down to the carried hexadecimal.

    These pin the tie behaviour: Tech 3285 rounds half away from zero, so
    -22.645 carries F727h and 12.765 carries 04FDh - and both of those
    products land exactly on a representable .5 in float64, so a
    round-half-to-even encoder really does miss the printed value by one
    code. (The tables' guard test recomputes the same rows from the
    printed formula in decimal arithmetic.)
    """
    encoded = _encode_loudness(value)
    assert encoded == carried_decimal
    assert encoded & 0xFFFF == carried_hex


# ---------------------------------------------------------------------------
# Refusals: a chunk that would lie about itself is not written
# ---------------------------------------------------------------------------


def test_oversize_fields_raise_instead_of_truncating(tmp_path: Path) -> None:
    long_originator = replace(fresh_metadata(), originator="x" * 33)
    with pytest.raises(ValueError, match=r"bext Originator is \d+ bytes"):
        write(tmp_path / "x.wav", np.zeros(4), FS, bext=long_originator)
    long_umid = replace(fresh_metadata(), umid=bytes(65))
    with pytest.raises(ValueError, match=r"bext UMID is \d+ bytes"):
        write(tmp_path / "x.wav", np.zeros(4), FS, bext=long_umid)


def test_version_gating_is_enforced_on_write(tmp_path: Path) -> None:
    umid_on_v0 = replace(fresh_metadata(), version=0, umid=bytes(64))
    with pytest.raises(ValueError, match=r"bext version \d+ has no UMID field"):
        write(tmp_path / "x.wav", np.zeros(4), FS, bext=umid_on_v0)
    loudness_on_v1 = replace(fresh_metadata(), version=1, loudness_value=-23.0)
    with pytest.raises(ValueError, match=r"bext version \d+ has no loudness fields"):
        write(tmp_path / "x.wav", np.zeros(4), FS, bext=loudness_on_v1)


def test_unknown_bext_argument_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"unknown bext 'auto'"):
        write(tmp_path / "x.wav", np.zeros(4), FS, bext="auto")
