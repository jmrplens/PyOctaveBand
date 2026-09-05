#  Copyright (c) 2026. Jose Manuel Requena Plens
r"""The control-character gate, held against the byte that made it necessary.

``scripts/check_control_characters.py`` exists because a form feed reached a
raw docstring in ``noise_control/hvac.py``. In a raw string ``\frac`` is six
characters, and the paste replaced the first two with one invisible byte, so
the published formula read ``10 \log_{10} rac{(r + 1)^{2}}{4r}``: the fraction
was gone, Python was happy, and every other gate was too. These tests fix the
reading of that byte, of the rest of the class, and of the two characters this
corpus is written with and must keep.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys

_SCRIPTS = str(pathlib.Path(__file__).resolve().parent.parent / "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import check_control_characters as ccc


def _write(tmp_path: pathlib.Path, text: str) -> pathlib.Path:
    """Write the bytes asked for, on every platform.

    ``newline=""`` turns off the translation Windows would otherwise apply,
    which would put a carriage return in front of every newline and make a
    fixture about one control character a fixture about two.
    """
    path = tmp_path / "module.py"
    path.write_text(text, encoding="utf-8", newline="")
    return path


def test_the_form_feed_that_ate_a_backslash_is_found(tmp_path: pathlib.Path) -> None:
    """The defect the gate exists for, at the line and column it sits on."""
    path = _write(tmp_path, 'r"""\n   10 \\log_{10} \x0crac{a}{b}\n"""\n')
    assert ccc.offences(path) == [(2, 17, "form feed")]


def test_the_report_names_file_line_and_character(tmp_path: pathlib.Path) -> None:
    """What CI prints, for a file outside the tree as well as inside it."""
    path = _write(tmp_path, "a\x0bb\n")
    reports, _stale = ccc.check([path])
    assert reports == [f"{path.as_posix()}:1:2: vertical tab"]


def test_tab_and_newline_are_what_the_corpus_is_written_with(
    tmp_path: pathlib.Path,
) -> None:
    """The two C0 characters that must never be reported."""
    path = _write(tmp_path, "def f():\n\treturn 1\n")
    assert ccc.offences(path) == []


def test_a_carriage_return_is_in_the_class(tmp_path: pathlib.Path) -> None:
    """This tree is LF throughout, so a CRLF line is a defect and not a style."""
    path = _write(tmp_path, "first\r\nsecond\n")
    assert ccc.offences(path) == [(1, 6, "carriage return")]


def test_every_offence_in_a_file_is_reported(tmp_path: pathlib.Path) -> None:
    """Not just the first: a paste can carry more than one."""
    path = _write(tmp_path, "\x0ca\nb\x00\n")
    assert ccc.offences(path) == [(1, 1, "form feed"), (2, 2, "null")]


def test_a_delivered_file_keeps_its_carriage_returns_and_nothing_else(
    tmp_path: pathlib.Path,
) -> None:
    """The exemption is one character wide, not a pass for the whole file."""
    path = _write(tmp_path, "line\r\nnext\x0c\n")
    assert ccc.offences(path, allow_carriage_return=True) == [(2, 5, "form feed")]


def test_an_exemption_whose_file_is_gone_is_stale(tmp_path: pathlib.Path) -> None:
    """The ratchet: the list cannot outlive the files that earned it."""
    path = _write(tmp_path, "no carriage returns here\n")
    _reports, stale = ccc.check([path])
    assert stale == sorted(ccc.AS_DELIVERED)


def test_every_exemption_names_a_file_the_scan_reads() -> None:
    """The hatch cannot point at something the check never opens.

    Whether those files still carry a carriage return is the gate's own
    question at run time, and it is answered on a checkout whose line endings
    are the repository's; asserting it here would make the test depend on how
    the checkout was made rather than on what the tree holds.
    """
    read = {ccc._named(path) for path in ccc.tracked_files()}
    assert set(ccc.AS_DELIVERED) <= read


def _is_binary(path: pathlib.Path) -> bool:
    """Whether a file's first 64 kB are bytes rather than text.

    A NUL is the usual tell and is what git itself uses, but it is not enough
    here: a PDF opens with several kilobytes of ASCII before its first
    compressed stream. What every binary in this tree does have is a byte
    sequence that is not UTF-8.
    """
    head = path.read_bytes()[:65536]
    if b"\x00" in head:
        return True
    try:
        head.decode("utf-8")
    except UnicodeDecodeError:
        return True
    return False


def test_nothing_the_scan_skips_is_text() -> None:
    """The partition has to be total, and its skipped half has to be earned.

    Naming the text suffixes and reading those is the version that shipped
    first, and its scope test asserted that every scanned file had a scanned
    suffix, which is true by construction and could never fail. It hid
    twenty-nine tracked files, seven of them CSV oracles carrying the very
    character the check is for. So the list names what to skip, and this
    asserts that everything skipped really is binary: a NUL byte in the first
    four kilobytes, or an SVG, which is text nobody reads as prose.
    """
    root = ccc.ROOT
    listed = subprocess.run(  # noqa: S603 - a fixed argument vector
        ["git", "-C", str(root), "ls-files", "-z"],  # noqa: S607 - git is on PATH
        capture_output=True,
        check=True,
        text=True,
    ).stdout
    tracked = [root / name for name in listed.split("\0") if name]
    scanned = set(ccc.tracked_files())
    assert scanned, "the checkout tracks text files"

    skipped = [path for path in tracked if path not in scanned]
    assert skipped, "the checkout tracks figures and media"
    prose_in_the_skipped_half = [
        path.relative_to(root).as_posix()
        for path in skipped
        if path.suffix.lower() != ".svg" and not _is_binary(path)
    ]
    assert prose_in_the_skipped_half == []
