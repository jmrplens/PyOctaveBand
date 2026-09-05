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
    assert reports == [f"{path}:1:2: vertical tab"]


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


def test_the_delivered_files_are_the_ones_that_still_need_it() -> None:
    """Every exemption in the tree is earned, which is what the ratchet asks."""
    _reports, stale = ccc.check(ccc.tracked_files())
    assert stale == []


def test_only_prose_and_code_are_read() -> None:
    """Figures and binaries are out of scope, by suffix and nothing else."""
    paths = ccc.tracked_files()
    assert paths, "the checkout tracks text files"
    assert {p.suffix for p in paths} <= ccc.SUFFIXES
