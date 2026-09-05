#!/usr/bin/env python3
#  Copyright (c) 2026. Jose Manuel Requena Plens
r"""Fail on a control character that a text file cannot survive carrying.

A form feed is one byte, it prints as nothing in every editor and diff this
project uses, and inside a raw docstring it is *not* a form feed to the reader
of the published page: ``\frac`` is written ``\f`` plus ``rac``, so the API
reference published a change-of-section formula as ``10 \log_{10} rac{...}``
with the fraction gone and no error anywhere. Python did not complain, because
in a raw string the backslash and the ``f`` are two characters; the byte that
replaced them arrived from a paste. The same byte in a guide is a paragraph
break Markdown does not make, and a vertical tab or a stray escape is the same
defect wearing another number.

So the rule is the whole class rather than the one byte that was found: no C0
control character other than tab and newline, and no delete. Carriage return
is in the class deliberately -- this tree is LF throughout, and a CRLF line
that slipped in would break the hard-wrap checks that read line ends.

:data:`AS_DELIVERED` is the escape hatch, and it is only ever about that one
character: two files here are copies of somebody else's, a licence and an
oracle, and their line ending is not this project's to choose. It is a ratchet
like the rest -- an entry whose file is gone, or whose carriage returns have
been taken out, fails too, so the list cannot outlive its reason.

The scan is over every tracked file except the suffixes :data:`NOT_TEXT`
names, which are the binary ones plus SVG. Naming what to skip rather than
what to read is deliberate: the first version named the text suffixes and so
never opened twenty-nine tracked files, among them seven CSV oracles that
carry exactly the character it looks for.

Usage::

    python scripts/check_control_characters.py

Exit status 0 when every file is clean, 1 otherwise, naming the file, the line
and the character by name.
"""

from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent

#: Suffixes whose bytes are not text anybody reads: figures, media, archives
#: and the binary oracles. Everything else tracked is scanned, which is the
#: safe way round -- a text format nobody thought of is read by default, and
#: only a new *binary* one has to be declared. The first version of this list
#: went the other way, naming the text suffixes, and silently skipped
#: twenty-nine tracked files, seven of which carry the very character the
#: check exists for.
#:
#: SVG is here for a different reason than the rest: it is text, but it is
#: generated, there are two thousand four hundred of them, and nothing in one
#: is read as prose.
NOT_TEXT = frozenset(
    {
        ".gif",
        ".ico",
        ".jpg",
        ".npz",
        ".pdf",
        ".png",
        ".svg",
        ".ttf",
        ".wav",
        ".webm",
        ".webp",
        ".zip",
    }
)

#: Carriage return, named once because it is the only exemptible character.
CARRIAGE_RETURN = 0x0D

#: Files whose line endings are not this project's to choose, and why. The
#: exemption is the carriage return and nothing else. Rewriting the first two
#: would alter a licence text or make a test's input no longer the file the
#: standard ships; the CNOSSOS-EU tables are extracts of a published workbook
#: and of the Official Journal, and a re-extraction to check them against the
#: source produces the same CRLF, so normalising them here would put a diff
#: between the oracle and the thing it was taken from.
AS_DELIVERED: dict[str, str] = {
    ".github/brand/fonts/LICENSE-IBMPlex.txt": (
        "the IBM Plex licence, as the foundry publishes it"
    ),
    "tests/data/cnossos/rail_emission_cases.csv": "as extracted from the workbook",
    "tests/data/cnossos/rail_frequency_tables_2015.csv": "as extracted from the catalogues",
    "tests/data/cnossos/rail_vehicles_2015.csv": "as extracted from the catalogue",
    "tests/data/cnossos/rail_wavelength_tables_2015.csv": "as extracted from the catalogues",
    "tests/data/cnossos/road_coefficients_2015.csv": "as extracted from the Official Journal",
    "tests/data/cnossos/road_emission_cases.csv": "as extracted from the workbook",
    "tests/data/cnossos/road_surfaces_2015.csv": "as extracted from the Official Journal",
    "tests/data/iso532_1/iso532_1_test_signal_1_levels.txt": (
        "the ISO 532-1 reference levels, as the standard delivers them"
    ),
}

#: The forbidden characters, by codepoint, with the name a report should use.
#: Tab and newline are the two C0 characters this corpus is written with.
FORBIDDEN: dict[int, str] = {
    0x00: "null",
    0x01: "start of heading",
    0x02: "start of text",
    0x03: "end of text",
    0x04: "end of transmission",
    0x05: "enquiry",
    0x06: "acknowledge",
    0x07: "bell",
    0x08: "backspace",
    0x0B: "vertical tab",
    0x0C: "form feed",
    0x0D: "carriage return",
    0x0E: "shift out",
    0x0F: "shift in",
    0x10: "data link escape",
    0x11: "device control 1",
    0x12: "device control 2",
    0x13: "device control 3",
    0x14: "device control 4",
    0x15: "negative acknowledge",
    0x16: "synchronous idle",
    0x17: "end of transmission block",
    0x18: "cancel",
    0x19: "end of medium",
    0x1A: "substitute",
    0x1B: "escape",
    0x1C: "file separator",
    0x1D: "group separator",
    0x1E: "record separator",
    0x1F: "unit separator",
    0x7F: "delete",
}


def tracked_files() -> list[pathlib.Path]:
    """Every tracked file this check reads, in the order git lists them."""
    listed = subprocess.run(  # noqa: S603 - a fixed argument vector
        ["git", "-C", str(ROOT), "ls-files", "-z"],  # noqa: S607 - git is on PATH
        capture_output=True,
        check=True,
        text=True,
    ).stdout
    return [
        ROOT / name
        for name in listed.split("\0")
        if name and pathlib.PurePath(name).suffix.lower() not in NOT_TEXT
    ]


def offences(
    path: pathlib.Path, *, allow_carriage_return: bool = False
) -> list[tuple[int, int, str]]:
    """The forbidden characters in one file, as line, column and name."""
    forbidden = (
        {code: name for code, name in FORBIDDEN.items() if code != CARRIAGE_RETURN}
        if allow_carriage_return
        else FORBIDDEN
    )
    try:
        # newline="" turns off universal-newline translation, without which a
        # CRLF arrives as a bare "\n" and the carriage return this looks for
        # is the one character it can never see.
        text = path.read_text(encoding="utf-8", newline="")
    except (OSError, UnicodeDecodeError):  # pragma: no cover - not in this tree
        return []
    found: list[tuple[int, int, str]] = []
    line = column = 1
    for character in text:
        name = forbidden.get(ord(character))
        if name is not None:
            found.append((line, column, name))
        if character == "\n":
            line, column = line + 1, 1
        else:
            column += 1
    return found


def _named(path: pathlib.Path) -> str:
    """The path as a report prints it and as :data:`AS_DELIVERED` keys it.

    Relative to the tree when it is inside it, and always with forward
    slashes: ``str()`` on a Windows path gives backslashes, which match no key
    in the hatch, so the two exempt files would be reported there and their
    entries called stale in the same run.
    """
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def check(paths: list[pathlib.Path]) -> tuple[list[str], list[str]]:
    """The offences over the given files, and the stale exemptions.

    :param paths: The files to read.
    :return: One report line per forbidden character, and the
        :data:`AS_DELIVERED` keys that no longer describe a file carrying a
        carriage return, either because it is not among ``paths`` or because
        it has been converted since.
    """
    reports: list[str] = []
    still_delivered: set[str] = set()
    for path in paths:
        name = _named(path)
        exempt = name in AS_DELIVERED
        for line, column, character in offences(path, allow_carriage_return=exempt):
            reports.append(f"{name}:{line}:{column}: {character}")
        if exempt and any(
            character == "\r"
            for character in path.read_text(encoding="utf-8", newline="")
        ):
            still_delivered.add(name)
    return reports, sorted(set(AS_DELIVERED) - still_delivered)


def main(argv: list[str] | None = None) -> int:
    """Report every forbidden control character in the tracked text corpus."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)

    paths = tracked_files()
    failures, stale = check(paths)
    if not failures and not stale:
        print(f"No stray control character in the {len(paths)} text files tracked.")
        return 0
    if failures:
        print("::error::a control character in a text file - see below")
        for failure in failures:
            print(f"  {failure}")
        print(
            "  -> delete it. A form feed inside a raw docstring eats the backslash "
            "of the command that follows it, and nothing downstream can see that."
        )
    for name in stale:
        print(
            f"::error::AS_DELIVERED lists {name}, which no longer needs the exemption"
        )
    return 1


if __name__ == "__main__":
    sys.exit(main())
