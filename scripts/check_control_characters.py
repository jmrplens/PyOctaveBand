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

The scan is over the tracked files whose content is prose or code, listed by
:data:`SUFFIXES`. Figures and other generated artefacts are out: an SVG may
legitimately carry anything inside a path, nobody reads it as text, and
scanning 2 400 of them would cost more than the check is worth.

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

#: What counts as prose or code here. Everything else a checkout holds is
#: either generated, binary, or a figure.
SUFFIXES = frozenset(
    {
        ".astro",
        ".cff",
        ".css",
        ".js",
        ".json",
        ".md",
        ".mdx",
        ".mjs",
        ".py",
        ".toml",
        ".ts",
        ".txt",
        ".yaml",
        ".yml",
    }
)

#: Carriage return, named once because it is the only exemptible character.
CARRIAGE_RETURN = 0x0D

#: Files that are copies of somebody else's, kept byte for byte, and why. The
#: exemption is the carriage return and nothing else: rewriting either file
#: would alter a licence text, or make a test's input no longer the file the
#: standard ships.
AS_DELIVERED: dict[str, str] = {
    ".github/brand/fonts/LICENSE-IBMPlex.txt": (
        "the IBM Plex licence, as the foundry publishes it"
    ),
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
        if name and pathlib.PurePath(name).suffix in SUFFIXES
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


def _named(path: pathlib.Path) -> pathlib.Path:
    """The path as a report should print it: relative to the tree when it is in it."""
    try:
        return path.relative_to(ROOT)
    except ValueError:
        return path


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
        name = str(_named(path))
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
