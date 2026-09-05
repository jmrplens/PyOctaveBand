#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Splitting a citation string into the document and the place inside it.

Every check registers one free-text citation: ``IEC 61260-1:2014 Table 1``,
``Long, Architectural Acoustics 2e, Table 8.1``, ``Ainslie (2010) §11.4.6``.
Read as a whole string, seven clauses of one book are seven documents, which is
how the report came to publish a count of "standards" that counts citations.
Read as ``(designation, edition, clause)`` the seven are one document cited
seven times, and a row can be joined to the bibliography, filtered by issuing
body, or grouped by document.

The split is done by :func:`parse` and **verified**, not asserted:
:func:`recompose` rebuilds the citation from the three fields and must
reproduce the original string character for character. A citation that cannot
be rebuilt carries its split explicitly in ``reference_overrides.txt``, a
two-way ratchet - a line that is no longer needed fails just as loudly as a
citation that is missing one - so the file can only shrink as the parser
improves.

``kind`` reuses the vocabulary the site bibliography already declares in
``site/src/content.config.ts`` (``standard``, ``book``, ``article``,
``report``, ``web``) and adds ``derivation`` for the checks that cite no
document at all: a closed form synthesised to a known result, such as "All-pass
decomposition of a pure latency", is evidence, but it is not a citation.
"""

from __future__ import annotations

import enum
import functools
import pathlib
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

#: The ratchet. One citation per line, tab-separated, holding the split the
#: parser cannot derive: ``cite<TAB>kind<TAB>designation<TAB>edition<TAB>clause``.
#: An empty edition or clause is written as ``-``.
OVERRIDES_PATH = pathlib.Path(__file__).with_name("reference_overrides.txt")


class ReferenceKind(enum.StrEnum):
    """What sort of document a check cites."""

    STANDARD = "standard"
    BOOK = "book"
    ARTICLE = "article"
    REPORT = "report"
    WEB = "web"
    DERIVATION = "derivation"


@dataclass(frozen=True)
class Reference:
    """One citation, split into the document and the place inside it."""

    kind: ReferenceKind
    designation: str
    edition: str | None
    clause: str | None
    cite: str


# Issuing bodies whose designations the parser recognises. A citation opening
# with one of these is a standard, a regulation or an official report; the list
# is explicit rather than a pattern because "AS" and "RD" are also English and
# Spanish words, and a guess here silently changes what `counts` counts.
#
# Some entries name a body and the series it publishes in, "ECAC Doc" or
# "SAE ARP", because the number alone does not identify the document: Doc 29 is
# the airport-noise method and Doc 32 the rotorcraft one, and reading the body
# as the designation filed both under "ECAC Doc" as though they were one book.
# The rule for the ones that follow is the same as for the bodies: a fixed
# prefix that every citation of that series is written with. Longest first,
# since the alternation takes the first branch that matches.
_BODIES = (
    "ISO/IEC Guide",
    "ISO/IEC",
    "ISO/PAS",
    "ISO/TR",
    "ISO/TS",
    "ISO",
    "IEC/TR",
    "IEC",
    "CEN/TS",
    "EN",
    "CEN",
    "ANSI/ASA",
    "ANSI",
    "ASTM",
    "ASA WG",
    "ASA",
    "AES",
    "EBU Tech",
    "EBU",
    "ITU-R",
    "ITU-T",
    "ECMA",
    "SAE ARP",
    "SAE",
    "ARP",
    "ICAO Annex",
    "ICAO Doc",
    "ICAO",
    "SMPTE",
    "IEEE",
    "NIST",
    "ASHRAE",
    "VDI",
    "NT ACOU",
    "NORDTEST",
    "DIN",
    "BS",
    "NF",
    "UNE",
    "JIS",
    "NASA",
    "ECAC Doc",
    "ECAC",
    "FAA",
    "EASA",
    "WHO",
    "NIOSH",
    "OSHA",
    "MIL",
    "CTE",
    "RD",
    "Directive (EU)",
    "Directive",
    "Regulation",
    "Reglamento",
    "Recommendation",
    "Real Decreto",
)

#: Bodies whose documents are reports rather than standards: they are published
#: findings, not normative texts, and the bibliography types them accordingly.
_REPORT_BODIES = frozenset({"NASA", "ECAC", "FAA", "EASA", "WHO", "NIOSH", "OSHA"})

_BODY_ALTERNATION = "|".join(re.escape(body) for body in _BODIES)

#: ``IEC 61260-1:2014 Table 1`` - body, designation, optional edition, clause.
#: The designation stops at the first space that is followed by something that
#: is not part of a designation token, which the non-greedy run plus the
#: optional edition group achieves. The edition takes an amendment marker,
#: because ``ISO 10140-5:2010+A1`` otherwise matches no year at all and the
#: whole of ``10140-5:2010+A1`` falls into the clause, leaving the bare body as
#: the document.
_STANDARD = re.compile(
    rf"^(?P<designation>(?:{_BODY_ALTERNATION})[ ]?[A-Za-z]?[\w./()-]*?)"
    r"(?:(?P<sep>[:-])(?P<edition>\d{4}(?:\+A\d+)?))?"
    r"(?:\s+(?P<clause>\S.*))?$"
)

#: ``Long, Architectural Acoustics 2e, Table 8.1`` - an edition marker splits
#: the work from the place in it. ``2e``, ``4th ed``, ``6e`` all appear.
_EDITION = re.compile(
    r"^(?P<designation>.+?)\s+(?P<edition>\d+(?:e|nd ed|rd ed|th ed|st ed))"
    r"(?:(?:,\s+|\s+)(?P<clause>\S.*))?$"
)

#: ``Havelock 2008 Part I Ch. 6`` / ``Vigran (2008) Eqs. (9.18)-(9.20)`` - an
#: author (or authors) then the year the work is cited by.
_YEAR = re.compile(
    r"^(?P<designation>[A-Z][^,]*?)\s+(?P<edition>\(?(?:19|20)\d{2}\)?)"
    r"(?:(?:,\s+|\s+)(?P<clause>\S.*))?$"
)

#: Words that open the "where in the document" half of a citation. A citation
#: with no edition is split here instead, so ``Fastl & Zwicker Eq (10.2)``
#: still names one work rather than one more.
_CLAUSE_OPENERS = (
    "Eq",
    "Eqs",
    "Equation",
    "Ch",
    "Chapter",
    "Sec",
    "Secs",
    "Sect",
    "Sects",
    "Section",
    "Table",
    "Tables",
    "Fig",
    "Figs",
    "Figure",
    "App",
    "Appendix",
    "Annex",
    "Clause",
    "clause",
    "Formula",
    "Formulas",
    "Part",
    "Example",
    "Ejemplo",
    "Art",
    "Article",
    "p",
    "pp",
    "Note",
)

_CLAUSE_START = re.compile(
    r"^(?P<designation>.+?)\s+(?P<clause>(?:"
    + "|".join(re.escape(word) for word in _CLAUSE_OPENERS)
    + r")\b.*|§.*)$"
)


def recompose(reference: Reference) -> str:
    """Rebuild the citation string from the three fields.

    The verification the whole split rests on. Only the separators a citation
    is actually written with are tried - ``:`` and ``-`` before a standard's
    year, a comma or a space before a clause - so a rebuild that succeeds
    proves the three fields carry every character of the original in the
    original order, and a rebuild that fails means something was dropped or
    moved.

    :param reference: The split to rebuild.
    :return: The citation, or the empty string when no separator reproduces it.
    """
    parts = [reference.designation]
    if reference.edition is not None:
        parts.append(reference.edition)
    if reference.clause is not None:
        parts.append(reference.clause)
    for joiners in _JOINERS:
        candidate = parts[0]
        for part, joiner in zip(parts[1:], joiners, strict=False):
            candidate += joiner + part
        if candidate == reference.cite:
            return candidate
    return ""


#: Separator pairs tried when rebuilding: first before the edition, then before
#: the clause. Ordered so the commonest shape is found first.
_JOINERS: tuple[tuple[str, str], ...] = (
    (":", " "),
    ("-", " "),
    (" ", " "),
    (" ", ", "),
    (":", ", "),
    ("-", ", "),
    (" ", " / "),
)


def _kind_for(designation: str) -> ReferenceKind:
    """Classify a designation that opens with a known issuing body."""
    body = designation.split(" ", 1)[0]
    if body in _REPORT_BODIES:
        return ReferenceKind.REPORT
    return ReferenceKind.STANDARD


def _as_standard(cite: str) -> Reference | None:
    """Split a citation that opens with a recognised issuing body."""
    match = _STANDARD.match(cite)
    if match is None:
        return None
    return Reference(
        kind=_kind_for(match["designation"]),
        designation=match["designation"],
        edition=match["edition"],
        clause=match["clause"],
        cite=cite,
    )


def _as_edition(cite: str) -> Reference | None:
    """Split a citation carrying a book edition marker (``2e``, ``4th ed``)."""
    match = _EDITION.match(cite)
    if match is None:
        return None
    return Reference(
        kind=ReferenceKind.BOOK,
        designation=match["designation"],
        edition=match["edition"],
        clause=match["clause"],
        cite=cite,
    )


def _as_year(cite: str) -> Reference | None:
    """Split ``Author 1999 Eq. (17)`` or ``Author (2010) §11.4.6``."""
    match = _YEAR.match(cite)
    if match is None:
        return None
    return Reference(
        kind=ReferenceKind.ARTICLE,
        designation=match["designation"],
        edition=match["edition"],
        clause=match["clause"],
        cite=cite,
    )


def _as_clause(cite: str) -> Reference | None:
    """Split an undated work from the clause word that follows it."""
    match = _CLAUSE_START.match(cite)
    if match is None:
        return None
    return Reference(
        kind=ReferenceKind.BOOK,
        designation=match["designation"],
        edition=None,
        clause=match["clause"],
        cite=cite,
    )


def _parsers() -> tuple[Callable[[str], Reference | None], ...]:
    """The splitters, in the order they are tried."""
    return (_as_standard, _as_edition, _as_year, _as_clause)


def parse(cite: str, overrides: dict[str, Reference] | None = None) -> Reference:
    """Split one citation into its document and the place inside it.

    :param cite: The citation exactly as the check registered it.
    :param overrides: The ratchet, keyed by citation; defaults to the committed
        file.
    :return: The split, guaranteed to rebuild the citation verbatim.
    """
    table = _load_overrides() if overrides is None else overrides
    recorded = table.get(cite)
    if recorded is not None:
        return recorded
    for parser in _parsers():
        reference = parser(cite)
        if reference is not None and recompose(reference) == cite:
            return reference
    # No split survives the round trip, so the citation is one indivisible
    # name: a closed-form derivation, or a work whose title is the whole of it.
    return Reference(
        kind=ReferenceKind.DERIVATION,
        designation=cite,
        edition=None,
        clause=None,
        cite=cite,
    )


@functools.cache
def _load_overrides() -> dict[str, Reference]:
    """Read the ratchet file, keyed by the citation it corrects.

    Cached: :func:`parse` is called once per check, and re-reading a file 554
    times to answer the same question is 554 answers to one question.
    """
    if not OVERRIDES_PATH.is_file():
        return {}
    table: dict[str, Reference] = {}
    for number, line in enumerate(
        OVERRIDES_PATH.read_text(encoding="utf8").splitlines(), start=1
    ):
        if not line.strip() or line.startswith("#"):
            continue
        table.update(_override_line(line, number))
    return table


def _override_line(line: str, number: int) -> dict[str, Reference]:
    """Parse one ratchet line into its citation and split.

    :raises ValueError: If the line does not carry five tab-separated fields,
        or names a kind outside the vocabulary.
    """
    fields = line.split("\t")
    if len(fields) != 5:
        msg = (
            f"{OVERRIDES_PATH.name}:{number}: expected 5 tab-separated fields "
            f"(cite, kind, designation, edition, clause), found {len(fields)}."
        )
        raise ValueError(msg)
    cite, kind, designation, edition, clause = fields
    if kind not in tuple(ReferenceKind):
        msg = (
            f"{OVERRIDES_PATH.name}:{number}: kind {kind!r} is not one of "
            f"{[str(k) for k in ReferenceKind]}."
        )
        raise ValueError(msg)
    return {
        cite: Reference(
            kind=ReferenceKind(kind),
            designation=designation,
            edition=None if edition == "-" else edition,
            clause=None if clause == "-" else clause,
            cite=cite,
        )
    }
