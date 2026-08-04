#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Guard the evidence of every entry in ``docs/ERRATA.md``.

Each entry in the registry is a permanent public statement that a named
standards body, author or publisher printed something wrong. Two failure modes
have actually shipped, and this script covers both.

**Extraction, not the page.** PDF text layers silently delete glyphs. Most of
the maths-bearing documents this project cites emit no ``U+221A`` at all over
their whole text layer, so every radical in them extracts as if it were absent
and ``f_T/sqrt(2)`` reads back as ``f_T/2``. An entry drafted on that basis
accused an author of printing ``2/(3 pi)`` where the page prints
``2/(sqrt(3) pi)``; three independent extractors agreed with each other and all
three were wrong, and it was caught in review. Hence :func:`check_pages`: every
entry must cite the page it quotes, by PDF page index and printed folio, or be
listed in :data:`PAGE_CITATION_ALLOWLIST` with a reason. A page index is a
property of the edition, so any copy of that edition resolves it.

**The arithmetic tells.** When a glyph is lost the printed and derived values
differ by the lost factor, so their ratio lands on one of a small set of
constants. That entry announced its own ratio twice, as "1,73 times
too small" and as "a fixed 2,4 dB offset" -- ``sqrt(3)`` and
``20 lg sqrt(3)``. Hence :func:`check_ratios`: any multiplicative claim in an
entry whose value sits within 0,5 % of ``sqrt(2)``, ``sqrt(3)``, ``pi``,
``2 pi``, ``1/sqrt(2)``, ``ln 2`` or a small integer is reported, and is an
error unless that entry cites the page it read. A genuine factor-of-ten
misprint trips the test too; that is the point, since the way to clear it is
to read the page as an image.

**The finding, not the procedure.** :func:`check_procedure_is_not_cited` keeps
the account of how a page was put on screen out of the registry. An entry says
what a document prints and where; the resolution someone rendered it at is
method, belongs in ``CONTRIBUTING.md``, and reads to an issuing body like a
machine's account of itself rather than a finding a maintainer stands behind.

Usage::

    python scripts/check_errata_evidence.py            # both checks
    python scripts/check_errata_evidence.py --ratios   # ratio report only

CI runs the first form. See ``CONTRIBUTING.md``, "Filing an errata entry".
"""

from __future__ import annotations

import argparse
import math
import pathlib
import re
import sys
from dataclasses import dataclass

ROOT = pathlib.Path(__file__).resolve().parent.parent
REGISTRY = ROOT / "docs" / "ERRATA.md"

#: Entries filed before the page rule, or whose evidence is not a reading of a
#: page at all. Each value is the reason. **This list is meant to shrink**:
#: clearing an entry means reading its page as an image and citing that page in
#: its Evidence bullet, not adding a new line here. A ratio hit is never
#: excused by membership of this list.
PAGE_CITATION_ALLOWLIST: dict[str, str] = {
    "ISO 12354-1:2017 Table L.3 / ISO 12354-2:2017 Table G.3 (perimeter sums)":
        "the claim is about five recomputed sums, not about printed characters",
    "ISO 12354-1:2017 Table L.3 / ISO 12354-2:2017 Table G.3 (external wall ηint)":
        "the claim is a recomputation of Formula (C.1), not a character reading",
    "ISO 12354-1:2017, Table L.4 (second path block labelled 2d)":
        "the claim is that three recomputed columns identify a different path",
    "ISO 12354-2:2017, Table G.1 (50 Hz to 80 Hz flanking columns)":
        "the claim is a band-by-band recomputation of two printed tables",
    "ISO 10848-1:2006, Clause 8.1.1, Formula (20) (spurious π in the critical frequency)":
        "filed before the page rule; corrected upstream in the 2017 edition",
    "ISO 12999-1:2020, Table 4 (missing 500 Hz row)":
        "the claim is that a row is absent, which no reading of a page can show",
    "ISO 10052:2021, Table 4 volume-range header":
        "filed before the page rule",
    "ECMA-418-2:2025 (4th edition), clause 5.1.5.2 (last block index)":
        "filed before the page rule",
    "ECMA-418-2:2025 (4th edition), clause 9.1.4, Formula (127) (HSA kernel phase)":
        "filed before the page rule",
    "ECMA-418-2:2025 (4th edition), clause 9.1.5, Formula (144) (bin offset)":
        "filed before the page rule",
    "ECMA-418-2:2025 (4th edition), clause 9.1.7 (units of the fine-tuning constants)":
        "filed before the page rule",
    "ECMA-418-2:2025 (4th edition), clause 9 introduction (broken cross-reference)":
        "filed before the page rule",
    "UNE-EN 61043:1999, clause 6.1 (class 2 frequency range dropped in translation)":
        "filed before the page rule; the claim is a comparison of two prose sentences",
    "ISO/PAS 1996-3:2022, Clause 5 (cross-references of r and d)":
        "filed before the page rule; the held copy is a scan without a text layer",
    "ISO 9613-2:1996, Table 2 (15 °C / 80 % / 1 kHz cell)":
        "filed before the page rule",
    "ANSI S3.5-1997, Annex C worked examples (official WG S3-79 errata)":
        "the standard is not held locally; the entry says so in a leading notice",
    "ANSI S3.5-1997, captions of Tables 1 to 4 (official WG S3-79 erratum)":
        "the standard is not held locally; the entry says so in a leading notice",
    "NORAH2 rotorcraft guidance SC01.D1.5d (EASA.2020.FC.06), Eq. (21)":
        "filed before the page rule",
    "NORAH2 rotorcraft guidance SC01.D1.5d (EASA.2020.FC.06), §A.3.1 triangulation":
        "the claim is about shipped lookup tables, not about printed characters",
    "NORAH2 rotorcraft guidance SC01.D1.5d (EASA.2020.FC.06), Eq. (46)":
        "filed before the page rule",
    "RANDI 3.1 Physics Description (NRL, Breeding et al.), Table 2":
        "filed before the page rule",
    "Jiménez, Groby, Pagneux & Romero-García (2017), Appl. Sci. 7(6), 618, Eqs. (7)-(8)":
        "filed before the page rule",
    "Jiménez et al. (2017), Appl. Sci. 7(6), 618 / Sci. Rep. 7, 5389, slit-radiation term":
        "filed before the page rule",
    "Bies, Hansen & Howard, Engineering Noise Control 5e (2017), Eq. (8.141)":
        "filed before the page rule",
}

#: A page citation: the PDF page index and the printed folio, anchored to the
#: DOCUMENT rather than to a file. The registry is a public statement about
#: published sources; a filesystem path pinned it to one machine, said nothing
#: a reader could follow, and leaked layout. A PDF page index is a property of
#: the edition, so any copy of that edition resolves it, and the printed folio
#: is what a reader holding the paper looks for.
#: A run of page indices is accepted as well as a single one, because several
#: entries quote a table that spans pages or a claim that needs two of them.
PAGE_CITATION = re.compile(
    r"PDF\s+pages?\s+\d+(?:\s*(?:,|and|to)\s*\d+)*\s*"
    r"\(\s*printed\s+pp?\.\s*[\w\s,/.-]+?\)",
    re.IGNORECASE,
)

#: The edition the page belongs to, which follows the page group. A page index
#: alone resolves in no library: "PDF page 45 (printed p. 39)" of *what*? The
#: designation is what a reader takes to a standards catalogue, so the citation
#: is only complete once it names one. Further page groups may sit in between,
#: because an entry that reads two pages of the same edition names it once.
_CITED_EDITION = re.compile(r"\bof\s+(?:the\s+)?[A-Z0-9(\u201c\"']", re.UNICODE)

#: How far after the page group the edition may sit: long enough for the second
#: and third page group of an entry that reads a table across a spread.
_EDITION_WINDOW = 200

#: How a page was read: a resolution, or the act of rendering one. A bare
#: resolution has no other meaning in this registry, so it is rejected wherever
#: it appears; "render" is an ordinary English verb an entry may need ("the
#: dropped exponent renders Formula (7) inconsistent"), so only the procedural
#: sense counts, and only in the Evidence bullet.
_RESOLUTION = re.compile(r"\b\d+[\s\u2010-\u2015-]*dpi\b", re.IGNORECASE)
_RENDERING = re.compile(
    r"\brender(?:ed|ing|s)?\s+(?:(?:of|at|from)\b|(?:the\s+)?(?:page|image|crop|scan)\b)",
    re.IGNORECASE,
)

#: Ratios that are the signature of a lost glyph rather than an author's error.
_NAMED_RATIOS: dict[str, float] = {
    "sqrt(2)": math.sqrt(2.0),
    "sqrt(3)": math.sqrt(3.0),
    "pi": math.pi,
    "2 pi": 2.0 * math.pi,
    "1/sqrt(2)": 1.0 / math.sqrt(2.0),
    "ln 2": math.log(2.0),
}
#: Small integers count too: a dropped digit or a mis-reduced constant lands on
#: one, and a factor of ten is exactly how a reduced formula fails.
_SMALL_INTEGERS = tuple(float(n) for n in range(2, 13))

#: Relative tolerance of the ratio test.
_TOLERANCE = 5e-3

#: "1,73 times too small", "3.14 times larger".
_TIMES = re.compile(
    r"(?P<value>\d+[.,]\d+|\d+)\s+times\s+(?:too\s+)?"
    r"(?:small|large|big|high|low|greater|smaller|more|less)",
    re.IGNORECASE,
)
#: "a factor of 1,73", "by a factor 10", "a factor-16 error". The word boundary
#: keeps "prefactor 4 N h" out: that is a term of a formula, not a ratio claim.
_FACTOR = re.compile(
    r"\bfactor[\s-]+(?:of\s+)?(?P<value>\d+[.,]\d+|\d+)",
    re.IGNORECASE,
)
#: "a fixed 2,4 dB offset", "2,4 dB too high", "off by 2,4 dB".
_DECIBEL = re.compile(
    r"(?P<value>\d+[.,]\d+|\d+)\s*dB\s+(?:offset|bias|too\s+(?:high|low))"
    r"|off\s+by\s+(?P<value2>\d+[.,]\d+|\d+)\s*dB",
    re.IGNORECASE,
)


#: The start of the bullet the citation has to live in. Anything quoted in "The
#: print" or narrated in "The problem" is not evidence, so the citation is only
#: read from here to the next top-level bullet.
_EVIDENCE_BULLET = re.compile(
    r"^- \*\*Evidence:?\*\*.*?(?=^- \*\*|\Z)",
    re.MULTILINE | re.DOTALL,
)


@dataclass(frozen=True)
class Entry:
    """One ``## `` section of the registry."""

    title: str
    body: str
    line: int

    @property
    def evidence(self) -> str:
        """The Evidence bullet, including its continuation lines."""
        return "".join(match.group(0) for match in _EVIDENCE_BULLET.finditer(self.body))

    @property
    def cites_page(self) -> bool:
        """A page group followed, within the same sentence run, by an edition."""
        return any(
            _CITED_EDITION.search(
                self.evidence[match.end() : match.end() + _EDITION_WINDOW]
            )
            for match in PAGE_CITATION.finditer(self.evidence)
        )


@dataclass(frozen=True)
class RatioHit:
    """A multiplicative claim that matches a lost-glyph signature."""

    entry: Entry
    quoted: str
    ratio: float
    name: str


def _number(text: str) -> float:
    """Read a decimal written with either separator."""
    return float(text.replace(",", "."))


def parse_entries(markdown: str) -> list[Entry]:
    """Split the registry into entries.

    An entry is a ``## `` section carrying a ``- **Status:**`` bullet; the
    preamble and the closing "Related source properties" section are not
    entries and are skipped.
    """
    entries: list[Entry] = []
    title: str | None = None
    start = 0
    buffer: list[str] = []
    for number, line in enumerate(markdown.splitlines(), 1):
        if line.startswith("## "):
            if title is not None:
                entries.append(Entry(title, "\n".join(buffer), start))
            title, start, buffer = line[3:].strip(), number, []
        elif title is not None:
            buffer.append(line)
    if title is not None:
        entries.append(Entry(title, "\n".join(buffer), start))
    return [e for e in entries if "- **Status:**" in e.body]


def _match_named(ratio: float) -> str | None:
    """Name the lost-glyph constant *ratio* sits on, if any."""
    for probe in (ratio, 1.0 / ratio) if ratio > 0.0 else ():
        for name, value in _NAMED_RATIOS.items():
            if abs(probe - value) <= _TOLERANCE * value:
                return name
        for value in _SMALL_INTEGERS:
            if abs(probe - value) <= _TOLERANCE * value:
                return f"{value:g}"
    return None


#: LaTeX that hides a number from the patterns below. ``$1{,}73$ times too
#: small`` states the same claim as ``1,73 times too small``, and the registry
#: writes its mathematics in LaTeX, so the linter has to read both.
_LATEX_NOISE = (
    (re.compile(r"\{,\}"), ","),
    (re.compile(r"\\text\{([^}]*)\}"), r"\1"),
    (re.compile(r"\\mathrm\{([^}]*)\}"), r"\1"),
    (re.compile(r"\\[ ,;]"), " "),
    (re.compile(r"[$]"), " "),
)


def _plain(text: str) -> str:
    """The entry as prose, so a claim typeset as maths still reads as a claim."""
    for pattern, replacement in _LATEX_NOISE:
        text = pattern.sub(replacement, text)
    return text


def _candidate_ratios(body: str) -> list[tuple[str, float]]:
    """Multiplicative claims stated in an entry, as (quotation, ratio)."""
    body = _plain(body)
    found: list[tuple[str, float]] = []
    for pattern in (_TIMES, _FACTOR):
        for match in pattern.finditer(body):
            found.append((match.group(0), _number(match.group("value"))))
    for match in _DECIBEL.finditer(body):
        raw = match.group("value") or match.group("value2")
        decibels = _number(raw)
        # A dB figure can be an amplitude or a power ratio; both readings are
        # tested, because a lost glyph shows up in whichever one the entry meant.
        found.append((match.group(0), 10.0 ** (decibels / 20.0)))
        found.append((match.group(0), 10.0 ** (decibels / 10.0)))
    return found


def check_ratios(entries: list[Entry]) -> list[RatioHit]:
    """Report every multiplicative claim that matches a lost-glyph signature."""
    hits: list[RatioHit] = []
    for entry in entries:
        for quoted, ratio in _candidate_ratios(entry.body):
            if ratio <= 0.0:
                continue
            name = _match_named(ratio)
            if name is not None:
                hits.append(RatioHit(entry, quoted.strip(), ratio, name))
    return hits


def check_procedure_is_not_cited(entries: list[Entry]) -> list[str]:
    """No entry describes how its page was read.

    An entry states what a document prints and where. How the page was put in
    front of a reader is method, not evidence: it belongs in CONTRIBUTING.md,
    where a contributor needs it, and nowhere in a registry that a standards
    body may one day read. "Verified on a 900 dpi render of page 130" also
    reads as a machine's account of itself rather than as a finding someone
    stands behind, which is the opposite of what the entry is for.
    """
    problems: list[str] = []
    for entry in entries:
        found = _RESOLUTION.search(entry.body) or _RENDERING.search(entry.evidence)
        if found is None:
            continue
        problems.append(
            f"{REGISTRY.name}:{entry.line}: entry '{entry.title}' describes how "
            f"the page was read ({found.group(0).strip()!r}). Cite the document, "
            "the PDF page and the printed folio; leave the method to "
            "CONTRIBUTING.md."
        )
    return problems


def check_pages(entries: list[Entry]) -> list[str]:
    """Every entry cites the page it quotes, or is allowlisted with a reason."""
    problems: list[str] = []
    for entry in entries:
        if entry.cites_page:
            continue
        if entry.title in PAGE_CITATION_ALLOWLIST:
            continue
        where = "in its Evidence bullet" if entry.evidence else "(it has no Evidence bullet)"
        problems.append(
            f"{REGISTRY.name}:{entry.line}: entry '{entry.title}' cites no page "
            f"{where}. Add 'Verified on PDF page N (printed p. M) of "
            "<edition>.' to its Evidence bullet, or list it in "
            "PAGE_CITATION_ALLOWLIST with a reason."
        )
    return problems


def check_allowlist(entries: list[Entry]) -> list[str]:
    """The allowlist must not outlive the entries it excuses."""
    titles = {entry.title for entry in entries}
    stale = sorted(set(PAGE_CITATION_ALLOWLIST) - titles)
    cleared = sorted(
        title
        for entry in entries
        if entry.cites_page and (title := entry.title) in PAGE_CITATION_ALLOWLIST
    )
    problems = [
        f"PAGE_CITATION_ALLOWLIST names '{title}', which is not an entry of "
        f"{REGISTRY.name} (renamed or removed?)."
        for title in stale
    ]
    problems += [
        f"PAGE_CITATION_ALLOWLIST still names '{title}', which now cites its page. "
        "Delete the line: the allowlist is meant to shrink."
        for title in cleared
    ]
    # A blank reason is an entry excused by nobody. The point of the allowlist
    # is that someone said, in writing, what the claim rests on instead of a
    # page reading; an empty string passes every check while saying nothing.
    problems += [
        f"PAGE_CITATION_ALLOWLIST excuses '{title}' with a blank reason. State what "
        "the claim rests on instead of a page reading."
        for title, reason in sorted(PAGE_CITATION_ALLOWLIST.items())
        if not reason.strip()
    ]
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--ratios",
        action="store_true",
        help="report every lost-glyph ratio signature and exit, without "
        "running the page-citation check",
    )
    args = parser.parse_args(argv)

    markdown = REGISTRY.read_text(encoding="utf-8")
    entries = parse_entries(markdown)
    if not entries:
        print(f"error: no entries parsed from {REGISTRY}", file=sys.stderr)
        return 1

    hits = check_ratios(entries)
    if args.ratios:
        for hit in hits:
            covered = "page cited" if hit.entry.cites_page else "NO PAGE CITED"
            print(
                f"{REGISTRY.name}:{hit.entry.line}: {hit.entry.title}\n"
                f"    '{hit.quoted}' -> ratio {hit.ratio:.6g} ~ {hit.name} "
                f"({covered})"
            )
        print(f"{len(hits)} ratio signature(s) over {len(entries)} entries.")
        return 0

    problems = (
        check_pages(entries)
        + check_allowlist(entries)
        + check_procedure_is_not_cited(entries)
    )
    problems += [
        f"{REGISTRY.name}:{hit.entry.line}: entry '{hit.entry.title}' states "
        f"'{hit.quoted}', a ratio of {hit.ratio:.6g} which is within "
        f"{_TOLERANCE:.1%} of {hit.name}. That is the signature of a glyph lost "
        "in extraction rather than an author's error, so the entry must cite "
        "the page it read. See CONTRIBUTING.md, 'Filing an errata entry'."
        for hit in hits
        if not hit.entry.cites_page
    ]

    for problem in problems:
        print(f"::error file=docs/ERRATA.md::{problem}" if _in_ci() else problem)
    if problems:
        print(f"\n{len(problems)} problem(s) in {REGISTRY}.", file=sys.stderr)
        return 1
    flagged = sum(1 for hit in hits if hit.entry.cites_page)
    print(
        f"{len(entries)} errata entries checked: "
        f"{len(entries) - len(PAGE_CITATION_ALLOWLIST)} cite their page, "
        f"{len(PAGE_CITATION_ALLOWLIST)} are allowlisted, "
        f"{flagged} lost-glyph ratio signature(s) are covered by a page citation."
    )
    return 0


def _in_ci() -> bool:
    import os

    return os.environ.get("GITHUB_ACTIONS") == "true"


if __name__ == "__main__":
    raise SystemExit(main())
