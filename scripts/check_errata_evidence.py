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
three were wrong, and it was caught in review. Hence :func:`check_renders`: every entry must record a render
of the page it quotes (source file, PDF page index, printed folio and dpi), or
be listed in :data:`RENDER_ALLOWLIST` with a reason.

**The arithmetic tells.** When a glyph is lost the printed and derived values
differ by the lost factor, so their ratio lands on one of a small set of
constants. That entry announced its own ratio twice, as "1,73 times
too small" and as "a fixed 2,4 dB offset" -- ``sqrt(3)`` and
``20 lg sqrt(3)``. Hence :func:`check_ratios`: any multiplicative claim in an
entry whose value sits within 0,5 % of ``sqrt(2)``, ``sqrt(3)``, ``pi``,
``2 pi``, ``1/sqrt(2)``, ``ln 2`` or a small integer is reported, and is an
error unless that entry cites a render. A genuine factor-of-ten misprint trips
the test too; that is the point, since the way to clear it is to read the page
as an image.

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

#: Entries filed before the render rule, or whose evidence cannot be a render.
#: Each value is the reason. **This list is meant to shrink**: clearing an
#: entry means reading its page as an image and recording the render in its
#: Evidence bullet, not adding a new line here. A ratio hit is never excused by
#: membership of this list.
RENDER_ALLOWLIST: dict[str, str] = {
    'ISO 12354-1:2017, Table D.1 (1 600 Hz covered by two rows)':
        'read on a 600 dpi render of the printed table, which showed two separately ruled rows rather than an extraction artefact; the structured citation predates this check',
    'Vigran, Building Acoustics (2008), Figure 8.37 caption (carpet stiffness exponent)':
        'read on a 600 dpi render of the caption and of p. 321; the structured citation predates this check',
    'Vigran, Building Acoustics (2008), Eq. (9.18) (receiving-side coefficient)':
        'the claim is a derivation of the receiving-side integral, not a character reading; the printed form is quoted as printed and is not in dispute',
    'Commission Directive (EU) 2015/996, Annex II 2.2.1 (octave-band range of the road source)':
        'the claim is that the clause contradicts its own Appendix F tables, settled by the corrigendum text fetched from the Publications Office rather than by a character reading',
    'Directive (EU) 2015/996, Annex II 2.3.2 (roughness conversion in km/h)':
        'read on a rendered page image of the Official Journal; the structured citation predates this check',
    'Directive (EU) 2015/996, Appendix G, Table G-1, second table (wrong symbol)':
        'read on a rendered page image of the Official Journal; the structured citation predates this check',
    'Directive (EU) 2015/996, Appendix G, Table G-5, 6 350 Hz row (50 dB notch)':
        'read on a rendered page image of the Official Journal and corroborated by the amendment that restores the two values; the structured citation predates this check',
    'Directive (EU) 2015/996, Appendix G, band and wavelength labels':
        'read on a rendered page image of the Official Journal; the structured citation predates this check',
    'Directive (EU) 2015/996, Annex II 2.3.2, curve squeal (unassigned endpoints)':
        'the claim is that two printed open intervals leave R = 300 m and 500 m unassigned, a reading of the interval endpoints rather than of a glyph',
    'Allard & Atalla, Propagation of Sound in Porous Media 2e (2009), Eq. (6.85)':
        "settled by the book's own Eq. (6.80) on the facing page, a derivation rather than a character reading; independently re-derived in review",
    'Allard & Atalla 2e (2009), Eq. (11.48) and Table 11.1 (poroelastic layer)':
        'read on a render in review, which also established the chapter page offsets; the structured citation predates this check',
    'Allard & Atalla 2e (2009), Sect. 6.6.3 (thickness of the second sample)':
        'the claim is that two printed statements of the same thickness disagree, a comparison of two sentences rather than a glyph reading',
    'Allard & Atalla 2e (2009), Sect. 6.5.4 (the frame-borne velocity ratio)':
        "the claim is that a printed sentence says modulus where the printed value is the real part, settled by computing both from the book's own Table 6.1 inputs",

    "ISO 12354-1:2017 Table L.3 / ISO 12354-2:2017 Table G.3 (perimeter sums)":
        "the claim is about five recomputed sums, not about printed characters",
    "ISO 12354-1:2017 Table L.3 / ISO 12354-2:2017 Table G.3 (external wall ηint)":
        "the claim is a recomputation of Formula (C.1), not a character reading",
    "ISO 12354-1:2017, Table L.4 (second path block labelled 2d)":
        "the claim is that three recomputed columns identify a different path",
    "ISO 12354-2:2017, Table G.1 (50 Hz to 80 Hz flanking columns)":
        "the claim is a band-by-band recomputation of two printed tables",
    "ISO 10848-1:2006, Clause 8.1.1, Formula (20) (spurious π in the critical frequency)":
        "pre-render entry; corrected upstream in the 2017 edition",
    "ISO 12999-1:2020, Table 4 (missing 500 Hz row)":
        "the claim is that a row is absent, which a render cannot show",
    "ISO 10052:2021, Table 4 volume-range header":
        "pre-render entry",
    "ECMA-418-2:2025 (4th edition), clause 5.1.5.2 (last block index)":
        "pre-render entry",
    "ECMA-418-2:2025 (4th edition), clause 9.1.4, Formula (127) (HSA kernel phase)":
        "pre-render entry",
    "ECMA-418-2:2025 (4th edition), clause 9.1.5, Formula (144) (bin offset)":
        "pre-render entry",
    "ECMA-418-2:2025 (4th edition), clause 9.1.7 (units of the fine-tuning constants)":
        "pre-render entry",
    "ECMA-418-2:2025 (4th edition), clause 9 introduction (broken cross-reference)":
        "pre-render entry",
    "UNE-EN 61043:1999, clause 6.1 (class 2 frequency range dropped in translation)":
        "pre-render entry; the claim is a comparison of two prose sentences",
    "ISO/PAS 1996-3:2022, Clause 5 (cross-references of r and d)":
        "pre-render entry; the local copy is a scan without a text layer",
    "ISO 9613-2:1996, Table 2 (15 °C / 80 % / 1 kHz cell)":
        "pre-render entry",
    "ANSI S3.5-1997, Annex C worked examples (official WG S3-79 errata)":
        "the standard is not held locally; the entry says so in a leading notice",
    "ANSI S3.5-1997, captions of Tables 1 to 4 (official WG S3-79 erratum)":
        "the standard is not held locally; the entry says so in a leading notice",
    "NORAH2 rotorcraft guidance SC01.D1.5d (EASA.2020.FC.06), Eq. (21)":
        "pre-render entry",
    "NORAH2 rotorcraft guidance SC01.D1.5d (EASA.2020.FC.06), §A.3.1 triangulation":
        "the claim is about shipped lookup tables, not about printed characters",
    "NORAH2 rotorcraft guidance SC01.D1.5d (EASA.2020.FC.06), Eq. (46)":
        "pre-render entry",
    "RANDI 3.1 Physics Description (NRL, Breeding et al.), Table 2":
        "pre-render entry",
    "Osses, García & Kohlrausch (2016), fluctuation-strength model, Eq. (3)":
        "pre-render entry",
    "Jiménez, Groby, Pagneux & Romero-García (2017), Appl. Sci. 7(6), 618, Eqs. (7)-(8)":
        "pre-render entry",
    "Jiménez et al. (2017), Appl. Sci. 7(6), 618 / Sci. Rep. 7, 5389, slit-radiation term":
        "pre-render entry",
    "Attenborough & Van Renterghem, Predicting Outdoor Sound 2e (2021), Table 5.1":
        "pre-render entry",
    "Attenborough & Van Renterghem, Predicting Outdoor Sound 2e (2021), Eq. (5.13)":
        "pre-render entry",
    "Bies, Hansen & Howard, Engineering Noise Control 5e (2017), Eq. (8.141)":
        "pre-render entry",
    "Norton & Karczub, Fundamentals of Noise and Vibration Analysis for Engineers 2e "
    "(2003), Eq. (6.56)":
        "pre-render entry",
    "Norton & Karczub 2e (2003), problem 6.13 answer (eta_21 column)":
        "pre-render entry",
    "Norton & Karczub 2e (2003), problem 6.10 (platform area)":
        "pre-render entry",
    "Real Decreto 1367/2007, Annex IV A.3.3 (Kf and Ki threshold tables)":
        "pre-render entry; the source is the BOE consolidated HTML text",
    "NMFS (2024) Updated Technical Guidance v3.0, Table 5 / Table ES2 (otariid C)":
        "pre-render entry; the issuing body documents the defect in its own footnote",
}

#: A complete render citation: the file, the PDF page index, the printed folio
#: and the resolution. All four are required, in that order, in one sentence.
RENDER = re.compile(
    r"`[^`]+\.pdf`[^.]*?PDF\s+page\s+\d+[^.]*?printed\s+p{1,2}\.\s*[\d\s,and-]+"
    r"[^.]*?\d+\s*dpi",
    re.IGNORECASE | re.DOTALL,
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
#: "a factor of 1,73", "by a factor 10", "a factor-16 error".
_FACTOR = re.compile(
    r"factor[\s-]+(?:of\s+)?(?P<value>\d+[.,]\d+|\d+)",
    re.IGNORECASE,
)
#: "a fixed 2,4 dB offset", "2,4 dB too high", "off by 2,4 dB".
_DECIBEL = re.compile(
    r"(?P<value>\d+[.,]\d+|\d+)\s*dB\s+(?:offset|bias|too\s+(?:high|low))"
    r"|off\s+by\s+(?P<value2>\d+[.,]\d+|\d+)\s*dB",
    re.IGNORECASE,
)


#: The start of the bullet the render has to live in. Anything quoted in "The
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
    def cites_render(self) -> bool:
        return RENDER.search(self.evidence) is not None


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


def _candidate_ratios(body: str) -> list[tuple[str, float]]:
    """Multiplicative claims stated in an entry, as (quotation, ratio)."""
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


def check_renders(entries: list[Entry]) -> list[str]:
    """Every entry names a render, or is allowlisted with a reason."""
    problems: list[str] = []
    for entry in entries:
        if entry.cites_render:
            continue
        if entry.title in RENDER_ALLOWLIST:
            continue
        where = "in its Evidence bullet" if entry.evidence else "(it has no Evidence bullet)"
        problems.append(
            f"{REGISTRY.name}:{entry.line}: entry '{entry.title}' cites no page "
            f"render {where}. Add 'Render: `plan/<file>.pdf`, PDF page N, "
            "printed p. M, D dpi.' to its Evidence bullet, or list it in "
            "RENDER_ALLOWLIST with a reason."
        )
    return problems


def check_allowlist(entries: list[Entry]) -> list[str]:
    """The allowlist must not outlive the entries it excuses."""
    titles = {entry.title for entry in entries}
    stale = sorted(set(RENDER_ALLOWLIST) - titles)
    cleared = sorted(
        title
        for entry in entries
        if entry.cites_render and (title := entry.title) in RENDER_ALLOWLIST
    )
    problems = [
        f"RENDER_ALLOWLIST names '{title}', which is not an entry of "
        f"{REGISTRY.name} (renamed or removed?)."
        for title in stale
    ]
    problems += [
        f"RENDER_ALLOWLIST still names '{title}', which now cites a render. "
        "Delete the line: the allowlist is meant to shrink."
        for title in cleared
    ]
    # A blank reason is an entry excused by nobody. The point of the allowlist
    # is that someone said, in writing, what the claim rests on instead of a
    # render; an empty string passes every other check while saying nothing.
    problems += [
        f"RENDER_ALLOWLIST excuses '{title}' with a blank reason. State what "
        "the claim rests on instead of a render."
        for title, reason in sorted(RENDER_ALLOWLIST.items())
        if not reason.strip()
    ]
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--ratios",
        action="store_true",
        help="report every lost-glyph ratio signature and exit, without "
        "running the render-evidence check",
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
            covered = "render cited" if hit.entry.cites_render else "NO RENDER"
            print(
                f"{REGISTRY.name}:{hit.entry.line}: {hit.entry.title}\n"
                f"    '{hit.quoted}' -> ratio {hit.ratio:.6g} ~ {hit.name} "
                f"({covered})"
            )
        print(f"{len(hits)} ratio signature(s) over {len(entries)} entries.")
        return 0

    problems = check_renders(entries) + check_allowlist(entries)
    problems += [
        f"{REGISTRY.name}:{hit.entry.line}: entry '{hit.entry.title}' states "
        f"'{hit.quoted}', a ratio of {hit.ratio:.6g} which is within "
        f"{_TOLERANCE:.1%} of {hit.name}. That is the signature of a glyph lost "
        "in extraction rather than an author's error, so the entry must cite a "
        "render of the page. See CONTRIBUTING.md, 'Filing an errata entry'."
        for hit in hits
        if not hit.entry.cites_render
    ]

    for problem in problems:
        print(f"::error file=docs/ERRATA.md::{problem}" if _in_ci() else problem)
    if problems:
        print(f"\n{len(problems)} problem(s) in {REGISTRY}.", file=sys.stderr)
        return 1
    flagged = sum(1 for hit in hits if hit.entry.cites_render)
    print(
        f"{len(entries)} errata entries checked: "
        f"{len(entries) - len(RENDER_ALLOWLIST)} cite a page render, "
        f"{len(RENDER_ALLOWLIST)} are allowlisted, "
        f"{flagged} lost-glyph ratio signature(s) are covered by a render."
    )
    return 0


def _in_ci() -> bool:
    import os

    return os.environ.get("GITHUB_ACTIONS") == "true"


if __name__ == "__main__":
    raise SystemExit(main())
