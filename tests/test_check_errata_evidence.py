#  Copyright (c) 2026. Jose Manuel Requena Plens
"""The errata evidence gate, exercised on the entry that made it necessary.

``scripts/check_errata_evidence.py`` exists because an entry was drafted that
accused an author of printing ``2/(3 pi)`` where the page prints
``2/(sqrt(3) pi)``: three text extractors had eaten the radical and agreed with
each other. It cited no page and stated its own ratio twice, as "1,73 times
too small" and as "a fixed 2,4 dB offset". Both tells are reproduced below,
together with the shapes a passing entry has to have.
"""

from __future__ import annotations

import pathlib
import sys

import pytest

_SCRIPTS = str(pathlib.Path(__file__).resolve().parent.parent / "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import check_errata_evidence as cee

#: The withdrawn entry, reduced to the two claims that give it away.
_WITHDRAWN = """
## Vigran, Building Acoustics (2008), Eq. (8.46)

- **Location:** Eq. (8.46).
- **The print:** the prefactor is printed as 2/(3 pi).
- **The problem:** the derivation gives 2/(sqrt(3) pi), so the printed
  constant is 1,73 times too small and the model carries a fixed 2,4 dB
  offset.
- **Evidence:** the printed text extracts as `2=3p`.
- **Library behaviour:** implements the derived form.
- **Status:** unreported.
"""

_WITH_PAGE = """
## Some standard, Formula (1)

- **Location:** Formula (1).
- **The print:** `A = 0,32 V`.
- **The problem:** the constant is a factor of 10 out.
- **Evidence:** direct algebra. Verified on PDF page 8 (printed p. 6) of Some standard:2020.
- **Status:** unreported.
"""

#: A citation-shaped string outside the Evidence bullet is not evidence.
_PAGE_IN_THE_WRONG_BULLET = """
## Some standard, Formula (1)

- **Location:** Formula (1).
- **The print:** `A = 0,32 V`, on PDF page 8 (printed p. 6) of Some standard:2020.
- **Evidence:** direct algebra.
- **Status:** unreported.
"""

#: The procedure, which the registry does not carry.
_WITH_PROCEDURE = """
## Some standard, Formula (1)

- **Location:** Formula (1).
- **The print:** `A = 0,32 V`.
- **Evidence:** verified on a 900 dpi render of PDF page 8 (printed p. 6) of Some standard:2020.
- **Status:** unreported.
"""

_NO_RATIO_NO_PAGE = """
## Some standard, Clause 4 (a broken cross-reference)

- **Location:** Clause 4.
- **The print:** "see Clause 0".
- **The problem:** Clause 0 does not exist.
- **Evidence:** the clause listing of the standard itself.
- **Status:** unreported.
"""


def _entries(markdown: str) -> list[cee.Entry]:
    return cee.parse_entries(markdown)


def test_the_withdrawn_entry_trips_both_tells() -> None:
    """"1,73 times too small" and "2,4 dB offset" are sqrt(3) twice over."""
    entries = _entries(_WITHDRAWN)
    assert len(entries) == 1
    assert not entries[0].cites_page
    hits = cee.check_ratios(entries)
    names = sorted({hit.name for hit in hits})
    assert names == ["sqrt(3)"]
    assert len(hits) >= 2, "both the 'times' claim and the dB offset must flag"
    quoted = " ".join(" ".join(hit.quoted.split()) for hit in hits)
    assert "1,73 times too small" in quoted
    assert "2,4 dB offset" in quoted


def test_the_withdrawn_entry_fails_the_page_check() -> None:
    problems = cee.check_pages(_entries(_WITHDRAWN))
    assert len(problems) == 1
    assert "cites no page" in problems[0]


def test_a_complete_page_citation_satisfies_the_check() -> None:
    entries = _entries(_WITH_PAGE)
    assert entries[0].cites_page
    assert cee.check_pages(entries) == []


def test_a_page_run_is_a_citation_too() -> None:
    """Several entries quote a table that spans two pages."""
    spanning = _WITH_PAGE.replace(
        "PDF page 8 (printed p. 6)", "PDF pages 8 and 9 (printed pp. 6 and 7)"
    )
    assert _entries(spanning)[0].cites_page


@pytest.mark.parametrize(
    ("field", "without"),
    [
        ("the PDF page index", "Verified on printed p. 6 of Some standard:2020."),
        ("the printed folio", "Verified on PDF page 8 of Some standard:2020."),
        ("the edition", "Verified on PDF page 8 (printed p. 6)."),
        ("both", "Verified against the source document."),
    ],
)
def test_an_incomplete_citation_does_not_satisfy_the_check(
    field: str, without: str
) -> None:
    """All three parts are required: the page index resolves in any copy of the
    edition, the printed folio is what a reader holding the paper looks for,
    and without the designation neither resolves in any library."""
    incomplete = _WITH_PAGE.replace(
        "Verified on PDF page 8 (printed p. 6) of Some standard:2020.", without
    )
    assert incomplete != _WITH_PAGE, "the fixture no longer contains that sentence"
    assert not _entries(incomplete)[0].cites_page, field


def test_a_citation_outside_the_evidence_bullet_does_not_count() -> None:
    """Only the Evidence bullet is read.

    A citation-shaped string quoted in "The print" or narrated in "The problem"
    would otherwise let an entry through with an Evidence bullet that still
    rests on an extraction.
    """
    entries = _entries(_PAGE_IN_THE_WRONG_BULLET)
    assert "PDF page 8" in entries[0].body
    assert "PDF page 8" not in entries[0].evidence
    assert not entries[0].cites_page
    problems = cee.check_pages(entries)
    assert len(problems) == 1
    assert "cites no page in its Evidence bullet" in problems[0]


def test_the_procedure_is_rejected() -> None:
    """How the page was read belongs in CONTRIBUTING.md, not in the registry.

    The entry is otherwise perfect: it cites the document, the PDF page and the
    printed folio, so every other check passes and this one has to be what
    fails.
    """
    entries = _entries(_WITH_PROCEDURE)
    assert entries[0].cites_page
    assert cee.check_pages(entries) == []
    problems = cee.check_procedure_is_not_cited(entries)
    assert len(problems) == 1
    assert "900 dpi" in problems[0]


@pytest.mark.parametrize(
    "procedure",
    [
        "a 1200 dpi render",
        "rendered at 600 DPI",
        "the rendering of page 8",
        "300dpi",
        "a rendered page image",
    ],
)
def test_every_wording_of_the_procedure_is_rejected(procedure: str) -> None:
    markdown = f"""
## Probe

- **Evidence:** read on {procedure}, PDF page 8 (printed p. 6) of X:2020.
- **Status:** unreported.
"""
    assert cee.check_procedure_is_not_cited(_entries(markdown))


def test_the_ordinary_verb_is_not_the_procedure() -> None:
    """"Renders" is a word an entry may need for what a misprint does.

    The check exists to keep an account of the reading out of the registry, not
    to ban an English verb: an entry that says a dropped exponent renders a
    formula inconsistent is describing the defect, which is its whole job.
    """
    markdown = """
## Probe

- **The problem:** the dropped exponent renders Formula (7) inconsistent.
- **Evidence:** Verified on PDF page 8 (printed p. 6) of X:2020.
- **Status:** unreported.
"""
    assert cee.check_procedure_is_not_cited(_entries(markdown)) == []


def test_a_clean_citation_carries_no_procedure() -> None:
    assert cee.check_procedure_is_not_cited(_entries(_WITH_PAGE)) == []


def test_an_entry_with_no_multiplicative_claim_is_not_flagged() -> None:
    assert cee.check_ratios(_entries(_NO_RATIO_NO_PAGE)) == []


@pytest.mark.parametrize(
    ("claim", "expected"),
    [
        ("1,41 times too small", "sqrt(2)"),
        ("1,73 times too small", "sqrt(3)"),
        ("3,14 times too large", "pi"),
        ("6,28 times too large", "2 pi"),
        ("0,707 times too small", "1/sqrt(2)"),
        ("0,693 times too small", "ln 2"),
        ("a factor of 10", "10"),
        # A dB figure is tested as both an amplitude and a power ratio, so
        # 3,01 dB names sqrt(2) on one reading and 2 on the other.
        ("3,01 dB offset", "sqrt(2)"),
        ("6,02 dB offset", "2"),
        ("10 dB offset", "10"),
        ("off by 4,77 dB", "3"),
    ],
)
def test_each_lost_glyph_signature_is_recognised(claim: str, expected: str) -> None:
    markdown = f"""
## Probe

- **The problem:** the print is {claim}.
- **Status:** unreported.
"""
    hits = cee.check_ratios(_entries(markdown))
    assert expected in [hit.name for hit in hits], claim


@pytest.mark.parametrize(
    "claim",
    [
        "$1{,}73$ times too small",
        "a fixed $2{,}4\\ \\text{dB}$ offset",
        "off by $4{,}77\\ \\text{dB}$",
    ],
)
def test_a_claim_typeset_as_maths_is_still_a_claim(claim: str) -> None:
    """The registry writes its mathematics in LaTeX, and the linter reads it.

    `$1{,}73$ times too small` states exactly what `1,73 times too small`
    states, so typesetting a claim must not be a way to hide it from the check
    that exists to catch a lost glyph.
    """
    markdown = f"""
## Probe

- **The problem:** the printed value is {claim}.
- **Status:** unreported.
"""
    assert cee.check_ratios(_entries(markdown)), claim


def test_a_formula_term_is_not_a_ratio_claim() -> None:
    """"The prefactor 4 N h" is a term of an equation, not a factor of four."""
    markdown = """
## Probe

- **The problem:** the prefactor $4 N h_1 c_{L1}/(\\sqrt{3}\\,\\omega S_1)$ is
  already dimensionless.
- **Status:** unreported.
"""
    assert cee.check_ratios(_entries(markdown)) == []


def test_an_honest_ratio_is_not_flagged() -> None:
    """A ratio that is not a lost-glyph signature passes untouched."""
    markdown = """
## Probe

- **The problem:** the printed value is 1,27 times too small.
- **Status:** unreported.
"""
    assert cee.check_ratios(_entries(markdown)) == []


def test_the_registry_itself_passes() -> None:
    """The committed docs/ERRATA.md satisfies the gate CI runs."""
    assert cee.main([]) == 0


def test_the_allowlist_only_names_real_entries() -> None:
    markdown = cee.REGISTRY.read_text(encoding="utf-8")
    assert cee.check_allowlist(cee.parse_entries(markdown)) == []


def test_a_blank_allowlist_reason_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """An excuse nobody wrote down is not an excuse.

    Every other check passes for a whitespace-only reason: the title is real,
    the entry cites no page, and nothing compares the value. That is the
    whole point of the allowlist, so it has to be the one thing that fails.

    The fixture is local rather than borrowed from the shipped allowlist,
    which is meant to shrink and may legitimately end up empty.
    """
    markdown = """
## Probe standard, Clause 1

- **The problem:** the printed coefficient cannot be right.
- **Evidence:** a recomputation, not a character reading.
- **Status:** unreported.
"""
    entries = _entries(markdown)
    title = entries[0].title
    monkeypatch.setattr(cee, "PAGE_CITATION_ALLOWLIST", {title: "   "})
    problems = cee.check_allowlist(entries)
    assert any("blank reason" in problem for problem in problems), problems
    assert any(title in problem for problem in problems), problems


def test_a_stated_allowlist_reason_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    """The same entry with a real reason raises nothing."""
    markdown = """
## Probe standard, Clause 1

- **The problem:** the printed coefficient cannot be right.
- **Evidence:** a recomputation, not a character reading.
- **Status:** unreported.
"""
    entries = _entries(markdown)
    monkeypatch.setattr(
        cee,
        "PAGE_CITATION_ALLOWLIST",
        {entries[0].title: "the claim is a recomputation"},
    )
    assert cee.check_allowlist(entries) == []
