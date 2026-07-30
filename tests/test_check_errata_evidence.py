#  Copyright (c) 2026. Jose M. Requena-Plens
"""The errata evidence gate, exercised on the entry that made it necessary.

``scripts/check_errata_evidence.py`` exists because an entry was drafted that
accused an author of printing ``2/(3 pi)`` where the page prints
``2/(sqrt(3) pi)``: three text extractors had eaten the radical and agreed with
each other. It cited no page render and stated its own ratio twice, as
"1,73 times too small" and as "a fixed 2,4 dB offset". Both tells are
reproduced below, together with the shapes a passing entry has to have.
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

_WITH_RENDER = """
## Some standard, Formula (1)

- **Location:** Formula (1).
- **The print:** `A = 0,32 V`.
- **The problem:** the constant is a factor of 10 out.
- **Evidence:** direct algebra. Render: `plan/some-standard.pdf`, PDF page 8, printed p. 6, 600 dpi.
- **Status:** unreported.
"""

#: A render-shaped string outside the Evidence bullet is not evidence.
_RENDER_IN_THE_WRONG_BULLET = """
## Some standard, Formula (1)

- **Location:** Formula (1).
- **The print:** `A = 0,32 V`, as `plan/some-standard.pdf`, PDF page 8, printed p. 6, 600 dpi shows.
- **Evidence:** direct algebra.
- **Status:** unreported.
"""

_NO_RATIO_NO_RENDER = """
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
    assert not entries[0].cites_render
    hits = cee.check_ratios(entries)
    names = sorted({hit.name for hit in hits})
    assert names == ["sqrt(3)"]
    assert len(hits) >= 2, "both the 'times' claim and the dB offset must flag"
    quoted = " ".join(" ".join(hit.quoted.split()) for hit in hits)
    assert "1,73 times too small" in quoted
    assert "2,4 dB offset" in quoted


def test_the_withdrawn_entry_fails_the_render_check() -> None:
    problems = cee.check_renders(_entries(_WITHDRAWN))
    assert len(problems) == 1
    assert "cites no page render" in problems[0]


def test_a_complete_render_citation_satisfies_the_check() -> None:
    entries = _entries(_WITH_RENDER)
    assert entries[0].cites_render
    assert cee.check_renders(entries) == []


@pytest.mark.parametrize(
    "missing",
    [
        "`plan/some-standard.pdf`, ",  # no file
        "PDF page 8, ",  # no PDF page index
        "printed p. 6, ",  # no printed folio
        ", 600 dpi",  # no resolution
    ],
)
def test_an_incomplete_render_citation_does_not_satisfy_the_check(
    missing: str,
) -> None:
    """All four fields are required; three of four is not a citation."""
    incomplete = _WITH_RENDER.replace(missing, "")
    assert incomplete != _WITH_RENDER, "the fixture no longer contains that field"
    assert not _entries(incomplete)[0].cites_render


def test_a_render_outside_the_evidence_bullet_does_not_count() -> None:
    """Only the Evidence bullet is read.

    A render-shaped string quoted in "The print" or narrated in "The problem"
    would otherwise let an entry through with an Evidence bullet that still
    rests on an extraction.
    """
    entries = _entries(_RENDER_IN_THE_WRONG_BULLET)
    assert "PDF page 8" in entries[0].body
    assert "PDF page 8" not in entries[0].evidence
    assert not entries[0].cites_render
    problems = cee.check_renders(entries)
    assert len(problems) == 1
    assert "cites no page render in its Evidence bullet" in problems[0]


def test_an_entry_with_no_multiplicative_claim_is_not_flagged() -> None:
    assert cee.check_ratios(_entries(_NO_RATIO_NO_RENDER)) == []


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
