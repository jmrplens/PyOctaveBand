#  Copyright (c) 2026. Jose Manuel Requena Plens
"""The committed conformance indicators say what the artefact says.

Six SVGs under ``.github/badges`` are generated and committed, so the same
three properties the artefact itself has to hold apply here: they must be a
function of the tree alone, they must carry the tree's own numbers rather than
a figure typed in beside them, and the documents that point at them must point
at files that exist.

The guards below are the ones that would otherwise fail somewhere far away. A
non-deterministic byte fails as a red ``conformance`` job on an unrelated pull
request. A hard-coded count fails as a banner that says 566 for ever. A renamed
badge fails as a broken image in a rendered document and nowhere else, because
the old file is still tracked, still unchanged, and still green. And a colour
chosen for one theme fails only for the half of the readership in the other,
which is why the contrast is measured here and not asserted in a comment.
"""

from __future__ import annotations

import pathlib
import re
import subprocess
import sys
import xml.etree.ElementTree as ET

import pytest

_SCRIPTS = str(pathlib.Path(__file__).resolve().parent.parent / "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import check_figure_contrast as cfc
import conformance_badges as cb
from conformance import artifact
from conformance.registry import Verdict

_ROOT = pathlib.Path(__file__).resolve().parent.parent

#: Every ground a mark is actually drawn on, read off the surface rather than
#: named from a theme. GitHub stripes its Markdown tables
#: (``tr:nth-child(2n)`` takes ``--bgColor-muted``) and forces table images
#: transparent, so half of the report's 566 rows sit on the stripe and not on
#: the canvas; the stripes were missing here entirely, which is how the one
#: ground below 3:1 went unmeasured. The site's two page colours and two card
#: colours come from the computed style of the built page: Starlight's
#: ``--sl-color-black`` is overridden in ``site/src/styles/theme.css``, so the
#: #17181c this tuple used to name is not a ground anywhere in this project.
_GROUNDS = (
    "#ffffff",  # GitHub light canvas, and the site's light page
    "#f6f8fa",  # GitHub light table stripe
    "#0d1117",  # GitHub dark canvas
    "#151b23",  # GitHub dark table stripe
    "#22272e",  # GitHub dimmed canvas
    "#2d333b",  # GitHub dimmed table stripe
    "#0d1114",  # the site's dark page
    "#e7edf1",  # the site's light summary card
    "#1a2126",  # the site's dark summary card
)

#: The one ground in :data:`_GROUNDS` no fill clears 3:1 on. Named here so the
#: exception is a value one test pins rather than a silence in another.
_DIMMED_STRIPE = "#2d333b"

#: The four verdict fills, as the drawn marks use them.
_FILLS = ("#1f883d", "#da3633", "#1f6feb", "#6e7781")

#: The pictographic blocks, and nothing else: emoji and dingbats (U+2600-27BF,
#: which is where the tick, the cross and the warning sign live), the symbols
#: and arrows block that holds the stars (U+2B00-2BFF), the supplementary
#: planes (U+1F000-1FAFF), and the variation selector that turns a character
#: into its emoji presentation.
#:
#: Deliberately narrower than "everything above U+2190". A conformance report
#: is full of mathematics - the quantities in it print √, ≤, ≫, ⟨⟩, ∂, ∞ - and
#: a class that swept from the arrows to the end of the symbol blocks reported
#: a square root as an emoji.
EMOJI = re.compile("[\U0001f000-\U0001faff☀-➿⬀-⯿️]")

#: A whole artefact, cut down to what the indicators read. Counts unlike the
#: real ones on purpose: a generator that hard-codes 566 passes against the
#: committed document and fails here.
FIXTURE = {
    "counts": {
        "checks": 7,
        "passing": 4,
        "failing": 3,
        "domains": 2,
        "standards": 5,
        "citations": 5,
        "designations": 5,
        "sources": 0,
    }
}


def _rgb(colour: str) -> tuple[float, float, float]:
    parsed = cfc._parse_color(colour)
    assert parsed is not None, colour
    return parsed


def _contrast(colour: str, ground: str) -> float:
    return cfc.contrast_ratio(_rgb(colour), _rgb(ground))


def _luminance(colour: str) -> float:
    return cfc._relative_luminance(_rgb(colour))


def _committed(name: str) -> str:
    return (cb.BADGE_DIR / name).read_text(encoding="utf8")


# --------------------------------------------------------------------------
# The manifest covers the vocabulary, and the files match the manifest
# --------------------------------------------------------------------------


def test_there_is_one_mark_for_every_verdict_a_check_can_carry() -> None:
    """A tick and a cross could only say two things; four verdicts need four.

    Pinned against the enum rather than against a list written twice, so
    adding a fifth verdict fails here instead of rendering as a blank cell.
    """
    assert {mark.verdict for mark in cb.MARKS} == {str(v) for v in Verdict}


def test_the_directory_holds_exactly_what_the_manifest_names() -> None:
    """An orphan is as bad as a missing file: it is what a rename leaves.

    The old file stays tracked and unchanged, so the working tree is clean and
    CI is green while the document points at a name nothing writes any more.
    """
    on_disk = {path.name for path in cb.BADGE_DIR.glob("*.svg")}
    assert on_disk == set(cb.asset_names())


def test_the_committed_files_are_what_the_generator_renders() -> None:
    """The staleness gate, in the suite rather than only in CI."""
    for name, source in cb.assets().items():
        assert _committed(name) == source, f"{name} is stale; run `make conformance`"


def test_every_reference_to_a_badge_resolves_to_a_generated_file() -> None:
    """No document may point at a badge this module does not write.

    Scanned across the tracked tree rather than a list of known consumers, so
    a badge cited from a page nobody thought of is still covered. This is the
    guard that turns a rename into a failing test instead of a 404 in a
    rendered page.
    """
    tracked = subprocess.run(  # noqa: S603
        ["git", "-C", str(_ROOT), "ls-files", "-z"],  # noqa: S607
        capture_output=True,
        check=True,
    ).stdout.decode()
    cited = re.compile(re.escape(cb.RAW_PATH) + r"/([\w.-]+\.svg)")
    known = set(cb.asset_names())
    missing: list[str] = []
    for name in tracked.split("\0"):
        path = _ROOT / name
        if not name or path.suffix in (".svg", ".png", ".webp", ".pdf", ".ttf"):
            continue
        try:
            text = path.read_text(encoding="utf8")
        except (OSError, UnicodeDecodeError):
            continue
        missing += [
            f"{name} cites {cb.RAW_PATH}/{cited_name}"
            for cited_name in cited.findall(text)
            if cited_name not in known
        ]
    assert missing == []


# --------------------------------------------------------------------------
# Determinism
# --------------------------------------------------------------------------


def test_two_runs_write_the_same_bytes(tmp_path: pathlib.Path) -> None:
    """The whole tree is diffed by CI, so a wobbling byte reddens a build."""
    first, second = tmp_path / "a", tmp_path / "b"
    cb.write(directory=first)
    cb.write(directory=second)
    for name in cb.asset_names():
        assert (first / name).read_bytes() == (second / name).read_bytes()


def test_nothing_written_carries_a_clock_or_a_version() -> None:
    """Provenance that changes by itself is what makes a diff meaningless."""
    for source in cb.assets().values():
        assert not re.search(r"\b(19|20)\d\d-\d\d-\d\d\b", source)
        assert "generator" not in source.lower()
        assert "matplotlib" not in source.lower()


def test_the_sweep_removes_a_file_the_manifest_no_longer_names(
    tmp_path: pathlib.Path,
) -> None:
    """What makes a rename self-cleaning rather than silently additive."""
    (tmp_path / "verdict-obsolete.svg").write_text("<svg/>", encoding="utf8")
    (tmp_path / "notes.txt").write_text("kept", encoding="utf8")
    cb.write(directory=tmp_path)
    assert not (tmp_path / "verdict-obsolete.svg").exists()
    # Confined to the SVGs this module authors; nothing else in the directory
    # is its business.
    assert (tmp_path / "notes.txt").read_text(encoding="utf8") == "kept"


# --------------------------------------------------------------------------
# The numbers come from the artefact
# --------------------------------------------------------------------------


def test_the_banner_reads_its_counts_and_never_carries_its_own() -> None:
    """Rendered on a document whose counts are nothing like the real ones."""
    alt = cb.banner_alt(FIXTURE["counts"])
    assert alt == (
        "4 of 7 conformance checks pass, 3 failing, across 2 domains and 5 standards"
    )
    svg = cb.render_banner(FIXTURE["counts"], cb.LIGHT)
    root = ET.fromstring(svg)  # noqa: S314 - our own output
    assert root.get("aria-label") == alt
    # Read off the title rather than searched for as a substring: the drawn
    # figures are outlines, so "566" turns up in path coordinates by chance.
    title = root.find("{http://www.w3.org/2000/svg}title")
    assert title is not None
    assert title.text is not None
    assert "566" not in title.text


def test_the_committed_banner_states_the_committed_counts() -> None:
    counts = artifact.load()["counts"]
    assert cb.banner_alt(counts) in _committed(cb.BANNER)
    assert cb.banner_alt(counts) in _committed(cb.BANNER_DARK)


def test_a_singular_count_is_not_captioned_in_the_plural() -> None:
    one = {"checks": 1, "passing": 1, "domains": 1, "standards": 1}
    assert cb.banner_alt(one) == (
        "All 1 conformance checks pass, across 1 domain and 1 standard"
    )


def test_the_bar_shows_a_failing_share_only_when_something_fails() -> None:
    """The bar is the live part: a red notch is the whole point of drawing it."""
    all_green = dict(FIXTURE["counts"], passing=7, failing=0)
    green = cb.render_banner(all_green, cb.LIGHT)
    assert cb.LIGHT.failing not in green
    red = cb.render_banner(FIXTURE["counts"], cb.LIGHT)
    assert cb.LIGHT.failing in red


# --------------------------------------------------------------------------
# Light and dark, measured rather than asserted
# --------------------------------------------------------------------------


def _drawn_fills(source: str) -> list[str]:
    """The colours one mark paints with, minus the glyph cut out of them."""
    found = set(re.findall(r'(?:fill|stroke)="(#[0-9a-f]{6})"', source))
    return sorted(found - {"#ffffff"})


def test_every_mark_clears_three_to_one_on_every_ground_but_the_dimmed_stripe() -> None:
    """One file per mark is only defensible if one palette really works.

    The alternative is a per-theme ``<picture>`` pair per row, which measures
    +162.8 kB on the 566-row table against the +10.0 kB the reference-style
    single-file mark costs, so this test is what the size decision rests on.
    Every ground in :data:`_GROUNDS` is one the
    marks are really drawn on, GitHub's two table stripes included, because
    half of the report's rows sit on a stripe rather than on the canvas.
    """
    poor: list[str] = []
    for name, source in cb.render_marks().items():
        for colour in _drawn_fills(source):
            poor += [
                f"{name}: {colour} on {ground} is {_contrast(colour, ground):.2f}:1"
                for ground in _GROUNDS
                if ground != _DIMMED_STRIPE and _contrast(colour, ground) < 3.0
            ]
    assert poor == []


def test_the_dimmed_table_stripe_is_the_only_ground_left_short_and_stays_close() -> (
    None
):
    """The exception, pinned so it cannot quietly widen or quietly worsen.

    GitHub's dimmed theme stripes its even table rows #2d333b, where the four
    fills measure 2.75 to 2.82:1. They are left there deliberately - the next
    test shows no colour could do better - so what this one guards is that the
    shortfall stays confined to that one ground and stays within a fifth of
    the target.
    """
    shortfalls = {
        colour: _contrast(colour, _DIMMED_STRIPE)
        for source in cb.render_marks().values()
        for colour in _drawn_fills(source)
    }
    assert sorted(shortfalls) == sorted(_FILLS)
    for colour, ratio in shortfalls.items():
        assert 2.7 <= ratio < 3.0, f"{colour} on {_DIMMED_STRIPE} is {ratio:.2f}:1"


def test_no_fill_can_clear_the_dimmed_stripe_and_still_carry_its_glyph() -> None:
    """Why that exception is arithmetic and not an unfixed bug.

    Clearing 3:1 on the dimmed stripe asks for a *brighter* fill; carrying a
    white glyph at 4.5:1 asks for a darker one. The two windows do not
    overlap, so the choice is which to keep, and the glyph wins because it is
    the part of the mark that is read - and because the verdict is spelled out
    in words beside the mark on every surface that draws one.
    """
    dimmest_that_clears_the_stripe = 3.0 * (_luminance(_DIMMED_STRIPE) + 0.05) - 0.05
    brightest_that_carries_a_white_glyph = 1.05 / 4.5 - 0.05
    assert dimmest_that_clears_the_stripe > brightest_that_carries_a_white_glyph
    for fill in _FILLS:
        assert _luminance(fill) <= brightest_that_carries_a_white_glyph


def test_a_white_glyph_stays_readable_on_the_fill_under_it() -> None:
    """The glyph is judged against the fills the marks are really drawn with.

    Not against a list written beside them: an assertion about a colour the
    tree no longer uses passes for ever and guards nothing.
    """
    drawn = {
        colour
        for source in cb.render_marks().values()
        for colour in _drawn_fills(source)
    }
    assert drawn == set(_FILLS)
    for fill in sorted(drawn):
        assert _contrast("#ffffff", fill) >= 4.5


def test_each_banner_palette_reads_on_its_own_card() -> None:
    for palette in (cb.LIGHT, cb.DARK):
        assert _contrast(palette.ink, palette.card) >= 7.0
        assert _contrast(palette.muted, palette.card) >= 4.5
        assert _contrast(palette.passing, palette.card) >= 3.0
        assert _contrast(palette.failing, palette.card) >= 3.0
        assert _contrast(palette.border, palette.card) >= 1.2


def test_the_marks_ship_as_one_file_each_with_no_theme_variant() -> None:
    """The measured reason the per-row markup stays 17 characters long."""
    assert not [mark for mark in cb.MARKS if "_dark" in mark.filename]


def test_the_banner_ships_as_a_pair_on_the_naming_the_repository_pairs_on() -> None:
    """``ThemeImage.astro`` derives the dark URL by this exact substitution."""
    assert cb.BANNER_DARK == cb.BANNER.replace(".svg", "_dark.svg")
    # The drawing side and the citing side name the same two files. If the dark
    # palette's suffix ever drifted from the pairing rule, the writer's sweep
    # would delete the committed dark banner as an orphan while every document
    # went on citing it.
    assert cb.banner_name(cb.LIGHT) == cb.BANNER
    assert cb.banner_name(cb.DARK) == cb.BANNER_DARK


def test_no_asset_asks_the_reader_operating_system_what_theme_to_use() -> None:
    """An in-SVG media query follows the OS, not the GitHub theme toggle.

    A reader with GitHub set to light and a dark desktop would get the dark
    palette on a white page, which is worse than no theming at all.
    """
    for name, source in cb.assets().items():
        assert "prefers-color-scheme" not in source, name
        assert "<style" not in source, name
        assert "currentColor" not in source, name


def test_no_asset_reaches_outside_itself() -> None:
    """GitHub sandboxes an SVG in an ``<img>``; anything external never loads."""
    for name, source in cb.assets().items():
        assert "xlink:href" not in source, name
        assert "<image" not in source, name
        for url in re.findall(r'https?://[^"\s]+', source):
            assert url == "http://www.w3.org/2000/svg", f"{name} reaches {url}"


# --------------------------------------------------------------------------
# The text alternative, and no emoji
# --------------------------------------------------------------------------


def test_every_asset_says_in_words_what_it_shows() -> None:
    """An image of a tick is not a verdict to anyone who cannot see it."""
    for name, source in cb.assets().items():
        root = ET.fromstring(source)  # noqa: S314 - our own output
        assert root.get("role") == "img", name
        label = root.get("aria-label")
        assert label, name
        title = root.find("{http://www.w3.org/2000/svg}title")
        assert title is not None, name
        assert title.text == label, name


def test_a_mark_alt_is_the_verdict_and_not_the_name_of_a_picture() -> None:
    assert [mark.alt for mark in cb.MARKS] == ["Pass", "Fail", "By design", "n/a"]


def test_nothing_generated_contains_an_emoji() -> None:
    """The defect this whole exercise exists to end, guarded at the source."""
    for name, source in cb.assets().items():
        assert not EMOJI.search(source), name
    for mark in cb.MARKS:
        assert not EMOJI.search(mark.alt), mark.filename


# --------------------------------------------------------------------------
# The files are files a renderer will accept
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", cb.asset_names())
def test_each_asset_is_well_formed_and_declares_its_own_size(name: str) -> None:
    source = _committed(name)
    root = ET.fromstring(source)  # noqa: S314 - our own output
    assert root.tag == "{http://www.w3.org/2000/svg}svg"
    # Intrinsic width and height as well as a viewBox: without them GitHub
    # cannot reserve the box before the bytes arrive, and a mark in a table
    # cell would render at whatever the cell is wide.
    assert root.get("width")
    assert root.get("height")
    assert root.get("viewBox")


def test_a_mark_is_small_enough_to_repeat_on_every_row() -> None:
    """566 rows will cite these; a heavy mark is 566 heavy requests."""
    for mark in cb.MARKS:
        assert len(_committed(mark.filename).encode("utf8")) < 1024


# --------------------------------------------------------------------------
# The documents that cite them
# --------------------------------------------------------------------------


def _report() -> str:
    return (_ROOT / "docs" / "CONFORMANCE.md").read_text(encoding="utf8")


def test_every_reference_in_the_report_has_a_definition() -> None:
    """A reference with no definition is published as its own source text.

    Nothing fails: the row still renders, the table still lines up, and the
    Status cell reads ``![Pass][cv-pass]``. Both halves come off one list in
    the renderer, so they cannot disagree - which is exactly the kind of
    invariant that stops being true the first time someone edits one of them.
    """
    report = _report()
    used = set(re.findall(r"!\[[^\]]*\]\[([\w-]+)\]", report))
    defined = set(re.findall(r"^\[([\w-]+)\]:\s", report, flags=re.MULTILINE))
    assert used, "the report cites no verdict mark at all"
    assert used == defined


def test_no_mark_in_the_report_stands_without_its_word() -> None:
    """The picture illustrates the verdict; the word is the verdict.

    A cell holding only an image says nothing to a reader whose images did not
    arrive, and nothing in a plain-text diff of a generated file. Caught by
    looking for a mark that runs straight into the end of its cell.
    """
    assert not re.search(r"!\[[^\]]*\]\[cv-[\w-]+\]\s*\|", _report())


def test_the_report_states_its_own_counts_and_pictures_nobody_elses() -> None:
    """The report's headline is a sentence, and deliberately not a banner.

    A banner can only be cited at a fixed ref, and this file is regenerated
    from whatever tree it sits in, so the picture would carry ``main``'s count
    above a sentence carrying the branch's. Nothing pinned to ``main`` in this
    file may state a number; the marks may, because their geometry never
    changes. The pull-request comment is where the banner belongs, pinned to
    the head commit.
    """
    counts = artifact.load()["counts"]
    report = _report()
    assert cb.BANNER not in report
    assert cb.BANNER_DARK not in report
    headline = f"**{counts['passing']}/{counts['checks']} conformance checks pass**"
    assert headline in report
    # Above the first collapsible section, or it is not a summary of them.
    assert report.index(headline) < report.index("<details>")


def test_the_report_legend_names_every_mark_it_draws_and_no_other() -> None:
    """A legend entry for a silhouette that appears nowhere is a false promise."""
    report = _report()
    legend = next(line for line in report.splitlines() if "Verdict marks." in line)
    cited = r"!\[[^\]]*\]\[([\w-]+)\]"
    assert set(re.findall(cited, legend)) == set(re.findall(cited, report))


def test_the_report_carries_no_emoji() -> None:
    """The defect this exercise began with, checked where it was published."""
    assert not EMOJI.search(_report())


def test_the_readme_banner_says_the_committed_counts_in_words() -> None:
    """The README is also the PyPI page, and PyPI freezes what it is given.

    The counts inside the ``alt`` are kept current by
    ``check_conformance_claims.py --write``, which ``make conformance`` runs;
    this is what fails if that sentence is ever worded so the rewriter cannot
    find the numbers in it.
    """
    readme = (_ROOT / "README.md").read_text(encoding="utf8")
    assert f'alt="{cb.banner_alt(artifact.load()["counts"])}"' in readme
    assert f"{cb.RAW_PATH}/{cb.BANNER}" in readme
    assert f"{cb.RAW_PATH}/{cb.BANNER_DARK}" in readme
