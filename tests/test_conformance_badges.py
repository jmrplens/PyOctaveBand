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

#: Every ground the marks are read on: GitHub light, GitHub dark, GitHub dim,
#: and Starlight's page colour on the documentation site.
_GROUNDS = ("#ffffff", "#0d1117", "#22272e", "#17181c")

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


def test_every_mark_clears_three_to_one_on_every_ground_it_lands_on() -> None:
    """One file per mark is only defensible if one palette really works.

    The alternative is a ``<picture>`` per row, which measures +163 kB on the
    566-row table against +9.6 kB for a single-file mark, so this test is what
    the size decision rests on.
    """
    marks = cb.render_marks()
    poor: list[str] = []
    for name, source in marks.items():
        for colour in sorted(
            set(re.findall(r'(?:fill|stroke)="(#[0-9a-f]{6})"', source))
        ):
            if colour == "#ffffff":  # the glyph, judged against its own fill
                continue
            poor += [
                f"{name}: {colour} on {ground} is {_contrast(colour, ground):.2f}:1"
                for ground in _GROUNDS
                if _contrast(colour, ground) < 3.0
            ]
    assert poor == []


def test_a_white_glyph_stays_readable_on_the_fill_under_it() -> None:
    for fill in ("#1f883d", "#da3633", "#0969da"):
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


def test_the_report_leads_with_the_banner_for_its_own_counts() -> None:
    counts = artifact.load()["counts"]
    report = _report()
    banner = cb.banner_picture(counts, "main")
    assert banner in report
    # Above the first collapsible section, or it is not a summary of them.
    assert report.index(banner) < report.index("<details>")


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
