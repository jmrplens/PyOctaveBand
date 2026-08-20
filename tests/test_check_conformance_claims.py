#  Copyright (c) 2026. Jose Manuel Requena Plens
"""The conformance-count writer rewrites claims and nothing else.

``scripts/check_conformance_claims.py`` reads the headline of the generated
``docs/CONFORMANCE.md`` and makes every sentence in the corpus that quotes a
count say the same thing. Two properties matter more than the happy path:

* it must move the number wherever it is quoted, in either language, in
  markdown prose, in the root ``README.md`` and the ``README_PYPI.md``
  published to PyPI, in MDX frontmatter and in the ``.zenodo.json``
  description;
* it must move *only* those numbers. A blind ``str.replace`` of the old count
  for the new one has already corrupted an unrelated figure in this repository
  (``replace("303", "316")`` also hit "303 operating days"), so the writer
  substitutes the span of the captured digits, and the fixture below plants
  decoys carrying the **stale** values in unrelated sentences, on the same
  line as a real claim and on lines of their own. A regression to textual
  replacement moves them and the tests fail.

The last test runs the scan over the real corpus and asserts the writer would
change nothing, which is the property CI gates on.
"""

from __future__ import annotations

import json
import pathlib
import sys

import pytest

_SCRIPTS = str(pathlib.Path(__file__).resolve().parent.parent / "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import check_conformance_claims as ccc

#: What the generated report says: the authoritative counts.
TOTAL, DOMAINS, STANDARDS = 501, 55, 338
#: What the prose still says, one release behind, everywhere.
OLD_TOTAL, OLD_DOMAINS, OLD_STANDARDS = 462, 53, 303

REPORT = (
    f"&#9989; **{TOTAL}/{TOTAL} conformance checks pass** across {DOMAINS} "
    f"domains and {STANDARDS} standards - filters class 1.\n"
)

#: Sentences using the stale counts for something that is not a conformance
#: count. They must survive the rewrite verbatim.
DECOY_LINES = (
    f"The campaign ran for {OLD_STANDARDS} operating days at "
    f"{OLD_DOMAINS} sites, over {OLD_TOTAL} measurement positions.\n"
    f"Annex {OLD_DOMAINS} of ISO {OLD_STANDARDS}-2 lists {OLD_TOTAL} rows.\n"
)


def _write(path: pathlib.Path, text: str) -> None:
    """Write a fixture file with exactly the bytes given, on every platform."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf8", newline="")


@pytest.fixture
def corpus(tmp_path: pathlib.Path) -> pathlib.Path:
    """A miniature repository quoting the counts everywhere the real one does.

    Every claim is deliberately stale, so a writer that silently did nothing
    would fail every assertion below.
    """
    root = tmp_path / "repo"
    _write(root / "docs" / "CONFORMANCE.md", REPORT)
    # The two most-read pages of prose, and the two that were outside the walk
    # for as long as it started at docs/. README_PYPI.md is the PyPI long
    # description, which PyPI freezes at upload.
    _write(
        root / "README.md",
        f"The report runs {OLD_TOTAL} conformance checks across {OLD_DOMAINS} "
        f"domains and {OLD_STANDARDS} standards, each pinning a value.\n" + DECOY_LINES,
    )
    _write(
        root / "README_PYPI.md",
        f"The report runs {OLD_TOTAL} conformance checks across {OLD_DOMAINS} "
        f"domains and {OLD_STANDARDS} standards, each pinning a value.\n",
    )
    _write(
        root / "docs" / "start" / "getting-started.md",
        f"- [Conformance report](CONFORMANCE.md): the value of all "
        f"{OLD_TOTAL} checks\n"
        f"The suite runs {OLD_TOTAL} numerical conformance checks across "
        f"{OLD_DOMAINS} domains and {OLD_STANDARDS} standards, one for each of "
        f"the {OLD_TOTAL} rows of Annex {OLD_DOMAINS}.\n" + DECOY_LINES,
    )
    _write(
        root / "site" / "src" / "content" / "docs" / "index.mdx",
        f'description: "Toolkit: {OLD_TOTAL} conformance checks against '
        f'{OLD_STANDARDS} standards, measured over {OLD_DOMAINS} weeks."\n',
    )
    _write(
        root / "site" / "src" / "content" / "docs" / "es" / "index.mdx",
        f'description: "Kit: {OLD_TOTAL} comprobaciones de conformidad frente '
        f'a {OLD_STANDARDS} normas."\n'
        f"La batería ejecuta {OLD_TOTAL} comprobaciones numéricas de "
        f"conformidad en {OLD_DOMAINS} dominios y {OLD_STANDARDS} normas, y "
        f"las {OLD_TOTAL} comprobaciones se publican.\n",
    )
    _write(
        root / ".zenodo.json",
        json.dumps(
            {
                "description": (
                    f"{OLD_TOTAL} numerical conformance checks across "
                    f"{OLD_DOMAINS} domains against {OLD_STANDARDS} IEC, ISO "
                    f"and EN standards, among them ISO {OLD_STANDARDS}-2, "
                    f"IEC 616{OLD_DOMAINS}-1 and ITU-R BS.{OLD_TOTAL}."
                )
            },
            ensure_ascii=False,
        )
        + "\n",
    )
    # Excluded by name and by directory: the report itself, the changelog, the
    # errata registry and the gitignored scratch space all quote historical
    # counts on purpose.
    _write(root / "CHANGELOG.md", f"- {OLD_TOTAL} conformance checks in 1.0.\n")
    _write(root / "docs" / "ERRATA.md", f"- covered by {OLD_TOTAL} checks.\n")
    _write(
        root / "docs" / "superpowers" / "notes.md",
        f"- a local plan quoting {OLD_TOTAL} conformance checks.\n",
    )
    # The root markdown is globbed, not walked: a README one level down belongs
    # to something else (the compatibility stub keeps its own, pinned to the
    # release it describes) and must not be rewritten from this repository's
    # report.
    _write(
        root / "stub" / "README.md",
        f"- the stub quotes {OLD_TOTAL} conformance checks of its own.\n",
    )
    return root


def _run(root: pathlib.Path, *args: str) -> int:
    return ccc.main([*args, "--root", str(root)])


def test_check_reports_every_stale_claim(corpus, capsys):
    """The read-only mode is the CI gate: it lists them all and exits 1."""
    assert _run(corpus) == 1
    errors = capsys.readouterr().err
    for quoted in (
        "README.md",
        "README_PYPI.md",
        "docs/start/getting-started.md",
        "site/src/content/docs/index.mdx",
        "site/src/content/docs/es/index.mdx",
        ".zenodo.json",
    ):
        assert quoted in errors
    assert f"claims {OLD_TOTAL} total, report says {TOTAL}" in errors
    assert f"claims {OLD_DOMAINS} domains, report says {DOMAINS}" in errors
    assert f"claims {OLD_STANDARDS} standards, report says {STANDARDS}" in errors


def test_write_moves_every_quoted_count(corpus, capsys):
    """Every place that quotes a count is brought into line in one run."""
    assert _run(corpus, "--write") == 0
    assert "Rewrote" in capsys.readouterr().out

    for readme in ("README.md", "README_PYPI.md"):
        text = (corpus / readme).read_text(encoding="utf8")
        assert (
            f"{TOTAL} conformance checks across {DOMAINS} domains and "
            f"{STANDARDS} standards" in text
        ), f"{readme} was not brought into line"

    started = (corpus / "docs" / "start" / "getting-started.md").read_text(
        encoding="utf8"
    )
    assert f"all {TOTAL} checks" in started
    assert (
        f"{TOTAL} numerical conformance checks across {DOMAINS} domains and "
        f"{STANDARDS} standards" in started
    )

    english = (corpus / "site/src/content/docs/index.mdx").read_text(encoding="utf8")
    assert f"{TOTAL} conformance checks against {STANDARDS} standards" in english

    spanish = (corpus / "site/src/content/docs/es/index.mdx").read_text(encoding="utf8")
    assert (
        f"{TOTAL} comprobaciones de conformidad frente a {STANDARDS} normas" in spanish
    )
    assert f"en {DOMAINS} dominios y {STANDARDS} normas" in spanish
    assert f"las {TOTAL} comprobaciones" in spanish

    zenodo = json.loads((corpus / ".zenodo.json").read_text(encoding="utf8"))
    assert (
        f"{TOTAL} numerical conformance checks across {DOMAINS} domains "
        f"against {STANDARDS} IEC" in zenodo["description"]
    )

    # And the corpus now passes the read-only gate.
    assert _run(corpus) == 0


def test_write_leaves_the_decoys_alone(corpus):
    """A number that is not a conformance count is never touched.

    The decoys carry the stale counts in unrelated sentences: on their own
    lines, on the same line as a real claim, and as part of a standard's
    designation. A textual replace of the old value for the new one rewrites
    all of them.
    """
    assert _run(corpus, "--write") == 0

    started = (corpus / "docs" / "start" / "getting-started.md").read_text(
        encoding="utf8"
    )
    assert started.endswith(DECOY_LINES)
    assert f"one for each of the {OLD_TOTAL} rows of Annex {OLD_DOMAINS}." in started

    assert (corpus / "README.md").read_text(encoding="utf8").endswith(DECOY_LINES)

    english = (corpus / "site/src/content/docs/index.mdx").read_text(encoding="utf8")
    assert f"measured over {OLD_DOMAINS} weeks" in english

    zenodo = json.loads((corpus / ".zenodo.json").read_text(encoding="utf8"))
    for designation in (
        f"ISO {OLD_STANDARDS}-2",
        f"IEC 616{OLD_DOMAINS}-1",
        f"ITU-R BS.{OLD_TOTAL}",
    ):
        assert designation in zenodo["description"]


def test_write_is_idempotent(corpus):
    """A second run changes nothing, byte for byte."""
    assert _run(corpus, "--write") == 0
    first = {p: p.read_bytes() for p in ccc.sources(corpus)}
    assert _run(corpus, "--write") == 0
    assert {p: p.read_bytes() for p in ccc.sources(corpus)} == first


def test_write_touches_only_the_files_that_quote_a_count(corpus):
    """Excluded and count-free files come out byte-identical.

    ``stub/README.md`` is in the list for the same reason the root markdown is
    globbed rather than walked: it is a different package's page, describing a
    different release, and this report says nothing about it.
    """
    untouched = [
        corpus / "CHANGELOG.md",
        corpus / "docs" / "ERRATA.md",
        corpus / "docs" / "superpowers" / "notes.md",
        corpus / "docs" / "CONFORMANCE.md",
        corpus / "stub" / "README.md",
    ]
    before = {p: p.read_bytes() for p in untouched}
    assert _run(corpus, "--write") == 0
    assert {p: p.read_bytes() for p in untouched} == before


def test_a_crlf_file_stays_crlf(tmp_path):
    """The writer changes digits, not the file's line endings."""
    _write(tmp_path / "docs" / "CONFORMANCE.md", REPORT)
    page = tmp_path / "docs" / "getting-started.md"
    _write(
        page,
        f"Intro paragraph.\r\nThe suite runs {OLD_TOTAL} conformance checks.\r\n",
    )
    assert _run(tmp_path, "--write") == 0
    assert (
        page.read_bytes()
        == (
            f"Intro paragraph.\r\nThe suite runs {TOTAL} conformance checks.\r\n"
        ).encode()
    )


def test_line_endings_and_surrounding_prose_are_preserved():
    """Only the digits move: indentation, quoting and CRLF stay put."""
    expected = {"total": TOTAL, "domains": DOMAINS, "standards": STANDARDS}
    line = (
        f'  description: "…; {OLD_TOTAL} conformance checks against '
        f'{OLD_STANDARDS} standards."\r\n'
    )
    corrected, fixed = ccc.rewrite_line(line, expected)
    assert corrected == (
        f'  description: "…; {TOTAL} conformance checks against '
        f'{STANDARDS} standards."\r\n'
    )
    assert [claim.field for claim in fixed] == ["total", "standards"]


def test_a_claim_read_two_ways_is_rejected(monkeypatch):
    """Two patterns that disagree about a number are a bug, not a coin toss."""
    import re

    monkeypatch.setattr(
        ccc,
        "CLAIMS",
        (
            (re.compile(r"(\d+)\s+conformance checks"), "total"),
            (re.compile(r"(\d+)\s+conformance checks"), "standards"),
        ),
    )
    with pytest.raises(ValueError, match="ambiguous conformance claim"):
        ccc.claims_in(f"the suite runs {OLD_TOTAL} conformance checks")


def test_an_ambiguous_pattern_pair_reports_instead_of_crashing(
    corpus, monkeypatch, capsys
):
    """The command exits 1 with the diagnostic, not with a traceback."""
    import re

    monkeypatch.setattr(
        ccc,
        "CLAIMS",
        (
            (re.compile(r"(\d+)\s+conformance checks"), "total"),
            (re.compile(r"(\d+)\s+conformance checks"), "standards"),
        ),
    )
    assert _run(corpus, "--write") == 1
    assert "ambiguous conformance claim" in capsys.readouterr().err


def test_a_report_without_a_headline_is_an_error(tmp_path, capsys):
    """A format change must fail loudly rather than rewrite prose to nothing."""
    _write(tmp_path / "docs" / "CONFORMANCE.md", "no headline here\n")
    assert _run(tmp_path, "--write") == 1
    assert "headline not found" in capsys.readouterr().err


def test_the_real_corpus_needs_no_rewriting():
    """The committed tree is already consistent, so the writer is a no-op.

    This is the same property the CI gate asserts, checked without touching
    the working tree: every source file is rewritten in memory and must come
    back unchanged.
    """
    expected = ccc.expected_counts()
    for path in ccc.sources():
        text = path.read_text(encoding="utf8")
        corrected, fixed = ccc.rewrite_text(text, expected)
        assert not fixed, f"{path} quotes a stale count: {fixed}"
        assert corrected == text
