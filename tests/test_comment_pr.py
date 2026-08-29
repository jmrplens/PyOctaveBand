#  Copyright (c) 2026. Jose Manuel Requena Plens
"""The pull-request comment reports what moved, and fits in a comment.

``.github/scripts/comment_pr.py`` used to paste the whole conformance report
into the sticky comment: 103 954 characters measured on #645, against a
documented API maximum of 65 536. It posted because GitHub appears to enforce
the limit on the compressed payload and a repetitive Markdown table compresses
well - undocumented headroom that a slightly less repetitive report would have
spent without warning.

So the two properties here are that the comment says what changed, and that it
cannot silently grow past the budget: the second is asserted before the file is
written, so the job fails with a sentence rather than at a 422 after every
other step has already run.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import re
import subprocess
import sys

import pytest

_ROOT = pathlib.Path(__file__).resolve().parent.parent
_SCRIPTS = str(_ROOT / "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)


def _load_comment_pr() -> object:
    """Import the workflow script, which is not on any import path."""
    path = _ROOT / ".github" / "scripts" / "comment_pr.py"
    spec = importlib.util.spec_from_file_location("comment_pr", path)
    if spec is None or spec.loader is None:  # pragma: no cover - import plumbing
        pytest.skip(f"{path} is not importable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cpr = _load_comment_pr()

#: A head commit, standing in for ``github.event.pull_request.head.sha``.
#: Deliberately not ``main``: everything drawn into this comment is pinned to
#: the commit under review, and a test that passed ``main`` could not tell the
#: two apart.
SHA = "0f1e2d3c4b5a69788796a5b4c3d2e1f009876543"


@pytest.fixture(scope="module")
def head() -> dict:
    """The committed artefact, standing in for the pull request's head."""
    return json.loads((_ROOT / "docs" / "conformance.json").read_text(encoding="utf8"))


def _copy(document: dict) -> dict:
    return json.loads(json.dumps(document))


def test_a_verdict_change_leads_the_comment(head: dict) -> None:
    """It is the only bucket that can be a regression, so it comes first."""
    base = _copy(head)
    base["checks"][0]["verdict"] = "fail"
    body = cpr.conformance_section(head, base, SHA)
    assert "Verdict changes (1)" in body
    assert body.index("Verdict changes") < body.index("Closest to their")


def test_a_reworded_quantity_is_a_rename_not_a_delete_and_an_add(head: dict) -> None:
    """The id is derived from the quantity, so rewording one renames the row.

    Matching on the unchanged (domain, designation, clause) is what keeps that
    from reading as an unrelated check appearing beside an unrelated one
    vanishing.
    """
    base = _copy(head)
    base["checks"][0]["id"] += "-as-it-used-to-be-worded"
    base["checks"][0]["quantity"] = "As it used to be worded"
    body = cpr.conformance_section(head, base, SHA)
    assert "Renamed (1)" in body
    assert "New checks" not in body
    assert "Removed checks" not in body


def test_a_move_within_the_display_quantum_is_not_reported(head: dict) -> None:
    """Otherwise every BLAS build would be a row in the comment."""
    base = _copy(head)
    check = next(c for c in base["checks"] if c["deviation"].get("value") is not None)
    check["deviation"]["value"] += 10.0 ** -max(int(check["precision"]), 3) / 10
    body = cpr.conformance_section(head, base, SHA)
    assert "Moved beyond their display precision" not in body


def test_a_move_beyond_the_display_quantum_is_reported(head: dict) -> None:
    base = _copy(head)
    check = next(c for c in base["checks"] if c["deviation"].get("value") is not None)
    check["deviation"]["value"] += 1000.0
    body = cpr.conformance_section(head, base, SHA)
    assert "Moved beyond their display precision (1)" in body


def test_the_closest_to_their_limit_run_even_when_nothing_moved(head: dict) -> None:
    """The standing metrology answer, which 554 static rows never surfaced."""
    body = cpr.conformance_section(head, _copy(head), SHA)
    assert "Nothing moved" in body
    assert "Closest to their published limit" in body


def test_a_missing_baseline_degrades_to_totals(head: dict) -> None:
    """A fork, or the first pull request after the artefact landed."""
    body = cpr.conformance_section(head, None, SHA)
    assert "totals only" in body
    assert "Closest to their published limit" in body


def test_a_missing_artefact_says_how_to_make_one() -> None:
    body = cpr.conformance_section(None, None, SHA)
    assert "make conformance" in body


def test_the_comment_is_far_inside_its_budget(head: dict) -> None:
    """What the whole rewrite bought: the body is a summary, not the report."""
    body = cpr.conformance_section(head, _copy(head), SHA)
    assert len(body) < cpr.BODY_LIMIT / 10


def test_a_body_over_budget_fails_before_it_is_written(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """Failing here names the problem; failing at the API names a 422."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cpr, "BODY_LIMIT", 10)
    monkeypatch.setattr(cpr, "head_document", lambda: None)
    monkeypatch.setattr(cpr, "base_document", lambda: None)
    with pytest.raises(SystemExit, match="over the 10 budget"):
        cpr.main()


def test_a_deviation_below_its_precision_keeps_its_digits(head: dict) -> None:
    """A row of zeros beside "97 %" reads as a contradiction, not a small number."""
    assert (
        cpr._deviation({"deviation": {"value": 4.2e-7}, "precision": 5, "unit": "m"})
        == "4.2e-07 m"
    )


# --------------------------------------------------------------------------
# The indicators: this branch's numbers, and no emoji
# --------------------------------------------------------------------------


def test_the_comment_still_imports_on_the_standard_library_alone() -> None:
    """The ``pr-comment`` job has no pip step, and that is deliberate.

    It used to install the whole scientific stack and spend 45 s regenerating
    a report the ``conformance`` job had already proven current. Now it reads
    two committed JSON files. Citing a badge must stay inside that budget,
    which is why the badge *vocabulary* lives in ``conformance.marks`` and the
    *drawing*, which needs matplotlib for glyph outlines, lives next door in
    ``conformance.badges``. A stray import across that line fails the job on
    every pull request; here it fails in one second.
    """
    blocked = ("numpy", "scipy", "matplotlib", "phonometry", "reference_data")
    program = (
        "import importlib.abc, importlib.util, sys\n"
        f"blocked = {blocked!r}\n"
        "class Blocker(importlib.abc.MetaPathFinder):\n"
        "    def find_spec(self, fullname, path=None, target=None):\n"
        "        if fullname.split('.')[0] in blocked:\n"
        "            raise ImportError('not installed in the pr-comment job: '"
        " + fullname)\n"
        "        return None\n"
        "sys.meta_path.insert(0, Blocker())\n"
        f"sys.path.insert(0, {str(_ROOT / 'scripts')!r})\n"
        "spec = importlib.util.spec_from_file_location('comment_pr', "
        f"{str(_ROOT / '.github' / 'scripts' / 'comment_pr.py')!r})\n"
        "spec.loader.exec_module(importlib.util.module_from_spec(spec))\n"
    )
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", program],
        capture_output=True,
        text=True,
        check=False,
        cwd=_ROOT,
    )
    assert result.returncode == 0, result.stderr


def test_the_banner_is_pinned_to_this_head_and_never_to_main(head: dict) -> None:
    """The defect a banner in a pull-request comment exists to avoid.

    A raw link to ``main`` renders ``main``'s count on every branch. It looks
    live, it is not, and a reviewer reading 566/566 on a branch that broke six
    checks is worse off than with no banner at all.
    """
    body = cpr.conformance_section(head, None, SHA)
    assert f"/{SHA}/.github/badges/conformance-summary.svg" in body
    assert f"/{SHA}/.github/badges/conformance-summary_dark.svg" in body
    assert "/main/.github/badges/" not in body


def test_the_banner_states_this_head_counts_in_words(head: dict) -> None:
    """A 404 on a fork's head commit degrades to the sentence, not to nothing."""
    counts = head["counts"]
    body = cpr.conformance_section(head, None, SHA)
    assert f'alt="All {counts["checks"]} conformance checks pass' in body


def test_a_failing_tree_says_so_on_the_banner_too(head: dict) -> None:
    """The banner is the live part, so it has to be able to carry bad news."""
    broken = _copy(head)
    broken["counts"]["passing"] -= 3
    broken["counts"]["failing"] = 3
    body = cpr.conformance_section(broken, None, SHA)
    assert "3 failing" in body


def test_the_comment_carries_no_emoji(head: dict, tmp_path: pathlib.Path) -> None:
    """Including its own test-status line, which is where the last ones were."""
    # The pictographic blocks only. A wider class that swept from the arrows
    # to the end of the symbol blocks reported the square root in a quantity
    # name as an emoji, and this comment prints quantity names.
    emoji = re.compile("[\U0001f000-\U0001faff☀-➿⬀-⯿️]")
    results = tmp_path / "results"
    results.mkdir()
    (results / "test-results-3.13.xml").write_text(
        '<testsuite tests="4" failures="1"/>', encoding="utf8"
    )
    table, tests, failures = cpr.parse_test_results(str(results), SHA)
    assert (tests, failures) == (4, 1)
    assert not emoji.search(table)
    assert not emoji.search(cpr.conformance_section(head, None, SHA))
    assert not emoji.search(cpr.conformance_section(None, None, SHA))


def test_a_test_row_shows_a_mark_and_still_says_the_word(
    tmp_path: pathlib.Path,
) -> None:
    """The word is what a reader gets when the image does not arrive."""
    results = tmp_path / "results"
    results.mkdir()
    (results / "test-results-3.13.xml").write_text(
        '<testsuite tests="4" failures="0"/>', encoding="utf8"
    )
    (results / "test-results-3.14.xml").write_text(
        '<testsuite tests="4" failures="2"/>', encoding="utf8"
    )
    table, _, _ = cpr.parse_test_results(str(results), SHA)
    assert (
        f"![Pass](https://raw.githubusercontent.com/jmrplens/phonometry/{SHA}" in table
    )
    assert "Passed" in table
    assert (
        f"![Fail](https://raw.githubusercontent.com/jmrplens/phonometry/{SHA}" in table
    )
    assert "Failed" in table
