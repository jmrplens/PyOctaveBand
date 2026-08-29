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
    body = cpr.conformance_section(head, base)
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
    body = cpr.conformance_section(head, base)
    assert "Renamed (1)" in body
    assert "New checks" not in body
    assert "Removed checks" not in body


def test_a_move_within_the_display_quantum_is_not_reported(head: dict) -> None:
    """Otherwise every BLAS build would be a row in the comment."""
    base = _copy(head)
    check = next(c for c in base["checks"] if c["deviation"].get("value") is not None)
    check["deviation"]["value"] += 10.0 ** -max(int(check["precision"]), 3) / 10
    body = cpr.conformance_section(head, base)
    assert "Moved beyond their display precision" not in body


def test_a_move_beyond_the_display_quantum_is_reported(head: dict) -> None:
    base = _copy(head)
    check = next(c for c in base["checks"] if c["deviation"].get("value") is not None)
    check["deviation"]["value"] += 1000.0
    body = cpr.conformance_section(head, base)
    assert "Moved beyond their display precision (1)" in body


def test_the_closest_to_their_limit_run_even_when_nothing_moved(head: dict) -> None:
    """The standing metrology answer, which 554 static rows never surfaced."""
    body = cpr.conformance_section(head, _copy(head))
    assert "Nothing moved" in body
    assert "Closest to their published limit" in body


def test_a_missing_baseline_degrades_to_totals(head: dict) -> None:
    """A fork, or the first pull request after the artefact landed."""
    body = cpr.conformance_section(head, None)
    assert "totals only" in body
    assert "Closest to their published limit" in body


def test_a_missing_artefact_says_how_to_make_one() -> None:
    body = cpr.conformance_section(None, None)
    assert "make conformance" in body


def test_the_comment_is_far_inside_its_budget(head: dict) -> None:
    """What the whole rewrite bought: the body is a summary, not the report."""
    body = cpr.conformance_section(head, _copy(head))
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
