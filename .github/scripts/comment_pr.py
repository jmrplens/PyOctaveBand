"""Assemble the PR comment body: what moved, not the whole report.

The comment used to be the entire conformance report, all 554 rows of it,
posted again on every run. Measured on #645 that body was 103 954 characters,
against a documented API maximum of 65 536 - it went through because GitHub
appears to enforce the limit on the compressed payload, and a highly repetitive
Markdown table compresses extremely well. That is undocumented headroom, and a
reviewer scrolling 554 unchanged rows to find the two that moved was paying for
it either way.

So the comment answers one question: **what moved.** It joins the committed
``docs/conformance.json`` at this head against the same file on the base branch
and reports verdict changes, new checks, removed checks, renames and numeric
moves - plus, unconditionally, the checks sitting closest to their published
limit, which is the standing metrology question and the one 554 static rows
never surfaced.

Two consequences worth stating. This runs no checks: the ``conformance`` job has
already failed the pull request if the committed artefact is stale, so a
comment built from committed files cannot be reading numbers nobody verified.
And it needs the base commit, so the workflow must fetch it - a checkout at the
default depth has only the merge commit, and a missing baseline degrades to
totals only rather than failing.

The full report is linked rather than pasted. A blob at the head SHA is a
permalink, renders as a table on github.com, and needs no login on a public
repo - which a workflow artefact, being a zip behind an authenticated download
that expires after 90 days, is not.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from conformance.marks import banner_picture, outcome_mark
from conformance.metrics import utilisation

#: Hard ceiling on the body, asserted before the file is written. The API's
#: documented maximum is 65 536 characters; failing here names the problem,
#: where failing at the API names a 422 after all the work is done.
BODY_LIMIT = 60_000

#: Rows shown per bucket before the rest are summarised as a count.
BUCKET_ROWS = 12

#: How many of the closest-to-their-limit checks to list.
CLOSEST_ROWS = 5

ARTIFACT = "docs/conformance.json"


def parse_test_results(test_dir: str, ref: str) -> tuple[str, int, int]:
    """The per-version test table, its total and its failure count.

    :param test_dir: Directory the workflow downloaded the JUnit XML into.
    :param ref: Commit to pin the pass and fail marks to, as everywhere else
        in this comment.
    """
    summary = []
    total_tests = 0
    total_failures = 0

    if not Path(test_dir).exists():
        return "No test results found.", 0, 0

    # Distinguish between test results and coverage reports
    test_files = []
    coverage_files = {}  # version -> path

    for root, _, filenames in os.walk(test_dir):
        for filename in filenames:
            f_path = Path(root) / filename
            if filename.startswith("test-results-") and filename.endswith(".xml"):
                test_files.append(f_path)
            elif filename == "coverage.xml":
                # Extract version from parent directory name
                version = Path(root).name.replace("test-results-", "")
                coverage_files[version] = f_path

    # Sorted as text, so the row order does not depend on whether the list
    # holds strings or Path objects: a Path compares component by component,
    # which reorders names where one is a prefix of another.
    test_files.sort(key=str)

    summary.append("| Python Version | Tests | Failures | Coverage | Status |")
    summary.append("|---|---|---|---|---|")

    for f_path in test_files:
        f_name = f_path.name
        try:
            tree = ET.parse(f_path)
            root = tree.getroot()

            if root.tag == "testsuites":
                tests = 0
                failures = 0
                for suite in root:
                    tests += int(suite.attrib.get("tests", 0))
                    failures += int(suite.attrib.get("failures", 0))
            else:
                tests = int(root.attrib.get("tests", 0))
                failures = int(root.attrib.get("failures", 0))

            version = f_name.replace("test-results-", "").replace(".xml", "")

            # Parse coverage for this version if available
            coverage_pct = "-"
            if version in coverage_files:
                try:
                    cov_tree = ET.parse(coverage_files[version])
                    cov_root = cov_tree.getroot()
                    line_rate = float(cov_root.attrib.get("line-rate", 0))
                    coverage_pct = f"{line_rate * 100:.1f}%"
                except Exception:  # noqa: BLE001 - degrade gracefully if the coverage artifact is malformed
                    coverage_pct = "error"

            # The mark, then the word. The word is what a reader gets if the
            # image never arrives, and this table is read on a page where an
            # image can 404: a fork's head commit is not always servable from
            # this repository's raw host.
            passed = failures == 0
            status = f"{outcome_mark(passed, ref)} {'Passed' if passed else 'Failed'}"
            summary.append(
                f"| {version} | {tests} | {failures} | {coverage_pct} | {status} |"
            )

            total_tests += tests
            total_failures += failures
        except Exception as e:  # noqa: BLE001 - degrade gracefully if the coverage artifact is malformed
            summary.append(f"| {f_name} | - | - | - | Error parsing: {e} |")

    return "\n".join(summary), total_tests, total_failures


def head_document() -> dict[str, Any] | None:
    """The artefact committed at this head, or ``None`` if it is missing."""
    path = Path(ARTIFACT)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def base_document() -> dict[str, Any] | None:
    """The artefact on the base branch, or ``None`` if it cannot be reached.

    A fork, a first pull request after the artefact landed, or a checkout that
    never fetched the base all end here. None of those is an error: the comment
    falls back to totals and says so.
    """
    base = os.environ.get("GITHUB_BASE_REF")
    if not base:
        return None
    for ref in (f"origin/{base}", base):
        # Fixed argv, no shell: the only interpolated value is the base branch
        # name the workflow passes in, and it lands in an argument, never in a
        # command line a shell would re-split.
        result = subprocess.run(
            ["git", "show", f"{ref}:{ARTIFACT}"],
            capture_output=True,
            check=False,
        )
        if result.returncode == 0:
            try:
                return json.loads(result.stdout.decode("utf-8"))
            except ValueError:
                return None
    return None


def _by_id(document: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {check["id"]: check for check in document["checks"]}


def _identity(check: dict[str, Any]) -> tuple[str, str, str]:
    """What a check is about, independent of how its quantity is worded.

    Renames are matched on this, so rewording a quantity shows up as one moved
    row rather than as a deletion beside an unrelated-looking addition.
    """
    reference = check["reference"]
    return (check["domain"], reference["designation"], reference.get("clause") or "")


def _moved(before: dict[str, Any], after: dict[str, Any]) -> bool:
    """Whether a value moved by more than the check's own display quantum.

    Anything smaller is the last digit of a number the check does not print,
    and reporting it would make every BLAS build a comment row.
    """
    old = before["deviation"].get("value")
    new = after["deviation"].get("value")
    if old is None or new is None:
        return old != new
    quantum = 10.0 ** -max(int(after["precision"]), 3)
    return abs(new - old) > quantum


def _used(check: dict[str, Any]) -> str:
    fraction = utilisation(check)
    return "-" if fraction is None else f"{fraction * 100:.0f} %"


def _deviation(check: dict[str, Any]) -> str:
    """The deviation as the report prints it: trailing zeros trimmed.

    A value below its own display precision falls back to three significant
    digits rather than printing as a row of zeros beside a utilisation of 97 %,
    which is the pair that makes a reader distrust both numbers.
    """
    value = check["deviation"].get("value")
    if value is None:
        return check["deviation"].get("label") or "-"
    places = max(int(check["precision"]), 3)
    text = f"{value:.{places}f}".rstrip("0").rstrip(".")
    if text in ("", "0", "-0") and value != 0:
        text = f"{value:.3g}"
    unit = check.get("unit")
    return f"{text}{f' {unit}' if unit else ''}"


def _row(check: dict[str, Any]) -> str:
    return (
        f"| {check['reference']['cite']} | {check['quantity']} "
        f"| {_deviation(check)} | {_used(check)} |"
    )


def _table(title: str, checks: list[dict[str, Any]]) -> list[str]:
    """One bucket, with its overflow named rather than dropped silently."""
    if not checks:
        return []
    lines = [
        f"**{title} ({len(checks)})**",
        "",
        "| Standard | Quantity | Deviation | Used |",
        "|---|---|---|---|",
    ]
    lines += [_row(check) for check in checks[:BUCKET_ROWS]]
    if len(checks) > BUCKET_ROWS:
        lines.append(f"| … | {len(checks) - BUCKET_ROWS} more | | |")
    lines.append("")
    return lines


def _verdict_rows(
    base: dict[str, dict[str, Any]], head: dict[str, dict[str, Any]]
) -> list[str]:
    """Verdict changes: the only bucket that can be a regression."""
    changed = [
        (base[check_id], check)
        for check_id, check in head.items()
        if check_id in base and base[check_id]["verdict"] != check["verdict"]
    ]
    if not changed:
        return []
    lines = [
        f"**Verdict changes ({len(changed)})**",
        "",
        "| Standard | Quantity | Base | Head |",
        "|---|---|---|---|",
    ]
    lines += [
        f"| {new['reference']['cite']} | {new['quantity']} "
        f"| {old['verdict']} | {new['verdict']} |"
        for old, new in changed[:BUCKET_ROWS]
    ]
    lines.append("")
    return lines


def _rename_rows(renamed: list[tuple[dict[str, Any], dict[str, Any]]]) -> list[str]:
    if not renamed:
        return []
    lines = [
        f"**Renamed ({len(renamed)})**",
        "",
        "| Standard | Base quantity | Head quantity |",
        "|---|---|---|",
    ]
    lines += [
        f"| {new['reference']['cite']} | {old['quantity']} | {new['quantity']} |"
        for old, new in renamed[:BUCKET_ROWS]
    ]
    lines.append("")
    return lines


def _split_renames(
    added: list[dict[str, Any]], removed: list[dict[str, Any]]
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[tuple[dict[str, Any], dict[str, Any]]],
]:
    """Pair an addition with a removal that is the same check, reworded."""
    pool = {_identity(check): check for check in removed}
    renamed = []
    still_added = []
    for check in added:
        match = pool.pop(_identity(check), None)
        if match is None:
            still_added.append(check)
        else:
            renamed.append((match, check))
    return still_added, list(pool.values()), renamed


def conformance_section(
    head: dict[str, Any] | None, base: dict[str, Any] | None, ref: str
) -> str:
    """The body's lead: the headline, then only what changed.

    :param head: The artefact committed at this head, or ``None``.
    :param base: The same file on the base branch, or ``None``.
    :param ref: The commit the banner is pinned to. This is the whole point of
        drawing a banner in a pull-request comment rather than linking the one
        on ``main``: a raw link to ``main`` renders ``main``'s count on every
        branch, which is worse than no banner because it looks live and is
        not. The workflow passes ``github.event.pull_request.head.sha``, and
        the same 40-character URL shape is what the "full report" links below
        already resolve with.
    """
    if head is None:
        return (
            "## Numerical conformance\n\n"
            f"**No artefact.** `{ARTIFACT}` is missing from this checkout, so "
            "nothing can be compared. Run `make conformance` and commit the "
            "result."
        )
    counts = head["counts"]
    lines = [
        "## Numerical conformance",
        "",
        # Pinned to this head, so the picture is the one this branch generated
        # and the numbers on it are the numbers in the tables below it.
        banner_picture(counts, ref),
        "",
        f"**{counts['passing']}/{counts['checks']} checks pass** across "
        f"{counts['domains']} domains and {counts['standards']} standards "
        f"({counts['designations']} normative designations, {counts['sources']} "
        "further published sources).",
        "",
        "<sub><b>Used</b> in the tables below is how much of that clause's "
        "published tolerance the deviation consumes: 100 % sits exactly on the "
        "limit, 5 % uses a twentieth of the allowance, and a dash means the "
        "clause states no two-sided tolerance for the quantity, so there is no "
        "budget to spend. It is reported and never used to decide a verdict, "
        "which is settled at full precision before any rounding.</sub>",
        "",
    ]
    lines += _changes(head, base)
    lines += _closest(head)
    return "\n".join(lines)


def _changes(head: dict[str, Any], base: dict[str, Any] | None) -> list[str]:
    """Every bucket of movement, in order of how much a reviewer should care."""
    if base is None:
        return [
            "_No baseline artefact on the base branch, so this run reports "
            "totals only._",
            "",
        ]
    base_checks, head_checks = _by_id(base), _by_id(head)
    added = [check for cid, check in head_checks.items() if cid not in base_checks]
    removed = [check for cid, check in base_checks.items() if cid not in head_checks]
    added, removed, renamed = _split_renames(added, removed)
    moved = [
        check
        for cid, check in head_checks.items()
        if cid in base_checks and _moved(base_checks[cid], check)
    ]
    lines = _verdict_rows(base_checks, head_checks)
    lines += _table("New checks", added)
    lines += _table("Removed checks", removed)
    lines += _rename_rows(renamed)
    lines += _table("Moved beyond their display precision", moved)
    if len(lines) == 0:
        base_counts = base["counts"]
        lines = [
            f"_Nothing moved: same {base_counts['checks']} checks, same "
            "verdicts, same numbers._",
            "",
        ]
    return lines


def _closest(head: dict[str, Any]) -> list[str]:
    """The standing question, asked whether or not anything moved."""
    ranked = sorted(
        (check for check in head["checks"] if utilisation(check) is not None),
        key=lambda check: utilisation(check) or 0.0,
        reverse=True,
    )
    if not ranked:
        return []
    lines = [
        f"**Closest to their published limit (top {CLOSEST_ROWS})**",
        "",
        "_The rows with the least room left, so the ones a change is most "
        "likely to push over._",
        "",
        "| Standard | Quantity | Deviation | Used |",
        "|---|---|---|---|",
    ]
    lines += [_row(check) for check in ranked[:CLOSEST_ROWS]]
    lines.append("")
    return lines


def main() -> None:
    repo = os.environ.get("GITHUB_REPOSITORY")
    run_id = os.environ.get("GITHUB_RUN_ID")
    sha = os.environ.get("GITHUB_HEAD_SHA") or os.environ.get("GITHUB_SHA") or "main"
    test_dir = "test-results"

    conformance = conformance_section(head_document(), base_document(), sha)
    test_table, tests, failures = parse_test_results(test_dir, sha)
    status = "all green" if failures == 0 else f"{failures} failing"
    # Raw <img>: the summary line below sits inside the <details> HTML block,
    # where Markdown image syntax ships as literal text.
    verdict = outcome_mark(failures == 0, sha, html=True)
    blob = f"https://github.com/{repo}/blob/{sha}"

    # Hidden marker so the CI updates one sticky comment instead of posting a
    # new one every run (see the "Post PR Comment" step in python-app.yml).
    body = f"""<!-- phonometry-ci-conformance -->
{conformance}

---

<details>
<summary>{verdict} Tests &amp; coverage: {tests} tests, {failures} failures ({status})</summary>

{test_table}

</details>

<sub>Full report at this commit: <a href="{blob}/docs/CONFORMANCE.md">docs/CONFORMANCE.md</a> · \
<a href="{blob}/docs/conformance.json">docs/conformance.json</a> · \
<a href="https://github.com/{repo}/actions/runs/{run_id}">full CI artifacts</a></sub>
"""

    if len(body) > BODY_LIMIT:
        msg = (
            f"the pull-request comment is {len(body)} characters, over the "
            f"{BODY_LIMIT} budget. GitHub rejects a body past 65 536; shrink a "
            "bucket in .github/scripts/comment_pr.py rather than hoping the "
            "compressed payload squeezes through."
        )
        raise SystemExit(msg)

    Path("pr_comment_body.md").write_text(body, encoding="utf-8")


if __name__ == "__main__":
    main()
