#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Turning the artefact into the Markdown the report is.

The Markdown is a pure function of ``docs/conformance.json``: this module
computes nothing and runs no check, it formats what the artefact already
decided. That is what lets ``docs/CONFORMANCE.md`` keep a byte gate while the
values behind it are compared within a tolerance - the bytes are derived from
bytes that are themselves committed, so two runs on different hardware render
the same file even when the underlying computation wobbles in its last digit.

It is also what makes the report testable on a three-check fixture instead of
only end to end on 552, and what stops the Markdown mirror and the site page
from ever disagreeing: both read one document.

The reader gets the headline count, the filters and weightings showcase, then
one collapsible section per domain in registration order. Only the committed
file also gets :data:`_DOC_HEADER`, the do-not-edit banner, so the pull-request
comment stays header-free.

The collapsing rule is the point of the layout: a section stays shut while
every one of its rows passes and springs open the moment one fails, so a long
green report reads as a single line and a regression is the only thing open on
the page. The escaping in :func:`_cell` exists for the same reason - an
unescaped ``|`` in a quantity written the way acousticians write it silently
drops the evidence columns off the right of the row.
"""

from __future__ import annotations

import argparse
import re
import sys
from typing import TYPE_CHECKING, Any

from .artifact import ARTIFACT_PATH, load, write
from .metrics import utilisation
from .registry import (
    _ROOT,
    Kind,
    ToleranceMode,
    Verdict,
    _fmt,
    _headroom,
    _snap,
    deviation_places,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = ["_snap"]


#: The four verdicts as the tables print them. A tick and a cross can only say
#: two things, so "by design" and "not applicable" had nowhere to go and were
#: carried in a display string instead; a word says all four, survives a
#: plain-text diff and needs no colour to be read.
_VERDICT_TEXT = {
    str(Verdict.PASS): "Pass",
    str(Verdict.FAIL): "**Fail**",
    str(Verdict.BY_DESIGN): "By design",
    str(Verdict.NOT_APPLICABLE): "n/a",
}


def _status(verdict: str) -> str:
    """One verdict as the table prints it."""
    return _VERDICT_TEXT.get(verdict, verdict)


def _used(check: Mapping[str, Any]) -> str:
    """How much of its published limit a check's deviation consumes.

    Blank where the question has no answer, which is a check that declares no
    tolerance or a mask bounded on one side only.
    """
    fraction = utilisation(check)
    return "-" if fraction is None else f"{fraction * 100:.0f} %"


def _cell(text: object) -> str:
    """Escape a value so it survives as a single Markdown table cell.

    An unescaped ``|`` reads as a column separator, so a quantity written the
    way acousticians write it -- ``max |diff|``, ``20 lg(|k|/k0)`` -- silently
    splits its row into more cells than the header has. Markdown renderers
    then drop the surplus from the right, which in this table means losing the
    computed value, the deviation and the pass mark: the row still renders, so
    nothing looks broken while the evidence is gone.

    Idempotent, so a value that already escapes its own bars is left alone.
    """
    return re.sub(r"(?<!\\)\|", r"\\|", str(text))


def _domains(document: Mapping[str, Any] | None = None) -> list[str]:
    """The domain titles, in registration order."""
    source = load() if document is None else document
    return [domain["title"] for domain in source["domains"]]


def _filter_verdict(row: Mapping[str, Any]) -> str:
    """The class verdict of one showcase row, as the table prints it."""
    if row["verdict"] == str(Verdict.BY_DESIGN):
        return f"By design ({row['reason']})"
    if "class" not in row:
        return "not compliant"
    if row["class"] == 1:
        return "Class 1 (default)" if row["architecture"] == "butter" else "Class 1"
    return f"Class {row['class']}"


def _panel(document: Mapping[str, Any], panel_id: str) -> Mapping[str, Any]:
    """One showcase panel by id."""
    panels: list[Mapping[str, Any]] = list(document["panels"])
    for panel in panels:
        if panel["id"] == panel_id:
            return panel
    msg = f"the artefact carries no {panel_id!r} panel."
    raise KeyError(msg)


def _numerical_validation_section(document: Mapping[str, Any], filters_ok: bool) -> str:
    # Collapsed like the per-domain groups so the report stays compact by
    # default; it springs open whenever the filters/weightings domain has a
    # failing row, exactly like those groups do.
    opened = "" if filters_ok else " open"
    lines: list[str] = []
    lines.append(f"<details{opened}>")
    lines.append(
        "<summary><b>Numerical validation - filters &amp; "
        "weightings</b>: class showcase (IEC 61260-1 · IEC 61672-1 · "
        "ISO 7196)</summary>"
    )
    lines.append("")
    lines.append(
        "**IEC 61260-1:2014 class per filter architecture** (order 6, "
        "one-third-octave, 100 Hz-10 kHz, fs = 48 kHz). For each architecture "
        "the table shows, at its *binding* band, the measured relative "
        "attenuation and the class-1 limit it must clear, so the number and "
        "the range it must sit in are both visible. A positive margin means "
        "the acceptance limits are met with that much room."
    )
    lines.append("")
    lines.append(
        "| Architecture | Class verdict | Binding band | Measured rel. atten. "
        "| Class-1 limit | Margin cl.1 | Margin cl.2 |"
    )
    lines.append("|:---|:---:|:---:|:---:|:---:|:---:|:---:|")
    for row in _panel(document, "filter-class")["rows"]:
        bind = row["binding"]
        comparator = "&le;" if bind["side"] == "ceil" else "&ge;"
        lines.append(
            f"| {row['architecture']} | {_filter_verdict(row)} "
            f"| {bind['frequency_hz']:.0f} Hz "
            f"| {bind['measured']:+.2f} dB | {comparator} {bind['limit']:+.2f} dB "
            f"| {row['margin_class1']:+.3f} dB | {row['margin_class2']:+.3f} dB |"
        )
    lines.append("")
    lines.append(
        "Only **Butterworth** (the library default) and **Chebyshev-II** are "
        "class-compliant architectures. Chebyshev-I and elliptic trade the "
        "mask for passband ripple, and Bessel for a maximally-flat group delay "
        "(soft rolloff); they cannot satisfy the IEC 61260-1 Class 1/2 "
        "attenuation mask by construction, so they are labelled *By design* - "
        "this is expected, not a failure or regression."
    )
    lines.append("")
    lines.append(
        "**Frequency-weighting conformance** (A/C: IEC 61672-1 Table 3; "
        "G: ISO 7196 A.3). The *max deviation from nominal* is informational "
        "(it falls at a frequency extreme where the tolerance is widest and "
        "asymmetric); compliance is judged at the *binding* frequency - the "
        "one with the least headroom - where the deviation, the applicable "
        "tolerance band and the headroom are shown together."
    )
    lines.append("")
    lines.append(
        "| Curve | fs | Max dev. from nominal (info) | Binding freq "
        "| Deviation there | Tolerance band | Headroom |"
    )
    lines.append("|:---|:---:|:---:|:---:|:---:|:---:|:---:|")
    for row in _panel(document, "weighting-deviation")["rows"]:
        worst, bind = row["worst"], row["binding"]
        band = f"[{bind['lower']:+.2f}, {bind['upper']:+.2f}] dB"
        lines.append(
            f"| {row['curve']} | {row['fs_hz'] // 1000} kHz "
            f"| {worst['deviation']:+.3f} dB @ {worst['frequency_hz']:.0f} Hz "
            f"| {bind['frequency_hz']:.0f} Hz | {bind['deviation']:+.3f} dB | {band} "
            f"| {row['headroom']:+.3f} dB |"
        )
    lines.append("")
    lines.append("</details>")
    return "\n".join(lines)


def expected_text(check: Mapping[str, Any]) -> str:
    """The Expected column of one row.

    A check that gave its normative side a label prints the label: the label is
    there precisely because the value alone does not say what the standard
    requires ("class 1", "m >= 0.5 (C.4.2 pass criterion)"). Everything else
    prints the value and the limit it is judged against, which is the pairing
    the whole report exists to show.
    """
    side, tolerance = check["expected"], check.get("tolerance")
    if "label" in side:
        return str(side["label"])
    unit = check.get("unit", "")
    value = _fmt(side["value"], unit, check["precision"])
    if tolerance is None or "value" not in tolerance:
        return value
    if tolerance["mode"] == str(ToleranceMode.RELATIVE):
        return f"{value} (+/-{tolerance['value'] * 100:g}%)"
    return f"{value} (+/-{_fmt(tolerance['value'], unit, check['precision'])})"


def computed_text(check: Mapping[str, Any]) -> str:
    """The Computed column of one row."""
    side = check["computed"]
    if "label" in side:
        return str(side["label"])
    return _fmt(side["value"], check.get("unit", ""), check["precision"])


def deviation_text(check: Mapping[str, Any]) -> str:
    """The delta column of one row.

    Four shapes, one per kind, plus the compatibility path: a check that
    declares no tolerance was built before the builders existed and its delta
    survives only as the string it was formatted as, so that string is printed
    rather than a number reconstructed from it.
    """
    deviation, unit = check["deviation"], check.get("unit", "")
    if "tolerance" not in check:
        return str(deviation["label"])
    kind, value = check["kind"], deviation.get("value")
    if kind in (str(Kind.RECORD), str(Kind.COUNT)):
        return "exact" if not value else f"{int(value)} differ"
    if kind == str(Kind.MASK):
        binding = check.get("binding", {})
        headroom = _headroom(value, binding.get("lower"), binding.get("upper"))
        return f"headroom {_fmt(headroom, unit, deviation_places(check['precision']))}"
    return _fmt(value, unit, deviation_places(check["precision"]))


def render_markdown(document: Mapping[str, Any] | None = None) -> tuple[str, int, int]:
    """Render the full conformance report from the artefact.

    :param document: The artefact; read from :data:`~conformance.artifact.
        ARTIFACT_PATH` when omitted.
    :return: The Markdown, the passing count and the total.
    """
    source = load() if document is None else document
    counts = source["counts"]
    passed, total = counts["passing"], counts["checks"]

    filters_ok = all(
        domain["passing"] == domain["checks"]
        for domain in source["domains"]
        if domain["title"] == "Filters & weightings"
    )
    failing = total - passed

    out: list[str] = []
    out.append("## Numerical conformance report")
    out.append("")
    summary = (
        f"**{passed}/{total} conformance checks pass** across "
        f"{counts['domains']} domains and {counts['standards']} standards"
    )
    if filters_ok:
        summary += " - filters class 1 - weightings within IEC 61672-1 class 1"
    out.append(f"{summary}." if not failing else f"{summary} - **{failing} failing**.")
    out.append("")
    out.append(
        "<sub><b>&#916;</b> is the difference between the computed value and "
        "the one the standard publishes. <b>Used</b> is how much of that "
        "clause's published tolerance the difference consumes: 100 % means it "
        "sits exactly on the limit, 5 % means it uses a twentieth of the "
        "allowance, and a dash means the clause states no two-sided tolerance "
        "for the quantity, so there is no budget to spend. It is reported and "
        "never used to decide a verdict, which is settled at full precision "
        "before any rounding.</sub>"
    )
    out.append("")
    out.append(
        "<sub>Each row pins a standard clause to its expected normative value "
        "and the value the library computes. Every section below is "
        "collapsible and stays collapsed while all of its rows pass; a "
        "section with any failing row opens automatically.</sub>"
    )
    out.append("")
    out.append(_numerical_validation_section(source, filters_ok))
    out.append("")

    by_domain: dict[str, list[Mapping[str, Any]]] = {}
    for check in source["checks"]:
        by_domain.setdefault(check["domain"], []).append(check)

    for domain in source["domains"]:
        rows = by_domain.get(domain["id"], [])
        passed_d, total_d = domain["passing"], domain["checks"]
        pct = 100.0 * passed_d / total_d if total_d else 100.0
        failing_d = total_d - passed_d
        # Each domain is a collapsible group labelled with its compliance
        # percentage (100 % = every row passes). Groups with any failing row are
        # opened by default so regressions stay visible.
        opened = " open" if passed_d != total_d else ""
        domain_html = domain["title"].replace("&", "&amp;")
        out.append(f"<details{opened}>")
        tail = f" - <b>{failing_d} failing</b>" if failing_d else ""
        out.append(
            f"<summary><b>{domain_html}</b>: {pct:.0f}% "
            f"({passed_d}/{total_d}){tail}</summary>"
        )
        out.append("")
        out.append(
            "| Standard | Quantity | Expected (norm) | Computed | &#916; | Used | "
            "Status |"
        )
        out.append("|:---|:---|:---|:---|:---|:---:|:---:|")
        out.extend(
            f"| {_cell(check['reference']['cite'])} | {_cell(check['quantity'])} "
            f"| {_cell(expected_text(check))} | {_cell(computed_text(check))} "
            f"| {_cell(deviation_text(check))} | {_used(check)} "
            f"| {_status(check['verdict'])} |"
            for check in rows
        )
        out.append("")
        out.append("</details>")
        out.append("")

    return "\n".join(out), passed, total


# Header prepended to the committed docs/CONFORMANCE.md (via `--file-header`,
# used by `make conformance`). Emitted only for the committed file, not for the
# CI PR-comment body, so the PR comment stays header-free.
_DOC_HEADER = """<!--
  AUTO-GENERATED FILE - DO NOT EDIT BY HAND.
  The chain is: scripts/conformance/domains/ -> docs/conformance.json -> this
  file. Regenerate the whole of it with `make conformance` (runs
  scripts/conformance_report.py). CI regenerates it on every pull request and
  fails the build if it drifts.
-->

> **Auto-generated conformance report - do not hand-edit.** Produced by
> `make conformance` from the library's own computations checked against the
> referenced standards, and rendered from
> [`docs/conformance.json`](https://github.com/jmrplens/phonometry/blob/main/docs/conformance.json),
> which carries the same rows as data. CI regenerates it on every pull request
> and fails if it is out of date, so edit the checks in
> `scripts/conformance/domains/`, not this file. Each row pins a standard and clause to its expected normative value and
> the value the library computes. Full standards list and methodology:
> [Theory](https://github.com/jmrplens/phonometry/blob/main/docs/reference/theory/index.md) -
> [Why phonometry](https://github.com/jmrplens/phonometry/blob/main/docs/start/why-phonometry.md).

"""


def main(argv: list[str] | None = None) -> int:
    """Run every check, refresh the artefact, and print the report.

    Two steps, in this order and never the other: the checks produce
    ``docs/conformance.json``, and the Markdown is rendered from whatever is on
    disk afterwards. When the fresh run agrees with the committed artefact
    within tolerance the file is left untouched, so the Markdown printed here
    is a function of committed bytes and a byte diff of it means something.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--file-header",
        action="store_true",
        help="prepend the do-not-edit banner (for the committed docs/ copy)",
    )
    parser.add_argument(
        "--no-artifact",
        action="store_true",
        help="render from the committed artefact without running the checks",
    )
    args = parser.parse_args(argv)

    document = load() if args.no_artifact else write()[0]
    markdown, passed, total = render_markdown(document)
    output = _DOC_HEADER + markdown if args.file_header else markdown
    print(output)
    print(
        f"\n[conformance] {passed}/{total} checks passed, artefact at "
        f"{ARTIFACT_PATH.relative_to(_ROOT)}",
        file=sys.stderr,
    )
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
