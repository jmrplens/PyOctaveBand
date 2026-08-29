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

from . import marks
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
    from collections.abc import Mapping, Sequence

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

#: The ref every image in the committed report is pinned to. ``main``, because
#: this file lives on ``main`` and is read there, on github.com and through
#: raw: a reader browsing it has no commit in hand, and there is no commit to
#: pin to at the moment the file is written, since the file is generated
#: before it is committed.
#:
#: Two consequences to know about. A badge added on a branch 404s in this file
#: until the branch lands; that is the same trade the README's figures already
#: make, and the ``alt`` is the verdict, so a 404 degrades to the word. And
#: nothing pinned this way may be a picture of a *number*, because a number on
#: ``main`` is not the number in a branch's copy of this file - which is why
#: the summary banner is not embedded here (see :func:`render_markdown`) while
#: the four verdict marks, whose geometry never changes, are.
_REF = "main"


def _status(verdict: str) -> str:
    """One verdict as the table prints it: the mark, then the word.

    Both, and not either. The word is what survives a plain-text diff, a
    `raw` view and an image that never arrives; the mark is what tells a
    failing row from a passing one before anything is read. The image is
    reference-style, so the 117-character URL is written once for the whole
    document instead of once per row.

    The two land on separate lines, not side by side, and that was measured
    rather than intended: under GitHub's own table stylesheet the Status
    column settles at about 53.5 px of content box while a 16 px mark plus
    the word needs about 54, so at a 1012 px viewport 563 of 566 cells wrap
    the word under the mark and at 1400 px 493 do. A non-breaking space
    between them changes nothing, because GitHub caps the table at the
    content width and the column has no room to grow into. It costs no
    height - the rows are already two lines tall from the Quantity column -
    so the stack is left alone and described here instead of being fought.
    """
    word = _VERDICT_TEXT.get(verdict, verdict)
    if verdict not in marks.MARK_OF:
        return word
    return f"{marks.mark_reference(verdict)} {word}"


def _marked(document: Mapping[str, Any]) -> list[str]:
    """The verdicts this report actually prints a mark for, in legend order.

    Read off the document rather than listed, so the legend explains the marks
    on the page and no others: a legend entry for a silhouette that appears
    nowhere is a promise the report does not keep, and the day a check fails
    the legend gains its red hexagon without anyone editing it.
    """
    used = {check["verdict"] for check in document["checks"]}
    used |= {row["verdict"] for row in _panel(document, "filter-class")["rows"]}
    return [mark.verdict for mark in marks.MARKS if mark.verdict in used]


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
    """The class verdict of one showcase row, as the table prints it.

    Marked like a conformance row, and this is the only table where a mark
    other than the pass disc appears today: three of the five architectures
    are excluded from the class mask on purpose, and a blue bar says "out of
    scope" where a red cross would say "wrong".
    """
    if row["verdict"] == str(Verdict.BY_DESIGN):
        text = f"By design ({row['reason']})"
    elif "class" not in row:
        text = "not compliant"
    elif row["class"] == 1:
        text = "Class 1 (default)" if row["architecture"] == "butter" else "Class 1"
    else:
        text = f"Class {row['class']}"
    if row["verdict"] not in marks.MARK_OF:
        return text
    return f"{marks.mark_reference(row['verdict'])} {text}"


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


#: How many rows the "closest to their limit" table prints. Ten, the same as
#: the site page, because the two are the same answer to the same question and
#: a reader comparing them should not have to work out which was truncated.
_CLOSEST_ROWS = 10


def _closest_section(document: Mapping[str, Any]) -> str:
    """The checks with the least room left, ranked.

    The single most informative thing the artefact holds, and until now the
    one thing the published report did not print: the ``Used`` column already
    gives it per row, but spread over 566 rows inside 59 collapsed sections
    nobody can see which rows are near their limit. The pull-request comment
    has ranked them since it was written and the site page ranks ten of them;
    this makes the three surfaces answer the question the same way.

    A high figure is not a failure - it is the room the check passes with -
    so the caption says so, in the same words the site uses.
    """
    ranked = sorted(
        (
            check
            for check in document["checks"]
            if utilisation(check) is not None  # nothing to be a fraction of
        ),
        key=lambda check: utilisation(check) or 0.0,
        reverse=True,
    )
    if not ranked:
        return ""
    lines = [
        f"**The {_CLOSEST_ROWS} checks closest to their published limit**",
        "",
        "<sub>The fraction of its published tolerance each check consumes. "
        "A high figure is not a failure: it is the room the check passes "
        "with, and it never decides the verdict, which is settled at full "
        "precision before any rounding. These are the rows a change is most "
        "likely to push over.</sub>",
        "",
        "| Standard | Quantity | Computed | &#916; | Used |",
        "|:---|:---|:---|:---|:---:|",
    ]
    lines += [
        f"| {_cell(check['reference']['cite'])} | {_cell(check['quantity'])} "
        f"| {_cell(computed_text(check))} | {_cell(deviation_text(check))} "
        f"| {_used(check)} |"
        for check in ranked[:_CLOSEST_ROWS]
    ]
    return "\n".join(lines)


def _verdict_legend(verdicts: Sequence[str]) -> str:
    """The marks on this page, each beside the word it stands for.

    A legend of pictures alone would only be a second puzzle, so every entry
    pairs the mark with the word the Status column prints next to it, and the
    sentence says the part a picture cannot: that the silhouette carries the
    meaning as much as the colour does, which is what keeps the four apart in
    greyscale and for a reader who cannot separate the hues.
    """
    entries = " · ".join(
        f"{marks.mark_reference(verdict)} {marks.MARK_OF[verdict].alt}"
        for verdict in verdicts
    )
    return (
        f"<sub><b>Verdict marks.</b> {entries}. Every mark travels with the "
        "verdict in words and never replaces it, so the verdict survives an "
        "image that never arrives; the silhouettes differ as much as the "
        "colours do, so a reader who cannot separate the hues can still tell "
        "them apart.</sub>"
    )


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
    # No summary banner here, and this is the one place in the project that
    # goes without one. This file is regenerated from the tree it sits in and
    # gated byte for byte, so it claims to be *this* tree's report; the banner
    # can only be cited at a fixed ref (see _REF), and a picture of main's
    # count sitting above a sentence carrying the branch's count is a document
    # arguing with itself. Any pull request that adds or removes a check would
    # show it. The pull-request comment still leads with the banner, pinned to
    # the head commit, where the picture and the sentence come from one tree;
    # here the sentence carries every number the picture would, and two class
    # claims no bar can state.
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
    marked = _marked(source)
    out.append(_verdict_legend(marked))
    out.append("")
    closest = _closest_section(source)
    if closest:
        out.append(closest)
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

    # One definition per mark, at the foot of the document. A link reference
    # resolves document-wide in CommonMark, which is what lets a row spend 16
    # characters on its mark instead of the 117 an inline URL costs.
    out.extend(marks.mark_definitions(marked, _REF))
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
