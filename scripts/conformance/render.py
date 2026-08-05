#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Turning the registry into the Markdown the report is.

One renderer serves two readers: the sticky pull-request comment CI posts, and
the committed ``docs/CONFORMANCE.md``. Both get the headline count, the
filters and weightings showcase, then one collapsible section per domain in
registration order. Only the committed file also gets :data:`_DOC_HEADER`, the
do-not-edit banner, so the PR comment stays header-free.

The collapsing rule is the point of the layout: a section stays shut while
every one of its rows passes and springs open the moment one fails, so a long
green report reads as a single line and a regression is the only thing open on
the page. The escaping in :func:`_cell` exists for the same reason - an
unescaped ``|`` in a quantity written the way acousticians write it silently
drops the evidence columns off the right of the row.
"""

from __future__ import annotations

import re
import sys

from .registry import _ROOT, CHECKS
from .shared import _FILTER_ARCHS, FilterClass, _filter_class, _weighting_deviation


def _snap(value: float, eps: float = 5e-4) -> float:
    """Snap a near-zero value to +0 so displays avoid a spurious ``-0.00``."""
    return 0.0 if abs(value) < eps else value


def _status(passed: bool) -> str:
    return "&#9989;" if passed else "&#10060;"


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


def _domains() -> list[str]:
    seen: list[str] = []
    for chk in CHECKS:
        if chk.domain not in seen:
            seen.append(chk.domain)
    return seen


# Architectures that cannot meet the IEC 61260-1 mask by construction, with
# the reason (for the "By design" label, not a failure verdict).
_BY_DESIGN: dict[str, str] = {
    "cheby1": "passband ripple",
    "ellip": "passband ripple",
    "bessel": "soft rolloff",
}


def _filter_verdict(arch: str, fc: FilterClass) -> str:
    if fc.overall_class == 1:
        return "Class 1 (default)" if arch == "butter" else "Class 1"
    if fc.overall_class == 2:
        return "Class 2"
    reason = _BY_DESIGN.get(arch)
    return f"By design ({reason})" if reason else "not compliant"


def _numerical_validation_section(filters_ok: bool) -> str:
    # Collapsed like the per-domain groups so the report stays compact by
    # default; it springs open whenever the filters/weightings domain has a
    # failing row, exactly like those groups do.
    emoji = "&#9989;" if filters_ok else "&#10060;"
    opened = "" if filters_ok else " open"
    lines: list[str] = []
    lines.append(f"<details{opened}>")
    lines.append(
        f"<summary>{emoji} <b>Numerical validation - filters &amp; "
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
    for arch in _FILTER_ARCHS:
        fc = _filter_class(arch, 3)
        if fc.bind_side == "ceil":
            req = f"&le; {fc.bind_limit_db:+.2f} dB"
        else:
            req = f"&ge; {fc.bind_limit_db:+.2f} dB"
        lines.append(
            f"| {arch} | {_filter_verdict(arch, fc)} | {fc.bind_freq:.0f} Hz "
            f"| {_snap(fc.bind_measured_db, 5e-3):+.2f} dB | {req} "
            f"| {fc.min_margin1:+.3f} dB | {fc.min_margin2:+.3f} dB |"
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
    for curve, fs in [("A", 48000), ("A", 96000), ("C", 48000), ("G", 48000)]:
        wd = _weighting_deviation(curve, fs)
        band = f"[{wd.bind_lower:+.2f}, {wd.bind_upper:+.2f}] dB"
        lines.append(
            f"| {curve} | {fs // 1000} kHz "
            f"| {wd.worst_dev:+.3f} dB @ {wd.worst_freq:.0f} Hz "
            f"| {wd.bind_freq:.0f} Hz | {_snap(wd.bind_dev):+.3f} dB | {band} "
            f"| {wd.min_headroom:+.3f} dB |"
        )
    lines.append("")
    lines.append("</details>")
    return "\n".join(lines)


def render_markdown() -> tuple[str, int, int]:
    """Render the full conformance report. Returns (markdown, passed, total)."""
    results = [(chk, chk.run()) for chk in CHECKS]
    passed = sum(1 for _, o in results if o.passed)
    total = len(results)

    filters_ok = all(
        o.passed for c, o in results if c.domain == "Filters & weightings"
    )
    headline_emoji = "&#9989;" if passed == total else "&#10060;"

    out: list[str] = []
    out.append("## Numerical conformance report")
    out.append("")
    summary = (
        f"**{passed}/{total} conformance checks pass** across "
        f"{len(_domains())} domains and {len({c.standard.split(':')[0].split(' Annex')[0] for c, _ in results})} standards"
    )
    if filters_ok:
        summary += " - filters class 1 - weightings within IEC 61672-1 class 1"
    out.append(f"{headline_emoji} {summary}.")
    out.append("")
    out.append(
        "<sub>Each row pins a standard clause to its expected normative value "
        "and the value the library computes. Every section below is "
        "collapsible and stays collapsed while all of its rows pass; a "
        "section with any failing row opens automatically.</sub>"
    )
    out.append("")
    out.append(_numerical_validation_section(filters_ok))
    out.append("")

    for domain in _domains():
        rows = [(chk, o) for chk, o in results if chk.domain == domain]
        passed_d = sum(1 for _, o in rows if o.passed)
        total_d = len(rows)
        pct = 100.0 * passed_d / total_d if total_d else 100.0
        emoji = "&#9989;" if passed_d == total_d else "&#10060;"
        # Each domain is a collapsible group labelled with its compliance
        # percentage (100 % = every row passes). Groups with any failing row are
        # opened by default so regressions stay visible.
        opened = " open" if passed_d != total_d else ""
        domain_html = domain.replace("&", "&amp;")
        out.append(f"<details{opened}>")
        out.append(
            f"<summary>{emoji} <b>{domain_html}</b>: {pct:.0f}% "
            f"({passed_d}/{total_d})</summary>"
        )
        out.append("")
        out.append("| Standard | Quantity | Expected (norm) | Computed | &#916; | Status |")
        out.append("|:---|:---|:---|:---|:---|:---:|")
        for chk, outcome in rows:
            out.append(
                f"| {_cell(chk.standard)} | {_cell(chk.quantity)} "
                f"| {_cell(outcome.expected)} | {_cell(outcome.computed)} "
                f"| {_cell(outcome.delta)} | {_status(outcome.passed)} |"
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
  Regenerate with `make conformance` (runs scripts/conformance_report.py).
  CI regenerates it on every pull request and fails the build if it drifts.
-->

> **Auto-generated conformance report - do not hand-edit.** Produced by
> `make conformance` from the library's own computations checked against the
> referenced standards. CI regenerates it on every pull request and fails if it
> is out of date, so edit the checks in `scripts/conformance/domains/`, not this
> file. Each row pins a standard and clause to its expected normative value and
> the value the library computes. Full standards list and methodology:
> [Theory](https://github.com/jmrplens/phonometry/blob/main/docs/reference/theory/index.md) -
> [Why phonometry](https://github.com/jmrplens/phonometry/blob/main/docs/start/why-phonometry.md).

"""


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    markdown, passed, total = render_markdown()
    # The root artifact feeds the CI PR comment; keep it header-free.
    (_ROOT / "conformance_report.md").write_text(markdown + "\n")
    output = _DOC_HEADER + markdown if "--file-header" in args else markdown
    print(output)
    print(f"\n[conformance] {passed}/{total} checks passed", file=sys.stderr)
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
