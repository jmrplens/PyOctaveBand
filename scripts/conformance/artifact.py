#  Copyright (c) 2026. Jose Manuel Requena Plens
"""The registry as one committed document.

``docs/conformance.json`` is what the checks produce and what every consumer
reads: the Markdown mirror, the two site pages, the counts quoted in the prose,
and the pull-request comment. Before it existed each of those parsed the
report's headline sentence with its own regular expression, in two languages,
and the limit every check is judged against - the number a metrology library is
asked for first - was not written down anywhere at all.

Three properties make it committable.

**It is a function of the source tree and nothing else.** No timestamp, no git
SHA, no numpy version. Two runs of the same tree produce the same bytes.

**Only computed values are rounded.** A published limit and a published
expected value are stored as they were authored, because rounding a normative
figure is not a display choice; the values the library computed are rounded to
the check's own ``precision``, which is what keeps the file stable across the
BLAS builds GitHub's runner fleet mixes. Deviations get the same precision with
a three-decimal floor, because the precision suits the value and a deviation is
a much smaller number than the value it is a deviation from. The old report
capped every deviation at three decimals for the cross-build reason alone,
coarsening the evidence of every check to suit the noisiest; the cap is gone and
the floor remains.

**The verdict is stored, never re-derived.** It is decided at full precision
inside the check, before anything is rounded. A deviation one quantum below its
limit rounds onto it, so a consumer that recomputed ``|deviation| <= tolerance``
from the stored numbers would flip a verdict the harness got right.
"""

from __future__ import annotations

import json
import math
import re
from typing import TYPE_CHECKING, Any

from .compare import document_problems
from .references import Reference, ReferenceKind, parse
from .registry import _ROOT, CHECKS, Kind, Outcome, Verdict, _snap, deviation_places
from .shared import _FILTER_ARCHS, _filter_class, _weighting_deviation
from .units import UNITS

if TYPE_CHECKING:
    import pathlib
    from collections.abc import Iterable, Mapping

    from .registry import Check

#: Bumped only when the shape changes in a way a reader cannot ignore.
SCHEMA = 1

#: Where the document is committed. Beside the Markdown it generates, in the
#: directory the project already declares as the home of its numerical
#: evidence, and on the path ``conformance-stats.mjs`` already walks up to find.
ARTIFACT_PATH = _ROOT / "docs" / "conformance.json"

#: Significant digits kept when a value is smaller than its own display
#: precision. Rounding 4.2e-7 to five decimals would store a zero deviation
#: against a non-zero limit, which is the coarsening this file exists to end;
#: three significant digits is orders of magnitude coarser than the ~1 ULP
#: cross-build drift and orders finer than any real movement.
_FALLBACK_SIGNIFICANT = 3

_SLUG_STRIP = re.compile(r"[^a-z0-9]+")


def slug(text: str) -> str:
    """Reduce a title, citation or quantity to an id fragment.

    :param text: Any of the three strings a check registers under.
    :return: Lowercase, hyphen-separated, ASCII.
    """
    return _SLUG_STRIP.sub("-", text.lower()).strip("-")


def check_id(check: Check) -> str:
    """The stable join key for one check.

    Derived from the three strings the check is registered under, so it is a
    function of the registry and of nothing else. Rewording a quantity renames
    the row and the pull-request diff reports it as a removal plus an addition,
    which is visible in review - preferable to a hidden identity that quietly
    reattaches history to a different check.

    :param check: The registered check.
    :return: ``domain/citation/quantity``, each part slugged.
    """
    return f"{slug(check.domain)}/{slug(check.standard)}/{slug(check.quantity)}"


def _rounded(value: float, precision: int) -> float:
    """Round a computed value to the precision its check reports at.

    :param value: The value the library computed.
    :param precision: Decimals the check declares.
    :return: The rounded value, with negative zero normalised away
        (``-0.0 + 0.0`` is ``0.0``), so a value that rounds down to nothing
        cannot flip the sign of a byte comparison.
    :raises ValueError: If the value is not finite. ``Infinity`` is not JSON
        and would throw in the site build, so it fails here, at write time,
        where the check that produced it can still be named.
    """
    if not math.isfinite(value):
        msg = f"non-finite value {value!r} cannot be stored in the artefact."
        raise ValueError(msg)
    coarse = round(value, precision) + 0.0
    if coarse == 0.0 and value != 0.0:
        return float(f"{value:.{_FALLBACK_SIGNIFICANT}g}") + 0.0
    return coarse


def _exact(value: float) -> float:
    """Store a published number as authored, with negative zero normalised.

    Expected values and tolerances come off a printed table; they are not
    measurements and are not rounded to a display precision.
    """
    if not math.isfinite(value):
        msg = f"non-finite published value {value!r} cannot be stored."
        raise ValueError(msg)
    return float(value) + 0.0


def _reference_document(reference: Reference) -> dict[str, Any]:
    """The reference of one check, as the document carries it."""
    return {
        "kind": str(reference.kind),
        "designation": reference.designation,
        "edition": reference.edition,
        "clause": reference.clause,
        "cite": reference.cite,
    }


def _expected_document(outcome: Outcome) -> dict[str, Any]:
    """The normative side of a check."""
    side = outcome.expected_data
    return {
        "value": None if side.value is None else _exact(side.value),
        "label": side.label,
        "record": None if side.record is None else dict(side.record),
    }


def _computed_document(outcome: Outcome) -> dict[str, Any]:
    """The library's side of a check, rounded to the check's precision."""
    side = outcome.computed_data
    record = side.record
    return {
        "value": (
            None if side.value is None else _rounded(side.value, outcome.precision)
        ),
        "label": side.label,
        "record": (
            None
            if record is None
            else {
                name: _rounded(value, outcome.precision)
                for name, value in record.items()
            }
        ),
    }


def _tolerance_document(outcome: Outcome) -> dict[str, Any] | None:
    """The published limit, or ``None`` where the check declares none."""
    tolerance = outcome.tolerance
    if tolerance is None:
        return None
    return {
        "mode": str(tolerance.mode),
        "value": None if tolerance.value is None else _exact(tolerance.value),
    }


def _binding_document(outcome: Outcome) -> dict[str, Any] | None:
    """Where a mask binds, and the band it must sit inside there."""
    binding = outcome.binding
    if binding is None:
        return None
    return {
        "frequency_hz": (
            None if binding.frequency_hz is None else _exact(binding.frequency_hz)
        ),
        "lower": None if binding.lower is None else _exact(binding.lower),
        "upper": None if binding.upper is None else _exact(binding.upper),
    }


def _check_document(
    check: Check, outcome: Outcome, reference: Reference
) -> dict[str, Any]:
    """One check, as the document carries it."""
    deviation = outcome.deviation
    value = None if deviation is None else deviation.value
    return {
        "id": check_id(check),
        "domain": slug(check.domain),
        "reference": _reference_document(reference),
        "quantity": check.quantity,
        # Nothing records which public symbol a check exercises yet. The field
        # is here so the shape does not change on the day it does, and so the
        # site can link a row to its API page without a schema bump.
        "implements": None,
        "kind": str(outcome.kind),
        "expected": _expected_document(outcome),
        "computed": _computed_document(outcome),
        "unit": outcome.unit,
        "tolerance": _tolerance_document(outcome),
        "deviation": {
            # Stored at the deviation floor, not at the value precision: a
            # distance reported to the foot still has a deviation of hundredths
            # of one, and rounding the two the same way reports it as zero.
            "value": (
                None
                if value is None
                else _rounded(value, deviation_places(outcome.precision))
            ),
            "label": None if deviation is None else deviation.label,
        },
        "binding": _binding_document(outcome),
        "precision": outcome.precision,
        "verdict": str(outcome.verdict),
    }


def _domain_documents(
    results: list[tuple[Check, Outcome]],
) -> list[dict[str, Any]]:
    """The domains, in registration order, each owning its display title."""
    order: list[str] = []
    for check, _ in results:
        if check.domain not in order:
            order.append(check.domain)
    documents = []
    for title in order:
        rows = [outcome for check, outcome in results if check.domain == title]
        documents.append(
            {
                "id": slug(title),
                "title": title,
                "checks": len(rows),
                "passing": sum(1 for outcome in rows if outcome.passed),
            }
        )
    return documents


# Architectures that cannot meet the IEC 61260-1 mask by construction, with the
# reason. The report has always labelled these "By design" in a display string;
# as a verdict it can be counted.
_BY_DESIGN: dict[str, str] = {
    "cheby1": "passband ripple",
    "ellip": "passband ripple",
    "bessel": "soft rolloff",
}

#: The weighting curves and rates the showcase table reports, in table order.
_WEIGHTING_PANEL_ROWS = (("A", 48000), ("A", 96000), ("C", 48000), ("G", 48000))


def _filter_panel() -> dict[str, Any]:
    """The IEC 61260-1 class showcase, as data.

    Kept out of ``checks`` deliberately: folding the five architectures in
    would move ``counts.checks`` off 552, and that number is published.
    """
    rows = []
    for architecture in _FILTER_ARCHS:
        result = _filter_class(architecture, 3)
        rows.append(
            {
                "architecture": architecture,
                "verdict": str(_panel_verdict(architecture, result.overall_class)),
                "class": result.overall_class,
                "reason": _BY_DESIGN.get(architecture),
                "binding": {
                    "frequency_hz": _rounded(result.bind_freq, 0),
                    # Snapped before rounding, as the table has always printed
                    # it: a measured value a thousandth of a decibel below zero
                    # is zero at two decimals, and printing it as "-0.00" reads
                    # as a deficit that is not there.
                    "measured": _rounded(_snap(result.bind_measured_db, 5e-3), 2),
                    "limit": _exact(result.bind_limit_db),
                    "side": result.bind_side,
                },
                "margin_class1": _rounded(result.min_margin1, 3),
                "margin_class2": _rounded(result.min_margin2, 3),
            }
        )
    return {
        "id": "filter-class",
        "title": "IEC 61260-1:2014 class per filter architecture",
        "unit": "dB",
        "rows": rows,
    }


def _panel_verdict(architecture: str, overall_class: int | None) -> Verdict:
    """Verdict of one showcase row, with ``by-design`` as a first-class value."""
    if overall_class in (0, 1, 2):
        return Verdict.PASS
    return Verdict.BY_DESIGN if architecture in _BY_DESIGN else Verdict.FAIL


def _weighting_panel() -> dict[str, Any]:
    """The A/C/G weighting deviation showcase, as data."""
    rows = []
    for curve, rate in _WEIGHTING_PANEL_ROWS:
        deviation = _weighting_deviation(curve, rate)
        rows.append(
            {
                "curve": curve,
                "fs_hz": rate,
                "worst": {
                    "frequency_hz": _rounded(deviation.worst_freq, 0),
                    "deviation": _rounded(deviation.worst_dev, 3),
                },
                "binding": {
                    "frequency_hz": _rounded(deviation.bind_freq, 0),
                    "deviation": _rounded(_snap(deviation.bind_dev), 3),
                    "lower": _exact(deviation.bind_lower),
                    "upper": _exact(deviation.bind_upper),
                },
                "headroom": _rounded(deviation.min_headroom, 3),
                "verdict": str(
                    Verdict.PASS if deviation.min_headroom >= 0.0 else Verdict.FAIL
                ),
            }
        )
    return {
        "id": "weighting-deviation",
        "title": "Frequency-weighting conformance",
        "unit": "dB",
        "rows": rows,
    }


def _counts(
    results: list[tuple[Check, Outcome]],
    references: Mapping[str, Reference],
    domains: int,
) -> dict[str, int]:
    """The integers the prose quotes, written rather than left to be derived.

    Written for two reasons. A consumer that needs only the headline should not
    have to reduce a 300 kB array to get it, and an explicit count lets a gate
    assert ``counts.checks == len(checks)`` - the one thing a derived count can
    never catch, because a truncated write derives a count that agrees with
    itself.

    ``standards`` is the figure the landing page, both READMEs, ``llms.txt``
    and ``.zenodo.json`` publish, and it counts *citation groups*: the citation
    string up to its first colon or " Annex". By that rule seven clause strings
    of one book are seven standards. ``designations`` and ``sources`` are the
    honest split the reference parser makes possible - distinct normative
    documents, and distinct further works - and are reported alongside rather
    than in place of it, because changing a published claim is the maintainer's
    call and not a side effect of a refactor.
    """
    passing = sum(1 for _, outcome in results if outcome.passed)
    designations = {
        reference.designation
        for reference in references.values()
        if reference.kind is ReferenceKind.STANDARD
    }
    sources = {
        reference.designation
        for reference in references.values()
        if reference.kind is not ReferenceKind.STANDARD
    }
    legacy = {check.standard.split(":")[0].split(" Annex")[0] for check, _ in results}
    return {
        "checks": len(results),
        "passing": passing,
        "failing": len(results) - passing,
        "domains": domains,
        "standards": len(legacy),
        "citations": len(references),
        "designations": len(designations),
        "sources": len(sources),
    }


def _library_version() -> str:
    """The version that produced these numbers."""
    return (_ROOT / "VERSION").read_text(encoding="utf8").strip()


def build_document() -> dict[str, Any]:
    """Run every registered check and assemble the whole document.

    :return: The document, ready to serialise.
    :raises ValueError: If two checks derive the same id, which would make the
        pull-request diff join rows that are not the same check.
    """
    results = [(check, check.run()) for check in CHECKS]
    references = {check.standard: parse(check.standard) for check, _ in results}
    domains = _domain_documents(results)
    checks = [
        _check_document(check, outcome, references[check.standard])
        for check, outcome in results
    ]
    _reject_duplicate_ids(checks)
    return _without_nulls(
        {
            "schema": SCHEMA,
            "library": _library_version(),
            "generator": "scripts/conformance_report.py",
            "counts": _counts(results, references, len(domains)),
            "units": sorted(UNITS),
            "domains": domains,
            "panels": [_filter_panel(), _weighting_panel()],
            "checks": checks,
        }
    )


def _without_nulls(document: dict[str, Any]) -> dict[str, Any]:
    """Drop every key whose value is null, throughout the document.

    A check that has no tolerance, no binding, no record and no label carries
    eight null leaves, and 554 checks carry them 554 times: writing them out
    costs a hundred kilobytes of committed file to say nothing. Absence and
    ``null`` mean the same thing here - the check does not have that - and
    every consumer reads these fields through a schema that already declares
    them optional, so nothing has to distinguish the two.

    :param value: Any part of the document.
    :return: The same structure with its null-valued keys removed.
    """
    return {key: _pruned(inner) for key, inner in document.items() if inner is not None}


def _pruned(value: object) -> object:
    """Drop the null-valued keys from one branch of the document."""
    if isinstance(value, dict):
        return {
            key: _pruned(inner) for key, inner in value.items() if inner is not None
        }
    if isinstance(value, list):
        return [_pruned(item) for item in value]
    return value


def _reject_duplicate_ids(checks: Iterable[dict[str, Any]]) -> None:
    """Fail on two checks sharing an id."""
    seen: set[str] = set()
    repeated: set[str] = set()
    for check in checks:
        if check["id"] in seen:
            repeated.add(check["id"])
        seen.add(check["id"])
    duplicates = sorted(repeated)
    if duplicates:
        msg = (
            "duplicate conformance check id(s): "
            + ", ".join(duplicates)
            + ". Two checks share a domain, a citation and a quantity; give one "
            "of them a quantity that says what it tests differently."
        )
        raise ValueError(msg)


def dumps(document: Mapping[str, Any]) -> str:
    """Serialise the document the one way it is ever serialised.

    ``allow_nan=False`` is the guard that matters: Python would happily emit
    ``Infinity``, which is not JSON, and ``JSON.parse`` in the site build would
    throw on it at a point far from the check that produced it.
    """
    return (
        json.dumps(
            document,
            indent=2,
            ensure_ascii=False,
            sort_keys=False,
            allow_nan=False,
        )
        + "\n"
    )


def write(path: pathlib.Path | None = None) -> tuple[dict[str, Any], bool]:
    """Write the document, leaving the committed bytes alone when nothing moved.

    The file is rewritten only when the fresh run differs from what is on disk
    by more than the numeric tolerance - a changed verdict, an added or removed
    check, a moved citation, or a number that moved further than a rounding
    wobble. Two runs of the same tree on different hardware therefore leave the
    committed file untouched, which is what lets ``docs/CONFORMANCE.md`` keep a
    byte gate: the Markdown is rendered from these bytes, so it is a pure
    function of something already committed.

    :param path: Where to write; defaults to :data:`ARTIFACT_PATH`.
    :return: The document that is now on disk, and whether it was rewritten.
    """
    target = ARTIFACT_PATH if path is None else path
    fresh = build_document()
    if target.is_file():
        committed = json.loads(target.read_text(encoding="utf8"))
        if not document_problems(committed, fresh):
            return committed, False
    target.write_text(dumps(fresh), encoding="utf8")
    return fresh, True


def load(path: pathlib.Path | None = None) -> dict[str, Any]:
    """Read the committed document.

    :param path: Where to read from; defaults to :data:`ARTIFACT_PATH`.
    :raises SystemExit: If the file is missing, which means the artefact was
        never generated in this checkout.
    """
    target = ARTIFACT_PATH if path is None else path
    if not target.is_file():
        msg = f"{target}: missing. Generate it with `make conformance`."
        raise SystemExit(msg)
    document: dict[str, Any] = json.loads(target.read_text(encoding="utf8"))
    return document


def is_pass(verdict: str) -> bool:
    """Whether a stored verdict counts towards the passing total."""
    return verdict == str(Verdict.PASS)


def kind_of(check: Mapping[str, Any]) -> Kind:
    """The stored shape of a check, as an enum."""
    return Kind(check["kind"])
