#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Comparing two conformance documents: exact where it matters, tolerant where
a byte comparison would lie.

``docs/conformance.json`` cannot be byte-gated the way a hand-written file can.
GitHub's runner fleet is hardware-heterogeneous, so the same pinned stack
computes some values a unit in the last place apart, and a value that happens
to sit on a rounding boundary then stores one quantum away from the committed
one. The figures already met this and answered it the same way
(:mod:`scripts.check_figures`); the difference here is that the comparison can
be structure-aware rather than a walk over numeric tokens.

So: **structure, strings, ids, citations, ordering and the verdict are compared
exactly**, and only the last digits of a number may move. An added check, a
removed check, a renamed one, a reworded citation, a changed unit or a flipped
verdict all fail. What passes is a value that agrees to within one quantum of
the precision its own check declares - which is, by construction, the smallest
change that check considers worth displaying.

The gate cannot tell a rounding-boundary flip from a real move of exactly one
display quantum. Two things cover that: the verdict is exact, so a move that
matters has already failed, and ``test_every_check_passes`` executes all 552
checks for real in the test job, against no artefact at all.
"""

from __future__ import annotations

import pathlib
import sys
from typing import TYPE_CHECKING, Any

_SCRIPTS = str(pathlib.Path(__file__).resolve().parents[1])
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from generated_assets import NumericTolerance, numbers_within_tolerance

from .registry import Kind, deviation_places

if TYPE_CHECKING:
    from collections.abc import Mapping

#: Relative floor applied on top of the per-check quantum, so the judgement
#: stays meaningful for a value of 10 000 as well as for one of 0.001.
_RELATIVE = 1e-9

#: Decimals the showcase panels are rounded to. They carry no per-row
#: precision, being a fixed table, so one figure covers them.
_PANEL_PRECISION = 3

#: Fields of a check compared exactly. Everything that identifies the check,
#: says what it cites, or states the answer.
_EXACT_CHECK_FIELDS = (
    "id",
    "domain",
    "quantity",
    "implements",
    "kind",
    "unit",
    "precision",
    "verdict",
)


#: Decimals a record check's values are guarded at. They are stored unrounded
#: and compared for equality, so the guard has to be tight enough to be an
#: equality in practice while still absorbing the last-bit wobble that moving
#: the same computation between machines produces.
_RECORD_PLACES = 12


def _tolerance(precision: int, relative: float = _RELATIVE) -> NumericTolerance:
    """The tolerance for a value stored at ``precision`` decimals.

    :param precision: Decimals the value is stored at.
    :param relative: Fraction of the larger value also tolerated. A record
        side passes zero: its guard is the last bit and nothing else, and the
        default relative floor would swallow it whole. At ``_RECORD_PLACES``
        the absolute term is 1e-12, while 1e-9 of a stored 0.6 is 6e-10, six
        hundred times looser than the guard is written to be.
    :return: The tolerance.
    """
    return NumericTolerance(absolute=10.0**-precision, relative=relative)


def _as_number(value: object) -> float | None:
    """The leaf as a float, or ``None`` when a tolerance does not apply to it.

    ``bool`` is excluded deliberately: it is a subclass of ``int``, so a verdict
    that had somehow become a boolean would otherwise be compared within a
    numeric tolerance instead of exactly.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _number_problem(
    where: str,
    old: object,
    new: object,
    precision: int,
    relative: float = _RELATIVE,
) -> str | None:
    """Compare one leaf that may be a number, absent, or a string."""
    old_number, new_number = _as_number(old), _as_number(new)
    if old_number is None or new_number is None:
        return None if old == new else f"{where}: {old!r} -> {new!r}"
    tolerance = _tolerance(precision, relative)
    if numbers_within_tolerance(old_number, new_number, tolerance):
        return None
    return f"{where}: {old!r} -> {new!r} (beyond {precision}-decimal tolerance)"


def _mapping_problems(
    where: str,
    old: Mapping[str, Any] | None,
    new: Mapping[str, Any] | None,
    precision: int,
    relative: float = _RELATIVE,
) -> list[str]:
    """Compare two nested mappings of leaves, key set included."""
    if old is None or new is None:
        return [] if old == new else [f"{where}: {old!r} -> {new!r}"]
    if set(old) != set(new):
        return [f"{where}: fields {sorted(old)} -> {sorted(new)}"]
    problems: list[str] = []
    for key in old:
        inner_old, inner_new = old[key], new[key]
        if isinstance(inner_old, dict) or isinstance(inner_new, dict):
            problems += _mapping_problems(
                f"{where}.{key}", inner_old, inner_new, precision, relative
            )
            continue
        problem = _number_problem(
            f"{where}.{key}", inner_old, inner_new, precision, relative
        )
        if problem is not None:
            problems.append(problem)
    return problems


def _check_problems(old: Mapping[str, Any], new: Mapping[str, Any]) -> list[str]:
    """Compare one check against its counterpart."""
    where = old["id"]
    problems = [
        f"{where}.{field}: {old.get(field)!r} -> {new.get(field)!r}"
        for field in _EXACT_CHECK_FIELDS
        if old.get(field) != new.get(field)
    ]
    if old["reference"] != new["reference"]:
        problems.append(f"{where}.reference: citation split changed")
    precision = int(old["precision"])
    # A record check compares its values for equality and stores them
    # unrounded, so guarding them at the check's own precision guards nothing:
    # `record` sets precision to zero because its *deviation* is a count of
    # names that disagreed, and a zero-decimal tolerance is 1.0 absolute. A
    # stored 1.0 against a computed 0.6 sat inside it, so the artefact kept a
    # normative figure the standard does not contain and no regeneration could
    # dislodge it, because `write` only rewrites what this function objects to.
    is_record = old.get("kind") == str(Kind.RECORD)
    side_places = _RECORD_PLACES if is_record else precision
    side_relative = 0.0 if is_record else _RELATIVE
    for field in ("expected", "computed"):
        problems += _mapping_problems(
            f"{where}.{field}",
            old.get(field),
            new.get(field),
            side_places,
            side_relative,
        )
    for field in ("tolerance", "binding"):
        problems += _mapping_problems(
            f"{where}.{field}", old.get(field), new.get(field), precision
        )
    # The deviation is printed finer than the check's own precision -- at
    # ``deviation_places``, never below three decimals -- so guarding it at
    # ``precision`` leaves the published digits unguarded. For the 95 checks
    # declaring 0, 1 or 2 decimals that gap is real: a computed value can move
    # far enough to take a deviation from 3.911 to 4.401, a figure the report
    # publishes to three decimals, without this gate seeing it. Guard it at the
    # precision it is shown at.
    problems += _mapping_problems(
        f"{where}.deviation",
        old.get("deviation"),
        new.get("deviation"),
        deviation_places(precision),
    )
    return problems


def _roster_problems(old: Mapping[str, Any], new: Mapping[str, Any]) -> list[str]:
    """Compare which checks are present, and in what order."""
    old_ids = [check["id"] for check in old["checks"]]
    new_ids = [check["id"] for check in new["checks"]]
    if old_ids == new_ids:
        return []
    added = [check for check in new_ids if check not in set(old_ids)]
    removed = [check for check in old_ids if check not in set(new_ids)]
    if added or removed:
        return [f"checks added: {added}", f"checks removed: {removed}"]
    return ["checks reordered: the registration order changed"]


def document_problems(old: Mapping[str, Any], new: Mapping[str, Any]) -> list[str]:
    """Describe every way ``new`` differs from ``old`` beyond tolerance.

    :param old: The committed document.
    :param new: A freshly built one.
    :return: One line per difference, empty when the two agree.
    """
    problems = [
        f"{field}: {old.get(field)!r} -> {new.get(field)!r}"
        for field in ("schema", "library", "generator", "counts", "units", "domains")
        if old.get(field) != new.get(field)
    ]
    problems += _roster_problems(old, new)
    if problems:
        return problems
    problems += _mapping_problems(
        "panels", _indexed(old["panels"]), _indexed(new["panels"]), _PANEL_PRECISION
    )
    for old_check, new_check in zip(old["checks"], new["checks"], strict=True):
        problems += _check_problems(old_check, new_check)
    return problems


def _indexed(panels: list[dict[str, Any]]) -> dict[str, Any]:
    """Panels keyed by id, with their rows keyed by position.

    Turns two ragged lists into mappings so one comparison walks both, and so a
    difference is reported at ``panels.filter-class.rows.2.margin_class1``
    rather than as two lists a reader has to diff by eye.
    """
    return {
        panel["id"]: {key: value for key, value in panel.items() if key != "rows"}
        | {"rows": {str(index): row for index, row in enumerate(panel["rows"])}}
        for panel in panels
    }
