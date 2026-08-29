#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Gate for the committed ``docs/conformance.json``.

The artefact cannot be gated by a byte diff alone. GitHub's runner fleet is
hardware-heterogeneous, so the same pinned stack computes a few values a unit
in the last place apart, and a value sitting on a rounding boundary then stores
one quantum away from the committed one. The figures met this first and the
answer is the same here: compare structure exactly and numbers within a
tolerance (:mod:`conformance.compare`).

Two modes, because they cost two very different things.

``--validate`` (the default) reads the committed document and checks that it is
internally consistent - the counts agree with the rows, every leaf is a
built-in type, no two checks share an id, every unit is in the vocabulary, and
every citation still rebuilds from its split. It runs no check and needs no
scientific stack, so it is cheap enough to run beside every other read-only
gate. It is what catches a truncated write, a hand-edit and a numpy scalar.

``--regenerate`` runs all the checks and compares the result against the
committed document. This is the authoritative staleness gate, and it costs the
same as the harness because it *is* the harness.
"""

from __future__ import annotations

import argparse
import functools
import pathlib
import sys
from typing import TYPE_CHECKING, Any

_SCRIPTS = pathlib.Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from conformance.artifact import SCHEMA, build_document, load
from conformance.compare import document_problems
from conformance.references import OVERRIDES_PATH, Reference, ReferenceKind, recompose
from conformance.registry import Kind, Verdict
from conformance.units import UNITS

if TYPE_CHECKING:
    from collections.abc import Mapping

#: Leaves that must be a built-in ``float`` if they are present at all.
#: Checked with ``type(x) is float`` and never ``isinstance``: ``numpy.float64``
#: is a subclass of ``float``, so an ``isinstance`` test passes on exactly the
#: value that breaks ``json.dumps`` further down the chain.
_FLOAT_LEAVES = (
    ("expected", "value"),
    ("computed", "value"),
    ("tolerance", "value"),
    ("deviation", "value"),
)


def _count_problems(document: Mapping[str, Any]) -> list[str]:
    """The written counts must agree with the rows they count.

    This is the check a derived count cannot make: a truncated write derives a
    total that agrees with the rows that survived it.
    """
    counts, checks = document["counts"], document["checks"]
    passing = sum(1 for check in checks if check["verdict"] == str(Verdict.PASS))
    problems = []
    if counts["checks"] != len(checks):
        problems.append(
            f"counts.checks is {counts['checks']} but the document carries "
            f"{len(checks)} checks."
        )
    if counts["passing"] != passing:
        problems.append(
            f"counts.passing is {counts['passing']} but {passing} rows say pass."
        )
    if counts["passing"] + counts["failing"] != counts["checks"]:
        problems.append(
            f"counts.passing + counts.failing is "
            f"{counts['passing'] + counts['failing']}, not {counts['checks']}."
        )
    if counts["domains"] != len(document["domains"]):
        problems.append(
            f"counts.domains is {counts['domains']} but the document carries "
            f"{len(document['domains'])} domains."
        )
    return problems


def _type_problems(check: Mapping[str, Any]) -> list[str]:
    """Every numeric leaf of one check must be a built-in ``float``."""
    problems = []
    for outer, inner in _FLOAT_LEAVES:
        holder = check.get(outer)
        if not isinstance(holder, dict) or inner not in holder:
            continue
        value = holder[inner]
        if type(value) is not float:
            problems.append(
                f"{check['id']}.{outer}.{inner} is {type(value).__name__}, not "
                "float. A numpy scalar passes every isinstance test and fails "
                "json.dumps; coerce it in the check."
            )
    return problems


def _vocabulary_problems(check: Mapping[str, Any]) -> list[str]:
    """A check's unit, kind and verdict must all be in their vocabularies."""
    problems = []
    unit = check.get("unit")
    if unit is not None and unit not in UNITS:
        problems.append(
            f"{check['id']}.unit is {unit!r}, which is not in the vocabulary "
            "declared at the head of the document."
        )
    if check["kind"] not in tuple(Kind):
        problems.append(f"{check['id']}.kind is {check['kind']!r}.")
    if check["verdict"] not in tuple(Verdict):
        problems.append(f"{check['id']}.verdict is {check['verdict']!r}.")
    return problems


def _reference_problems(check: Mapping[str, Any]) -> list[str]:
    """The split must still rebuild the citation it came from.

    The whole reference split rests on this: three fields that cannot be
    reassembled into the original string have lost or moved something, and the
    designation count is then counting the wrong thing.
    """
    reference = check["reference"]
    if reference["kind"] not in tuple(ReferenceKind):
        return [f"{check['id']}.reference.kind is {reference['kind']!r}."]
    rebuilt = recompose(
        Reference(
            kind=ReferenceKind(reference["kind"]),
            designation=reference["designation"],
            edition=reference.get("edition"),
            clause=reference.get("clause"),
            cite=reference["cite"],
        )
    )
    if rebuilt == reference["cite"] or reference["cite"] in _overridden():
        return []
    return [
        f"{check['id']}: the citation split does not rebuild its citation. "
        f"{reference['designation']!r} + {reference.get('edition')!r} + "
        f"{reference.get('clause')!r} does not reproduce {reference['cite']!r}. "
        f"Fix the parser, or record the split in {OVERRIDES_PATH.name}."
    ]


@functools.cache
def _overridden() -> frozenset[str]:
    """Citations whose split is recorded by hand and is exempt from rebuilding.

    Cached: the round-trip is checked once per check, and the answer is one
    file that does not change while the process runs.
    """
    if not OVERRIDES_PATH.is_file():
        return frozenset()
    return frozenset(
        line.split("\t")[0]
        for line in OVERRIDES_PATH.read_text(encoding="utf8").splitlines()
        if line.strip() and not line.startswith("#")
    )


def _ratchet_problems(document: Mapping[str, Any]) -> list[str]:
    """The override file may only shrink.

    A line for a citation no check registers any more is dead weight that hides
    the next real one, so it fails just as loudly as a missing entry would.
    """
    cited = {check["reference"]["cite"] for check in document["checks"]}
    return [
        f"{OVERRIDES_PATH.name}: no check cites {cite!r} any more; delete the line."
        for cite in sorted(_overridden() - cited)
    ]


def validate(document: Mapping[str, Any]) -> list[str]:
    """Everything the committed document must be true about itself."""
    problems: list[str] = []
    if document.get("schema") != SCHEMA:
        problems.append(
            f"schema is {document.get('schema')!r}, this checkout reads {SCHEMA}."
        )
    problems += _count_problems(document)
    seen: set[str] = set()
    for check in document["checks"]:
        if check["id"] in seen:
            problems.append(f"duplicate check id {check['id']!r}.")
        seen.add(check["id"])
        problems += _type_problems(check)
        problems += _vocabulary_problems(check)
        problems += _reference_problems(check)
    problems += _ratchet_problems(document)
    return problems


def main(argv: list[str] | None = None) -> int:
    """Validate the committed artefact, or compare it against a fresh run."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--regenerate",
        action="store_true",
        help=(
            "run every check and compare the result against the committed "
            "document, instead of only validating what is committed"
        ),
    )
    args = parser.parse_args(argv)

    document = load()
    problems = validate(document)
    if args.regenerate:
        problems += document_problems(document, build_document())

    if problems:
        print(
            "docs/conformance.json is not consistent with the checks. "
            "Regenerate it with `make conformance` and commit the result:\n",
            file=sys.stderr,
        )
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        return 1

    counts = document["counts"]
    print(
        f"docs/conformance.json consistent: {counts['passing']}/{counts['checks']} "
        f"checks, {counts['domains']} domains, {counts['designations']} normative "
        f"designations, {counts['sources']} further sources."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
