#  Copyright (c) 2026. Jose Manuel Requena Plens
"""The conformance artefact says what the checks said, and stays saying it.

``docs/conformance.json`` is committed, so three properties have to hold or the
file is worse than no file: it must be a function of the source tree alone
(same tree, same bytes, on any machine), everything in it must be a built-in
type ``json.dumps`` can write, and the verdict it carries must be the verdict
the check decided at full precision rather than one re-derived from rounded
numbers.

The guards below are the ones that would otherwise fail somewhere far away: a
numpy scalar fails at serialisation, a non-finite value fails inside the site
build's ``JSON.parse``, a lost citation fragment fails as a wrong published
count, and a verdict re-derived at the boundary fails as a green report for a
check that does not pass.
"""

from __future__ import annotations

import json
import math
import pathlib
import re
import sys

import numpy as np
import pytest

_SCRIPTS = str(pathlib.Path(__file__).resolve().parent.parent / "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import check_conformance_artifact as gate
import conformance_report as cr
from conformance import artifact, compare, metrics, references, registry, units

# One xdist worker runs this module with the report smoke tests: the registry
# memoizes each check per process, so building the document here and rendering
# the report there compute every check once instead of once per worker.
pytestmark = pytest.mark.xdist_group("conformance-report")


@pytest.fixture(scope="module")
def committed() -> dict:
    """The artefact as committed, which is what every consumer reads."""
    return artifact.load()


@pytest.fixture(scope="module")
def fresh() -> dict:
    """A document built from the registry in this process."""
    return artifact.build_document()


# --------------------------------------------------------------------------
# Types at the boundary
# --------------------------------------------------------------------------


def test_a_numpy_verdict_is_coerced_to_a_builtin_bool() -> None:
    """``numpy.bool_`` is not JSON-serialisable and nine checks produced one.

    ``numpy.float64`` subclasses ``float``, so a numpy scalar reaches
    :func:`numeric` with no annotation and no type checker objecting, and
    ``abs(delta) <= limit`` then hands back a ``numpy.bool_``. Coercing in the
    constructor is what catches the outcomes built by hand as well.
    """
    outcome = registry.Outcome(
        expected="1", computed="1", delta="0", passed=np.bool_(True)
    )
    assert type(outcome.passed) is bool
    assert outcome.verdict is registry.Verdict.PASS


def test_numeric_coerces_numpy_scalars_on_the_way_in() -> None:
    outcome = registry.numeric(np.float64(2.0), np.float64(2.5), np.float64(1.0))
    assert type(outcome.passed) is bool
    assert type(outcome.deviation.value) is float


def test_every_stored_number_is_a_builtin_float(committed: dict) -> None:
    """Checked with ``type(x) is float``, never ``isinstance``.

    ``isinstance(numpy.float64(1.0), float)`` is ``True``, so an isinstance
    test passes on exactly the value that breaks the write.
    """
    assert gate.validate(committed) == []


def test_the_committed_document_is_internally_consistent(committed: dict) -> None:
    counts = committed["counts"]
    assert counts["checks"] == len(committed["checks"])
    assert counts["passing"] + counts["failing"] == counts["checks"]
    assert counts["domains"] == len(committed["domains"])


# --------------------------------------------------------------------------
# Rounding, verdicts and reproducibility
# --------------------------------------------------------------------------


def test_the_verdict_survives_a_deviation_that_rounds_onto_its_limit() -> None:
    """The boundary case the stored verdict exists for.

    A deviation of 0.0499 against a limit of 0.05 passes at full precision and
    rounds to 0.05 at two decimals, where a consumer recomputing
    ``|deviation| <= tolerance`` would see equality and a consumer using ``<``
    would flip it. The artefact stores what the check decided.
    """
    outcome = registry.numeric(1.0, 1.0499, 0.05, places=2)
    assert outcome.verdict is registry.Verdict.PASS
    stored = artifact._rounded(outcome.deviation.value, 2)
    assert stored == pytest.approx(0.05)


def test_a_deviation_is_never_reported_coarser_than_three_decimals() -> None:
    """A distance printed to the foot still has a deviation of hundredths.

    Applying the value's precision to its deviation reported a real 0.036 ft as
    zero, which is the coarsening the per-check precision was meant to end.
    """
    outcome = registry.numeric(5280.0, 5280.036, 0.1, unit="ft", places=0)
    assert outcome.delta == "0.036 ft"
    assert registry.deviation_places(0) == 3
    assert registry.deviation_places(6) == 6


def test_negative_zero_is_normalised_away(committed: dict) -> None:
    """A rounded-down negative is ``-0.0``, which is not ``0.0`` to a byte diff.

    Both paths into the document normalise it, and the whole committed file is
    checked for the literal, because one occurrence is one byte a fresh run on
    another machine might not produce.
    """
    assert math.copysign(1.0, artifact._rounded(-0.0, 3)) > 0
    assert math.copysign(1.0, artifact._exact(-0.0)) > 0
    assert re.search(r"-0\.0(?=[,\n}])", artifact.dumps(committed)) is None


def test_a_value_smaller_than_its_precision_keeps_its_digits() -> None:
    """Rounding 4.2e-7 to five decimals would store a zero against a real limit."""
    assert artifact._rounded(4.2e-7, 5) == pytest.approx(4.2e-7)


def test_a_non_finite_value_fails_at_write_time() -> None:
    """``Infinity`` is not JSON and would throw inside the site build."""
    with pytest.raises(ValueError, match="non-finite value"):
        artifact._rounded(math.inf, 3)


def test_the_document_serialises_the_same_way_twice(fresh: dict) -> None:
    """Committed means reproducible: no timestamp, no SHA, no library version."""
    assert artifact.dumps(fresh) == artifact.dumps(artifact.build_document())


def test_the_document_carries_no_provenance_that_changes_by_itself(
    committed: dict,
) -> None:
    text = artifact.dumps(committed)
    assert "timestamp" not in text
    assert "numpy" not in text


# --------------------------------------------------------------------------
# The reference split
# --------------------------------------------------------------------------


def test_every_citation_rebuilds_from_its_split(committed: dict) -> None:
    """The whole split rests on this: three fields that cannot be reassembled
    into the original string have lost or moved something, and the designation
    count is then counting the wrong thing.
    """
    overridden = gate._overridden()
    for check in committed["checks"]:
        reference = check["reference"]
        if reference["cite"] in overridden:
            continue
        rebuilt = references.recompose(
            references.Reference(
                kind=references.ReferenceKind(reference["kind"]),
                designation=reference["designation"],
                edition=reference.get("edition"),
                clause=reference.get("clause"),
                cite=reference["cite"],
            )
        )
        assert rebuilt == reference["cite"], check["id"]


def test_the_override_ratchet_carries_no_dead_lines(committed: dict) -> None:
    """A line for a citation nobody makes hides the next real one."""
    assert gate._ratchet_problems(committed) == []


def test_a_standard_splits_into_designation_edition_and_clause() -> None:
    reference = references.parse("IEC 61260-1:2014 Table 1", overrides={})
    assert reference.kind is references.ReferenceKind.STANDARD
    assert (reference.designation, reference.edition, reference.clause) == (
        "IEC 61260-1",
        "2014",
        "Table 1",
    )


def test_a_book_edition_is_a_string_not_a_year() -> None:
    """An edition has to hold "2e" and "4th ed" as well as "2014"."""
    reference = references.parse(
        "Long, Architectural Acoustics 2e, Table 8.1", overrides={}
    )
    assert reference.kind is references.ReferenceKind.BOOK
    assert reference.edition == "2e"
    assert reference.designation == "Long, Architectural Acoustics"


def test_a_closed_form_is_a_derivation_and_not_a_document() -> None:
    reference = references.parse("Model identity (uniform absorption)", overrides={})
    assert reference.kind is references.ReferenceKind.DERIVATION
    assert reference.clause is None


def test_an_override_line_needs_five_fields() -> None:
    with pytest.raises(ValueError, match="expected 5 tab-separated fields"):
        references._override_line("only\ttwo", 3)


def test_an_override_line_needs_a_known_kind() -> None:
    with pytest.raises(ValueError, match="is not one of"):
        references._override_line("cite\tpamphlet\tX\t-\t-", 4)


# --------------------------------------------------------------------------
# Units
# --------------------------------------------------------------------------


def test_a_unit_spelling_collapses_onto_one_form() -> None:
    assert units.canonical_unit("dB(A)") == "dBA"
    assert units.canonical_unit("Pa s/m2") == "Pa·s/m²"
    assert units.canonical_unit("m/s^2") == "m/s²"


def test_an_unknown_unit_is_rejected_rather_than_passed_through() -> None:
    """A silent new spelling is how the report reached 58 spellings of 54 units."""
    with pytest.raises(ValueError, match="not in the conformance unit vocabulary"):
        units.canonical_unit("furlongs")


def test_the_vocabulary_at_the_document_head_covers_every_row(
    committed: dict,
) -> None:
    declared = set(committed["units"])
    used = {check["unit"] for check in committed["checks"] if check.get("unit")}
    assert used <= declared


# --------------------------------------------------------------------------
# The builders
# --------------------------------------------------------------------------


def test_a_record_check_compares_the_same_names_on_both_sides() -> None:
    with pytest.raises(ValueError, match="record check compares different names"):
        registry.record({"Rw": 52.0}, {"Rw": 52.0, "C": -1.0})


def test_a_record_check_counts_the_names_that_disagree() -> None:
    outcome = registry.record({"Rw": 52.0, "C": -1.0}, {"Rw": 52.0, "C": -2.0})
    assert outcome.kind is registry.Kind.RECORD
    assert outcome.deviation.value == 1.0
    assert outcome.verdict is registry.Verdict.FAIL


def test_a_count_check_carries_no_unit() -> None:
    """ "mismatches" sat in the unit column among the newtons and the pascals."""
    outcome = registry.count(160, 160, subject="coefficients")
    assert outcome.unit is None
    assert outcome.kind is registry.Kind.COUNT
    assert outcome.computed == "160/160 coefficients"


def test_a_mask_is_judged_by_the_nearer_edge_of_its_band() -> None:
    """The Tech 3341 true-peak window is +0.2/-0.4 dB: no single figure says it."""
    outcome = registry.mask(
        expected="-6 dBTP (+0.2/-0.4 dB)",
        computed="-5.9 dBTP",
        deviation=0.1,
        lower=-0.4,
        upper=0.2,
        unit="dBTP",
    )
    assert outcome.kind is registry.Kind.MASK
    assert outcome.delta == "headroom 0.1 dBTP"
    assert outcome.verdict is registry.Verdict.PASS


def test_a_one_sided_criterion_has_an_unbounded_edge() -> None:
    outcome = registry.mask(
        expected="m >= 0.5", computed="0.981", deviation=0.981, lower=0.5
    )
    assert outcome.delta == "headroom 0.481"


def test_an_outcome_built_from_strings_alone_keeps_working() -> None:
    """Four checks live in a module this pipeline does not own, and a new row
    added there must land in the artefact without an edit here.
    """
    outcome = registry.Outcome(
        expected="class 0",
        computed="class 0 (margin +0.650 dB)",
        delta="+0.650 dB",
        passed=True,
    )
    assert outcome.kind is registry.Kind.MASK
    assert outcome.deviation.value == pytest.approx(0.65)
    assert outcome.deviation.label == "+0.650 dB"


def test_a_delta_carrying_two_quantities_is_not_guessed_at() -> None:
    """ "+0.12 / -0.34 Hz" is two numbers; storing either one would be a lie."""
    deviation = registry._inferred("+0.12 / -0.34 Hz")
    assert deviation.value is None
    assert deviation.label == "+0.12 / -0.34 Hz"


# --------------------------------------------------------------------------
# Tolerance utilisation
# --------------------------------------------------------------------------


def _scalar(deviation: float, limit: float, mode: str = "absolute") -> dict:
    return {
        "deviation": {"value": deviation},
        "tolerance": {"mode": mode, "value": limit},
        "expected": {"value": 10.0},
    }


def test_utilisation_is_the_fraction_of_the_limit_a_deviation_spends() -> None:
    assert metrics.utilisation(_scalar(0.05, 0.1)) == pytest.approx(0.5)
    assert metrics.utilisation(_scalar(-0.1, 0.1)) == pytest.approx(1.0)


def test_a_relative_tolerance_is_a_fraction_of_the_expected_value() -> None:
    assert metrics.utilisation(_scalar(0.5, 0.1, "relative")) == pytest.approx(0.5)


def test_a_check_with_no_declared_limit_has_no_utilisation() -> None:
    assert metrics.utilisation({"deviation": {"value": 1.0}}) is None


def test_a_one_sided_mask_has_no_utilisation() -> None:
    """Half of an unbounded band is not a fraction of anything."""
    check = {
        "deviation": {"value": 0.9},
        "tolerance": {"mode": "mask", "value": 0.0},
        "binding": {"lower": 0.5},
        "expected": {},
    }
    assert metrics.utilisation(check) is None


def test_no_committed_check_spends_more_than_its_limit(committed: dict) -> None:
    """A utilisation over one with a passing verdict would mean the stored
    limit and the stored verdict disagree about the same check.
    """
    for check in committed["checks"]:
        used = metrics.utilisation(check)
        if used is None or check["verdict"] != "pass":
            continue
        assert used <= 1.0 + 1e-9, check["id"]


# --------------------------------------------------------------------------
# The comparator
# --------------------------------------------------------------------------


def test_an_unchanged_document_reports_nothing(committed: dict) -> None:
    assert compare.document_problems(committed, json.loads(json.dumps(committed))) == []


def test_a_flipped_verdict_is_never_within_tolerance(committed: dict) -> None:
    moved = json.loads(json.dumps(committed))
    moved["checks"][0]["verdict"] = "fail"
    problems = compare.document_problems(committed, moved)
    assert any("verdict" in problem for problem in problems)


def test_a_last_digit_wobble_is_within_tolerance(committed: dict) -> None:
    """The drift the gate exists to tolerate: one quantum of the check's own
    precision, which is what a rounding boundary moves by across BLAS builds.
    """
    moved = json.loads(json.dumps(committed))
    check = next(c for c in moved["checks"] if c["deviation"].get("value") is not None)
    check["deviation"]["value"] += 10.0 ** -max(int(check["precision"]), 3)
    assert compare.document_problems(committed, moved) == []


def test_a_real_move_is_not_within_tolerance(committed: dict) -> None:
    moved = json.loads(json.dumps(committed))
    check = next(c for c in moved["checks"] if c["deviation"].get("value") is not None)
    check["deviation"]["value"] += 1000.0
    assert compare.document_problems(committed, moved) != []


def test_an_added_check_fails_the_comparison(committed: dict) -> None:
    grown = json.loads(json.dumps(committed))
    extra = json.loads(json.dumps(grown["checks"][0]))
    extra["id"] += "-copy"
    grown["checks"].append(extra)
    grown["counts"]["checks"] += 1
    assert compare.document_problems(committed, grown) != []


def test_two_checks_may_not_share_an_id() -> None:
    """A shared id would make the pull-request diff join two different checks."""
    with pytest.raises(ValueError, match="duplicate conformance check id"):
        artifact._reject_duplicate_ids([{"id": "a/b/c"}, {"id": "a/b/c"}])


# --------------------------------------------------------------------------
# The two renderers read one document
# --------------------------------------------------------------------------


def test_the_markdown_shows_every_check_exactly_once(committed: dict) -> None:
    """The drift guard: Markdown and the site component render one artefact,
    and the Markdown must account for all of it.
    """
    markdown, passed, total = cr.render_markdown(committed)
    assert (passed, total) == (
        committed["counts"]["passing"],
        committed["counts"]["checks"],
    )
    for check in committed["checks"]:
        quantity = cr._cell(check["quantity"])
        assert markdown.count(f"| {quantity} |") >= 1, check["id"]


def test_the_markdown_headline_states_the_recorded_counts(committed: dict) -> None:
    markdown, _, _ = cr.render_markdown(committed)
    counts = committed["counts"]
    assert (
        f"**{counts['passing']}/{counts['checks']} conformance checks pass** "
        f"across {counts['domains']} domains and {counts['standards']} standards"
    ) in markdown


def test_the_markdown_is_a_pure_function_of_the_artefact(committed: dict) -> None:
    """Rendering twice from one document must give one file, which is what
    lets docs/CONFORMANCE.md keep a byte gate while the values behind it are
    compared within a tolerance.
    """
    assert cr.render_markdown(committed)[0] == cr.render_markdown(committed)[0]


def test_the_committed_markdown_is_what_the_artefact_renders(committed: dict) -> None:
    markdown, _, _ = cr.render_markdown(committed)
    expected = cr._DOC_HEADER + markdown + "\n"
    on_disk = (artifact._ROOT / "docs" / "CONFORMANCE.md").read_text(encoding="utf8")
    assert on_disk == expected


#: A whole document with three checks, one of each interesting shape. The
#: renderer is a pure function of the artefact, so it can be exercised on this
#: instead of only end to end on 554 rows.
FIXTURE = {
    "schema": 1,
    "library": "0.0.0",
    "generator": "test",
    "counts": {
        "checks": 3,
        "passing": 3,
        "failing": 0,
        "domains": 1,
        "standards": 2,
        "citations": 3,
        "designations": 2,
        "sources": 0,
    },
    "units": ["dB"],
    "domains": [{"id": "d", "title": "D", "checks": 3, "passing": 3}],
    "panels": [
        {
            "id": "filter-class",
            "title": "T",
            "rows": [
                {
                    "architecture": "butter",
                    "verdict": "pass",
                    "class": 1,
                    "binding": {
                        "frequency_hz": 1000.0,
                        "measured": 0.0,
                        "limit": -0.3,
                        "side": "ceil",
                    },
                    "margin_class1": 0.4,
                    "margin_class2": 0.9,
                }
            ],
        },
        {
            "id": "weighting-deviation",
            "title": "W",
            "rows": [
                {
                    "curve": "A",
                    "fs_hz": 48000,
                    "worst": {"frequency_hz": 20.0, "deviation": -0.7},
                    "binding": {
                        "frequency_hz": 20.0,
                        "deviation": -0.7,
                        "lower": -2.0,
                        "upper": 2.0,
                    },
                    "headroom": 1.3,
                    "verdict": "pass",
                }
            ],
        },
    ],
    "checks": [
        {
            "id": "d/iso-1-2020-table-1/scalar",
            "domain": "d",
            "reference": {
                "kind": "standard",
                "designation": "ISO 1",
                "edition": "2020",
                "clause": "Table 1",
                "cite": "ISO 1:2020 Table 1",
            },
            "quantity": "Scalar",
            "kind": "scalar",
            "expected": {"value": 1.0},
            "computed": {"value": 1.02},
            "unit": "dB",
            "tolerance": {"mode": "absolute", "value": 0.05},
            "deviation": {"value": 0.02},
            "precision": 2,
            "verdict": "pass",
        },
        {
            "id": "d/iso-1-2020-table-2/mask",
            "domain": "d",
            "reference": {
                "kind": "standard",
                "designation": "ISO 1",
                "edition": "2020",
                "clause": "Table 2",
                "cite": "ISO 1:2020 Table 2",
            },
            "quantity": "Mask",
            "kind": "mask",
            "expected": {"label": "within band"},
            "computed": {"label": "+0.10 dB"},
            "unit": "dB",
            "tolerance": {"mode": "mask", "value": 0.0},
            "deviation": {"value": 0.1},
            "binding": {"frequency_hz": 1000.0, "lower": -1.0, "upper": 1.0},
            "precision": 3,
            "verdict": "pass",
        },
        {
            "id": "d/iso-2-2020-annex-a/record",
            "domain": "d",
            "reference": {
                "kind": "standard",
                "designation": "ISO 2",
                "edition": "2020",
                "clause": "Annex A",
                "cite": "ISO 2:2020 Annex A",
            },
            "quantity": "Record",
            "kind": "record",
            "expected": {"label": "Rw = 52", "record": {"Rw": 52.0}},
            "computed": {"label": "Rw = 52", "record": {"Rw": 52.0}},
            "tolerance": {"mode": "absolute", "value": 0.0},
            "deviation": {"value": 0.0},
            "precision": 0,
            "verdict": "pass",
        },
    ],
}


def test_the_renderer_works_on_a_three_check_document() -> None:
    """The point of making the Markdown a pure function of the artefact."""
    markdown, passed, total = cr.render_markdown(FIXTURE)
    assert (passed, total) == (3, 3)
    assert (
        "**3/3 conformance checks pass** across 1 domains and 2 standards" in markdown
    )
    assert "| 1.02 dB |" in markdown  # the scalar's computed value
    assert "| headroom 0.9 dB |" in markdown  # the mask's headroom, derived
    assert "| exact |" in markdown  # the record's deviation
    assert "ISO 1:2020 Table 1" in markdown


def test_a_verdict_is_printed_and_never_re_derived() -> None:
    """The stored verdict wins over anything the stored numbers imply.

    A deviation that rounds onto its limit would flip a re-derived judgement,
    so the renderer must not have one. Handed a row whose deviation is ten
    times its tolerance and whose verdict says pass, it prints pass.
    """
    document = json.loads(json.dumps(FIXTURE))
    document["checks"][0]["deviation"]["value"] = 0.5
    markdown, _, _ = cr.render_markdown(document)
    row = next(line for line in markdown.splitlines() if "| Scalar |" in line)
    assert row.endswith("| &#9989; |")


def test_a_missing_panel_is_named() -> None:
    with pytest.raises(KeyError, match="carries no 'filter-class' panel"):
        cr._numerical_validation_section({"panels": []}, True)


def test_a_missing_artefact_says_how_to_make_one(tmp_path: pathlib.Path) -> None:
    with pytest.raises(SystemExit, match="make conformance"):
        artifact.load(tmp_path / "conformance.json")
