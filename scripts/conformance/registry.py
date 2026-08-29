#  Copyright (c) 2026. Jose Manuel Requena Plens
"""The check registry, and the Outcome vocabulary every check speaks.

A conformance check is an argument-free callable returning an :class:`Outcome`
- what the standard says, what the library computed, the difference between
them and the verdict - registered under a domain, a standard designation and a
quantity by the :func:`register` decorator. That decorator is the only way
into :data:`CHECKS`, and :data:`CHECKS` keeps registration order, so the order
the domain modules are imported in is the order of the report.

:func:`numeric` is the builder nearly every check uses. It applies an absolute
or relative tolerance and records what the comparison was: the two values, the
published limit it was judged against, the signed deviation between them and
the decimals the check reports at. :func:`mask`, :func:`record` and
:func:`count` are the builders for the three comparisons that are not one
scalar against another.

An :class:`Outcome` is data first and three strings second. It used to be the
three strings alone, which meant the limit a check was judged against - the
number a metrology library is most often asked for - existed nowhere after the
comparison, and the deviation survived only as pre-formatted text with five
different meanings ("exact", "-", "headroom +0.003 dB", "sum 32.0 dB"). The
strings are still carried, because a check may legitimately describe its two
sides in prose, but they are now the fallback rather than the record.

:class:`Verdict` is an enum, not a bool, for a reason that had already bitten:
``numpy.float64`` is a subclass of ``float``, so a numpy scalar reached
:func:`numeric` with mypy's blessing and ``abs(delta) <= limit`` handed back a
``numpy.bool_``, which is not JSON-serialisable. Nine checks did this and
nothing noticed, because the only consumer was ``if outcome.passed``. Every
value crossing this boundary is now coerced to a built-in type.

This module also puts ``tests/`` on ``sys.path``. The expected values are not
re-typed for the report: the checks read the same ``reference_data`` tables the
test suite reads, and the ``iso12354_building`` and ``diffuser_prediction``
helpers it shares, so the report and the tests can never validate the library
against different numbers.
"""

from __future__ import annotations

import enum
import functools
import math
import pathlib
import re
import sys
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .units import canonical_unit

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

# The checks do not re-type their expected values: they read the test suite's
# shared tables (``reference_data``) and its worked-example helpers
# (``iso12354_building``, ``diffuser_prediction``) straight from ``tests/``,
# which is what this puts on the path. Two copies of a normative table could
# drift; one cannot.
_ROOT = pathlib.Path(__file__).resolve().parents[2]
_TESTS = _ROOT / "tests"
_DATA = _TESTS / "data"
for _p in (str(_TESTS),):
    if _p not in sys.path:
        sys.path.insert(0, _p)


class Verdict(enum.StrEnum):
    """What a check decided, as a name rather than a bit.

    ``BY_DESIGN`` is not reachable from a registered check and is used by the
    showcase panels: the Chebyshev-I, elliptic and Bessel architectures cannot
    meet the IEC 61260-1 mask by construction, which the report has always said
    in a display string where nothing could count it.
    """

    PASS = "pass"
    FAIL = "fail"
    BY_DESIGN = "by-design"
    NOT_APPLICABLE = "not-applicable"


class Kind(enum.StrEnum):
    """The shape of the comparison a check makes.

    Four shapes cover every registered check. ``SCALAR`` is one number against
    another within a tolerance; ``MASK`` is a response judged against an
    acceptance band, where what matters is the headroom to the nearest edge;
    ``RECORD`` is a set of named values compared exactly (the ISO 717 single
    numbers, ``Rw (C; Ctr)``); ``COUNT`` is a tally of agreeing fields, which
    is what the byte-for-byte metadata checks report.
    """

    SCALAR = "scalar"
    MASK = "mask"
    RECORD = "record"
    COUNT = "count"


class ToleranceMode(enum.StrEnum):
    """How a tolerance value is applied to the deviation.

    Replaces the boolean ``rel=``, which could express two of these and never
    the third: a mask limit is a band that varies with frequency, so the number
    the deviation is judged against is the band edge carried in
    :class:`Binding`, not a single figure.
    """

    ABSOLUTE = "absolute"
    RELATIVE = "relative"
    MASK = "mask"


@dataclass(frozen=True)
class Tolerance:
    """The published limit a deviation is judged against.

    First-class because a limit that exists only inside the comparison cannot
    be reported: "how much of its published tolerance does this check consume"
    is unanswerable from a pass mark alone.
    """

    mode: ToleranceMode
    value: float | None = None


@dataclass(frozen=True)
class Binding:
    """Where a mask check binds, and the band it must sit inside there.

    A mask is judged at its least comfortable point, so the frequency and the
    two edges of the acceptance band at that point are the evidence; the
    headroom is their difference and is therefore not stored.
    """

    frequency_hz: float | None = None
    lower: float | None = None
    upper: float | None = None


@dataclass(frozen=True)
class Side:
    """One side of a comparison: the normative value, or what was computed.

    ``value`` and ``record`` are mutually exclusive - a side is one number or a
    set of named ones, never both. ``label`` is prose the report shows in place
    of the formatted number, for a side whose meaning does not survive being
    printed as a figure ("class 1", "bext v2 fields byte-identical").
    """

    value: float | None = None
    label: str | None = None
    record: Mapping[str, float] | None = None


@dataclass(frozen=True)
class Deviation:
    """How far the computed side fell from the expected one.

    ``value`` is signed and in the check's unit: for a scalar it is
    ``computed - expected``, for a mask it is the headroom to the binding edge,
    for a count it is the number of fields that disagreed. ``label`` carries
    the deviation of a check that has not been converted to data yet, so the
    report can still print what it always printed.
    """

    value: float | None = None
    label: str | None = None


@dataclass(frozen=True)
class Outcome:
    """The result of a single conformance check, as data and as three columns.

    The three strings are what the Markdown report prints. Everything below
    them is the same result as data, which is what the artefact carries and
    what lets a consumer rank checks by the fraction of their published
    tolerance they consume.

    A check that supplies only the strings still works - four of them live in a
    module this pipeline does not own - and its shape is inferred by
    :func:`_inferred`, which is why ``kind`` and ``deviation`` have defaults.
    """

    expected: str
    computed: str
    delta: str
    passed: bool
    kind: Kind = Kind.SCALAR
    unit: str | None = None
    tolerance: Tolerance | None = None
    deviation: Deviation | None = None
    precision: int = 4
    expected_data: Side = field(default_factory=Side)
    computed_data: Side = field(default_factory=Side)
    binding: Binding | None = None

    def __post_init__(self) -> None:
        """Coerce the verdict to a built-in ``bool`` and fill in the shape.

        ``numpy.bool_`` is not JSON-serialisable and ``numpy.float64`` passes
        every ``isinstance(x, float)`` test there is, so a numpy scalar reached
        this constructor unchallenged on nine checks. Coercing here is the only
        place that catches the ones built by hand.
        """
        object.__setattr__(self, "passed", bool(self.passed))
        if self.tolerance is None:
            object.__setattr__(self, "kind", _inferred_kind(self.expected, self.delta))
        if self.deviation is None:
            object.__setattr__(self, "deviation", _inferred(self.delta))
        if _is_empty(self.expected_data):
            object.__setattr__(self, "expected_data", Side(label=self.expected))
        if _is_empty(self.computed_data):
            object.__setattr__(self, "computed_data", Side(label=self.computed))

    @property
    def verdict(self) -> Verdict:
        """The verdict as a name. Decided at full precision, before rounding."""
        return Verdict.PASS if self.passed else Verdict.FAIL


@dataclass(frozen=True)
class Check:
    """A registered (standard, quantity) conformance check."""

    domain: str
    standard: str
    quantity: str
    run: Callable[[], Outcome]


CHECKS: list[Check] = []


def register(
    domain: str, standard: str, quantity: str
) -> Callable[[Callable[[], Outcome]], Callable[[], Outcome]]:
    """Register a check callable under a domain / standard / quantity."""

    def deco(fn: Callable[[], Outcome]) -> Callable[[], Outcome]:
        # Checks are deterministic and argument-free, so the outcome is
        # cached per process: rendering the full report and re-running an
        # individual check in the same process computes the check once.
        CHECKS.append(Check(domain, standard, quantity, functools.cache(fn)))
        return fn

    return deco


# A signed decimal with an optional unit and nothing else: the shape of a
# delta string that still carries one number, so an outcome built before the
# builders existed can be read back as data rather than as text.
_LONE_NUMBER = re.compile(r"^(?P<value>[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)(?: \S+)?$")

# The same, behind the word the mask checks prefix it with.
_HEADROOM = re.compile(r"^headroom (?P<value>[-+]?\d+(?:\.\d+)?)(?: \S+)?$")


def _inferred(delta: str) -> Deviation:
    """Read a deviation back out of a pre-formatted delta string.

    The compatibility path, for checks that hand this module three strings and
    no data. Two shapes carry exactly one number - ``"+0.036 ft"`` and
    ``"headroom +0.003 dB"`` - and are recovered; anything else (``"exact"``,
    ``"-"``, ``"sum 32.0 dB"``, ``"+0.12 / -0.34 Hz"``) is two quantities or
    none, so the string is kept as a label and the value stays ``None`` rather
    than being guessed at.

    :param delta: The delta column as the check formatted it.
    :return: The deviation, with ``value`` set only when the string is
        unambiguously one number.
    """
    for pattern in (_LONE_NUMBER, _HEADROOM):
        match = pattern.match(delta)
        if match is not None:
            return Deviation(value=float(match["value"]), label=delta)
    return Deviation(label=delta)


def _is_empty(side: Side) -> bool:
    """Whether a side carries nothing, so the display string is all there is."""
    return side.value is None and side.label is None and side.record is None


def _inferred_kind(expected: str, delta: str) -> Kind:
    """Read the shape of a comparison back out of its two display strings.

    The same compatibility path as :func:`_inferred`, for a check that supplies
    no tolerance. Three rules, each keyed on the report's own vocabulary rather
    than on a guess: a delta that names a headroom is a mask; an expected side
    that reads ``class N`` is also a mask, because a filter class *is* the
    verdict of the IEC 61260-1 attenuation mask and its margin is the headroom
    to it; a delta that is one number is a scalar. Everything else compares
    named things exactly, which is what a record is.

    :param expected: The expected column as the check formatted it.
    :param delta: The delta column as the check formatted it.
    :return: The inferred shape.
    """
    if delta.startswith("headroom ") or expected.startswith("class "):
        return Kind.MASK
    if _LONE_NUMBER.match(delta):
        return Kind.SCALAR
    return Kind.RECORD


#: Decimals a deviation is never reported coarser than, whatever precision the
#: check displays its *values* at. A precision is chosen to suit the value - a
#: distance of 5 280 ft reads at zero decimals - and the deviation from it is a
#: much smaller number, so applying the same precision to both would report a
#: real deviation of 0.036 ft as zero. This is the floor the old
#: ``_DELTA_PLACES`` was, with its cap removed: a check that declares more
#: decimals now gets them, and no check gets fewer than three.
_MIN_DEVIATION_PLACES = 3


def deviation_places(precision: int) -> int:
    """Decimals a deviation is reported at, given the check's precision."""
    return max(precision, _MIN_DEVIATION_PLACES)


def _snap(value: float, eps: float = 5e-4) -> float:
    """Snap a near-zero value to +0 so displays avoid a spurious ``-0.00``."""
    return 0.0 if abs(value) < eps else value


def _fmt(value: float, unit: str = "", places: int = 4) -> str:
    """Compact fixed/again-significant formatting with an optional unit."""
    if not math.isfinite(value):
        return "inf" if value > 0 else "-inf"
    text = f"{value:.{places}f}".rstrip("0").rstrip(".")
    if text in ("", "-0"):
        text = "0"
    return f"{text} {unit}".strip()


def numeric(
    expected: float,
    computed: float,
    tol: float,
    *,
    unit: str = "",
    rel: bool = False,
    places: int = 4,
    expected_label: str | None = None,
    computed_label: str | None = None,
) -> Outcome:
    """Build an Outcome for ``|computed - expected| <= tol`` (abs or rel).

    Every argument is coerced to a built-in ``float`` on the way in. A caller
    that hands over a ``numpy.float64`` - which is a subclass of ``float``, so
    no annotation and no type checker objects - would otherwise make
    ``abs(delta) <= limit`` a ``numpy.bool_``, and nine checks did exactly that
    until the outcome had to be serialised.

    :param expected: The value the standard publishes.
    :param computed: The value the library computes.
    :param tol: The published limit, absolute or as a fraction of ``expected``.
    :param unit: Unit of both values, in any spelling
        :func:`~conformance.units.canonical_unit` knows.
    :param rel: Judge ``tol`` as a fraction of ``|expected|`` rather than as an
        absolute limit.
    :param places: Decimals this check reports at, for both the report and the
        artefact.
    :param expected_label: Prose to print instead of the formatted expected
        value; the value itself is still recorded.
    :param computed_label: Prose to print instead of the formatted computed
        value, for a check whose result reads as a sentence ("max absolute
        deviation 0.000 dB"); the value itself is still recorded.
    :return: The outcome, with its verdict decided at full precision.
    """
    expected, computed, tol = float(expected), float(computed), float(tol)
    delta = computed - expected
    limit = tol * abs(expected) if rel else tol
    passed = abs(delta) <= limit
    tol_txt = f"{tol * 100:g}%" if rel else _fmt(tol, unit, places)
    exp_txt = expected_label or _fmt(expected, unit, places)
    mode = ToleranceMode.RELATIVE if rel else ToleranceMode.ABSOLUTE
    return Outcome(
        expected=f"{exp_txt} (+/-{tol_txt})" if not expected_label else exp_txt,
        computed=computed_label or _fmt(computed, unit, places),
        # The report prints the deviation at the check's own precision, with a
        # three-decimal floor. It used to be capped at three decimals so the
        # committed Markdown survived a byte diff across numpy/scipy/BLAS
        # builds; the artefact is compared within a tolerance instead, so the
        # evidence no longer has to be rounded away to keep a text diff quiet.
        delta=_fmt(delta, unit, deviation_places(places)),
        passed=passed,
        kind=Kind.SCALAR,
        unit=canonical_unit(unit),
        tolerance=Tolerance(mode, tol),
        deviation=Deviation(value=delta),
        precision=places,
        expected_data=Side(value=expected, label=expected_label),
        computed_data=Side(value=computed, label=computed_label),
    )


def _headroom(deviation: float, lower: float | None, upper: float | None) -> float:
    """Distance from ``deviation`` to the nearest edge of its band.

    A missing edge is unbounded on that side. With neither edge the deviation
    is already the headroom, which is how a check that knows its margin but not
    the band that produced it reports.
    """
    if lower is None and upper is None:
        return deviation
    to_upper = math.inf if upper is None else upper - deviation
    to_lower = math.inf if lower is None else deviation - lower
    return min(to_upper, to_lower)


def mask(
    *,
    expected: str,
    computed: str,
    deviation: float,
    lower: float | None = None,
    upper: float | None = None,
    frequency_hz: float | None = None,
    unit: str = "",
    places: int = 3,
) -> Outcome:
    """Build an Outcome for a value judged against an acceptance band.

    A mask has no single limit: the band varies with frequency, and it may be
    asymmetric about the nominal value, so what decides the verdict is the
    headroom to the nearest edge at the point where that headroom is least. The
    band edges there are recorded, and the signed deviation with them, so the
    headroom stays derivable rather than being the only thing kept.

    :param expected: What the band requires, in prose.
    :param computed: What was measured, in prose.
    :param deviation: Signed deviation from nominal, or the margin itself when
        no band is given.
    :param lower: Lower edge of the acceptance band, ``None`` if unbounded.
    :param upper: Upper edge, ``None`` if unbounded.
    :param frequency_hz: Frequency the band binds at, when there is one.
    :param unit: Unit of the deviation and of the band edges.
    :param places: Decimals to report the headroom at.
    :return: The outcome, verdict decided at full precision.
    """
    deviation = float(deviation)
    headroom = _headroom(deviation, lower, upper)
    return Outcome(
        expected=expected,
        computed=computed,
        delta=f"headroom {_fmt(headroom, unit, deviation_places(places))}",
        passed=headroom >= 0.0,
        kind=Kind.MASK,
        unit=canonical_unit(unit),
        tolerance=Tolerance(ToleranceMode.MASK, 0.0),
        deviation=Deviation(value=deviation),
        precision=places,
        expected_data=Side(label=expected),
        computed_data=Side(label=computed),
        binding=Binding(frequency_hz=frequency_hz, lower=lower, upper=upper),
    )


def record(
    expected: Mapping[str, float],
    computed: Mapping[str, float],
    *,
    unit: str = "",
    label: str | None = None,
    computed_label: str | None = None,
) -> Outcome:
    """Build an Outcome for named values that must match exactly.

    The ISO 717 single numbers are the case this exists for: ``Rw (C; Ctr)`` is
    three integers read off one table, and a tolerance would be meaningless
    because the standard rounds them itself. Equality is therefore the whole
    comparison, and the deviation is the count of names that disagreed.

    :param expected: The names and values the standard publishes.
    :param computed: The names and values the library produced.
    :param unit: Unit shared by every value.
    :param label: Prose for the expected side; defaults to ``k = v`` pairs.
    :param computed_label: Prose for the computed side; same default.
    :return: The outcome, passing only on an exact match of every name.
    :raises ValueError: If the two sides do not carry the same names, which is
        a mistake in the check rather than a failure of the library.
    """
    if set(expected) != set(computed):
        msg = (
            f"record check compares different names: expected "
            f"{sorted(expected)}, computed {sorted(computed)}."
        )
        raise ValueError(msg)
    expected = {name: float(value) for name, value in expected.items()}
    computed = {name: float(value) for name, value in computed.items()}
    differing = sum(1 for name in expected if expected[name] != computed[name])
    expected_text = label or _pairs(expected, unit)
    computed_text = computed_label or _pairs(computed, unit)
    return Outcome(
        expected=expected_text,
        computed=computed_text,
        delta="exact" if not differing else f"{differing} differ",
        passed=not differing,
        kind=Kind.RECORD,
        unit=canonical_unit(unit),
        tolerance=Tolerance(ToleranceMode.ABSOLUTE, 0.0),
        deviation=Deviation(value=float(differing)),
        precision=0,
        expected_data=Side(label=expected_text, record=expected),
        computed_data=Side(label=computed_text, record=computed),
    )


def count(
    matching: int,
    total: int,
    *,
    subject: str,
    expected_label: str | None = None,
) -> Outcome:
    """Build an Outcome for a tally of fields that had to agree.

    The byte-for-byte metadata checks report ``N/N`` of something, and the
    thing being counted was written into the unit column, where ``"mismatches"``
    sat among the newtons and the pascals. A count has no unit; it has a
    subject.

    :param matching: How many agreed.
    :param total: How many were compared.
    :param subject: What was counted, e.g. ``"bext fields"``.
    :param expected_label: Prose for the expected side, when ``all N subject``
        is not how the check reads.
    :return: The outcome, passing only when every one agreed.
    """
    matching, total = int(matching), int(total)
    expected_text = expected_label or f"{total}/{total} {subject}"
    computed_text = f"{matching}/{total} {subject}"
    return Outcome(
        expected=expected_text,
        computed=computed_text,
        delta="exact" if matching == total else f"{total - matching} differ",
        passed=matching == total,
        kind=Kind.COUNT,
        unit=None,
        tolerance=Tolerance(ToleranceMode.ABSOLUTE, 0.0),
        deviation=Deviation(value=float(total - matching)),
        precision=0,
        expected_data=Side(value=float(total), label=expected_text),
        computed_data=Side(value=float(matching), label=computed_text),
    )


def _pairs(values: Mapping[str, float], unit: str) -> str:
    """Render named values as the report has always printed them."""
    return "; ".join(
        f"{name} = {_fmt(value, unit, 4)}" for name, value in values.items()
    )
