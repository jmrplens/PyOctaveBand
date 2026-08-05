#  Copyright (c) 2026. Jose Manuel Requena Plens
"""The check registry, and the Outcome vocabulary every check speaks.

A conformance check is an argument-free callable returning an :class:`Outcome`
- what the standard says, what the library computed, the difference between
them and the verdict - registered under a domain, a standard designation and a
quantity by the :func:`register` decorator. That decorator is the only way
into :data:`CHECKS`, and :data:`CHECKS` keeps registration order, so the order
the domain modules are imported in is the order of the report.

:func:`numeric` is the builder nearly every check uses. It applies an absolute
or relative tolerance and formats the three columns, including the
deliberately coarse delta that keeps the committed ``docs/CONFORMANCE.md``
byte-stable across numpy/scipy/BLAS builds.

This module also puts ``tests/`` on ``sys.path``. The expected values are not
re-typed for the report: the checks read the same ``reference_data`` tables the
test suite reads, and the ``iso12354_building`` and ``diffuser_prediction``
helpers it shares, so the report and the tests can never validate the library
against different numbers.
"""

from __future__ import annotations

import functools
import math
import pathlib
import sys
from collections.abc import Callable
from dataclasses import dataclass

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

@dataclass(frozen=True)
class Outcome:
    """The rendered result of a single conformance check."""

    expected: str
    computed: str
    delta: str
    passed: bool


@dataclass(frozen=True)
class Check:
    """A registered (standard, quantity) conformance check."""

    domain: str
    standard: str
    quantity: str
    run: Callable[[], Outcome]


CHECKS: list[Check] = []


def register(domain: str, standard: str, quantity: str) -> Callable[
    [Callable[[], Outcome]], Callable[[], Outcome]
]:
    """Register a check callable under a domain / standard / quantity."""

    def deco(fn: Callable[[], Outcome]) -> Callable[[], Outcome]:
        # Checks are deterministic and argument-free, so the outcome is
        # cached per process: rendering the full report and re-running an
        # individual check in the same process computes the check once.
        CHECKS.append(Check(domain, standard, quantity, functools.cache(fn)))
        return fn

    return deco


# Decimal places for the informational delta column. Coarse on purpose so the
# committed docs/CONFORMANCE.md stays byte-stable across numpy/scipy/BLAS
# builds (see the note in ``numeric``).
_DELTA_PLACES = 3


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
) -> Outcome:
    """Build an Outcome for ``|computed - expected| <= tol`` (abs or rel)."""
    delta = computed - expected
    limit = tol * abs(expected) if rel else tol
    passed = abs(delta) <= limit
    tol_txt = f"{tol * 100:g}%" if rel else _fmt(tol, unit, places)
    exp_txt = expected_label or _fmt(expected, unit, places)
    return Outcome(
        expected=f"{exp_txt} (+/-{tol_txt})" if not expected_label else exp_txt,
        computed=_fmt(computed, unit, places),
        # The delta column is informational (pass/fail comes from ``passed``),
        # so it is coarsened to 3 decimals. This collapses sub-milli residuals
        # of the heavy DSP chains (ECMA/Moore-Glasberg/Zwicker, FFT intensity),
        # whose 4th-5th digit is BLAS/FFT-build dependent, to a stable value -
        # keeping the committed report diff-stable without hiding a regression
        # (that still shows in the Computed column and flips the status).
        delta=_fmt(delta, unit, _DELTA_PLACES),
        passed=passed,
    )
