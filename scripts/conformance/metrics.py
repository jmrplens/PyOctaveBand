#  Copyright (c) 2026. Jose Manuel Requena Plens
"""What the artefact's numbers mean, once they are numbers.

Stdlib only, and it reads the committed document rather than the registry, so
the pull-request comment can answer these questions without installing the
scientific stack or re-running a 45-second harness to learn what a file it can
already read says.

The one question here is the one the Markdown report could never answer: **how
much of its published tolerance does a check consume?** Nothing computed it
before because nothing recorded the limit - the tolerance lived inside the
comparison and was gone the moment the verdict was decided.

``Conformance.astro`` computes the same figure in TypeScript for the site. Two
implementations of one formula is the price of two runtimes; the formula is
short, it is stated in both places, and the fixtures in
``tests/test_conformance_artifact.py`` pin this one.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping


def utilisation(check: Mapping[str, Any]) -> float | None:
    """The fraction of its published limit a check's deviation consumes.

    Never used to decide a verdict. The stored verdict was decided at full
    precision before any rounding, and a deviation one quantum under its limit
    rounds onto it, so re-deriving a judgement from this number would flip a
    verdict the harness got right.

    :param check: One check, as the artefact carries it.
    :return: The fraction, where ``1.0`` means the deviation sits exactly on the
        limit; ``None`` where the question has no answer, which is a check that
        declares no tolerance or a mask bounded on only one side.
    """
    stored = check["deviation"].get("value")
    tolerance = check.get("tolerance")
    if stored is None or tolerance is None:
        return None
    deviation = float(stored)
    if tolerance["mode"] == "mask":
        return _mask_utilisation(deviation, check.get("binding") or {})
    limit = _limit(tolerance, check)
    if limit > 0:
        return abs(deviation) / limit
    # A limit of zero is an exact-match check: it is either spent or it is not.
    return 0.0 if deviation == 0 else 1.0


def _limit(tolerance: Mapping[str, Any], check: Mapping[str, Any]) -> float:
    """The absolute limit, whichever way the tolerance was declared."""
    value = float(tolerance.get("value") or 0.0)
    if tolerance["mode"] == "relative":
        return value * abs(float(check["expected"].get("value") or 0.0))
    return value


def _mask_utilisation(deviation: float, binding: Mapping[str, Any]) -> float | None:
    """How far across its acceptance band a mask deviation sits.

    Measured from the centre of the band outwards, so ``1.0`` is an edge and
    ``0`` is dead centre. A band open on one side has no centre and no width,
    so it has no answer.
    """
    stored_lower, stored_upper = binding.get("lower"), binding.get("upper")
    if stored_lower is None or stored_upper is None:
        return None
    lower, upper = float(stored_lower), float(stored_upper)
    half = (upper - lower) / 2.0
    if half <= 0:
        return None
    return abs(deviation - (upper + lower) / 2.0) / half
