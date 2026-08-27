#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Shared assertion for the ``Signal`` overload contract tests.

Every surface that adopts ``phonometry.io._resolve`` is held to the same
statement: the call that hands a ``Signal`` must produce the *identical*
result the equivalent bare-array call produces, never a nearby number. The
results are dataclasses of arrays, so the comparison has to walk them field
by field; doing that once here is what lets each package's test file be
only its own list of functions and its own exemptions.
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from typing import Any

import numpy as np


def assert_same(a: Any, b: Any, path: str = "result") -> None:  # noqa: ANN401 - deep-compares results of any shape: dataclasses, dicts, sequences, arrays, scalars
    """Assert two results are identical, walking result objects field by field.

    Every failure names the field path it reached and, at a leaf, what the two
    sides hold. The path used to start empty, so a mismatch at the top level
    raised an ``AssertionError`` carrying no message at all, and the callers
    that assert a difference IS expected could only write a bare
    ``pytest.raises(AssertionError)``. That accepts any failure of the helper,
    including a shape mismatch that has nothing to do with the contract the
    test is named for, which is the one thing a refusal test must not do.

    :param a: The result of the call under test.
    :param b: The result of the reference call.
    :param path: Field path reached so far, for the failure message.
    :raises AssertionError: On the first field that differs, naming the path.
    """
    if is_dataclass(a) and not isinstance(a, type):
        assert type(a) is type(b), path
        for f in fields(a):
            assert_same(getattr(a, f.name), getattr(b, f.name), f"{path}.{f.name}")
        return
    if isinstance(a, dict):
        assert a.keys() == b.keys(), path
        for k in a:
            assert_same(a[k], b[k], f"{path}[{k!r}]")
        return
    if isinstance(a, (list, tuple)):
        assert len(a) == len(b), path
        for i, (ai, bi) in enumerate(zip(a, b, strict=True)):
            assert_same(ai, bi, f"{path}[{i}]")
        return
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        # equal_nan: a NaN in the same place is the same result. Several
        # of these carry one by design (an unqualified decay time, an empty
        # band), and without this the helper would reject a match.
        arr_a, arr_b = np.asarray(a), np.asarray(b)
        assert np.array_equal(arr_a, arr_b, equal_nan=True), (
            f"{path} differs: arrays of shape {arr_a.shape} and {arr_b.shape}"
        )
        return
    assert a == b, f"{path} differs: {a!r} != {b!r}"
