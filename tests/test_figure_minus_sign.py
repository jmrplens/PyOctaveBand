#  Copyright (c) 2026. Jose Manuel Requena Plens
"""The minus sign a figure label writes its numbers with.

The axis tick labels of every figure carry the typographic minus (U+2212) in
both languages, so a reading interpolated into a label has to carry it too:
``format`` writes the ASCII hyphen, which is shorter and sits lower, and the
two signs then share a figure with the short one always on the number the
reader came for. ``scripts/figures/i18n.py`` routes those numbers through
``_fmt_minus``, next to the decimal-comma pass that does the same job on the
other half of the notation.

It takes the number and its format spec rather than the finished sentence,
which is what lets it tell a sign from the hyphens around it. These tests pin
that reach: the width padding a monospace column depends on, the hyphen
inside the number's own exponent, and the sign that survives a rounding to
zero.
"""

import pathlib
import sys

import pytest

_SCRIPTS = str(pathlib.Path(__file__).resolve().parent.parent / "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from figures.i18n import _decimal_comma, _fmt_minus

MINUS = "−"


@pytest.mark.parametrize(
    ("value", "spec", "expected"),
    [
        # The sign of a negative reading, in the specs the clips use.
        (-4.82, ".2f", f"{MINUS}4.82"),
        (-6.56, ".2f", f"{MINUS}6.56"),
        (-3, "d", f"{MINUS}3"),
        (-171.4, "6.1f", f"{MINUS}171.4"),
        # Width padding is what holds a readout's columns still while the
        # number under it changes: the pad stays in front of the sign.
        (-4.82, "6.2f", f" {MINUS}4.82"),
        (-4.82, "8.2f", f"   {MINUS}4.82"),
        (-50.5, "6.1f", f" {MINUS}50.5"),
        # A positive number keeps its own notation, padded or signed.
        (4.82, "6.2f", "  4.82"),
        (4.82, "+.2f", "+4.82"),
        (0.82, "+.2f", "+0.82"),
        # An explicit "+" spec still writes the negative sign as a minus.
        (-4.82, "+.2f", f"{MINUS}4.82"),
        # Negative zero, and a negative reading that rounds onto zero: the
        # sign is drawn, so it is drawn as a minus.
        (-0.0, "", f"{MINUS}0.0"),
        (-0.0, ".1f", f"{MINUS}0.0"),
        (-0.001, ".1f", f"{MINUS}0.0"),
        (-0.004, ".2f", f"{MINUS}0.00"),
        (0.0, ".1f", "0.0"),
        # The hyphen of an exponent belongs to the number, not to its sign.
        (1e-05, "", "1e-05"),
        (-1e-05, "", f"{MINUS}1e-05"),
        (-3.2e-07, ".1e", f"{MINUS}3.2e-07"),
        (3.2e-07, ".1e", "3.2e-07"),
    ],
)
def test_only_the_sign_becomes_a_minus(value: float, spec: str, expected: str) -> None:
    assert _fmt_minus(value, spec) == expected


@pytest.mark.parametrize(
    ("value", "spec"),
    [(-4.82, ".1e"), (-3.2e-07, ".1e"), (-1.0e-12, ".2e")],
)
def test_a_number_keeps_every_hyphen_but_the_first(value: float, spec: str) -> None:
    written = _fmt_minus(value, spec)
    assert written.startswith(MINUS)
    # One ASCII hyphen left where ``format`` wrote two: the exponent's.
    assert written.count("-") == format(value, spec).count("-") - 1


@pytest.mark.parametrize("value", [-4.82, -0.5, 4.82, 0.0, -171.4])
def test_the_width_of_a_padded_field_is_unchanged(value: float) -> None:
    assert len(_fmt_minus(value, "8.2f")) == len(format(value, "8.2f")) == 8


def test_the_spanish_pass_still_reads_the_number() -> None:
    """The comma pass runs over the sign, and does not trip on the glyph."""
    assert _decimal_comma(_fmt_minus(-4.82, ".2f")) == f"{MINUS}4,82"
    assert _decimal_comma(_fmt_minus(-171.4, "7.1f")) == f" {MINUS}171,4"
