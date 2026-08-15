#  Copyright (c) 2026. Jose Manuel Requena Plens
"""The minus sign the library's own ``.plot()`` labels write their numbers with.

The axis tick labels matplotlib draws carry the typographic minus (U+2212), so
a reading interpolated into a label has to carry it too: ``format`` writes the
ASCII hyphen, which is a shorter glyph sitting lower on the line, and the two
then share one figure with the short one always on the number the reader came
for. ``phonometry._i18n`` signs its numbers through :func:`format_number` and,
where a caller needs its own format spec, :func:`fmt_minus`.

The documentation figures settled the same question first, in
``scripts/figures/i18n.py``; ``tests/test_figure_minus_sign.py`` pins that
side. These tests pin the library's, and the reach is the same: the sign, and
nothing else that happens to be a hyphen. The one deliberate difference is the
reading that rounds onto zero -- the figures draw its sign, the library drops
it -- which is asserted here so the divergence stays a decision rather than a
surprise.
"""

from __future__ import annotations

import pytest

from phonometry._i18n import decimal_comma, fmt_minus, format_number

MINUS = "−"


@pytest.mark.parametrize(
    ("value", "spec", "expected"),
    [
        # The sign of a negative reading, in the specs the renderers use.
        (-6.6, "+.1f", f"{MINUS}6.6"),
        (-4.82, ".2f", f"{MINUS}4.82"),
        (-3, "d", f"{MINUS}3"),
        (-171.4, "6.1f", f"{MINUS}171.4"),
        (-12.5, "g", f"{MINUS}12.5"),
        # Width padding is what holds a readout column still while the number
        # under it changes: the pad stays in front of the sign.
        (-4.82, "6.2f", f" {MINUS}4.82"),
        (-4.82, "8.2f", f"   {MINUS}4.82"),
        # A positive number keeps its own notation, padded or signed.
        (4.82, "6.2f", "  4.82"),
        (6.6, "+.1f", "+6.6"),
        # The hyphen of an exponent belongs to the number, not to its sign,
        # which is why this takes the value and the spec and not the finished
        # string: ``:g`` reaches an exponent on its own.
        (1e-05, "g", "1e-05"),
        (-1.25e-05, "g", f"{MINUS}1.25e-05"),
        (-3.2e-07, ".1e", f"{MINUS}3.2e-07"),
    ],
)
def test_fmt_minus_signs_only_the_sign(value: float, spec: str, expected: str) -> None:
    assert fmt_minus(value, spec) == expected


@pytest.mark.parametrize(("value", "spec"), [(-4.82, ".1e"), (-1.0e-12, ".2e")])
def test_fmt_minus_keeps_every_hyphen_but_the_first(value: float, spec: str) -> None:
    written = fmt_minus(value, spec)
    assert written.startswith(MINUS)
    # One ASCII hyphen left where ``format`` wrote two: the exponent's.
    assert written.count("-") == format(value, spec).count("-") - 1


@pytest.mark.parametrize("value", [-4.82, -0.5, 4.82, 0.0, -171.4])
def test_fmt_minus_leaves_a_padded_field_its_width(value: float) -> None:
    assert len(fmt_minus(value, "8.2f")) == len(format(value, "8.2f")) == 8


@pytest.mark.parametrize(
    ("value", "language", "kwargs", "expected"),
    [
        (-6.6, "en", {}, f"{MINUS}6.6"),
        (-6.6, "es", {}, f"{MINUS}6,6"),
        (6.6, "en", {}, "6.6"),
        (-90.0, "en", {"trim": True}, f"{MINUS}90"),
        (-90.0, "es", {"trim": True}, f"{MINUS}90"),
        (-0.25, "es", {"decimals": 2}, f"{MINUS}0,25"),
    ],
)
def test_format_number_signs_with_a_minus(
    value: float, language: str, kwargs: dict[str, object], expected: str
) -> None:
    assert format_number(value, language, **kwargs) == expected  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [-0.0, -0.001, -0.04])
def test_a_reading_that_rounds_onto_zero_is_unsigned(value: float) -> None:
    """No sign at all, rather than a minus in front of a zero.

    The deliberate difference from the figures' helper, which draws the sign it
    was given. A library caller asking for one decimal of a value that is not
    zero but rounds to it is better served by ``0.0`` than by a ``−0.0`` that
    claims a direction the printed digits do not support.
    """
    assert format_number(value, decimals=1) == "0.0"
    assert format_number(value, "es", decimals=1) == "0,0"


def test_the_spanish_pass_still_reads_the_number() -> None:
    """The comma pass runs over the sign, and does not trip on the glyph."""
    assert decimal_comma(fmt_minus(-4.82, ".2f"), "es") == f"{MINUS}4,82"
    assert decimal_comma(fmt_minus(-171.4, "7.1f"), "es") == f" {MINUS}171,4"
    # ... and the exponent's hyphen is not a decimal point, so it survives too.
    assert decimal_comma(fmt_minus(-1.25e-05, "g"), "es") == f"{MINUS}1,25e-05"
