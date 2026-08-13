#  Copyright (c) 2026. Jose Manuel Requena Plens
"""The ``$...$`` composer of the hand-drawn diagram canvas.

``diagrams.canvas.SVG.text`` composes a string carrying ``$...$`` markup
into styled ``<tspan>`` runs: italic variables, dropped/raised scripts at
reduced size, upright prose, operators and acronyms. These tests pin the
emitted SVG exactly -- the diagrams are committed byte for byte, so the
composer's output is part of the repository's contract -- and they pin the
path a plain string takes, which must stay byte-identical to what the
canvas emitted before the composer existed.
"""

from __future__ import annotations

import pathlib
import sys

import pytest

_SCRIPTS = str(pathlib.Path(__file__).resolve().parent.parent / "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from diagrams import canvas
from diagrams.canvas import _FONT, LIGHT, SVG


def _element(s: str, **kwargs: object) -> str:
    svg = SVG(900, 560, LIGHT)
    svg.text(450, 100, s, **kwargs)  # type: ignore[arg-type]
    assert len(svg.parts) == 1
    return svg.parts[0]


def test_plain_string_element_unchanged() -> None:
    # The exact element the canvas emitted before the composer existed.
    element = _element("probe A, 30 mm above", size=14, anchor="start")
    assert element == (
        f'<text x="450" y="100" font-family="{_FONT}" font-size="14" '
        f'fill="#1a1a1a" text-anchor="start">probe A, 30 mm above</text>'
    )


def test_plain_string_still_escapes_metacharacters() -> None:
    assert ">A &amp; B &lt;C&gt;<" in _element("A & B <C>")


def test_subscript_composition() -> None:
    element = _element("$L_p$", size=20)
    assert element == (
        f'<text x="450" y="100" font-family="{_FONT}" font-size="20" '
        f'fill="#1a1a1a" text-anchor="middle" xml:space="preserve">'
        f'<tspan font-style="italic">L</tspan>'
        f'<tspan dy="4.4" font-size="14.0" font-style="italic">p</tspan>'
        f"</text>"
    )


def test_superscript_composition() -> None:
    element = _element("$c^2$", size=20)
    assert '<tspan font-style="italic">c</tspan>' in element
    assert '<tspan dy="-7.6" font-size="14.0">2</tspan>' in element


def test_baseline_restored_after_script() -> None:
    # After Z's subscript the minus sign returns to the baseline: the dy
    # bookkeeping must emit the exact opposite shift.
    element = _element("$Z_2−Z_1$", size=20)
    assert '<tspan dy="4.4" font-size="14.0">2</tspan>' in element
    assert '<tspan dy="-4.4">−</tspan>' in element


def test_braced_subscript_retokenized_at_script_size() -> None:
    # Letters inside a braced script are variables at script size; the
    # comma stays upright at script size beside them.
    element = _element("$L_{p,s}$", size=20)
    assert (
        '<tspan dy="4.4" font-size="14.0" font-style="italic">p</tspan>'
        '<tspan font-size="14.0">,</tspan>'
        '<tspan font-size="14.0" font-style="italic">s</tspan>'
    ) in element


def test_operator_names_and_acronyms_stay_upright() -> None:
    element = _element("$CN = c·dt·√2/dx ≤ 1$", size=17)
    assert "<tspan>CN = </tspan>" in element
    assert '<tspan font-style="italic">c</tspan>' in element
    assert "<tspan>·dt·√2/dx ≤ 1</tspan>" in element


def test_greek_letters_are_italic_variables() -> None:
    # Adjacent single-letter variables share one italic tspan; the prime
    # is an upright glyph after its greek base.
    assert '<tspan font-style="italic">ρc</tspan>' in _element("$ρc$")
    element = _element("$κ′$")
    assert '<tspan font-style="italic">κ</tspan><tspan>′</tspan>' in element


def test_combining_mark_travels_with_its_letter() -> None:
    assert '<tspan font-style="italic">T̂</tspan>' in _element("$T̂$")


def test_prose_upright_and_single_text_chunk() -> None:
    element = _element("measurement cell → $L_{p,s}$ ($h_s$)", size=15)
    assert "<tspan>measurement cell → </tspan>" in element
    assert 'xml:space="preserve"' in element
    # No tspan carries coordinates of its own: one text chunk, so the
    # text-anchor of the element keeps positioning the composed whole.
    assert element.count('x="') == 1
    assert element.count(' y="') == 1


def test_anchor_and_bold_carry_over() -> None:
    element = _element("$f_n$ = 295 kHz", size=15, anchor="end", bold=True)
    assert 'text-anchor="end"' in element
    assert 'font-weight="600"' in element


def test_mono_and_italic_ignored_for_math() -> None:
    element = _element("$L_p$", mono=True, italic=True)
    assert f'font-family="{_FONT}"' in element
    assert "Consolas" not in element
    # The only italics are the composed variables, not the whole string.
    assert 'text-anchor="middle" xml:space=' in element


def test_math_and_prose_escape_metacharacters() -> None:
    element = _element("A & B $a<b$")
    assert "<tspan>A &amp; B </tspan>" in element
    assert '<tspan font-style="italic">a</tspan>' in element
    assert "<tspan>&lt;</tspan>" in element


def test_unbalanced_markup_raises() -> None:
    with pytest.raises(ValueError, match="unbalanced"):
        _element("broken $L_p")


def test_translation_happens_before_composition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_lookup(s: str, *, translate: bool) -> str:
        assert translate
        return {"speed $c_0$": "celeridad $c_0$"}[s]

    monkeypatch.setattr(canvas, "lookup", fake_lookup)
    svg = SVG(900, 560, LIGHT, lang="es")
    svg.text(450, 100, "speed $c_0$")
    assert "<tspan>celeridad </tspan>" in svg.parts[0]
    assert '<tspan font-style="italic">c</tspan>' in svg.parts[0]


def test_render_composes_math_in_the_title() -> None:
    plain = SVG(900, 560, LIGHT).render("Plain title")
    assert ">Plain title</text>" in plain
    composed = SVG(900, 560, LIGHT).render("Rating $R_w$ measured")
    assert 'xml:space="preserve"' in composed
    assert (
        '<tspan font-style="italic">R</tspan>'
        '<tspan dy="5.7" font-size="18.2" font-style="italic">w</tspan>'
        '<tspan dy="-5.7"> measured</tspan>'
    ) in composed


def test_composition_is_deterministic() -> None:
    def build() -> str:
        svg = SVG(900, 560, LIGHT)
        svg.text(450, 100, "$v ← v − (dt/ρ·dx)·grad p$,  then  $p$", size=13)
        svg.text(450, 130, "$TL(f) = 20 log_{10} |I(f) / T(f)|$", size=20)
        return svg.render("A title with $f_c$ inside")

    assert build() == build()
